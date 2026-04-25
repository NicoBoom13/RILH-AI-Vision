"""
RILH-AI-Vision — p1_a_detect
Detect & track players + puck with YOLO + ByteTrack — the entry point of
the identification pipeline.

Outputs:
- annotated.mp4   — source video with bounding boxes, IDs, traces overlaid
- detections.json — per-frame detection records, consumed by every
                    downstream stage (b/c/d/e and f/g)

Two model backends:
- Default: YOLO11 pretrained on COCO (class 0 person, class 32 "sports ball"
  as a puck proxy — unreliable on hockey pucks)
- --hockey-model: HockeyAI (YOLOv8m fine-tuned on ice hockey, 7 classes).
  Auto-downloaded from HuggingFace on first use. Classes are remapped so
  the detections.json output schema stays uniform across backends:
  class_id=0 for any skater/goaltender, class_id=32 for puck.
"""

import argparse
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# Output class IDs (what Phase 2 expects)
PERSON_CLASS = 0
PUCK_CLASS = 32

# All model weights are kept under models/ so repo root stays clean and the
# ultralytics auto-download lands in a predictable location.
MODELS_DIR = Path("models")


def resolve_model_path(name: str) -> Path:
    """Bare YOLO filenames (e.g. 'yolo11m.pt') are rewritten to models/<name>
    so ultralytics auto-downloads into models/ instead of CWD. Paths that
    already include a directory component are used verbatim."""
    p = Path(name)
    if len(p.parts) == 1:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return MODELS_DIR / p.name
    return p


# --- HockeyAI backend ---------------------------------------------------------

HOCKEY_MODEL_URL = (
    "https://huggingface.co/SimulaMet-HOST/HockeyAI/resolve/main/"
    "HockeyAI_model_weight.pt"
)
HOCKEY_MODEL_PATH = MODELS_DIR / "HockeyAI_model_weight.pt"

# Native HockeyAI class IDs → (remapped class_id for Phase 2, readable label)
# Dropped classes (Center Ice 0, Faceoff Dots 1, Goal Frame 2, Referee 6) are
# absent from this map, so detections of those classes are discarded.
HOCKEY_CLASS_MAP = {
    3: (PERSON_CLASS, "goaltender"),
    4: (PERSON_CLASS, "player"),
    5: (PUCK_CLASS, "puck"),
}


def ensure_hockey_model() -> Path:
    """Lazily download the HockeyAI weights into ``models/`` if missing.

    Returns:
        Path to the local checkpoint, ready to be passed to YOLO().
    """
    if HOCKEY_MODEL_PATH.exists():
        return HOCKEY_MODEL_PATH
    HOCKEY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading HockeyAI weights (~52 MB) → {HOCKEY_MODEL_PATH}")
    urllib.request.urlretrieve(HOCKEY_MODEL_URL, HOCKEY_MODEL_PATH)
    return HOCKEY_MODEL_PATH


# --- Main pipeline ------------------------------------------------------------

def run(
    video_path: Path,
    output_dir: Path,
    model_name: str,
    conf: float,
    imgsz: int,
    hockey_mode: bool,
    tracker: str,
    training_mode: bool = False,
):
    """Run YOLO detection + ByteTrack tracking on a video.

    Streams every frame through the model, applies the HockeyAI class
    remap if requested, drops extra puck detections in match mode, and
    writes both an annotated MP4 and a per-frame ``detections.json``.

    Args:
        video_path: Source MP4.
        output_dir: Where to write annotated.mp4 + detections.json. Created if missing.
        model_name: YOLO weights path (resolved by ``resolve_model_path``).
        conf: Detection confidence floor for the YOLO predictor.
        imgsz: Inference image size; bigger helps small-object recall (puck).
        hockey_mode: If True use HockeyAI weights + class remap; else COCO.
        tracker: Tracker config (yaml) — bytetrack.yaml or a path under configs/.
        training_mode: If True, keep all puck detections per frame
            (multi-puck drills). If False (default = match mode), keep
            only the highest-confidence puck per frame.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / "annotated.mp4"
    detections_path = output_dir / "detections.json"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Video: {width}x{height} @ {fps:.2f}fps, {total_frames} frames")
    print(f"Backend: {'HockeyAI' if hockey_mode else 'YOLO11 COCO'}")
    print(f"Model: {model_name}, conf={conf}, imgsz={imgsz}")
    print(f"Tracker: {tracker}")
    if training_mode:
        print("Training mode: keeping ALL puck detections per frame "
              "(multi-puck drills)")
    else:
        print("Match mode (default): keeping only the highest-confidence "
              "puck per frame")

    model = YOLO(model_name)

    if hockey_mode:
        native_classes = list(HOCKEY_CLASS_MAP.keys())
    else:
        native_classes = [PERSON_CLASS, PUCK_CLASS]

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.4, text_thickness=1)
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=30)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (width, height))

    all_frames = []

    results = model.track(
        source=str(video_path),
        persist=True,
        tracker=tracker,
        conf=conf,
        imgsz=imgsz,
        classes=native_classes,
        stream=True,
        verbose=False,
    )

    frame_idx = 0
    for r in results:
        frame = r.orig_img.copy()
        frame_record = {"frame": frame_idx, "boxes": []}

        if r.boxes is not None and len(r.boxes) > 0:
            detections = sv.Detections.from_ultralytics(r)

            # Supervision >=0.27 requires tracker_id to be set on the
            # Detections object itself (not just as a local variable).
            if detections.tracker_id is None:
                detections.tracker_id = np.full(len(detections), -1, dtype=int)

            if hockey_mode:
                native_ids = detections.class_id.copy()
                keep_mask = np.array(
                    [cid in HOCKEY_CLASS_MAP for cid in native_ids], dtype=bool
                )
                detections = detections[keep_mask]
                native_ids = native_ids[keep_mask]
                if len(detections) > 0:
                    detections.class_id = np.array(
                        [HOCKEY_CLASS_MAP[cid][0] for cid in native_ids], dtype=int
                    )
                    labels_native = [HOCKEY_CLASS_MAP[cid][1] for cid in native_ids]
            else:
                labels_native = [model.names[cid] for cid in detections.class_id]

            # Default (match mode): keep only the highest-confidence puck
            # per frame — a real match has exactly 1 puck on the ice, so
            # extra detections are false positives (shadows, pads, sticks,
            # spectator objects). Player + goaltender detections pass
            # through untouched. Done AFTER the tracker so puck tracks
            # already have IDs assigned; the dropped duplicates simply
            # never reach detections.json.
            # --training-mode disables this filter (e.g. drills with
            # multiple pucks on the ice at once).
            if not training_mode and len(detections) > 0:
                puck_mask = detections.class_id == PUCK_CLASS
                if puck_mask.sum() > 1:
                    puck_confs = detections.confidence[puck_mask]
                    puck_idx = np.where(puck_mask)[0]
                    keep_puck = puck_idx[np.argmax(puck_confs)]
                    drop = np.zeros(len(detections), dtype=bool)
                    drop[puck_idx] = True
                    drop[keep_puck] = False
                    keep = ~drop
                    detections = detections[keep]
                    labels_native = [labels_native[i] for i in range(len(keep)) if keep[i]]

            tracker_ids = detections.tracker_id

            for i in range(len(detections)):
                frame_record["boxes"].append({
                    "xyxy": [float(v) for v in detections.xyxy[i]],
                    "class_id": int(detections.class_id[i]),
                    "class_name": labels_native[i],
                    "track_id": int(tracker_ids[i]),
                    "conf": float(detections.confidence[i]),
                })

            if len(detections) > 0:
                labels = [
                    f"#{tid} {name}"
                    for tid, name in zip(tracker_ids, labels_native)
                ]
                frame = box_annotator.annotate(scene=frame, detections=detections)
                frame = label_annotator.annotate(
                    scene=frame, detections=detections, labels=labels
                )
                person_dets = detections[detections.class_id == PERSON_CLASS]
                if len(person_dets) > 0:
                    frame = trace_annotator.annotate(scene=frame, detections=person_dets)

        all_frames.append(frame_record)
        writer.write(frame)

        frame_idx += 1
        if frame_idx % 60 == 0:
            pct = 100 * frame_idx / max(total_frames, 1)
            print(f"  {frame_idx}/{total_frames} frames ({pct:.1f}%)")

    writer.release()

    with open(detections_path, "w") as f:
        json.dump({
            "video": str(video_path),
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": frame_idx,
            "model": model_name,
            "backend": "hockeyai" if hockey_mode else "coco",
            "tracker": tracker,
            "frames": all_frames,
        }, f)

    print(f"\nDone.")
    print(f"  Annotated video: {annotated_path}")
    print(f"  Detections data: {detections_path}")


def main():
    """CLI entry point — parse arguments and dispatch to ``run``."""
    parser = argparse.ArgumentParser(description="RILH-AI-Vision — P1.a: detect & track")
    parser.add_argument("video", type=str, help="Path to input video")
    parser.add_argument("--output", type=str, default="runs/latest", help="Output directory")
    parser.add_argument("--model", type=str, default="yolo11m.pt",
                        help="YOLO model (COCO mode only). Ignored when "
                             "--hockey-model is set. Examples: yolo11n.pt "
                             "(fast), yolo11m.pt (default), yolo11x.pt (best "
                             "YOLO11), yolo26l.pt (YOLO26 large, ~51MB; "
                             "auto-downloaded on first use).")
    parser.add_argument("--hockey-model", action="store_true",
                        help="Use HockeyAI (YOLOv8m fine-tuned on ice hockey) instead of COCO YOLO11.")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml",
                        help="Tracker config: bytetrack.yaml (default), botsort.yaml, "
                             "or a path to a custom YAML (e.g. configs/botsort_reid.yaml).")
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Inference image size — bigger helps puck detection")
    parser.add_argument("--training-mode", action="store_true",
                        help="Disable the default 1-puck-per-frame filter. "
                             "By default Phase 1 assumes match conditions "
                             "(exactly 1 puck on the ice) and keeps only "
                             "the highest-confidence puck per frame, "
                             "dropping false positives. Pass --training-mode "
                             "to keep every puck detection — useful for "
                             "drills where multiple pucks are in play at once.")
    args = parser.parse_args()

    if args.hockey_model:
        model_name = str(ensure_hockey_model())
    else:
        model_name = str(resolve_model_path(args.model))

    run(
        Path(args.video), Path(args.output), model_name,
        args.conf, args.imgsz, hockey_mode=args.hockey_model,
        tracker=args.tracker, training_mode=args.training_mode,
    )


if __name__ == "__main__":
    main()
