"""
RILH-AI-Vision — Phase 1
Detect & track players + puck with YOLO11 + ByteTrack.

Outputs:
- annotated.mp4  — source video with bounding boxes, IDs, traces overlaid
- tracks.json    — per-frame detection records, consumed by Phase 2

Note: COCO class 32 ("sports ball") is used as a proxy for the puck.
It is unreliable on roller hockey pucks; a custom fine-tune (roadmap
Phase 4) is required for production-quality puck detection.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# COCO class IDs we care about
PERSON_CLASS = 0
PUCK_CLASS = 32  # "sports ball" in COCO (used as proxy until Phase 4 fine-tune)


def run(video_path: Path, output_dir: Path, model_name: str, conf: float, imgsz: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / "annotated.mp4"
    tracks_path = output_dir / "tracks.json"

    # Probe video for writer setup
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Video: {width}x{height} @ {fps:.2f}fps, {total_frames} frames")
    print(f"Model: {model_name}, conf={conf}, imgsz={imgsz}")

    model = YOLO(model_name)

    # Annotators for the visualization video
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.4, text_thickness=1)
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=30)

    # MP4 writer (mp4v works without extra codecs on most installs;
    # if you have FFmpeg + H.264, swap in 'avc1' for smaller files)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (width, height))

    all_frames = []  # list of {"frame": i, "boxes": [...]}

    # Stream tracking — yields one Result per frame, with persistent IDs
    results = model.track(
        source=str(video_path),
        persist=True,
        tracker="bytetrack.yaml",
        conf=conf,
        imgsz=imgsz,
        classes=[PERSON_CLASS, PUCK_CLASS],
        stream=True,
        verbose=False,
    )

    frame_idx = 0
    for r in results:
        frame = r.orig_img.copy()
        frame_record = {"frame": frame_idx, "boxes": []}

        if r.boxes is not None and len(r.boxes) > 0:
            detections = sv.Detections.from_ultralytics(r)

            # tracker_id may be None for very early frames or untracked dets
            if detections.tracker_id is None:
                tracker_ids = np.full(len(detections), -1, dtype=int)
            else:
                tracker_ids = detections.tracker_id

            for i in range(len(detections)):
                frame_record["boxes"].append({
                    "xyxy": [float(v) for v in detections.xyxy[i]],
                    "class_id": int(detections.class_id[i]),
                    "track_id": int(tracker_ids[i]),
                    "conf": float(detections.confidence[i]),
                })

            # Build labels and annotate
            labels = [
                f"#{tid} {model.names[cid]}"
                for tid, cid in zip(tracker_ids, detections.class_id)
            ]
            frame = box_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
            # Trace only persons (puck trace is jittery and not very useful)
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

    # Persist tracks
    with open(tracks_path, "w") as f:
        json.dump({
            "video": str(video_path),
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": frame_idx,
            "model": model_name,
            "frames": all_frames,
        }, f)

    print(f"\nDone.")
    print(f"  Annotated video: {annotated_path}")
    print(f"  Tracks data:     {tracks_path}")


def main():
    parser = argparse.ArgumentParser(description="RILH-AI-Vision — Phase 1: detect & track")
    parser.add_argument("video", type=str, help="Path to input video")
    parser.add_argument("--output", type=str, default="runs/latest", help="Output directory")
    parser.add_argument("--model", type=str, default="yolo11m.pt",
                        help="YOLO model: yolo11n.pt | yolo11s.pt | yolo11m.pt | yolo11l.pt | yolo11x.pt")
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Inference image size — bigger helps puck detection")
    args = parser.parse_args()

    run(Path(args.video), Path(args.output), args.model, args.conf, args.imgsz)


if __name__ == "__main__":
    main()
