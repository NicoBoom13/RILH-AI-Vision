"""
RILH-AI-Vision — pose cache shared by Stage 1.b and Stage 1.c.

Both stages need YOLO-pose results on a sampled subset of frames per
track. Running pose twice on overlapping frame sets is wasteful — this
module runs pose once on the union of needed frames and caches the
result to disk so each stage just reads it.

Cache file layout (pickle protocol 4):
  {"meta": {"pose_model": str, "pose_imgsz": int, "samples_per_track": int},
   "cache": {source_frame_idx: {"boxes": ndarray, "kp_xy": ndarray,
                                "kp_conf": ndarray} | None}}
A None entry means the pose model returned zero people on that frame.

Used as both a library (imported by p1_b_teams + p1_c_numbers) and a
CLI (run by the project orchestrator before Stage 1.b / 1.c).
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

PERSON_CLASS = 0
MODELS_DIR = Path("models")


def pick_device():
    """mps > cuda > cpu — same selection rule as the stage scripts."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_yolo_path(name: str) -> Path:
    """Route a bare YOLO filename into ``models/`` so Ultralytics
    auto-downloads land there. Paths with a directory component
    are kept as-is."""
    p = Path(name)
    if len(p.parts) == 1:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return MODELS_DIR / p.name
    return p


def needed_frames_top_n(detections_data, samples_per_track):
    """Frames the cache needs to cover: union of top-N highest-conf
    player detections per track. Stage 1.c default is 15, Stage 1.b
    is 8, so 15 covers both."""
    by_tid = defaultdict(list)
    for fr in detections_data["frames"]:
        for b in fr["boxes"]:
            if b["class_id"] != PERSON_CLASS or b["track_id"] < 0:
                continue
            by_tid[b["track_id"]].append((fr["frame"], b["conf"]))
    needed = set()
    for tid, dets in by_tid.items():
        for fi, _ in sorted(dets, key=lambda d: -d[1])[:samples_per_track]:
            needed.add(int(fi))
    return needed


def run_pose_for_frames(video_path, frame_indices, pose_model,
                        pose_imgsz, device):
    """Iterate the video sequentially once, run pose on each requested
    frame, return {frame_idx: pose_record | None}. Reading sequentially
    is much faster than seeking — same approach as the stage scripts."""
    cache = {}
    needed = set(int(i) for i in frame_indices)
    if not needed:
        return cache
    max_needed = max(needed)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    current = 0
    n_done = 0
    n_total = len(needed)
    while current <= max_needed:
        ok, frame = cap.read()
        if not ok:
            break
        if current in needed:
            res = pose_model.predict(
                source=frame, imgsz=pose_imgsz, conf=0.10,
                classes=[PERSON_CLASS], verbose=False, device=device,
            )[0]
            if (res.keypoints is not None and res.boxes is not None
                    and len(res.boxes) > 0):
                cache[current] = {
                    "boxes": res.boxes.xyxy.cpu().numpy(),
                    "kp_xy": res.keypoints.xy.cpu().numpy(),
                    "kp_conf": res.keypoints.conf.cpu().numpy(),
                }
            else:
                cache[current] = None
            n_done += 1
            if n_done % 100 == 0 or n_done == n_total:
                pct = 100 * n_done / max(n_total, 1)
                print(f"  pose {n_done}/{n_total} frames ({pct:.1f}%)")
        current += 1
    cap.release()
    return cache


def save_cache(path, cache, meta):
    """Persist cache + meta as a single pickle file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"meta": meta, "cache": cache}, f, protocol=4)


def load_cache(path):
    """Return (meta, cache_dict) or (None, None) if the file is missing.
    Stage scripts treat that as a soft miss and fall back to running
    pose inline so they stay independently runnable."""
    p = Path(path)
    if not p.exists():
        return None, None
    with open(p, "rb") as f:
        payload = pickle.load(f)
    return payload.get("meta"), payload.get("cache", {})


def main():
    """CLI: pre-extract pose for Stage 1.b + 1.c."""
    p = argparse.ArgumentParser(
        description="Pre-extract YOLO-pose results once for Stage 1.b + 1.c "
                    "(saves the duplicated pose pass between the two stages, "
                    "and lets them run in parallel without GPU contention "
                    "on the pose model).",
    )
    p.add_argument("detections_json", type=str,
                   help="Stage 1.a output (p1_a_detections.json).")
    p.add_argument("video", type=str)
    p.add_argument("--output", type=str, default=None,
                   help="Cache file path (default: "
                        "<detections_dir>/p1_pose_cache.pkl).")
    p.add_argument("--pose-model", type=str, default="yolo11n-pose.pt",
                   help="YOLO pose weights. Same default selection as the "
                        "stages; pass yolo26l-pose.pt for the larger model.")
    p.add_argument("--pose-imgsz", type=int, default=1280)
    p.add_argument("--samples-per-track", type=int, default=15,
                   help="Pre-extract pose for the union of top-N highest-conf "
                        "player detections per track. 15 is the Stage 1.c "
                        "default (Stage 1.b uses 8, so it's a subset).")
    args = p.parse_args()

    detections_path = Path(args.detections_json)
    video_path = Path(args.video)
    cache_path = (Path(args.output) if args.output
                  else detections_path.with_name("p1_pose_cache.pkl"))

    device = pick_device()
    print(f"Device: {device}")
    print(f"Loading detections: {detections_path}")
    detections_data = json.loads(detections_path.read_text())
    needed = needed_frames_top_n(detections_data, args.samples_per_track)
    print(f"Pose pre-extract: top-{args.samples_per_track} per track → "
          f"{len(needed)} unique frames "
          f"(over {len(detections_data['frames'])} processed source frames)")

    pose_path = str(resolve_yolo_path(args.pose_model))
    print(f"Loading pose model: {pose_path}")
    pose_model = YOLO(pose_path)

    cache = run_pose_for_frames(video_path, needed, pose_model,
                                args.pose_imgsz, device)
    meta = {
        "pose_model": pose_path,
        "pose_imgsz": int(args.pose_imgsz),
        "samples_per_track": int(args.samples_per_track),
    }
    save_cache(cache_path, cache, meta)
    n_with_pose = sum(1 for v in cache.values() if v is not None)
    print(f"\nDone.")
    print(f"  Cache: {cache_path}")
    print(f"  Frames with pose: {n_with_pose}/{len(cache)}")


if __name__ == "__main__":
    main()
