"""
Throwaway — sanity-check HockeyRink.pt transfer from ice to roller inline hockey.

Samples N evenly-spaced frames from a clip, runs HockeyRink inference, overlays
the detected keypoints with their native index + confidence, saves PNGs.
"""

import argparse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

HOCKEYRINK_URL = (
    "https://huggingface.co/SimulaMet-HOST/HockeyRink/resolve/main/HockeyRink.pt"
)
MODELS_DIR = Path("models")
HOCKEYRINK_PATH = MODELS_DIR / "HockeyRink.pt"


def ensure_hockeyrink() -> Path:
    if HOCKEYRINK_PATH.exists():
        return HOCKEYRINK_PATH
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading HockeyRink weights (~93 MB) → {HOCKEYRINK_PATH}")
    urllib.request.urlretrieve(HOCKEYRINK_URL, HOCKEYRINK_PATH)
    return HOCKEYRINK_PATH


def sample_frames(video_path: Path, n: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Evenly spaced, avoid first/last frame which may be black
    indices = [int(i * (total - 1) / max(n - 1, 1)) for i in range(n)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append((idx, frame))
    cap.release()
    return frames, total


def overlay_keypoints(frame, kp_xy, kp_conf, min_conf=0.3):
    """Draw all keypoints with conf > min_conf, labeled with their native index."""
    vis = frame.copy()
    n_drawn = 0
    for i in range(len(kp_xy)):
        x, y = float(kp_xy[i, 0]), float(kp_xy[i, 1])
        c = float(kp_conf[i])
        if c < min_conf or (x == 0 and y == 0):
            continue
        # Color-ramp by confidence: green = high, red = low
        color = (0, int(255 * min(c, 1.0)), int(255 * (1.0 - min(c, 1.0))))
        cv2.circle(vis, (int(x), int(y)), 6, color, 2)
        cv2.circle(vis, (int(x), int(y)), 2, (255, 255, 255), -1)
        label = f"{i}:{c:.2f}"
        cv2.putText(vis, label, (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        n_drawn += 1
    return vis, n_drawn


def main():
    p = argparse.ArgumentParser(description="HockeyRink transfer sanity check")
    p.add_argument("video", type=str)
    p.add_argument("--output", type=str, default="runs/test05")
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--conf", type=float, default=0.25,
                   help="Detector confidence threshold")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--min-kp-conf", type=float, default=0.3,
                   help="Per-keypoint confidence floor for drawing")
    args = p.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = ensure_hockeyrink()
    print(f"Loading {model_path}")
    model = YOLO(str(model_path))

    video_path = Path(args.video)
    frames, total = sample_frames(video_path, args.samples)
    print(f"Video has {total} frames — sampling {len(frames)} frames at indices "
          f"{[i for i, _ in frames]}")

    per_kp_conf = np.zeros(56)  # accumulate mean conf per native keypoint index
    per_kp_count = np.zeros(56)

    for frame_idx, frame in frames:
        results = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf,
                                verbose=False)[0]

        n_rink = 0 if results.boxes is None else len(results.boxes)
        print(f"\nFrame {frame_idx}: {n_rink} rink detections")

        if (results.keypoints is None or results.boxes is None
                or len(results.boxes) == 0):
            out_path = output_dir / f"frame_{frame_idx:05d}.png"
            cv2.imwrite(str(out_path), frame)
            print(f"  no detections; saved raw frame → {out_path}")
            continue

        # Pick the highest-conf rink detection (should be only one, but be safe)
        confs = results.boxes.conf.cpu().numpy()
        best = int(np.argmax(confs))
        kp_xy = results.keypoints.xy.cpu().numpy()[best]
        kp_conf = results.keypoints.conf.cpu().numpy()[best]

        visible = int((kp_conf >= args.min_kp_conf).sum())
        print(f"  rink bbox conf: {confs[best]:.2f}")
        print(f"  keypoints visible (conf ≥ {args.min_kp_conf}): {visible}/56")

        for i in range(56):
            if kp_conf[i] >= args.min_kp_conf and not (
                kp_xy[i, 0] == 0 and kp_xy[i, 1] == 0
            ):
                per_kp_conf[i] += kp_conf[i]
                per_kp_count[i] += 1

        vis, n_drawn = overlay_keypoints(frame, kp_xy, kp_conf, args.min_kp_conf)
        out_path = output_dir / f"frame_{frame_idx:05d}.png"
        cv2.imwrite(str(out_path), vis)
        print(f"  saved overlay ({n_drawn} kps drawn) → {out_path}")

    print("\n=== Per-keypoint detection rate across sampled frames ===")
    print(f"(out of {len(frames)} frames, mean conf shown when detected)")
    for i in range(56):
        if per_kp_count[i] > 0:
            mean = per_kp_conf[i] / per_kp_count[i]
            print(f"  kp {i:2d}: seen {int(per_kp_count[i])}/{len(frames)}  "
                  f"mean conf {mean:.2f}")

    # Which keypoints are NEVER seen?
    never = [i for i in range(56) if per_kp_count[i] == 0]
    print(f"\nKeypoints never detected: {len(never)}/56 — indices {never}")

    ratio = (per_kp_count > 0).mean()
    print(f"\nTransfer score (fraction of kps detected at least once): {ratio:.1%}")


if __name__ == "__main__":
    main()
