"""
RILH-AI-Vision — Phase 2
Virtual follow-cam.

Reads tracks.json from Phase 1 + the original video, computes a smooth
focus-point trajectory, and crops a 16:9 broadcast-style window that
follows the action.

Strategy:
  - Per-frame focus point = blend(puck position, players centroid).
    Puck is preferred when detected; otherwise we fall back to the
    players' centroid. Recently-seen puck positions are extrapolated
    for a short window to bridge missed detections (very common with
    COCO-pretrained models on hockey pucks).
  - Smoothing: exponential moving average on the focus trajectory.
    Optional second pass (centered moving average) for extra polish.
  - Crop window: fixed size derived from --zoom, clamped to frame bounds.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

PERSON_CLASS = 0
PUCK_CLASS = 32

# How many frames to keep "remembering" the last puck position when
# detection drops. Beyond this, fall back to players centroid.
PUCK_MEMORY_FRAMES = 15


def compute_focus_point(boxes, last_puck, puck_age, players_fallback,
                        puck_weight, frame_w, frame_h):
    """Return (focus_xy, new_last_puck, new_puck_age, new_players_fallback)."""
    puck_pts = [b for b in boxes if b["class_id"] == PUCK_CLASS]
    player_pts = [b for b in boxes if b["class_id"] == PERSON_CLASS]

    def center(b):
        """Return the bbox centroid as a numpy 2-vector."""
        x1, y1, x2, y2 = b["xyxy"]
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

    # Puck this frame: take the highest-confidence detection
    if puck_pts:
        puck_pts.sort(key=lambda b: -b["conf"])
        puck_xy = center(puck_pts[0])
        last_puck = puck_xy
        puck_age = 0
    elif last_puck is not None and puck_age < PUCK_MEMORY_FRAMES:
        puck_xy = last_puck
        puck_age += 1
    else:
        puck_xy = None
        last_puck = None

    # Players centroid (with persistence across rare empty frames)
    if player_pts:
        players_centroid = np.mean([center(b) for b in player_pts], axis=0)
        players_fallback = players_centroid
    else:
        players_centroid = players_fallback  # may be None on very first frames

    # Combine
    if puck_xy is not None and players_centroid is not None:
        focus = puck_weight * puck_xy + (1 - puck_weight) * players_centroid
    elif puck_xy is not None:
        focus = puck_xy
    elif players_centroid is not None:
        focus = players_centroid
    else:
        focus = np.array([frame_w / 2.0, frame_h / 2.0])

    return focus, last_puck, puck_age, players_fallback


def ema_smooth(points, alpha):
    """Causal exponential moving average. Lower alpha = smoother (laggier)."""
    smoothed = np.empty_like(points)
    state = points[0].copy()
    for i, p in enumerate(points):
        state = alpha * p + (1 - alpha) * state
        smoothed[i] = state
    return smoothed


def boxcar_smooth(points, window):
    """Centered moving average for extra polish; window forced odd."""
    if window <= 1:
        return points
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.concatenate([
        np.repeat(points[:1], pad, axis=0),
        points,
        np.repeat(points[-1:], pad, axis=0),
    ])
    kernel = np.ones(window) / window
    out = np.empty_like(points)
    out[:, 0] = np.convolve(padded[:, 0], kernel, mode="valid")
    out[:, 1] = np.convolve(padded[:, 1], kernel, mode="valid")
    return out


def get_crop_window(focus, frame_w, frame_h, crop_w, crop_h):
    """Return the (x1, y1, x2, y2) crop centred on ``focus`` and clamped
    to the frame bounds — never produces black bars at the edges."""
    cx, cy = focus
    x1 = int(round(cx - crop_w / 2))
    y1 = int(round(cy - crop_h / 2))
    x1 = max(0, min(x1, frame_w - crop_w))
    y1 = max(0, min(y1, frame_h - crop_h))
    return x1, y1, x1 + crop_w, y1 + crop_h


def run(tracks_path: Path, video_path: Path, output_path: Path,
        zoom: float, alpha: float, puck_weight: float,
        polish_window: int, debug_overlay: bool):
    """Render a virtual broadcast follow-cam from detections.json.

    Computes a focus point per frame as a weighted blend of the puck
    position and the players' centroid, smooths the trajectory with
    EMA + an optional centred boxcar pass, then crops a fixed-aspect
    window centred on that focus and writes it to ``output_path``."""
    with open(tracks_path) as f:
        data = json.load(f)

    frame_w = data["width"]
    frame_h = data["height"]
    fps = data["fps"]
    n_frames = data["total_frames"]

    # Crop dimensions: 16:9 derived from zoom factor
    crop_w = int(frame_w / zoom)
    crop_h = int(crop_w * 9 / 16)
    if crop_h > frame_h:
        crop_h = frame_h
        crop_w = int(crop_h * 16 / 9)
    # Even dimensions (some codecs are picky)
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2

    print(f"Source {frame_w}x{frame_h} @ {fps:.2f}fps")
    print(f"Crop window: {crop_w}x{crop_h} (zoom={zoom})")
    print(f"Smoothing: alpha={alpha}, polish_window={polish_window}")

    # 1) Compute raw focus per frame
    frame_to_boxes = {f["frame"]: f["boxes"] for f in data["frames"]}
    raw = np.zeros((n_frames, 2), dtype=np.float64)
    last_puck = None
    puck_age = 0
    players_fallback = None
    for i in range(n_frames):
        boxes = frame_to_boxes.get(i, [])
        focus, last_puck, puck_age, players_fallback = compute_focus_point(
            boxes, last_puck, puck_age, players_fallback,
            puck_weight, frame_w, frame_h,
        )
        raw[i] = focus

    # 2) Smooth: causal EMA + optional centered boxcar polish
    smoothed = ema_smooth(raw, alpha=alpha)
    if polish_window > 1:
        smoothed = boxcar_smooth(smoothed, polish_window)

    # 3) Render
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    if debug_overlay:
        debug_path = output_path.with_name(output_path.stem + "_debug.mp4")
        debug_writer = cv2.VideoWriter(str(debug_path), fourcc, fps, (frame_w, frame_h))
    else:
        debug_writer = None

    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (crop_w, crop_h))

    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break

        focus = smoothed[i]
        x1, y1, x2, y2 = get_crop_window(focus, frame_w, frame_h, crop_w, crop_h)
        cropped = frame[y1:y2, x1:x2]
        if cropped.shape[1] != crop_w or cropped.shape[0] != crop_h:
            cropped = cv2.resize(cropped, (crop_w, crop_h))
        writer.write(cropped)

        if debug_writer is not None:
            dbg = frame.copy()
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(dbg, (int(focus[0]), int(focus[1])), 8, (0, 0, 255), -1)
            cv2.circle(dbg, (int(raw[i, 0]), int(raw[i, 1])), 5, (255, 0, 0), -1)
            debug_writer.write(dbg)

        if i % 60 == 0:
            print(f"  rendered {i}/{n_frames}")

    cap.release()
    writer.release()
    if debug_writer is not None:
        debug_writer.release()
        print(f"  Debug video:   {debug_path}")
    print(f"\nDone.")
    print(f"  Follow-cam:    {output_path}")


def main():
    """CLI entry point — parse arguments and dispatch to ``run``."""
    parser = argparse.ArgumentParser(description="RILH-AI-Vision — stage_f_followcam : virtual follow-cam")
    parser.add_argument("tracks", type=str, help="Path to detections.json from stage_a")
    parser.add_argument("video", type=str, help="Path to original video")
    parser.add_argument("--output", type=str, default="runs/latest/followcam.mp4")
    parser.add_argument("--zoom", type=float, default=2.0,
                        help="1.0 = full frame; 2.0 default for roller rinks")
    parser.add_argument("--alpha", type=float, default=0.08,
                        help="EMA smoothing factor (0.03 silky, 0.2 reactive)")
    parser.add_argument("--puck-weight", type=float, default=0.7,
                        help="Bias toward puck vs. players centroid")
    parser.add_argument("--polish-window", type=int, default=15,
                        help="Centered moving-average window (frames). 1 = off.")
    parser.add_argument("--debug-overlay", action="store_true",
                        help="Also render a debug video with focus point + crop rect")
    args = parser.parse_args()

    run(
        tracks_path=Path(args.tracks),
        video_path=Path(args.video),
        output_path=Path(args.output),
        zoom=args.zoom,
        alpha=args.alpha,
        puck_weight=args.puck_weight,
        polish_window=args.polish_window,
        debug_overlay=args.debug_overlay,
    )


if __name__ == "__main__":
    main()
