# Project context for Claude Code

## Project: RILH-AI-Vision

Computer-vision pipeline for roller inline hockey: automatic AI-driven
match recording, broadcast-style virtual follow-cam, and post-match
analytics. Built from open-source modules + custom code only.

## Current status
- **Phase 1** implemented: YOLO11 + ByteTrack on a wide-angle source video
- **Phase 2** implemented: virtual follow-cam, smooth 16:9 crop driven by tracked positions

## Architecture
Two-pass pipeline:
1. `src/phase1_detect_track.py` → produces `tracks.json` (per-frame bounding boxes, class IDs, persistent track IDs)
2. `src/phase2_followcam.py` → reads `tracks.json` + original video → produces follow-cam MP4

Why two passes: detection is the slow step. Decoupling lets us iterate
on cinematography (zoom, smoothing, framing logic) without re-running inference.

## Key design choices
- **YOLO11 pretrained on COCO**, classes 0 (person) and 32 (sports ball).
  No fine-tuning yet. Player detection is solid; puck detection is the
  weak link — see "Known limitations" below.
- **ByteTrack** via Ultralytics' built-in `model.track(..., tracker="bytetrack.yaml")`
  for persistent IDs across frames.
- **Focus point = weighted blend of puck position and players centroid**.
  Puck gets high weight when detected; players-centroid fallback otherwise.
  Recently-seen puck positions are extrapolated for ~15 frames to bridge
  missed detections.
- **Smoothing = exponential moving average** on the focus trajectory,
  optionally followed by a centered boxcar pass for extra polish.
- **Crop window clamped to frame bounds** so we never show black bars.

## Known limitations (to address in later phases)
- COCO "sports ball" class doesn't reliably detect roller hockey pucks.
  This is the #1 thing to fix.
- No player identification yet (jersey number OCR comes in Phase 5).
- No event detection yet (goals, shots — Phase 4).
- No rink calibration (top-down map projection — Phase 3).
- Single-camera assumption. Multi-camera stitching = Phase 7.

## Conventions
- Python 3.10+
- All paths via `pathlib.Path`
- CLI scripts use `argparse`
- Outputs go under `runs/` (gitignored)
- Don't commit videos or model weights

## Roadmap

**Phase 3 — Rink calibration & 2D map** (1–2 weeks)
- Detect rink lines (segmentation: U-Net or SAM-based)
- Compute homography between camera plane and rink plane
- Project player positions onto a top-down 2D rink map
- Unlocks heatmaps, distance-traveled, zone occupancy stats

**Phase 4 — Custom puck detector** (2–4 weeks)
- Build a roller-hockey-specific dataset (annotation in CVAT or Roboflow)
- Fine-tune YOLO11 with the puck as a dedicated class
- Target: >0.7 mAP on the puck class
- This single phase makes Phase 2 dramatically better

**Phase 5 — Event detection** (3–6 weeks)
- Goals, shots, penalties — temporal action models (TSN, MoViNet, or
  SlowFast architectures)
- Likely needs custom labeled dataset

**Phase 6 — Player identification** (1–2 weeks)
- Jersey number OCR (PARSeq) on dorsal crops
- Per-player highlight reels, stats

**Phase 7 — Web platform** (2–3 weeks)
- FastAPI backend + Next.js frontend
- Match library, clip editor, tagging, sharing

**Phase 8+** — Multi-camera stitching, live streaming (RTMP/HLS via
MediaMTX), mobile control app.

## Datasets (for fine-tuning later)
No public roller-hockey CV dataset currently. Build our own:
- Use Phase 1 output to bootstrap annotations (semi-automatic labeling)
- Roboflow Universe sometimes has hockey datasets — search "ice hockey"
  as a starting point (transfers reasonably to roller inline)
- Annotate ~500–1000 frames with puck visible for first fine-tune

## Things to know when iterating
- Test on a 60s clip first: `ffmpeg -i input.mp4 -ss 0 -t 60 -c copy clip.mp4`
- Camera too slow → raise `--alpha`. Camera jittery → lower it, or raise `--polish-window`.
- Puck rarely detected → `--imgsz 1280` or 1536 (slower but much better on small objects)
- Use `--debug-overlay` to visually understand the focus trajectory
