# Project context for Claude Code

## Project: RILH-AI-Vision

Computer-vision pipeline for roller inline hockey: automatic AI-driven
match recording, broadcast-style virtual follow-cam, and post-match
analytics. Built from open-source modules + custom code only.

## Current status
- **Phase 1** implemented: dual-backend detection + tracking (YOLO11 COCO by default, HockeyAI via `--hockey-model`) with ByteTrack, on a wide-angle source video
- **Phase 2** implemented: virtual follow-cam, smooth 16:9 crop driven by tracked positions

## Architecture
Two-pass pipeline:
1. `src/phase1_detect_track.py` → produces `tracks.json` (per-frame bounding boxes, class IDs, persistent track IDs)
2. `src/phase2_followcam.py` → reads `tracks.json` + original video → produces follow-cam MP4

Why two passes: detection is the slow step. Decoupling lets us iterate
on cinematography (zoom, smoothing, framing logic) without re-running inference.

## Key design choices
- **Two detector backends**, selectable at runtime in `phase1_detect_track.py`:
  - Default — **YOLO11 pretrained on COCO**, classes 0 (person) and 32
    (sports ball). Player detection is solid; puck detection via "sports ball"
    is unreliable (<1% of frames).
  - `--hockey-model` — **[HockeyAI](https://huggingface.co/SimulaMet-HOST/HockeyAI)**,
    a YOLOv8m fine-tuned on ice hockey with 7 classes (center ice, faceoff dots,
    goal frame, goaltender, players, puck, referee). Auto-downloaded on first
    run to `models/HockeyAI_model_weight.pt`. Transfers well to roller inline
    hockey. Classes are remapped at the source so the output schema stays
    Phase-2-compatible: player + goaltender → `class_id=0`, puck → `class_id=32`,
    referee + rink markers are dropped.
  Both backends write the same `tracks.json` schema (plus a `class_name`
  string per detection for introspection), so Phase 2 is identical regardless.
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
- Puck detection is workable on a 60s wide-angle clip with `--hockey-model`
  (~43% of frames vs <1% with COCO), but coverage still drops on motion blur,
  small pucks, poor lighting. A roller-specific fine-tune would narrow the gap.
- No player identification yet (jersey number OCR comes in Phase 6).
- No event detection yet (goals, shots — Phase 5).
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

**Phase 4 — Roller-specific fine-tune** (2–4 weeks)
- HockeyAI (ice-hockey-trained YOLOv8m) already covers ~43% puck coverage on
  roller hockey out-of-the-box via `--hockey-model`. This phase narrows the
  remaining gap with a roller-specific fine-tune.
- Build a roller-hockey-specific dataset (CVAT or Roboflow). Bootstrap
  with HockeyAI detections as pre-labels to cut annotation time.
- Fine-tune from HockeyAI weights (preferred) or from YOLO11.
- Target: >0.7 mAP on the puck class in roller conditions.

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
- For any serious puck work, pass `--hockey-model` — the COCO default is only
  useful when iterating fast and puck quality doesn't matter.
- Camera too slow → raise `--alpha`. Camera jittery → lower it, or raise `--polish-window`.
- Still missing puck detections → `--imgsz 1280` or 1536 (slower but much better on small objects)
- Use `--debug-overlay` to visually understand the focus trajectory
- Run outputs are kept incrementally: `runs/test01/`, `runs/test02/`, ...
  Don't overwrite a previous run even if it failed — it's useful for diffing
  a parameter change against the prior result.
