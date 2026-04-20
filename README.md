# RILH-AI-Vision

**Roller Inline Hockey — AI Vision.** Open-source computer-vision pipeline: automatic match recording, broadcast-style virtual follow-cam, and post-match analytics. Built from open-source modules + custom code only.

## Phases delivered here

- **Phase 1** — `src/phase1_detect_track.py` — detects + tracks players and puck on a match video, exports an annotated MP4 and a `tracks.json`. Two backends: **YOLO11 COCO** (default, fast, weak on puck) or **HockeyAI** via `--hockey-model` (ice-hockey-trained YOLOv8m, auto-downloaded, dramatically better on puck — see "Puck detection" below).
- **Phase 2** — `src/phase2_followcam.py` — reads the tracks, computes a smooth virtual camera trajectory, and crops a 16:9 broadcast-style follow-cam window from the source video.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

First run downloads YOLO11 weights (~50 MB) automatically.

GPU recommended but not required. CPU-only runs at roughly 1/5 of real-time on a modern laptop; a decent GPU runs 2–5x faster than real-time.

## Usage

### Phase 1 — Detect & track

```bash
# COCO default — fast, but puck detection is very weak
python src/phase1_detect_track.py path/to/match.mp4 --output runs/match01

# HockeyAI — recommended for anything serious (see "Puck detection" below)
python src/phase1_detect_track.py path/to/match.mp4 --output runs/match01 --hockey-model
```

Outputs:
- `runs/match01/annotated.mp4` — original video with bounding boxes, IDs, traces
- `runs/match01/tracks.json` — per-frame detections, fed to Phase 2

Useful flags:
- `--hockey-model` — use HockeyAI (YOLOv8m fine-tuned on ice hockey) instead of COCO YOLO11. Auto-downloads the weights to `models/` on first run. See "Puck detection" section.
- `--model yolo11n.pt` (faster) | `yolo11m.pt` (default) | `yolo11x.pt` (best, slowest) — only used when `--hockey-model` is not set.
- `--conf 0.3` — detection confidence threshold (lower = more detections including false positives)
- `--imgsz 1280` — inference resolution; helps small-object detection (still useful even with HockeyAI)

### Phase 2 — Virtual follow-cam

```bash
python src/phase2_followcam.py runs/match01/tracks.json path/to/match.mp4 \
  --output runs/match01/followcam.mp4 \
  --zoom 2.0 \
  --alpha 0.08 \
  --debug-overlay
```

Tunables:
- `--zoom` — 1.0 = full frame, 2.0 = show half width. Roller rinks are smaller than football pitches, so 2.0 is a good default.
- `--alpha` — EMA smoothing (0.03 silky / 0.2 reactive). Roller hockey is fast — start at 0.08 and adjust.
- `--puck-weight` — focus bias toward puck vs. players centroid (default 0.7).
- `--polish-window` — second-pass moving average for extra polish (default 15 frames, set 1 to disable).
- `--debug-overlay` — also produces a debug video showing focus point + crop rectangle on the source.

## Puck detection — two backends

### Default: COCO YOLO11 (not recommended for puck work)

Uses class 32 ("sports ball") as a puck proxy. Almost never catches a roller hockey puck (~0.1% of frames in internal tests). Fine when you only care about players.

### `--hockey-model`: HockeyAI

[HockeyAI](https://huggingface.co/SimulaMet-HOST/HockeyAI) is a YOLOv8m fine-tuned on ice hockey (SimulaMet-HOST). Seven classes: center ice, faceoff dots, goal frame, goaltender, players, puck, referee. The weights transfer well to roller inline hockey and are auto-downloaded (~52 MB) to `models/HockeyAI_model_weight.pt` on first use. Class IDs are remapped at the source so the `tracks.json` output stays compatible with Phase 2 (player + goaltender → `class_id=0`, puck → `class_id=32`, referee + rink markers are dropped).

On a typical 60s wide-angle clip (1920×1080 @ 60 fps):

| metric                     | COCO YOLO11n | HockeyAI YOLOv8m |
|----------------------------|--------------|------------------|
| player detections          | 63,224       | 17,860 (more selective) |
| player track IDs           | 1,804        | **433 (~4× more stable)** |
| frames with puck detected  | 0.1%         | **42.6%**        |

HockeyAI is slower (medium vs. nano) but the tracking output is dramatically cleaner and puck data is actually useful. Referees are excluded at the source.

### Phase 2 still has fallbacks for puck gaps

Even with HockeyAI, the puck is missed in ~57% of frames. Phase 2 handles this with:
1. **Short-term puck memory** — uses last known puck position for ~15 frames after detection drops
2. **Players-centroid fallback** — when the puck is lost too long, the camera tracks the cluster of players

For near-perfect puck tracking, a roller-specific fine-tune is the next step (Phase 4 of the roadmap — see `CLAUDE.md`).

## Workflow tips

- Trim to a 60-second test clip first: `ffmpeg -i full_match.mp4 -ss 0 -t 60 -c copy clip.mp4`
- Iterate Phase 2 parameters on a single Phase 1 run — you don't need to re-detect each time
- Open `tracks.json` to inspect the data structure for custom analytics
- The `--debug-overlay` output is your best friend for understanding why the camera moves the way it does

## Roadmap

See `CLAUDE.md` for full roadmap and architecture notes.
