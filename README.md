# RILH-AI-Vision

**Roller Inline Hockey — AI Vision.** Open-source computer-vision pipeline: automatic match recording, broadcast-style virtual follow-cam, and post-match analytics. Built from open-source modules + custom code only.

## Phases delivered here

- **Phase 1** — `src/phase1_detect_track.py` — runs YOLO11 + ByteTrack on a match video, exports an annotated MP4 and a `tracks.json` with all detections (players + puck).
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
python src/phase1_detect_track.py path/to/match.mp4 --output runs/match01
```

Outputs:
- `runs/match01/annotated.mp4` — original video with bounding boxes, IDs, traces
- `runs/match01/tracks.json` — per-frame detections, fed to Phase 2

Useful flags:
- `--model yolo11n.pt` (faster) | `yolo11m.pt` (default) | `yolo11x.pt` (best, slowest)
- `--conf 0.25` — detection confidence threshold (lower = more detections including false positives)
- `--imgsz 1280` — inference resolution; **strongly recommended for puck detection** (small, fast object)

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

## Important — puck detection

The COCO-pretrained YOLO11 model was not trained on roller hockey pucks. It uses class 32 ("sports ball"), which sometimes catches the puck but is unreliable, especially when:
- The puck is small in frame
- It moves fast (motion blur)
- Lighting is uneven (typical indoor rinks)

Phase 2 handles this with two fallbacks:
1. **Short-term puck memory** — uses last known puck position for ~15 frames after detection drops
2. **Players-centroid fallback** — when the puck is lost too long, the camera tracks the cluster of players

For broadcast-quality puck tracking, a custom fine-tune is required (Phase 4 of the roadmap, see `CLAUDE.md`).

## Workflow tips

- Trim to a 60-second test clip first: `ffmpeg -i full_match.mp4 -ss 0 -t 60 -c copy clip.mp4`
- Iterate Phase 2 parameters on a single Phase 1 run — you don't need to re-detect each time
- Open `tracks.json` to inspect the data structure for custom analytics
- The `--debug-overlay` output is your best friend for understanding why the camera moves the way it does

## Roadmap

See `CLAUDE.md` for full roadmap and architecture notes.
