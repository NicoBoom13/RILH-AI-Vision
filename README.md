# RILH-AI-Vision

**Roller Inline Hockey — AI Vision.** Open-source computer-vision pipeline: automatic match recording, broadcast-style virtual follow-cam, and post-match analytics. Built from open-source modules + custom code only.

## Status by phase

| Phase | Module | Status | Notes |
|-------|--------|--------|-------|
| 1 — Detect & track | `src/phase1_detect_track.py` | ✅ | Dual backend (COCO YOLO11 or HockeyAI), configurable tracker (ByteTrack / BoT-SORT+ReID). |
| 2 — Follow-cam | `src/phase2_followcam.py` | ✅ | Works; not heavily iterated on. |
| 3 — Rink calibration | — | ❌ deferred | HockeyRink ice-hockey model doesn't transfer to roller rinks. Needs 200–300 annotated frames to fine-tune. |
| 4 — Roller fine-tune | — | ⏳ later | HockeyAI covers 43% puck coverage OOB; fine-tune when needed. |
| 5 — Event detection | — | ⏳ later | Not started. |
| 6 — Player identification | `src/phase6_identify.py` + `src/phase6_annotate.py` | 🟡 partial | PARSeq dorsal OCR + team color clustering + track merging. Limited by tracker fragmentation (next: post-hoc Re-ID clustering). |
| 7 — Web platform | — | ⏳ later | Not started. |

See `CLAUDE.md` for the full test log, design decisions, and open blockers.

## Scripts

- **`src/phase1_detect_track.py`** — Phase 1 detection + tracking.
- **`src/phase2_followcam.py`** — Phase 2 virtual follow-cam.
- **`src/phase6_identify.py`** — Phase 6 jersey-number OCR (PARSeq) on tracked players.
- **`src/phase6_annotate.py`** — viz with `#NN`/`#??` labels + green/blue team boxes.
- **`configs/bytetrack_tuned.yaml`** — longer-memory ByteTrack config.
- **`configs/botsort_reid.yaml`** — BoT-SORT with GMC + ReID (appearance).
- **`scripts/phase3_transfer_test.py`** — throwaway sanity check for the HockeyRink pretrained keypoint model (showed that transfer to roller fails; kept for reproducibility).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

First run auto-downloads YOLO11 weights (~50 MB) into `models/`. The HockeyAI weights land there too when you pass `--hockey-model`.

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
- `--tracker bytetrack.yaml` (default) | `configs/bytetrack_tuned.yaml` | `configs/botsort_reid.yaml` — choose the tracker backend. See "Tracking stability" below.
- `--conf 0.3` — detection confidence threshold (lower = more detections including false positives)
- `--imgsz 1280` — inference resolution; helps small-object detection (still useful even with HockeyAI)

### Phase 6 — Player identification

```bash
# 1) run OCR on the tracks produced by Phase 1 (HockeyAI strongly recommended)
python src/phase6_identify.py runs/match01/tracks.json path/to/match.mp4 \
  --output runs/match01/tracks_identified.json --samples-per-track 15

# 2) render the annotated video (#NN / #?? labels, green/blue team boxes)
python src/phase6_annotate.py runs/match01/tracks.json runs/match01/tracks_identified.json path/to/match.mp4 \
  --output runs/match01/annotated_numbered.mp4
```

Pipeline: YOLO11-pose → filter back-facing samples → torso-back crop → PARSeq digit OCR → per-track majority vote → merge tracks with the same number + non-overlapping time spans.

Tunables for `phase6_identify.py`:
- `--samples-per-track` — default 15. Raise if tracks are long-lived and OCR coverage is too low.
- `--ocr-min-conf` — default 0.4. Lower to widen coverage at the cost of more false positives.
- `--pose-model` — default `yolo11n-pose.pt` (auto-downloaded into `models/`).

On `test08` (HockeyAI tracks from a 60s clip), 52/433 tracks were identified (12%), and 11 player identities emerged from number-based track merging. Coverage is currently gated by tracker fragmentation — see "Tracking stability" below.

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

## Tracking stability

With 12 real entities on a roller rink (2 × 5 skaters + 2 goalies + 1 puck), the default ByteTrack tracker produces 300–450 track IDs per 60s clip because occlusions and camera cuts break identity. Two off-the-shelf mitigations are plumbed in via `--tracker`:

| Tracker | 60s clip result | When to use |
|---------|----------------|-------------|
| `bytetrack.yaml` (default) | ~433 tracks | Fast iteration, baseline. |
| `configs/bytetrack_tuned.yaml` | ~312 tracks | Longer memory (3s buffer) + more permissive matching. Same speed. |
| `configs/botsort_reid.yaml` | ~349 tracks, longest = 10.7s | BoT-SORT + GMC (camera-motion compensation) + ReID (appearance from YOLO backbone). Makes individual tracks longer even when total count isn't much lower. |

**None of these hit the ideal ~12 entities.** The planned next step (not implemented) is a post-hoc Re-ID clustering pass under a hard team constraint (max 5 skaters + goalie per team). Phase 6's number-based merging already recovers 11 player identities across broken tracks — see `CLAUDE.md` for the full breakdown.

## Workflow tips

- Trim to a 60-second test clip first: `ffmpeg -i full_match.mp4 -ss 0 -t 60 -c copy clip.mp4`
- Iterate Phase 2/6 parameters on a single Phase 1 run — you don't need to re-detect each time
- Open `tracks.json` to inspect the data structure for custom analytics
- The `--debug-overlay` output is your best friend for understanding why the camera moves the way it does
- Always pass `python -u` for long runs — stdout is block-buffered by default, which makes progress invisible
- Outputs are kept incrementally: `runs/test01/`, `runs/test02/`, … never overwrite a previous run, even if it failed

## Roadmap

See `CLAUDE.md` for full roadmap and architecture notes.
