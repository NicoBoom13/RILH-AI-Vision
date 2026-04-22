# RILH-AI-Vision

**Roller Inline Hockey — AI Vision.** Open-source computer-vision pipeline: automatic match recording, broadcast-style virtual follow-cam, and post-match analytics. Built from open-source modules + custom code only.

## Status by phase

| Phase | Module | Status | Notes |
|-------|--------|--------|-------|
| 1 — Detect & track | `src/phase1_detect_track.py` | ✅ | Dual backend (COCO YOLO11 or HockeyAI), configurable tracker (ByteTrack / BoT-SORT+ReID). |
| 1.5 — Team classification | `src/phase1_5_teams.py` | ✅ | Pose-based torso crop + multi-point dominant color + per-crop k=2 vote. HSV default; skater-only fit, goalies classified post-hoc. |
| 2 — Follow-cam | `src/phase2_followcam.py` | ✅ | Works; not heavily iterated on. |
| 3 — Rink calibration | — | ❌ deferred | HockeyRink ice-hockey model doesn't transfer to roller rinks. Needs 200–300 annotated frames to fine-tune. |
| 4 — Roller fine-tune | — | ⏳ later | HockeyAI covers 43% puck coverage OOB; fine-tune when needed. |
| 5 — Event detection | — | ⏳ later | Not started. |
| 6 — Player identification | `src/phase6_identify.py` + `src/phase6_annotate.py` | 🟡 partial | Dorsal jersey-number OCR — PARSeq (default) or TrOCR (`--ocr-engine trocr`). Tight torso-band crop + letterbox pad to PARSeq's 4:1 aspect. |
| 1.6 — Entity Re-ID clustering | `src/phase1_6_entities.py` | ✅ | OSNet medoid embedding per track + greedy merge under team + non-overlap + OCR constraints. Collapses ~200 fragments into ~20–40 entities. Design doc: `docs/phase_1_6_design.md`. |
| 7 — Web platform | — | ⏳ later | Not started. |

See `CLAUDE.md` for the full test log, design decisions, and open blockers.

## Scripts

- **`src/phase1_detect_track.py`** — Phase 1 detection + tracking.
- **`src/phase1_5_teams.py`** — Phase 1.5 team classification (green vs blue) from per-track jersey color.
- **`src/phase2_followcam.py`** — Phase 2 virtual follow-cam.
- **`src/phase6_identify.py`** — Phase 6 jersey-number OCR on tracked players; PARSeq (default) or TrOCR.
- **`src/phase1_6_entities.py`** — Phase 1.6 entity Re-ID clustering; collapses track fragments into stable entities.
- **`src/phase6_annotate.py`** — viz with `#NN`/`#??` labels + green/blue team boxes (entity-aware if `tracks_entities.json` is present).
- **`configs/bytetrack_tuned.yaml`** — longer-memory ByteTrack config.
- **`configs/botsort_reid.yaml`** — BoT-SORT with GMC + ReID (appearance).
- **`src/phase3_transfer_test.py`** — throwaway sanity check for the HockeyRink pretrained keypoint model (showed that transfer to roller fails; kept for reproducibility).

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
- `--model <weights>` — YOLO weights (COCO mode only, ignored when `--hockey-model` is set). All are auto-downloaded into `models/` on first use:
  - `yolo11n.pt` (fastest, lowest quality)
  - `yolo11m.pt` (default, balanced)
  - `yolo11x.pt` (best YOLO11 quality, slowest)
  - `yolo26l.pt` — **YOLO26 large** (~51 MB), newer architecture (released Jan 2026), a good middle ground between `yolo11m` and `yolo11x` on most benchmarks. Worth trying when the COCO detector is the bottleneck.
- `--tracker bytetrack.yaml` (default) | `configs/bytetrack_tuned.yaml` | `configs/botsort_reid.yaml` — choose the tracker backend. See "Tracking stability" below.
- `--conf 0.3` — detection confidence threshold (lower = more detections including false positives)
- `--imgsz 1280` — inference resolution; helps small-object detection (still useful even with HockeyAI)

### Phase 1.5 — Team classification

```bash
python src/phase1_5_teams.py runs/match01/tracks.json path/to/match.mp4
# writes runs/match01/tracks_teams.json + runs/match01/teams_preview.png
```

Pipeline: YOLO11-pose → torso-band crop (shoulders→hips, bbox-fallback for dark jerseys) → 3×2 multi-point dominant color averaging → k=2 k-means on skater tracks (HSV default) → majority vote per track. Goalies classified post-hoc against the skater centroids so their often-contrasting pads don't pull the team centres.

Tunables: `--space {hsv,bgr}`, `--grid RxC` (default `3x2`), `--samples-per-track`, `--pose-model` (default `yolo11n-pose.pt`; pass `yolo26l-pose.pt` for the newer YOLO26-large pose model — ~55 MB, better keypoint localisation on dark / low-contrast jerseys at the cost of ~3× inference time). See `teams_preview.png` + the JSON's `cluster_margin` to judge whether the two teams actually separate in your video.

### Phase 6 — Player identification

```bash
# 1) run OCR on the tracks from Phase 1 (HockeyAI strongly recommended)
python src/phase6_identify.py runs/match01/tracks.json path/to/match.mp4 \
  --output runs/match01/tracks_identified.json

# 2) render the annotated video (#NN / #?? labels, green/blue team boxes)
python src/phase6_annotate.py runs/match01/tracks.json runs/match01/tracks_identified.json path/to/match.mp4 \
  --output runs/match01/annotated_numbered.mp4
```

Pipeline: YOLO11-pose → filter back-facing samples → **tight back-of-torso crop (number band)** → **letterbox pad to OCR aspect** → PARSeq or TrOCR digit recognition → per-track majority vote → merge tracks with the same number + non-overlapping time spans.

Tunables for `phase6_identify.py`:
- `--ocr-engine {parseq,trocr}` — PARSeq (default, fast, via `torch.hub`) vs TrOCR (`microsoft/trocr-base-printed`, heavier ~340 MB but ~2× recall on difficult text). Both are plumbed in behind the same batch interface.
- `--samples-per-track` — default 15. Raise if tracks are long-lived and OCR coverage is too low.
- `--ocr-min-conf` — default 0.4. Lower to widen coverage at the cost of more false positives.
- `--pose-model` — default `yolo11n-pose.pt` (~6 MB, fast). Pass `yolo11x-pose.pt` for the best YOLO11 pose or `yolo26l-pose.pt` (~55 MB, newer architecture) for better keypoint recall on dark or motion-blurred torsos; both are auto-downloaded into `models/`.

Typical coverage on a 60 s clip: 11–23 % of tracks numbered (depending on source video quality and OCR engine). This is enough for Phase 1.6 below to seed entity merges.

### Phase 1.6 — Entity Re-ID clustering

```bash
python src/phase1_6_entities.py \
  runs/match01/tracks.json \
  runs/match01/tracks_teams.json \
  runs/match01/tracks_identified.json \
  path/to/match.mp4
# writes runs/match01/tracks_entities.json

# then re-render the annotated video — phase6_annotate auto-picks it up
python src/phase6_annotate.py runs/match01/tracks.json runs/match01/tracks_identified.json path/to/match.mp4 \
  --output runs/match01/annotated_entities.mp4
```

Collapses the fragmented tracks from Phase 1 into stable entities (one entity = one player/goalie/ref). Strategy: per-track OSNet x0_25 medoid appearance embedding, greedy merge under hard constraints — same team (from Phase 1.5, with confidence threshold), zero temporal overlap, no OCR-number conflict. OCR-matching pairs get a high merge bonus (10× base similarity). See `docs/phase_1_6_design.md` for the full rationale.

On our 60s test videos, 167–250 Phase-1 tracks collapse to 22–40 entities. Doesn't replace the tracker — it post-processes its output.

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

With ~12 real entities on a roller rink (2 × (4 skaters + 1 goalie) + 1–2 refs + 1 puck), the default ByteTrack tracker produces 150–450 track IDs per 60 s clip because occlusions and camera cuts break identity. Three mitigations, in order of impact:

| Approach | Effect on test12 (250 player tracks) | Notes |
|---|---|---|
| `--tracker bytetrack.yaml` (default) | 250 tracks | Baseline. |
| `--tracker configs/bytetrack_tuned.yaml` | ~28 % fewer tracks | Longer memory (3 s buffer) + looser matching. Same speed. |
| `--tracker configs/botsort_reid.yaml` | similar count, longer individual tracks | BoT-SORT + GMC + ReID (appearance from YOLO backbone). |
| **Phase 1.6 entity clustering** (`src/phase1_6_entities.py`) | **250 → 40 entities (test12), 167 → 24 (test13)** | Post-hoc OSNet embedding + team + non-overlap + OCR constraint. Works on top of any tracker. |

None of these hit the ideal ~12 entities on fragmented source video. Phase 1.6 takes you most of the way; the rest is capped by OCR recall on small/motion-blurred numbers and by ambiguous team colours — both source-quality issues.

## Workflow tips

- Trim to a 60-second test clip first: `ffmpeg -i full_match.mp4 -ss 0 -t 60 -c copy clip.mp4`
- Iterate Phase 2/6 parameters on a single Phase 1 run — you don't need to re-detect each time
- Open `tracks.json` to inspect the data structure for custom analytics
- The `--debug-overlay` output is your best friend for understanding why the camera moves the way it does
- Always pass `python -u` for long runs — stdout is block-buffered by default, which makes progress invisible
- Outputs are kept incrementally: `runs/test01/`, `runs/test02/`, … never overwrite a previous run, even if it failed

## Roadmap

See `CLAUDE.md` for full roadmap and architecture notes.
