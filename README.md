# RILH-AI-Vision

**Roller Inline Hockey — AI Vision.** Open-source computer-vision pipeline: automatic match recording, broadcast-style virtual follow-cam, and post-match analytics. Built from open-source modules + custom code only.

## Project phases

The project is organised in **7 phases**. The first 5 are part of the
per-run pipeline (orchestrated by `src/run_project.py`); Phase 6 and Phase 7 are
separate concerns that consume the run folders.

| # | Phase | In pipeline? | Stages | Status |
|---|---|---|---|---|
| **Phase 1** | Detect & track | ✅ yes | a (detect), b (teams), c (numbers), d (entities), e (annotate) | ✅ implemented |
| **Phase 2** | Virtual follow-cam | ✅ yes | a (followcam) | ✅ implemented |
| **Phase 3** | Rink calibration | ✅ yes (tolerant of failure) | a (rink keypoints) | ❌ parked — HockeyRink doesn't transfer to roller rinks; needs 200-300 annotated frames |
| **Phase 4** | Event detection | ✅ yes (stub) | a (events) | ⏳ stub no-op — placeholder for goals / shots / fouls via temporal action models |
| **Phase 5** | Statistics creation | ✅ yes (stub) | a (stats) | ⏳ stub no-op — placeholder for per-player / per-team aggregation |
| **Phase 6** | Web platform | ❌ external | — | ⏳ later — FastAPI + Next.js consuming the `runs/runNN/` folders |
| **Phase 7** | Multi-cam stitching, live RTMP/HLS, app mobile | ❌ external | — | ⏳ later — infra + mobile control app |

See `CLAUDE.md` for the full test log, design decisions, and open blockers.

## Scripts

- **`src/run_project.py`** — orchestrator. Runs Phase 1 → Phase 5 with per-phase
  gates (`--skip-pN`, `--force`). Pass-through flags for backend / pose /
  OCR weights. Use this for normal end-to-end runs.

Phase 1 stages (the identification sub-pipeline):

- **`src/p1_a_detect.py`** — detection + tracking. Outputs `detections.json`.
- **`src/p1_b_teams.py`** — team classification from per-track jersey color. Outputs `teams.json` + `teams_preview.png`.
- **`src/p1_c_numbers.py`** — jersey-number OCR via PARSeq. Outputs `numbers.json`.
- **`src/p1_d_entities.py`** — entity Re-ID clustering. Outputs `entities.json`.
- **`src/p1_e_annotate.py`** — final video annotation. Reads everything above; outputs `annotated.mp4`.

Other phase scripts:

- **`src/p2_a_followcam.py`** — broadcast follow-cam. Reads `detections.json`; outputs `followcam.mp4`.
- **`src/p3_a_rink.py`** — parked rink-calibration sanity check (kept for future Phase 3 work).
- **`src/p4_a_events.py`** — STUB. Writes `p4_events.json` marker.
- **`src/p5_a_stats.py`** — STUB. Writes `p5_stats.json` marker.

Configs + tools:

- **`configs/bytetrack_tuned.yaml`** — longer-memory ByteTrack config.
- **`configs/botsort_reid.yaml`** — BoT-SORT with GMC + ReID (appearance).
- **`tools/`** — helpers (annotation web UI, dataset builder, fine-tune script, smoke tests). See `CLAUDE.md > Tools` for the inventory.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

First run auto-downloads YOLO11 weights (~50 MB) into `models/`. The HockeyAI weights land there too when you pass `--hockey-model`.

GPU recommended but not required. CPU-only runs at roughly 1/5 of real-time on a modern laptop; a decent GPU runs 2–5x faster than real-time.

## Usage

### One-shot orchestrator (recommended)

The `run_project.py` orchestrator runs Phase 1 → Phase 5 in order with sane defaults
and per-phase gates:

```bash
# Full pipeline with HockeyAI + RILH-fine-tuned PARSeq:
python src/run_project.py videos/match.mp4 --output runs/run23 \
    --hockey-model \
    --pose-model yolo26l-pose.pt \
    --parseq-checkpoint models/parseq_hockey_rilh.pt

# Skip a phase entirely:
python src/run_project.py videos/match.mp4 --output runs/run23 --skip-p3 --skip-p4

# Force re-run all stages even if their outputs already exist:
python src/run_project.py videos/match.mp4 --output runs/run23 --force
```

By default each phase is ON; each stage is skipped if its output file
exists (incremental re-runs cost nothing). The orchestrator forwards
backend / model flags to the relevant stage; everything else uses the
stage script's own default.

The sections below document each stage script for direct standalone use.

### Phase 1 — stage a — Detect & track

```bash
# Recommended: HockeyAI for clean player + puck detection
python src/p1_a_detect.py path/to/match.mp4 --output runs/match01 --hockey-model

# Default: COCO YOLO11 (fast, but puck detection is very weak)
python src/p1_a_detect.py path/to/match.mp4 --output runs/match01
```

Outputs:
- `runs/match01/annotated.mp4` — original video with bounding boxes, IDs, traces
- `runs/match01/detections.json` — per-frame detections, consumed by every downstream stage

Useful flags:
- `--hockey-model` — HockeyAI (YOLOv8m fine-tuned on ice hockey) instead of COCO. Auto-downloads to `models/`. Strongly recommended for anything beyond toy clips.
- `--training-mode` — disable the default 1-puck-per-frame filter. Use for drills with multiple pucks; otherwise the match-mode default keeps only the highest-confidence puck per frame.
- `--tracker bytetrack.yaml` (default) | `configs/bytetrack_tuned.yaml` | `configs/botsort_reid.yaml`
- `--conf 0.3` — detection confidence threshold
- `--imgsz 1280` — inference resolution

### Phase 1 — stage b — Teams

```bash
python src/p1_b_teams.py runs/match01/detections.json path/to/match.mp4 \
  --pose-model yolo26l-pose.pt
# writes runs/match01/teams.json + runs/match01/teams_preview.png
```

Pipeline: YOLO pose → pose-guided torso crop → multi-point dominant color
→ k=2 k-means (HSV default) on skater tracks → goalies classified post-hoc
against those centroids → per-track majority vote.

Tunables: `--space {hsv,bgr}`, `--grid RxC` (default `3x2`),
`--samples-per-track` (default 8), `--pose-model`. Inspect
`teams_preview.png` + `cluster_margin` in the JSON to judge separation.

### Phase 1 — stage c — Numbers

```bash
# Use the RILH-fine-tuned PARSeq Hockey checkpoint (recommended)
python src/p1_c_numbers.py runs/match01/detections.json path/to/match.mp4 \
  --pose-model yolo26l-pose.pt \
  --parseq-checkpoint models/parseq_hockey_rilh.pt
# writes runs/match01/numbers.json
```

Pipeline: YOLO pose → keep back-facing samples → **Koshkina-style dorsal
crop** (full shoulder→hip torso + 5 px pad) → PARSeq inference → digit
filter → per-track majority vote (≥ 2 votes required).

Without `--parseq-checkpoint`, the default is baudm/parseq pretrained on
generic STR (much weaker on jersey numbers). The checkpoint
`models/parseq_hockey_rilh.pt` is produced by
`tools/finetune_parseq_hockey.py` from Maria Koshkina's hockey baseline
+ our 1063 manually-annotated RILH crops.

Tunables: `--samples-per-track` (default 15), `--ocr-min-conf`
(default 0.30), `--pose-model`, `--debug-crops-dir`.

### Phase 1 — stage d — Entities

```bash
python src/p1_d_entities.py \
  runs/match01/detections.json \
  runs/match01/teams.json \
  runs/match01/numbers.json \
  path/to/match.mp4
# writes runs/match01/entities.json
```

Collapses fragmented stage-a tracks into stable entities. Strategy: per-track
OSNet x0_25 medoid appearance embedding + greedy merge under hard constraints
(same team, zero temporal overlap, no OCR conflict). Number-matching pairs
get a 10× merge bonus. See `docs/p1_d_entities_design.md`.

On our 60 s test videos, ~200–400 stage-a tracks collapse to ~20–40 entities.

### Phase 1 — stage e — Annotate

```bash
python src/p1_e_annotate.py \
  runs/match01/detections.json \
  runs/match01/numbers.json \
  path/to/match.mp4 \
  --output runs/match01/annotated.mp4
# auto-discovers teams.json + entities.json next to detections.json if present
```

Final MP4 with team-coloured boxes, `t{id} {G|S} #NN` per-track labels
(track id always shown), gray puck box, short traces. Pass
`--debug-frames-dir` to also dump 1 frame every N for visual review.

### Phase 2 — stage a — Virtual follow-cam

```bash
python src/p2_a_followcam.py runs/match01/detections.json path/to/match.mp4 \
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

[HockeyAI](https://huggingface.co/SimulaMet-HOST/HockeyAI) is a YOLOv8m fine-tuned on ice hockey (SimulaMet-HOST). Seven classes: center ice, faceoff dots, goal frame, goaltender, players, puck, referee. The weights transfer well to roller inline hockey and are auto-downloaded (~52 MB) to `models/HockeyAI_model_weight.pt` on first use. Class IDs are remapped at the source so the `detections.json` output stays uniform across backends (player + goaltender → `class_id=0`, puck → `class_id=32`, referee + rink markers are dropped).

On a typical 60s wide-angle clip (1920×1080 @ 60 fps):

| metric                     | COCO YOLO11n | HockeyAI YOLOv8m |
|----------------------------|--------------|------------------|
| player detections          | 63,224       | 17,860 (more selective) |
| player track IDs           | 1,804        | **433 (~4× more stable)** |
| frames with puck detected  | 0.1%         | **42.6%**        |

HockeyAI is slower (medium vs. nano) but the tracking output is dramatically cleaner and puck data is actually useful. Referees are excluded at the source.

### Stage f still has fallbacks for puck gaps

Even with HockeyAI, the puck is missed in ~50 % of frames. The follow-cam (`Phase 2.a`) handles this with:
1. **Short-term puck memory** — uses last known puck position for ~15 frames after detection drops
2. **Players-centroid fallback** — when the puck is lost too long, the camera tracks the cluster of players

For near-perfect puck tracking, a roller-specific fine-tune is the next step (Phase 4 of the roadmap — see `CLAUDE.md`).

## Tracking stability

With ~12 real entities on a roller rink (2 × (4 skaters + 1 goalie) + 1–2 refs + 1 puck), ByteTrack produces 150–450 track IDs per 60 s clip because occlusions and camera cuts break identity. Three mitigations, in order of impact:

| Approach | Effect on run12 (250 player tracks) | Notes |
|---|---|---|
| `--tracker bytetrack.yaml` (default) | 250 tracks | Baseline. |
| `--tracker configs/bytetrack_tuned.yaml` | ~28 % fewer tracks | Longer memory (3 s buffer) + looser matching. |
| `--tracker configs/botsort_reid.yaml` | similar count, longer individual tracks | BoT-SORT + GMC + ReID (appearance from YOLO backbone). |
| **Phase 1.d entity clustering** | **250 → 40 entities (run12), 167 → 24 (run13)** | Post-hoc OSNet embedding + team + non-overlap + OCR constraint. Works on top of any tracker. |

None of these hit the ideal ~12 entities on fragmented source video. Phase 1.d takes you most of the way; the rest is capped by OCR recall on small/motion-blurred numbers and by ambiguous team colours — both source-quality issues.

## Workflow tips

- Trim to a 60-second test clip first: `ffmpeg -i full_match.mp4 -ss 0 -t 60 -c copy clip.mp4`
- Iterate stage-c / stage-e parameters on a single stage-a run — you don't need to re-detect each time
- Open `detections.json` to inspect the data structure for custom analytics
- The `--debug-overlay` output (Phase 2.a) is your best friend for understanding why the camera moves the way it does
- Always pass `python -u` for long runs — stdout is block-buffered by default, which makes progress invisible
- Outputs are kept incrementally: `runs/run01/`, `runs/run02/`, … never overwrite a previous run, even if it failed
- Outputs are kept incrementally: `runs/test01/`, `runs/test02/`, … never overwrite a previous run, even if it failed

## Roadmap

See `CLAUDE.md` for full roadmap and architecture notes.
