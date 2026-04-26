# RILH-AI-Vision

**Roller Inline Hockey — AI Vision.** Open-source computer-vision pipeline: automatic match recording, broadcast-style virtual follow-cam, and post-match analytics. Built from open-source modules + custom code only.

## Project phases

The project is organised in **7 phases**. The first 5 are part of the
per-run pipeline (orchestrated by `src/run_project.py`); Phase 6 and Phase 7 are
separate concerns that consume the run folders.

| # | Phase | In pipeline? | Stages | Status |
|---|---|---|---|---|
| **Phase 1** | Detect & track | ✅ yes | a (detect), b (teams), c (numbers) | ✅ implemented |
| **Phase 2** | Rink calibration | ✅ yes (tolerant of failure) | a (rink keypoints) | 🚧 **HIGH PRIORITY** — HockeyRink doesn't transfer to roller rinks off the shelf; next step is a 200-300 frame roller-rink keypoint dataset + fine-tune. Unlocks the on-ice / off-ice geometric filter that downstream entity quality depends on. |
| **Phase 3** | Entity recognition + final annotated MP4 | ✅ yes | a (entities), b (annotate) | ✅ implemented |
| **Phase 4** | Event detection | ✅ yes (stub) | a (events) | ⏳ stub no-op — placeholder for goals / shots / fouls via temporal action models |
| **Phase 5** | Statistics creation | ✅ yes (stub) | a (stats) | ⏳ stub no-op — placeholder for per-player / per-team aggregation |
| **Phase 6** | Web platform | ❌ external | — | ⏳ later — FastAPI + Next.js consuming the `runs/runNN/` folders |
| **Phase 7** | Multi-cam stitching, live RTMP/HLS, app mobile | ❌ external | — | ⏳ later — infra + mobile control app |

See `CLAUDE.md` for the full test log, design decisions, and open blockers.

## Scripts

- **`src/run_project.py`** — orchestrator. Runs Phase 1 → Phase 5 with per-phase
  gates (`--skip-pN`, `--force`). Pass-through flags for backend / pose /
  OCR weights. Use this for normal end-to-end runs.

Phase 1 stages (per-frame detection → per-track team / number):

- **`src/p1_a_detect.py`** — detection + tracking. Outputs `p1_a_detections.json`.
- **`src/p1_b_teams.py`** — team classification from per-track jersey color. Outputs `p1_b_teams.json` + `teams_preview.png`.
- **`src/p1_c_numbers.py`** — jersey-number OCR via PARSeq. Outputs `p1_c_numbers.json`.

Phase 2 — rink calibration (high-priority, in progress):

- **`src/p2_a_rink.py`** — HockeyRink keypoint sanity check / future homography. Today HockeyRink doesn't transfer cleanly to roller rinks; orchestrator runs it tolerantly so the wiring is exercised end-to-end while the fine-tune dataset is being built.

Phase 3 — entity recognition + final annotated MP4:

- **`src/p3_a_entities.py`** — entity Re-ID clustering across fragmented Phase 1 tracks. Outputs `p3_a_entities.json`.
- **`src/p3_b_annotate.py`** — final video annotation. Reads everything above; outputs `annotated.mp4`.

Stubs:

- **`src/p4_a_events.py`** — STUB. Writes `p4_a_events.json` marker.
- **`src/p5_a_stats.py`** — STUB. Writes `p5_a_stats.json` marker.

Configs + tools:

- **`configs/bytetrack_tuned.yaml`** — longer-memory ByteTrack config.
- **`configs/botsort_reid.yaml`** — BoT-SORT with GMC + ReID (appearance).
- **`tools/`** — helpers (annotation web UI, dataset builder, fine-tune script, smoke tests). See `CLAUDE.md > Tools` for the inventory.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r docs/requirements.md   # yes, .md — see docs/requirements.md for why
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
python src/run_project.py videos/match.mp4 --output runs/run30 --skip-p4 --skip-p5

# Force re-run all stages even if their outputs already exist:
python src/run_project.py videos/match.mp4 --output runs/run30 --force
```

By default every phase is ON. Each stage is skipped if its output
file exists (incremental re-runs cost nothing). Phase 2 (rink) runs
tolerantly while the fine-tune dataset is being built; if its
keypoint detector fails, downstream phases continue. The orchestrator
forwards backend / model flags to the relevant stage; everything else
uses the stage script's own default.

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
- `runs/match01/p1_a_detections.json` — per-frame detections, consumed by every downstream stage

Useful flags:
- `--hockey-model` — HockeyAI (YOLOv8m fine-tuned on ice hockey) instead of COCO. Auto-downloads to `models/`. Strongly recommended for anything beyond toy clips.
- `--training-mode` — disable the default 1-puck-per-frame filter. Use for drills with multiple pucks; otherwise the match-mode default keeps only the highest-confidence puck per frame.
- `--tracker bytetrack.yaml` (default) | `configs/bytetrack_tuned.yaml` | `configs/botsort_reid.yaml`
- `--conf 0.3` — detection confidence threshold
- `--imgsz 1280` — inference resolution

### Phase 1 — stage b — Teams

```bash
python src/p1_b_teams.py runs/match01/p1_a_detections.json path/to/match.mp4 \
  --pose-model yolo26l-pose.pt
# writes runs/match01/p1_b_teams.json + runs/match01/teams_preview.png
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
python src/p1_c_numbers.py runs/match01/p1_a_detections.json path/to/match.mp4 \
  --pose-model yolo26l-pose.pt \
  --parseq-checkpoint models/parseq_hockey_rilh.pt
# writes runs/match01/p1_c_numbers.json
```

Pipeline: YOLO pose → keep back-facing samples → **Koshkina-style dorsal
crop** (full shoulder→hip torso + 5 px pad) → PARSeq inference → digit
filter → per-track majority vote (≥ 2 votes required).

Without `--parseq-checkpoint`, the default is baudm/parseq pretrained on
generic STR (much weaker on jersey numbers). The checkpoint
`models/parseq_hockey_rilh.pt` is produced by
`tools/finetune_parseq_hockey.py` from Maria Koshkina's hockey baseline
+ our 2078 manually-annotated RILH crops (across 6 videos, 48 unique numbers).

Tunables: `--samples-per-track` (default 15), `--ocr-min-conf`
(default 0.30), `--pose-model`, `--debug-crops-dir`.

### Phase 2 — stage a — Rink calibration (high priority)

> 🚧 In progress. HockeyRink (ice) doesn't transfer to roller rinks
> off the shelf — the model recognises "a rink" but collapses every
> keypoint into a small cluster instead of localising them. The next
> step is a 200-300 frame roller-rink keypoint dataset + fine-tune.
> The orchestrator runs this stage by default but tolerates the
> failure so downstream phases keep going.

```bash
python src/p2_a_rink.py path/to/match.mp4 --output runs/match01
```

Once the fine-tune lands this stage will produce `p2_a_rink_keypoints.json`,
which Stage 3.a will use as a geometric on-ice / off-ice filter — that's the
single biggest expected fix for spectator pollution and ref leakage in
entity recognition.

### Phase 3 — stage a — Entities

```bash
python src/p3_a_entities.py \
  runs/match01/p1_a_detections.json \
  runs/match01/p1_b_teams.json \
  runs/match01/p1_c_numbers.json \
  path/to/match.mp4
# writes runs/match01/p3_a_entities.json
```

Collapses fragmented stage-1.a tracks into stable entities. Strategy: per-track
OSNet x0_25 medoid appearance embedding + greedy merge under hard constraints
(same team, zero temporal overlap, no OCR conflict). Number-matching pairs
get a 10× merge bonus. See `docs/p3_a_entities_design.md`.

On our 60 s test videos, ~200–400 stage-1.a tracks collapse to ~20–40 entities.

### Phase 3 — stage b — Annotate

```bash
python src/p3_b_annotate.py \
  runs/match01/p1_a_detections.json \
  runs/match01/p1_c_numbers.json \
  path/to/match.mp4 \
  --output runs/match01/annotated.mp4
# auto-discovers p1_b_teams.json + p3_a_entities.json next to p1_a_detections.json if present
```

Final MP4 with team-coloured boxes, `t{id} {G|S} #NN` per-track labels
(track id always shown), gray puck box, short traces. Pass
`--debug-frames-dir` to also dump 1 frame every N for visual review.

## Puck detection — two backends

### Default: COCO YOLO11 (not recommended for puck work)

Uses class 32 ("sports ball") as a puck proxy. Almost never catches a roller hockey puck (~0.1% of frames in internal tests). Fine when you only care about players.

### `--hockey-model`: HockeyAI

[HockeyAI](https://huggingface.co/SimulaMet-HOST/HockeyAI) is a YOLOv8m fine-tuned on ice hockey (SimulaMet-HOST). Seven classes: center ice, faceoff dots, goal frame, goaltender, players, puck, referee. The weights transfer well to roller inline hockey and are auto-downloaded (~52 MB) to `models/HockeyAI_model_weight.pt` on first use. Class IDs are remapped at the source so the `p1_a_detections.json` output stays uniform across backends (player + goaltender → `class_id=0`, puck → `class_id=32`, referee + rink markers are dropped).

On a typical 60s wide-angle clip (1920×1080 @ 60 fps):

| metric                     | COCO YOLO11n | HockeyAI YOLOv8m |
|----------------------------|--------------|------------------|
| player detections          | 63,224       | 17,860 (more selective) |
| player track IDs           | 1,804        | **433 (~4× more stable)** |
| frames with puck detected  | 0.1%         | **42.6%**        |

HockeyAI is slower (medium vs. nano) but the tracking output is dramatically cleaner and puck data is actually useful. Referees are excluded at the source.

### Puck-gap handling downstream

Even with HockeyAI, the puck is missed in ~50 % of frames. Downstream stages
that need a continuous puck signal (Phase 4 events, future virtual cam)
will need short-term memory + players-centroid fallback like the previous
follow-cam stage had. For near-perfect puck tracking, a roller-specific
fine-tune is the next step (Phase 4 fine-tune — see `CLAUDE.md`).

## Tracking stability

With ~12 real entities on a roller rink (2 × (4 skaters + 1 goalie) + 1–2 refs + 1 puck), ByteTrack produces 150–450 track IDs per 60 s clip because occlusions and camera cuts break identity. Three mitigations, in order of impact:

| Approach | Effect on run12 (250 player tracks) | Notes |
|---|---|---|
| `--tracker bytetrack.yaml` (default) | 250 tracks | Baseline. |
| `--tracker configs/bytetrack_tuned.yaml` | ~28 % fewer tracks | Longer memory (3 s buffer) + looser matching. |
| `--tracker configs/botsort_reid.yaml` | similar count, longer individual tracks | BoT-SORT + GMC + ReID (appearance from YOLO backbone). |
| **Stage 3.a entity clustering** | **250 → 40 entities (run12), 167 → 24 (run13)** | Post-hoc OSNet embedding + team + non-overlap + OCR constraint. Works on top of any tracker. |

None of these hit the ideal ~12 entities on fragmented source video. Stage 3.a takes you most of the way; the rest is capped by OCR recall on small/motion-blurred numbers and by ambiguous team colours — both source-quality issues. The next step (Phase 2 rink fine-tune → on-ice/off-ice geometric filter) is expected to drop spectator entities and ref leakage substantially.

## Workflow tips

- Trim to a 60-second test clip first: `ffmpeg -i full_match.mp4 -ss 0 -t 60 -c copy clip.mp4`
- Iterate stage-1.c / stage-3.b parameters on a single stage-1.a run — you don't need to re-detect each time
- Open `p1_a_detections.json` to inspect the data structure for custom analytics
- The `--debug-overlay` output (Stage 2.a) is your best friend for understanding why the camera moves the way it does
- Always pass `python -u` for long runs — stdout is block-buffered by default, which makes progress invisible
- Outputs are kept incrementally: `runs/run01/`, `runs/run02/`, … never overwrite a previous run, even if it failed
- Outputs are kept incrementally: `runs/test01/`, `runs/test02/`, … never overwrite a previous run, even if it failed

## Roadmap

See `CLAUDE.md` for full roadmap and architecture notes.
