# Project context for Claude Code

## Project: RILH-AI-Vision

Computer-vision pipeline for roller inline hockey: automatic AI-driven
match recording, broadcast-style virtual follow-cam, and post-match
analytics. Built from open-source modules + custom code only.

## Current status
- **Stage 1.a (Detect & track)** ✅ : dual-backend detection (YOLO11 COCO or
  HockeyAI) + configurable tracker (ByteTrack default, BoT-SORT/ReID
  available). Match-mode default = max 1 puck per frame; pass
  `--training-mode` to lift that.
- **Stage 1.b (Teams)** ✅ : team clustering via pose-based torso crop
  (YOLO pose shoulders→hips, bbox-fallback for dark jerseys). Engine
  pluggable via `--team-engine`:
  - `hsv` (default) — multi-point dominant color averaging (3×2 grid)
    + per-crop k=2 k-means (HSV space). User-validated as "practically
    perfect" on run12 except referees (HockeyAI mislabels them 'player'
    on roller — can't filter by class_name) and dark-vs-dark teams
    (France-Monde plateau — k-means can't separate dark-blue from
    dark-green torsos with colour alone).
  - `osnet` — k=2 on per-track OSNet x0_25 medoid embeddings (same
    model Stage 3.a uses for entity Re-ID). Captures pattern + colour,
    so it tends to beat HSV on similar-coloured teams.
  - `siglip` — Roboflow recipe: SigLIP encoder → mean-pool → UMAP-3D
    → k-means k=2. Heaviest engine (~370 MB checkpoint download on
    first run) but zero training data required.
  - `contrastive` — Koshkina 2021: small CNN trained with triplet loss
    + 50 % grayscale aug on torso crops. Trained per-deployment via
    `tools/finetune_contrastive_team.py`. Output checkpoint at
    `models/contrastive_team_rilh.pt`. The grayscale augmentation is
    the load-bearing trick that handles dark-vs-dark teams.
  Plus an orthogonal `--ref-classifier` flag that loads a binary head
  trained on `track_truth.json` (`tools/finetune_ref_classifier.py`,
  output `models/ref_classifier_rilh.pt`) and tags each track with
  `is_referee` + `ref_score` post-hoc, no matter which team engine ran.
  `is_goaltender` per track still uses HockeyAI majority (>50 % of
  detections tagged `goaltender`). The four engines can be benchmarked
  side-by-side via `tools/bench_team_engines.py` against a truth set
  produced by `tools/annotate_tracks.py`. **6-clip bench (2100 truth
  tracks, 1497 team-tagged) ranks contrastive 0.896 > siglip 0.838 >
  hsv 0.788 > osnet 0.717**: contrastive wins on 4 of 6 clips and
  never drops below 0.74; SigLIP wins on the two dark-vs-dark cases
  (run26 / run27). See the test log for the full per-clip table.
  Default stays `hsv` for backwards compatibility; `--team-engine
  contrastive` is the recommended analytical choice. The ref
  classifier (v2 on 2100 tracks) ships at test recall 0.78 /
  precision 0.53 — useful for flagging but the in-bench precision
  (0.10) drops because the sigmoid threshold of 0.5 is too loose for
  the natural ref/non-ref prevalence; threshold tuning is the next
  iter. Opt-in via `--ref-classifier` only.
- **Stage 1.c (Numbers)** ✅ : per-track jersey-number OCR via **PARSeq
  Hockey + RILH fine-tune**. Single engine (PARSeq); the previous
  TrOCR / together engines were removed in the refactor — the digit
  number is now the sole player-clustering key, and TrOCR brought
  receipt-vocabulary hallucinations + 340 MB of weights for limited
  benefit. PARSeq accepts a custom `--parseq-checkpoint` to load
  Maria Koshkina's hockey baseline
  ([github.com/mkoshkina/jersey-number-pipeline](https://github.com/mkoshkina/jersey-number-pipeline),
  CC-BY-NC license) or our `parseq_hockey_rilh.pt` (trained on 2078
  numbered RILH crops + 3119 X-negatives across 6 videos, 48 unique
  numbers, via `tools/finetune_parseq_hockey.py`). Crop strategy is
  Koshkina-style (bbox of 4 torso keypoints + 5 px padding). When the
  checkpoint is loaded, letterbox is skipped (Koshkina trained on
  direct-resize). **Quality on held-out test set (407 crops, 6 videos):
  exact match 35.4 % (Koshkina alone) → 96.6 % (Koshkina + RILH)** —
  best epoch 3, val 0.968. The earlier 4-video test-set scored 43 % →
  97 %; the new test set is harder (more diversity / more X-negatives /
  Video 06 + 07 added) and the model still holds. On Video 04 + 05
  truth tracks: recall 55–58 % → **81–84 %** ; precision 63–68 % →
  **95–96 %**. Names are not identified — PARSeq Hockey is digit-only;
  a name fine-tune is a separate later step.
- **Stage 2.a (Rink calibration)** 🚧 HIGH PRIORITY (was Phase 3, promoted).
  HockeyRink keypoints don't transfer to roller rinks off the shelf —
  the model recognises "a rink" but collapses all 56 keypoints into a
  small cluster instead of localising them (see run05–run07). Promoted
  from "parked" because rink calibration is the cleanest way to add
  a geometric on-ice / off-ice classifier, which directly fixes the
  spectator-pollution and ref-mis-classification limits in entity
  recognition (Phase 3). Unblocking requires a fine-tune dataset —
  200–300 annotated roller-rink frames — which is now the top
  out-of-pipeline blocker. The orchestrator still tolerates failure on
  this stage so the wiring stays exercised end-to-end.
- **Stage 3.a (Entity recognition)** ✅ (was Stage 1.d). Post-hoc
  **Re-ID clustering** that collapses fragmented Phase 1 tracks into
  stable entities (one entity = one real player / goalie). Uses OSNet
  x0_25 (via `torchreid`) medoid embedding per track + greedy merge
  under **same-team constraint** (from Stage 1.b), **strict temporal
  non-overlap**, and **OCR bonus** (from Stage 1.c). OCR conflicts are
  a hard block. Output: `p3_a_entities.json`, consumed by Stage 3.b
  (annotate), Phase 4 (events), and Phase 5 (stats). On run12 (250
  tracks → 40 entities), run13 (167 tracks → 24 entities). Doesn't
  replace the tracker — it post-processes its output. Entity-level
  `is_goaltender` is weighted by frame coverage to absorb HockeyAI
  class flips. Design doc: `docs/p3_a_entities_design.md`.
- **Stage 3.b (Annotate)** ✅ (was Stage 1.e). Final MP4 with team-
  coloured boxes, per-track label `t{id} {G|S} #NN` (track id always
  shown), dark-gray puck box, short traces. Auto-discovers
  `p1_b_teams.json` and `p3_a_entities.json` next to
  `p1_a_detections.json` if present. Optional `--debug-frames-dir`
  writes 1 PNG every N frames.
- **Phase 2 follow-cam (former)** — REMOVED from the project for now.
  Output wasn't usable and rink calibration is the bottleneck for
  every other downstream improvement. Will resurface as its own phase
  later, after Phase 2 (rink) is unblocked.

## Architecture

The project has **two levels** of organisation:

### Level 1 — project phases (axis of `graph3D` / `run_project.py`)

Big chunks of work, each with its own concern:

| Phase | Name | In pipeline? |
|---|---|---|
| **Phase 1** | Detect & track (a detect, b teams, c numbers) | ✅ orchestrated |
| **Phase 2** | Rink calibration | 🚧 orchestrated, **HIGH PRIORITY** (tolerant of failure pending the fine-tune dataset) |
| **Phase 3** | Entity recognition (a entities, b annotated MP4) | ✅ orchestrated |
| **Phase 4** | Event detection | ✅ orchestrated (stub) |
| **Phase 5** | Statistics creation | ✅ orchestrated (stub) |
| **Phase 6** | Web platform (FastAPI + Next.js) | ❌ external — consumes `runs/runNN/` folders |
| **Phase 7** | Multi-cam stitching, live RTMP/HLS, app mobile | ❌ external — infra + mobile |

The orchestrator `src/run_project.py` runs Phase 1 → Phase 5 in sequence with
per-phase gates (`--skip-pN`, `--force`). All five are ON by default; the
former Phase 2 (Virtual follow-cam) was removed in this restructure (see
`Phase 2 follow-cam (former)` above for the rationale). Phase 6 and Phase 7
are services / infra that live outside the per-run pipeline.

### Level 2 — internal stages of each phase

Each phase decomposes into one or more stages, named `pN_x_*.py`:

**Phase 1 — Detect & track** (3 stages, sequential):

1. `src/p1_a_detect.py` → `p1_a_detections.json` (per-frame bboxes, class IDs,
   persistent track IDs). HockeyAI YOLO + ByteTrack. Match-mode default
   keeps top-1 puck per frame; pass `--training-mode` for drills.
2. `src/p1_b_teams.py` → `p1_b_teams.json` + `teams_preview.png`: team_id
   (0/1) per player track via k=2 on pose-based torso color.
3. `src/p1_c_numbers.py` → `p1_c_numbers.json`: per-track jersey number via
   YOLO pose + PARSeq (`--parseq-checkpoint models/parseq_hockey_rilh.pt`
   for our RILH-fine-tuned model). Crop is Koshkina-style.

**Phase 2 — Rink calibration** (1 stage, high priority):

- `src/p2_a_rink.py` → `p2_a_rink_keypoints.json` (intended). HockeyRink
  off the shelf doesn't transfer to roller — the next step is a roller-
  specific keypoint dataset + fine-tune. Orchestrator runs it by default
  and tolerates the failure so the wiring stays exercised.

**Phase 3 — Entity recognition** (2 stages, sequential):

1. `src/p3_a_entities.py` → `p3_a_entities.json`: fragmented Phase 1 tracks
   collapsed into stable entities via OSNet embeddings + team/overlap/OCR
   constraints. Was Stage 1.d.
2. `src/p3_b_annotate.py` → annotated MP4 with team-coloured boxes +
   `t{id} {G|S} #NN` labels. Auto-discovers `p1_b_teams.json` +
   `p3_a_entities.json`. Was Stage 1.e.

**Phase 4 — Event detection** (1 stage, stub):

- `src/p4_a_events.py` → `p4_a_events.json` marker. Real impl pending.

**Phase 5 — Statistics creation** (1 stage, stub):

- `src/p5_a_stats.py` → `p5_a_stats.json` marker. Real impl pending.

Why multi-pass: detection is the slow step. Decoupling lets us iterate on
cinematography, identification, and analytics without re-running inference.

## Key design choices

### Detector backends (Stage 1.a)
Two detector backends, selectable at runtime in `p1_a_detect.py`:
- Default — **COCO-pretrained YOLO**, classes 0 (person) and 32 (sports ball).
  Player detection is solid; puck detection via "sports ball" is unreliable
  (<1% of frames on roller). Weights are chosen via `--model`:
  - `yolo11n.pt` / `yolo11m.pt` (default) / `yolo11x.pt`
  - `yolo26l.pt` — YOLO26 large (~51 MB, released January 2026). Newer
    architecture, stronger than YOLO11m on COCO benchmarks. Auto-downloaded
    on first use. Still doesn't solve the puck class — that needs HockeyAI.
- `--hockey-model` — **[HockeyAI](https://huggingface.co/SimulaMet-HOST/HockeyAI)**,
  a YOLOv8m fine-tuned on ice hockey with 7 native classes (center ice,
  faceoff dots, goal frame, goaltender, players, puck, referee).
  Auto-downloaded to `models/HockeyAI_model_weight.pt`. Transfers well to
  roller inline hockey. Classes are remapped at the source so the output
  schema stays uniform across backends: player + goaltender → `class_id=0`,
  puck → `class_id=32`, referee + rink markers are dropped.

Both backends write the same `p1_a_detections.json` schema (plus a `class_name`
string per detection), so every downstream stage is backend-agnostic.

**Match vs training mode.** Stage 1.a enforces 1-puck-per-frame by default
(real match conditions): when multiple puck detections appear in the
same frame, only the highest-confidence one is kept (after the tracker
has assigned IDs, so the dropped duplicates simply never reach
`p1_a_detections.json`). Pass `--training-mode` to disable the filter —
useful for drills where multiple pucks are intentionally on the ice.

### Tracker backends (Stage 1.a)
Tracker is configurable via `--tracker <yaml>`:
- `bytetrack.yaml` (default) — motion-only, fast, fragmented on occlusions.
- `configs/bytetrack_tuned.yaml` — same tracker with `track_buffer=180` and
  `match_thresh=0.9` (longer memory, more permissive association).
- `configs/botsort_reid.yaml` — BoT-SORT with GMC (camera motion
  compensation) and ReID (appearance features from the YOLO backbone).
  Slower but makes individual tracks longer — see run11.

### Jersey identification (Stage 1.c)
`p1_c_numbers.py` pipeline, one pass per track (not per frame — keeps
inference tractable):
1. Sample the N highest-confidence detections for each track_id.
2. Run the **YOLO pose model** on the full frames, match by IoU to the
   tracked bbox. Default is `yolo11n-pose.pt` (~6 MB); `--pose-model
   yolo26l-pose.pt` (~55 MB) is available for better keypoint recall on
   dark / low-contrast / motion-blurred torsos. Both auto-download into
   `models/` on first use.
3. Classify **orientation** from pose keypoints (nose + eyes → front;
   ears without nose → back; shoulders only → side).
4. On back-facing samples, crop the dorsal region **Koshkina-style** (since
   run19/20/21 — replaces the previous tight 15-65 % band): bbox of the
   four torso keypoints (LSHO, RSHO, LHIP, RHIP) plus 5 px padding on
   `x_min`, `x_max`, and `y_min` (no padding on `y_max` so the crop ends
   cleanly at the hips). This produces ~square crops covering the full
   shoulder→hip torso, **2× taller** than the old band. Matches the
   distribution Koshkina trained on, so off-the-shelf inference no
   longer faces a distribution shift.
5. **Letterbox-pad** the crop to PARSeq's 32×128 (1:4 h:w) before
   inference — used by the default baudm/parseq pretrained engine. When
   loading a custom checkpoint via `--parseq-checkpoint` (e.g.
   `models/parseq_hockey.pt` or `parseq_hockey_rilh.pt`), the letterbox
   is **disabled**: Koshkina trained with direct stretch resize and her
   model expects horizontally-stretched digits (skipping the letterbox
   was the difference between 20 % and 80 % exact-match in the run19
   smoke test).
6. Run **PARSeq OCR** (`pytorch_lightning` + `timm` + `nltk` via
   `torch.hub`). Single engine — TrOCR and the PARSeq+TrOCR `together`
   vote were removed in the refactor (the digit number is now the sole
   player-clustering key, names not used). Pass `--parseq-checkpoint
   <path>` to load a custom Lightning checkpoint; the loader auto-
   detects whether keys carry the `model.` wrapper prefix (Koshkina's
   vendored fork doesn't, our fine-tuned checkpoint does). Default
   `models/parseq_hockey_rilh.pt`.
7. Keep digits only, 1–2 chars. **Vote** per track (≥ 2 agreeing votes
   required) — singletons are usually noise from a single bad crop.
8. **Merge** tracks that share the same confident number and don't overlap
   in time — they're the same player with a broken track.

### Entity clustering (Stage 3.a)
`p3_a_entities.py` post-processes fragmented Phase 1 tracks into stable
entities:
1. Per-track **OSNet x0_25 medoid embedding** (512-d, L2-normalised) over
   the top-N confidence detections.
2. Build candidate merge graph: all pairs `(a, b)` with **same team_id
   from Stage 1.b** (with vote_confidence ≥ 0.67 on both sides), **zero
   temporal overlap**, no OCR conflict (same team + different confident
   numbers rejects the pair).
3. Edge weight = `cos_sim(emb_a, emb_b) + 10·1[same_jersey] +
   0.05·1[both_goalie]`. OCR-seeded pairs always win (weight ≥ 10).
4. Greedy merge in descending weight until similarity drops below
   `--sim-threshold` (default 0.65). Re-checks overlap on the merged
   cluster each time.
5. Output: `p3_a_entities.json` with `track_ids` lists, derived
   `team_id` / `is_goaltender` / `jersey_number` / frame ranges, plus a
   list of unmatched singleton tracks.

**Goalie weighting** (added run16, kept after the refactor):
- Stage 1.b `is_goaltender` per track uses a **majority threshold**
  (>50 % of detections tagged `goaltender`), replacing the previous
  `any` rule that flipped a track on a single noisy frame.
- Stage 3.a entity-level `is_goaltender` is **weighted by frame
  coverage**: an entity is goalie only if more than half of its merged
  tracks' total frame count comes from goalie-tagged tracks.
- Stage 1.c aggregation requires **≥2 agreeing votes** for the winning
  number — singletons are usually OCR noise from a single bad crop.

**Spectator handling** — see run17 for why a motion+position filter was
attempted then **removed**: it dropped ~50 % of real player fragments to
catch ~30 spectators, while the 7 spectators HockeyAI tags as goalies
slipped through anyway. The plan is now **Phase 2 (rink calibration)**:
once a roller rink keypoint detector is fine-tuned, every Stage 3.a
candidate can be filtered geometrically ("is this bbox on the ice?")
before clustering. That single change is expected to clean both
spectator pollution and the ref leak into team clusters.

Annotation (`p3_b_annotate.py`):
- Supervision-style boxes/labels/traces. Box color is **forced green or
  blue** per team.
- **Entity-aware**: if `p3_a_entities.json` exists, the label + team come
  from the entity (so every merged fragment shares the same `#NN`, name
  and colour across the video). Otherwise, per-track values from
  `p1_b_teams.json` + `p1_c_numbers.json`.
- Labels read `t{id} {G|S} #NN NAME` — track_id always shown (so user
  can give frame-level feedback), `G`/`S` from `is_goaltender`, `#??`
  if no number, name omitted if not identified.
- **Puck**: rendered as a dark gray bbox (60,60,60) + short trace.
  No label.
- **Spectators**: not filtered until Phase 2 rink calibration is online.
  All tracked detections render — including stationary bystanders that
  HockeyAI tags as players.

## Known limitations (in priority order)
1. **Phase 2 rink calibration is THE current bottleneck** for entity
   quality. HockeyRink (ice) doesn't transfer to roller rinks — the
   model "recognises" a rink but collapses all 56 keypoints to a
   cluster. Unblocking requires 200–300 annotated frames of roller
   rinks + a fine-tune. Once unlocked, an "on-ice / off-ice"
   geometric filter applied to Stage 3.a candidates removes the
   spectator pollution AND most of the ref leakage in one shot —
   that's why this jumped to high priority. Sub-blockers below are
   expected to fall away or shrink dramatically afterwards.
2. **Stationary spectators / refs leaking into team clusters** —
   today they form their own entities or get absorbed into one of the
   two teams (HockeyAI tags refs as `class_name='player'` on roller).
   Run17 tried a motion+position filter in Stage 1.b and reverted it
   (over-filtered real players). Clean fix waits for #1.
3. **Track fragmentation** is real but **partly absorbed** by Stage 3.a
   (run16: 435 tracks → 37 entities, of which only 5 G after the
   goalie majority + frame-coverage rules). HockeyAI still
   intermittently flips goaltender/player class on individual frames,
   but the majority threshold + frame-coverage weighting mostly
   neutralises that.
4. **Video source quality** still affects team-colour clustering and
   any per-frame analysis on small / blurry / oblique players. Number
   OCR is no longer the bottleneck (post run21: 95-96 % precision,
   81-84 % recall on truth tracks via PARSeq Hockey + RILH fine-tune).
   The remaining ~15-20 % missed numbers are mostly tracks where the
   back is never cleanly visible.
5. **Puck detection quality** is workable (~43% of frames with HockeyAI vs
   <1% with COCO) but drops on motion blur, small pucks, and uneven lighting.
   Roller-specific fine-tune would narrow the gap.
6. No event detection yet (Phase 4).
7. No statistics yet (Phase 5).
8. Single-camera assumption. Multi-camera stitching = Phase 7.

## Conventions
- Python 3.10+ (project uses 3.12 in the dev venv)
- All paths via `pathlib.Path`
- CLI scripts use `argparse`
- All model weights live under `models/` (both explicit downloads and
  Ultralytics auto-downloads — `p1_a_detect.py` routes bare YOLO
  names like `yolo11m.pt` into `models/` via `resolve_model_path`).
- Tracker configs live under `configs/` (YAML)
- Outputs go under `runs/testNN/` (gitignored). **Never overwrite** a
  previous run — incremental numbering is the convention, so past results
  stay available for comparison.
- Don't commit videos (`videos/`) or model weights (`*.pt`)

## Repo layout
```
src/
├── run_project.py         — orchestrator (Phase 1 → Phase 5 with per-phase gates)
├── p1_a_detect.py         — Phase 1 stage a — detect & track
├── p1_b_teams.py          — Phase 1 stage b — teams
├── p1_c_numbers.py        — Phase 1 stage c — numbers (PARSeq OCR)
├── p2_a_rink.py           — Phase 2 stage a — rink calibration (high prio)
├── p3_a_entities.py       — Phase 3 stage a — entities (Re-ID merge)
├── p3_b_annotate.py       — Phase 3 stage b — annotate (final MP4)
├── p4_a_events.py         — Phase 4 stage a — event detection (STUB)
└── p5_a_stats.py          — Phase 5 stage a — statistics (STUB)

configs/                   — tracker YAMLs (bytetrack_tuned, botsort_reid)
docs/                      — design notes (p3_a_entities_design.md)
models/                    — model weights (gitignored). Ultralytics YOLOs,
                              HockeyAI, parseq_hockey.pt (Koshkina),
                              parseq_hockey_rilh.pt (our fine-tune).
videos/                    — source clips (gitignored)
runs/                      — pipeline outputs per run (gitignored)
data/                      — license-clean datasets (committed). Today:
                              data/jersey_numbers/ (5197 crops + annotations
                              + train/val/test splits + LICENSE/README).
tools/                     — utilities (annotation web UI, dataset/splits
                              builders, fine-tune script, smoke tests).
                              Not pipeline stages.
graphify-out/              — local 3D visualization (mostly gitignored;
                              regen.py + orchestration.json + style.css +
                              REGEN.md tracked as source).
```

## Tools (`tools/`)
- `tools/annotate_crops.py` — localhost web UI for manually labeling
  jersey crops. Pre-fills each crop's input with the OCR engine's
  filename hint, saves incrementally, resumes at first un-annotated.
  Used to produce `data/jersey_numbers/annotations.json`.
- `tools/build_jersey_dataset.py` — consolidates the runs-based crops
  into the portable `data/jersey_numbers/` dataset.
- `tools/build_jersey_splits.py` — stratified 80/10/10 train/val/test
  splits with X-negatives subsampled to balance positives.
- `tools/finetune_parseq_hockey.py` — minimal PyTorch loop fine-tuning
  Koshkina's PARSeq Hockey on the user's RILH crops. Outputs
  `models/parseq_hockey_rilh.pt`. Self-evaluates baseline vs fine-tuned
  on the held-out test set.
- `tools/smoke_parseq_hockey.py` — standalone smoke test that loads the
  Koshkina checkpoint into baudm/parseq architecture and predicts on N
  random annotated crops, printing a table of truth vs prediction.
- `tools/annotate_tracks.py` — track-level (not crop-level) web UI for
  the Stage 1.b team-engine bench. Walks every player track in N run
  folders, shows up to 6 thumbnail crops, single-key shortcuts for
  team A / team B / referee / not-a-player. Output:
  `data/jersey_numbers/track_truth.json` with thumbnails cached under
  `data/jersey_numbers/_track_thumbs/`.
- `tools/bench_team_engines.py` — compares `hsv` / `osnet` / `siglip` /
  `contrastive` (+ optional ref classifier) on existing run folders
  using `track_truth.json`. Re-runs only Stage 1.b (detection is the
  slow step and stays cached). Per-engine output saved as
  `runs/runNN/p1_b_teams_<engine>.json` so the canonical
  `p1_b_teams.json` is untouched. Produces
  `runs/bench_team_engines_YYYYMMDD_HHMMSS/{results.json, summary.json,
  summary.txt}`.
- `tools/finetune_ref_classifier.py` — trains a 512→64→1 MLP head on
  frozen OSNet x0_25 embeddings to classify referee vs non-referee
  torso crops. Reads truth from `track_truth.json`, writes
  `models/ref_classifier_rilh.pt`. Track-level negatives = team A/B
  positives (X tracks excluded as noisy negatives).
- `tools/finetune_contrastive_team.py` — Koshkina 2021 small CNN
  (3 conv + 2 FC, 128-d L2-normalised output) trained with triplet
  loss + 50 % grayscale augmentation (the load-bearing trick for
  similar-coloured teams). Writes `models/contrastive_team_rilh.pt`.

## Visualization (`graphify-out/` — local-only)
3D interactive knowledge graph of the pipeline (~225 nodes, ~325
edges, ~13 communities). **Two-level architecture pinned on an X-axis
rail**: 7 project phases (Phase 1-5 cyan = orchestrated, Phase 6-7
amber = external) with vertical sub-pipelines of internal stages
(Stage 1.a … Stage 5.a, smaller + dimmer than their parent phase).
Script files (`pN_x_*.py`) are pinned next to their stage in soft
mint. Anything not reachable from any phase via BFS is dropped into
an "Orphans" cluster past Phase 7 with a visible gap (red rings) so
dead/unwired files read at a glance.

Built by `graphify` (PyPI `graphifyy`) for AST extraction from
`src/*.py` + a custom 3D renderer (`3d-force-graph` / ThreeJS) that
merges that with hand-curated pipeline metadata (phase order, data
flow, design rationale) from `orchestration.json`.

Regenerate after code or doc changes (~2 s, zero LLM tokens):
```bash
/Users/nico/.local/share/uv/tools/graphifyy/bin/python graphify-out/regen.py
```
The whole `graphify-out/` folder is gitignored. Full doc + edge
legend in `graphify-out/graphify.md` (PDF: `graphify-out/graphify.pdf`).

## Test run log

Condensed: only the runs cross-referenced from this file (status, design choices, limitations). Full per-run journal — including iterations that got reverted or were stepping stones to a kept run — lives in `docs/experiments.md`.

- **2026-04-27 6-clip team-engine bench** — full pass with ref classifier
  v2 + contrastive v3 (both retrained on the expanded 2100-track truth
  covering all 6 clips run24-run29). The contrastive engine now wins
  decisively across the board: **+10.8 pts vs HSV overall, top accuracy
  on 4 of 6 clips, never below 0.74**. SigLIP is the only engine
  that beats it on a clip (run26 dark-vs-dark, 0.924 vs 0.819, and
  run27 0.933 vs 0.906). OSNet underperforms on the Villeneuve-Vierzon
  and Garges-Rouen clips (0.625 / 0.611) — its embedding doesn't help
  when colours are already easy.

  **6-clip bench (2100 truth tracks, 1497 team-tagged across 6 clips):**

  | engine        | team_acc | Δ vs hsv |
  |---------------|---------:|---------:|
  | hsv           |    0.788 |   —      |
  | osnet         |    0.717 |   −7.1   |
  | siglip        |    0.838 |   +5.0   |
  | **contrastive** | **0.896** | **+10.8** |

  Per-clip:

  | clip                              | hsv   | osnet | siglip   | contrastive |
  |-----------------------------------|-------|-------|----------|-------------|
  | run24 (clip60)                    | 0.546 | 0.734 | 0.555    | **0.843**   |
  | run25 (clip60-2)                  | 0.665 | 0.591 | 0.591    | **0.740**   |
  | run26 (Video04 France-Monde)      | 0.688 | 0.910 | **0.924**| 0.819       |
  | run27 (Video05 Villeneuve-Vierzon)| 0.839 | 0.625 | **0.933**| 0.906       |
  | run28 (Video06 Grenoble-Varces)   | 0.932 | 0.835 | 0.965    | **0.975**   |
  | run29 (Video07 Garges-Rouen)      | 0.881 | 0.611 | 0.954    | **0.975**   |

  **Ref classifier v2** retrained on 2100 truth (vs 1099 v1):
    - Test accuracy: 0.964 (up 0.947 → 0.964)
    - Precision: 0.531 (up 0.373 → 0.531) — 5× more data lifted precision
      cleanly
    - Recall: 0.782 (similar)
    - In-bench performance though: recall 0.663, precision 0.104. The
      bench measures across all 1497 team-tagged tracks where many
      team-tagged tracks get high ref_score because the threshold
      (sigmoid > 0.5) is too loose for the natural prevalence (44 refs
      / 2100). Sigmoid threshold tuning is the next iter — tightening
      to ~0.85 should lift precision into the 0.4-0.6 range without
      losing too much recall.

  **Decision**: leave `hsv` as orchestrator default (no breaking change),
  promote `--team-engine contrastive` as the recommended choice for any
  new analytical run. Keep ref classifier opt-in via `--ref-classifier`
  until precision is tuned.

- **2026-04-27 team-engine plumbing + first bench** — Stage 1.b refactored
  to support pluggable team engines (`--team-engine
  {hsv,osnet,siglip,contrastive}`) + an orthogonal `--ref-classifier`
  flag. Default stays `hsv` so every prior run reproduces unchanged.
  Three new engines shipped wired-up; the contrastive engine needs
  `models/contrastive_team_rilh.pt` (trained via
  `tools/finetune_contrastive_team.py` from the truth set produced by
  `tools/annotate_tracks.py`). Bench harness `tools/bench_team_engines.py`
  re-runs only Stage 1.b (detections stay cached) and writes engine-
  tagged `p1_b_teams_<engine>.json` so no canonical output is
  overwritten. Scoring is permutation-invariant per Koshkina 2021.

  **First bench results — 4 engines × ref on/off, 1099 truth tracks
  across run24-run27 (run28/29 not yet annotated):**

  | engine        | team_acc | Δ vs hsv | best clip                   |
  |---------------|---------:|---------:|------------------------------|
  | hsv           |    0.672 |   —      | run27 (0.838) Villeneuve-Vierzon (clear colours) |
  | osnet         |    0.719 | +4.7 pt  | run26 (0.910) France-Monde (dark-vs-dark)        |
  | siglip        |    0.715 | +4.3 pt  | run26 (0.924) — narrow win over osnet on dark-vs-dark |
  | **contrastive** | **0.775** | **+10.3 pt** | run24 (0.755), run25 (0.684), run27 (0.862) — most consistent across clips |

  Per-clip table:

  | clip  | hsv   | osnet | siglip | contrastive |
  |-------|-------|-------|--------|-------------|
  | run24 | 0.546 | 0.734 | 0.555  | **0.755**   |
  | run25 | 0.665 | 0.591 | 0.591  | **0.684**   |
  | run26 | 0.688 | 0.910 | **0.924** | 0.840    |
  | run27 | 0.838 | 0.701 | **0.916** | 0.862    |

  Verdict: **contrastive is the new default candidate** (best overall,
  no clip below 0.68). Add the Koshkina 50 % grayscale aug + intra-clip
  triplet sampling — the v1 trainer cross-clip triplets failed to
  converge (holdout pair-acc 0.485, worse than random) because the A/B
  label is per-clip and absolute identity isn't preserved across
  clips; v2 enforces intra-clip triplets and converges cleanly.

  **Ref classifier** (`models/ref_classifier_rilh.pt`, 512 → 64 → 1
  MLP head on frozen OSNet x0_25 embeddings, 30 epochs):
    - Test accuracy: 0.947
    - **Recall: 0.734** (catches 27 of 37 truth refs across the 4 clips)
    - **Precision: 0.110** (false alarm rate is high — every engine
      gets the same 0.11 number because the head doesn't depend on the
      team engine)
    - Verdict: **needs work before going to default.** 73 % of refs
      caught is great, but 89 % of "ref" tags wrong means we'd pollute
      the player buckets if we hard-blocked them. Likely fixes:
      (a) raise the sigmoid threshold above 0.5 to trade recall for
      precision; (b) more ref training tracks (only 44 in current
      truth); (c) add a ref re-confirmation via temporal majority
      across the track.

  **What's next**: keep `hsv` as the orchestrator default for now (no
  regression for unbenched setups), but `--team-engine contrastive`
  for any new analytical run. Re-run the bench when run28/run29
  truth lands. Iterate the ref classifier threshold separately
  before flipping it on by default.
- **2026-04-26 restructure** — phase numbering rewritten so the
  pipeline reads in causal order: detection → rink → entity → events
  → stats. **Phase 2 (Virtual follow-cam) removed** (output wasn't
  usable, was wasting wall-clock and stalling the actual blocker).
  **Rink calibration promoted from parked Phase 3 to high-priority
  Phase 2** (`src/p2_a_rink.py`); rink-aware on-ice/off-ice filtering
  is the cleanest path to fix spectator pollution + ref leakage in
  entity recognition. **Old Stage 1.d (Entities)** moved to Stage
  3.a (`src/p3_a_entities.py`, output `p3_a_entities.json`). **Old
  Stage 1.e (Annotate)** moved to Stage 3.b (`src/p3_b_annotate.py`)
  since it depends on entity rollups. Phase 4 + 5 stubs rewired to
  read `p3_a_entities.json`. The orchestrator's `--run-p2` flag is
  gone; new `--skip-p2` (rink) is OFF by default like every other
  phase. Historical run01–run29 entries in this log keep their
  original phase numbers — they describe what shipped at that moment.
- **run24–run29** — full Phase 1 → Phase 5 validation of the freshly
  fine-tuned `parseq_hockey_rilh.pt` (the run23 model trained on 6 videos)
  across all six dataset videos. Total wall-clock 3h54m on MPS, sequential.
  Per-video summary (tracks → entities, numbered tracks):
    • run24 (clip60)   : 343 → 33,  37 numbered (10.8 %)
    • run25 (clip60-2) : 433 → 37,  29 numbered ( 6.7 %)
    • run26 (Video 04) : 167 → 24,  28 numbered (16.8 %)  [30fps, 30s]
    • run27 (Video 05) : 435 → 37,  41 numbered ( 9.4 %)
    • run28 (Video 06) : 475 → 40,  72 numbered (15.2 %)  [highest — FT'd on it]
    • run29 (Video 07) : 396 → 35,  31 numbered ( 7.8 %)
  All six runs produced the full artefact set (annotated.mp4 +
  p1_a..d.json + p4/p5 stubs + teams_preview.png). Phase 2 stayed off
  (parked), Phase 3 skipped (parked). No regressions on the historic
  4-video set. Confirms the parking + merge-mode + new model deploy.
- **run23** — extends the RILH jersey dataset with **Video 06 (Grenoble vs
  Varces)** and **Video 07 (Garges vs Rouen)**. New `--frame-stride 10`
  flag on Stage 1.c densifies sampling for dataset use (uniform temporal
  coverage instead of top-N highest-confidence). 1670 new crops →
  manually reviewed via `tools/annotate_crops.py` → 1669 annotations,
  fed into the dataset via `tools/build_jersey_dataset.py` (which now
  defaults to **merge** mode — `--replace` for full rebuild). Total
  dataset: **5197 crops, 48 unique numbers, 6 videos**. Re-fine-tuned
  PARSeq Hockey RILH (20 epochs, lr 5e-5, batch 16, best epoch 3). Test
  exact_match: 0.354 (Koshkina alone) → **0.966 (Koshkina + RILH)** on
  407 held-out crops (199 pos + 208 X). Annotation typos (XX/XXX → X)
  normalised pre-split. Phase 2 follow-cam parked in this session
  (orchestrator opt-in via `--run-p2`).
- **run22** — first end-to-end run via `src/run_project.py` orchestrator on
  Video 05. Phase 1 → Phase 5 sequenced cleanly (~84 min wall-clock).
  Surfaced + fixed the Stage 1.a/1.e `annotated.mp4` filename collision
  (Stage 1.a now writes `annotated_raw.mp4`).
- **run21** — full pipeline with PARSeq Hockey + RILH fine-tune on
  Video 04 + 05. Current OCR baseline: precision 95–96 %, recall 81–84 %
  on truth tracks (vs Koshkina-only 63–68 %/55–58 %). Stage 1.d entities:
  Video 04 → 24, Video 05 → 36. Player names not identified (PARSeq
  Hockey is digit-only). Loader bug fixed mid-run: auto-detect of the
  `model.` checkpoint prefix.
- **run19** — dorsal-crop dataset collection on 4 videos. Pivoted from a
  narrow band crop to Koshkina-style (full shoulder→hip + 5 px pad);
  median height doubled (~80 px), aspect 1.81 → 0.78. **3 528 crops**
  manually annotated, consolidated into `data/jersey_numbers/`
  (license-clean). 1 068 with a number, 38 unique numbers.
- **run17** — spectator filter (motion + position) attempted on Video 05.
  Dropped 60 % of tracks → Stage 1.d 37 → 20 entities (closer to the real
  count) but **~50 % of real player fragments were also dropped**, and the
  7 spectators HockeyAI tags as goalies still slipped through. Filter
  reverted; clean fix waits for Phase 2 rink calibration (geometric
  on/off-ice test). TrOCR receipt-vocab hallucinations
  (`CASHIER`/`AMOUNT`/…) confirmed at the entity level — one of the
  reasons the refactor dropped TrOCR entirely.
- **run16** — Video 05, 5 fixes consolidated: `--match-mode` (top-1
  puck/frame), goalie majority rule on Stage 1.b, entity-level goalie
  weighted by frame coverage on Stage 1.d, OCR ≥ 2 votes,
  `ocr_min_conf` 0.40 → 0.30, puck rendered in the final video. Goalie
  entities 17 → 5 (-71 %). 435 tracks → 37 entities.
- **run13** — full pipeline on Video 04 (France vs Monde, 60 s @ 30 fps).
  167 player tracks → Stage 1.d 24 entities. Team margin 1.46 (low —
  France dark-blue vs Monde light-green). Source video quality
  (motion blur, 30 fps) confirmed as the OCR ceiling.
- **run12** — full pipeline on Video 03 (Vierzon vs Pont de Metz, 30 s @
  60 fps). 250 tracks → 40 entities. Stage 1.b iterated v1 → v3 ;
  user-validated v2 as "practically perfect" except refs (HockeyAI
  doesn't tag them on roller) and the Pont de Metz goalie (white pads
  → classified pale).
- **run11** — Stage 1.a HockeyAI + BoT-SORT + ReID + GMC. Slightly longer
  tracks than ByteTrack tuned (run10), but ReID from the YOLO backbone
  doesn't discriminate within a team (5 identical jerseys), so the gain
  is marginal — see `Tracker backends`.
- **run05 / run06 / run07** — rink-stage HockeyRink transfer tests on
  three videos (clip60-2 with relaxed thresholds; Video 03 30 s; Video
  02 12 min). The model recognises that there *is* a rink but cannot
  localise its keypoints on roller markings — keypoints either cluster
  in a single area or are too few for a RANSAC homography. This is
  what motivated keeping the rink stage non-blocking, then promoting
  it to Phase 2 high-priority once entity quality plateaued.
- **run04** — **canonical Stage 1.a baseline**. clip60-2 + HockeyAI +
  ByteTrack default. 17 860 player detections, 433 tracks; puck in
  1 431 / 3 360 frames (**42.6 %**). Re-used as input for run08, 09,
  10, 11.
- **run02** — Stage 1.a COCO yolo11n baseline. 63 224 person dets, 1 804
  track IDs, only 2 puck frames out of 3 360 (**0.1 %**). Quantifies the
  baseline weakness that motivated HockeyAI adoption (note: ran on
  `yolo11n.pt` for speed, not `yolo11x.pt`).

The intermediate runs that informed these (run01, run03, run08, run09,
run10, run14, run15, run18, run20) are kept verbatim in
`docs/experiments.md`.


## Backlog

État courant des chantiers actifs, priorisé. Items concrets avec effort estimé
(XS <30min, S ~1h, M ~1j, L ~1sem, XL >1sem). Pour les chantiers structurels
multi-semaines (Phase 4+ et plateformes externes), voir la "Roadmap" en dessous.

### 🔥 Priorité haute — Phase 2 (Rink calibration)
- **Annoter 200–300 frames roller rink** (boards, cercles, buts, lignes —
  56 keypoints suivant le schéma HockeyRink) puis fine-tuner HockeyRink
  sur ce dataset. Bloque tout le reste de la chaîne aval : on-ice/off-ice
  geometric filter pour Stage 3.a → fix spectateurs + fix refs. — XL
- **Outil d'annotation dédié** (cousin de `tools/annotate_crops.py`) pour
  picker les keypoints sur des frames samplées. Sans outil ergonomique
  l'annotation va ramer. — M
- **Pipeline data → fine-tune HockeyRink** (transfer learning depuis le
  HockeyRink ice puisque la topologie de keypoints est identique). — M

### Phase 1 — Court terme
- **Cache pose entre Stage 1.b et Stage 1.c** : aujourd'hui yolo26l-pose
  tourne 2× sur les mêmes frames. Écrire `pose_cache.json` en Stage 1.b
  et le lire en Stage 1.c sauve ~20-30 % wall-clock pipeline. — S
- **Stage 1.b + Stage 1.c en parallèle** (processus séparés). Gain
  ~10-20 % supplémentaire (overlap GPU/CPU). — S

### Phase 1 / Phase 3 — Court-moyen terme
- **Annoter `track_truth.json` sur run24-run29** via `tools/annotate_tracks.py`
  (UI localhost, ~250 tracks à passer en A/B/Ref/X) puis lancer
  `tools/bench_team_engines.py` pour comparer hsv / osnet / siglip / contrastive
  + ref classifier. Donne la première mesure factuelle sur **nos** vidéos
  pour décider quel engine devient le défaut Stage 1.b. — S
- **Train `models/ref_classifier_rilh.pt`** une fois track_truth.json existe
  (`tools/finetune_ref_classifier.py`). Solve le ref-leak indépendamment
  de Phase 2. — XS (entraînement) après l'annotation
- **Train `models/contrastive_team_rilh.pt`** sur le même truth set
  (`tools/finetune_contrastive_team.py`, ~30 epochs CPU). — S
- **Vote temporel team entité-level seul** comme quick-win complémentaire :
  mode des `team_id` des membres d'une entité au lieu de
  `next(iter(team_ids))` non-déterministe. — S
- **WBF (Weighted Boxes Fusion)** ensemble palet HockeyAI + détecteur
  dédié (style sieve-data). Gain attendu couverture palet 42% → 55–65%.
  Plus utile *après* fine-tune Phase 4. — M

### Long terme (cf. Roadmap)
- **Phase 4 fine-tune HockeyAI** sur 500–1000 frames RILH annotées.
  Bloqueur = annotation. Débloque puck + goalie + classes refs. — XL
- **Fine-tune nom de joueur** (séparé du PARSeq Hockey digit-only). — L
- **Phase 4 events** : détection événements (buts, tirs, fautes). — XL
- **Phase 6 web platform** : FastAPI + Next.js. — L
- **Phase 7+** : multi-cam stitching, live RTMP/HLS. — XL

## Roadmap (multi-week horizons)

Six headlines; per-chantier scope, blockers, target metrics, and the surrounding technical surveys (public hockey models, WBF ensembling, cost/benefit ranking, datasets-to-build) live in `docs/experiments.md`.

- **Phase 2 — Rink calibration & 2D map** (HIGH PRIORITY) — blocked on 200–300 roller rink keypoint annotations + fine-tune HockeyRink. Unlocks the on-ice / off-ice filter that downstream entity quality depends on.
- **Phase 4 — Roller-specific YOLO fine-tune** (2–4 wk) — needs a 500–1000 frame RILH dataset; bootstrap with HockeyAI pre-labels. Target > 0.7 mAP on puck.
- **Phase 4 — Event detection** (3–6 wk) — temporal action models (TSN / MoViNet / SlowFast); custom labelled set.
- **Stage 1.c — Names + entity-level multi-frame OCR consensus** — recall ceiling currently set by source video quality + per-track voting; entity-level vote could push it further.
- **Phase 6 — Web platform** (2–3 wk) — FastAPI + Next.js consuming `runs/runNN/`.
- **Phase 7+** — Multi-cam stitching, live RTMP/HLS via MediaMTX, mobile control app.

## Things to know when iterating
- Test on a 60s clip first: `ffmpeg -i input.mp4 -ss 0 -t 60 -c copy clip.mp4`
- **Always use `python -u`** when launching long runs to a file — stdout is
  block-buffered otherwise, which makes progress invisible and triggered a
  false "stuck" diagnosis in run03.
- For any serious puck work, pass `--hockey-model` — the COCO default is
  only useful when iterating fast and puck quality doesn't matter.
- Camera too slow in Stage 2.a → raise `--alpha`. Camera jittery → lower it,
  or raise `--polish-window`.
- Still missing puck detections → `--imgsz 1280` or 1536 (slower but much
  better on small objects).
- Use `--debug-overlay` (Stage 2.a) to understand the focus trajectory.
- Tracker comparisons: pass `--tracker configs/bytetrack_tuned.yaml` or
  `configs/botsort_reid.yaml`. BoT-SORT+ReID is ~same speed as ByteTrack
  in practice (ReID features come from the YOLO backbone, no extra model
  download).
- Run outputs are incremental: `runs/run01/`, `runs/run02/`, … Don't
  overwrite a previous run even if it failed — past tracks + videos are
  useful for diffing the effect of a parameter change.
