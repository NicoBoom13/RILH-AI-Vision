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
  (YOLO pose shoulders→hips, bbox-fallback for dark jerseys) +
  multi-point dominant color averaging (3×2 grid) + per-crop k=2
  k-means (fit on skaters only, goalies classified post-hoc). HSV by
  default. `is_goaltender` per track via majority threshold (>50 % of
  detections tagged `goaltender`). User-validated on run12 as
  "practically perfect" except referees (HockeyAI mislabels them
  'player' on roller — can't filter by class_name) and the Pont de Metz
  goalie (white pads → classified pale). Improvement needs position-
  based assignment (waits for Stage 3.a) or custom goalie sampling —
  deferred.
- **Stage 1.c (Numbers)** ✅ : per-track jersey-number OCR via **PARSeq
  Hockey + RILH fine-tune**. Single engine (PARSeq); the previous
  TrOCR / together engines were removed in the refactor — the digit
  number is now the sole player-clustering key, and TrOCR brought
  receipt-vocabulary hallucinations + 340 MB of weights for limited
  benefit. PARSeq accepts a custom `--parseq-checkpoint` to load
  Maria Koshkina's hockey baseline
  ([github.com/mkoshkina/jersey-number-pipeline](https://github.com/mkoshkina/jersey-number-pipeline),
  CC-BY-NC license) or our `parseq_hockey_rilh.pt` (trained on 1063
  RILH crops via `tools/finetune_parseq_hockey.py`). Crop strategy is
  Koshkina-style (bbox of 4 torso keypoints + 5 px padding). When the
  checkpoint is loaded, letterbox is skipped (Koshkina trained on
  direct-resize). **Quality on held-out test set (208 crops): exact
  match 43.3 % (Koshkina alone) → 97.1 % (Koshkina + RILH).** On
  Video 04 + 05 truth tracks: recall 55–58 % → **81–84 %** ;
  precision 63–68 % → **95–96 %**. Names are not identified — PARSeq
  Hockey is digit-only; a name fine-tune is a separate later step.
- **Stage 1.d (Entities)** ✅ : post-hoc **Re-ID clustering** that
  collapses fragmented tracks into stable entities (one entity = one
  real player / goalie). Uses OSNet x0_25 (via `torchreid`) medoid
  embedding per track + greedy merge under **same-team constraint**
  (from Stage 1.b), **strict temporal non-overlap**, and **OCR bonus**
  (from Stage 1.c). OCR conflicts are a hard block. Output:
  `p1_d_entities.json` consumed downstream by `p1_e_annotate.py`. On
  run12 (250 tracks → 40 entities), on run13 (167 tracks → 24
  entities). Doesn't replace the tracker — it post-processes its
  output. Entity-level `is_goaltender` is weighted by frame coverage
  to absorb HockeyAI class flips. Design doc:
  `docs/p1_d_entities_design.md`.
- **Stage 1.e (Annotate)** ✅ : final MP4 with team-coloured boxes,
  per-track label `t{id} {G|S} #NN` (track id always shown),
  dark-gray puck box, short traces. Auto-discovers `p1_b_teams.json` and
  `p1_d_entities.json` next to `p1_a_detections.json` if present. Optional
  `--debug-frames-dir` writes 1 PNG every N frames.
- **Stage 2.a (Follow-cam)** ✅ : virtual broadcast cam. Built but not
  heavily iterated on; runs in parallel to stages b–e (only depends
  on `p1_a_detections.json`).
- **Stage 3.a (Rink calibration)** ❌ PARKED. The obvious shortcut
  (HockeyRink keypoints) **does not transfer to roller rinks** — see
  run05–run07 in the test log. Needs a roller-specific annotated
  dataset (200–300 frames) + fine-tune to unblock.

## Architecture

The project has **two levels** of organisation:

### Level 1 — project phases (axis of `graph3D` / `run_project.py`)

Big chunks of work, each with its own concern:

| Phase | Name | In pipeline? |
|---|---|---|
| **Phase 1** | Detect & track (full identification) | ✅ orchestrated |
| **Phase 2** | Virtual follow-cam | ✅ orchestrated |
| **Phase 3** | Rink calibration | ✅ orchestrated (parked, tolerant of failure) |
| **Phase 4** | Event detection | ✅ orchestrated (stub) |
| **Phase 5** | Statistics creation | ✅ orchestrated (stub) |
| **Phase 6** | Web platform (FastAPI + Next.js) | ❌ external — consumes `runs/runNN/` folders |
| **Phase 7** | Multi-cam stitching, live RTMP/HLS, app mobile | ❌ external — infra + mobile |

The orchestrator `src/run_project.py` runs Phase 1 → Phase 5 in sequence with
per-phase gates (`--skip-pN`, `--force`). Phase 6 and Phase 7 are services /
infra that live outside the per-run pipeline.

### Level 2 — internal stages of each phase

Each phase decomposes into one or more stages, named `pN_x_*.py`:

**Phase 1 — Detect & track** (5 stages, sequential):

1. `src/p1_a_detect.py` → `p1_a_detections.json` (per-frame bboxes, class IDs,
   persistent track IDs). HockeyAI YOLO + ByteTrack. Match-mode default
   keeps top-1 puck per frame; pass `--training-mode` for drills.
2. `src/p1_b_teams.py` → `p1_b_teams.json` + `teams_preview.png`: team_id
   (0/1) per player track via k=2 on pose-based torso color.
3. `src/p1_c_numbers.py` → `p1_c_numbers.json`: per-track jersey number via
   YOLO pose + PARSeq (`--parseq-checkpoint models/parseq_hockey_rilh.pt`
   for our RILH-fine-tuned model). Crop is Koshkina-style.
4. `src/p1_d_entities.py` → `p1_d_entities.json`: fragmented tracks collapsed
   into stable entities via OSNet embeddings + team/overlap/OCR constraints.
5. `src/p1_e_annotate.py` → annotated MP4 with team-coloured boxes +
   `t{id} {G|S} #NN` labels.

**Phase 2 — Virtual follow-cam** (1 stage):

- `src/p2_a_followcam.py` → `followcam.mp4`. Reads only `p1_a_detections.json`,
  runs in parallel to Phase 1 stages b-e.

**Phase 3 — Rink calibration** (1 stage, parked):

- `src/p3_a_rink.py` → `p3_a_rink_keypoints.json` (would be), but HockeyRink
  doesn't transfer to roller. Orchestrator tolerates the failure.

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

### Follow-cam (Stage 2.a)
- **Focus point = weighted blend of puck position and players centroid**.
  Puck gets high weight when detected; players-centroid fallback otherwise.
  Recently-seen puck positions are extrapolated for ~15 frames to bridge
  missed detections.
- **Smoothing = exponential moving average** on the focus trajectory,
  optionally followed by a centered boxcar pass for extra polish.
- **Crop window clamped to frame bounds** so we never show black bars.

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

### Entity clustering (Stage 1.d)
`p1_d_entities.py` post-processes fragmented tracks into stable entities:
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
5. Output: `p1_d_entities.json` with `track_ids` lists, derived
   `team_id` / `is_goaltender` / `jersey_number` / frame ranges, plus a
   list of unmatched singleton tracks.

**Goalie weighting** (added run16, kept after the refactor):
- Stage 1.b `is_goaltender` per track uses a **majority threshold**
  (>50 % of detections tagged `goaltender`), replacing the previous
  `any` rule that flipped a track on a single noisy frame.
- Stage 1.d entity-level `is_goaltender` is **weighted by frame
  coverage**: an entity is goalie only if more than half of its merged
  tracks' total frame count comes from goalie-tagged tracks.
- Stage 1.c aggregation requires **≥2 agreeing votes** for the winning
  number — singletons are usually OCR noise from a single bad crop.

**Spectator handling** — see run17 for why a motion+position filter was
attempted then **removed**: it dropped ~50 % of real player fragments to
catch ~30 spectators, while the 7 spectators HockeyAI tags as goalies
slipped through anyway. The plan is to wait for **Stage 3.a rink
calibration** (geometric "is this bbox on the ice?") to do this cleanly.

Annotation (`p1_e_annotate.py`):
- Supervision-style boxes/labels/traces. Box color is **forced green or
  blue** per team.
- **Entity-aware**: if `p1_d_entities.json` exists, the label + team come
  from the entity (so every merged fragment shares the same `#NN`, name
  and colour across the video). Otherwise, per-track values from
  `p1_b_teams.json` + `p1_c_numbers.json`.
- Labels read `t{id} {G|S} #NN NAME` — track_id always shown (so user
  can give frame-level feedback), `G`/`S` from `is_goaltender`, `#??`
  if no number, name omitted if not identified.
- **Puck**: rendered as a dark gray bbox (60,60,60) + short trace.
  No label.
- **Spectators**: not filtered (will be addressed by Stage 3.a rink
  calibration). All tracked detections render — including stationary
  bystanders that HockeyAI tags as players.

## Known limitations (in priority order)
1. **Video source quality** still affects team-colour clustering and
   any per-frame analysis on small / blurry / oblique players. Number
   OCR is no longer the bottleneck (post run21: 95-96 % precision,
   81-84 % recall on truth tracks via PARSeq Hockey + RILH fine-tune).
   The remaining ~15-20 % missed numbers are mostly tracks where the
   back is never cleanly visible.
2. **Track fragmentation** is real but **partly absorbed** by Stage 1.d
   (run16: 435 tracks → 37 entities, of which only 5 G after the
   goalie majority + frame-coverage rules). HockeyAI still
   intermittently flips goaltender/player class on individual frames,
   but the majority threshold + frame-coverage weighting mostly
   neutralises that. Refs still merge into the two teams (no third
   cluster) — see #5. **Stationary spectators that HockeyAI mistags as
   `player` or `goaltender` still pollute entities** — the Stage 1.b
   spectator filter attempted in run17 over-filtered real players and
   was reverted; clean fix waits for Stage 3.a rink calibration.
3. **Stage 3.a calibration** is blocked on annotation. HockeyRink (ice) does
   not transfer to roller rinks — the model "recognises" a rink but collapses
   all 56 keypoints to a cluster instead of localising them individually.
   Unblocking requires 200–300 annotated frames of roller rinks.
4. **Puck detection quality** is workable (~43% of frames with HockeyAI vs
   <1% with COCO) but drops on motion blur, small pucks, and uneven lighting.
   Roller-specific fine-tune (Phase 4) would narrow the gap.
5. **Refs leak into team clusters** because HockeyAI doesn't recognise
   roller-hockey referee uniforms — they're tagged `class_name='player'`.
   Fixable by k=3 clustering in Stage 1.b or a dedicated ref detector.
6. No event detection yet (Phase 5).
7. Single-camera assumption. Multi-camera stitching = Phase 7.

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
├── p1_d_entities.py       — Phase 1 stage d — entities (Re-ID merge)
├── p1_e_annotate.py       — Phase 1 stage e — annotate (final MP4)
├── p2_a_followcam.py      — Phase 2 stage a — virtual follow-cam
├── p3_a_rink.py           — Phase 3 stage a — rink calibration (parked)
├── p4_a_events.py         — Phase 4 stage a — event detection (STUB)
└── p5_a_stats.py          — Phase 5 stage a — statistics (STUB)

configs/                   — tracker YAMLs (bytetrack_tuned, botsort_reid)
docs/                      — design notes (p1_d_entities_design.md)
models/                    — model weights (gitignored). Ultralytics YOLOs,
                              HockeyAI, parseq_hockey.pt (Koshkina),
                              parseq_hockey_rilh.pt (our fine-tune).
videos/                    — source clips (gitignored)
runs/                      — pipeline outputs per run (gitignored)
data/                      — license-clean datasets (committed). Today:
                              data/jersey_numbers/ (3528 crops + annotations
                              + train/val/test splits + LICENSE/README).
tools/                     — utilities (annotation web UI, dataset/splits
                              builders, fine-tune script, smoke tests).
                              Not pipeline stages.
graphify-out/              — local 3D visualization (gitignored).
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
  reverted; clean fix waits for Stage 3.a rink calibration (geometric
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
- **run05 / run06 / run07** — Stage 3.a HockeyRink transfer tests on three
  videos (clip60-2 with relaxed thresholds; Video 03 30 s; Video 02
  12 min). The model recognises that there *is* a rink but cannot
  localise its keypoints on roller markings — keypoints either cluster
  in a single area or are too few for a RANSAC homography. Confirms the
  parked status of Stage 3.a.
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
multi-semaines (Phases 3–8+), voir la "Roadmap" en dessous.

### Phase 1 — Court terme
- **Cache pose entre Stage 1.b et Stage 1.c** : aujourd'hui yolo26l-pose
  tourne 2× sur les mêmes frames. Écrire `pose_cache.json` en Stage 1.b
  et le lire en Stage 1.c sauve ~20-30 % wall-clock pipeline. — S
- **Stage 1.b + Stage 1.c en parallèle** (processus séparés). Gain
  ~10-20 % supplémentaire (overlap GPU/CPU). — S

### Phase 2 — Court-moyen terme
- **Fix team classification Stage 1.b** : sur run15, 6 tracks/435 mal
  classés avec `vote_confidence` 0.50–1.00 (donc le k-means capture la
  mauvaise couleur, pas un floor issue). Pistes par effort croissant :
  (a) augmenter `--samples-per-track` 8 → 20 ;
  (b) vote temporel entité-level (mode des `team_id` des membres au lieu
  de `next(iter(team_ids))` non-déterministe) ;
  (c) seeder centroïdes sur joueurs OCR-confiants ;
  (d) fine-tune classifieur team. — M–L
- **Vote temporel team entité-level seul** (option b ci-dessus) comme
  quick-win avant le fix complet. — S
- **WBF (Weighted Boxes Fusion)** ensemble palet HockeyAI + détecteur
  dédié (style sieve-data). Gain attendu couverture palet 42% → 55–65%.
  Plus utile *après* fine-tune Phase 4. — M

### Phase 3 — Long terme (cf. Roadmap)
- **Phase 4** : fine-tune HockeyAI sur 500–1000 frames RILH annotées.
  Bloqueur = annotation. Débloque puck + goalie + classes refs. — XL
- **Fine-tune TrOCR** sur crops jersey RILH (numéros + noms). Élimine les
  hallucinations receipt-style. — L
- **Stage 3.a** : calibration rink + map 2D. Bloqué sur 200–300 annotations
  keypoints rink roller. — XL
- **Phase 5** : détection événements (buts, tirs, fautes). — XL
- **Phase 7** : plateforme web FastAPI + Next.js. — L
- **Phase 8+** : multi-cam stitching, live RTMP/HLS. — XL

## Roadmap (multi-week horizons)

Six headlines; per-chantier scope, blockers, target metrics, and the surrounding technical surveys (public hockey models, WBF ensembling, cost/benefit ranking, datasets-to-build) live in `docs/experiments.md`.

- **Stage 3.a — Rink calibration & 2D map** — blocked on 200–300 roller rink keypoint annotations.
- **Phase 4 — Roller-specific YOLO fine-tune** (2–4 wk) — needs a 500–1000 frame RILH dataset; bootstrap with HockeyAI pre-labels. Target > 0.7 mAP on puck.
- **Phase 5 — Event detection** (3–6 wk) — temporal action models (TSN / MoViNet / SlowFast); custom labelled set.
- **Stage 1.c — Names + entity-level multi-frame OCR consensus** — recall ceiling currently set by source video quality + per-track voting; entity-level vote could push it further.
- **Phase 7 — Web platform** (2–3 wk) — FastAPI + Next.js consuming `runs/runNN/`.
- **Phase 8+** — Multi-cam stitching, live RTMP/HLS via MediaMTX, mobile control app.

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
