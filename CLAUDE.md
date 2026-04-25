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
  `entities.json` consumed downstream by `p1_e_annotate.py`. On
  run12 (250 tracks → 40 entities), on run13 (167 tracks → 24
  entities). Doesn't replace the tracker — it post-processes its
  output. Entity-level `is_goaltender` is weighted by frame coverage
  to absorb HockeyAI class flips. Design doc:
  `docs/p1_d_entities_design.md`.
- **Stage 1.e (Annotate)** ✅ : final MP4 with team-coloured boxes,
  per-track label `t{id} {G|S} #NN` (track id always shown),
  dark-gray puck box, short traces. Auto-discovers `teams.json` and
  `entities.json` next to `detections.json` if present. Optional
  `--debug-frames-dir` writes 1 PNG every N frames.
- **Stage 2.a (Follow-cam)** ✅ : virtual broadcast cam. Built but not
  heavily iterated on; runs in parallel to stages b–e (only depends
  on `detections.json`).
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

1. `src/p1_a_detect.py` → `detections.json` (per-frame bboxes, class IDs,
   persistent track IDs). HockeyAI YOLO + ByteTrack. Match-mode default
   keeps top-1 puck per frame; pass `--training-mode` for drills.
2. `src/p1_b_teams.py` → `teams.json` + `teams_preview.png`: team_id
   (0/1) per player track via k=2 on pose-based torso color.
3. `src/p1_c_numbers.py` → `numbers.json`: per-track jersey number via
   YOLO pose + PARSeq (`--parseq-checkpoint models/parseq_hockey_rilh.pt`
   for our RILH-fine-tuned model). Crop is Koshkina-style.
4. `src/p1_d_entities.py` → `entities.json`: fragmented tracks collapsed
   into stable entities via OSNet embeddings + team/overlap/OCR constraints.
5. `src/p1_e_annotate.py` → annotated MP4 with team-coloured boxes +
   `t{id} {G|S} #NN` labels.

**Phase 2 — Virtual follow-cam** (1 stage):

- `src/p2_a_followcam.py` → `followcam.mp4`. Reads only `detections.json`,
  runs in parallel to Phase 1 stages b-e.

**Phase 3 — Rink calibration** (1 stage, parked):

- `src/p3_a_rink.py` → `rink_keypoints.json` (would be), but HockeyRink
  doesn't transfer to roller. Orchestrator tolerates the failure.

**Phase 4 — Event detection** (1 stage, stub):

- `src/p4_a_events.py` → `p4_events.json` marker. Real impl pending.

**Phase 5 — Statistics creation** (1 stage, stub):

- `src/p5_a_stats.py` → `p5_stats.json` marker. Real impl pending.

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

Both backends write the same `detections.json` schema (plus a `class_name`
string per detection), so every downstream stage is backend-agnostic.

**Match vs training mode.** Stage 1.a enforces 1-puck-per-frame by default
(real match conditions): when multiple puck detections appear in the
same frame, only the highest-confidence one is kept (after the tracker
has assigned IDs, so the dropped duplicates simply never reach
`detections.json`). Pass `--training-mode` to disable the filter —
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
5. Output: `entities.json` with `track_ids` lists, derived
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
- **Entity-aware**: if `entities.json` exists, the label + team come
  from the entity (so every merged fragment shares the same `#NN`, name
  and colour across the video). Otherwise, per-track values from
  `teams.json` + `numbers.json`.
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
3D interactive knowledge graph of the pipeline (174 nodes, 250 edges,
20 communities, 10 phases pinned along an X-axis rail), generated by
`graphify` (PyPI `graphifyy`) + a custom 3D renderer using
`3d-force-graph` (ThreeJS). Open `graphify-out/graph3d.html` in any
browser — no server needed. Combines AST extraction from `src/*.py`
+ hand-curated pipeline orchestration (phase order, data flow,
rationales) from `orchestration.json`.

Regenerate after code or doc changes:
```bash
/Users/nico/.local/share/uv/tools/graphifyy/bin/python graphify-out/regen.py
```
~2 seconds, zero LLM tokens, no network access. The whole `graphify-out/`
folder is gitignored — it's an aid to local navigation, not project
deliverable. Full doc + edge legend in `graphify-out/graphify.md`.

## Test run log

Reverse-chronological. Each entry = one `runs/testNN/`. Params only list
what differs from the immediate predecessor.

- **run21 — Full pipeline avec PARSeq Hockey + RILH fine-tune sur
  Video 04 + 05.** Première run de l'OCR fine-tuné en pipeline complet.
  Réutilise `detections.json` existants (run13 pour Video 04, run18 pour
  Video 05) — Phases 1.5, 1.6, 6 identify, 6 annotate refaites.
  * **Stage 1.c** : `--ocr-engine parseq --parseq-checkpoint
    models/parseq_hockey_rilh.pt`. Crop Koshkina-style (full torso).
  * **Validation contre annotations utilisateur** :
    - Video 04 (31 truth tracks) : 27 prédits, **26 corrects → précision
      96 %, recall 84 %** (vs Koshkina seul 63 %/55 %, vs TrOCR 53 %/29 %).
    - Video 05 (48 truth tracks) : 41 prédits, **39 corrects → précision
      95 %, recall 81 %** (vs Koshkina seul 68 %/58 %, vs TrOCR 40 %/17 %).
  * **Stage 1.d entités** : Video 04 → 24 entités (1 G, top-10 = 9 numéros
    identifiés). Video 05 → 36 entités (5 G, top-10 = 9 numéros).
    Beaucoup plus propre qu'avant (run16 avait 17 G erronés, run18
    avait des hallucinations TrOCR `CASHIER`/`AMOUNT` sur les noms).
  * **Vidéos finales** : `runs/run21/video04/annotated_numbered.mp4`
    (120 MB), `runs/run21/video05/annotated_numbered.mp4` (206 MB).
    Debug frames (final + phase1) au 1/10.
  * **Limitation restante** : noms de joueurs à 0 (PARSeq Hockey est
    digit-only). À traiter par soit garder TrOCR pour les noms, soit
    annoter + fine-tuner PARSeq sur des crops noms.
  Bug intermédiaire : premier essai de run21 → 0 prédiction parce que
  le loader doublait le préfixe `model.` (Koshkina sans préfixe vs notre
  checkpoint fine-tuné qui en a un). Détection auto du préfixe ajoutée
  → re-run OK.
- **run20 — Stage 1.c avec PARSeq Hockey baseline (sans fine-tune)
  sur Video 04 + 05.** Première intégration du loader Koshkina dans
  `p1_c_numbers.py` (option `--parseq-checkpoint`). Crop Koshkina-
  style (full torso). Direct-resize sans letterbox.
  * Video 04 : 59/167 tracks identifiés (35.3 %), recall 55 %.
  * Video 05 : 156/435 tracks identifiés (35.9 %), recall 58 %.
  * Précision 63 % / 68 % — ×2-3 le recall vs TrOCR baseline run19.
  * Names : 0 (PARSeq Hockey digit-only).
  Smoke test loader sur 200 crops annotés : 74 % exact match.
- **run19 — Collecte crops dorsal sur 4 vidéos pour annotation +
  fine-tune.** Génération de matériel d'entraînement, pas un test
  pipeline. Réutilise `detections.json` de run04 (clip60-2), run13
  (Video 04), run18 (Video 05). Stage 1.a nouveau sur clip60.mp4.
  * Première passe : crop bande étroite 15-65 % torse + 30 % pad.
    3 233 crops produits, hauteur médiane ~35 px → utilisateur trouve
    ça difficile à annoter à l'œil.
  * **Pivot** : crop refait à la Koshkina (full shoulder→hip + 5px pad),
    régénéré sur les 4 vidéos. **Hauteur médiane doublée** (~80 px),
    aspect ratio passé de 1.81 (large) à 0.78 (carré-ish).
  * **3 528 crops** dans `runs/run19/{clip60,clip60-2,video04,video05}/
    debug_crops/numbers/`. Annotation manuelle via
    `tools/annotate_crops.py` (web UI localhost). Annotations sauvées
    dans `runs/run19/annotations.json` puis consolidées dans
    `data/jersey_numbers/` (license-clean, indépendant Koshkina).
  * **1 068 crops avec numéro** (38 numéros uniques) + **2 460 X**
    (no number visible). Top-10 numéros : #9 (112), #20 (90), #5 (77),
    #11 (73), #77 (71), #6 (69), #13 (62), #14 (60), #87 (49), #92 (43).
- **run17 — Villeneuve vs Vierzon (Video 05) — filtre spectator multi-signal.**
  Pipeline complet relancé. Stage 1.a inchangée vs run16 (HockeyAI +
  match-mode default, 1 puck/frame). Nouveauté Stage 1.b : `is_static` =
  `NOT goaltender AND (median_disp < 5px OR median_y > 0.74 ×
  frame_height)`. Goalies exemptés du filtre (préservation des vrais
  goalies stationnaires). Résultats :
  * **Static tracks** : **261 / 435 (60 %)** flaggés spectator,
    excluded de Stage 1.c, 1.6, et annotate.
  * **Stage 1.c** : 1605 samples (vs 3696 run16, **-57 %
    compute**). 16 / 174 tracks identifiés (9.2 %), 19 noms (10.9 %).
    Numéros : #2, #77, #6, #4 — moins variés qu'avant car beaucoup de
    tracks éliminées avant OCR.
  * **Stage 1.d** : **20 entités** (vs 37 run16, **-46 %**).
    Approche du compte réel attendu (2 équipes × 4-5 ≈ 10 entities/team).
    Goalie entities : 6 (vs 5 run16, +1 — les 7 spectators-tagged-
    goalie passent toujours le filtre, contribuent à des entités G).
    Top-10 : 3G/7S (vs 2G/8S run16).
  * **Vidéo finale** : 176 MB (vs 209 run16). Annotated propre.
  * **Hallucinations TrOCR persistent au niveau entity** : entity [0]
    `S #49 CASH`, [2] `G #2 AMOUNT`, [4] `S #79 TAX`, [6] `S #79 QTY`,
    [7] `G #4 CASHIER`, [8] `S #69 MCAUD`. Le filtre spectator a viré
    les tracks isolées contenant ces hallucinations, mais TrOCR continue
    à les produire sur certains crops illisibles de vrais joueurs.
    Stop-list = next-step Phase 1.
  Verdict utilisateur après revue visuelle : **filtre spectator
  RETIRÉ** — trop de vrais joueurs faux-droppés (~50 % des fragments)
  pour ne catcher que ~30 spectators sur 37 listés, et les 7
  spectators-tagged-goalie passaient quand même. Le bon outil sera la
  calibration rink (Stage 3.a) qui répondra géométriquement « ce bbox
  est-il sur la glace ou en tribune ? ». Tout le reste de la v2 est
  conservé (name OCR, debug, palet gris, goalie majority + frame-
  weighting, OCR ≥2 votes, ocr_min_conf 0.30, ocr_conflict 0.40,
  training-mode default, label `t{id} G/S #num NAME`).
- **run16 — Villeneuve vs Vierzon (Video 05) — 5 fix consolidés.** Première
  run avec `--match-mode` (top-1 puck/frame), goalie majority rule,
  entity goalie weighted by frame coverage, OCR ≥2 votes, `ocr_min_conf`
  0.40→0.30, `ocr_conflict_conf_floor` 0.55→0.40, palet rendu dans la
  vidéo finale. Stage 1.a rejouée (1h wall-clock), Phases 1.5/6/1.6/annotate
  neuves. Résultats :
  * **Palet** : 0 frame avec 2+ pucks ✓ (run15: 12.9% en avaient, max 5).
    260 → 185 puck tracks (filtrage des faux positifs).
  * **Goalies flag Stage 1.b** : 122 → **73 tracks** (-40%) via majority
    rule (>50% des détections tagées goaltender).
  * **Goalie entities Stage 1.d** : 17 → **5** (-71%) via frame-coverage
    weighting. Top-10 passé de 9G/1S à 2G/8S.
  * **Numéros** : 66 (run15) → 49 tracks identifiés, **noms** 96 → 66.
    Perte de recall attendue (≥2 votes filtre noise ET signal) —
    compensée par qualité supérieure (fin des `#3` 1-vote hallucinés).
  * **Entités** : 36 → 37. Player groups (merge par #) : 16 → 11.
  Verdict utilisateur sur review debug_frames : (a) track 2 encore
  classé G sur frames 10-310 (cause: entité mergée avec 4 spectator
  tracks taguées goalie par HockeyAI, dont 739/1348 = 54.8% des frames
  → passait le seuil 50%) ; (b) **37 track IDs listés comme spectators**
  (4, 6, 11, 12, 15, 30, 157, 159, 162, 186, 193, 201, 224, 246, 293,
  319, 881, 883, 884, 992, 999, 1035, 1054, 1154, 1415, 1483, 1490,
  1555, 1572, 1574, 1592, 1644, 1649, 1664, 1733, 1735, 1827) —
  presque toujours taggués avec les hallucinations TrOCR `CASHIER` /
  `CASH` / `AMOUNT` ; (c) palet gris clair (180,180,180) insuffisamment
  visible. Fix suivants codés (non re-testé) : filter spectator par
  median per-frame displacement < 5px (propagé Stage 1.b→6_identify→
  1.6→annotate), palet BGR(60,60,60), inversion `--match-mode` →
  `--training-mode` (1-puck désormais par défaut).
- **run15 — Villeneuve vs Vierzon (Video 05) — debug mode + name OCR.**
  Réutilise `detections.json` + `teams.json` de run14. Ajouts dans le
  code : name OCR (crop au-dessus du numéro, TrOCR max_new_tokens=16
  pour noms / 6 pour numéros), `--debug-crops-dir` (2606 crops number+
  name sauvés), `--debug-frames-dir --debug-frames-step 10` dans
  phase6_annotate (360 PNG). Label final enrichi `t{id} {G/S} #num NAME`.
  Résultats :
  * Numéros : **66/435 (15.2%)** vs run14 59/435 (+12%). Noms :
    **96/435 (21.6%)** (nouveau). 16 player groups (== run14).
  * Entités Stage 1.d : 36 (vs 37 run14). **17 taguées goalie** (9 du
    top-10 en G). Anomalie confirmée.
  * Hallucinations TrOCR identifiées : `CASHIER`, `CASH`, `AMOUNT`,
    `TAX`, `QTY`, `ITEM`, `MAY` — vocabulaire "reçus de caisse" du
    training set microsoft/trocr-base-printed.
  * Debug crops + frames ont permis à l'utilisateur de catégoriser les
    bugs : 11 frames pour goalie over-tag (27 tracks culprit, 21 Type-B
    propagation + 6 Type-A seuil any→majority), 18 frames pour OCR
    (6 misread + 5 missed malgré back samples + 4 entity misassignment
    + 2 no back sample), 6 frames pour team (vote conf 0.50-1.00
    → k-means capture mauvaise couleur, pas un floor issue), 16 pucks
    fake (12.9% frames avec 2+ pucks, max 5 simultanés). Diagnostic
    complet qui a mené aux 5 fix de run16.
- **run14 — Villeneuve vs Vierzon (Video 05, 60s, 1920×1080 @ 60fps).**
  Pipeline complète, première run production de YOLO26L-pose.
  * **Stage 1.a** ✅ HockeyAI + ByteTrack default. 3600 frames. 435 player
    tracks (313 skaters + 122 goaltenders), 260 puck tracks, puck dans
    **1870/3600 frames (51,9 %)** — meilleur qu'run04 (42,6 %) et
    run12 (28,6 %). Runtime ~1h wall-clock. (Note : un premier essai
    accidentel avec `--model yolo26l.pt` (COCO) a été stoppé pour
    repartir sur HockeyAI.)
  * **Stage 1.b** ✅ HSV k=2 avec `--pose-model yolo26l-pose.pt` (premier
    run production de YOLO26L-pose). 282/152 split, marge 1,95,
    27 mixed-vote tracks, 1 seul track sans sample. Pose-based : 1712
    crops (vs 656 bbox-fallback) = **72 % de crops pose** contre ~57 %
    sur run13 avec yolo11n-pose → YOLO26L-pose améliore bien le taux
    de succès des keypoints sur les torses dark/blurry.
  * **Stage 1.c** ✅ TrOCR + yolo26l-pose. **59/435 (13,6 %)**,
    16 groupes joueurs. Numbers trouvés (top par fragments) : #1 (10),
    #4 (4), #2/#9/#6 (3), #77/#5/#09/#19 (2). En absolu, 59 tracks
    identifiés > run13 (39) — vidéo plus longue + plus de tracks aide.
  * **Stage 1.d** ✅ OSNet x0_25 + greedy merge. **435 tracks → 37
    entités** (28 unmatched), 370 merges, 6061 paires skippées (overlap).
    Split 25 team 0 / 12 team 1. **Anomalie** : 19/37 entités taguées
    goaltender, dont 9 du top 10 — HockeyAI flippe massivement la classe
    `goaltender`/`player` sur cette vidéo (limitation #2 amplifiée ici).
  * **Stage 1.e** ✅ entity-aware (37 entités couvrent 407 tracks).
    Sortie : `runs/run14/annotated_numbered.mp4`.
  Verdict utilisateur : détections équipe + joueurs « pas mal du tout
  mais pas parfaites ». Points de bug précis à documenter au prochain pass.
- **run13 — Full pipeline + Stage 1.d + TrOCR on Video 04 (France vs
  Monde, 60s, 1920×1080 @ 30fps).** Stage 1.a (HockeyAI + ByteTrack default):
  167 player tracks (130 skaters + 37 goaltender fragments). Stage 1.b
  (HSV): 94/73 split, margin **1.46** (low — France dark-blue vs Monde
  light-green are less contrasted than the run12 white-vs-black; 41
  mixed-vote tracks). Stage 1.c iterated PARSeq→TrOCR:
  * **PARSeq** (new tight crop + letterbox): 19/167 numbered (11.4%),
    3 player groups.
  * **TrOCR** (same crops): 39/167 (**23.4%**), **10 player groups**.
    ~2× recall. Numbers found: #1, #2, #6, #9, #10, #19, #26, #35, #98.
  Stage 1.d (OSNet + greedy merge) on TrOCR output: **24 entities**
  (12/12 team split, 41 unmatched). 7/10 top entities labelled. User
  feedback: team classification still has errors on this clip (margin
  1.46 reflects real trouble); numbers not stable enough across the
  clip. Source video quality (motion blur, angle, 30 fps → less texture
  on small torsos) is now the bottleneck, not the algorithms.
- **run12 — Full pipeline on Video 03 (Vierzon vs Pont de Metz, 30s,
  1920×1080 @ 60fps).** Stage 1.a (HockeyAI + ByteTrack default): 250 player
  tracks (221 skaters + 29 goaltender fragments; HockeyAI does NOT tag
  refs on roller video — they leak as class_name='player'), 176 puck
  tracks, puck in **28.6%** of frames. Stage 1.a ran ~22 min wall-clock (MPS
  + YOLOv8m @ 1280 on 60fps pushes compute). Stage 1.c: 26/250
  (**10.4%**), 4 player groups (#5, #7×2, #10). Stage 1.b iterated v1→v3:
  * **v1** (BGR median-per-track, bbox-torso 10–40% × full width, sat
    filter): 123/127 split, margin 2.94. Centroids OK but labels visually
    wrong — at 9s all tracks blue, at 20s goalie + skater split across
    clusters. Root cause: sat-filter excludes the actual jersey pixels
    for near-grayscale teams (white/black), and per-track median is
    fragile on short tracks.
  * **v2** (pose-based torso + 3×2 multi-point dominant avg + per-crop
    majority vote + tight bbox-fallback 15–45% × 25–75% + val-only
    filter): 185/65 split, margin 2.15, centroids correctly pale
    (Vierzon) vs dark (Pont de Metz), only 12 mixed-vote tracks.
    User-validated as "practically perfect" except refs at 1s/11s
    (all green) and Pont de Metz goalie + neighbours at 14–17s (green).
  * **v3** (+ skater-only centroid fit, goalies classified post-hoc):
    183/67 split, goalie split 19/10 (identical to v2). Fix is
    technically correct but goalie color signatures (white pads) are
    fundamentally closer to the pale cluster. Accepted as final.
  Outputs preserved colocated: `tracks_teams_v1.json`,
  `tracks_teams_v2nogoalfix.json`, `annotated_numbered_v1.mp4`,
  `annotated_numbered_v2nogoalfix.mp4`; current `teams.json` /
  `annotated_numbered.mp4` = v3.
- **run11 — Stage 1.a HockeyAI + BoT-SORT+ReID+GMC (clip60-2).** 349 player
  tracks, 221 puck tracks. Longest track 639 frames (10.7s), 5 tracks ≥500
  frames. Longer tracks than run10 but slightly more of them — ReID from
  YOLO backbone features doesn't discriminate within a team (5 identical
  jerseys).
- **run10 — Stage 1.a HockeyAI + ByteTrack tuned (buffer=180, match=0.9).**
  312 player tracks, 196 puck tracks. ~28% fewer tracks than run04. Helps a
  bit; not enough.
- **run09 — Stage 1.c annotation v2.** Supervision-style boxes (green = team
  0, blue = team 1) + `#NN`/`#??` labels. Uses run04 tracks + run08
  identifications. `annotated_numbered.mp4` — user-validated look.
- **run08 — Stage 1.c OCR on run04 tracks.** 52/433 tracks identified (12%),
  21 distinct jersey numbers, **11 player groups** after number-based merge.
  High-confidence numbers: #14 (424 dets across 4 tracks), #24, #25, #77,
  #78, #19, #33, #92, #93, #91. Goalies correctly labelled (class_name =
  "goaltender"). Gated by track fragmentation (many tracks too short to
  have any back-facing sample).
- **run07 — Stage 3.a HockeyRink transfer test on Video 02 (Vierzon vs
  Rethel, 12 min).** 5/20 sampled frames had detections, best ones showed
  19–21 keypoints at high confidence — **but visually all keypoints were
  clustered in the same area**. Model recognises "there is a rink" but
  cannot localise landmarks on roller rink markings. Transfer failed.
- **run06 — Stage 3.a HockeyRink transfer on Video 03 (Vierzon vs Pont de
  Metz, 30s).** 1/10 frames with detection, 19 keypoints but geometrically
  wrong. Same failure mode as run07.
- **run05 / run05b — Stage 3.a HockeyRink transfer on clip60-2.** Even with
  very relaxed thresholds (conf=0.05, imgsz=1920, min-kp-conf=0.15),
  transfer score = 16–59%, per-frame keypoint count too low for RANSAC
  homography. Fundamentally insufficient.
- **run04 — Stage 1.a HockeyAI baseline (clip60-2, ByteTrack default).**
  This is **the canonical Stage 1.a output**. 17,860 player detections,
  433 tracks; puck detected in 1,431 / 3,360 frames (**42.6%**); 254 puck
  tracks. All downstream experiments (run08, 09, 10, 11) use this as input
  or as a comparison baseline.
- **run03 — First HockeyAI attempt.** Killed prematurely: output was not
  stuck, just Python's buffered stdout making progress invisible. Learning:
  always use `python -u` for long runs.
- **run02 — Stage 1.a COCO yolo11n baseline (clip60-2).** 63,224 person dets,
  1804 track IDs, only 2 puck frames out of 3360 (0.1%). Quantifies the
  baseline weakness that motivated HockeyAI adoption. **Note: ran on
  `yolo11n.pt` (nano) for speed, not `yolo11x.pt` (best quality). If the
  COCO path is ever revisited for a fair comparison, re-run with `yolo11x`.**
- **run01 — Stage 1.a first-ever run (clip60).** Crashed at ~35% of frames:
  `supervision >=0.27` requires `tracker_id` on the `Detections` object
  itself, not just as a local variable. Fixed in `p1_a_detect.py`
  (~line 83).

## Backlog

État courant des chantiers actifs, priorisé. Items concrets avec effort estimé
(XS <30min, S ~1h, M ~1j, L ~1sem, XL >1sem). Pour les chantiers structurels
multi-semaines (Phases 3–8+), voir la "Roadmap" en dessous.

### P0 — Itération courante
- **Revue visuelle des MP4 run21** par utilisateur sur Video 04 + 05.
  Confirmer : (a) numéros stables sur les players principaux,
  (b) goalies bien tagués, (c) erreurs résiduelles à catégoriser
  pour la prochaine itération. Aucun travail à faire de mon côté
  tant que le verdict n'est pas posé. — XS

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

## Roadmap

**Stage 3.a — Rink calibration & 2D map** (blocked on annotation)
- HockeyRink transfer failed (see run05–07). Unblocking requires 200–300
  roller-hockey rink keypoint annotations and fine-tuning HockeyRink on them.
- Alternative path: classical line detection + manual keyframe
  recalibration (every N seconds). Not prototyped.

**Phase 4 — Roller-specific fine-tune** (2–4 weeks, when we need it)
- HockeyAI covers the player and puck cases reasonably (run04). Would
  narrow the gap on motion-blurred / small / poorly-lit pucks, and help
  Stage 1.c by producing longer tracks.
- Build a roller-hockey dataset (CVAT or Roboflow); bootstrap with HockeyAI
  pre-labels.
- Fine-tune from HockeyAI weights (preferred) or from YOLO11.
- Target: >0.7 mAP on the puck class in roller conditions.

**Phase 5 — Event detection** (3–6 weeks)
- Goals, shots, penalties — temporal action models (TSN, MoViNet,
  SlowFast). Likely needs a custom labelled dataset.

**Stage 1.c — Player identification** (partial)
- OCR paths implemented: PARSeq (default) + TrOCR (`--ocr-engine trocr`).
- Stage 1.d (entity clustering) is done and absorbs a lot of the
  fragmentation. Remaining bottleneck is **OCR recall on small / blurry
  numbers** — addressed either by better source video (see limitation 1)
  or by an eventual Phase 4 fine-tune on jersey-number crops.
- Not done: ref isolation (Known limitations #5); entity-level
  multi-frame OCR consensus (could boost recall by voting across all
  crops of a merged entity, not just per track).

**Phase 7 — Web platform** (2–3 weeks)
- FastAPI backend + Next.js frontend. Match library, clip editor, tagging,
  sharing.

**Phase 8+** — Multi-camera stitching, live streaming (RTMP/HLS via
MediaMTX), mobile control app.

## Pistes explorées (non implémentées)

Veille technique archivée pour référence future. Rien ici n'est codé — voir
"Roadmap" pour les pistes engagées.

### Modèles hockey publics inventoriés (mars 2026)

Aucun modèle public n'est entraîné sur du roller inline hockey avec palet.
Le champ d'options se réduit à :

- **HockeyAI (utilisé)** — YOLOv8m, 2101 frames SHL, 7 classes. Meilleur
  socle disponible. Ref : [huggingface.co/SimulaMet-HOST/HockeyAI](https://huggingface.co/SimulaMet-HOST/HockeyAI).
- **Rink hockey YOLOv7** (Lopes et al., MDPI 2024) — 2525 frames, 7 classes
  (ball, player, stick, referee, crowd, goalkeeper, goal). **Piège** :
  "rink hockey" = variante quad portugaise/espagnole jouée avec une *balle*,
  pas un palet. Inutile pour la détection palet. Utile potentiellement pour
  ajouter une classe `stick` ou affiner `referee`/`goalkeeper` (contexte
  visuel indoor plus proche du RILH que la glace SHL). Dataset Roboflow :
  `visao-computacional/roller-hockey`.
- **SportsVision-YOLO** (forzasys-students) — YOLOv8 sur SHL, focus palet.
  Doublon probable avec HockeyAI.
- **sieve-data/hockey-vision-analytics** — pipeline complet GitHub qui
  utilise un modèle palet dédié (`hockey-puck-detection.pt`) avec inference
  slicing. Approche intéressante à reproduire si on cible la couverture
  palet.
- Petits modèles Roboflow (Sportobal 229 imgs, Trampoline 428 imgs, Peyton
  Burns 57 imgs) — trop petits pour être sérieux.

### Technique d'ensembling — Weighted Boxes Fusion (WBF)

Méthode de référence pour combiner les sorties de N détecteurs sur la même
frame. Contrairement à NMS qui supprime des boîtes, WBF utilise les scores
de confiance de toutes les boîtes pour construire des boîtes moyennes
pondérées, sans rien jeter. Pondération par modèle possible (ex: HockeyAI
×2, modèle palet ×1).

- Paper : Solovyev et al., "Weighted boxes fusion: Ensembling boxes from
  different object detection models", Image and Vision Computing 2021.
- Implémentation : [github.com/ZFTurbo/Weighted-Boxes-Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
  (`pip install ensemble-boxes`). API :
  `weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=...,
  iou_thr=0.5, skip_box_thr=0.0001)`.
- Point d'insertion dans notre pipeline : juste avant l'écriture de
  `detections.json` dans `p1_a_detect.py`. Tout l'aval reste compatible.

### Coût/bénéfice — pas une priorité actuelle

Options classées du plus rentable au moins rentable :

1. **`--imgsz 1280` ou `1536` avec HockeyAI seul.** Gain probable sur palet,
   coût quasi-nul. À tester *avant* toute approche ensemble.
2. **Re-ID clustering post-hoc sous contrainte d'équipe** — *déjà implémenté
   en Stage 1.d*. Ciblait le vrai blocage actuel (fragmentation tracks
   300 → ~12 entités, facteur 25). Aucun ensemble de détecteurs ne fera ça.
3. **Fine-tune HockeyAI sur 500–1000 frames RILH annotées** (Roadmap
   Phase 4). Battra tout ensemble de modèles ice-hockey à coup sûr.
4. **WBF HockeyAI + détecteur palet spécialisé** (style sieve-data). Gain
   estimé : couverture palet 42,6 % → ~55–65 %. Coût : inférence ~2×.
   Risque secondaire : fusion peut légèrement perturber l'association
   ByteTrack frame-à-frame. À envisager *après* Phase 4.
5. **WBF avec rink hockey YOLOv7 pour ajouter la classe `stick`.** Pas une
   amélioration de détection — une nouvelle capacité utile pour la détection
   d'événements (Phase 5).

L'ensembling devient pertinent une fois qu'on aura un modèle RILH-spécifique
(Phase 4), pour le marier avec HockeyAI et combiner connaissance du domaine
et robustesse pré-entraînée.

## Datasets (for later fine-tunes)
No public roller-hockey CV dataset currently. Build our own:
- Use HockeyAI (Stage 1.a output) to bootstrap annotations semi-automatically
- Roboflow Universe has ice hockey datasets — transfer is partial (HockeyAI
  works, HockeyRink doesn't)
- Annotate ~500–1000 frames with puck visible for the first puck fine-tune
- For Stage 3.a: ~200–300 frames with rink keypoints annotated (56-keypoint
  schema from the HockeyRink dataset, restricted to landmarks visible on
  roller rinks — centre circle, face-off dots, goal lines, corners)

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
