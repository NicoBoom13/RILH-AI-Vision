# Project context for Claude Code

## Project: RILH-AI-Vision

Computer-vision pipeline for roller inline hockey: automatic AI-driven
match recording, broadcast-style virtual follow-cam, and post-match
analytics. Built from open-source modules + custom code only.

## Current status
- **Phase 1** ✅ implemented: dual-backend detection (YOLO11 COCO or HockeyAI)
  + configurable tracker (ByteTrack default, BoT-SORT/ReID available).
- **Phase 2** ✅ implemented: virtual follow-cam. Built but not heavily iterated
  on; quality is acceptable but not the current priority.
- **Phase 3** ❌ attempted, deferred. The obvious shortcut (the HockeyRink
  pretrained keypoint detector from the same team as HockeyAI) **does not
  transfer to roller rinks** — see "Test run log" below. Requires a
  roller-specific annotated dataset + fine-tune. Parked.
- **Phase 1.5** ✅ implemented (v3): team clustering via
  pose-based torso crop (YOLO11-pose shoulders→hips, bbox-fallback
  15–45% × 25–75% for dark jerseys the pose model misses) + multi-point
  dominant averaging (3×2 grid inside the torso) + per-crop k-means
  majority vote (fit on skaters only, goalies classified post-hoc against
  those centroids). HSV by default (`--space bgr` also available).
  `dominant_bgr` uses a val-only filter (no sat bias) because near-
  grayscale jerseys like white/black carry their signal in V, not H/S.
  User-validated on test12 as "practically perfect". Residual ~0.3% of
  frames: referees (HockeyAI mislabels them 'player' on roller → can't
  filter by class_name) and the Pont de Metz goalie (white pads →
  classified pale). Further improvement needs position-based assignment
  (needs Phase 3) or custom goalie sampling — deferred.
- **Phase 1.6** ✅ implemented: post-hoc **Re-ID clustering** that collapses
  fragmented tracks into stable entities (one entity = one real player /
  goalie / ref). Uses OSNet x0_25 (via `torchreid`) medoid embedding per
  track + greedy merge under **team constraint** (from Phase 1.5),
  **strict temporal non-overlap**, and **OCR bonus** (from Phase 6 identify).
  OCR conflicts are a hard block. Output: `tracks_entities.json` consumed
  downstream by `phase6_annotate.py`. On test12 (250 tracks → 40 entities),
  on test13 (167 tracks → 24 entities). Doesn't replace the tracker — it
  post-processes its output. Design doc: `docs/phase_1_6_design.md`.
- **Phase 6** 🟡 partially implemented: jersey-number OCR on dorsal crops.
  Two OCR engines plumbed in — PARSeq (default, fast, via torch.hub) and
  Microsoft TrOCR (`--ocr-engine trocr`, heavier ~340 MB, ~2× recall on
  difficult video). Crop is a **tight horizontal band of the back** with
  **letterbox padding to PARSeq's 4:1 aspect** (added after observing that
  stretched-square crops confuse digit recognition). On `test08` (old
  crop): 52/433 tracks identified (12%), 11 player groups. On `test13`
  (new crop + TrOCR): 39/167 (23%), 10 player groups. **Upstream blocker
  (fragmentation) is partly addressed by Phase 1.6**, but OCR itself
  caps out on small / motion-blurred / oblique-angle jersey numbers —
  video quality is the bottleneck after ~test13.

## Architecture
Multi-pass pipeline:
1. `src/phase1_detect_track.py` → `tracks.json` (per-frame bboxes, class IDs,
   persistent track IDs)
2. `src/phase1_5_teams.py` → `tracks_teams.json`: team_id (0/1) per player
   track via k=2 on pose-based torso color (HSV default, skater-only fit,
   goalies classified post-hoc)
3. `src/phase2_followcam.py` → follow-cam MP4 (uses only `tracks.json`)
4. `src/phase6_identify.py` → `tracks_identified.json`: per-track jersey
   number via YOLO11-pose + PARSeq or TrOCR. Tight torso-back crop +
   letterbox pad to 4:1
5. `src/phase1_6_entities.py` → `tracks_entities.json`: fragmented tracks
   collapsed into stable entities via OSNet embeddings + team/overlap/OCR
   constraints (greedy merge)
6. `src/phase6_annotate.py` → annotated MP4 with entity-level color +
   `#NN`/`#??` labels (entity labels shared across all merged tracks).
   Falls back to per-track labels if (5) hasn't been run.

Why multi-pass: detection is the slow step. Decoupling lets us iterate on
cinematography, identification, and analytics without re-running inference.

## Key design choices

### Detector backends (Phase 1)
Two detector backends, selectable at runtime in `phase1_detect_track.py`:
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
  schema stays Phase-2-compatible: player + goaltender → `class_id=0`,
  puck → `class_id=32`, referee + rink markers are dropped.

Both backends write the same `tracks.json` schema (plus a `class_name`
string per detection), so Phases 2 and 6 are backend-agnostic.

**Match vs training mode.** Phase 1 enforces 1-puck-per-frame by default
(real match conditions): when multiple puck detections appear in the
same frame, only the highest-confidence one is kept (after the tracker
has assigned IDs, so the dropped duplicates simply never reach
`tracks.json`). Pass `--training-mode` to disable the filter — useful
for drills where multiple pucks are intentionally on the ice.

### Tracker backends (Phase 1)
Tracker is configurable via `--tracker <yaml>`:
- `bytetrack.yaml` (default) — motion-only, fast, fragmented on occlusions.
- `configs/bytetrack_tuned.yaml` — same tracker with `track_buffer=180` and
  `match_thresh=0.9` (longer memory, more permissive association).
- `configs/botsort_reid.yaml` — BoT-SORT with GMC (camera motion
  compensation) and ReID (appearance features from the YOLO backbone).
  Slower but makes individual tracks longer — see test11.

### Follow-cam (Phase 2)
- **Focus point = weighted blend of puck position and players centroid**.
  Puck gets high weight when detected; players-centroid fallback otherwise.
  Recently-seen puck positions are extrapolated for ~15 frames to bridge
  missed detections.
- **Smoothing = exponential moving average** on the focus trajectory,
  optionally followed by a centered boxcar pass for extra polish.
- **Crop window clamped to frame bounds** so we never show black bars.

### Jersey identification (Phase 6)
`phase6_identify.py` pipeline, one pass per track (not per frame — keeps
inference tractable):
1. Sample the N highest-confidence detections for each track_id.
2. Run the **YOLO pose model** on the full frames, match by IoU to the
   tracked bbox. Default is `yolo11n-pose.pt` (~6 MB); `--pose-model
   yolo26l-pose.pt` (~55 MB) is available for better keypoint recall on
   dark / low-contrast / motion-blurred torsos. Both auto-download into
   `models/` on first use.
3. Classify **orientation** from pose keypoints (nose + eyes → front;
   ears without nose → back; shoulders only → side).
4. On back-facing samples, crop a **tight horizontal band** of the back
   (15–65 % of shoulder→hip height, shoulder width + 30 % pad). This is
   narrower than the full torso but matches the region where the jersey
   number actually sits.
5. **Letterbox-pad** the crop to the OCR engine's expected aspect (1:4 h:w
   for PARSeq's 32×128) before inference — avoids horizontal stretching
   of digits that used to cause 6↔7 and 0↔6 confusions.
6. Run OCR (`--ocr-engine {parseq,trocr}`):
   - **PARSeq** (default, via `torch.hub`, `pytorch_lightning`+`timm`+`nltk`)
   - **TrOCR** (`microsoft/trocr-base-printed`, via `transformers`+
     `sentencepiece`): heavier (~340 MB) but ~2× recall on difficult text.
7. Keep digits only, 1–2 chars. **Vote** per track; the jersey number is
   the majority winner.
8. **Merge** tracks that share the same confident number and don't overlap
   in time — they're the same player with a broken track.

### Entity clustering (Phase 1.6)
`phase1_6_entities.py` post-processes fragmented tracks into stable entities:
1. Per-track **OSNet x0_25 medoid embedding** (512-d, L2-normalised) over
   the top-N confidence detections.
2. Build candidate merge graph: all pairs `(a, b)` with **same team_id
   from Phase 1.5** (with vote_confidence ≥ 0.67 on both sides), **zero
   temporal overlap**, no OCR conflict (same team + different confident
   numbers rejects the pair).
3. Edge weight = `cos_sim(emb_a, emb_b) + 10·1[same_jersey] +
   0.05·1[both_goalie]`. OCR-seeded pairs always win (weight ≥ 10).
4. Greedy merge in descending weight until similarity drops below
   `--sim-threshold` (default 0.65). Re-checks overlap on the merged
   cluster each time.
5. Output: `tracks_entities.json` with `track_ids` lists, derived
   `team_id`/`is_goaltender`/`jersey_number`/`name`/frame ranges, plus a
   list of unmatched singleton tracks.

**Goalie weighting** (added test16):
- Phase 1.5 `is_goaltender` per track now uses **majority threshold**
  (>50 % of detections tagged `goaltender`), replacing the previous
  `any` rule that flipped a track on a single noisy frame.
- Phase 1.6 entity-level `is_goaltender` is **weighted by frame
  coverage**: an entity is goalie only if more than half of its merged
  tracks' total frame count comes from goalie-tagged tracks.
- Phase 6 identify aggregation requires **≥2 agreeing votes** for the
  winning number/name (singletons are usually noise — TrOCR can produce
  a confident wrong digit on a single bad crop).

**Spectator handling** — see test17 for why a motion+position filter was
attempted then **removed**: it dropped ~50 % of real player fragments to
catch ~30 spectators, while the 7 spectators HockeyAI tags as goalies
slipped through anyway. The plan is to wait for **Phase 3 rink
calibration** (geometric "is this bbox on the ice?") to do this cleanly.

Annotation (`phase6_annotate.py`):
- Supervision-style boxes/labels/traces. Box color is **forced green or
  blue** per team.
- **Entity-aware**: if `tracks_entities.json` exists, the label + team come
  from the entity (so every merged fragment shares the same `#NN`, name
  and colour across the video). Otherwise, per-track values from
  `tracks_teams.json` + `tracks_identified.json`.
- Labels read `t{id} {G|S} #NN NAME` — track_id always shown (so user
  can give frame-level feedback), `G`/`S` from `is_goaltender`, `#??`
  if no number, name omitted if not identified.
- **Puck**: rendered as a dark gray bbox (60,60,60) + short trace.
  No label.
- **Spectators**: not filtered (will be addressed by Phase 3 rink
  calibration). All tracked detections render — including stationary
  bystanders that HockeyAI tags as players.

## Known limitations (in priority order)
1. **Video source quality is the current bottleneck** (post test13). Torso
   crops of 30–50 px with motion blur and oblique angles defeat even TrOCR;
   team colour clustering margin drops when jerseys aren't sharply contrasted.
   No algorithmic fix short of fine-tuning — needs better input footage
   (stable camera, good lighting, close distance, high contrast jerseys).
2. **Track fragmentation** is real but **partly absorbed** by Phase 1.6
   (test16: 435 tracks → 37 entities, of which only 5 G after the
   goalie majority + frame-coverage rules). HockeyAI still
   intermittently flips goaltender/player class on individual frames,
   but the majority threshold + frame-coverage weighting mostly
   neutralises that. Refs still merge into the two teams (no third
   cluster) — see #5. **Stationary spectators that HockeyAI mistags as
   `player` or `goaltender` still pollute entities** — the Phase 1.5
   spectator filter attempted in test17 over-filtered real players and
   was reverted; clean fix waits for Phase 3 rink calibration.
3. **Phase 3 calibration** is blocked on annotation. HockeyRink (ice) does
   not transfer to roller rinks — the model "recognises" a rink but collapses
   all 56 keypoints to a cluster instead of localising them individually.
   Unblocking requires 200–300 annotated frames of roller rinks.
4. **Puck detection quality** is workable (~43% of frames with HockeyAI vs
   <1% with COCO) but drops on motion blur, small pucks, and uneven lighting.
   Roller-specific fine-tune (Phase 4) would narrow the gap.
5. **Refs leak into team clusters** because HockeyAI doesn't recognise
   roller-hockey referee uniforms — they're tagged `class_name='player'`.
   Fixable by k=3 clustering in Phase 1.5 or a dedicated ref detector.
6. No event detection yet (Phase 5).
7. Single-camera assumption. Multi-camera stitching = Phase 7.

## Conventions
- Python 3.10+ (project uses 3.12 in the dev venv)
- All paths via `pathlib.Path`
- CLI scripts use `argparse`
- All model weights live under `models/` (both explicit downloads and
  Ultralytics auto-downloads — `phase1_detect_track.py` routes bare YOLO
  names like `yolo11m.pt` into `models/` via `resolve_model_path`).
- Tracker configs live under `configs/` (YAML)
- Outputs go under `runs/testNN/` (gitignored). **Never overwrite** a
  previous run — incremental numbering is the convention, so past results
  stay available for comparison.
- Don't commit videos (`videos/`) or model weights (`*.pt`)

## Test run log

Reverse-chronological. Each entry = one `runs/testNN/`. Params only list
what differs from the immediate predecessor.

- **test17 — Villeneuve vs Vierzon (Video 05) — filtre spectator multi-signal.**
  Pipeline complet relancé. Phase 1 inchangée vs test16 (HockeyAI +
  match-mode default, 1 puck/frame). Nouveauté Phase 1.5 : `is_static` =
  `NOT goaltender AND (median_disp < 5px OR median_y > 0.74 ×
  frame_height)`. Goalies exemptés du filtre (préservation des vrais
  goalies stationnaires). Résultats :
  * **Static tracks** : **261 / 435 (60 %)** flaggés spectator,
    excluded de Phase 6 identify, 1.6, et annotate.
  * **Phase 6 identify** : 1605 samples (vs 3696 test16, **-57 %
    compute**). 16 / 174 tracks identifiés (9.2 %), 19 noms (10.9 %).
    Numéros : #2, #77, #6, #4 — moins variés qu'avant car beaucoup de
    tracks éliminées avant OCR.
  * **Phase 1.6** : **20 entités** (vs 37 test16, **-46 %**).
    Approche du compte réel attendu (2 équipes × 4-5 ≈ 10 entities/team).
    Goalie entities : 6 (vs 5 test16, +1 — les 7 spectators-tagged-
    goalie passent toujours le filtre, contribuent à des entités G).
    Top-10 : 3G/7S (vs 2G/8S test16).
  * **Vidéo finale** : 176 MB (vs 209 test16). Annotated propre.
  * **Hallucinations TrOCR persistent au niveau entity** : entity [0]
    `S #49 CASH`, [2] `G #2 AMOUNT`, [4] `S #79 TAX`, [6] `S #79 QTY`,
    [7] `G #4 CASHIER`, [8] `S #69 MCAUD`. Le filtre spectator a viré
    les tracks isolées contenant ces hallucinations, mais TrOCR continue
    à les produire sur certains crops illisibles de vrais joueurs.
    Stop-list = next-step P1.
  Verdict utilisateur après revue visuelle : **filtre spectator
  RETIRÉ** — trop de vrais joueurs faux-droppés (~50 % des fragments)
  pour ne catcher que ~30 spectators sur 37 listés, et les 7
  spectators-tagged-goalie passaient quand même. Le bon outil sera la
  calibration rink (Phase 3) qui répondra géométriquement « ce bbox
  est-il sur la glace ou en tribune ? ». Tout le reste de la v2 est
  conservé (name OCR, debug, palet gris, goalie majority + frame-
  weighting, OCR ≥2 votes, ocr_min_conf 0.30, ocr_conflict 0.40,
  training-mode default, label `t{id} G/S #num NAME`).
- **test16 — Villeneuve vs Vierzon (Video 05) — 5 fix consolidés.** Première
  run avec `--match-mode` (top-1 puck/frame), goalie majority rule,
  entity goalie weighted by frame coverage, OCR ≥2 votes, `ocr_min_conf`
  0.40→0.30, `ocr_conflict_conf_floor` 0.55→0.40, palet rendu dans la
  vidéo finale. Phase 1 rejouée (1h wall-clock), Phases 1.5/6/1.6/annotate
  neuves. Résultats :
  * **Palet** : 0 frame avec 2+ pucks ✓ (test15: 12.9% en avaient, max 5).
    260 → 185 puck tracks (filtrage des faux positifs).
  * **Goalies flag Phase 1.5** : 122 → **73 tracks** (-40%) via majority
    rule (>50% des détections tagées goaltender).
  * **Goalie entities Phase 1.6** : 17 → **5** (-71%) via frame-coverage
    weighting. Top-10 passé de 9G/1S à 2G/8S.
  * **Numéros** : 66 (test15) → 49 tracks identifiés, **noms** 96 → 66.
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
  median per-frame displacement < 5px (propagé Phase 1.5→6_identify→
  1.6→annotate), palet BGR(60,60,60), inversion `--match-mode` →
  `--training-mode` (1-puck désormais par défaut).
- **test15 — Villeneuve vs Vierzon (Video 05) — debug mode + name OCR.**
  Réutilise `tracks.json` + `tracks_teams.json` de test14. Ajouts dans le
  code : name OCR (crop au-dessus du numéro, TrOCR max_new_tokens=16
  pour noms / 6 pour numéros), `--debug-crops-dir` (2606 crops number+
  name sauvés), `--debug-frames-dir --debug-frames-step 10` dans
  phase6_annotate (360 PNG). Label final enrichi `t{id} {G/S} #num NAME`.
  Résultats :
  * Numéros : **66/435 (15.2%)** vs test14 59/435 (+12%). Noms :
    **96/435 (21.6%)** (nouveau). 16 player groups (== test14).
  * Entités Phase 1.6 : 36 (vs 37 test14). **17 taguées goalie** (9 du
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
    complet qui a mené aux 5 fix de test16.
- **test14 — Villeneuve vs Vierzon (Video 05, 60s, 1920×1080 @ 60fps).**
  Pipeline complète, première run production de YOLO26L-pose.
  * **Phase 1** ✅ HockeyAI + ByteTrack default. 3600 frames. 435 player
    tracks (313 skaters + 122 goaltenders), 260 puck tracks, puck dans
    **1870/3600 frames (51,9 %)** — meilleur qu'test04 (42,6 %) et
    test12 (28,6 %). Runtime ~1h wall-clock. (Note : un premier essai
    accidentel avec `--model yolo26l.pt` (COCO) a été stoppé pour
    repartir sur HockeyAI.)
  * **Phase 1.5** ✅ HSV k=2 avec `--pose-model yolo26l-pose.pt` (premier
    run production de YOLO26L-pose). 282/152 split, marge 1,95,
    27 mixed-vote tracks, 1 seul track sans sample. Pose-based : 1712
    crops (vs 656 bbox-fallback) = **72 % de crops pose** contre ~57 %
    sur test13 avec yolo11n-pose → YOLO26L-pose améliore bien le taux
    de succès des keypoints sur les torses dark/blurry.
  * **Phase 6 identify** ✅ TrOCR + yolo26l-pose. **59/435 (13,6 %)**,
    16 groupes joueurs. Numbers trouvés (top par fragments) : #1 (10),
    #4 (4), #2/#9/#6 (3), #77/#5/#09/#19 (2). En absolu, 59 tracks
    identifiés > test13 (39) — vidéo plus longue + plus de tracks aide.
  * **Phase 1.6** ✅ OSNet x0_25 + greedy merge. **435 tracks → 37
    entités** (28 unmatched), 370 merges, 6061 paires skippées (overlap).
    Split 25 team 0 / 12 team 1. **Anomalie** : 19/37 entités taguées
    goaltender, dont 9 du top 10 — HockeyAI flippe massivement la classe
    `goaltender`/`player` sur cette vidéo (limitation #2 amplifiée ici).
  * **Phase 6 annotate** ✅ entity-aware (37 entités couvrent 407 tracks).
    Sortie : `runs/test14/annotated_numbered.mp4`.
  Verdict utilisateur : détections équipe + joueurs « pas mal du tout
  mais pas parfaites ». Points de bug précis à documenter au prochain pass.
- **test13 — Full pipeline + Phase 1.6 + TrOCR on Video 04 (France vs
  Monde, 60s, 1920×1080 @ 30fps).** Phase 1 (HockeyAI + ByteTrack default):
  167 player tracks (130 skaters + 37 goaltender fragments). Phase 1.5
  (HSV): 94/73 split, margin **1.46** (low — France dark-blue vs Monde
  light-green are less contrasted than the test12 white-vs-black; 41
  mixed-vote tracks). Phase 6 identify iterated PARSeq→TrOCR:
  * **PARSeq** (new tight crop + letterbox): 19/167 numbered (11.4%),
    3 player groups.
  * **TrOCR** (same crops): 39/167 (**23.4%**), **10 player groups**.
    ~2× recall. Numbers found: #1, #2, #6, #9, #10, #19, #26, #35, #98.
  Phase 1.6 (OSNet + greedy merge) on TrOCR output: **24 entities**
  (12/12 team split, 41 unmatched). 7/10 top entities labelled. User
  feedback: team classification still has errors on this clip (margin
  1.46 reflects real trouble); numbers not stable enough across the
  clip. Source video quality (motion blur, angle, 30 fps → less texture
  on small torsos) is now the bottleneck, not the algorithms.
- **test12 — Full pipeline on Video 03 (Vierzon vs Pont de Metz, 30s,
  1920×1080 @ 60fps).** Phase 1 (HockeyAI + ByteTrack default): 250 player
  tracks (221 skaters + 29 goaltender fragments; HockeyAI does NOT tag
  refs on roller video — they leak as class_name='player'), 176 puck
  tracks, puck in **28.6%** of frames. Phase 1 ran ~22 min wall-clock (MPS
  + YOLOv8m @ 1280 on 60fps pushes compute). Phase 6 identify: 26/250
  (**10.4%**), 4 player groups (#5, #7×2, #10). Phase 1.5 iterated v1→v3:
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
  `annotated_numbered_v2nogoalfix.mp4`; current `tracks_teams.json` /
  `annotated_numbered.mp4` = v3.
- **test11 — Phase 1 HockeyAI + BoT-SORT+ReID+GMC (clip60-2).** 349 player
  tracks, 221 puck tracks. Longest track 639 frames (10.7s), 5 tracks ≥500
  frames. Longer tracks than test10 but slightly more of them — ReID from
  YOLO backbone features doesn't discriminate within a team (5 identical
  jerseys).
- **test10 — Phase 1 HockeyAI + ByteTrack tuned (buffer=180, match=0.9).**
  312 player tracks, 196 puck tracks. ~28% fewer tracks than test04. Helps a
  bit; not enough.
- **test09 — Phase 6 annotation v2.** Supervision-style boxes (green = team
  0, blue = team 1) + `#NN`/`#??` labels. Uses test04 tracks + test08
  identifications. `annotated_numbered.mp4` — user-validated look.
- **test08 — Phase 6 OCR on test04 tracks.** 52/433 tracks identified (12%),
  21 distinct jersey numbers, **11 player groups** after number-based merge.
  High-confidence numbers: #14 (424 dets across 4 tracks), #24, #25, #77,
  #78, #19, #33, #92, #93, #91. Goalies correctly labelled (class_name =
  "goaltender"). Gated by track fragmentation (many tracks too short to
  have any back-facing sample).
- **test07 — Phase 3 HockeyRink transfer test on Video 02 (Vierzon vs
  Rethel, 12 min).** 5/20 sampled frames had detections, best ones showed
  19–21 keypoints at high confidence — **but visually all keypoints were
  clustered in the same area**. Model recognises "there is a rink" but
  cannot localise landmarks on roller rink markings. Transfer failed.
- **test06 — Phase 3 HockeyRink transfer on Video 03 (Vierzon vs Pont de
  Metz, 30s).** 1/10 frames with detection, 19 keypoints but geometrically
  wrong. Same failure mode as test07.
- **test05 / test05b — Phase 3 HockeyRink transfer on clip60-2.** Even with
  very relaxed thresholds (conf=0.05, imgsz=1920, min-kp-conf=0.15),
  transfer score = 16–59%, per-frame keypoint count too low for RANSAC
  homography. Fundamentally insufficient.
- **test04 — Phase 1 HockeyAI baseline (clip60-2, ByteTrack default).**
  This is **the canonical Phase 1 output**. 17,860 player detections,
  433 tracks; puck detected in 1,431 / 3,360 frames (**42.6%**); 254 puck
  tracks. All downstream experiments (test08, 09, 10, 11) use this as input
  or as a comparison baseline.
- **test03 — First HockeyAI attempt.** Killed prematurely: output was not
  stuck, just Python's buffered stdout making progress invisible. Learning:
  always use `python -u` for long runs.
- **test02 — Phase 1 COCO yolo11n baseline (clip60-2).** 63,224 person dets,
  1804 track IDs, only 2 puck frames out of 3360 (0.1%). Quantifies the
  baseline weakness that motivated HockeyAI adoption. **Note: ran on
  `yolo11n.pt` (nano) for speed, not `yolo11x.pt` (best quality). If the
  COCO path is ever revisited for a fair comparison, re-run with `yolo11x`.**
- **test01 — Phase 1 first-ever run (clip60).** Crashed at ~35% of frames:
  `supervision >=0.27` requires `tracker_id` on the `Detections` object
  itself, not just as a local variable. Fixed in `phase1_detect_track.py`
  (~line 83).

## Backlog

État courant des chantiers actifs, priorisé. Items concrets avec effort estimé
(XS <30min, S ~1h, M ~1j, L ~1sem, XL >1sem). Pour les chantiers structurels
multi-semaines (Phases 3–8+), voir la "Roadmap" en dessous.

### P0 — Prochaine itération
- **Pipeline test18** : pipeline complet sans le filtre spectator
  (retiré). Sert de baseline propre pour mesurer Phase 3 quand elle
  arrivera, et pour valider que les wins v2 (goalie majority/weighting,
  OCR ≥2 votes, name OCR, training-mode default, palet rendu) tiennent
  hors filtre spectator. Phase 1 ~1h. — M

### P1 — Court terme
- **Stop-list TrOCR** pour filtrer les hallucinations type `CASHIER`,
  `CASH`, `AMOUNT`, `TAX`, `QTY`, `ITEM`, `MCAUD`, `MAY` (vocabulaire
  reçus). À placer dans `_filter_name` de `phase6_identify.py`. Sans
  filtre spectator, ces hallucinations vont apparaître sur encore plus
  d'entités → priorité bumpée. — XS
- **Cache pose entre Phase 1.5 et Phase 6 identify** : aujourd'hui
  yolo26l-pose tourne 2× sur les mêmes frames. Écrire `pose_cache.json`
  en P1.5 et le lire en P6 sauve ~20-30 % wall-clock pipeline. — S
- **Phase 1.5 + Phase 6 identify en parallèle** (processus séparés).
  Gain ~10-20 % supplémentaire (overlap GPU/CPU). — S

### P2 — Court-moyen terme
- **Fix team classification Phase 1.5** : sur test15, 6 tracks/435 mal
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

### P3 — Long terme (cf. Roadmap)
- **Phase 4** : fine-tune HockeyAI sur 500–1000 frames RILH annotées.
  Bloqueur = annotation. Débloque puck + goalie + classes refs. — XL
- **Fine-tune TrOCR** sur crops jersey RILH (numéros + noms). Élimine les
  hallucinations receipt-style. — L
- **Phase 3** : calibration rink + map 2D. Bloqué sur 200–300 annotations
  keypoints rink roller. — XL
- **Phase 5** : détection événements (buts, tirs, fautes). — XL
- **Phase 7** : plateforme web FastAPI + Next.js. — L
- **Phase 8+** : multi-cam stitching, live RTMP/HLS. — XL

## Roadmap

**Phase 3 — Rink calibration & 2D map** (blocked on annotation)
- HockeyRink transfer failed (see test05–07). Unblocking requires 200–300
  roller-hockey rink keypoint annotations and fine-tuning HockeyRink on them.
- Alternative path: classical line detection + manual keyframe
  recalibration (every N seconds). Not prototyped.

**Phase 4 — Roller-specific fine-tune** (2–4 weeks, when we need it)
- HockeyAI covers the player and puck cases reasonably (test04). Would
  narrow the gap on motion-blurred / small / poorly-lit pucks, and help
  Phase 6 by producing longer tracks.
- Build a roller-hockey dataset (CVAT or Roboflow); bootstrap with HockeyAI
  pre-labels.
- Fine-tune from HockeyAI weights (preferred) or from YOLO11.
- Target: >0.7 mAP on the puck class in roller conditions.

**Phase 5 — Event detection** (3–6 weeks)
- Goals, shots, penalties — temporal action models (TSN, MoViNet,
  SlowFast). Likely needs a custom labelled dataset.

**Phase 6 — Player identification** (partial)
- OCR paths implemented: PARSeq (default) + TrOCR (`--ocr-engine trocr`).
- Phase 1.6 (entity clustering) is done and absorbs a lot of the
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
  `tracks.json` dans `phase1_detect_track.py`. Tout l'aval reste compatible.

### Coût/bénéfice — pas une priorité actuelle

Options classées du plus rentable au moins rentable :

1. **`--imgsz 1280` ou `1536` avec HockeyAI seul.** Gain probable sur palet,
   coût quasi-nul. À tester *avant* toute approche ensemble.
2. **Re-ID clustering post-hoc sous contrainte d'équipe** — *déjà implémenté
   en Phase 1.6*. Ciblait le vrai blocage actuel (fragmentation tracks
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
- Use HockeyAI (Phase 1 output) to bootstrap annotations semi-automatically
- Roboflow Universe has ice hockey datasets — transfer is partial (HockeyAI
  works, HockeyRink doesn't)
- Annotate ~500–1000 frames with puck visible for the first puck fine-tune
- For Phase 3: ~200–300 frames with rink keypoints annotated (56-keypoint
  schema from the HockeyRink dataset, restricted to landmarks visible on
  roller rinks — centre circle, face-off dots, goal lines, corners)

## Things to know when iterating
- Test on a 60s clip first: `ffmpeg -i input.mp4 -ss 0 -t 60 -c copy clip.mp4`
- **Always use `python -u`** when launching long runs to a file — stdout is
  block-buffered otherwise, which makes progress invisible and triggered a
  false "stuck" diagnosis in test03.
- For any serious puck work, pass `--hockey-model` — the COCO default is
  only useful when iterating fast and puck quality doesn't matter.
- Camera too slow in Phase 2 → raise `--alpha`. Camera jittery → lower it,
  or raise `--polish-window`.
- Still missing puck detections → `--imgsz 1280` or 1536 (slower but much
  better on small objects).
- Use `--debug-overlay` (Phase 2) to understand the focus trajectory.
- Tracker comparisons: pass `--tracker configs/bytetrack_tuned.yaml` or
  `configs/botsort_reid.yaml`. BoT-SORT+ReID is ~same speed as ByteTrack
  in practice (ReID features come from the YOLO backbone, no extra model
  download).
- Run outputs are incremental: `runs/test01/`, `runs/test02/`, … Don't
  overwrite a previous run even if it failed — past tracks + videos are
  useful for diffing the effect of a parameter change.
