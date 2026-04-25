# RILH-AI-Vision — experiments archive

Material that was inline in `CLAUDE.md` until it grew too large to be re-read on every Claude turn. Lives here for on-demand consultation: full test log, long-term roadmap, technical surveys, dataset notes.

`CLAUDE.md` carries a short summary of each item with a pointer here.

---

## 1. Test run log (full)

Reverse-chronological. Each entry = one `runs/runNN/`. Params only list what differs from the immediate predecessor.

- **run22 — First end-to-end run via `src/run_project.py` orchestrator on Video 05.**
  Phase 1 → Phase 5 sequenced by the orchestrator (`--hockey-model
  --pose-model yolo26l-pose.pt --parseq-checkpoint
  models/parseq_hockey_rilh.pt`). Wall-clock ~84 min total. Phase 3
  produced 0/56 keypoints (parked, tolerant). Phase 4 + Phase 5 stubs
  wrote marker JSONs. Bug surfaced + fixed in this run: Stage 1.e was
  silently skipping because Stage 1.a's debug `annotated.mp4`
  (raw bboxes) collided with Stage 1.e's expected output filename → fixed by
  renaming Stage 1.a's output to `annotated_raw.mp4` (commit `5d37f3d`).
  Final artifacts in `runs/run22/`: `p1_a_detections.json`,
  `p1_b_teams.json`, `p1_c_numbers.json`, `p1_d_entities.json`,
  `annotated.mp4` (197 MB Stage 1.e), `annotated_raw.mp4` (196 MB
  Stage 1.a), `followcam.mp4` (50 MB), 5 rink debug PNGs.
- **run21 — Full pipeline avec PARSeq Hockey + RILH fine-tune sur
  Video 04 + 05.** Première run de l'OCR fine-tuné en pipeline complet.
  Réutilise `p1_a_detections.json` existants (run13 pour Video 04, run18 pour
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
  pipeline. Réutilise `p1_a_detections.json` de run04 (clip60-2), run13
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
  Réutilise `p1_a_detections.json` + `p1_b_teams.json` de run14. Ajouts dans le
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
  `annotated_numbered_v2nogoalfix.mp4`; current `p1_b_teams.json` /
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

---

## 2. Roadmap (long-term, multi-week chantiers)

`CLAUDE.md` carries a 6-line summary; the detail lives here.

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

---

## 3. Pistes explorées (non implémentées)

Veille technique archivée pour référence future. Rien ici n'est codé — voir "Roadmap" pour les pistes engagées.

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
  `p1_a_detections.json` dans `p1_a_detect.py`. Tout l'aval reste compatible.

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

---

## 4. Datasets (for later fine-tunes)

No public roller-hockey CV dataset currently. Build our own:
- Use HockeyAI (Stage 1.a output) to bootstrap annotations semi-automatically
- Roboflow Universe has ice hockey datasets — transfer is partial (HockeyAI
  works, HockeyRink doesn't)
- Annotate ~500–1000 frames with puck visible for the first puck fine-tune
- For Stage 3.a: ~200–300 frames with rink keypoints annotated (56-keypoint
  schema from the HockeyRink dataset, restricted to landmarks visible on
  roller rinks — centre circle, face-off dots, goal lines, corners)
