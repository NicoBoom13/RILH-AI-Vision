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
- **Phase 6** 🟡 partially implemented: jersey-number OCR on dorsal crops
  works when a track has enough back-facing samples. On `test08`, 52/433
  tracks were identified (12%) and 11 player groups emerged via number-based
  track merging. **The hard blocker is upstream**: tracker fragmentation
  produces ~300 tracks for ~12 real entities. Neither tuning ByteTrack nor
  switching to BoT-SORT+ReID+GMC solves this (31–28% fewer tracks, still far
  off). Next planned step is post-hoc Re-ID clustering under a hard team
  constraint (option "propre", not started).

## Architecture
Two/three-pass pipeline:
1. `src/phase1_detect_track.py` → produces `tracks.json` (per-frame bounding
   boxes, class IDs, persistent track IDs)
2. `src/phase2_followcam.py` → reads `tracks.json` + original video →
   produces follow-cam MP4
3. `src/phase6_identify.py` → reads `tracks.json` + video →
   `tracks_identified.json` with per-track jersey number + team merge groups
4. `src/phase6_annotate.py` → renders an annotated video with `#NN` / `#??`
   labels and green/blue team boxes using the outputs of (1) and (3)

Why multi-pass: detection is the slow step. Decoupling lets us iterate on
cinematography, identification, and analytics without re-running inference.

## Key design choices

### Detector backends (Phase 1)
Two detector backends, selectable at runtime in `phase1_detect_track.py`:
- Default — **YOLO11 pretrained on COCO**, classes 0 (person) and 32
  (sports ball). Player detection is solid; puck detection via "sports ball"
  is unreliable (<1% of frames on roller).
- `--hockey-model` — **[HockeyAI](https://huggingface.co/SimulaMet-HOST/HockeyAI)**,
  a YOLOv8m fine-tuned on ice hockey with 7 native classes (center ice,
  faceoff dots, goal frame, goaltender, players, puck, referee).
  Auto-downloaded to `models/HockeyAI_model_weight.pt`. Transfers well to
  roller inline hockey. Classes are remapped at the source so the output
  schema stays Phase-2-compatible: player + goaltender → `class_id=0`,
  puck → `class_id=32`, referee + rink markers are dropped.

Both backends write the same `tracks.json` schema (plus a `class_name`
string per detection), so Phases 2 and 6 are backend-agnostic.

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
2. Run **YOLO11-pose** on the full frames, match by IoU to the tracked bbox.
3. Classify **orientation** from pose keypoints (nose + eyes → front;
   ears without nose → back; shoulders only → side).
4. On back-facing samples, crop the **torso-back region** (shoulders → hips).
5. Run **PARSeq** scene-text recognition (via `torch.hub`, needs
   `pytorch_lightning`, `timm`, `nltk`), keep digits only, 1–2 chars.
6. **Vote** per track; the jersey number is the majority winner.
7. **Merge** tracks that share the same confident number and don't overlap
   in time — they're the same player with a broken track.

Annotation (`phase6_annotate.py`):
- Supervision-style boxes/labels/traces (same look as Phase 1's annotated
  MP4), but box color is **forced green or blue** based on k=2 k-means
  clustering over per-track median jersey color (sampled on torso crops
  10–40% of bbox height, HSV-filtered for saturation).
- Labels read `#NN` when identified, `#??` otherwise. No class name suffix.

## Known limitations (in priority order)
1. **Track fragmentation is the #1 blocker now.** ~12 real entities on the
   clip but trackers produce 300–430 track IDs. Neither a longer buffer nor
   BoT-SORT+ReID solves it (see test10/test11). Next step: post-hoc Re-ID
   clustering with hard constraint (5 skaters + goalie per team × 2 teams).
2. **Phase 3 calibration is blocked on annotation.** HockeyRink (ice) does
   not transfer to roller rinks — the model "recognises" a rink but collapses
   all 56 keypoints to a cluster instead of localising them individually.
   Unblocking requires 200–300 annotated frames of roller rinks.
3. **Puck detection quality** is workable (~43% of frames with HockeyAI vs
   <1% with COCO) but drops on motion blur, small pucks, and uneven lighting.
   Roller-specific fine-tune (Phase 4) would narrow the gap.
4. **Phase 6 coverage** is 12% of tracks in test08; this is gated by (1) —
   short fragment tracks rarely have enough back-facing samples for OCR.
5. No event detection yet (Phase 5).
6. Single-camera assumption. Multi-camera stitching = Phase 7.

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

**Phase 6 — Player identification** (in progress)
- OCR path implemented (`phase6_identify.py`). Current blocker:
  track fragmentation limits OCR coverage. **Next milestone: post-hoc Re-ID
  clustering under team constraint** — force ~300 tracks down to ~12
  entities via appearance embedding + max 6 entities per team + temporal
  non-overlap. Opt-in: won't replace the tracker, just merges its output.

**Phase 7 — Web platform** (2–3 weeks)
- FastAPI backend + Next.js frontend. Match library, clip editor, tagging,
  sharing.

**Phase 8+** — Multi-camera stitching, live streaming (RTMP/HLS via
MediaMTX), mobile control app.

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
