# RILH-AI-Vision — Jersey number dataset

Manually annotated jersey-number crops collected from four roller inline
hockey video clips.

## Contents

- `crops/<video>/*.png` — dorsal jersey crops produced by Phase 6 identify
  with the Koshkina-style pose-guided crop strategy (full shoulder→hip
  torso + 5 px padding, see `src/phase6_identify.py:torso_back_crop`).
- `annotations.json` — `{ "annotations": { <relative crop path>: <label>, … }, "metadata": {…} }`.
  Label is `"0"`–`"99"` for a visible jersey number, or `"X"` if no number
  is readable in the crop (back not facing, occluded, blurred, partial).

## Source videos

| Folder | Source clip | Match | Approx. duration |
|---|---|---|---|
| `crops/clip60/` | clip60.mp4 | unidentified test clip | 60 s |
| `crops/clip60-2/` | clip60-2.mp4 | unidentified test clip | 60 s |
| `crops/video04/` | Video 04 - France vs. Monde (42m30-43m30).mp4 | France vs Monde | 60 s |
| `crops/video05/` | Video 05 - Villeneuve vs. Vierzon (1min).mp4 | Villeneuve vs Vierzon | 60 s |

## How crops were generated

1. Phase 1 (HockeyAI YOLOv8m) detects players + tracks them across frames.
2. Phase 6 identify samples the top-N highest-confidence detections per
   track, runs YOLO26L-pose, classifies orientation, and on back-facing
   samples crops the dorsal region using the four torso keypoints
   (left/right shoulders, left/right hips) plus 5 px padding.
3. The crop filename encodes track id, frame index, and the OCR engine's
   predicted label (a hint, often wrong on hard crops):
   `t{track_id:04d}_f{frame:05d}_num-{ocr_pred}_c{ocr_conf:02d}.png`

## How annotations were collected

Manual annotation via `tools/annotate_crops.py` — a localhost web UI
that displays each crop one at a time, pre-fills the input with the
OCR engine's filename hint, and lets the annotator confirm or correct
with a single keystroke.

## Dataset stats

See `metadata` block in `annotations.json` for the per-video breakdown.
