"""
RILH-AI-Vision — consolidate jersey-number annotations into a portable
license-clean dataset.

Reads:
  runs/test19/<video>/debug_crops/numbers/*.png  (image files)
  runs/test19/annotations.json                   (path → label mapping)

Writes:
  data/jersey_numbers/
    crops/<video>/<original_filename>.png
    annotations.json    (paths remapped to crops/<video>/...)
    README.md
    LICENSE.md

Usage:
  python tools/build_jersey_dataset.py [--dry-run]

The dataset is independent of any third-party model weights (it is just
your annotated crops), so you can later choose any license for it
without affecting the licence of any model fine-tuned on it.
"""

import argparse
import json
import shutil
from pathlib import Path


SOURCE = Path("runs/test19")
DEST = Path("data/jersey_numbers")
VIDEOS = ["clip60", "clip60-2", "video04", "video05"]


README = """\
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
"""


LICENSE_PLACEHOLDER = """\
# License

This dataset (the annotated crops + `annotations.json`) was produced by
the RILH-AI-Vision project. Choose your preferred license — the dataset
itself is independent of any third-party model weights used during the
crop-generation pipeline.

Common options for a small CV dataset:
  - **CC-BY 4.0** — attribution required, no other restrictions, free
    for commercial and non-commercial use.
  - **CC-BY-NC 4.0** — attribution required, non-commercial only.
  - **CC0** — public domain, no restrictions.

→ Replace this placeholder with your actual license text.
"""


def main():
    p = argparse.ArgumentParser(description="Build license-clean jersey dataset")
    p.add_argument("--dry-run", action="store_true",
                   help="Don't copy/write anything, just report counts.")
    p.add_argument("--source", default=str(SOURCE),
                   help="runs/<test>/ root containing <video>/debug_crops/numbers/")
    p.add_argument("--dest", default=str(DEST),
                   help="Destination root for the consolidated dataset.")
    args = p.parse_args()

    src = Path(args.source).resolve()
    dest = Path(args.dest).resolve()

    ann_in_path = src / "annotations.json"
    if not ann_in_path.exists():
        raise SystemExit(f"Missing {ann_in_path}")
    ann_in = json.loads(ann_in_path.read_text())["annotations"]
    print(f"Source: {src}")
    print(f"Destination: {dest}")
    print(f"Input annotations: {len(ann_in)} entries\n")

    # Map old path → new path. Skip entries whose crop file is missing.
    remapped = {}
    n_copied = 0
    n_missing = 0
    per_video_counts = {v: 0 for v in VIDEOS}
    per_video_labels = {v: {"X": 0, "with_number": 0} for v in VIDEOS}

    for old_rel, label in ann_in.items():
        # old_rel = "<video>/debug_crops/numbers/<filename>.png"
        parts = Path(old_rel).parts
        if len(parts) < 4 or parts[1] != "debug_crops" or parts[2] != "numbers":
            print(f"WARN: unexpected path layout, skipping: {old_rel}")
            continue
        video, fname = parts[0], parts[-1]
        if video not in VIDEOS:
            print(f"WARN: unknown video {video!r}, skipping: {old_rel}")
            continue
        src_path = src / old_rel
        if not src_path.exists():
            n_missing += 1
            continue
        new_rel = f"crops/{video}/{fname}"
        remapped[new_rel] = label
        per_video_counts[video] += 1
        if label == "X":
            per_video_labels[video]["X"] += 1
        else:
            per_video_labels[video]["with_number"] += 1

        if not args.dry_run:
            new_path = dest / new_rel
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, new_path)
        n_copied += 1

    print(f"\nPer-video counts:")
    for v in VIDEOS:
        ct = per_video_counts[v]
        wn = per_video_labels[v]["with_number"]
        x = per_video_labels[v]["X"]
        print(f"  {v:<10} : {ct:>5} crops ({wn} numbers, {x} X)")

    n_with_num = sum(p["with_number"] for p in per_video_labels.values())
    n_x = sum(p["X"] for p in per_video_labels.values())

    metadata = {
        "n_total": n_copied,
        "n_with_number": n_with_num,
        "n_no_number_X": n_x,
        "n_missing_in_source": n_missing,
        "n_unique_numbers": len({lbl for lbl in remapped.values() if lbl != "X"}),
        "per_video": per_video_counts,
        "produced_by": "tools/build_jersey_dataset.py",
        "source_run": str(src),
    }

    if args.dry_run:
        print(f"\n--dry-run, not writing. Would have copied {n_copied} files.")
        print(f"Metadata that would be written:")
        print(json.dumps(metadata, indent=2))
        return

    dest.mkdir(parents=True, exist_ok=True)
    (dest / "annotations.json").write_text(json.dumps(
        {"annotations": remapped, "metadata": metadata},
        indent=2, sort_keys=True,
    ))
    (dest / "README.md").write_text(README)
    (dest / "LICENSE.md").write_text(LICENSE_PLACEHOLDER)

    print(f"\n✓ Wrote {n_copied} crops + annotations.json + README.md + LICENSE.md")
    print(f"  → {dest}")
    if n_missing:
        print(f"  ({n_missing} annotated entries had no matching crop file)")


if __name__ == "__main__":
    main()
