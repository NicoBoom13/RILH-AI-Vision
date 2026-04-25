"""
RILH-AI-Vision — build train/val/test splits from the consolidated
jersey-number dataset.

Reads:
  data/jersey_numbers/annotations.json

Writes:
  data/jersey_numbers/splits/train.json
  data/jersey_numbers/splits/val.json
  data/jersey_numbers/splits/test.json

Each split JSON: list of {"path": "crops/.../<file>.png", "label": "12"}.
"X" labels (no readable number) are kept as empty-string targets so the
fine-tuned model learns to output EOS immediately for unreadable crops
(prevents hallucination at inference time on bad crops).

Strategy:
- X negatives are subsampled to match positive count (1:1 balance).
- Per-label stratified 80/10/10 split:
    * labels with >= 10 samples: stratified
    * labels with < 10 samples: all in train (rare labels need max exposure)
- Random seed for reproducibility (--seed, default 42).
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


DEFAULT_DATASET = Path("data/jersey_numbers")
TRAIN_FRAC, VAL_FRAC = 0.80, 0.10  # test = remainder
RARE_LABEL_MIN = 10                # below this, send all to train


def main():
    p = argparse.ArgumentParser(description="Build train/val/test splits")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--neg-ratio", type=float, default=1.0,
                   help="Number of X negatives per positive sample. "
                        "1.0 = balanced; 2.0 = twice as many negatives; "
                        "0 = no negatives in training set.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    rng = random.Random(args.seed)

    root = Path(args.dataset)
    ann = json.loads((root / "annotations.json").read_text())["annotations"]

    # Group by label
    by_label = defaultdict(list)
    for path, label in ann.items():
        if not label:
            continue
        by_label[label].append(path)
    print(f"Loaded {sum(len(v) for v in by_label.values())} annotated entries, "
          f"{len(by_label)} unique labels")

    # Separate positives + negatives
    pos_labels = [l for l in by_label if l != "X"]
    n_pos = sum(len(by_label[l]) for l in pos_labels)
    n_neg = len(by_label.get("X", []))
    print(f"  Positives (numbered): {n_pos} across {len(pos_labels)} labels")
    print(f"  Negatives (X)       : {n_neg}")

    # Subsample negatives to N pos × neg_ratio
    target_negs = int(round(n_pos * args.neg_ratio))
    if "X" in by_label and target_negs < len(by_label["X"]):
        rng.shuffle(by_label["X"])
        by_label["X"] = by_label["X"][:target_negs]
        print(f"  Subsampled X to {len(by_label['X'])} (neg_ratio={args.neg_ratio})")

    # Per-label stratified split
    splits = {"train": [], "val": [], "test": []}
    for label, paths in by_label.items():
        rng.shuffle(paths)
        n = len(paths)
        if n < RARE_LABEL_MIN:
            # Send all rare-label samples to training
            splits["train"].extend((p, label) for p in paths)
            continue
        n_train = int(round(n * TRAIN_FRAC))
        n_val = int(round(n * VAL_FRAC))
        # Ensure at least 1 in val + test
        n_val = max(1, n_val)
        n_test = max(1, n - n_train - n_val)
        n_train = n - n_val - n_test
        splits["train"].extend((p, label) for p in paths[:n_train])
        splits["val"].extend((p, label) for p in paths[n_train:n_train + n_val])
        splits["test"].extend((p, label) for p in paths[n_train + n_val:])

    # Shuffle each split (so labels aren't grouped)
    for split in splits.values():
        rng.shuffle(split)

    # Reporting
    print(f"\nSplit sizes:")
    for name, items in splits.items():
        n_pos_split = sum(1 for _, lbl in items if lbl != "X")
        n_neg_split = sum(1 for _, lbl in items if lbl == "X")
        print(f"  {name:<6}: {len(items):>5} ({n_pos_split} numbers, {n_neg_split} X)")

    # Per-label distribution check (val + test should have all "common" labels)
    common_labels = [l for l in pos_labels if len(by_label[l]) >= RARE_LABEL_MIN]
    rare_labels = [l for l in pos_labels if len(by_label[l]) < RARE_LABEL_MIN]
    print(f"\nLabels with ≥{RARE_LABEL_MIN} samples (stratified): {len(common_labels)}")
    print(f"Labels with <{RARE_LABEL_MIN} samples (all → train) : {len(rare_labels)}")
    if rare_labels:
        print(f"  rare: {sorted(rare_labels)}")

    if args.dry_run:
        print("\n--dry-run, not writing")
        return

    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        out = [{"path": p, "label": lbl} for p, lbl in items]
        (splits_dir / f"{name}.json").write_text(json.dumps(out, indent=2))
    # Manifest with seed + ratio for reproducibility
    (splits_dir / "manifest.json").write_text(json.dumps({
        "seed": args.seed,
        "neg_ratio": args.neg_ratio,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "rare_label_min": RARE_LABEL_MIN,
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
    }, indent=2))
    print(f"\n✓ Wrote splits/ to {splits_dir}")


if __name__ == "__main__":
    main()
