"""
RILH-AI-Vision — referee binary classifier fine-tune.

Trains a small MLP head on top of frozen OSNet x0_25 embeddings to
classify each torso crop as referee vs. non-referee. Output:
``models/ref_classifier_rilh.pt`` — a state dict + the matching
architecture description, loaded by Stage 1.b's ``--ref-classifier``
flag (orthogonal to the team engine).

Why this exists: HockeyAI mis-tags roller-hockey referees as `player`
(its training data only has ice-hockey ref uniforms). They leak into
both team clusters and pollute downstream entity stats. Koshkina 2021
recommends this exact split — supervised binary for refs (uniforms
are league-stable, not video-specific), unsupervised k-means for teams
(team colours change every clip).

Why OSNet + MLP and not end-to-end fine-tune: we already pay the
OSNet inference cost in Stage 1.b's osnet/contrastive engines; piggy-
backing a 512→64→1 MLP on the same embeddings adds ~negligible compute
at inference. The MLP trains in seconds on CPU.

Inputs:
  data/jersey_numbers/track_truth.json  — track-level labels from
      tools/annotate_tracks.py. Tracks with `is_referee=True` provide
      positives; tracks with `is_referee=False` provide negatives
      (both team and X tracks).
  data/jersey_numbers/_track_thumbs/<run>/<tid>_<idx>.png — per-track
      thumbnails decoded by the annotator.

Output:
  models/ref_classifier_rilh.pt — torch state dict + meta (arch sizes
      + held-out accuracy / precision / recall on 20 % test split).

Usage:
  python tools/finetune_ref_classifier.py [--epochs 30] [--lr 1e-3]

Eval is automatic on a 20 % stratified test split. The summary is
printed and saved alongside the checkpoint.
"""

import argparse
import datetime
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRUTH = PROJECT_ROOT / "data/jersey_numbers/track_truth.json"
DEFAULT_THUMBS = PROJECT_ROOT / "data/jersey_numbers/_track_thumbs"
DEFAULT_OUTPUT = PROJECT_ROOT / "models/ref_classifier_rilh.pt"
EMB_DIM = 512   # OSNet x0_25 output dimension
HIDDEN = 64


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def list_thumbs(thumbs_dir: Path, run: str, tid: int):
    """Return all decoded thumbnail paths for one (run, tid)."""
    return sorted((thumbs_dir / run).glob(f"{tid}_*.png"))


def build_dataset(truth_path: Path, thumbs_dir: Path):
    """Walk track_truth.json, return a list of (image_path, label,
    track_key) records — one per thumbnail. Tracks tagged 'X' (not a
    player) are excluded from negatives because they're a noisy
    superset of refs and would bias the negative class."""
    truth = json.loads(truth_path.read_text())["tracks"]
    records = []
    for key, meta in truth.items():
        if "/" not in key:
            continue
        run, tid_str = key.split("/", 1)
        try:
            tid = int(tid_str)
        except ValueError:
            continue
        # Positives: explicit refs. Negatives: confirmed players (team A/B).
        if meta.get("is_referee"):
            label = 1
        elif meta.get("team") in ("A", "B"):
            label = 0
        else:
            continue   # "X" or unlabelled — skip
        for thumb in list_thumbs(thumbs_dir, run, tid):
            records.append((thumb, label, key))
    return records


def stratified_split(records, test_frac=0.2, seed=0):
    """Group by track_key so that all crops of one track land in the
    same split (no leakage), stratified by label. Returns (train, test)
    lists of records."""
    by_track = defaultdict(list)
    track_label = {}
    for thumb, label, key in records:
        by_track[key].append((thumb, label, key))
        track_label[key] = label

    rng = random.Random(seed)
    pos_keys = [k for k, l in track_label.items() if l == 1]
    neg_keys = [k for k, l in track_label.items() if l == 0]
    rng.shuffle(pos_keys)
    rng.shuffle(neg_keys)
    n_pos_test = max(1, int(len(pos_keys) * test_frac))
    n_neg_test = max(1, int(len(neg_keys) * test_frac))
    test_keys = set(pos_keys[:n_pos_test] + neg_keys[:n_neg_test])
    train, test = [], []
    for k, recs in by_track.items():
        (test if k in test_keys else train).extend(recs)
    return train, test


class CropDataset(Dataset):
    """Lazy-decodes thumbnails to BGR ndarrays. The OSNet feature
    extractor accepts BGR ndarrays directly (torchreid convention)."""
    def __init__(self, records):
        self.records = records
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        thumb, label, _ = self.records[idx]
        img = cv2.imread(str(thumb))
        if img is None:
            raise RuntimeError(f"Failed to decode {thumb}")
        return img, label


def extract_embeddings(records, extractor, batch_size=64):
    """Run OSNet over every crop once; return a (N, 512) array of
    L2-normalised embeddings + a parallel (N,) label array. Cached in
    memory because the dataset is small enough."""
    feats, labels = [], []
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        crops = [cv2.imread(str(r[0])) for r in batch]
        with torch.no_grad():
            f = extractor(crops).cpu().numpy()
        feats.append(f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9))
        labels.extend(int(r[1]) for r in batch)
    feats = np.vstack(feats) if feats else np.zeros((0, EMB_DIM), dtype=np.float32)
    return feats.astype(np.float32), np.array(labels, dtype=np.int64)


class RefHead(nn.Module):
    """512 → 64 → 1 MLP, sigmoid at inference. Small enough to train in
    seconds on CPU; the heavy lifting is OSNet, run once per crop."""
    def __init__(self, emb_dim=EMB_DIM, hidden=HIDDEN):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(0.2)
    def forward(self, x):
        return self.fc2(self.drop(torch.relu(self.fc1(x)))).squeeze(-1)


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        pred = (torch.sigmoid(logits) > 0.5).numpy().astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    acc = (tp + tn) / max(len(y), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"accuracy": acc, "precision": prec, "recall": rec,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    p = argparse.ArgumentParser(description="Fine-tune ref-binary head on OSNet embeddings")
    p.add_argument("--truth", type=str, default=str(DEFAULT_TRUTH))
    p.add_argument("--thumbs-dir", type=str, default=str(DEFAULT_THUMBS))
    p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    truth_path = Path(args.truth)
    thumbs_dir = Path(args.thumbs_dir)
    if not truth_path.exists():
        raise SystemExit(f"Truth file not found: {truth_path}\n"
                         f"Run tools/annotate_tracks.py first.")

    print(f"Truth: {truth_path}")
    records = build_dataset(truth_path, thumbs_dir)
    print(f"Crops with usable label: {len(records)}")
    n_pos = sum(1 for r in records if r[1] == 1)
    print(f"  positives (referee): {n_pos}")
    print(f"  negatives (player):  {len(records) - n_pos}")
    if n_pos < 5 or (len(records) - n_pos) < 5:
        raise SystemExit("Not enough labels yet — annotate more tracks first.")

    train, test = stratified_split(records, test_frac=args.test_frac,
                                   seed=args.seed)
    print(f"Split: train={len(train)} crops, test={len(test)} crops")

    device = pick_device()
    print(f"Device: {device}")
    print("Loading OSNet x0_25 (frozen — embeddings only)...")
    from torchreid.reid.utils import FeatureExtractor
    extractor = FeatureExtractor(
        model_name="osnet_x0_25", model_path="", device=device,
    )

    print("Extracting embeddings…")
    X_train, y_train = extract_embeddings(train, extractor)
    X_test, y_test = extract_embeddings(test, extractor)
    print(f"  train: {X_train.shape}, test: {X_test.shape}")

    model = RefHead()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr,
                             weight_decay=1e-4)
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) /
                                max(y_train.sum(), 1)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    rng = np.random.default_rng(args.seed)
    print(f"\nTraining {args.epochs} epochs (lr={args.lr}, "
          f"batch={args.batch_size}, pos_weight={pos_weight.item():.2f})…")
    for epoch in range(1, args.epochs + 1):
        model.train()
        idx = rng.permutation(len(X_train))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(idx), args.batch_size):
            sl = idx[i:i + args.batch_size]
            xb = torch.from_numpy(X_train[sl])
            yb = torch.from_numpy(y_train[sl]).float()
            optim.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            train_m = evaluate(model, X_train, y_train)
            test_m = evaluate(model, X_test, y_test)
            print(f"  epoch {epoch:3d}  loss={epoch_loss / max(n_batches,1):.4f}  "
                  f"train acc={train_m['accuracy']:.3f} "
                  f"prec={train_m['precision']:.3f} rec={train_m['recall']:.3f} | "
                  f"test acc={test_m['accuracy']:.3f} "
                  f"prec={test_m['precision']:.3f} rec={test_m['recall']:.3f}")

    final_test = evaluate(model, X_test, y_test)
    final_train = evaluate(model, X_train, y_train)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "arch": {"emb_dim": EMB_DIM, "hidden": HIDDEN},
        "meta": {
            "produced_by": "tools/finetune_ref_classifier.py",
            "trained_at": datetime.datetime.now().isoformat(),
            "epochs": args.epochs,
            "lr": args.lr,
            "n_train_crops": int(len(X_train)),
            "n_test_crops": int(len(X_test)),
            "n_train_pos": int(y_train.sum()),
            "n_test_pos": int(y_test.sum()),
            "test_metrics": final_test,
            "train_metrics": final_train,
        },
    }
    torch.save(payload, str(out_path))
    print(f"\n✓ Saved {out_path}")
    print(f"  test accuracy:  {final_test['accuracy']:.3f}")
    print(f"  test precision: {final_test['precision']:.3f}")
    print(f"  test recall:    {final_test['recall']:.3f}")
    print(f"  confusion: tp={final_test['tp']} fp={final_test['fp']} "
          f"fn={final_test['fn']} tn={final_test['tn']}")


if __name__ == "__main__":
    main()
