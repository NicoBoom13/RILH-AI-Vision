"""
RILH-AI-Vision — Koshkina-style contrastive team-embedding fine-tune.

Trains a small CNN that maps a torso crop to an embedding where
same-team crops cluster and other-team crops are pushed apart, using
triplet loss on pseudo-labels bootstrapped from the existing HSV
engine. Critically, **50 % of training triplets are converted to
grayscale** so the network learns jersey shape / pattern rather than
just colour — that's the trick from Koshkina, Pidaparthy, Elder 2021
("Contrastive Learning for Sports Video: Unsupervised Player
Classification") that handles dark-vs-dark teams (e.g. France-Monde)
where pure colour clustering plateaus.

Output: ``models/contrastive_team_rilh.pt`` — state dict + arch meta,
loaded at inference time by Stage 1.b's `--team-engine contrastive`
(see ContrastiveEngine in src/p1_b_teams.py).

Inputs:
  data/jersey_numbers/track_truth.json (optional, preferred)
      Truth labels from tools/annotate_tracks.py; tracks tagged team
      A or B provide clean positives. When present, training uses
      the truth directly — no pseudo-label loop needed.
  data/jersey_numbers/_track_thumbs/<run>/<tid>_<idx>.png
      Per-track thumbnails decoded by the annotator.

When track_truth.json is missing or has < 8 labelled tracks per team,
the script falls back to the Koshkina pseudo-label bootstrap: run the
HSV engine on each clip's existing detections, keep only tracks with
vote_confidence >= 0.85, treat their HSV team_id as the soft label.

Usage:
  python tools/finetune_contrastive_team.py [--epochs 30]
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRUTH = PROJECT_ROOT / "data/jersey_numbers/track_truth.json"
DEFAULT_THUMBS = PROJECT_ROOT / "data/jersey_numbers/_track_thumbs"
DEFAULT_OUTPUT = PROJECT_ROOT / "models/contrastive_team_rilh.pt"

CROP_W, CROP_H = 64, 128   # Koshkina input size (62×128 in paper, rounded
                            # to a power of 2 for cleaner conv arithmetic)
EMB_DIM = 128


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ContrastiveTeamNet(nn.Module):
    """Tiny CNN — 3 conv blocks (16/32/64) + 2 FC. Matches the Koshkina
    2021 architecture; trains in seconds to minutes per epoch on CPU.
    Output is L2-normalised so triplet loss runs in cosine space and
    inference k-means is rotation-invariant."""
    def __init__(self, emb_dim=EMB_DIM):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 2)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 2, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, emb_dim),
        )
    def forward(self, x):
        z = self.head(self.features(x))
        return F.normalize(z, p=2, dim=1)


def preprocess(bgr, grayscale=False):
    """Letterbox-resize a BGR ndarray into (CROP_H, CROP_W) keeping
    aspect, channel-wise affine intensity stretch (Koshkina trick for
    illumination invariance), optional desaturation. Returns a torch
    float32 (3, H, W) tensor."""
    h, w = bgr.shape[:2]
    target_ar = CROP_W / CROP_H
    cur_ar = w / max(h, 1)
    if cur_ar > target_ar:
        new_w, new_h = CROP_W, max(1, int(CROP_W / cur_ar))
    else:
        new_w, new_h = max(1, int(CROP_H * cur_ar)), CROP_H
    resized = cv2.resize(bgr, (new_w, new_h))
    canvas = np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)
    yy, xx = (CROP_H - new_h) // 2, (CROP_W - new_w) // 2
    canvas[yy:yy + new_h, xx:xx + new_w] = resized

    if grayscale:
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    arr = canvas.astype(np.float32)
    # Per-channel intensity stretch — pulls min→0, max→255 so absolute
    # brightness varies less. Falls through if the channel is constant.
    for ci in range(3):
        c = arr[..., ci]
        lo, hi = float(c.min()), float(c.max())
        if hi - lo > 1e-3:
            arr[..., ci] = (c - lo) * (255.0 / (hi - lo))
    arr = arr / 255.0
    arr = arr.transpose(2, 0, 1)   # HWC → CHW
    return torch.from_numpy(arr).float()


class TripletDataset(Dataset):
    """Each item is an (anchor, positive, negative) triplet where all
    three crops come from THE SAME CLIP (intra-clip), so the team-label
    convention (A/B = left/right team in that specific match) stays
    consistent inside the triplet. 50 % of triplets are emitted in
    grayscale (independently per triplet) so the model learns shape /
    pattern cues alongside colour — Koshkina's load-bearing trick.

    by_clip: ``{clip: {"A": [path, ...], "B": [path, ...]}}``."""
    def __init__(self, by_clip, n_triplets=4000, gray_prob=0.5, seed=0):
        self.gray_prob = gray_prob
        self.rng = random.Random(seed)
        self.triplets = []
        clips = list(by_clip.keys())
        if not clips:
            raise SystemExit("Need at least 1 eligible clip with both teams")
        # Sample clips proportional to their pair-count budget
        # (prevents one big clip from monopolising the triplets).
        weights = []
        for c in clips:
            a, b = len(by_clip[c]["A"]), len(by_clip[c]["B"])
            weights.append(min(a, b) * (a + b))
        for _ in range(n_triplets):
            c = self.rng.choices(clips, weights=weights)[0]
            roster = by_clip[c]
            tA = self.rng.choice(["A", "B"])
            tB = "B" if tA == "A" else "A"
            if len(roster[tA]) < 2 or not roster[tB]:
                continue
            a, p = self.rng.sample(roster[tA], 2)
            n = self.rng.choice(roster[tB])
            gray = self.rng.random() < gray_prob
            self.triplets.append((a, p, n, gray))
    def __len__(self):
        return len(self.triplets)
    def __getitem__(self, idx):
        a, p, n, gray = self.triplets[idx]
        return (
            preprocess(cv2.imread(str(a)), grayscale=gray),
            preprocess(cv2.imread(str(p)), grayscale=gray),
            preprocess(cv2.imread(str(n)), grayscale=gray),
        )


def list_thumbs(thumbs_dir, run, tid):
    return sorted((thumbs_dir / run).glob(f"{tid}_*.png"))


def collect_truth_team_crops(truth_path, thumbs_dir):
    """Group thumbnail paths by clip + team. Returns nested dict
    ``{clip: {"A": [path, ...], "B": [path, ...]}}`` — refs and X
    tracks excluded.

    A/B labels are PER-CLIP (annotator's "left vs right team"
    convention), so triplets must stay intra-clip — pairing an A-track
    from run24 (where A is, say, dark blue) with an A-track from run25
    (where A is white) is meaningless. The model should learn
    *intra-clip* team separation; at inference k=2 k-means picks the
    two cluster centroids per clip and the absolute label is
    permutation-invariant. Returns None if too few labelled tracks
    in any single clip to form meaningful triplets."""
    if not truth_path.exists():
        return None
    truth = json.loads(truth_path.read_text())["tracks"]
    by_clip = {}
    track_counts = {}
    for key, meta in truth.items():
        if "/" not in key:
            continue
        team = meta.get("team")
        if team not in ("A", "B"):
            continue
        run, tid_str = key.split("/", 1)
        try:
            tid = int(tid_str)
        except ValueError:
            continue
        thumbs = list_thumbs(thumbs_dir, run, tid)
        if not thumbs:
            continue
        by_clip.setdefault(run, {"A": [], "B": []})
        by_clip[run][team].extend(thumbs)
        track_counts.setdefault(run, {"A": 0, "B": 0})[team] += 1

    # Drop clips that don't have at least 2 tracks per team — can't
    # form a triplet (anchor + positive + negative) without that.
    eligible = {c: by_clip[c] for c, tc in track_counts.items()
                if tc["A"] >= 2 and tc["B"] >= 2}
    if len(eligible) < 1:
        return None
    return eligible


def evaluate_split(model, by_clip, device, n_pairs=200, seed=0):
    """Held-out evaluation, intra-clip: sample n_pairs
    (anchor, positive, negative) triplets all from the same clip,
    embed, check whether anchor-positive distance < anchor-negative.
    Returns the % of triplets satisfied — coarse but honest signal of
    intra-clip team separation in the learned embedding."""
    rng = random.Random(seed)
    clips = list(by_clip.keys())
    if not clips:
        return 0.0
    correct = 0
    for _ in range(n_pairs):
        c = rng.choice(clips)
        roster = by_clip[c]
        tA = rng.choice(["A", "B"])
        tB = "B" if tA == "A" else "A"
        if len(roster[tA]) < 2 or not roster[tB]:
            continue
        a, p = rng.sample(roster[tA], 2)
        n = rng.choice(roster[tB])
        with torch.no_grad():
            za = model(preprocess(cv2.imread(str(a))).unsqueeze(0).to(device))
            zp = model(preprocess(cv2.imread(str(p))).unsqueeze(0).to(device))
            zn = model(preprocess(cv2.imread(str(n))).unsqueeze(0).to(device))
        d_pos = float((za - zp).pow(2).sum())
        d_neg = float((za - zn).pow(2).sum())
        if d_pos < d_neg:
            correct += 1
    return correct / max(n_pairs, 1)


def main():
    p = argparse.ArgumentParser(
        description="Koshkina contrastive team-embedding fine-tune")
    p.add_argument("--truth", type=str, default=str(DEFAULT_TRUTH))
    p.add_argument("--thumbs-dir", type=str, default=str(DEFAULT_THUMBS))
    p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-triplets", type=int, default=4000,
                   help="Triplets per epoch (resampled each epoch).")
    p.add_argument("--gray-prob", type=float, default=0.5,
                   help="Fraction of triplets converted to grayscale "
                        "(Koshkina's load-bearing trick).")
    p.add_argument("--margin", type=float, default=0.5,
                   help="Triplet loss margin (cosine space).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    truth_path = Path(args.truth)
    thumbs_dir = Path(args.thumbs_dir)
    by_clip = collect_truth_team_crops(truth_path, thumbs_dir)
    if by_clip is None:
        raise SystemExit(
            f"Need labelled team crops at {truth_path}.\n"
            f"Run tools/annotate_tracks.py to label at least 2 tracks per team\n"
            f"in at least one clip, then retry.")

    print(f"Truth: {truth_path}")
    for c in sorted(by_clip):
        print(f"  {c}: A={len(by_clip[c]['A'])} crops, "
              f"B={len(by_clip[c]['B'])} crops")

    rng = random.Random(args.seed)
    # Hold out 15 % of crops per team per clip for eval, stratified
    # by team. Triplets stay intra-clip on both train and eval sides.
    train_by_clip, holdout_by_clip = {}, {}
    for clip, roster in by_clip.items():
        train_by_clip[clip] = {"A": [], "B": []}
        holdout_by_clip[clip] = {"A": [], "B": []}
        for team, crops in roster.items():
            crops_shuffled = list(crops)
            rng.shuffle(crops_shuffled)
            n_hold = max(1, int(len(crops_shuffled) * 0.15))
            holdout_by_clip[clip][team] = crops_shuffled[:n_hold]
            train_by_clip[clip][team] = crops_shuffled[n_hold:]
    n_train = sum(len(r["A"]) + len(r["B"]) for r in train_by_clip.values())
    n_hold = sum(len(r["A"]) + len(r["B"]) for r in holdout_by_clip.values())
    print(f"Train: {n_train} crops across {len(train_by_clip)} clips")
    print(f"Held-out: {n_hold} crops")

    device = pick_device()
    print(f"Device: {device}")
    model = ContrastiveTeamNet().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr,
                             weight_decay=1e-4)

    print(f"\nTraining {args.epochs} epochs (lr={args.lr}, "
          f"batch={args.batch_size}, gray_prob={args.gray_prob}, "
          f"margin={args.margin})…")
    eval_log = []
    base_acc = evaluate_split(model, holdout_by_clip, device, seed=args.seed)
    print(f"  baseline (random init) holdout pair-acc = {base_acc:.3f}")
    eval_log.append({"epoch": 0, "holdout_pair_acc": base_acc})

    for epoch in range(1, args.epochs + 1):
        ds = TripletDataset(train_by_clip, n_triplets=args.n_triplets,
                            gray_prob=args.gray_prob, seed=args.seed + epoch)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=0)
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for a, pos, neg in loader:
            a = a.to(device); pos = pos.to(device); neg = neg.to(device)
            za = model(a); zp = model(pos); zn = model(neg)
            d_pos = (za - zp).pow(2).sum(dim=1)
            d_neg = (za - zn).pow(2).sum(dim=1)
            loss = F.relu(d_pos - d_neg + args.margin).mean()
            optim.zero_grad(); loss.backward(); optim.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        avg = epoch_loss / max(n_batches, 1)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            acc = evaluate_split(model, holdout_by_clip, device,
                                 seed=args.seed + epoch)
            print(f"  epoch {epoch:3d}  loss={avg:.4f}  "
                  f"holdout pair-acc={acc:.3f}")
            eval_log.append({"epoch": epoch, "loss": avg,
                             "holdout_pair_acc": acc})

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "arch": {"emb_dim": EMB_DIM, "crop_w": CROP_W, "crop_h": CROP_H},
        "meta": {
            "produced_by": "tools/finetune_contrastive_team.py",
            "trained_at": datetime.datetime.now().isoformat(),
            "epochs": args.epochs,
            "lr": args.lr,
            "n_train_clips": len(train_by_clip),
            "n_train_crops": n_train,
            "n_holdout_crops": n_hold,
            "per_clip_train": {c: {"A": len(r["A"]), "B": len(r["B"])}
                                for c, r in train_by_clip.items()},
            "gray_prob": args.gray_prob,
            "margin": args.margin,
            "eval_log": eval_log,
            "final_holdout_pair_acc": eval_log[-1]["holdout_pair_acc"],
        },
    }
    torch.save(payload, str(out_path))
    print(f"\n✓ Saved {out_path}")
    print(f"  final holdout pair-acc: {eval_log[-1]['holdout_pair_acc']:.3f}")


if __name__ == "__main__":
    main()
