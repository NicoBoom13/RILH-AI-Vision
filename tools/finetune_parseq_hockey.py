"""
RILH-AI-Vision — fine-tune Maria Koshkina's PARSeq Hockey checkpoint on
the user's manually-annotated RILH jersey crops.

Inputs:
  models/parseq_hockey.pt              — starting weights (Koshkina, NC license)
  data/jersey_numbers/                  — license-clean dataset
    crops/<video>/*.png
    splits/{train,val,test}.json

Outputs:
  models/parseq_hockey_rilh.pt         — best model by val exact-match
  runs/finetune_<timestamp>/
    train_log.json                     — per-epoch loss + accuracy
    eval_test.json                     — held-out test set predictions
    summary.txt                        — human-readable comparison

Usage:
  python tools/finetune_parseq_hockey.py [--epochs 20] [--batch-size 16] [--lr 5e-5]
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Make strhub importable from the cached torch.hub clone
HUB = Path("/Users/nico/.cache/torch/hub/baudm_parseq_main")
sys.path.insert(0, str(HUB))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision import transforms as T  # noqa: E402

from strhub.models.parseq.system import PARSeq  # noqa: E402


CKPT_BASE = Path("models/parseq_hockey.pt")
DATASET_ROOT = Path("data/jersey_numbers")
SPLITS_DIR = DATASET_ROOT / "splits"
OUT_MODEL = Path("models/parseq_hockey_rilh.pt")


# ----- Dataset --------------------------------------------------------------

class JerseyDataset(Dataset):
    """Load split JSON entries → (image_tensor, label_string)."""

    def __init__(self, split_json: Path, dataset_root: Path, img_size,
                 augment: bool = False):
        """Load a split JSON file (list of {path, label}) and prepare
        the BICUBIC + normalise transform that matches Koshkina's
        training pipeline (direct stretch resize, no letterbox)."""
        self.entries = json.loads(split_json.read_text())
        self.root = dataset_root
        self.augment = augment
        # Match baudm/parseq's SceneTextDataModule transform exactly: direct
        # bicubic resize, no letterbox. Koshkina trained on this.
        self.preprocess = T.Compose([
            T.Resize(img_size, T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])

    def __len__(self):
        """Number of entries in this split."""
        return len(self.entries)

    def __getitem__(self, idx):
        """Return one (image_tensor, label_string) pair. ``X`` labels
        are mapped to the empty string so the model learns to predict
        EOS immediately on unreadable crops."""
        e = self.entries[idx]
        img = Image.open(self.root / e["path"]).convert("RGB")
        if self.augment:
            img = self._augment(img)
        img = self.preprocess(img)
        label = "" if e["label"] == "X" else e["label"]
        return img, label

    @staticmethod
    def _augment(pil):
        """Apply mild brightness / contrast jitter (±20 %). No rotation
        — jersey numbers are upright in the source frames."""
        from torchvision.transforms.functional import adjust_brightness, adjust_contrast
        import random
        if random.random() < 0.5:
            pil = adjust_brightness(pil, 1 + (random.random() - 0.5) * 0.4)
        if random.random() < 0.5:
            pil = adjust_contrast(pil, 1 + (random.random() - 0.5) * 0.4)
        return pil


def collate_fn(batch):
    """Stack image tensors and pass labels through as a list (PARSeq's
    tokenizer encodes them inside training_step, so they stay strings)."""
    images = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    return images, labels


# ----- Model loading --------------------------------------------------------

def load_pretrained(device: str) -> PARSeq:
    """Build PARSeq with Koshkina's hyperparameters and load her weights."""
    ckpt = torch.load(str(CKPT_BASE), map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    hp = ckpt["hyper_parameters"]

    # Drop hp keys not accepted by PARSeq.__init__ (e.g. 'name')
    init_kwargs = {k: v for k, v in hp.items() if k != "name"}
    model = PARSeq(**init_kwargs)

    # baudm's PARSeq wraps the actual encoder/decoder/head modules under
    # `self.model.*`, so the state_dict has keys like `model.encoder.*`.
    # Koshkina's vendored fork stored weights at the top level (`encoder.*`),
    # so we add the `model.` prefix when loading.
    remapped = {f"model.{k}": v for k, v in sd.items()}
    result = model.load_state_dict(remapped, strict=True)
    print(f"  loaded Koshkina weights: missing={len(result.missing_keys)}, "
          f"unexpected={len(result.unexpected_keys)}")
    model._device = torch.device(device)
    model.to(device)
    return model


# ----- Eval -----------------------------------------------------------------

@torch.no_grad()
def evaluate(model: PARSeq, loader: DataLoader, device: str):
    """Compute exact-match accuracy + per-bucket stats."""
    model.eval()
    n_total = n_correct = 0
    n_total_pos = n_correct_pos = 0
    n_total_neg = n_correct_neg = 0
    samples = []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = logits.softmax(-1)
        preds, _ = model.tokenizer.decode(probs)
        for pred, truth in zip(preds, labels):
            digits = "".join(c for c in pred if c.isdigit())
            ok = digits == truth
            n_total += 1
            n_correct += int(ok)
            if truth == "":
                n_total_neg += 1
                n_correct_neg += int(digits == "")
            else:
                n_total_pos += 1
                n_correct_pos += int(ok)
            if len(samples) < 30:
                samples.append({
                    "truth": truth or "X",
                    "pred_raw": pred,
                    "pred_digits": digits,
                    "ok": ok,
                })
    return {
        "n_total": n_total,
        "exact_match": n_correct / max(n_total, 1),
        "exact_match_positives": n_correct_pos / max(n_total_pos, 1),
        "exact_match_negatives_X": n_correct_neg / max(n_total_neg, 1),
        "n_pos": n_total_pos,
        "n_neg": n_total_neg,
        "samples": samples,
    }


# ----- Train ----------------------------------------------------------------

def train(args):
    """Run the full fine-tune loop end-to-end: load Koshkina baseline,
    train on train.json with eval on val.json each epoch, save the
    best model by val exact-match, then evaluate baseline vs fine-tune
    on the held-out test set and write the comparison summary."""
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading Koshkina checkpoint: {CKPT_BASE}")
    model = load_pretrained(device)

    img_size = tuple(model.hparams.img_size)
    print(f"Image size: {img_size}")

    train_ds = JerseyDataset(SPLITS_DIR / "train.json", DATASET_ROOT,
                             img_size, augment=True)
    val_ds = JerseyDataset(SPLITS_DIR / "val.json", DATASET_ROOT,
                           img_size, augment=False)
    test_ds = JerseyDataset(SPLITS_DIR / "test.json", DATASET_ROOT,
                            img_size, augment=False)
    print(f"Splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, collate_fn=collate_fn)

    # Mock Lightning's `self.log` (training_step calls it)
    model.log = lambda *a, **kw: None

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs * len(train_loader), eta_min=args.lr * 0.1)

    # Eval baseline (epoch 0)
    print(f"\nBaseline (Koshkina without RILH fine-tune):")
    base_val = evaluate(model, val_loader, device)
    print(f"  val: exact_match={base_val['exact_match']:.3f} "
          f"(pos={base_val['exact_match_positives']:.3f}, "
          f"neg={base_val['exact_match_negatives_X']:.3f})")

    best_acc = base_val["exact_match"]
    best_epoch = 0
    log = {
        "args": vars(args),
        "img_size": list(img_size),
        "device": device,
        "epochs": [{"epoch": 0, "val": base_val, "loss": None}],
    }
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"runs/finetune_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFine-tuning {args.epochs} epochs (lr={args.lr}, batch={args.batch_size})…")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for batch in train_loader:
            images, labels = batch
            images = images.to(device)
            optim.zero_grad()
            loss = model.training_step((images, labels), 0)
            loss.backward()
            optim.step()
            sched.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        val_metrics = evaluate(model, val_loader, device)
        dt = time.time() - t0
        print(f"  epoch {epoch:>2d}: loss={avg_loss:.4f}  "
              f"val_acc={val_metrics['exact_match']:.3f} "
              f"(pos={val_metrics['exact_match_positives']:.3f}, "
              f"neg={val_metrics['exact_match_negatives_X']:.3f})  "
              f"[{dt:.0f}s]")
        log["epochs"].append({
            "epoch": epoch, "loss": avg_loss, "val": val_metrics, "wall_s": dt,
        })

        if val_metrics["exact_match"] > best_acc:
            best_acc = val_metrics["exact_match"]
            best_epoch = epoch
            # Save in our compact format (state_dict only, no Lightning wrapper)
            torch.save({
                "state_dict": model.state_dict(),
                "hyper_parameters": dict(model.hparams),
                "best_val_accuracy": best_acc,
                "epoch": epoch,
                "source_checkpoint": str(CKPT_BASE),
            }, str(OUT_MODEL))
            print(f"    ✓ saved {OUT_MODEL} (val_acc {best_acc:.3f})")

    # Final eval on test set with the best model
    print(f"\nLoading best model (epoch {best_epoch}, val_acc {best_acc:.3f}) "
          f"for test eval…")
    best_sd = torch.load(str(OUT_MODEL), map_location=device, weights_only=False)
    model.load_state_dict(best_sd["state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    print(f"  test: exact_match={test_metrics['exact_match']:.3f} "
          f"(pos={test_metrics['exact_match_positives']:.3f}, "
          f"neg={test_metrics['exact_match_negatives_X']:.3f})")

    log["best_epoch"] = best_epoch
    log["best_val_accuracy"] = best_acc
    log["test_metrics"] = test_metrics
    (run_dir / "train_log.json").write_text(json.dumps(log, indent=2, default=str))

    # Comparison: baseline (Koshkina only) vs fine-tuned, on the test set
    print(f"\nLoading baseline Koshkina for test comparison…")
    baseline = load_pretrained(device)
    baseline_test = evaluate(baseline, test_loader, device)
    summary = (
        f"=== Fine-tune Koshkina → RILH summary ===\n"
        f"epochs              : {args.epochs}\n"
        f"best epoch          : {best_epoch} (val_acc {best_acc:.3f})\n"
        f"\n"
        f"On held-out test set ({test_metrics['n_total']} crops: "
        f"{test_metrics['n_pos']} pos + {test_metrics['n_neg']} X):\n"
        f"  Koshkina (baseline)         : exact_match {baseline_test['exact_match']:.3f}  "
        f"(pos {baseline_test['exact_match_positives']:.3f}, "
        f"neg {baseline_test['exact_match_negatives_X']:.3f})\n"
        f"  Koshkina + RILH fine-tune   : exact_match {test_metrics['exact_match']:.3f}  "
        f"(pos {test_metrics['exact_match_positives']:.3f}, "
        f"neg {test_metrics['exact_match_negatives_X']:.3f})\n"
        f"\n"
        f"  Δ exact_match overall  : {test_metrics['exact_match'] - baseline_test['exact_match']:+.3f}\n"
        f"  Δ exact_match positives: {test_metrics['exact_match_positives'] - baseline_test['exact_match_positives']:+.3f}\n"
        f"  Δ exact_match X (neg)  : {test_metrics['exact_match_negatives_X'] - baseline_test['exact_match_negatives_X']:+.3f}\n"
    )
    (run_dir / "summary.txt").write_text(summary)
    (run_dir / "eval_test.json").write_text(json.dumps({
        "baseline_koshkina": baseline_test,
        "finetuned_rilh": test_metrics,
    }, indent=2, default=str))
    print(f"\n{summary}")
    print(f"All artifacts → {run_dir}")


def main():
    """CLI entry point — parse arguments and dispatch to ``train``."""
    p = argparse.ArgumentParser(description="Fine-tune PARSeq Hockey on RILH crops")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
