"""
Smoke test for loading the Koshkina PARSeq Hockey checkpoint into the
baudm/parseq architecture, and running predictions on a sample of
user-annotated crops.

Usage:
  python tools/smoke_parseq_hockey.py
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


CKPT = Path("models/parseq_hockey.pt")
ANN = Path("runs/run19/annotations.json")
ROOT = Path("runs/run19")
N_SAMPLES = 200


def load_koshkina(device: str):
    """Load baudm/parseq architecture with Koshkina's hyperparams + weights.

    Discovery: baudm/parseq wraps the actual PARSeq module so all internal
    state-dict keys are prefixed `model.` (e.g. `model.encoder.blocks.0.…`).
    Koshkina's checkpoint stores the same keys without that prefix. Both
    use the same 95-char vocab so the shapes already match — we only need
    to remap the keys."""
    print(f"Loading checkpoint {CKPT}")
    ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    hp = ckpt["hyper_parameters"]
    print(f"  state_dict has {len(sd)} keys, charset_test={len(hp['charset_test'])} chars")

    print("\nBuilding baudm/parseq architecture (random init)…")
    model = torch.hub.load(
        "baudm/parseq", "parseq", pretrained=False, trust_repo=True
    )

    # Remap: prefix every Koshkina key with `model.`
    remapped = {f"model.{k}": v for k, v in sd.items()}
    print(f"\nLoading remapped state_dict (with `model.` prefix)…")
    result = model.load_state_dict(remapped, strict=False)
    n_miss = len(result.missing_keys)
    n_unexpect = len(result.unexpected_keys)
    print(f"  missing keys   : {n_miss}{(' — ' + str(result.missing_keys[:3])) if n_miss else ''}")
    print(f"  unexpected keys: {n_unexpect}{(' — ' + str(result.unexpected_keys[:3])) if n_unexpect else ''}")
    if n_miss == 0 and n_unexpect == 0:
        print("  ✓ all 175 weights loaded cleanly")

    # Tokenizer: keep baudm's built-in. Both baudm and Koshkina use the
    # same 94-char training charset (digits + a-z + A-Z + 32 symbols), so
    # the default tokenizer aligns correctly with Koshkina's head. We
    # filter to digits-only post-decode.
    print(f"  using baudm's built-in tokenizer (charset matches Koshkina)")

    model.eval().to(device)
    print(f"  model on {device}")
    return model, hp


def letterbox_to_aspect(bgr, target_w_over_h, pad_value=0):
    """Same letterbox as in our ParseqOCR — pad to target aspect."""
    h, w = bgr.shape[:2]
    if h <= 0 or w <= 0:
        return bgr
    current = w / h
    if current < target_w_over_h:
        new_w = int(round(h * target_w_over_h))
        pad = new_w - w
        left = pad // 2
        padded = np.full((h, new_w, 3), pad_value, dtype=np.uint8)
        padded[:, left:left + w] = bgr
        return padded
    if current > target_w_over_h:
        new_h = int(round(w / target_w_over_h))
        pad = new_h - h
        top = pad // 2
        padded = np.full((new_h, w, 3), pad_value, dtype=np.uint8)
        padded[top:top + h, :] = bgr
        return padded
    return bgr


def predict_batch(model, device, bgr_crops, letterbox: bool = False):
    """Run PARSeq on a list of BGR crops.

    If `letterbox` is False (default — matches baudm's SceneTextDataModule
    transform that Koshkina also used), crops are resized directly to
    img_size, horizontally stretching square-ish jersey crops to the 4:1
    text-line aspect. The model was trained on stretched-text inputs so
    this is what it expects.
    If True, we letterbox-pad to the target aspect first — useful if the
    crops are already roughly text-shaped (e.g. our older tight band)."""
    if not bgr_crops:
        return []
    img_size = model.hparams.img_size  # (h, w)
    target_aspect = img_size[1] / img_size[0]
    from torchvision import transforms as T
    preprocess = T.Compose([
        T.Resize(img_size, T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ])
    if letterbox:
        crops_in = [letterbox_to_aspect(c, target_aspect) for c in bgr_crops]
    else:
        crops_in = bgr_crops
    pil_imgs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
                for c in crops_in]
    batch = torch.stack([preprocess(im) for im in pil_imgs]).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = logits.softmax(-1)
        preds, confs = model.tokenizer.decode(probs)
    out = []
    for text, conf in zip(preds, confs):
        conf_val = float(conf.mean()) if hasattr(conf, "mean") else float(conf)
        out.append((text, conf_val))
    return out


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    model, hp = load_koshkina(device)

    # Pick N samples that have a real number annotation (not "X")
    ann = json.loads(ANN.read_text())["annotations"]
    samples_with_num = [(p, lbl) for p, lbl in ann.items() if lbl and lbl != "X"]
    print(f"\n{len(samples_with_num)} numbered crops in annotations.json")
    rng = random.Random(42)
    sample = rng.sample(samples_with_num, min(N_SAMPLES, len(samples_with_num)))

    crops = []
    truths = []
    paths = []
    for rel_path, truth in sample:
        f = ROOT / rel_path
        if not f.exists():
            print(f"  WARN: file missing: {f}")
            continue
        crop = cv2.imread(str(f))
        if crop is None:
            print(f"  WARN: cannot read: {f}")
            continue
        crops.append(crop)
        truths.append(truth)
        paths.append(rel_path)

    if not crops:
        print("No crops loaded — exiting")
        return

    print(f"\nRunning PARSeq Hockey on {len(crops)} crops…")
    preds = predict_batch(model, device, crops)

    print(f"\n{'truth':>6} | {'pred':>10} | {'conf':>5} | {'OK':>3} | path")
    print("-" * 100)
    n_ok = 0
    for path, truth, (pred, conf) in zip(paths, truths, preds):
        digits_only = "".join(c for c in pred if c.isdigit())
        ok = digits_only == truth
        n_ok += int(ok)
        print(f"{truth:>6} | {pred:>10} | {conf:.2f}  | {'✓' if ok else '✗':>3} | {path}")
    print(f"\nExact match (digits-only): {n_ok}/{len(crops)} = {100*n_ok/len(crops):.0f}%")


if __name__ == "__main__":
    main()
