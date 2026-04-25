"""
RILH-AI-Vision — p1_c_numbers
Per-track jersey-number OCR via PARSeq.

Inputs : p1_detections.json (p1_a_detect) + the source video.
Output : p1_numbers.json — per player-track jersey number + confidence,
         plus optional track-merge groups for players whose track id was
         broken and re-issued by the tracker.

Pipeline (per track id, not per frame — keeps inference tractable):
  1) sample the N highest-confidence detections of that track
  2) run YOLO pose on each sample's full frame, match by IoU to the bbox
  3) classify orientation from pose keypoints (nose / ears / shoulders)
  4) on back-facing samples, crop the dorsal jersey region Koshkina-style
     (bbox of LSHO, RSHO, LHIP, RHIP + 5 px padding)
  5) run PARSeq (default baudm pretrained, or our fine-tuned hockey
     checkpoint via --parseq-checkpoint), keep digits only (1-2 chars)
  6) vote per track (≥ 2 agreeing votes required); merge tracks sharing
     the same number with non-overlapping time spans (same player, but
     the tracker re-issued the id)
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# In our remapped class space, every "person" detection (skater +
# goaltender after HockeyAI's classes are collapsed in P1.a) is class 0.
PERSON_CLASS = 0

# COCO-17 keypoint indices used by every Ultralytics pose model.
KP_NOSE, KP_LEYE, KP_REYE, KP_LEAR, KP_REAR = 0, 1, 2, 3, 4
KP_LSHO, KP_RSHO = 5, 6
KP_LHIP, KP_RHIP = 11, 12

MODELS_DIR = Path("models")


def pick_device():
    """Return the best PyTorch device available (mps > cuda > cpu)."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_yolo_path(name: str) -> Path:
    """Route a bare YOLO filename into ``models/`` for Ultralytics
    auto-download. Paths with a directory component are kept as-is."""
    p = Path(name)
    if len(p.parts) == 1:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return MODELS_DIR / p.name
    return p


def safe_crop(frame, xyxy):
    """Clamp xyxy to the frame and return the BGR sub-image, or None
    if the resulting region is too small (≤ 4 px on either dimension)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))
    if x2 <= x1 + 4 or y2 <= y1 + 4:
        return None
    return frame[y1:y2, x1:x2]


def orientation_from_kps(kp_xy, kp_conf, thr=0.3):
    """Classify body orientation from COCO-17 keypoints.

    Returns one of: 'front' (nose+eyes visible), 'back' (ears visible
    without nose), 'side' (only shoulders confidently visible), or
    'unknown' (shoulders missing). Used to gate which samples get the
    dorsal jersey OCR — only back-facing samples are useful."""
    nose = kp_conf[KP_NOSE] > thr
    eyes = (kp_conf[KP_LEYE] > thr) or (kp_conf[KP_REYE] > thr)
    ears = (kp_conf[KP_LEAR] > thr) or (kp_conf[KP_REAR] > thr)
    shos = (kp_conf[KP_LSHO] > thr) and (kp_conf[KP_RSHO] > thr)
    if not shos:
        return "unknown"
    if nose and eyes:
        return "front"
    if ears and not nose:
        return "back"
    return "side"


def torso_back_crop(frame, kp_xy, kp_conf, thr=0.3, padding_px=5):
    """Crop the dorsal jersey region — Koshkina-style (matches the training
    distribution of `parseq_hockey.pt`).

    Bounding box from the four torso keypoints (shoulders + hips):
      x_min/x_max = leftmost / rightmost of {LSHO, RSHO, LHIP, RHIP}
      y_min       = topmost  (= top of shoulders)
      y_max       = bottom   (= hip line, no extra padding)
    Then `padding_px` (default 5 px, matches Koshkina's PADDING constant)
    is added on x_min, x_max, and y_min — but NOT y_max so the crop ends
    cleanly at the hips.

    Produces roughly square crops covering the FULL shoulder-to-hip
    torso. Matches the data PARSeq Hockey was trained on, so off-the-
    shelf inference no longer faces a distribution shift."""
    shos_ok = kp_conf[KP_LSHO] > thr and kp_conf[KP_RSHO] > thr
    hips_ok = kp_conf[KP_LHIP] > thr and kp_conf[KP_RHIP] > thr
    if not shos_ok:
        return None
    pts = [(kp_xy[KP_LSHO, 0], kp_xy[KP_LSHO, 1]),
           (kp_xy[KP_RSHO, 0], kp_xy[KP_RSHO, 1])]
    if hips_ok:
        pts.append((kp_xy[KP_LHIP, 0], kp_xy[KP_LHIP, 1]))
        pts.append((kp_xy[KP_RHIP, 0], kp_xy[KP_RHIP, 1]))
    else:
        # No hip keypoints — fall back to a 2.2× shoulder-width vertical span.
        sho_w = abs(kp_xy[KP_LSHO, 0] - kp_xy[KP_RSHO, 0])
        sho_cy = (kp_xy[KP_LSHO, 1] + kp_xy[KP_RSHO, 1]) / 2.0
        pts.append((kp_xy[KP_LSHO, 0], sho_cy + 2.2 * sho_w))
        pts.append((kp_xy[KP_RSHO, 0], sho_cy + 2.2 * sho_w))
    x_min = min(p[0] for p in pts) - padding_px
    x_max = max(p[0] for p in pts) + padding_px
    y_min = min(p[1] for p in pts) - padding_px
    y_max = max(p[1] for p in pts)
    return safe_crop(frame, [x_min, y_min, x_max, y_max])


def ious(xyxy, boxes):
    """Vectorised IoU between one xyxy box and an (N, 4) array of boxes."""
    bx1 = np.maximum(xyxy[0], boxes[:, 0])
    by1 = np.maximum(xyxy[1], boxes[:, 1])
    bx2 = np.minimum(xyxy[2], boxes[:, 2])
    by2 = np.minimum(xyxy[3], boxes[:, 3])
    iw = np.clip(bx2 - bx1, 0, None)
    ih = np.clip(by2 - by1, 0, None)
    inter = iw * ih
    a0 = max(0.0, (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
    a1 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a0 + a1 - inter + 1e-6)


class ParseqOCR:
    """PARSeq scene-text recognition for jersey-number digits.

    Two modes:
      - Default (`checkpoint_path=None`): baudm/parseq pretrained on
        generic STR. Crops are letterbox-padded to the model's 4:1
        aspect before resize so generic STR doesn't see horizontally
        stretched digits.
      - Hockey (`checkpoint_path=models/parseq_hockey_rilh.pt` or
        `models/parseq_hockey.pt`): PARSeq fine-tuned on hockey jersey
        numbers (Koshkina + our RILH fine-tune). Same baudm
        architecture, weights remapped via the `model.` prefix.
        Trained with DIRECT resize, so we skip the letterbox to match
        the training distribution.

    Returns digit-only OCR results; non-digit characters are stripped
    by the caller's `_filter_number`."""

    def __init__(self, device, checkpoint_path: Path | None = None):
        """Build the OCR engine.

        Args:
            device: torch device string (e.g. 'mps', 'cuda', 'cpu').
            checkpoint_path: optional path to a custom Lightning .pt
                checkpoint. None → load baudm/parseq pretrained on STR.
                Otherwise load the custom weights via ``_load_external_checkpoint``.
        """
        self.device = device
        self.use_letterbox = True  # baudm default behaviour
        if checkpoint_path is None:
            print("Loading PARSeq baudm pretrained (STR generic)…")
            self.model = torch.hub.load(
                "baudm/parseq", "parseq", pretrained=True, trust_repo=True
            ).eval().to(device)
        else:
            print(f"Loading PARSeq from checkpoint: {checkpoint_path}")
            self.model = self._load_external_checkpoint(checkpoint_path).to(device)
            # Both Koshkina and our fine-tune used direct stretch resize.
            self.use_letterbox = False
            print("  (direct-resize mode: no letterbox padding before resize)")
        self.img_size = self.model.hparams.img_size  # (h, w), usually (32, 128)
        from torchvision import transforms as T
        self.preprocess = T.Compose([
            T.Resize(self.img_size, T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])
        self.target_w_over_h = float(self.img_size[1]) / float(self.img_size[0])

    @staticmethod
    def _load_external_checkpoint(checkpoint_path: Path):
        """Load a Lightning-format PARSeq checkpoint into the baudm/parseq
        architecture. Two checkpoint flavours are supported, distinguished
        by the first state-dict key:
          - Koshkina (vendored fork): keys at top level (`encoder.*`,
            `head.*`). We add the `model.` prefix to match baudm's
            wrapper module layout.
          - Our fine-tuned model (saved through baudm-wrapped PARSeq):
            keys ALREADY have the `model.` prefix. Loaded as-is."""
        ckpt = torch.load(str(checkpoint_path), map_location="cpu",
                          weights_only=False)
        if "state_dict" not in ckpt:
            raise SystemExit(f"{checkpoint_path}: no 'state_dict' key — "
                             f"unexpected checkpoint format")
        sd = ckpt["state_dict"]
        model = torch.hub.load(
            "baudm/parseq", "parseq", pretrained=False, trust_repo=True
        ).eval()
        first_key = next(iter(sd.keys()))
        if first_key.startswith("model."):
            sd_to_load = sd
        else:
            sd_to_load = {f"model.{k}": v for k, v in sd.items()}
        result = model.load_state_dict(sd_to_load, strict=False)
        if result.missing_keys or result.unexpected_keys:
            print(f"  load report: {len(result.missing_keys)} missing, "
                  f"{len(result.unexpected_keys)} unexpected")
            if result.missing_keys:
                print(f"  missing[:3]: {result.missing_keys[:3]}")
            if result.unexpected_keys:
                print(f"  unexpected[:3]: {result.unexpected_keys[:3]}")
        else:
            print(f"  ✓ all {len(sd_to_load)} weights loaded cleanly")
        return model

    @staticmethod
    def _letterbox_to_aspect(bgr, target_w_over_h, pad_value=0):
        """Pad crop with `pad_value` so w/h == target aspect, without
        changing the content region."""
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

    @torch.no_grad()
    def read_batch(self, bgr_crops):
        """Return list of (raw_text, confidence) per crop. Caller
        applies the digit filter."""
        if not bgr_crops:
            return []
        if self.use_letterbox:
            crops_in = [self._letterbox_to_aspect(c, self.target_w_over_h)
                        for c in bgr_crops]
        else:
            crops_in = bgr_crops
        pil_imgs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
                    for c in crops_in]
        batch = torch.stack([self.preprocess(im) for im in pil_imgs]).to(self.device)
        logits = self.model(batch)
        probs = logits.softmax(-1)
        preds, confs = self.model.tokenizer.decode(probs)
        out = []
        for text, conf in zip(preds, confs):
            conf_val = float(conf.mean()) if hasattr(conf, "mean") else float(conf)
            out.append((text, conf_val))
        return out


def group_detections_by_track(tracks_data):
    """Pivot detections by track id, keeping only PERSON-class entries
    with a real track id. Each per-track entry preserves frame, xyxy,
    confidence, and class_name (player vs goaltender)."""
    by_tid = defaultdict(list)
    for fr in tracks_data["frames"]:
        fi = fr["frame"]
        for box in fr["boxes"]:
            if box["class_id"] != PERSON_CLASS:
                continue
            tid = box["track_id"]
            if tid < 0:
                continue
            by_tid[tid].append({
                "frame": fi,
                "xyxy": box["xyxy"],
                "conf": box["conf"],
                "class_name": box.get("class_name", "player"),
            })
    return by_tid


def pick_samples(by_tid, top_n):
    """For each track, keep its ``top_n`` highest-confidence detections.
    Returned in confidence-descending order so the caller can take the
    first N if it needs even fewer."""
    samples = {}
    for tid, dets in by_tid.items():
        samples[tid] = sorted(dets, key=lambda d: d["conf"], reverse=True)[:top_n]
    return samples


def stream_needed_frames(video_path, indices):
    """Yield (frame_index, BGR frame) for the requested indices via a
    single linear pass through the video (much faster than seeking)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    needed = set(indices)
    max_needed = max(needed) if needed else -1
    current = 0
    while current <= max_needed:
        ok, frame = cap.read()
        if not ok:
            break
        if current in needed:
            yield current, frame
        current += 1
    cap.release()


def merge_tracks_by_number(enriched, by_tid):
    """Tracks with the same confident jersey number whose time spans do not
    overlap are considered the same player (tracker break)."""
    number_to_tids = defaultdict(list)
    for tid, data in enriched.items():
        jn = data["jersey_number"]
        if jn is None:
            continue
        number_to_tids[jn].append(tid)

    merged_groups = []
    for number, tids in number_to_tids.items():
        spans = []
        for tid in tids:
            dets = by_tid[int(tid)]
            spans.append({
                "tid": tid,
                "start": min(d["frame"] for d in dets),
                "end": max(d["frame"] for d in dets),
            })
        spans.sort(key=lambda s: s["start"])

        clusters = []
        for span in spans:
            if clusters and clusters[-1][-1]["end"] < span["start"]:
                clusters[-1].append(span)
            else:
                clusters.append([span])

        for cluster in clusters:
            if len(cluster) >= 2:
                merged_groups.append({
                    "jersey_number": number,
                    "track_ids": [c["tid"] for c in cluster],
                    "frame_start": cluster[0]["start"],
                    "frame_end": cluster[-1]["end"],
                })
    return merged_groups


def _filter_number(text: str) -> str:
    """Keep digits only, return when length is 1-2 (jersey numbers)."""
    digits = "".join(c for c in text if c.isdigit())
    return digits if 1 <= len(digits) <= 2 else ""


def _safe_filename(s: str) -> str:
    """Strip filesystem-hostile characters from a debug-crop filename."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)[:32]


def run(
    detections_json: Path,
    video_path: Path,
    output_path: Path,
    pose_model_name: str,
    samples_per_track: int,
    ocr_min_conf: float,
    pose_imgsz: int,
    ocr_batch_size: int,
    debug_crops_dir: Path | None = None,
    parseq_checkpoint: Path | None = None,
):
    """Run the full per-track jersey-number identification pipeline.

    For each player track in ``p1_detections.json``, sample N high-conf
    detections, run YOLO pose on those frames, isolate back-facing
    samples, crop the dorsal jersey region Koshkina-style, OCR with
    PARSeq, vote per track (≥ 2 agreeing votes required), and write
    ``p1_numbers.json`` plus an optional debug-crop dump.
    """
    device = pick_device()
    print(f"Device: {device}")

    detections_data = json.loads(detections_json.read_text())
    by_tid = group_detections_by_track(detections_data)
    print(f"Player tracks: {len(by_tid)}")

    samples = pick_samples(by_tid, samples_per_track)
    frame_work = defaultdict(list)  # frame_idx -> [(tid, xyxy)]
    for tid, dets in samples.items():
        for d in dets:
            frame_work[d["frame"]].append((tid, d["xyxy"]))
    needed = sorted(frame_work.keys())
    print(f"Unique frames to read: {len(needed)}")

    pose_path = str(resolve_yolo_path(pose_model_name))
    print(f"Loading pose model: {pose_path}")
    pose_model = YOLO(pose_path)

    ocr = ParseqOCR(device=device, checkpoint_path=parseq_checkpoint)

    if debug_crops_dir is not None:
        (debug_crops_dir / "numbers").mkdir(parents=True, exist_ok=True)
        print(f"Debug crops → {debug_crops_dir}/")

    # sample_results[tid] = [{"frame", "orient",
    #                        "number", "number_conf", "number_raw"}, ...]
    sample_results = defaultdict(list)

    pending_crops = []
    pending_meta = []  # (tid, sample_idx, frame_idx)

    def flush_ocr():
        """Run OCR on the queued crops and dispatch results to
        ``sample_results`` + the optional debug-crops dump. Called
        when the queue reaches ``ocr_batch_size`` and once at the end."""
        if not pending_crops:
            return
        results = ocr.read_batch(pending_crops)
        for (tid, idx, fi), crop, (raw_text, conf) in zip(
                pending_meta, pending_crops, results):
            kept = _filter_number(raw_text)
            if kept and conf >= ocr_min_conf:
                sample_results[tid][idx]["number"] = kept
                sample_results[tid][idx]["number_conf"] = conf
            sample_results[tid][idx]["number_raw"] = raw_text

            if debug_crops_dir is not None:
                fname = (
                    f"t{tid:04d}_f{fi:05d}"
                    f"_num-{_safe_filename(kept) or 'X'}"
                    f"_c{int(round(conf * 100)):02d}.png"
                )
                cv2.imwrite(str(debug_crops_dir / "numbers" / fname), crop)

        pending_crops.clear()
        pending_meta.clear()

    n_processed = 0
    total_samples = sum(len(v) for v in samples.values())

    for fi, frame in stream_needed_frames(video_path, needed):
        work = frame_work[fi]
        pose_res = pose_model.predict(
            source=frame, imgsz=pose_imgsz, conf=0.25,
            classes=[PERSON_CLASS], verbose=False, device=device,
        )[0]

        pose_boxes = None
        pose_kp_xy = None
        pose_kp_conf = None
        if (pose_res.keypoints is not None and pose_res.boxes is not None
                and len(pose_res.boxes) > 0):
            pose_boxes = pose_res.boxes.xyxy.cpu().numpy()
            pose_kp_xy = pose_res.keypoints.xy.cpu().numpy()
            pose_kp_conf = pose_res.keypoints.conf.cpu().numpy()

        for tid, xyxy in work:
            idx = len(sample_results[tid])
            sample_results[tid].append({
                "frame": fi, "orient": "unknown",
                "number": None, "number_conf": 0.0, "number_raw": None,
            })

            matched = -1
            if pose_boxes is not None:
                i_best = int(np.argmax(ious(np.array(xyxy), pose_boxes)))
                if ious(np.array(xyxy), pose_boxes)[i_best] > 0.5:
                    matched = i_best

            if matched < 0:
                n_processed += 1
                continue

            kp_xy = pose_kp_xy[matched]
            kp_conf = pose_kp_conf[matched]
            orient = orientation_from_kps(kp_xy, kp_conf)
            sample_results[tid][idx]["orient"] = orient

            if orient != "back":
                n_processed += 1
                continue

            num_crop = torso_back_crop(frame, kp_xy, kp_conf)
            if num_crop is not None and min(num_crop.shape[:2]) >= 16:
                pending_crops.append(num_crop)
                pending_meta.append((tid, idx, fi))

            if len(pending_crops) >= ocr_batch_size:
                flush_ocr()

            n_processed += 1
            if n_processed % 200 == 0:
                print(f"  {n_processed}/{total_samples} samples processed")

    flush_ocr()

    # Aggregate per track
    enriched = {}
    for tid, srs in sample_results.items():
        orient_counts = Counter(s["orient"] for s in srs)
        back_count = orient_counts.get("back", 0)

        num_votes = Counter()
        num_conf_acc = defaultdict(list)
        for s in srs:
            if s["number"]:
                num_votes[s["number"]] += 1
                num_conf_acc[s["number"]].append(s["number_conf"])

        # Require ≥2 agreeing votes for the winning number. Single votes
        # are too often noise — one stray vote would otherwise stamp the
        # whole track with a fake number.
        MIN_VOTES = 2
        if num_votes:
            jn, jn_count = num_votes.most_common(1)[0]
            if jn_count >= MIN_VOTES:
                jersey_conf = float(np.mean(num_conf_acc[jn]))
            else:
                jn, jersey_conf = None, 0.0
        else:
            jn, jersey_conf = None, 0.0

        enriched[str(tid)] = {
            "n_detections_total": len(by_tid[tid]),
            "samples_used": len(srs),
            "back_samples": back_count,
            "orientation_counts": dict(orient_counts),
            "jersey_number": jn,
            "jersey_votes": dict(num_votes),
            "jersey_conf": jersey_conf,
            "class_name": by_tid[tid][0].get("class_name", "player"),
        }

    merged_groups = merge_tracks_by_number(enriched, by_tid)

    output = {
        "source_detections": str(detections_json),
        "source_video": str(video_path),
        "samples_per_track": samples_per_track,
        "ocr_min_conf": ocr_min_conf,
        "pose_model": pose_path,
        "ocr_engine": "parseq",
        "parseq_checkpoint": str(parseq_checkpoint) if parseq_checkpoint else None,
        "tracks": enriched,
        "player_groups": merged_groups,
    }
    output_path.write_text(json.dumps(output, indent=2))

    n_numbered = sum(1 for t in enriched.values() if t["jersey_number"])
    print(f"\nDone.\n  Output: {output_path}")
    print(f"  Tracks with number: {n_numbered}/{len(enriched)} "
          f"({100*n_numbered/max(len(enriched),1):.1f}%)")
    print(f"  Player groups (merged tracks): {len(merged_groups)}")
    if merged_groups:
        for g in sorted(merged_groups, key=lambda x: -len(x["track_ids"]))[:10]:
            print(f"    #{g['jersey_number']}: tracks {g['track_ids']} "
                  f"(frames {g['frame_start']}–{g['frame_end']})")


def main():
    """CLI entry point — parse arguments and dispatch to ``run``."""
    p = argparse.ArgumentParser(
        description="RILH-AI-Vision — p1_c_numbers : per-track jersey OCR"
    )
    p.add_argument("detections_json", type=str,
                   help="Path to p1_a_detect output (p1_detections.json)")
    p.add_argument("video", type=str)
    p.add_argument("--output", type=str, default=None,
                   help="Output JSON path (default: <detections_dir>/p1_numbers.json)")
    p.add_argument("--pose-model", type=str, default="yolo11n-pose.pt",
                   help="YOLO pose model. Examples: yolo11n-pose.pt (default, "
                        "~6MB, fast), yolo11x-pose.pt (best YOLO11 pose), "
                        "yolo26l-pose.pt (YOLO26 large pose, ~55MB, newer "
                        "architecture). Auto-downloaded into models/.")
    p.add_argument("--samples-per-track", type=int, default=15)
    p.add_argument("--ocr-min-conf", type=float, default=0.30,
                   help="Per-sample minimum OCR confidence to count toward "
                        "the per-track vote.")
    p.add_argument("--pose-imgsz", type=int, default=1280)
    p.add_argument("--ocr-batch", type=int, default=32)
    p.add_argument("--debug-crops-dir", type=str, default=None,
                   help="If set, save every OCR'd crop into <dir>/numbers/ "
                        "with a filename encoding track id, frame, OCR "
                        "result and confidence.")
    p.add_argument("--parseq-checkpoint", type=str, default=None,
                   help="Path to a custom PARSeq checkpoint (Lightning .pt). "
                        "Default = baudm/parseq pretrained on STR. Use "
                        "models/parseq_hockey_rilh.pt for our RILH-fine-tuned "
                        "Koshkina hockey model (non-commercial license).")
    args = p.parse_args()

    detections_json = Path(args.detections_json)
    video = Path(args.video)
    output = (Path(args.output) if args.output
              else detections_json.with_name("p1_numbers.json"))
    debug_dir = Path(args.debug_crops_dir) if args.debug_crops_dir else None
    parseq_ckpt = (Path(args.parseq_checkpoint)
                   if args.parseq_checkpoint else None)

    run(
        detections_json, video, output,
        args.pose_model, args.samples_per_track, args.ocr_min_conf,
        args.pose_imgsz, args.ocr_batch,
        debug_crops_dir=debug_dir,
        parseq_checkpoint=parseq_ckpt,
    )


if __name__ == "__main__":
    main()
