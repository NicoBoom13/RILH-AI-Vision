"""
RILH-AI-Vision — Phase 6
Player identification via dorsal jersey-number OCR.

Input:  tracks.json from Phase 1 (HockeyAI strongly recommended) + source video
Output: tracks_identified.json — per player-track jersey number + confidence,
        plus optional track-merge groups for players whose track-id was broken
        and re-assigned by the tracker.

Pipeline (per track_id, not per frame — keeps inference tractable):
  1) sample the N highest-confidence detections of that track
  2) run YOLO11-pose on each sample's full frame, match by IoU with the track bbox
  3) classify orientation from pose keypoints (nose/ears/shoulders)
  4) on back-facing samples, crop the torso-back region (shoulders -> hips)
  5) run PARSeq scene-text recognition, keep digits only (1-2 chars)
  6) vote per track; merge tracks sharing the same number with non-overlapping
     time spans (same player, tracker break)
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

PERSON_CLASS = 0

# COCO-17 keypoint indices used by YOLO11-pose
KP_NOSE, KP_LEYE, KP_REYE, KP_LEAR, KP_REAR = 0, 1, 2, 3, 4
KP_LSHO, KP_RSHO = 5, 6
KP_LHIP, KP_RHIP = 11, 12

MODELS_DIR = Path("models")
DIGIT_RE = re.compile(r"\d+")


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_yolo_path(name: str) -> Path:
    p = Path(name)
    if len(p.parts) == 1:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return MODELS_DIR / p.name
    return p


def safe_crop(frame, xyxy):
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


def torso_back_crop(frame, kp_xy, kp_conf, thr=0.3,
                    y_top_frac=0.15, y_bot_frac=0.65, x_pad_ratio=0.30):
    """Crop the dorsal number band — the rectangle where the back number
    actually sits.

    Vertical range: `y_top_frac` → `y_bot_frac` of the shoulder-to-hip span
    (default 15–65 %, i.e. the upper half of the torso where the number is
    centred — NOT the full torso).

    Horizontal range: shoulder-width inflated by `x_pad_ratio` on each side
    (default 30 %) to catch wide digits without cutting off edges.

    This tight band has an aspect ratio (w:h) close to 2:1 instead of the
    ~0.8:1 of the old full-torso crop, which is much closer to what PARSeq
    expects (4:1 = 128:32). The remaining 2× discrepancy is absorbed by
    letterbox padding inside ParseqOCR.read_batch, so PARSeq never sees a
    horizontally-stretched digit."""
    shos_ok = kp_conf[KP_LSHO] > thr and kp_conf[KP_RSHO] > thr
    hips_ok = kp_conf[KP_LHIP] > thr and kp_conf[KP_RHIP] > thr
    if not shos_ok:
        return None
    lsx, lsy = kp_xy[KP_LSHO, 0], kp_xy[KP_LSHO, 1]
    rsx, rsy = kp_xy[KP_RSHO, 0], kp_xy[KP_RSHO, 1]
    sho_cx = (lsx + rsx) / 2.0
    sho_cy = (lsy + rsy) / 2.0
    sho_w = abs(lsx - rsx)
    if hips_ok:
        hip_cy = (kp_xy[KP_LHIP, 1] + kp_xy[KP_RHIP, 1]) / 2.0
    else:
        hip_cy = sho_cy + 2.2 * sho_w
    torso_h = hip_cy - sho_cy
    y1 = sho_cy + y_top_frac * torso_h
    y2 = sho_cy + y_bot_frac * torso_h
    half_w = sho_w * (0.5 + x_pad_ratio)
    x1 = sho_cx - half_w
    x2 = sho_cx + half_w
    return safe_crop(frame, [x1, y1, x2, y2])


def ious(xyxy, boxes):
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


class TrOCR_OCR:
    """Microsoft TrOCR (VisionEncoderDecoder, base, printed-text variant).
    Heavier than PARSeq (~334M params) but handles arbitrary input aspect
    ratios natively via its ViT encoder — no letterbox trick needed.

    Interface matches ParseqOCR.read_batch: returns list of (digits, conf).
    Confidence is the mean softmax probability of the selected tokens in
    the generated sequence."""

    def __init__(self, device, model_name="microsoft/trocr-base-printed"):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self.device = device
        print(f"Loading {model_name} (first run downloads ~340 MB)…")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = (VisionEncoderDecoderModel.from_pretrained(model_name)
                      .to(device).eval())

    @torch.no_grad()
    def read_batch(self, bgr_crops):
        if not bgr_crops:
            return []
        pil_imgs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
                    for c in bgr_crops]
        pixel_values = self.processor(
            images=pil_imgs, return_tensors="pt"
        ).pixel_values.to(self.device)
        gen = self.model.generate(
            pixel_values,
            max_new_tokens=6,
            output_scores=True,
            return_dict_in_generate=True,
        )
        sequences = gen.sequences
        texts = self.processor.batch_decode(sequences, skip_special_tokens=True)

        # Per-sample confidence = mean softmax prob of the chosen tokens
        # across the n_gen steps. Sequences start with decoder_start_token at
        # position 0; token at step i is sequences[:, i + 1].
        import torch.nn.functional as F
        if gen.scores:
            step_probs = []
            for i, scores in enumerate(gen.scores):
                probs = F.softmax(scores, dim=-1)
                chosen = sequences[:, i + 1].unsqueeze(1)
                step_probs.append(probs.gather(1, chosen).squeeze(1))
            confs = torch.stack(step_probs, dim=1).mean(dim=1).cpu().numpy()
        else:
            confs = [1.0] * len(pil_imgs)

        out = []
        for text, conf in zip(texts, confs):
            digits = "".join(ch for ch in text if ch.isdigit())
            if 1 <= len(digits) <= 2:
                out.append((digits, float(conf)))
            else:
                out.append(("", 0.0))
        return out


class ParseqOCR:
    """PARSeq scene-text recognition via torch.hub.

    Upstream PARSeq resizes its input directly to `model.hparams.img_size`
    (typically 32×128, a 1:4 h:w aspect). Our torso crops are close to
    square, so a naive resize stretches digits horizontally by ~3-4×,
    which degrades recognition badly. We letterbox-pad each crop to 1:4
    BEFORE the resize so PARSeq sees geometrically-correct digits."""

    def __init__(self, device):
        self.device = device
        print("Loading PARSeq (first run downloads from github.com/baudm/parseq)…")
        self.model = torch.hub.load(
            "baudm/parseq", "parseq", pretrained=True, trust_repo=True
        ).eval().to(device)
        self.img_size = self.model.hparams.img_size  # (h, w), usually (32, 128)
        from torchvision import transforms as T
        self.preprocess = T.Compose([
            T.Resize(self.img_size, T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])
        self.target_w_over_h = float(self.img_size[1]) / float(self.img_size[0])

    @staticmethod
    def _letterbox_to_aspect(bgr, target_w_over_h, pad_value=0):
        """Pad crop with `pad_value` (black) so w/h == target aspect, without
        changing the content region. If the crop is already wider than target,
        pad top/bottom instead."""
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
        """Return list of (digit_string, confidence) per crop. Empty string
        means no digit found or low confidence."""
        if not bgr_crops:
            return []
        letterboxed = [
            self._letterbox_to_aspect(c, self.target_w_over_h) for c in bgr_crops
        ]
        pil_imgs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
                    for c in letterboxed]
        batch = torch.stack([self.preprocess(im) for im in pil_imgs]).to(self.device)
        logits = self.model(batch)
        probs = logits.softmax(-1)
        preds, confs = self.model.tokenizer.decode(probs)
        out = []
        for text, conf in zip(preds, confs):
            conf_val = float(conf.mean()) if hasattr(conf, "mean") else float(conf)
            digits = "".join(ch for ch in text if ch.isdigit())
            if 1 <= len(digits) <= 2:
                out.append((digits, conf_val))
            else:
                out.append(("", 0.0))
        return out


def group_detections_by_track(tracks_data):
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
    samples = {}
    for tid, dets in by_tid.items():
        samples[tid] = sorted(dets, key=lambda d: d["conf"], reverse=True)[:top_n]
    return samples


def stream_needed_frames(video_path, indices):
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

        # Greedy chaining: consecutive spans with no overlap go together
        clusters = []
        for span in spans:
            if clusters and clusters[-1][-1]["end"] < span["start"]:
                clusters[-1].append(span)
            else:
                clusters.append([span])

        # Output the multi-track clusters only (single-track tids stay as-is)
        for cluster in clusters:
            if len(cluster) >= 2:
                merged_groups.append({
                    "jersey_number": number,
                    "track_ids": [c["tid"] for c in cluster],
                    "frame_start": cluster[0]["start"],
                    "frame_end": cluster[-1]["end"],
                })
    return merged_groups


def run(
    tracks_json: Path,
    video_path: Path,
    output_path: Path,
    pose_model_name: str,
    samples_per_track: int,
    ocr_min_conf: float,
    pose_imgsz: int,
    ocr_batch_size: int,
    ocr_engine: str = "parseq",
):
    device = pick_device()
    print(f"Device: {device}")

    tracks_data = json.loads(tracks_json.read_text())
    by_tid = group_detections_by_track(tracks_data)
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

    if ocr_engine == "trocr":
        ocr = TrOCR_OCR(device=device)
    else:
        ocr = ParseqOCR(device=device)
    print(f"OCR engine: {ocr_engine}")

    # sample_results[tid] = [{"frame", "orient", "number", "conf"}, ...]
    sample_results = defaultdict(list)

    # Collect back-crops across many tracks to run PARSeq in batches
    pending_crops = []
    pending_meta = []  # (tid, sample_idx)

    def flush_ocr():
        if not pending_crops:
            return
        results = ocr.read_batch(pending_crops)
        for (tid, idx), (digits, conf) in zip(pending_meta, results):
            if digits and conf >= ocr_min_conf:
                sample_results[tid][idx]["number"] = digits
                sample_results[tid][idx]["ocr_conf"] = conf
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
                "number": None, "ocr_conf": 0.0,
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

            tbc = torso_back_crop(frame, kp_xy, kp_conf)
            if tbc is not None and min(tbc.shape[:2]) >= 16:
                pending_crops.append(tbc)
                pending_meta.append((tid, idx))
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

        votes = Counter()
        conf_by_num = defaultdict(list)
        for s in srs:
            if s["number"]:
                votes[s["number"]] += 1
                conf_by_num[s["number"]].append(s["ocr_conf"])

        if votes:
            jn, n = votes.most_common(1)[0]
            jersey_conf = float(np.mean(conf_by_num[jn]))
        else:
            jn, n, jersey_conf = None, 0, 0.0

        enriched[str(tid)] = {
            "n_detections_total": len(by_tid[tid]),
            "samples_used": len(srs),
            "back_samples": back_count,
            "orientation_counts": dict(orient_counts),
            "jersey_number": jn,
            "jersey_votes": dict(votes),
            "jersey_conf": jersey_conf,
            "class_name": by_tid[tid][0].get("class_name", "player"),
        }

    merged_groups = merge_tracks_by_number(enriched, by_tid)

    output = {
        "source_tracks": str(tracks_json),
        "source_video": str(video_path),
        "samples_per_track": samples_per_track,
        "ocr_min_conf": ocr_min_conf,
        "pose_model": pose_path,
        "ocr_model": "parseq",
        "tracks": enriched,
        "player_groups": merged_groups,
    }
    output_path.write_text(json.dumps(output, indent=2))

    n_numbered = sum(1 for t in enriched.values() if t["jersey_number"])
    print(f"\nDone.\n  Output: {output_path}")
    print(f"  Tracks identified: {n_numbered}/{len(enriched)} "
          f"({100*n_numbered/max(len(enriched),1):.1f}%)")
    print(f"  Player groups (merged tracks): {len(merged_groups)}")
    if merged_groups:
        for g in sorted(merged_groups, key=lambda x: -len(x["track_ids"]))[:10]:
            print(f"    #{g['jersey_number']}: tracks {g['track_ids']} "
                  f"(frames {g['frame_start']}–{g['frame_end']})")


def main():
    p = argparse.ArgumentParser(
        description="RILH-AI-Vision — Phase 6: player identification by dorsal OCR"
    )
    p.add_argument("tracks_json", type=str)
    p.add_argument("video", type=str)
    p.add_argument("--output", type=str, default=None,
                   help="Output JSON path (default: <tracks_dir>/tracks_identified.json)")
    p.add_argument("--pose-model", type=str, default="yolo11n-pose.pt")
    p.add_argument("--samples-per-track", type=int, default=15)
    p.add_argument("--ocr-min-conf", type=float, default=0.4)
    p.add_argument("--pose-imgsz", type=int, default=1280)
    p.add_argument("--ocr-batch", type=int, default=32)
    p.add_argument("--ocr-engine", choices=["parseq", "trocr"], default="parseq",
                   help="OCR engine. parseq = PARSeq via torch.hub (fast). "
                        "trocr = Microsoft TrOCR base-printed (heavier ~340MB, "
                        "more robust on difficult / small / angled text).")
    args = p.parse_args()

    tracks_json = Path(args.tracks_json)
    video = Path(args.video)
    output = (Path(args.output) if args.output
              else tracks_json.with_name("tracks_identified.json"))

    run(
        tracks_json, video, output,
        args.pose_model, args.samples_per_track, args.ocr_min_conf,
        args.pose_imgsz, args.ocr_batch,
        ocr_engine=args.ocr_engine,
    )


if __name__ == "__main__":
    main()
