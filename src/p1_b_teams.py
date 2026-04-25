"""
RILH-AI-Vision — p1_b_teams
Classify each player track into one of two teams.

Pipeline (per sampled detection, not per frame — keeps inference tractable):
  1) For each track, pick the N highest-confidence detections.
  2) Run YOLO pose on the unique frames needed; match by IoU to the track bbox.
  3) Crop the torso rectangle shoulders→hips (any orientation).
  4) Sub-divide the torso into a rows×cols grid; get the dominant BGR in each
     sub-region (k-means on color-filtered pixels); average the sub-region
     dominants into one robust color per crop.
  5) k-means (HSV by default) over *all* per-crop colors — not per-track medians.
     Every crop gets a team label; each track inherits the majority vote.

Inputs:
  p1_a_detections.json — produced by p1_a_detect (per-frame detections + tracks).
  video.mp4       — the source video, sampled for the cropped frames.

Outputs:
  p1_b_teams.json        — per-track team_id, vote_distribution, vote_confidence,
                      n_color_samples; plus team_centers_bgr and cluster_margin.
  teams_preview.png — grid of torso thumbnails, one row per team, sorted by
                      vote confidence desc, with tid and vote counts overlaid.
                      Left gutter shows the team label and the centroid swatch.

Roster (roller inline hockey): 4 skaters + 1 goalie per team, 1–2 refs.
Refs are filtered upstream when P1.a used HockeyAI; they leak with COCO.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PERSON_CLASS = 0

# COCO-17 keypoint indices (same as phase6_identify)
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
    """Route a bare YOLO filename (e.g. ``yolo11n-pose.pt``) into the
    project ``models/`` folder so Ultralytics auto-downloads land there
    instead of CWD. Paths with a directory component are kept as-is."""
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


def torso_crop_from_pose(frame, kp_xy, kp_conf, thr=0.3):
    """Crop shoulders → hips regardless of orientation.
    Returns None if both shoulders aren't visible above thr."""
    if not (kp_conf[KP_LSHO] > thr and kp_conf[KP_RSHO] > thr):
        return None
    lsx, lsy = kp_xy[KP_LSHO, 0], kp_xy[KP_LSHO, 1]
    rsx, rsy = kp_xy[KP_RSHO, 0], kp_xy[KP_RSHO, 1]
    sho_cx = (lsx + rsx) / 2.0
    sho_cy = (lsy + rsy) / 2.0
    sho_w = abs(lsx - rsx)
    if kp_conf[KP_LHIP] > thr and kp_conf[KP_RHIP] > thr:
        hip_cy = (kp_xy[KP_LHIP, 1] + kp_xy[KP_RHIP, 1]) / 2.0
    else:
        hip_cy = sho_cy + 2.0 * sho_w
    half_w = sho_w * 0.7
    x1 = sho_cx - half_w
    x2 = sho_cx + half_w
    y1 = sho_cy - sho_w * 0.05
    y2 = hip_cy
    return safe_crop(frame, [x1, y1, x2, y2])


def torso_crop_from_bbox(frame, xyxy):
    """Fallback crop (no pose): 15–45 % of bbox height, middle 50 % of bbox
    width. Tight crop to avoid background bleed (ice / opponents / boards)
    between the player's arms and torso. Less accurate than pose-based but
    keeps coverage high on dark jerseys where YOLO11-pose often misses
    keypoints."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    bh = y2 - y1
    bw = x2 - x1
    ty1 = max(0, int(y1 + 0.15 * bh))
    ty2 = min(h, int(y1 + 0.45 * bh))
    tx1 = max(0, int(x1 + 0.25 * bw))
    tx2 = min(w, int(x2 - 0.25 * bw))
    if ty2 <= ty1 + 4 or tx2 <= tx1 + 4:
        return None
    return frame[ty1:ty2, tx1:tx2]


def dominant_bgr(bgr_crop, min_mask_pixels=30):
    """Dominant BGR of a crop. Filter is val-only (30–245) — we intentionally
    keep low-saturation pixels because for near-grayscale jerseys (white,
    black) those ARE the signal; a sat filter would bias the dominant toward
    any colored logo instead of the actual jersey color."""
    if bgr_crop is None or bgr_crop.size == 0:
        return None
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    val = hsv[..., 2]
    mask = (val > 30) & (val < 245)
    if mask.sum() < min_mask_pixels:
        return None
    pixels = bgr_crop[mask].astype(np.float32)
    k = 3 if len(pixels) >= 30 else 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3,
                                    cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    return tuple(int(c) for c in centers[int(np.argmax(counts))])


def multi_point_color(torso_crop, rows=3, cols=2):
    """Split torso into rows×cols sub-regions, compute dominant BGR in each,
    return the mean of the dominants. More robust to localized noise
    (logo patches, shadows, sticker stripes) than a single whole-crop dominant."""
    if torso_crop is None or torso_crop.size == 0:
        return None
    h, w = torso_crop.shape[:2]
    # Each sub-region must be at least ~6 px per side to have a meaningful dominant
    if h < rows * 6 or w < cols * 6:
        return dominant_bgr(torso_crop)
    colors = []
    for i in range(rows):
        y0 = i * h // rows
        y1 = (i + 1) * h // rows
        for j in range(cols):
            x0 = j * w // cols
            x1 = (j + 1) * w // cols
            c = dominant_bgr(torso_crop[y0:y1, x0:x1], min_mask_pixels=10)
            if c is not None:
                colors.append(c)
    if not colors:
        # All sub-regions failed (tiny / uniform / mostly masked out)
        return dominant_bgr(torso_crop)
    return tuple(int(v) for v in np.mean(np.array(colors, dtype=np.float32), axis=0))


def group_detections_by_track(tracks_data):
    """Pivot detections by track id, keeping only PERSON class with a
    real (non-negative) track id. Returns ``{tid: [(frame, xyxy, conf), ...]}``."""
    by_tid = defaultdict(list)
    for fr in tracks_data["frames"]:
        fi = fr["frame"]
        for b in fr["boxes"]:
            if b["class_id"] != PERSON_CLASS or b["track_id"] < 0:
                continue
            by_tid[b["track_id"]].append((fi, b["xyxy"], b["conf"]))
    return by_tid


def goaltender_tids(tracks_data, threshold=0.5):
    """Return the set of track_ids where MORE THAN `threshold` fraction of
    detections are tagged class_name='goaltender' (HockeyAI backend).

    Strict majority by default — a few mis-tagged frames in an otherwise-
    player track no longer flip the entire track to goalie. The previous
    `any` rule was poisoning entities downstream (Phase 1.6 propagated
    the goalie tag to every other player merged into the same entity)."""
    from collections import defaultdict
    counts = defaultdict(lambda: [0, 0])  # tid -> [n_player, n_goalie]
    for fr in tracks_data["frames"]:
        for b in fr["boxes"]:
            if b["class_id"] != PERSON_CLASS or b["track_id"] < 0:
                continue
            if b.get("class_name") == "goaltender":
                counts[b["track_id"]][1] += 1
            else:
                counts[b["track_id"]][0] += 1
    return {tid for tid, (np_, ng) in counts.items()
            if ng / max(np_ + ng, 1) > threshold}


def stream_needed_frames(video_path, indices):
    """Yield (frame_index, BGR frame) pairs for the requested frame
    indices, reading the video sequentially (single linear pass — much
    faster than seeking back and forth)."""
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


def sample_jersey_colors(tracks_data, video_path, pose_model, device,
                         samples_per_track, pose_imgsz, multi_grid):
    """Extract per-crop colors for each track via pose + multi-point averaging.
    Returns {tid: {"crop_colors": [(b,g,r), ...], "preview_crop": ndarray}}.
    Tracks with zero usable crops are omitted."""
    by_tid = group_detections_by_track(tracks_data)
    frame_work = defaultdict(list)  # frame_idx -> [(tid, xyxy)]
    for tid, dets in by_tid.items():
        for fi, xyxy, _ in sorted(dets, key=lambda d: -d[2])[:samples_per_track]:
            frame_work[fi].append((tid, xyxy))
    needed = sorted(frame_work.keys())
    print(f"Unique frames to read: {len(needed)}")

    crops_by_tid = defaultdict(list)
    preview_by_tid = {}

    n_total_samples = sum(len(v) for v in frame_work.values())
    n_processed = 0
    n_pose_crops = 0
    n_bbox_fallback = 0
    n_color_ok = 0

    for fi, frame in stream_needed_frames(video_path, needed):
        work = frame_work[fi]
        pose_res = pose_model.predict(
            source=frame, imgsz=pose_imgsz, conf=0.10,
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
            n_processed += 1

            torso = None
            if pose_boxes is not None:
                iou_arr = ious(np.array(xyxy), pose_boxes)
                i_best = int(np.argmax(iou_arr))
                if iou_arr[i_best] > 0.5:
                    torso = torso_crop_from_pose(
                        frame, pose_kp_xy[i_best], pose_kp_conf[i_best]
                    )
                    if torso is not None:
                        n_pose_crops += 1

            if torso is None:
                # Fallback: bbox-based crop — coarser but keeps coverage up.
                torso = torso_crop_from_bbox(frame, xyxy)
                if torso is not None:
                    n_bbox_fallback += 1

            if torso is None:
                continue

            color = multi_point_color(torso, *multi_grid)
            if color is None:
                continue
            n_color_ok += 1

            crops_by_tid[tid].append(color)
            if tid not in preview_by_tid:
                preview_by_tid[tid] = torso

        if n_processed % 300 == 0 or n_processed == n_total_samples:
            print(f"  {n_processed}/{n_total_samples} samples — "
                  f"pose {n_pose_crops}, bbox-fallback {n_bbox_fallback}, "
                  f"color {n_color_ok}")

    out = {}
    for tid, colors in crops_by_tid.items():
        out[tid] = {
            "crop_colors": colors,
            "preview_crop": preview_by_tid.get(tid),
        }
    return out


def _colors_to_space(bgr_list, space):
    """Convert a list of BGR triplets to a (N, 3) float32 array in either
    BGR or HSV — used by the k-means step. HSV typically separates
    teams better when one wears a near-grayscale jersey."""
    bgr = np.array(bgr_list, dtype=np.uint8).reshape(-1, 1, 3)
    if space == "hsv":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    if space == "bgr":
        return bgr.reshape(-1, 3).astype(np.float32)
    raise ValueError(f"Unknown color space: {space}")


def classify_teams(crops_by_tid, k=2, space="hsv", fit_tids=None):
    """k-means then majority vote per track.

    If fit_tids is not None, the centroids are fit on crops from those tids
    only (e.g. skaters) — then ALL tracks (including the excluded ones, e.g.
    goaltenders) are classified crop-by-crop against those centroids and
    majority-voted. This keeps goalie pads from pulling the skater team
    centers toward grey.

    Returns (team_of_tid, team_centers_bgr, votes_by_tid, margin)."""
    all_flat = [(tid, c) for tid, d in crops_by_tid.items() for c in d["crop_colors"]]
    if len(all_flat) < k:
        empty_of = {tid: 0 for tid in crops_by_tid}
        empty_votes = {
            tid: [len(d["crop_colors"])] + [0] * (k - 1)
            for tid, d in crops_by_tid.items()
        }
        return empty_of, [(128, 128, 128)] * k, empty_votes, 0.0

    fit_set = set(crops_by_tid.keys()) if fit_tids is None else set(fit_tids)
    fit_colors = [c for tid, c in all_flat if tid in fit_set]
    if len(fit_colors) < k:
        # Not enough fit data — fall back to clustering on everything
        fit_colors = [c for _, c in all_flat]

    fit_data = _colors_to_space(fit_colors, space)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, fit_labels, centers_in_space = cv2.kmeans(
        fit_data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    fit_labels = fit_labels.flatten()

    # Classify every crop (fit + out-of-fit) against those centroids.
    all_data = _colors_to_space([c for _, c in all_flat], space)
    # Squared-distance argmin = nearest centroid.
    dists = ((all_data[:, None, :] - centers_in_space[None, :, :]) ** 2).sum(axis=2)
    all_labels = dists.argmin(axis=1)

    # Back-convert centers to BGR for rendering.
    if space == "hsv":
        hsv_centers = centers_in_space.astype(np.uint8).reshape(-1, 1, 3)
        bgr_centers = cv2.cvtColor(hsv_centers, cv2.COLOR_HSV2BGR).reshape(-1, 3)
    else:
        bgr_centers = centers_in_space
    team_centers = [tuple(int(c) for c in bgr_centers[i]) for i in range(k)]

    votes_by_tid = {tid: [0] * k for tid in crops_by_tid}
    for (tid, _), lbl in zip(all_flat, all_labels):
        votes_by_tid[tid][int(lbl)] += 1
    team_of = {tid: int(np.argmax(v)) for tid, v in votes_by_tid.items()}

    margin = 0.0
    if k == 2:
        inter = float(np.linalg.norm(centers_in_space[0] - centers_in_space[1]))
        spreads = []
        for ci in range(k):
            pts = fit_data[fit_labels == ci]
            if len(pts) > 0:
                spreads.append(float(
                    np.mean(np.linalg.norm(pts - centers_in_space[ci], axis=1))
                ))
        if spreads:
            margin = inter / (float(np.mean(spreads)) + 1e-6)
    return team_of, team_centers, votes_by_tid, margin


def render_preview(crops_by_tid, team_of, votes_by_tid, team_centers_bgr,
                   output_path, cols=10, thumb=80):
    """One row per team, sorted by vote confidence desc. Each tile shows the
    track id (top) and vote split like '5/1' (bottom)."""
    by_team = defaultdict(list)
    for tid, team in team_of.items():
        entry = crops_by_tid.get(tid)
        if entry is None or entry.get("preview_crop") is None:
            continue
        votes = votes_by_tid[tid]
        total = sum(votes)
        conf = votes[team] / total if total else 0.0
        by_team[team].append((conf, tid, entry["preview_crop"], votes))

    if not by_team:
        return

    rows = []
    for team in sorted(by_team):
        picks = sorted(by_team[team], key=lambda x: (-x[0], x[1]))[:cols]
        tiles = []
        for conf, tid, c, votes in picks:
            tile = cv2.resize(c, (thumb, thumb))
            cv2.putText(tile, f"{tid}", (2, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(tile, f"{votes[0]}/{votes[1]}", (2, thumb - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        while len(tiles) < cols:
            tiles.append(np.zeros((thumb, thumb, 3), dtype=np.uint8))
        rows.append(np.hstack(tiles))

    grid = np.vstack(rows)
    gutter_w = 140
    gutter = np.zeros((grid.shape[0], gutter_w, 3), dtype=np.uint8)
    for i, team in enumerate(sorted(by_team)):
        y0 = i * thumb
        cv2.putText(gutter, f"Team {team}",
                    (8, y0 + thumb // 2 - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        sx1, sy1 = gutter_w - 46, y0 + thumb // 2 - 4
        sx2, sy2 = gutter_w - 8, y0 + thumb // 2 + 24
        cv2.rectangle(gutter, (sx1, sy1), (sx2, sy2),
                      tuple(int(c) for c in team_centers_bgr[team]), -1)
        cv2.rectangle(gutter, (sx1, sy1), (sx2, sy2), (255, 255, 255), 1)
    grid = np.hstack([gutter, grid])
    cv2.imwrite(str(output_path), grid)


def run(detections_json, video_path, output, samples_per_track, pose_model_name,
        pose_imgsz, preview_cols, space, multi_grid):
    """Run the team-classification pipeline end-to-end.

    Loads ``p1_a_detections.json``, samples crops from the source video,
    extracts dominant torso colors, fits a k=2 k-means on skater-only
    crops (goalies are classified post-hoc against those centroids),
    and writes ``p1_b_teams.json`` plus a debug ``teams_preview.png``.
    """
    device = pick_device()
    print(f"Device: {device}")

    detections_data = json.loads(detections_json.read_text())
    all_tids = {
        b["track_id"]
        for fr in detections_data["frames"] for b in fr["boxes"]
        if b["class_id"] == PERSON_CLASS and b["track_id"] >= 0
    }
    goalie_tids = goaltender_tids(detections_data)
    skater_tids = all_tids - goalie_tids
    print(f"Total player tracks in {detections_json.name}: {len(all_tids)} "
          f"(skaters {len(skater_tids)}, goaltenders {len(goalie_tids)})")

    pose_path = str(resolve_yolo_path(pose_model_name))
    print(f"Loading pose model: {pose_path}")
    pose_model = YOLO(pose_path)

    print(f"Sampling jersey colors (≤{samples_per_track}/track, pose-based, "
          f"{multi_grid[0]}×{multi_grid[1]} multi-point)…")
    crops_by_tid = sample_jersey_colors(
        detections_data, video_path, pose_model, device,
        samples_per_track, pose_imgsz, multi_grid,
    )
    n_crops = sum(len(v["crop_colors"]) for v in crops_by_tid.values())
    print(f"  colors extracted for {len(crops_by_tid)}/{len(all_tids)} tracks "
          f"({n_crops} crops total)")

    fit_tids = skater_tids & set(crops_by_tid.keys())
    print(f"Clustering teams (k=2, {space.upper()}, per-crop k-means, "
          f"fitting on {len(fit_tids)} skater tracks, classifying all)…")
    team_of, centers, votes_by_tid, margin = classify_teams(
        crops_by_tid, k=2, space=space, fit_tids=fit_tids,
    )

    n0 = sum(1 for t in team_of.values() if t == 0)
    n1 = sum(1 for t in team_of.values() if t == 1)
    low_conf = 0
    goalie_split = [0, 0]
    for tid, team in team_of.items():
        total = sum(votes_by_tid[tid])
        if total > 0 and votes_by_tid[tid][team] / total < 0.67:
            low_conf += 1
        if tid in goalie_tids:
            goalie_split[team] += 1
    verdict = "OK" if margin >= 1.0 else "LOW — inspect preview"
    print(f"  Team 0 (center BGR={centers[0]}): {n0} tracks")
    print(f"  Team 1 (center BGR={centers[1]}): {n1} tracks")
    print(f"  Goaltender tracks split: team 0 = {goalie_split[0]}, "
          f"team 1 = {goalie_split[1]} (classified post-hoc, didn't fit centers)")
    print(f"  Tracks without usable sample: {len(all_tids) - len(crops_by_tid)}")
    print(f"  Tracks with mixed votes (<67% agreement): {low_conf}")
    print(f"  Cluster margin: {margin:.2f} — {verdict}")

    out_json = {
        "source_detections": str(detections_json),
        "source_video": str(video_path),
        "version": "v2",
        "method": "pose_torso + multi_point_avg + per_crop_vote",
        "samples_per_track": samples_per_track,
        "color_space": space,
        "multi_point_grid": list(multi_grid),
        "roster_note": (
            "Roller inline hockey: 4 skaters + 1 goalie per team, 1–2 refs. "
            "Refs leak into clustering when Phase 1 used COCO (HockeyAI drops them)."
        ),
        "team_centers_bgr": [list(c) for c in centers],
        "cluster_margin": margin,
        "tracks": {
            str(tid): {
                "team_id": team_of[tid],
                "vote_distribution": votes_by_tid[tid],
                "vote_confidence": (
                    votes_by_tid[tid][team_of[tid]] / sum(votes_by_tid[tid])
                    if sum(votes_by_tid[tid]) > 0 else 0.0
                ),
                "n_color_samples": len(crops_by_tid[tid]["crop_colors"]),
                "is_goaltender": tid in goalie_tids,
            }
            for tid in crops_by_tid
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out_json, indent=2))

    preview_path = output.with_name("teams_preview.png")
    render_preview(crops_by_tid, team_of, votes_by_tid, centers,
                   preview_path, cols=preview_cols)
    print(f"\nDone.")
    print(f"  JSON:    {output}")
    print(f"  Preview: {preview_path}")


def main():
    """CLI entry point — parse arguments and dispatch to ``run``."""
    p = argparse.ArgumentParser(
        description="RILH-AI-Vision — p1_b_teams (pose torso + multi-point "
                    "+ per-crop vote)"
    )
    p.add_argument("detections_json", type=str,
                   help="P1.a output (p1_a_detections.json)")
    p.add_argument("video", type=str)
    p.add_argument("--output", type=str, default=None,
                   help="Output JSON (default: <detections_dir>/p1_b_teams.json)")
    p.add_argument("--samples-per-track", type=int, default=8,
                   help="Crops per track to vote on (higher = more robust, "
                        "slower inference)")
    p.add_argument("--pose-model", type=str, default="yolo11n-pose.pt",
                   help="YOLO pose model. Examples: yolo11n-pose.pt (default, "
                        "~6MB, fast), yolo11x-pose.pt (best YOLO11 pose), "
                        "yolo26l-pose.pt (YOLO26 large pose, ~55MB, newer "
                        "architecture). Auto-downloaded into models/.")
    p.add_argument("--pose-imgsz", type=int, default=1280)
    p.add_argument("--preview-cols", type=int, default=10)
    p.add_argument("--space", choices=["hsv", "bgr"], default="hsv",
                   help="Color space for k-means (hsv default)")
    p.add_argument("--grid", type=str, default="3x2",
                   help="Multi-point grid inside the torso (rows x cols). "
                        "3x2 = 6 sub-regions.")
    args = p.parse_args()

    try:
        rows, cols = (int(x) for x in args.grid.lower().split("x"))
    except Exception as e:
        raise SystemExit(f"Invalid --grid {args.grid!r}: expected 'RxC'") from e

    detections_json = Path(args.detections_json)
    video = Path(args.video)
    output = (Path(args.output) if args.output
              else detections_json.with_name("p1_b_teams.json"))

    run(detections_json, video, output,
        args.samples_per_track, args.pose_model, args.pose_imgsz,
        args.preview_cols, args.space, (rows, cols))


if __name__ == "__main__":
    main()
