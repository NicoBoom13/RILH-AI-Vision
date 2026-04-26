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
                         samples_per_track, pose_imgsz, multi_grid,
                         keep_torso_crops=False):
    """Extract per-crop torso color samples for each track via pose +
    multi-point averaging. When ``keep_torso_crops=True`` the raw BGR
    torso ndarray for every kept sample is also returned alongside the
    colour stat — needed by the embedding-based team engines (osnet,
    siglip, contrastive) that consume pixels directly. The HSV engine
    only needs ``crop_colors`` and ignores ``torso_crops``.

    Returns ``{tid: {"crop_colors": [(b,g,r), ...],
                     "torso_crops": [ndarray | None, ...],
                     "preview_crop": ndarray}}``.
    Tracks with zero usable crops are omitted."""
    by_tid = group_detections_by_track(tracks_data)
    frame_work = defaultdict(list)  # frame_idx -> [(tid, xyxy)]
    for tid, dets in by_tid.items():
        for fi, xyxy, _ in sorted(dets, key=lambda d: -d[2])[:samples_per_track]:
            frame_work[fi].append((tid, xyxy))
    needed = sorted(frame_work.keys())
    print(f"Unique frames to read: {len(needed)}")

    crops_by_tid = defaultdict(list)
    torso_by_tid = defaultdict(list)
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
            if keep_torso_crops:
                torso_by_tid[tid].append(torso)
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
            "torso_crops": torso_by_tid.get(tid, []) if keep_torso_crops else [],
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


# ---------------------------------------------------------------------------
# Pluggable team engines
# ---------------------------------------------------------------------------
# Each engine takes the per-track sample dict (output of
# `sample_jersey_colors`) and returns the same 4-tuple as the original
# `classify_teams`:
#     (team_of, team_centers_bgr, votes_by_tid, margin)
# so the rest of the pipeline (preview rendering + JSON output) is
# engine-agnostic. Add a new engine by subclassing TeamEngine and
# registering it in TEAM_ENGINES below.
class TeamEngine:
    """Base class — `cluster_tracks` returns the standard 4-tuple.
    `name` and `needs_torso_crops` drive the dispatcher in run()."""
    name = "base"
    needs_torso_crops = False
    def cluster_tracks(self, samples_by_tid, fit_tids):
        raise NotImplementedError


class HSVEngine(TeamEngine):
    """Default engine: per-crop k=2 k-means on dominant torso colours
    (HSV by default; --space bgr also supported). Centroids fit on
    skater crops only; goalies + low-confidence tracks classified
    post-hoc against those centroids. Behaviour-equivalent to the
    pre-refactor `classify_teams` call."""
    name = "hsv"
    needs_torso_crops = False
    def __init__(self, space="hsv"):
        self.space = space
    def cluster_tracks(self, samples_by_tid, fit_tids):
        return classify_teams(samples_by_tid, k=2, space=self.space,
                              fit_tids=fit_tids)


# Engines below are registered as stubs and wired up incrementally —
# OSNet (next commit), SigLIP (after that), Contrastive (last). Each
# raises a clear NotImplementedError until shipped so a `--team-engine`
# typo doesn't silently fall back to HSV.
class OSNetEngine(TeamEngine):
    """OSNet x0_25 medoid embedding per track + k=2 k-means on the
    skater medoids. Goalies + tracks below the fit set get classified
    post-hoc against the resulting two centroids. Same OSNet model
    Stage 3.a uses for entity Re-ID, but clustered at k=2 (team) here
    instead of k≈12 (entity) there. Captures both colour and torso
    pattern, so it tends to separate dark-vs-dark teams that a pure
    HSV k-means struggles with."""
    name = "osnet"
    needs_torso_crops = True
    OSNET_MODEL = "osnet_x0_25"

    def __init__(self):
        self._extractor = None

    def _get_extractor(self):
        # Lazy-load: importing torchreid pulls in tensorboard etc., so
        # only do it when this engine is actually selected.
        if self._extractor is None:
            from torchreid.reid.utils import FeatureExtractor
            self._extractor = FeatureExtractor(
                model_name=self.OSNET_MODEL, model_path="",
                device=pick_device(),
            )
        return self._extractor

    def _medoid(self, feats):
        if len(feats) == 0:
            return None
        arr = np.stack(feats)
        arr = arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9)
        sim = arr @ arr.T
        return arr[int(np.argmax(sim.sum(axis=1)))]

    def cluster_tracks(self, samples_by_tid, fit_tids):
        extractor = self._get_extractor()

        # Per-track medoid embedding from the kept torso crops. Tracks
        # whose torso_crops list is empty (rare — only happens if the
        # caller forgot keep_torso_crops=True) are skipped.
        emb_by_tid = {}
        for tid, info in samples_by_tid.items():
            crops = [c for c in info.get("torso_crops", []) if c is not None]
            if not crops:
                continue
            with torch.no_grad():
                feats = extractor(crops).cpu().numpy()
            med = self._medoid(feats)
            if med is not None:
                emb_by_tid[tid] = med

        if not emb_by_tid:
            empty_of = {tid: 0 for tid in samples_by_tid}
            empty_votes = {
                tid: [len(d["crop_colors"]), 0]
                for tid, d in samples_by_tid.items()
            }
            return empty_of, [(128, 128, 128), (128, 128, 128)], empty_votes, 0.0

        fit_tid_list = sorted([t for t in fit_tids if t in emb_by_tid])
        if len(fit_tid_list) < 2:
            fit_tid_list = sorted(emb_by_tid.keys())

        fit_X = np.stack([emb_by_tid[t] for t in fit_tid_list]).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        _, fit_labels, centers = cv2.kmeans(
            fit_X, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS,
        )
        fit_labels = fit_labels.flatten()

        # Classify ALL tracks (incl. goalies) against the centroids.
        all_tids = sorted(emb_by_tid.keys())
        all_X = np.stack([emb_by_tid[t] for t in all_tids]).astype(np.float32)
        dists = ((all_X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        all_labels = dists.argmin(axis=1)
        team_of = {t: int(lbl) for t, lbl in zip(all_tids, all_labels)}

        # Tracks with no usable embedding land in team 0 with zero confidence
        # (consistent with the HSV engine's empty-vote shape).
        for tid in samples_by_tid:
            team_of.setdefault(tid, 0)

        # Vote distribution for the JSON: derive a pseudo-vote count so
        # downstream entity merging (Stage 3.a) can still apply its
        # vote_confidence floor uniformly. Use 1.0 for the assigned team
        # and 0.0 for the other when the track has an embedding;
        # tracks without one get [0, 0].
        votes_by_tid = {}
        for tid in samples_by_tid:
            v = [0, 0]
            if tid in all_tids:
                v[team_of[tid]] = max(1, len(samples_by_tid[tid]["crop_colors"]))
            votes_by_tid[tid] = v

        # Approximate the visual team centre by averaging the dominant
        # colours of the tracks in each cluster — gives the preview PNG
        # a recognisable swatch even though the engine itself didn't use
        # the colour stat. Skater-only mean keeps goalie pads from
        # pulling the swatch towards grey.
        team_centers = []
        for ci in range(2):
            colours = []
            for tid in fit_tid_list:
                if team_of.get(tid) == ci:
                    colours.extend(samples_by_tid[tid]["crop_colors"])
            if colours:
                arr = np.array(colours, dtype=np.float32)
                team_centers.append(tuple(int(c) for c in arr.mean(axis=0)))
            else:
                team_centers.append((128, 128, 128))

        # Margin = inter-centroid distance / mean intra-cluster spread,
        # in OSNet embedding space. Comparable order-of-magnitude to the
        # HSV engine's margin (both bounded ~0..3).
        inter = float(np.linalg.norm(centers[0] - centers[1]))
        spreads = []
        for ci in range(2):
            pts = fit_X[fit_labels == ci]
            if len(pts) > 0:
                spreads.append(float(np.mean(np.linalg.norm(
                    pts - centers[ci], axis=1))))
        margin = inter / (float(np.mean(spreads)) + 1e-6) if spreads else 0.0
        return team_of, team_centers, votes_by_tid, margin


class SigLIPEngine(TeamEngine):
    """Roboflow-style team classifier: SigLIP vision encoder
    (google/siglip-base-patch16-224, ~370 MB) → mean-pool patch
    tokens → UMAP to 3-D (better-conditioned for k-means on a high-
    dimensional manifold than raw pooled features) → k=2 k-means.
    Centroids fit on skater crops only; goalies + low-confidence
    tracks classified post-hoc against those centroids. Per-track
    label = majority vote over the track's crops."""
    name = "siglip"
    needs_torso_crops = True
    SIGLIP_ID = "google/siglip-base-patch16-224"
    UMAP_DIM = 3

    def __init__(self):
        self._processor = None
        self._model = None
        self._device = None

    def _get_model(self):
        # Lazy-load to avoid the ~370 MB download when --team-engine
        # is something else.
        if self._model is None:
            from transformers import AutoProcessor, SiglipVisionModel
            self._device = pick_device()
            self._processor = AutoProcessor.from_pretrained(self.SIGLIP_ID)
            self._model = SiglipVisionModel.from_pretrained(
                self.SIGLIP_ID).to(self._device).eval()
        return self._processor, self._model, self._device

    def _encode(self, crops_bgr):
        from PIL import Image
        processor, model, device = self._get_model()
        # SigLIP wants RGB PIL Images. Convert in batches to keep peak
        # memory bounded; OSNet's internal batching is already enough
        # for our crop counts but SigLIP weights are larger.
        feats = []
        BATCH = 32
        for i in range(0, len(crops_bgr), BATCH):
            batch = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
                     for c in crops_bgr[i:i + BATCH] if c is not None]
            if not batch:
                continue
            with torch.no_grad():
                inputs = processor(images=batch, return_tensors="pt").to(device)
                out = model(**inputs)
                # last_hidden_state shape: (B, T, D) — mean-pool the
                # T patch tokens, no CLS in SigLIP vision tower.
                pooled = out.last_hidden_state.mean(dim=1).cpu().numpy()
            feats.extend(pooled)
        return np.stack(feats) if feats else np.zeros((0, 768), dtype=np.float32)

    def cluster_tracks(self, samples_by_tid, fit_tids):
        # Encode every kept crop; remember which (tid, idx) each row
        # belongs to so we can majority-vote per track at the end.
        all_crops = []
        all_owners = []  # parallel list of tid
        for tid, info in samples_by_tid.items():
            for c in info.get("torso_crops", []):
                if c is None:
                    continue
                all_crops.append(c)
                all_owners.append(tid)
        if not all_crops:
            empty_of = {tid: 0 for tid in samples_by_tid}
            empty_votes = {tid: [len(d["crop_colors"]), 0]
                           for tid, d in samples_by_tid.items()}
            return empty_of, [(128, 128, 128), (128, 128, 128)], empty_votes, 0.0

        feats = self._encode(all_crops)

        # UMAP to 3-D — Roboflow recipe. UMAP preserves local structure
        # better than PCA on these high-dim semantic features.
        try:
            import umap
            reducer = umap.UMAP(
                n_components=self.UMAP_DIM, n_neighbors=15, min_dist=0.0,
                random_state=0, metric="cosine",
            )
            X = reducer.fit_transform(feats)
        except ImportError:
            # Fall back to PCA if umap-learn isn't installed; the
            # quality drops but the engine still runs.
            from sklearn.decomposition import PCA
            X = PCA(n_components=self.UMAP_DIM, random_state=0).fit_transform(feats)

        X = X.astype(np.float32)

        # Fit k=2 on skater crops only.
        fit_mask = np.array([t in fit_tids for t in all_owners])
        if fit_mask.sum() < 2:
            fit_mask = np.ones(len(all_owners), dtype=bool)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        _, _, centers = cv2.kmeans(
            X[fit_mask], 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS,
        )
        dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)

        votes_by_tid = {tid: [0, 0] for tid in samples_by_tid}
        for tid, lbl in zip(all_owners, labels):
            votes_by_tid[tid][int(lbl)] += 1
        team_of = {tid: int(np.argmax(v)) if sum(v) > 0 else 0
                   for tid, v in votes_by_tid.items()}

        # Visual swatches from the colour stat (same trick as OSNet).
        team_centers = []
        for ci in range(2):
            colours = []
            for tid in fit_tids:
                if team_of.get(tid) == ci:
                    colours.extend(samples_by_tid[tid]["crop_colors"])
            if colours:
                arr = np.array(colours, dtype=np.float32)
                team_centers.append(tuple(int(c) for c in arr.mean(axis=0)))
            else:
                team_centers.append((128, 128, 128))

        # Margin in the reduced space.
        inter = float(np.linalg.norm(centers[0] - centers[1]))
        spreads = []
        for ci in range(2):
            pts = X[fit_mask][labels[fit_mask] == ci]
            if len(pts) > 0:
                spreads.append(float(np.mean(np.linalg.norm(
                    pts - centers[ci], axis=1))))
        margin = inter / (float(np.mean(spreads)) + 1e-6) if spreads else 0.0
        return team_of, team_centers, votes_by_tid, margin


class ContrastiveEngine(TeamEngine):
    """Koshkina 2021 — small CNN trained with triplet loss + 50 %
    grayscale augmentation on torso crops, so the embedding learns
    configural cues (jersey shape / pattern) instead of just colour.
    The trained checkpoint is produced by
    tools/finetune_contrastive_team.py and loaded here at inference.

    Inference pipeline: for every crop of every track, preprocess the
    same way the trainer did (letterbox to 64×128, per-channel intensity
    stretch, NO grayscale at inference), embed with the small CNN, then
    take the per-track L2-medoid. Skater medoids drive a k=2 k-means;
    goalies + low-confidence tracks classified post-hoc against the
    centroids — same shape as the OSNet engine, just a different
    embedding space optimised explicitly for team separation."""
    name = "contrastive"
    needs_torso_crops = True

    def __init__(self, checkpoint_path=None):
        self.checkpoint_path = (Path(checkpoint_path)
                                if checkpoint_path
                                else Path("models/contrastive_team_rilh.pt"))
        self._model = None
        self._device = None
        self._crop_size = (64, 128)   # (W, H) — overridden by checkpoint meta

    def _load(self):
        if self._model is not None:
            return self._model
        if not self.checkpoint_path.exists():
            raise SystemExit(
                f"Contrastive checkpoint not found: {self.checkpoint_path}\n"
                f"Train it first via tools/finetune_contrastive_team.py")
        payload = torch.load(str(self.checkpoint_path),
                             map_location="cpu", weights_only=False)
        arch = payload.get("arch", {})
        # Re-build the architecture from tools/finetune_contrastive_team.py
        # locally to avoid an import cycle (tools/ depends on src/, not the
        # other way around).
        import torch.nn as nn
        import torch.nn.functional as F

        class ContrastiveTeamNet(nn.Module):
            def __init__(self, emb_dim):
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
                    nn.Dropout(0.0),
                    nn.Linear(256, emb_dim),
                )
            def forward(self, x):
                z = self.head(self.features(x))
                return F.normalize(z, p=2, dim=1)

        self._model = ContrastiveTeamNet(arch.get("emb_dim", 128))
        self._model.load_state_dict(payload["state_dict"])
        self._model.eval()
        self._device = pick_device()
        self._model = self._model.to(self._device)
        self._crop_size = (arch.get("crop_w", 64), arch.get("crop_h", 128))
        return self._model

    def _preprocess(self, bgr):
        """Mirror tools/finetune_contrastive_team.py's preprocess: letterbox
        + per-channel intensity stretch. Inference is always full-colour
        (no grayscale aug)."""
        crop_w, crop_h = self._crop_size
        h, w = bgr.shape[:2]
        cur_ar = w / max(h, 1)
        target_ar = crop_w / crop_h
        if cur_ar > target_ar:
            new_w, new_h = crop_w, max(1, int(crop_w / cur_ar))
        else:
            new_w, new_h = max(1, int(crop_h * cur_ar)), crop_h
        resized = cv2.resize(bgr, (new_w, new_h))
        canvas = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        yy, xx = (crop_h - new_h) // 2, (crop_w - new_w) // 2
        canvas[yy:yy + new_h, xx:xx + new_w] = resized

        arr = canvas.astype(np.float32)
        for ci in range(3):
            c = arr[..., ci]
            lo, hi = float(c.min()), float(c.max())
            if hi - lo > 1e-3:
                arr[..., ci] = (c - lo) * (255.0 / (hi - lo))
        arr = arr / 255.0
        arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr).float()

    def _medoid(self, feats):
        if len(feats) == 0:
            return None
        arr = np.stack(feats)
        sim = arr @ arr.T
        return arr[int(np.argmax(sim.sum(axis=1)))]

    def cluster_tracks(self, samples_by_tid, fit_tids):
        model = self._load()

        emb_by_tid = {}
        for tid, info in samples_by_tid.items():
            crops = [c for c in info.get("torso_crops", []) if c is not None]
            if not crops:
                continue
            tensors = torch.stack([self._preprocess(c) for c in crops]).to(self._device)
            with torch.no_grad():
                feats = model(tensors).cpu().numpy()
            med = self._medoid(feats)
            if med is not None:
                emb_by_tid[tid] = med

        if not emb_by_tid:
            empty_of = {tid: 0 for tid in samples_by_tid}
            empty_votes = {tid: [len(d["crop_colors"]), 0]
                           for tid, d in samples_by_tid.items()}
            return empty_of, [(128, 128, 128), (128, 128, 128)], empty_votes, 0.0

        fit_tid_list = sorted([t for t in fit_tids if t in emb_by_tid])
        if len(fit_tid_list) < 2:
            fit_tid_list = sorted(emb_by_tid.keys())
        fit_X = np.stack([emb_by_tid[t] for t in fit_tid_list]).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        _, fit_labels, centers = cv2.kmeans(
            fit_X, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS,
        )
        fit_labels = fit_labels.flatten()

        all_tids = sorted(emb_by_tid.keys())
        all_X = np.stack([emb_by_tid[t] for t in all_tids]).astype(np.float32)
        dists = ((all_X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        all_labels = dists.argmin(axis=1)
        team_of = {t: int(lbl) for t, lbl in zip(all_tids, all_labels)}
        for tid in samples_by_tid:
            team_of.setdefault(tid, 0)

        votes_by_tid = {tid: [0, 0] for tid in samples_by_tid}
        for tid in samples_by_tid:
            if tid in all_tids:
                votes_by_tid[tid][team_of[tid]] = max(
                    1, len(samples_by_tid[tid]["crop_colors"]))

        team_centers = []
        for ci in range(2):
            colours = []
            for tid in fit_tid_list:
                if team_of.get(tid) == ci:
                    colours.extend(samples_by_tid[tid]["crop_colors"])
            if colours:
                arr = np.array(colours, dtype=np.float32)
                team_centers.append(tuple(int(c) for c in arr.mean(axis=0)))
            else:
                team_centers.append((128, 128, 128))

        inter = float(np.linalg.norm(centers[0] - centers[1]))
        spreads = []
        for ci in range(2):
            pts = fit_X[fit_labels == ci]
            if len(pts) > 0:
                spreads.append(float(np.mean(np.linalg.norm(
                    pts - centers[ci], axis=1))))
        margin = inter / (float(np.mean(spreads)) + 1e-6) if spreads else 0.0
        return team_of, team_centers, votes_by_tid, margin


TEAM_ENGINES = {
    "hsv":         lambda args: HSVEngine(space=args.space),
    "osnet":       lambda args: OSNetEngine(),
    "siglip":      lambda args: SigLIPEngine(),
    "contrastive": lambda args: ContrastiveEngine(
                       checkpoint_path=args.contrastive_checkpoint),
}


def apply_ref_classifier(samples_by_tid, ckpt_path):
    """Post-process the team engine's output: tag each track with an
    `is_referee` flag using a small MLP head trained on OSNet
    embeddings (see tools/finetune_ref_classifier.py). Per-track vote
    is the mean sigmoid score over the track's crops; threshold 0.5.
    Returns ``{tid: {"is_referee": bool, "ref_score": float}}``.
    Tracks with no usable torso crops get is_referee=False, score=0.0.
    """
    import torch.nn as nn
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    arch = payload["arch"]

    class RefHead(nn.Module):
        def __init__(self, emb_dim, hidden):
            super().__init__()
            self.fc1 = nn.Linear(emb_dim, hidden)
            self.fc2 = nn.Linear(hidden, 1)
            self.drop = nn.Dropout(0.0)   # eval-only — drop is identity
        def forward(self, x):
            return self.fc2(self.drop(torch.relu(self.fc1(x)))).squeeze(-1)

    model = RefHead(arch["emb_dim"], arch["hidden"])
    model.load_state_dict(payload["state_dict"])
    model.eval()

    from torchreid.reid.utils import FeatureExtractor
    extractor = FeatureExtractor(
        model_name="osnet_x0_25", model_path="", device=pick_device(),
    )

    out = {}
    for tid, info in samples_by_tid.items():
        crops = [c for c in info.get("torso_crops", []) if c is not None]
        if not crops:
            out[tid] = {"is_referee": False, "ref_score": 0.0}
            continue
        with torch.no_grad():
            feats = extractor(crops).cpu().numpy()
        feats = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-9)
        with torch.no_grad():
            logits = model(torch.from_numpy(feats.astype(np.float32)))
            scores = torch.sigmoid(logits).numpy()
        mean_score = float(scores.mean())
        out[tid] = {"is_referee": bool(mean_score > 0.5),
                    "ref_score": mean_score}
    return out


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
        pose_imgsz, preview_cols, space, multi_grid, engine,
        ref_classifier_path=None):
    """Run the team-classification pipeline end-to-end.

    Loads ``p1_a_detections.json``, samples crops from the source video,
    dispatches to the requested team engine (``engine.cluster_tracks``),
    and writes ``p1_b_teams.json`` plus a debug ``teams_preview.png``.
    The engine sees both the dominant-colour stat (used by HSV) and the
    raw torso BGR crops (used by embedding engines), set by
    ``engine.needs_torso_crops``.
    """
    device = pick_device()
    print(f"Device: {device}")
    print(f"Team engine: {engine.name}")

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
        keep_torso_crops=engine.needs_torso_crops,
    )
    n_crops = sum(len(v["crop_colors"]) for v in crops_by_tid.values())
    print(f"  colors extracted for {len(crops_by_tid)}/{len(all_tids)} tracks "
          f"({n_crops} crops total)")

    fit_tids = skater_tids & set(crops_by_tid.keys())
    print(f"Clustering teams ({engine.name}, fitting on {len(fit_tids)} "
          f"skater tracks, classifying all)…")
    team_of, centers, votes_by_tid, margin = engine.cluster_tracks(
        crops_by_tid, fit_tids,
    )

    # Optional referee binary classifier — post-hoc, orthogonal to the
    # team engine. Tags each track with is_referee + ref_score.
    ref_results = {}
    if ref_classifier_path is not None:
        if not engine.needs_torso_crops:
            # The HSV engine doesn't keep torso crops by default. Re-sample
            # them so the classifier has something to look at; cheap given
            # we already loaded the pose model + decoded the frames once.
            print(f"Re-sampling torso crops for ref classifier "
                  f"(engine {engine.name} didn't keep them)…")
            crops_by_tid_with_torso = sample_jersey_colors(
                detections_data, video_path, pose_model, device,
                samples_per_track, pose_imgsz, multi_grid,
                keep_torso_crops=True,
            )
        else:
            crops_by_tid_with_torso = crops_by_tid
        print(f"Running ref classifier ({ref_classifier_path})…")
        ref_results = apply_ref_classifier(
            crops_by_tid_with_torso, ref_classifier_path,
        )
        n_refs = sum(1 for r in ref_results.values() if r["is_referee"])
        print(f"  Tagged {n_refs}/{len(ref_results)} tracks as referee")

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
        "version": "v3",
        "method": f"pose_torso + engine={engine.name}",
        "team_engine": engine.name,
        "samples_per_track": samples_per_track,
        "color_space": space,
        "multi_point_grid": list(multi_grid),
        "roster_note": (
            "Roller inline hockey: 4 skaters + 1 goalie per team, 1–2 refs. "
            "Refs leak into clustering when Phase 1 used COCO (HockeyAI drops them)."
        ),
        "team_centers_bgr": [list(c) for c in centers],
        "cluster_margin": margin,
        "ref_classifier": str(ref_classifier_path) if ref_classifier_path else None,
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
                "is_referee": ref_results.get(tid, {}).get("is_referee", False),
                "ref_score": ref_results.get(tid, {}).get("ref_score", 0.0),
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
                   help="Color space for HSV-engine k-means (default hsv). "
                        "Ignored by other engines.")
    p.add_argument("--grid", type=str, default="3x2",
                   help="Multi-point grid inside the torso (rows x cols). "
                        "3x2 = 6 sub-regions.")
    p.add_argument("--team-engine", choices=list(TEAM_ENGINES.keys()),
                   default="hsv",
                   help="Team-classification backend. "
                        "hsv: per-crop k-means on dominant torso colour (default, "
                        "behaviour-equivalent to v2). "
                        "osnet: k=2 on OSNet x0_25 medoid embeddings. "
                        "siglip: SigLIP encode + UMAP + k-means (Roboflow recipe). "
                        "contrastive: Koshkina-style triplet model "
                        "(--contrastive-checkpoint required).")
    p.add_argument("--contrastive-checkpoint", type=str, default=None,
                   help="Path to a contrastive-team checkpoint (used only when "
                        "--team-engine contrastive). Default location: "
                        "models/contrastive_team_rilh.pt")
    p.add_argument("--ref-classifier", type=str, default=None,
                   help="Optional path to a referee binary classifier "
                        "(produced by tools/finetune_ref_classifier.py). When "
                        "set, every track is tagged with is_referee + "
                        "ref_score in p1_b_teams.json. Orthogonal to "
                        "--team-engine — runs after the team clustering.")
    args = p.parse_args()

    try:
        rows, cols = (int(x) for x in args.grid.lower().split("x"))
    except Exception as e:
        raise SystemExit(f"Invalid --grid {args.grid!r}: expected 'RxC'") from e

    detections_json = Path(args.detections_json)
    video = Path(args.video)
    output = (Path(args.output) if args.output
              else detections_json.with_name("p1_b_teams.json"))

    engine = TEAM_ENGINES[args.team_engine](args)

    run(detections_json, video, output,
        args.samples_per_track, args.pose_model, args.pose_imgsz,
        args.preview_cols, args.space, (rows, cols), engine,
        ref_classifier_path=Path(args.ref_classifier) if args.ref_classifier else None)


if __name__ == "__main__":
    main()
