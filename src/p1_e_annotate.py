"""
RILH-AI-Vision — p1_e_annotate
Render the final annotated MP4 from upstream stage outputs:
  - BoxAnnotator + LabelAnnotator + TraceAnnotator from supervision
  - One fixed color per team (green / blue), inherited from P1.b
  - Per-track label : `t{id} {G|S} #NN` (track id always shown, role
    G/S from is_goaltender, jersey number from P1.c via P1.d
    entity rollup; `#??` if no number identified)
  - Puck rendered as a dark gray bbox (no label) + short trace

Inputs : p1_detections.json (P1.a), p1_numbers.json (P1.c), the source
         video; auto-discovers p1_teams.json (P1.b) and p1_entities.json
         (P1.d) next to p1_detections.json if present.
Output : an annotated MP4 at the given --output path.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

PERSON_CLASS = 0
PUCK_CLASS = 32

TEAM_COLORS = [
    sv.Color(r=0, g=220, b=0),     # green — team 0
    sv.Color(r=30, g=120, b=255),  # blue  — team 1
]
PUCK_COLOR = sv.Color(r=60, g=60, b=60)  # dark gray — stands out on ice


# --- Jersey color sampling & team clustering ---------------------------------

def torso_crop_from_bbox(frame, xyxy):
    """Upper-chest region of a bbox — 10%-40% of its height."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    bh = y2 - y1
    ty1 = max(0, int(y1 + 0.10 * bh))
    ty2 = min(h, int(y1 + 0.40 * bh))
    tx1 = max(0, int(x1))
    tx2 = min(w, int(x2))
    if ty2 <= ty1 + 4 or tx2 <= tx1 + 4:
        return None
    return frame[ty1:ty2, tx1:tx2]


def dominant_bgr(bgr_crop):
    """Largest-cluster BGR of a crop, filtering low-sat / too-dark pixels."""
    if bgr_crop is None or bgr_crop.size == 0:
        return None
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1]
    val = hsv[..., 2]
    mask = (val > 40) & (val < 240)
    sat_mask = mask & (sat > 30)
    if sat_mask.sum() >= 40:
        mask = sat_mask
    elif mask.sum() < 40:
        return None
    pixels = bgr_crop[mask].astype(np.float32)
    k = 3 if len(pixels) >= 30 else 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3,
                                     cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    biggest = int(np.argmax(counts))
    return tuple(int(c) for c in centers[biggest])


def sample_track_colors(tracks_data, video_path, max_samples=6):
    """Per-track jersey color (used as a fallback when p1_teams.json is
    missing). Samples ``max_samples`` highest-conf detections per track,
    crops the upper-chest, returns the median dominant BGR per track."""
    by_tid = defaultdict(list)
    for fr in tracks_data["frames"]:
        fi = fr["frame"]
        for b in fr["boxes"]:
            if b["class_id"] != PERSON_CLASS or b["track_id"] < 0:
                continue
            by_tid[b["track_id"]].append((fi, b["xyxy"], b["conf"]))

    needed_by_frame = defaultdict(list)
    for tid, dets in by_tid.items():
        for fi, xyxy, _ in sorted(dets, key=lambda d: -d[2])[:max_samples]:
            needed_by_frame[fi].append((tid, xyxy))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    max_needed = max(needed_by_frame) if needed_by_frame else -1

    collected = defaultdict(list)
    current = 0
    while current <= max_needed:
        ok, frame = cap.read()
        if not ok:
            break
        if current in needed_by_frame:
            for tid, xyxy in needed_by_frame[current]:
                c = dominant_bgr(torso_crop_from_bbox(frame, xyxy))
                if c is not None:
                    collected[tid].append(c)
        current += 1
    cap.release()

    out = {}
    for tid, colors in collected.items():
        arr = np.array(colors)
        out[tid] = tuple(int(v) for v in np.median(arr, axis=0))
    return out


def cluster_teams(colors_by_tid):
    """k=2 k-means → {tid: team_id}. Tracks without an extracted color fall
    back to team 0 (we could do per-frame color later if it matters)."""
    tids = list(colors_by_tid.keys())
    if len(tids) < 2:
        return {tid: 0 for tid in tids}, [(128, 128, 128), (128, 128, 128)]
    data = np.array([colors_by_tid[t] for t in tids], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(data, 2, None, criteria, 10,
                                     cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()
    team_of = {tid: int(labels[i]) for i, tid in enumerate(tids)}
    team_colors = [tuple(int(c) for c in centers[i]) for i in range(2)]
    return team_of, team_colors


# --- Rendering ----------------------------------------------------------------

def build_detections(boxes):
    """Convert a list of ``{xyxy, class_id, track_id, conf}`` dicts into
    a supervision.Detections object that the BoxAnnotator can consume."""
    if not boxes:
        return sv.Detections.empty()
    xyxy = np.array([b["xyxy"] for b in boxes], dtype=np.float32)
    class_id = np.array([b["class_id"] for b in boxes], dtype=int)
    tracker_id = np.array([b["track_id"] for b in boxes], dtype=int)
    confidence = np.array([b["conf"] for b in boxes], dtype=np.float32)
    return sv.Detections(
        xyxy=xyxy,
        class_id=class_id,
        tracker_id=tracker_id,
        confidence=confidence,
    )


def render(detections_data, numbers, team_of, video_path, output,
           entity_of_tid=None, entity_by_id=None,
           per_track_goalie=None,
           debug_frames_dir=None, debug_frames_step=10):
    """Render. If entity_of_tid + entity_by_id are provided (P1.d
    output), each merged track inherits its entity's team_id,
    jersey_number, and is_goaltender so every member of the same entity
    is drawn with the same color + label across frames. Tracks outside
    any entity fall back to per-track team_id (P1.b), per-track
    jersey number (P1.c), and per-track is_goaltender (P1.b).

    If debug_frames_dir is given, every `debug_frames_step`-th frame is
    also written as a PNG to that folder for visual review."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (w, h))

    if debug_frames_dir is not None:
        debug_frames_dir.mkdir(parents=True, exist_ok=True)
        print(f"Debug frames → {debug_frames_dir}/ (1 every {debug_frames_step})")

    numbers_tracks = numbers.get("tracks", {})
    per_track_goalie = per_track_goalie or {}
    frames_map = {fr["frame"]: fr["boxes"] for fr in detections_data["frames"]}

    def team_for(tid_i):
        """Resolve a track id to its team. Prefer the entity-level
        team_id (P1.d) when available; otherwise fall back to the
        per-track team from P1.b."""
        if entity_of_tid is not None:
            eid = entity_of_tid.get(tid_i)
            if eid is not None:
                et = entity_by_id[eid].get("team_id")
                if et is not None:
                    return et
        return team_of.get(tid_i, 0)

    def jersey_for(tid_i):
        """Resolve a track id to its jersey number, preferring the
        entity-level value (P1.d) when present."""
        if entity_of_tid is not None:
            eid = entity_of_tid.get(tid_i)
            if eid is not None:
                return entity_by_id[eid].get("jersey_number")
        info = numbers_tracks.get(str(tid_i))
        return info.get("jersey_number") if info else None

    def goalie_for(tid_i):
        """Resolve a track id to its is_goaltender flag, preferring the
        entity-level value (P1.d) when present."""
        if entity_of_tid is not None:
            eid = entity_of_tid.get(tid_i)
            if eid is not None:
                return bool(entity_by_id[eid].get("is_goaltender", False))
        return bool(per_track_goalie.get(tid_i, False))

    # One annotator trio per team, each locked to its team color
    annotators = []
    for color in TEAM_COLORS:
        annotators.append({
            "box": sv.BoxAnnotator(color=color, thickness=2),
            "label": sv.LabelAnnotator(color=color, text_scale=0.5,
                                       text_thickness=1),
            "trace": sv.TraceAnnotator(color=color, thickness=2,
                                       trace_length=30),
        })
    # Puck annotator — neutral gray box, no label, short trace.
    puck_box = sv.BoxAnnotator(color=PUCK_COLOR, thickness=2)
    puck_trace = sv.TraceAnnotator(color=PUCK_COLOR, thickness=2,
                                   trace_length=20)

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = frames_map.get(fi, [])
        player_boxes = [b for b in boxes
                        if b["class_id"] == PERSON_CLASS and b["track_id"] >= 0]
        puck_boxes = [b for b in boxes
                      if b["class_id"] == PUCK_CLASS and b["track_id"] >= 0]

        by_team = [[], []]
        for b in player_boxes:
            by_team[team_for(b["track_id"])].append(b)

        for team_id, team_boxes in enumerate(by_team):
            if not team_boxes:
                continue
            dets = build_detections(team_boxes)
            labels = []
            for tid in dets.tracker_id:
                ti = int(tid)
                role = "G" if goalie_for(ti) else "S"
                num = jersey_for(ti)
                num_str = f"#{num}" if num else "#??"
                # Label: `t{id} {G|S} #{num}` — track id always shown
                # so the user can call out frame-level bugs by tid.
                labels.append(f"t{ti} {role} {num_str}")
            a = annotators[team_id]
            frame = a["box"].annotate(scene=frame, detections=dets)
            frame = a["label"].annotate(scene=frame, detections=dets,
                                        labels=labels)
            frame = a["trace"].annotate(scene=frame, detections=dets)

        # Puck — gray box + short trace, no label
        if puck_boxes:
            puck_dets = build_detections(puck_boxes)
            frame = puck_box.annotate(scene=frame, detections=puck_dets)
            frame = puck_trace.annotate(scene=frame, detections=puck_dets)

        writer.write(frame)
        if debug_frames_dir is not None and fi % debug_frames_step == 0:
            cv2.imwrite(str(debug_frames_dir / f"frame_{fi:05d}.png"), frame)

        fi += 1
        if fi % 120 == 0:
            pct = 100 * fi / max(total, 1)
            print(f"  rendered {fi}/{total} ({pct:.1f}%)")

    cap.release()
    writer.release()
    print(f"\nDone.\n  Annotated video: {output}")


def main():
    """CLI entry point — parse arguments and dispatch to ``render``."""
    p = argparse.ArgumentParser(
        description="Annotate video: t{id} {G|S} #NN labels + team-coloured boxes"
    )
    p.add_argument("detections_json", help="P1.a output (p1_detections.json)")
    p.add_argument("numbers_json", help="P1.c output (p1_numbers.json)")
    p.add_argument("video", help="Source video")
    p.add_argument("--output", required=True, help="Output MP4 path")
    p.add_argument("--color-samples", type=int, default=6,
                   help="Crops per track used to estimate jersey color (only "
                        "used when p1_teams.json is missing)")
    p.add_argument("--debug-frames-dir", type=str, default=None,
                   help="If set, save 1 annotated frame per --debug-frames-step "
                        "into this folder (PNG). Useful for visual review.")
    p.add_argument("--debug-frames-step", type=int, default=10,
                   help="Sampling stride for debug frames (default: every 10).")
    args = p.parse_args()

    detections_json = Path(args.detections_json)
    numbers_json = Path(args.numbers_json)
    video = Path(args.video)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    detections_data = json.loads(detections_json.read_text())
    numbers = json.loads(numbers_json.read_text())

    per_track_goalie = {}
    teams_json_path = detections_json.with_name("p1_teams.json")
    if teams_json_path.exists():
        print(f"Using precomputed teams from {teams_json_path}")
        teams_data = json.loads(teams_json_path.read_text())
        team_of = {int(tid): info["team_id"]
                   for tid, info in teams_data["tracks"].items()}
        team_colors = [tuple(c) for c in teams_data["team_centers_bgr"]]
        per_track_goalie = {int(tid): bool(info.get("is_goaltender", False))
                            for tid, info in teams_data["tracks"].items()}
    else:
        print("Sampling jersey colors per track…")
        colors = sample_track_colors(detections_data, video, args.color_samples)
        print(f"  colors extracted for {len(colors)} tracks")
        print("Clustering teams (k=2)…")
        team_of, team_colors = cluster_teams(colors)

    print(f"  cluster 0 mean BGR: {team_colors[0]} → GREEN")
    print(f"  cluster 1 mean BGR: {team_colors[1]} → BLUE")
    n0 = sum(1 for t in team_of.values() if t == 0)
    n1 = sum(1 for t in team_of.values() if t == 1)
    print(f"  tracks assigned: team 0 = {n0}, team 1 = {n1}")

    # Prefer entity-level team + jersey when P1.d has been run.
    entities_json_path = detections_json.with_name("p1_entities.json")
    entity_of_tid = None
    entity_by_id = None
    if entities_json_path.exists():
        print(f"Using entities from {entities_json_path}")
        entities_data = json.loads(entities_json_path.read_text())
        entity_by_id = entities_data["entities"]
        entity_of_tid = {}
        for eid, e in entity_by_id.items():
            for tid in e["track_ids"]:
                entity_of_tid[tid] = eid
        print(f"  {len(entity_by_id)} entities covering "
              f"{len(entity_of_tid)} tracks")

    debug_frames_dir = (Path(args.debug_frames_dir)
                        if args.debug_frames_dir else None)

    print("\nRendering annotated video…")
    render(detections_data, numbers, team_of, video, output,
           entity_of_tid=entity_of_tid, entity_by_id=entity_by_id,
           per_track_goalie=per_track_goalie,
           debug_frames_dir=debug_frames_dir,
           debug_frames_step=args.debug_frames_step)


if __name__ == "__main__":
    main()
