"""
RILH-AI-Vision — p3_b_annotate (Phase 3 stage b — final annotated MP4)
Render the final annotated MP4 from upstream stage outputs:
  - BoxAnnotator + LabelAnnotator + TraceAnnotator from supervision
  - One fixed color per team (green / blue), inherited from P1.b
  - Per-track label : `t{id} {G|S} #NN` (track id always shown, role
    G/S from is_goaltender, jersey number from P1.c via P3.a
    entity rollup; `#??` if no number identified)
  - Puck rendered as a dark gray bbox (no label) + short trace

Annotate lives in Phase 3 (alongside entity clustering) because it
needs the entity rollup to colour every track of one player with the
same team colour and number. Phase 2 (rink) runs before Phase 3 so
that future versions of entity clustering can reject off-ice tracks
geometrically.

Inputs : p1_a_detections.json (P1.a), p1_c_numbers.json (P1.c), the source
         video; auto-discovers p1_b_teams.json (P1.b) and p3_a_entities.json
         (P3.a) next to p1_a_detections.json if present.
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
REF_COLOR = sv.Color(r=10, g=10, b=10)   # black — referees, when --ref-classifier ran
GOALIE_DARKEN = 0.55                      # darken team colour for goalies


def _darken(color: sv.Color, factor: float = GOALIE_DARKEN) -> sv.Color:
    """Return a darker variant of a supervision Color (used to mark
    goalies with the same team hue as their teammates)."""
    return sv.Color(
        r=max(0, int(color.r * factor)),
        g=max(0, int(color.g * factor)),
        b=max(0, int(color.b * factor)),
    )


GOALIE_TEAM_COLORS = [_darken(c) for c in TEAM_COLORS]


def draw_multiline_label(frame, lines, anchor_xy, bg_bgr,
                         text_scale=0.5, text_thickness=1, pad=2):
    """Draw a 2-line label with an opaque rectangular background just
    above the bbox. anchor_xy is the bbox top-left corner; the label
    flows upward from there. Each line gets its own row, sized by
    cv2.getTextSize so the bg matches the actual text bounds.

    Used for players (team-coloured bg + white text). Refs share the
    same code path with a black bg.
    """
    if not lines:
        return frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    sizes = [cv2.getTextSize(t, font, text_scale, text_thickness)[0]
             for t in lines]
    line_h = max(s[1] for s in sizes) + 4  # vertical step per line
    box_w = max(s[0] for s in sizes) + 2 * pad
    box_h = line_h * len(lines) + pad
    x0 = int(anchor_xy[0])
    # Anchor the bottom of the label box flush with the top of the bbox.
    y1 = int(anchor_xy[1]) - 1
    y0 = y1 - box_h
    h, w = frame.shape[:2]
    if y0 < 0:
        # Bbox is at the top of the frame — flip the label inside the bbox
        y0 = int(anchor_xy[1]) + 1
        y1 = y0 + box_h
    x1 = min(w, x0 + box_w)
    x0 = max(0, x0)
    y0 = max(0, min(h - box_h - 1, y0))
    y1 = y0 + box_h
    cv2.rectangle(frame, (x0, y0), (x1, y1), bg_bgr, thickness=-1)
    text_x = x0 + pad
    for i, t in enumerate(lines):
        baseline_y = y0 + pad + (i + 1) * line_h - 4
        cv2.putText(frame, t, (text_x, baseline_y), font, text_scale,
                    (255, 255, 255), text_thickness, cv2.LINE_AA)
    return frame


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
    """Per-track jersey color (used as a fallback when p1_b_teams.json is
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
           per_track_referee=None,
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
    per_track_referee = per_track_referee or {}

    # Stage 1.a writes one record per processed source frame, with
    # `frame` set to the SOURCE index. When detection ran at a lower
    # fps than the source (stride > 1), the gap frames have no
    # detection — replicate each record's boxes onto the next
    # `stride - 1` source frames so the rendered video stays smooth
    # instead of flashing every other frame.
    stride = max(1, int(detections_data.get("stride", 1)))
    frames_map = {}
    for fr in detections_data["frames"]:
        for offset in range(stride):
            frames_map[fr["frame"] + offset] = fr["boxes"]

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

    def referee_for(tid_i):
        """Per-track is_referee from p1_b_teams.json (set only when
        Stage 1.b ran with --ref-classifier). Refs render in black so
        they read at a glance and don't pollute a team's colour."""
        return bool(per_track_referee.get(tid_i, False))

    # One annotator pair per visual category (box + trace only).
    # Labels are drawn manually (see draw_multiline_label) so that
    # they can span 2 lines (`e{eid}·t{tid}` then `S #NN`) — supervision's
    # LabelAnnotator collapses newlines into a single line.
    # Five categories total: team 0 skater (green) / team 0 goalie (dark
    # green) / team 1 skater (blue) / team 1 goalie (dark blue) / referee
    # (black). supervision binds colour at construction time so each
    # category needs its own annotator pair.
    def _make_pair(color, trace_len=30):
        return {
            "box": sv.BoxAnnotator(color=color, thickness=2),
            "trace": sv.TraceAnnotator(color=color, thickness=2,
                                       trace_length=trace_len),
        }

    annotators_by_key = {
        # ("team", team_id, is_goalie)
        ("team", 0, False): _make_pair(TEAM_COLORS[0]),
        ("team", 0, True):  _make_pair(GOALIE_TEAM_COLORS[0]),
        ("team", 1, False): _make_pair(TEAM_COLORS[1]),
        ("team", 1, True):  _make_pair(GOALIE_TEAM_COLORS[1]),
        ("ref",):           _make_pair(REF_COLOR),
    }
    # BGR for label backgrounds (cv2.rectangle takes BGR tuples)
    color_bgr_by_key = {
        ("team", 0, False): (TEAM_COLORS[0].b, TEAM_COLORS[0].g, TEAM_COLORS[0].r),
        ("team", 0, True):  (GOALIE_TEAM_COLORS[0].b, GOALIE_TEAM_COLORS[0].g, GOALIE_TEAM_COLORS[0].r),
        ("team", 1, False): (TEAM_COLORS[1].b, TEAM_COLORS[1].g, TEAM_COLORS[1].r),
        ("team", 1, True):  (GOALIE_TEAM_COLORS[1].b, GOALIE_TEAM_COLORS[1].g, GOALIE_TEAM_COLORS[1].r),
        ("ref",):           (REF_COLOR.b, REF_COLOR.g, REF_COLOR.r),
    }
    # Puck — drawn as a CIRCLE (cv2.circle) instead of a bbox so it
    # reads as a different object class than the player rectangles.
    # Same dark-gray colour, plus a short trace by reusing the
    # supervision TraceAnnotator (centroid-based, so the box-vs-circle
    # difference doesn't matter for the trace).
    puck_color_bgr = (PUCK_COLOR.b, PUCK_COLOR.g, PUCK_COLOR.r)
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

        # Bucket each box into one of the 5 visual categories.
        # Referee tag (when present) takes precedence over team colour
        # so refs always render black, regardless of which team the
        # engine assigned them to.
        by_key = defaultdict(list)
        for b in player_boxes:
            tid = b["track_id"]
            if referee_for(tid):
                by_key[("ref",)].append(b)
            else:
                by_key[("team", team_for(tid), goalie_for(tid))].append(b)

        for key, boxes in by_key.items():
            if not boxes:
                continue
            dets = build_detections(boxes)
            a = annotators_by_key[key]
            bg_bgr = color_bgr_by_key[key]
            frame = a["box"].annotate(scene=frame, detections=dets)
            frame = a["trace"].annotate(scene=frame, detections=dets)
            # Two-line custom label per detection. Line 1 = identity
            # (`e{eid}·t{tid}` for players, `t{tid}` for refs). Line 2
            # = role + jersey number (`S #14`) for players, `REF` for
            # refs. Anchored just above the bbox top-left corner; the
            # bg uses the team colour so the bbox + label match.
            for b in boxes:
                ti = int(b["track_id"])
                x1, y1, x2, y2 = b["xyxy"]
                if key[0] == "ref":
                    line1 = f"t{ti}"
                    line2 = "REF"
                else:
                    role = "G" if goalie_for(ti) else "S"
                    num = jersey_for(ti)
                    num_str = f"#{num}" if num else "#??"
                    eid = entity_of_tid.get(ti) if entity_of_tid else None
                    # Team prefix in the label so the same number on
                    # team 0 and team 1 reads as two distinct identities,
                    # even if the colour rendering goes wrong on the
                    # bbox border. Hardcoded T<id>; entity-level team_id
                    # is the team_for() lookup we already do above.
                    tid_team = team_for(ti)
                    line1 = f"e{eid} - t{ti}" if eid is not None else f"t{ti}"
                    line2 = f"{role} T{tid_team} {num_str}"
                frame = draw_multiline_label(
                    frame, [line1, line2], (x1, y1), bg_bgr,
                    text_scale=0.5, text_thickness=1)

        # Puck — drawn as a circle (so it reads as a different object
        # class than the player rectangles). The radius is half the
        # bbox width so the circle hugs the puck the same way the bbox
        # did. Trace stays the same (centroid-based polyline).
        if puck_boxes:
            for b in puck_boxes:
                x1, y1, x2, y2 = b["xyxy"]
                cx = int(round((x1 + x2) / 2))
                cy = int(round((y1 + y2) / 2))
                radius = max(3, int(round(max(x2 - x1, y2 - y1) / 2)))
                cv2.circle(frame, (cx, cy), radius, puck_color_bgr,
                           thickness=2, lineType=cv2.LINE_AA)
            puck_dets = build_detections(puck_boxes)
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
    p.add_argument("detections_json", help="P1.a output (p1_a_detections.json)")
    p.add_argument("numbers_json", help="P1.c output (p1_c_numbers.json)")
    p.add_argument("video", help="Source video")
    p.add_argument("--output", required=True, help="Output MP4 path")
    p.add_argument("--color-samples", type=int, default=6,
                   help="Crops per track used to estimate jersey color (only "
                        "used when p1_b_teams.json is missing)")
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
    per_track_referee = {}
    teams_json_path = detections_json.with_name("p1_b_teams.json")
    if teams_json_path.exists():
        print(f"Using precomputed teams from {teams_json_path}")
        teams_data = json.loads(teams_json_path.read_text())
        team_of = {int(tid): info["team_id"]
                   for tid, info in teams_data["tracks"].items()}
        team_colors = [tuple(c) for c in teams_data["team_centers_bgr"]]
        per_track_goalie = {int(tid): bool(info.get("is_goaltender", False))
                            for tid, info in teams_data["tracks"].items()}
        # Only populated when Stage 1.b ran with --ref-classifier;
        # otherwise every track silently falls back to is_referee=False.
        per_track_referee = {int(tid): bool(info.get("is_referee", False))
                             for tid, info in teams_data["tracks"].items()}
        n_refs = sum(1 for v in per_track_referee.values() if v)
        if n_refs:
            print(f"  {n_refs} tracks tagged as referee (will render black)")
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
    entities_json_path = detections_json.with_name("p3_a_entities.json")
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
           per_track_referee=per_track_referee,
           debug_frames_dir=debug_frames_dir,
           debug_frames_step=args.debug_frames_step)


if __name__ == "__main__":
    main()
