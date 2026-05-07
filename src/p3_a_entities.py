"""
RILH-AI-Vision — p3_a_entities (Phase 3 stage a)
Merge fragmented Phase 1 tracks into stable entities (one entity = one
real player / goalie) via post-hoc Re-ID clustering.

Phase 3 follows Phase 2 (rink calibration) so that future versions can
reject off-ice tracks geometrically before clustering. Today the rink
constraint isn't applied yet — Phase 2 is being unblocked separately.

Inputs (three JSONs produced by Phase 1 + the source video):
  - p1_a_detections.json   (p1_a_detect)
  - p1_b_teams.json        (p1_b_teams)        — team_id + vote_conf
  - p1_c_numbers.json      (p1_c_numbers)      — OCR jersey number
  - video.mp4                                — for embedding crops

Output: p3_a_entities.json
  Maps entity_id -> list of merged track_ids, with team_id, is_goaltender,
  jersey_number (if OCR'd), per-entity frame range and coverage, plus a
  list of unmatched singleton tracks.

Signals and constraints (see docs/p3_a_entities_design.md for the rationale):
  - Appearance embedding: OSNet x0_25 medoid over top-conf crops per track
  - Hard constraint — same team_id (from P1.b), both sides with
    vote_confidence >= `--team-conf-floor`
  - Hard constraint — strict temporal non-overlap (0 frames shared)
  - Hard constraint — OCR conflict (same team, different confident
    jersey numbers) blocks the merge
  - Merge weight = cos_sim(emb) + ocr_bonus*same_number + goalie_bonus*both_goalie
  - Greedy merge in descending weight order; stop when weight drops below
    `--sim-threshold` (so non-OCR pairs with poor appearance similarity
    are left as singletons)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torchreid.reid.utils import FeatureExtractor

PERSON_CLASS = 0


def pick_device():
    """Return the best PyTorch device available (mps > cuda > cpu)."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def safe_crop(frame, xyxy, min_w=8, min_h=16):
    """Clamp xyxy to the frame and return the BGR sub-image, or None
    if the resulting region is below the minimum dimensions for OSNet
    (8×16 — Re-ID embeddings on tinier crops are noise)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))
    if x2 <= x1 + min_w or y2 <= y1 + min_h:
        return None
    return frame[y1:y2, x1:x2]


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


def group_detections_by_track(tracks_data):
    """Pivot detections by track id, keeping only PERSON class with a
    real (non-negative) track id. Returns ``{tid: [(frame, xyxy, conf), ...]}``."""
    by_tid = defaultdict(list)
    for fr in tracks_data["frames"]:
        for b in fr["boxes"]:
            if b["class_id"] == PERSON_CLASS and b["track_id"] >= 0:
                by_tid[b["track_id"]].append((fr["frame"], b["xyxy"], b["conf"]))
    return by_tid


def track_frame_sets(tracks_data):
    """tid -> frozenset of frame indices (for exact overlap checking)."""
    by_tid = defaultdict(set)
    for fr in tracks_data["frames"]:
        for b in fr["boxes"]:
            if b["class_id"] == PERSON_CLASS and b["track_id"] >= 0:
                by_tid[b["track_id"]].add(fr["frame"])
    return {tid: frozenset(s) for tid, s in by_tid.items()}


def extract_track_embeddings(tracks_data, video_path, extractor,
                             samples_per_track, batch_size):
    """Return {tid: 512-dim L2-normalized medoid embedding}."""
    by_tid = group_detections_by_track(tracks_data)

    # Per-frame work list
    frame_work = defaultdict(list)
    for tid, dets in by_tid.items():
        for fi, xyxy, _ in sorted(dets, key=lambda d: -d[2])[:samples_per_track]:
            frame_work[fi].append((tid, xyxy))
    needed = sorted(frame_work.keys())
    total = sum(len(v) for v in frame_work.values())
    print(f"Extracting embeddings: {len(needed)} frames, {total} crops…")

    # Batched OSNet inference
    feats_per_tid = defaultdict(list)
    pending_crops = []
    pending_tids = []

    def flush():
        """Run OSNet on the queued crops and accumulate per-tid feature
        vectors. Called when the queue reaches ``batch_size`` and once
        at the end of the frame stream."""
        if not pending_crops:
            return
        with torch.no_grad():
            feats = extractor(pending_crops).cpu().numpy()  # (N, 512)
        for tid, feat in zip(pending_tids, feats):
            feats_per_tid[tid].append(feat)
        pending_crops.clear()
        pending_tids.clear()

    done = 0
    for fi, frame in stream_needed_frames(video_path, needed):
        for tid, xyxy in frame_work[fi]:
            crop = safe_crop(frame, xyxy)
            if crop is not None:
                pending_crops.append(crop)
                pending_tids.append(tid)
                if len(pending_crops) >= batch_size:
                    flush()
            done += 1
        if done % 400 == 0:
            print(f"  {done}/{total} crops processed…")
    flush()

    # Medoid (L2-normalized) per track
    embeddings = {}
    for tid, feats in feats_per_tid.items():
        if not feats:
            continue
        arr = np.stack(feats)
        arr = arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9)
        sim = arr @ arr.T
        medoid = int(np.argmax(sim.sum(axis=1)))
        embeddings[tid] = arr[medoid]
    print(f"  embeddings for {len(embeddings)}/{len(by_tid)} tracks")
    return embeddings


def index_p1b_p1c(teams, numbers, team_conf_floor):
    """Return per-tid dicts: team_id (best), team_conf, team_votes
    (multi-engine list), is_goaltender, jersey_number, jersey_conf.

    team_id is the BEST prior from Stage 1.b (max-confidence engine).
    team_votes is the full per-engine list `[{engine, team_id,
    confidence}, ...]` used by collect_entities for downstream voting.
    Tracks with no usable prior (team_id is None or team_conf < floor)
    still return their full team_votes so a strong signal in one
    engine can re-instate them at the entity level.
    """
    team_of = {}
    team_conf = {}
    team_votes = {}
    is_goalie = {}
    for tid_str, info in (teams or {}).get("tracks", {}).items():
        tid = int(tid_str)
        vc = float(info.get("vote_confidence", info.get("team_score", 0.0)))
        team_conf[tid] = vc
        # New multi-engine field, falls back to a synthetic single-engine
        # vote when reading old single-engine p1_b_teams.json.
        votes = info.get("team_votes")
        if not votes:
            t = info.get("team_id")
            if t is not None:
                votes = [{"engine": info.get("team_engine", "primary"),
                          "team_id": int(t), "confidence": vc}]
            else:
                votes = []
        team_votes[tid] = votes
        if vc >= team_conf_floor and info.get("team_id") is not None:
            team_of[tid] = info["team_id"]
        is_goalie[tid] = bool(info.get("is_goaltender", False))

    jersey = {}
    jersey_conf = {}
    for tid_str, info in (numbers or {}).get("tracks", {}).items():
        tid = int(tid_str)
        num = info.get("jersey_number")
        if num:
            jersey[tid] = str(num)
            jersey_conf[tid] = float(info.get("jersey_conf", 0.0))
    return team_of, team_conf, team_votes, is_goalie, jersey, jersey_conf


def build_edges(embeddings, frame_sets, team_of, is_goalie,
                jersey, jersey_conf,
                ocr_bonus, goalie_bonus, ocr_conflict_conf_floor,
                max_overlap_frames, cross_team_penalty):
    """All eligible merge candidate pairs, sorted by weight desc.

    Hard rejects (never considered):
      - temporal overlap > max_overlap_frames
      - OCR conflict: different confident jersey numbers, same team
        (the conflict block is intentionally team-aware: T0#14 vs
        T1#14 is two players, not a conflict)

    Soft penalty (still candidate but with reduced weight):
      - Cross-team pair (or one side has no team): subtract
        `cross_team_penalty`. Strong OCR/appearance signal can still
        carry the pair through; weak appearance-only edges drop below
        the merge threshold and stay rejected.
    """
    tids = sorted(embeddings.keys())
    edges = []
    for i in range(len(tids)):
        a = tids[i]
        ta = team_of.get(a)
        fa = frame_sets.get(a, frozenset())
        ja = jersey.get(a)
        ja_conf = jersey_conf.get(a, 0.0)
        emb_a = embeddings[a]

        for j in range(i + 1, len(tids)):
            b = tids[j]
            tb = team_of.get(b)
            fb = frame_sets.get(b, frozenset())
            overlap = len(fa & fb)
            if overlap > max_overlap_frames:
                continue
            jb = jersey.get(b)
            # Same-team OCR conflict: hard reject. Cross-team "same
            # number" is not a conflict — it's two distinct identities.
            if ja and jb and ja != jb and ta is not None and ta == tb:
                if ja_conf >= ocr_conflict_conf_floor and \
                   jersey_conf.get(b, 0.0) >= ocr_conflict_conf_floor:
                    continue
            sim = float(np.dot(emb_a, embeddings[b]))
            w = sim
            # OCR bonus only when same number AND same team (or one
            # team unknown). Cross-team same-number is intentionally
            # NOT bonused because that's two different players.
            if ja and jb and ja == jb and (ta is None or tb is None or ta == tb):
                w += ocr_bonus
            if is_goalie.get(a) and is_goalie.get(b):
                w += goalie_bonus
            # Soft same-team constraint: cross-team or unknown-team
            # pair gets a fixed penalty. Stage 1.b is a prior, not a
            # lock — strong signals (OCR-bonus +10, sim ~0.85) still
            # win; weak appearance-only edges drop below threshold.
            if ta is not None and tb is not None and ta != tb:
                w -= cross_team_penalty
            elif ta is None or tb is None:
                # One side unknown — half penalty
                w -= 0.5 * cross_team_penalty
            edges.append((w, a, b))
    edges.sort(key=lambda e: -e[0])
    return edges


class UnionFind:
    """Disjoint-set forest with path compression for the greedy merge.

    Keys are track ids; each root identifies one entity cluster after
    merges complete. Path compression keeps lookups near-constant time
    even on long merge chains."""

    def __init__(self, tids):
        """Initialise each track id as its own singleton cluster."""
        self.parent = {t: t for t in tids}

    def find(self, x):
        """Return the root of x's cluster, compressing the path along the way."""
        r = x
        while self.parent[r] != r:
            r = self.parent[r]
        # Path compression — points every visited node directly at the root.
        while self.parent[x] != r:
            self.parent[x], x = r, self.parent[x]
        return r

    def roots(self):
        """Return the set of cluster roots (one per merged group)."""
        return {self.find(t) for t in self.parent}


def greedy_merge(edges, frame_sets, tids_all, sim_threshold,
                 ocr_bonus, max_overlap_frames):
    """Greedy union under strict no-overlap. Returns (uf, n_merges)."""
    uf = UnionFind(tids_all)
    merged_frames = {t: set(frame_sets.get(t, frozenset())) for t in tids_all}

    n_merges = 0
    n_skipped_overlap = 0
    for w, a, b in edges:
        # OCR-bonus edges always have w >= ocr_bonus (>> sim_threshold).
        # Non-OCR edges have w == sim; once w drops below threshold we stop.
        if w < sim_threshold and w < ocr_bonus:
            break
        ra = uf.find(a)
        rb = uf.find(b)
        if ra == rb:
            continue
        # Re-check overlap on the merged cluster (tracks absorbed earlier
        # may have widened one side's frame set).
        if len(merged_frames[ra] & merged_frames[rb]) > max_overlap_frames:
            n_skipped_overlap += 1
            continue
        # Merge rb into ra
        uf.parent[rb] = ra
        merged_frames[ra] = merged_frames[ra] | merged_frames[rb]
        del merged_frames[rb]
        n_merges += 1
    return uf, n_merges, n_skipped_overlap


def collect_entities(uf, frame_sets, team_of, team_conf, team_votes_by_tid,
                     is_goalie, jersey, jersey_conf, total_frames):
    """Walk clusters → entity records.

    Per-entity decisions use **weighted majority voting** that aggregates
    every team vote from every engine across every member track, with
    track duration × engine-confidence as the weight. Long high-conf
    fragments and unanimous-engine fragments both outweigh single-engine
    noisy ones:

      - team_id: ∑ over (members, engines) of frames × engine_conf,
        bucketed by the team_id that engine assigned. Winner = argmax.
        team_score = winning_weight / total_weight (in [0, 1];
        1.0 = unanimous across every member × every engine).
      - is_goaltender: > 50 % of the entity's union frames come from
        tracks tagged goalie (no confidence component because Stage 1.b
        emits a binary tag, not a probability).
      - jersey_number: ∑ (track_frames × jersey_conf) per candidate
        number; winner = argmax. jersey_score = winning_weight /
        total_weight (or 0 if no track had an OCR'd number).

    The winning decisions are then **propagated** back to every member
    track via the entity_id pointer Stage 3.b uses (entity_for_tid),
    which means a track whose Stage 1.b said "team 0" but whose entity
    voted "team 1" will render as team 1 in the final video.
    """
    by_root = defaultdict(list)
    for tid in uf.parent:
        by_root[uf.find(tid)].append(tid)

    entities = []
    unmatched = []

    for root, members in by_root.items():
        # ---- weighted vote: team_id (multi-engine, multi-track) ----
        # Aggregate every engine vote of every member track. The weight
        # is (track_frames × engine_confidence): a long fragment + a
        # high-conf engine vote both push the same team. Cross-engine
        # disagreement on a single track is rare but real (it's why we
        # multi-engine in the first place); when it happens the vote
        # naturally resolves toward whichever side has more weight.
        vote_buckets = defaultdict(float)
        for t in members:
            tf = len(frame_sets.get(t, frozenset()))
            for v in team_votes_by_tid.get(t, []):
                tid_team = v.get("team_id")
                if tid_team is None:
                    continue
                vote_buckets[int(tid_team)] += tf * float(v.get("confidence", 0.0))
        if vote_buckets:
            team_id = max(vote_buckets, key=vote_buckets.get)
            total_w = sum(vote_buckets.values())
            team_score = vote_buckets[team_id] / total_w if total_w > 0 else 0.0
        else:
            team_id, team_score = None, 0.0

        # ---- frame-weighted is_goaltender (unchanged) ----
        goalie_frames = sum(len(frame_sets.get(t, frozenset()))
                            for t in members if is_goalie.get(t, False))
        total_member_frames = sum(len(frame_sets.get(t, frozenset()))
                                  for t in members)
        any_goalie = (goalie_frames / max(total_member_frames, 1)) > 0.5

        # ---- weighted vote: jersey_number ----
        # Weight = (frames covered by track) × (OCR confidence).
        # Multiplying by frame coverage avoids one bad short fragment
        # with a high-conf wrong OCR outvoting a long fragment with a
        # consistent moderate-conf right OCR.
        jn_votes = defaultdict(float)
        for t in members:
            if t in jersey:
                w = len(frame_sets.get(t, frozenset())) * jersey_conf.get(t, 0.0)
                jn_votes[jersey[t]] += w
        if jn_votes:
            jn_best = max(jn_votes, key=jn_votes.get)
            total_jn = sum(jn_votes.values())
            jn_score = jn_votes[jn_best] / total_jn if total_jn > 0 else 0.0
        else:
            jn_best, jn_score = None, 0.0

        frames_union = set()
        for t in members:
            frames_union |= frame_sets.get(t, frozenset())
        if not frames_union:
            continue
        first_frame = min(frames_union)
        last_frame = max(frames_union)
        covered = len(frames_union)

        # Identity string. Team-namespaced jersey number — two players
        # with the same number on different teams must read as
        # different identities. Used by Stage 3.b for the visible label.
        if team_id is not None and jn_best:
            identity = f"T{team_id}#{jn_best}"
        elif team_id is not None:
            identity = f"T{team_id}#??"
        else:
            identity = None

        record = {
            "track_ids": sorted(members),
            "team_id": team_id,
            "team_score": round(team_score, 3),
            "is_goaltender": any_goalie,
            "jersey_number": jn_best,
            "jersey_score": round(jn_score, 3),
            "identity": identity,
            "first_frame": first_frame,
            "last_frame": last_frame,
            "total_frames_covered": covered,
            "coverage_pct": round(100 * covered / max(total_frames, 1), 2),
        }
        # Singleton with no team label → unmatched
        if len(members) == 1 and record["team_id"] is None:
            unmatched.append(members[0])
        else:
            entities.append(record)

    entities.sort(key=lambda e: -e["total_frames_covered"])
    return entities, sorted(unmatched)


def verify_invariants(entities, frame_sets, max_overlap_frames):
    """Return list of (entity_idx, issue) tuples. Empty list = all good."""
    issues = []
    for i, e in enumerate(entities):
        tids = e["track_ids"]
        # Check pairwise non-overlap inside each entity
        for a_idx in range(len(tids)):
            a = tids[a_idx]
            fa = frame_sets.get(a, frozenset())
            for b_idx in range(a_idx + 1, len(tids)):
                b = tids[b_idx]
                fb = frame_sets.get(b, frozenset())
                overlap = len(fa & fb)
                if overlap > max_overlap_frames:
                    issues.append((i, f"entity {i} tracks {a},{b} overlap {overlap} frames"))
    return issues


def report(entities, unmatched, team_of, is_goalie):
    """Print a human-readable summary of the merge result: per-team
    entity / goalie counts and the top 10 entities by frame coverage."""
    per_team = defaultdict(lambda: {"n": 0, "goalies": 0})
    for e in entities:
        per_team[e["team_id"]]["n"] += 1
        if e["is_goaltender"]:
            per_team[e["team_id"]]["goalies"] += 1

    print(f"\nEntities: {len(entities)}   Unmatched tracks: {len(unmatched)}")
    for team in sorted(per_team):
        p = per_team[team]
        team_label = "none" if team is None else f"team {team}"
        print(f"  {team_label}: {p['n']} entities ({p['goalies']} goaltender)")

    print(f"\nTop 10 entities by frame coverage:")
    for i, e in enumerate(entities[:10]):
        num = f"#{e['jersey_number']}" if e["jersey_number"] else "#??"
        role = "G" if e["is_goaltender"] else "S"
        ts = e.get("team_score", 0.0)
        js = e.get("jersey_score", 0.0)
        print(f"  [{i}] team={e['team_id']}({ts:.2f}) {role} {num:>4s}({js:.2f})  "
              f"tracks={len(e['track_ids']):3d}  "
              f"frames={e['first_frame']:>4d}-{e['last_frame']:<4d}  "
              f"cover={e['coverage_pct']:>5.1f}%")


def run(detections_json, teams_json, numbers_json, video_path, output,
        samples_per_track, batch_size, sim_threshold,
        ocr_bonus, ocr_conflict_conf_floor, goalie_bonus,
        team_conf_floor, max_overlap_frames, cross_team_penalty,
        osnet_model):
    """Run the entity-clustering pipeline end-to-end.

    Loads P1.a/b/c outputs, computes per-track OSNet medoid
    embeddings, builds a pair-wise merge graph under team / non-overlap
    / OCR-conflict constraints, runs greedy union-find merging, and
    writes ``p3_a_entities.json`` with the resulting clusters.
    """
    device = pick_device()
    print(f"Device: {device}")

    detections_data = json.loads(detections_json.read_text())
    teams = json.loads(teams_json.read_text()) if teams_json.exists() else None
    numbers = (json.loads(numbers_json.read_text())
               if numbers_json.exists() else None)
    if teams is None:
        raise SystemExit(f"Missing {teams_json}: run p1_b_teams first")
    if numbers is None:
        print(f"Warning: {numbers_json} missing — OCR signal won't be used")

    team_of, team_conf, team_votes_by_tid, is_goalie, jersey, jersey_conf = \
        index_p1b_p1c(teams, numbers, team_conf_floor)
    n_eligible = sum(1 for _ in team_of)
    n_teamless = sum(
        1 for tid_str, info in teams["tracks"].items()
        if info.get("vote_confidence", 0) < team_conf_floor
    )
    print(f"Tracks with team label (conf ≥ {team_conf_floor}): {n_eligible}, "
          f"below floor: {n_teamless}")
    print(f"Tracks with OCR number: {len(jersey)}")
    print(f"Tagged as goaltender: {sum(1 for v in is_goalie.values() if v)}")

    print(f"\nLoading OSNet ({osnet_model})…")
    extractor = FeatureExtractor(model_name=osnet_model, device=device)

    embeddings = extract_track_embeddings(
        detections_data, video_path, extractor,
        samples_per_track=samples_per_track, batch_size=batch_size,
    )
    frame_sets = track_frame_sets(detections_data)
    tids_all = set(frame_sets.keys()) | set(embeddings.keys())

    print(f"\nBuilding merge graph (cross-team penalty={cross_team_penalty})…")
    edges = build_edges(
        embeddings, frame_sets, team_of, is_goalie,
        jersey, jersey_conf,
        ocr_bonus=ocr_bonus, goalie_bonus=goalie_bonus,
        ocr_conflict_conf_floor=ocr_conflict_conf_floor,
        max_overlap_frames=max_overlap_frames,
        cross_team_penalty=cross_team_penalty,
    )
    print(f"  {len(edges)} eligible pairs")

    print(f"Greedy merge (sim_threshold={sim_threshold}, OCR bonus={ocr_bonus})…")
    uf, n_merges, n_skipped_overlap = greedy_merge(
        edges, frame_sets, tids_all,
        sim_threshold=sim_threshold, ocr_bonus=ocr_bonus,
        max_overlap_frames=max_overlap_frames,
    )
    print(f"  {n_merges} merges performed, {n_skipped_overlap} skipped (overlap)")

    entities, unmatched = collect_entities(
        uf, frame_sets, team_of, team_conf, team_votes_by_tid,
        is_goalie, jersey, jersey_conf,
        detections_data["total_frames"],
    )

    issues = verify_invariants(entities, frame_sets, max_overlap_frames)
    if issues:
        print(f"\nWARNING: {len(issues)} invariant violations:")
        for i, msg in issues[:10]:
            print(f"  {msg}")
    else:
        print("\nAll invariants OK (no temporal overlaps inside any entity).")

    report(entities, unmatched, team_of, is_goalie)

    out = {
        "source_detections": str(detections_json),
        "source_teams": str(teams_json),
        "source_numbers": str(numbers_json),
        "source_video": str(video_path),
        "method": "OSNet medoid + greedy merge under team + non-overlap + OCR bonus",
        "params": {
            "samples_per_track": samples_per_track,
            "sim_threshold": sim_threshold,
            "ocr_bonus": ocr_bonus,
            "goalie_bonus": goalie_bonus,
            "team_conf_floor": team_conf_floor,
            "ocr_conflict_conf_floor": ocr_conflict_conf_floor,
            "max_overlap_frames": max_overlap_frames,
            "cross_team_penalty": cross_team_penalty,
            "osnet_model": osnet_model,
        },
        "n_entities": len(entities),
        "n_unmatched": len(unmatched),
        "entities": {str(i): e for i, e in enumerate(entities)},
        "unmatched_track_ids": unmatched,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {output}")


def main():
    """CLI entry point — parse arguments and dispatch to ``run``."""
    p = argparse.ArgumentParser(
        description="RILH-AI-Vision — p3_a_entities : merge tracks into entities"
    )
    p.add_argument("detections_json", type=str,
                   help="P1.a output (p1_a_detections.json)")
    p.add_argument("teams_json", type=str,
                   help="P1.b output (p1_b_teams.json)")
    p.add_argument("numbers_json", type=str,
                   help="P1.c output (p1_c_numbers.json)")
    p.add_argument("video", type=str)
    p.add_argument("--output", type=str, default=None,
                   help="Output JSON (default: <detections_dir>/p3_a_entities.json)")
    p.add_argument("--samples-per-track", type=int, default=15,
                   help="OSNet medoid uses up to N highest-conf detections "
                        "per track. Higher = more stable embedding, slower "
                        "embedding wall-clock. Default 15 (was 8).")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--sim-threshold", type=float, default=0.65,
                   help="Cosine similarity floor for non-OCR merges")
    p.add_argument("--ocr-bonus", type=float, default=10.0)
    p.add_argument("--ocr-conflict-conf-floor", type=float, default=0.40,
                   help="If both tracks carry confident but different numbers "
                        "AND the same team, reject the pair. Cross-team "
                        "same-number is intentionally NOT a conflict — that's "
                        "two distinct identities (T0#14 vs T1#14).")
    p.add_argument("--goalie-bonus", type=float, default=0.05)
    p.add_argument("--team-conf-floor", type=float, default=0.50,
                   help="Tracks with team_score below this floor still get "
                        "their team_votes propagated (so engine vote drives "
                        "the entity decision); they're just not used as the "
                        "anchor for the soft same-team prior. Lowered from "
                        "0.67 since the soft constraint replaces hard reject.")
    p.add_argument("--max-overlap-frames", type=int, default=2,
                   help="Tracks in the same entity may share up to this many "
                        "frames (default 2). Loosened from 0 to absorb the "
                        "1-frame box-collision artefact during scrums where "
                        "the tracker hands a player off cleanly but with a "
                        "single overlapping frame.")
    p.add_argument("--cross-team-penalty", type=float, default=0.30,
                   help="Soft same-team constraint: cross-team pair weight is "
                        "reduced by this amount (instead of hard reject). "
                        "Strong signals (OCR bonus +10, sim ~0.85) still "
                        "carry the merge; weak appearance-only edges drop "
                        "below threshold. Set to 100.0 to recover the old "
                        "hard-reject behaviour.")
    p.add_argument("--osnet", type=str, default="osnet_x0_25",
                   help="Torchreid model name (osnet_x0_25 = smallest).")
    args = p.parse_args()

    detections_json = Path(args.detections_json)
    teams_json = Path(args.teams_json)
    numbers_json = Path(args.numbers_json)
    video_path = Path(args.video)
    output = (Path(args.output) if args.output
              else detections_json.with_name("p3_a_entities.json"))

    run(
        detections_json, teams_json, numbers_json, video_path, output,
        samples_per_track=args.samples_per_track,
        batch_size=args.batch_size,
        sim_threshold=args.sim_threshold,
        ocr_bonus=args.ocr_bonus,
        ocr_conflict_conf_floor=args.ocr_conflict_conf_floor,
        goalie_bonus=args.goalie_bonus,
        team_conf_floor=args.team_conf_floor,
        max_overlap_frames=args.max_overlap_frames,
        cross_team_penalty=args.cross_team_penalty,
        osnet_model=args.osnet,
    )


if __name__ == "__main__":
    main()
