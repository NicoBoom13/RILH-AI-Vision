"""
RILH-AI-Vision — team-engine benchmark.

Runs each Stage 1.b team engine (hsv, osnet, siglip, optionally
contrastive) on a set of existing run folders, scores the per-track
team labels against the ground-truth file produced by
`tools/annotate_tracks.py` (`data/jersey_numbers/track_truth.json`),
and writes a comparative report.

Why per-engine vs per-pipeline-rerun: the slow part of the pipeline is
Stage 1.a (detection + tracking — half an hour per 60 s clip). Stage
1.b is fast, so we re-run only it for each engine, keeping the same
upstream detections.

Usage:
  python tools/bench_team_engines.py runs/run24 runs/run25 ... \\
      --engines hsv osnet siglip \\
      [--contrastive-checkpoint models/contrastive_team_rilh.pt] \\
      [--output runs/bench_team_engines_YYYYMMDD]

Outputs in --output:
  results.json    — per-(engine, run, track) prediction + the truth label
  summary.json    — per-engine aggregate metrics (precision, recall,
                    intra-clip purity)
  summary.txt     — human-readable comparison table

Scoring rule — per-clip, ignore A/B label permutation:
  team-classification accuracy = max(intra_clip_purity_under_swap_0,
                                     intra_clip_purity_under_swap_1)
  where intra_clip_purity = sum_of_correctly_assigned_player_tracks /
                            n_player_tracks_with_truth_in_this_clip.
  Tracks labelled X (not-a-player) and REF are excluded from the score
  but reported separately. This matches Koshkina 2021's convention —
  absolute team identity is meaningless across clips.

Referee handling: tracks whose engine output != REF but truth IS REF
count as a separate "ref leakage" miss; tracks whose engine == REF but
truth != REF count as "ref false-alarm". These are reported alongside
team accuracy but do NOT enter the team-accuracy metric (engines below
the contrastive level don't predict REF).
"""

import argparse
import datetime
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"


def run_engine_on_clip(engine, run_dir, video_path, contrastive_ckpt,
                       force=False):
    """Re-run Stage 1.b on a single clip with the given engine. Saves
    p1_b_teams_<engine>.json so each engine's output coexists in the
    same run folder, leaving the canonical p1_b_teams.json untouched.
    Returns the output path."""
    out = run_dir / f"p1_b_teams_{engine}.json"
    if out.exists() and not force:
        return out
    cmd = [sys.executable, "-u", str(SRC / "p1_b_teams.py"),
           str(run_dir / "p1_a_detections.json"), str(video_path),
           "--output", str(out),
           "--pose-model", "yolo26l-pose.pt",
           "--team-engine", engine]
    if engine == "contrastive" and contrastive_ckpt:
        cmd.extend(["--contrastive-checkpoint", str(contrastive_ckpt)])
    print(f"  $ {' '.join(cmd)}")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise RuntimeError(f"Stage 1.b failed for {run_dir.name} / {engine}")
    return out


def score_clip(pred, truth_for_clip):
    """Compute (team_accuracy, n_team_tracks, ref_recall, ref_precision,
    ref_counts) for one clip. team_accuracy is permutation-invariant
    (max over A↔B swap)."""
    if not truth_for_clip:
        return None
    truth_team = {tid: m["team"] for tid, m in truth_for_clip.items()
                  if m.get("team") in ("A", "B")}
    truth_ref = {tid: m["is_referee"] for tid, m in truth_for_clip.items()}

    pred_team_of = {int(t): info["team_id"]
                    for t, info in pred.get("tracks", {}).items()}

    # Team accuracy (permutation-invariant)
    if truth_team:
        n_match_id = sum(1 for tid, t in truth_team.items()
                         if pred_team_of.get(tid) == (0 if t == "A" else 1))
        n_match_swap = sum(1 for tid, t in truth_team.items()
                           if pred_team_of.get(tid) == (1 if t == "A" else 0))
        team_correct = max(n_match_id, n_match_swap)
        team_acc = team_correct / len(truth_team)
    else:
        team_correct, team_acc = 0, 0.0

    # Referee metrics — only meaningful when the engine emits an
    # is_referee flag (or when the user pairs the engine with the ref
    # binary classifier, see tools/finetune_ref_classifier.py).
    ref_pred = {int(t): bool(info.get("is_referee", False))
                for t, info in pred.get("tracks", {}).items()}
    truth_ref_set = {tid for tid, is_ref in truth_ref.items() if is_ref}
    pred_ref_set  = {tid for tid, is_ref in ref_pred.items() if is_ref}
    if truth_ref_set:
        ref_recall = len(truth_ref_set & pred_ref_set) / len(truth_ref_set)
    else:
        ref_recall = None
    if pred_ref_set:
        ref_precision = len(truth_ref_set & pred_ref_set) / len(pred_ref_set)
    else:
        ref_precision = None

    return {
        "n_team_tracks": len(truth_team),
        "team_correct": team_correct,
        "team_accuracy": team_acc,
        "n_truth_ref": len(truth_ref_set),
        "n_pred_ref": len(pred_ref_set),
        "ref_recall": ref_recall,
        "ref_precision": ref_precision,
    }


def find_video(run_dir, video_root):
    det = json.loads((run_dir / "p1_a_detections.json").read_text())
    sv = det.get("source_video") or ""
    p = Path(sv)
    if p.exists():
        return p
    candidates = list(Path(video_root).glob(f"*{p.name}*"))
    return candidates[0] if candidates else None


def main():
    p = argparse.ArgumentParser(description="Compare Stage 1.b team engines")
    p.add_argument("runs", nargs="+", type=str)
    p.add_argument("--engines", nargs="+",
                   default=["hsv", "osnet", "siglip"],
                   choices=["hsv", "osnet", "siglip", "contrastive"])
    p.add_argument("--contrastive-checkpoint", type=str, default=None)
    p.add_argument("--truth", type=str,
                   default="data/jersey_numbers/track_truth.json")
    p.add_argument("--video-root", type=str, default="videos")
    p.add_argument("--output", type=str, default=None,
                   help="Bench output dir (default: runs/bench_team_engines_YYYYMMDD_HHMMSS)")
    p.add_argument("--force", action="store_true",
                   help="Re-run engines even if p1_b_teams_<engine>.json exists.")
    args = p.parse_args()

    truth_path = Path(args.truth)
    if not truth_path.exists():
        raise SystemExit(f"Truth file not found: {truth_path}\n"
                         f"Run tools/annotate_tracks.py first.")
    truth_data = json.loads(truth_path.read_text())["tracks"]
    # Group truth by clip name (first path segment of "<run>/<tid>")
    truth_by_clip = defaultdict(dict)
    for key, meta in truth_data.items():
        if "/" not in key:
            continue
        clip, tid_str = key.split("/", 1)
        try:
            truth_by_clip[clip][int(tid_str)] = meta
        except ValueError:
            pass

    out_dir = Path(args.output) if args.output else (
        PROJECT_ROOT / "runs" /
        f"bench_team_engines_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Bench output: {out_dir}")

    run_dirs = [Path(r).resolve() for r in args.runs]

    # 1) Re-run each engine on each clip (skipping if cached).
    pred_paths = {}
    for r in run_dirs:
        video = find_video(r, args.video_root)
        if video is None:
            print(f"  WARN: no video for {r.name} — skipping")
            continue
        for eng in args.engines:
            print(f"\n[{eng}] {r.name}")
            try:
                pred_paths[(eng, r.name)] = run_engine_on_clip(
                    eng, r, video, args.contrastive_checkpoint, force=args.force,
                )
            except Exception as e:
                print(f"  FAIL: {e}")

    # 2) Score each (engine, clip) pair.
    per_clip = defaultdict(dict)   # engine → clip → metrics
    for (eng, clip), path in pred_paths.items():
        pred = json.loads(path.read_text())
        truth_for_clip = truth_by_clip.get(clip, {})
        scored = score_clip(pred, truth_for_clip)
        if scored is None:
            continue
        per_clip[eng][clip] = scored

    # 3) Aggregate per engine: weighted accuracy + per-clip table.
    summary = {}
    for eng, clips in per_clip.items():
        n_total_team = sum(c["n_team_tracks"] for c in clips.values())
        n_correct    = sum(c["team_correct"] for c in clips.values())
        agg_acc = n_correct / n_total_team if n_total_team else 0.0
        ref_recalls = [c["ref_recall"] for c in clips.values()
                       if c["ref_recall"] is not None]
        ref_precs   = [c["ref_precision"] for c in clips.values()
                       if c["ref_precision"] is not None]
        summary[eng] = {
            "team_accuracy_weighted": agg_acc,
            "n_team_tracks": n_total_team,
            "n_correct": n_correct,
            "ref_recall_mean": sum(ref_recalls) / len(ref_recalls)
                                if ref_recalls else None,
            "ref_precision_mean": sum(ref_precs) / len(ref_precs)
                                  if ref_precs else None,
            "per_clip": clips,
        }

    (out_dir / "results.json").write_text(json.dumps(per_clip, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # 4) Human-readable table.
    lines = []
    lines.append("=" * 78)
    lines.append(f"RILH team-engine bench — {datetime.datetime.now().isoformat()}")
    lines.append(f"Truth: {truth_path}  ({len(truth_data)} labelled tracks)")
    lines.append(f"Clips: {', '.join(r.name for r in run_dirs)}")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"{'engine':12s} {'team_acc':>10s} {'n_tracks':>10s} "
                 f"{'ref_recall':>11s} {'ref_prec':>10s}")
    lines.append("-" * 78)
    for eng in args.engines:
        if eng not in summary:
            lines.append(f"{eng:12s} (no results — engine failed or not run)")
            continue
        s = summary[eng]
        rr = f"{s['ref_recall_mean']:.3f}" if s['ref_recall_mean'] is not None else "  —  "
        rp = f"{s['ref_precision_mean']:.3f}" if s['ref_precision_mean'] is not None else "  —  "
        lines.append(f"{eng:12s} {s['team_accuracy_weighted']:>10.3f} "
                     f"{s['n_team_tracks']:>10d} {rr:>11s} {rp:>10s}")
    lines.append("")
    lines.append("Per-clip team accuracy:")
    clips_seen = sorted({c for s in summary.values() for c in s["per_clip"]})
    header = f"  {'clip':14s}" + "".join(f"{e:>10s}" for e in args.engines)
    lines.append(header)
    for clip in clips_seen:
        row = f"  {clip:14s}"
        for e in args.engines:
            s = summary.get(e, {}).get("per_clip", {}).get(clip)
            row += f"{s['team_accuracy']:>10.3f}" if s else f"{'—':>10s}"
        lines.append(row)
    lines.append("")
    text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
