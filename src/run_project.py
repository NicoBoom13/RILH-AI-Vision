"""
RILH-AI-Vision — project orchestrator.

Runs the full project pipeline (Phase 1 → Phase 5) on a single video,
with per-phase gates. Each stage script is invoked as a subprocess so
that it stays independently runnable; this orchestrator only knows the
glue (which file feeds which, which flags pass through, what to skip).

Pipeline (in execution order):
  Phase 1 — Detect & track       — 3 stages (a detect, b teams, c numbers)
  Phase 2 — Rink calibration     — 1 stage (a), HIGH PRIORITY / in progress
                                    (HockeyRink doesn't transfer to roller
                                    yet — needs a fine-tune dataset; Phase 2
                                    is run by default but tolerant of failure)
  Phase 3 — Entity recognition   — 2 stages (a entities, b annotated MP4),
                                    formerly Phase 1 stages d + e
  Phase 4 — Event detection      — 1 stage (a), STUB
  Phase 5 — Statistics creation  — 1 stage (a), STUB

Phase 6 (Web platform) and Phase 7 (Multi-cam / live) are not part of the per-run
pipeline — they are separate services that consume the run folder.

The previous Phase 2 (Virtual follow-cam) was removed; its output wasn't
usable and rink calibration is the bottleneck for entity quality.

Usage:
  python src/run_project.py videos/match.mp4 --output runs/run30 \\
      --hockey-model --pose-model yolo26l-pose.pt \\
      --parseq-checkpoint models/parseq_hockey_rilh.pt

  # Skip a phase entirely:
  python src/run_project.py videos/match.mp4 --output runs/run30 --skip-p2

  # Force re-run even if outputs already exist:
  python src/run_project.py videos/match.mp4 --output runs/run30 --force

By default every phase is ON; each stage skips its own work if its
output file already exists (incremental re-runs cost nothing). Pass
`--force` to re-run all enabled phases from scratch.
"""

import argparse
import datetime
import subprocess
import sys
import time
from pathlib import Path


# Always invoke stage scripts with the same Python that's running the
# orchestrator (avoids any venv mismatch when running via .venv/bin/python).
PYTHON = sys.executable

# Project root resolved from THIS script's location, so the orchestrator
# can be launched from anywhere (CI, IDE, cron, …) and still find src/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"


def step(label, cmd, expected_outputs, force):
    """Run a stage subprocess, but skip if every expected output already exists.

    Args:
        label: Human-readable name printed in the log.
        cmd: argv list passed to subprocess.run.
        expected_outputs: list of Path objects the stage should produce.
            All must exist for the stage to be considered "done".
        force: bool. If True, ignore existing outputs and re-run.

    Returns:
        True if the stage ran (or was skipped because outputs exist),
        False if the stage failed (subprocess exit != 0).

    Raises:
        Nothing — caller decides what to do on failure (e.g. Phase 2 is
        tolerant, others may want to re-raise).
    """
    print(f"\n----- {label} -----")
    if not force and all(o.exists() for o in expected_outputs):
        print(f"  ✓ outputs already exist, skipping (use --force to re-run)")
        for o in expected_outputs:
            print(f"      {o}")
        return True

    t0 = time.time()
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd)
    dt = time.time() - t0
    if result.returncode != 0:
        print(f"  ✗ FAILED after {dt:.0f}s (exit code {result.returncode})")
        return False
    print(f"  ✓ done in {dt:.0f}s")
    return True


def steps_parallel(items, force, log_dir):
    """Run several stage subprocesses concurrently, each redirecting its
    own stdout+stderr to a log file under ``log_dir``. Returns a list of
    bools (one per item) reporting per-stage success.

    ``items`` is a list of ``(label, cmd, expected_outputs, log_basename)``
    tuples. Stages whose expected outputs already exist are skipped (and
    counted as success), unless ``force=True``.

    Each running stage writes to ``log_dir/<log_basename>.log``; tail
    those files to follow live progress (the orchestrator only reports
    start/end here, since interleaved stdout from concurrent stages
    would be unreadable)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n----- {' + '.join(it[0] for it in items)} (parallel) -----")
    procs = []   # parallel to items: (popen|None, log_file|None, t0|None)
    skipped = []
    for label, cmd, expected_outputs, log_basename in items:
        if not force and all(o.exists() for o in expected_outputs):
            print(f"  ✓ {label}: outputs already exist, skipping")
            for o in expected_outputs:
                print(f"      {o}")
            procs.append((None, None, None))
            skipped.append(True)
            continue
        skipped.append(False)
        log_path = log_dir / f"{log_basename}.log"
        log_f = open(log_path, "w")
        print(f"  ▶ {label}")
        print(f"      $ {' '.join(str(c) for c in cmd)}")
        print(f"      log → {log_path}")
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((proc, log_f, time.time()))

    results = []
    for (label, _cmd, _exp, _basename), (proc, log_f, t0), was_skipped in zip(
            items, procs, skipped):
        if was_skipped:
            results.append(True)
            continue
        rc = proc.wait()
        log_f.close()
        dt = time.time() - t0
        if rc != 0:
            print(f"  ✗ {label} FAILED after {dt:.0f}s (exit {rc})")
            results.append(False)
        else:
            print(f"  ✓ {label} done in {dt:.0f}s")
            results.append(True)
    return results


def run_p1_detect_track(video, out, args):
    """Run Phase 1 (a detect → b teams → c numbers) sequentially.

    Halts on first failure: each stage's outputs feed the next, so a
    failed Stage 1.a means Stage 1.b can't run. Annotation + entity
    clustering moved to Phase 3 — they need rink-aware filtering.
    """
    print(f"\n========== Phase 1 — Detect & Track ==========")

    # Stage 1.a — detect + track
    cmd = [PYTHON, "-u", str(SRC / "p1_a_detect.py"),
           str(video), "--output", str(out)]
    if args.hockey_model:
        cmd.append("--hockey-model")
    if args.training_mode:
        cmd.append("--training-mode")
    if args.model:
        cmd.extend(["--model", args.model])
    cmd.extend(["--conf", str(args.conf), "--imgsz", str(args.imgsz),
                "--tracker", args.tracker,
                "--detect-fps", str(args.detect_fps)])
    if not step("Stage 1.a — Detect & track", cmd,
                [out / "p1_a_detections.json"], args.force):
        raise SystemExit("Stage 1.a failed — halting Phase 1")

    # Pose pre-extract — runs YOLO pose ONCE on the union of frames
    # that 1.b and 1.c need (top-15 highest-conf detections per track,
    # which subsumes 1.b's top-8). Both stages then look up the cache
    # instead of re-running pose, which (a) saves a full pose pass
    # vs the old sequential 1.b → 1.c, and (b) unblocks running 1.b
    # and 1.c in parallel without GPU contention on the pose model.
    cmd = [PYTHON, "-u", str(SRC / "pose_cache.py"),
           str(out / "p1_a_detections.json"), str(video),
           "--pose-model", args.pose_model,
           "--samples-per-track", "15"]
    if not step("Stage 1.b/1.c — Pose pre-extract", cmd,
                [out / "p1_pose_cache.pkl"], args.force):
        # Soft-fail: 1.b and 1.c can still run inline-pose if the
        # cache isn't there. We just lose the wall-clock saving.
        print("  ⚠ Pose pre-extract failed — 1.b/1.c will fall back "
              "to inline pose. Continuing.")

    # Stage 1.b → Stage 1.c run **sequentially**: 1.c reads
    # p1_b_teams.json so it can namespace jersey numbers by team
    # (T0#14 ≠ T1#14). The earlier parallel layout broke that
    # invariant — without team labels at OCR time, two players with
    # the same number on opposite teams could merge into one
    # player_group. Both stages still benefit from the shared pose
    # cache produced above (so neither re-runs pose).
    cmd_b = [PYTHON, "-u", str(SRC / "p1_b_teams.py"),
             str(out / "p1_a_detections.json"), str(video),
             "--pose-model", args.pose_model]
    if args.team_engines:
        cmd_b.extend(["--team-engines", args.team_engines])
    else:
        cmd_b.extend(["--team-engine", args.team_engine])
    if args.contrastive_checkpoint:
        cmd_b.extend(["--contrastive-checkpoint", args.contrastive_checkpoint])
    if args.ref_classifier:
        cmd_b.extend(["--ref-classifier", args.ref_classifier])
    if not step("Stage 1.b — Teams", cmd_b,
                [out / "p1_b_teams.json"], args.force):
        raise SystemExit("Stage 1.b failed — halting Phase 1")

    cmd_c = [PYTHON, "-u", str(SRC / "p1_c_numbers.py"),
             str(out / "p1_a_detections.json"), str(video),
             "--pose-model", args.pose_model,
             "--teams-json", str(out / "p1_b_teams.json")]
    if args.parseq_checkpoint:
        cmd_c.extend(["--parseq-checkpoint", args.parseq_checkpoint])
    if not step("Stage 1.c — Numbers", cmd_c,
                [out / "p1_c_numbers.json"], args.force):
        raise SystemExit("Stage 1.c failed — halting Phase 1")


def run_p2_rink(video, out, args):
    """Run Phase 2 — Rink calibration. High priority but currently
    tolerant of failure: HockeyRink (ice) doesn't transfer to roller
    rinks yet, so off-the-shelf inference will run but produce a
    cluster of useless keypoints. The wiring is exercised so that a
    future fine-tune drops cleanly into place; downstream phases
    don't depend on a successful rink output yet."""
    print(f"\n========== Phase 2 — Rink calibration ==========")
    cmd = [PYTHON, "-u", str(SRC / "p2_a_rink.py"),
           str(video), "--output", str(out)]
    ok = step("Stage 2.a — Rink keypoints", cmd,
              [out / "p2_a_rink_keypoints.json"], args.force)
    if not ok:
        print("  ⚠ Phase 2 not yet usable on roller rinks — fine-tune is the "
              "next blocker. Continuing.")


def run_p3_entity(video, out, args):
    """Run Phase 3 — Entity recognition + final annotated MP4.

    Stage 3.a re-IDs fragmented Phase 1 tracks into stable entities via
    OSNet medoid embeddings + team / non-overlap / OCR constraints.
    Stage 3.b renders the annotated.mp4 from those entities. Both
    stages depend on Phase 1 outputs; Stage 3.b also auto-discovers
    p3_a_entities.json so per-track labels stay consistent across
    fragments.
    """
    print(f"\n========== Phase 3 — Entity recognition ==========")

    needed = [out / "p1_a_detections.json",
              out / "p1_b_teams.json",
              out / "p1_c_numbers.json"]
    for n in needed:
        if not n.exists():
            print(f"  ⚠ Phase 3 needs {n.name} — skipping")
            return

    # Stage 3.a — entities (Re-ID merge)
    cmd = [PYTHON, "-u", str(SRC / "p3_a_entities.py"),
           str(out / "p1_a_detections.json"), str(out / "p1_b_teams.json"),
           str(out / "p1_c_numbers.json"), str(video)]
    if not step("Stage 3.a — Entities", cmd,
                [out / "p3_a_entities.json"], args.force):
        raise SystemExit("Stage 3.a failed — halting Phase 3")

    # Stage 3.b — annotate (final MP4)
    cmd = [PYTHON, "-u", str(SRC / "p3_b_annotate.py"),
           str(out / "p1_a_detections.json"), str(out / "p1_c_numbers.json"),
           str(video), "--output", str(out / "annotated.mp4")]
    if not step("Stage 3.b — Annotate", cmd,
                [out / "annotated.mp4"], args.force):
        raise SystemExit("Stage 3.b failed — halting Phase 3")


def run_p4_events(out, args):
    """Run Phase 4 — Event detection (stub). Writes a marker JSON."""
    print(f"\n========== Phase 4 — Event detection (stub) ==========")
    if not (out / "p1_a_detections.json").exists() or not (out / "p3_a_entities.json").exists():
        print("  ⚠ Phase 4 needs p1_a_detections.json + p3_a_entities.json — skipping")
        return
    cmd = [PYTHON, "-u", str(SRC / "p4_a_events.py"),
           str(out / "p1_a_detections.json"), str(out / "p3_a_entities.json"),
           "--output", str(out / "p4_a_events.json")]
    step("Stage 4.a — Events", cmd, [out / "p4_a_events.json"], args.force)


def run_p5_stats(out, args):
    """Run Phase 5 — Statistics creation (stub). Writes a marker JSON."""
    print(f"\n========== Phase 5 — Statistics (stub) ==========")
    if not (out / "p3_a_entities.json").exists() or not (out / "p1_c_numbers.json").exists():
        print("  ⚠ Phase 5 needs p3_a_entities.json + p1_c_numbers.json — skipping")
        return
    cmd = [PYTHON, "-u", str(SRC / "p5_a_stats.py"),
           str(out / "p3_a_entities.json"), str(out / "p1_c_numbers.json"),
           "--output", str(out / "p5_a_stats.json")]
    step("Stage 5.a — Stats", cmd, [out / "p5_a_stats.json"], args.force)


def main():
    """CLI entry point — parse arguments and run the requested phases."""
    p = argparse.ArgumentParser(
        description="RILH-AI-Vision project orchestrator (Phase 1 → Phase 5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required positional + output
    p.add_argument("video", type=str, help="Input video path (.mp4)")
    p.add_argument("--output", type=str, required=True,
                   help="Output run folder, e.g. runs/run30/")

    # Phase gates — each phase is ON by default
    g = p.add_argument_group("Phase gates")
    g.add_argument("--skip-p1", action="store_true",
                   help="Skip Phase 1 (Detect & track + teams + numbers)")
    g.add_argument("--skip-p2", action="store_true",
                   help="Skip Phase 2 (Rink calibration — high priority but "
                        "currently tolerant of failure pending the fine-tune)")
    g.add_argument("--skip-p3", action="store_true",
                   help="Skip Phase 3 (Entity recognition + annotated MP4)")
    g.add_argument("--skip-p4", action="store_true",
                   help="Skip Phase 4 (Event detection, stub)")
    g.add_argument("--skip-p5", action="store_true",
                   help="Skip Phase 5 (Statistics, stub)")
    g.add_argument("--force", action="store_true",
                   help="Force re-run all enabled stages even if outputs exist")

    # Pass-through to Phase 1 stages
    g1 = p.add_argument_group("Phase 1 options (passed through)")
    g1.add_argument("--hockey-model", action="store_true",
                    help="Stage 1.a: use HockeyAI weights (recommended)")
    g1.add_argument("--training-mode", action="store_true",
                    help="Stage 1.a: disable 1-puck-per-frame filter")
    g1.add_argument("--detect-fps", type=float, default=30.0,
                    help="Stage 1.a: target detection frame-rate (default 30). "
                         "On 60fps source video this halves the detection "
                         "wall-clock; pass 60 to disable sub-sampling. "
                         "Frame indices in p1_a_detections.json stay aligned "
                         "to the source video, and Stage 3.b carries "
                         "detections forward across skipped frames.")
    g1.add_argument("--model", type=str, default=None,
                    help="Stage 1.a: COCO YOLO weights (ignored when --hockey-model)")
    g1.add_argument("--conf", type=float, default=0.3,
                    help="Stage 1.a: detection confidence threshold")
    g1.add_argument("--imgsz", type=int, default=1920,
                    help="Stage 1.a: inference image size")
    g1.add_argument("--tracker", type=str, default="bytetrack.yaml",
                    help="Stage 1.a: tracker config")
    g1.add_argument("--pose-model", type=str, default="yolo26l-pose.pt",
                    help="Stage 1.b + Stage 1.c: YOLO pose weights")
    g1.add_argument("--team-engine", type=str, default="hsv",
                    choices=["hsv", "osnet", "siglip", "contrastive"],
                    help="Stage 1.b: SINGLE-engine backend (legacy mode). "
                         "Ignored when --team-engines is set. "
                         "hsv (default — fast, no extra model), "
                         "osnet (OSNet x0_25 medoid embeddings), "
                         "siglip (Roboflow SigLIP+UMAP), "
                         "contrastive (Koshkina triplet model — best "
                         "overall in the 6-clip bench, 0.896 vs hsv 0.788; "
                         "needs models/contrastive_team_rilh.pt).")
    g1.add_argument("--team-engines", type=str, default=None,
                    help="Stage 1.b: MULTI-engine mode — comma-separated list "
                         "(e.g. 'contrastive,hsv'). Each track gets a "
                         "team_votes array with one entry per engine, plus a "
                         "team_id_best summary. Stage 3.a uses these as a "
                         "SOFT prior (cross-team penalty instead of hard "
                         "reject) so a strong appearance/OCR signal can "
                         "correct a single mis-classified engine. Overrides "
                         "--team-engine when set.")
    g1.add_argument("--contrastive-checkpoint", type=str, default=None,
                    help="Stage 1.b: path to contrastive-team checkpoint "
                         "(used only with --team-engine contrastive). "
                         "Default location: models/contrastive_team_rilh.pt")
    g1.add_argument("--ref-classifier", type=str, default=None,
                    help="Stage 1.b: optional path to a referee binary "
                         "classifier (tools/finetune_ref_classifier.py "
                         "output). When set, every track is tagged with "
                         "is_referee + ref_score in p1_b_teams.json. "
                         "Orthogonal to --team-engine.")
    g1.add_argument("--parseq-checkpoint", type=str, default=None,
                    help="Stage 1.c: custom PARSeq checkpoint (default = baudm "
                         "pretrained); use models/parseq_hockey_rilh.pt for "
                         "the RILH-fine-tuned hockey model")

    args = p.parse_args()

    video = Path(args.video).resolve()
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    out = out.resolve()

    enabled = [f"Phase {n}" for n in (1, 2, 3, 4, 5)
               if not getattr(args, f"skip_p{n}")]
    print("=" * 64)
    print(f"RILH-AI-Vision project orchestrator")
    print(f"  started : {datetime.datetime.now(datetime.UTC).isoformat()}")
    print(f"  video   : {video}")
    print(f"  output  : {out}")
    print(f"  phases  : {' → '.join(enabled) if enabled else '(none — all skipped)'}")
    print(f"  force   : {args.force}")
    print("=" * 64)

    t_start = time.time()
    if not args.skip_p1:
        run_p1_detect_track(video, out, args)
    if not args.skip_p2:
        run_p2_rink(video, out, args)
    if not args.skip_p3:
        run_p3_entity(video, out, args)
    if not args.skip_p4:
        run_p4_events(out, args)
    if not args.skip_p5:
        run_p5_stats(out, args)

    dt = time.time() - t_start
    print("\n" + "=" * 64)
    print(f"All requested phases complete in {dt:.0f}s.")
    print(f"Artifacts in: {out}")
    print("=" * 64)


if __name__ == "__main__":
    main()
