"""
RILH-AI-Vision — project orchestrator.

Runs the full project pipeline (Phase 1 → Phase 5) on a single video,
with per-phase gates. Each stage script is invoked as a subprocess so
that it stays independently runnable; this orchestrator only knows the
glue (which file feeds which, which flags pass through, what to skip).

Pipeline (in execution order):
  Phase 1 — Detect & track       — 5 stages (a..e), full identification
  Phase 2 — Virtual follow-cam   — 1 stage (a), broadcast camera
  Phase 3 — Rink calibration     — 1 stage (a), parked / tolerant of failure
  Phase 4 — Event detection      — 1 stage (a), STUB
  Phase 5 — Statistics creation  — 1 stage (a), STUB

Phase 6 (Web platform) and Phase 7 (Multi-cam / live) are not part of the per-run
pipeline — they are separate services that consume the run folder.

Usage:
  python src/run_project.py videos/match.mp4 --output runs/run23 \\
      --hockey-model --pose-model yolo26l-pose.pt \\
      --parseq-checkpoint models/parseq_hockey_rilh.pt

  # Skip a phase entirely:
  python src/run_project.py videos/match.mp4 --output runs/run23 --skip-p3

  # Force re-run even if outputs already exist:
  python src/run_project.py videos/match.mp4 --output runs/run23 --force

By default each phase is ON, and each stage skips its own work if its
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
        Nothing — caller decides what to do on failure (e.g. Phase 3 is
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


def run_p1_detect_track(video, out, args):
    """Run all 5 stages of Phase 1 (a → b → c → d → e) sequentially.

    Halts on first failure: each stage's outputs feed the next, so a
    failed Stage 1.a means Stage 1.b can't run.
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
                "--tracker", args.tracker])
    if not step("Stage 1.a — Detect & track", cmd,
                [out / "p1_a_detections.json"], args.force):
        raise SystemExit("Stage 1.a failed — halting Phase 1")

    # Stage 1.b — teams
    cmd = [PYTHON, "-u", str(SRC / "p1_b_teams.py"),
           str(out / "p1_a_detections.json"), str(video),
           "--pose-model", args.pose_model]
    if not step("Stage 1.b — Teams", cmd,
                [out / "p1_b_teams.json"], args.force):
        raise SystemExit("Stage 1.b failed — halting Phase 1")

    # Stage 1.c — numbers (jersey OCR)
    cmd = [PYTHON, "-u", str(SRC / "p1_c_numbers.py"),
           str(out / "p1_a_detections.json"), str(video),
           "--pose-model", args.pose_model]
    if args.parseq_checkpoint:
        cmd.extend(["--parseq-checkpoint", args.parseq_checkpoint])
    if not step("Stage 1.c — Numbers", cmd,
                [out / "p1_c_numbers.json"], args.force):
        raise SystemExit("Stage 1.c failed — halting Phase 1")

    # Stage 1.d — entities (Re-ID merge)
    cmd = [PYTHON, "-u", str(SRC / "p1_d_entities.py"),
           str(out / "p1_a_detections.json"), str(out / "p1_b_teams.json"),
           str(out / "p1_c_numbers.json"), str(video)]
    if not step("Stage 1.d — Entities", cmd,
                [out / "p1_d_entities.json"], args.force):
        raise SystemExit("Stage 1.d failed — halting Phase 1")

    # Stage 1.e — annotate (final MP4)
    cmd = [PYTHON, "-u", str(SRC / "p1_e_annotate.py"),
           str(out / "p1_a_detections.json"), str(out / "p1_c_numbers.json"),
           str(video), "--output", str(out / "annotated.mp4")]
    if not step("Stage 1.e — Annotate", cmd,
                [out / "annotated.mp4"], args.force):
        raise SystemExit("Stage 1.e failed — halting Phase 1")


def run_p2_followcam(video, out, args):
    """Run Phase 2 — Virtual follow-cam. Depends on p1_a_detections.json from Stage 1.a."""
    print(f"\n========== Phase 2 — Virtual follow-cam ==========")
    if not (out / "p1_a_detections.json").exists():
        print("  ⚠ p1_a_detections.json missing — Phase 2 needs Stage 1.a; skipping")
        return
    cmd = [PYTHON, "-u", str(SRC / "p2_a_followcam.py"),
           str(out / "p1_a_detections.json"), str(video),
           "--output", str(out / "followcam.mp4")]
    step("Stage 2.a — Follow-cam", cmd, [out / "followcam.mp4"], args.force)


def run_p3_rink(video, out, args):
    """Run Phase 3 — Rink calibration. Currently parked: HockeyRink doesn't
    transfer to roller rinks. We still run it so the orchestrator wiring
    is exercised, but a non-zero exit code is logged and ignored — it
    must not block downstream phases."""
    print(f"\n========== Phase 3 — Rink calibration (parked) ==========")
    cmd = [PYTHON, "-u", str(SRC / "p3_a_rink.py"),
           str(video), "--output", str(out)]
    ok = step("Stage 3.a — Rink keypoints", cmd,
              [out / "p3_a_rink_keypoints.json"], args.force)
    if not ok:
        print("  ⚠ Phase 3 is parked (HockeyRink doesn't transfer to roller). "
              "Continuing.")


def run_p4_events(out, args):
    """Run Phase 4 — Event detection (stub). Writes a marker JSON."""
    print(f"\n========== Phase 4 — Event detection (stub) ==========")
    if not (out / "p1_a_detections.json").exists() or not (out / "p1_d_entities.json").exists():
        print("  ⚠ Phase 4 needs p1_a_detections.json + p1_d_entities.json — skipping")
        return
    cmd = [PYTHON, "-u", str(SRC / "p4_a_events.py"),
           str(out / "p1_a_detections.json"), str(out / "p1_d_entities.json"),
           "--output", str(out / "p4_a_events.json")]
    step("Stage 4.a — Events", cmd, [out / "p4_a_events.json"], args.force)


def run_p5_stats(out, args):
    """Run Phase 5 — Statistics creation (stub). Writes a marker JSON."""
    print(f"\n========== Phase 5 — Statistics (stub) ==========")
    if not (out / "p1_d_entities.json").exists() or not (out / "p1_c_numbers.json").exists():
        print("  ⚠ Phase 5 needs p1_d_entities.json + p1_c_numbers.json — skipping")
        return
    cmd = [PYTHON, "-u", str(SRC / "p5_a_stats.py"),
           str(out / "p1_d_entities.json"), str(out / "p1_c_numbers.json"),
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
                   help="Output run folder, e.g. runs/run23/")

    # Phase gates — each phase is ON by default
    g = p.add_argument_group("Phase gates")
    g.add_argument("--skip-p1", action="store_true",
                   help="Skip Phase 1 (Detect & track + teams + numbers + entities + annotate)")
    g.add_argument("--skip-p2", action="store_true",
                   help="Skip Phase 2 (Virtual follow-cam)")
    g.add_argument("--skip-p3", action="store_true",
                   help="Skip Phase 3 (Rink calibration, parked)")
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
    g1.add_argument("--model", type=str, default=None,
                    help="Stage 1.a: COCO YOLO weights (ignored when --hockey-model)")
    g1.add_argument("--conf", type=float, default=0.3,
                    help="Stage 1.a: detection confidence threshold")
    g1.add_argument("--imgsz", type=int, default=1280,
                    help="Stage 1.a: inference image size")
    g1.add_argument("--tracker", type=str, default="bytetrack.yaml",
                    help="Stage 1.a: tracker config")
    g1.add_argument("--pose-model", type=str, default="yolo26l-pose.pt",
                    help="Stage 1.b + Stage 1.c: YOLO pose weights")
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

    enabled = []
    for n in (1, 2, 3, 4, 5):
        if not getattr(args, f"skip_p{n}"):
            enabled.append(f"Phase {n}")
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
        run_p2_followcam(video, out, args)
    if not args.skip_p3:
        run_p3_rink(video, out, args)
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
