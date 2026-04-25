"""
RILH-AI-Vision — phase 5, stage a : statistics creation (STUB).

Placeholder for the future per-player / per-team statistics-aggregation
stage. Will consume the entity-level outputs (track positions, jersey
numbers, team assignments) and roll them up into ice-time, shots,
saves, possession, etc. — once event detection (P4) is real, this
stage will key on those events too.

Inputs:
  p1_d_entities.json (P1.d)
  p1_c_numbers.json  (P1.c)

Outputs:
  p5_a_stats.json — {status: "stub_no_op", ran_at: "<ISO timestamp>"}

The orchestrator (`src/run_project.py`) calls this script as part of
the default pipeline so the wiring is exercised. The stub does nothing
but write a marker proving it ran.
"""

import argparse
import datetime
import json
from pathlib import Path


def main():
    """CLI entry point — write a no-op marker so the orchestrator can
    confirm the stage ran in a given run folder."""
    p = argparse.ArgumentParser(description="P5.a statistics stub (no-op)")
    p.add_argument("entities_json", type=str,
                   help="P1.d output (p1_d_entities.json) — read by future impl")
    p.add_argument("numbers_json", type=str,
                   help="P1.c output (p1_c_numbers.json) — read by future impl")
    p.add_argument("--output", required=True, type=str,
                   help="Output marker JSON path")
    args = p.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "stub_no_op",
        "phase": "P5 (statistics creation)",
        "stage": "a",
        "ran_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "inputs": {
            "entities_json": str(args.entities_json),
            "numbers_json": str(args.numbers_json),
        },
        "note": (
            "Placeholder. Replace with real per-player / per-team "
            "statistics aggregation once requirements are set."
        ),
    }
    output.write_text(json.dumps(payload, indent=2))
    print(f"P5.a stats: stub no-op marker → {output}")


if __name__ == "__main__":
    main()
