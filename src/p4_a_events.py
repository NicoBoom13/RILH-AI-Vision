"""
RILH-AI-Vision — phase 4, stage a : event detection (STUB).

Placeholder for the future event-detection stage (goals, shots, fouls
via temporal action models like TSN / MoViNet / SlowFast). The
orchestrator (`src/run_project.py`) calls this script as part of the
default pipeline so the wiring is exercised end-to-end, but the stub
does no real inference — it only writes a marker JSON proving it ran.

Inputs:
  p1_a_detections.json (P1.a)
  p3_a_entities.json   (P3.a)

Outputs:
  p4_a_events.json — {status: "stub_no_op", ran_at: "<ISO timestamp>"}

When the real detector lands, replace this stub with the actual
implementation and keep the same CLI signature so the orchestrator
contract is unchanged.
"""

import argparse
import datetime
import json
from pathlib import Path


def main():
    """CLI entry point — write a no-op marker so the orchestrator can
    confirm the stage ran in a given run folder."""
    p = argparse.ArgumentParser(description="P4.a events stub (no-op)")
    p.add_argument("detections_json", type=str,
                   help="P1.a output (p1_a_detections.json) — read by future impl")
    p.add_argument("entities_json", type=str,
                   help="P3.a output (p3_a_entities.json) — read by future impl")
    p.add_argument("--output", required=True, type=str,
                   help="Output marker JSON path")
    args = p.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "stub_no_op",
        "phase": "P4 (event detection)",
        "stage": "a",
        "ran_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "inputs": {
            "detections_json": str(args.detections_json),
            "entities_json": str(args.entities_json),
        },
        "note": (
            "Placeholder. Replace with real temporal-action-model inference "
            "(goals / shots / fouls) once that work starts."
        ),
    }
    output.write_text(json.dumps(payload, indent=2))
    print(f"P4.a events: stub no-op marker → {output}")


if __name__ == "__main__":
    main()
