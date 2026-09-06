"""Prepare the Phase 13.10 RF-FLEX pre-execution attestation.

This command never authorizes or launches the confirmatory evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from researchforge.benchmarks.rf_flex_execution import prepare_attestation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare RF-FLEX Phase 13.10 attestation")
    parser.add_argument("--protocol", required=True, help="Frozen Phase 13.9 protocol JSON")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", default="rf_flex_13_10_pre_execution_attestation.json")
    args = parser.parse_args()
    attestation = prepare_attestation(Path(args.protocol), Path(args.repo_root).resolve())
    data = attestation.write(args.output)
    print(json.dumps({
        "preflight_pass": data["preflight_pass"],
        "authorization": data["authorization"],
        "software_commit": data["software_commit"],
        "dirty_worktree": data["dirty_worktree"],
        "attestation_fingerprint": data["attestation_fingerprint"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()