"""Create the clean-commit RF-FLEX Phase 13.11 release seal.

The command fails on a dirty repository and never authorizes execution.
Write output outside the repository so generated artifacts cannot dirty the
execution commit being sealed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from researchforge.benchmarks.rf_flex_execution import create_release_seal  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Create RF-FLEX Phase 13.11 release seal")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", required=True, help="Output directory outside the repository")
    args = parser.parse_args()
    seal = create_release_seal(args.protocol, Path(args.repo_root).resolve(), args.output)
    print(json.dumps(seal.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()