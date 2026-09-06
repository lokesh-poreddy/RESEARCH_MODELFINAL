"""Run the bounded RF-FLEX Phase 13.7 pilot."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from researchforge.benchmarks.rf_flex import RFFlexPilot


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RF-FLEX full-level pilot")
    parser.add_argument("--output", default="researchforge_full_eval", help="artifact directory")
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population-size", type=int, default=4)
    args = parser.parse_args()
    summary = RFFlexPilot(
        n_generations=args.generations,
        population_size=args.population_size,
    ).run(Path(args.output))
    print(summary)


if __name__ == "__main__":
    main()