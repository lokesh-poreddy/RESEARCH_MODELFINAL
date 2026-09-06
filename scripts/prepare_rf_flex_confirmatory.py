"""Freeze the Phase 13.9 design and calibrate resources from a Phase 13.8 pilot."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from researchforge.benchmarks.rf_flex_confirmatory import (  # noqa: E402
    RFFlexConfirmatoryProtocol,
    calibrate_from_pilot,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze and calibrate RF-FLEX Phase 13.9")
    parser.add_argument("--pilot", required=True, help="Phase 13.8 pilot artifact directory")
    parser.add_argument("--output", default="researchforge_full_eval", help="artifact directory")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = RFFlexConfirmatoryProtocol()
    protocol.freeze(output / "rf_flex_13_9_protocol.json")
    calibration = calibrate_from_pilot(args.pilot, protocol)
    (output / "rf_flex_13_9_calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"protocol": protocol.fingerprint(), "calibration": calibration}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()