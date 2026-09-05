"""scripts/run_phase12_validation.py — Executes Phase 12A Statistical Method Validation.

Loads the frozen Phase 11 pilot artifact, evaluates pre-registered hypotheses H1–H4,
computes blocked bootstrap CIs and permutation tests, applies Holm-Bonferroni correction
to the secondary family (H2–H4) without penalizing primary H1, runs 5 sensitivity analyses,
and writes the fingerprinted statistical validation artifact:
`phase12_statistical_validation.json`.

EXPLICIT STATUS: METHOD_VALIDATION (NOT CONFIRMATORY).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchforge.benchmarks.statistical import (
    StatisticalAnalysisEngine,
    ValidationStatus,
    create_canonical_sap,
)


def run_phase12_validation() -> None:
    print("================================================================================")
    print("      RESEARCHFORGE PHASE 12A: STATISTICAL ANALYSIS PLAN & METHOD VALIDATION    ")
    print("================================================================================")
    print("Status: METHOD VALIDATION ONLY (Pilot Data) — NOT CONFIRMATORY")
    print("Pre-registered Hypotheses: H1 (Primary), H2-H4 (Secondary)")
    print("Primary Endpoint: Decision Quality (DQ)")
    print("--------------------------------------------------------------------------------")

    sap = create_canonical_sap()
    print(f"Locked SAP Version: {sap.plan_version}")
    print(f"SAP Fingerprint: {sap.plan_fingerprint}")
    print(f"Primary Endpoint: {sap.primary_endpoint}")
    print(f"Secondary Family Correction: {sap.multiplicity_procedure_secondary}")
    print(f"Resampling Unit: {sap.resampling_unit}")
    print("--------------------------------------------------------------------------------")

    raw_pilot_path = PROJECT_ROOT / "phase11_continuity_benchmark_result.json"
    if not raw_pilot_path.exists():
        raise FileNotFoundError(f"Frozen Phase 11 pilot artifact not found at {raw_pilot_path}")

    out_path = PROJECT_ROOT / "phase12_statistical_validation.json"

    engine = StatisticalAnalysisEngine(plan=sap, random_seed=42)
    artifact = engine.generate_statistical_artifact(
        raw_benchmark_artifact_path=str(raw_pilot_path),
        validation_status=ValidationStatus.METHOD_VALIDATION,
        output_path=str(out_path),
    )

    print("=== HYPOTHESIS EVALUATION RESULTS (PILOT METHOD VALIDATION) ===")
    for h in artifact.hypothesis_results:
        print(f"  [{h['hypothesis_id']}] {h['description']}")
        print(f"       Endpoint: {h['endpoint']}")
        print(f"       Raw p-value: {h['raw_p_value']:.6f} | Adjusted p-value: {h['adjusted_p_value']:.6f} ({h['multiplicity_method']})")
        print(f"       Effect ({h['effect_metric']}): {h['effect_size']:+.4f} | 95% CI: [{h['ci_95'][0]:.4f}, {h['ci_95'][1]:.4f}]")
        print(f"       Decision: {h['decision']}")

    print("\n=== SENSITIVITY ANALYSES SUMMARY ===")
    print(f"Total sensitivity checks executed: {len(artifact.sensitivity_results)}")
    for s in artifact.sensitivity_results[:5]:
        print(f"  - {s['dimension']} [{s['stratum']}]: mean_diff={s['mean_difference']:+.4f}, CI=[{s['ci_lower_95']:.4f}, {s['ci_upper_95']:.4f}], status={s['stability']}")

    print("--------------------------------------------------------------------------------")
    print(f"Statistical Artifact Fingerprint: {artifact.artifact_fingerprint}")
    print(f"Validation Artifact Written to: {out_path}")
    print("================================================================================")


if __name__ == "__main__":
    run_phase12_validation()
