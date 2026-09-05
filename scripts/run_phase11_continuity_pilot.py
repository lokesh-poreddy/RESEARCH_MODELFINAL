"""scripts/run_phase11_continuity_pilot.py — Executes Phase 11 Continuity Benchmark Pilot.

Executes:
- 3 tasks (digits_0_4 [SAME_FAMILY], digits_5_9 [SAME_FAMILY], synthetic_ecg_lead1 [CROSS_FAMILY])
- 3 task orderings (forward, reverse, cross_first)
- 2 seeds ([0, 1])
- Experimental conditions:
  - COLD_START: Fresh researcher state for each task
  - NO_MEMORY: Memory contribution explicitly disabled
  - FLAT_ECRM: Flat strategy-keyed ECRM accumulated across tasks
  - TRAJECTORY_MEMORY: Capacity-bucketed trajectory memory accumulated across tasks
  - ADAPTIVE_TRAJECTORY: Hierarchical context-sensitive memory with backoff
  - CONTINUOUS_EXPERIENCE: Unified accumulated experience inherited sequentially

Saves canonical immutable artifact to: phase11_continuity_benchmark_result.json
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchforge.benchmarks.continuity import (
    ContinuityCondition,
    ContinuityHarness,
    get_pilot_task_sequence,
)


def run_phase11_pilot(
    output_filename: str = "phase11_continuity_benchmark_result.json",
    n_generations: int = 5,
    population_size: int = 6,
) -> None:
    print("================================================================================")
    print("           RESEARCHFORGE PHASE 11: RESEARCH CONTINUITY BENCHMARK PILOT          ")
    print("================================================================================")
    print("Primary Scientific Question: Does accumulated research experience improve")
    print("ResearchForge's later research decisions on subsequent problems?")
    print("Non-causal, empirical evaluation. Negative and neutral transfer are valid outcomes.")
    print("--------------------------------------------------------------------------------")

    tasks = get_pilot_task_sequence(seed=0)
    orderings = ["forward", "reverse", "cross_first"]
    seeds = [0, 1]
    conditions = [
        ContinuityCondition.COLD_START,
        ContinuityCondition.NO_MEMORY,
        ContinuityCondition.FLAT_ECRM,
        ContinuityCondition.TRAJECTORY_MEMORY,
        ContinuityCondition.ADAPTIVE_TRAJECTORY,
        ContinuityCondition.CONTINUOUS_EXPERIENCE,
    ]

    print(f"Tasks ({len(tasks)}): {[t.name for t in tasks]}")
    print(f"Orderings ({len(orderings)}): {orderings}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Conditions ({len(conditions)}): {[c.value for c in conditions]}")
    print(f"Budget per task run: {n_generations} generations, population={population_size}")
    print("--------------------------------------------------------------------------------")

    harness = ContinuityHarness(
        tasks=tasks,
        conditions=conditions,
        orderings=orderings,
        seeds=seeds,
        n_generations=n_generations,
        population_size=population_size,
    )

    t0 = time.time()
    out_path = PROJECT_ROOT / output_filename
    artifact = harness.run(output_path=str(out_path))
    elapsed = time.time() - t0

    print("--------------------------------------------------------------------------------")
    print(f"Pilot execution completed in {elapsed:.2f}s.")
    print(f"Total raw observations collected: {len(artifact.raw_observations)}")
    print(f"Total transfer events audited: {len(artifact.transfer_classifications)}")
    print(f"Learning-curve evaluation points: {len(artifact.learning_curve_telemetry)}")
    print(f"Artifact fingerprint: {artifact.artifact_fingerprint}")
    print(f"Canonical benchmark artifact written to: {out_path}")
    print("================================================================================")


if __name__ == "__main__":
    run_phase11_pilot()
