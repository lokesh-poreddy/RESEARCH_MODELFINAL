"""Phase 7 Example: Adaptive Trajectory ECRM in Action.

Demonstrates:
1. Running ResearchController under condition='adaptive_trajectory'.
2. Storing and querying context-sensitive trajectory records.
3. Hierarchical deterministic backoff (L3 -> L2 -> L1 -> L0).
4. Evidence-based confidence (derived from dispersion, not context level).
5. Failure safety checks (FailureCheckResult).
6. Auditable memory influence and trial telemetry.
7. Diagnostic oracle-backoff retrieval.
8. Provenance chain reconstruction into RDG/VRDEG.

Run: `python3 scripts/phase7_example.py`
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchforge.benchmarks.tasks import digits_task
from researchforge.pipeline.controller import ResearchController
from researchforge.memory.adaptive_trajectory import AdaptiveTrajectoryMemory


def main() -> None:
    print("=== ResearchForge Phase 7: Adaptive Trajectory ECRM Example ===\n")

    task = digits_task(seed=0)
    print(f"Task: {task.name} (target: {task.target_metric})")

    ctrl = ResearchController(task, condition="adaptive_trajectory", seed=42)
    print("Running ResearchController with condition='adaptive_trajectory' for 6 generations...")
    run_result = ctrl.run(n_generations=6)

    print(f"Run completed in {run_result.wall_time_s:.2f}s")
    print(f"Best Metric: {run_result.best_metric:.4f}")
    print(f"Total Trials: {len(run_result.trials)} (1 baseline + 6 generations)")

    mem: AdaptiveTrajectoryMemory = ctrl.adaptive_trajectory_memory
    stats = mem.stats()
    print(f"\nAdaptive Trajectory Stats: {json.dumps(stats, indent=2)}")

    print("\n--- Trial Decision Auditing (What memory told the system & how it influenced choices) ---")
    for t in run_result.trials[1:]:
        print(f"\nGeneration {t.generation} | Strategy: {t.strategy} | Model: {t.model_type} | Metric: {t.metric:.4f}")
        print(f"  Context Level: {t.context_level} (Requested: {t.requested_context_level}) | Samples: {t.sample_count}")
        print(f"  Fallback Reason: {t.fallback_reason}")
        if t.memory_decision_contribution:
            contrib = t.memory_decision_contribution
            print(f"  Memory Recommendation: {contrib.get('memory_recommendation')}")
            print(f"  Baseline Score: {contrib.get('baseline_strategy_score'):.4f} -> Final Score: {contrib.get('final_strategy_score'):.4f}")

    print("\n--- Diagnostic Oracle-Backoff Demonstration ---")
    parent_type = "LogisticRegression"
    parent_bucket = "medium"
    strategy = "tune_regularization"

    auto_res = mem.query_context(strategy, parent_type, parent_bucket)
    print(f"Automatic Query for ({strategy}, {parent_type}, {parent_bucket}):")
    print(f"  Selected Level: {auto_res.context_level} | Samples: {auto_res.sample_count} | Confidence: {auto_res.confidence:.4f}")
    print(f"  Fallback Reason: {auto_res.fallback_reason}")

    oracle_l3 = mem.query_oracle(strategy, parent_type, parent_bucket, oracle_level=3)
    print(f"\nOracle Query (Forced Level 3):")
    print(f"  Level: {oracle_l3.context_level} | Samples: {oracle_l3.sample_count} | Mean: {oracle_l3.mean_metric}")
    print(f"  Reason: {oracle_l3.fallback_reason}")

    oracle_l0 = mem.query_oracle(strategy, parent_type, parent_bucket, oracle_level=0)
    print(f"\nOracle Query (Forced Level 0 - Global):")
    print(f"  Level: {oracle_l0.context_level} | Samples: {oracle_l0.sample_count} | Mean: {oracle_l0.mean_metric}")

    print("\nPhase 7 example executed successfully.")


if __name__ == "__main__":
    main()
