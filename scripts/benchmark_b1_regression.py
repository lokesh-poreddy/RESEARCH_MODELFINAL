"""scripts/benchmark_b1_regression.py — B1: RF-2 Regression Benchmark.

Verifies that the RF-2 contextual policy (Phase 12) does not regress against
the RF-1 baseline on a standard cold-start task (digits_all, n=25 generations).

Metrics reported:
  - avg_best_metric: Mean of best per-trial metric across seeds.
  - auc_running_best: Area under the running-best curve (measures anytime performance).
  - time_to_target: Mean generations to first exceed target accuracy (0.90).
  - std_best_metric: Standard deviation across seeds (stability).

Pass/Fail criterion:
  RF-2 contextual_policy avg_best_metric must be >= RF-1 full * 0.98.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from researchforge.benchmarks.continuity.tasks import digits_all_task
from researchforge.pipeline.controller import ResearchController

TARGET_ACCURACY = 0.90
TOLERANCE_FACTOR = 0.98  # RF-2 must reach ≥ 98 % of RF-1 full


def _running_best(metrics: list[float]) -> list[float]:
    """Converts a per-trial metric list into a running-best curve."""
    best = float("-inf")
    curve = []
    for m in metrics:
        best = max(best, m)
        curve.append(best)
    return curve


def _auc(curve: list[float]) -> float:
    """Normalised AUC of the running-best curve in [0, 1]."""
    return float(np.mean(curve))


def _time_to_target(curve: list[float], target: float) -> int:
    """First generation (1-indexed) where running-best >= target, else len(curve)."""
    for i, v in enumerate(curve):
        if v >= target:
            return i + 1
    return len(curve)


def run_b1() -> dict:
    print("=" * 60)
    print("BENCHMARK B1: RF-2 Regression")
    print("=" * 60)

    conditions = ["no_memory", "random", "full", "contextual_policy"]
    seeds = [42, 43, 44]
    n_gens = 25

    results_summary: dict = {}

    for cond in conditions:
        print(f"\n  Condition: {cond}")
        best_metrics, aucs, ttts = [], [], []

        for seed in seeds:
            task = digits_all_task(seed=seed).task
            ctrl = ResearchController(task, condition=cond, seed=seed)
            res = ctrl.run(n_generations=n_gens)

            trial_metrics = [t.metric for t in res.trials]
            curve = _running_best(trial_metrics)

            best_metrics.append(curve[-1])
            aucs.append(_auc(curve))
            ttts.append(_time_to_target(curve, TARGET_ACCURACY))
            print(f"    seed={seed}  best={curve[-1]:.4f}  auc={_auc(curve):.4f}  ttt={_time_to_target(curve, TARGET_ACCURACY)}")

        results_summary[cond] = {
            "avg_best_metric": float(np.mean(best_metrics)),
            "std_best_metric": float(np.std(best_metrics)),
            "avg_auc_running_best": float(np.mean(aucs)),
            "avg_time_to_target": float(np.mean(ttts)),
        }

    print("\n--- RESULTS B1 ---")
    print(json.dumps(results_summary, indent=2))

    rf1 = results_summary["full"]["avg_best_metric"]
    rf2 = results_summary["contextual_policy"]["avg_best_metric"]
    pass_threshold = rf1 * TOLERANCE_FACTOR

    print(f"\nRF-1 full:              {rf1:.4f}")
    print(f"RF-2 contextual_policy: {rf2:.4f}")
    print(f"Pass threshold (98%):   {pass_threshold:.4f}")

    if rf2 >= pass_threshold:
        print("RESULT: ✅ PASS — RF-2 does not regress against RF-1.")
    else:
        print(f"RESULT: ❌ FAIL — RF-2 regressed by {(rf1 - rf2) / rf1 * 100:.1f}%.")

    return results_summary


if __name__ == "__main__":
    run_b1()
