"""scripts/benchmark_b10_ablation.py — B10: Full RF-2 Ablation Study.

Ablates the Phase 12 mechanisms one component at a time to isolate
which parts of the RF-2 architecture are contributing to performance.

Conditions compared:
  A: Full RF-2 (contextual_policy + active counterfactual sampling)
  B: RF-2 without active sampling (sampling disabled)
  C: RF-1 legacy transfer gate (trajectory_memory)
  D: Fully random (no learning)

Metrics:
  - avg_best_metric ± std (3 seeds)
  - avg_auc_running_best
  - avg_time_to_target (>= 0.88)

Statistical comparison: Cohen's d (effect size) between A and each ablation.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from researchforge.benchmarks.continuity.tasks import digits_all_task
from researchforge.pipeline.controller import ResearchController

TARGET_ACCURACY = 0.88
N_SEEDS = 3
N_GENS = 15


def _running_best(metrics: list[float]) -> list[float]:
    best = float("-inf")
    result = []
    for m in metrics:
        best = max(best, m)
        result.append(best)
    return result


def _time_to_target(curve: list[float], target: float) -> int:
    for i, v in enumerate(curve):
        if v >= target:
            return i + 1
    return len(curve)


def cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d effect size (pooled std denominator)."""
    mean_a, mean_b = np.mean(a), np.mean(b)
    std_a, std_b = np.std(a, ddof=1), np.std(b, ddof=1)
    pooled_std = np.sqrt((std_a**2 + std_b**2) / 2.0)
    if pooled_std < 1e-9:
        return 0.0
    return float((mean_a - mean_b) / pooled_std)


def run_condition(label: str, condition: str, disable_sampling: bool = False) -> dict:
    print(f"\n  [{label}] Condition: {condition}  (disable_sampling={disable_sampling})")
    best_metrics, aucs, ttts = [], [], []

    for seed in range(42, 42 + N_SEEDS):
        task = digits_all_task(seed=seed).task
        ctrl = ResearchController(task, condition=condition, seed=seed)

        if disable_sampling:
            # Disable active counterfactual sampling by zeroing the budget
            ctrl._sampling_policy.budget_fraction = 0.0

        res = ctrl.run(n_generations=N_GENS)

        trial_metrics = [t.metric for t in res.trials]
        curve = _running_best(trial_metrics)
        bm = curve[-1]
        auc = float(np.mean(curve))
        ttt = _time_to_target(curve, TARGET_ACCURACY)

        best_metrics.append(bm)
        aucs.append(auc)
        ttts.append(ttt)
        print(f"    seed={seed}  best={bm:.4f}  auc={auc:.4f}  ttt={ttt}")

    return {
        "label": label,
        "condition": condition,
        "disable_sampling": disable_sampling,
        "avg_best_metric": float(np.mean(best_metrics)),
        "std_best_metric": float(np.std(best_metrics)),
        "avg_auc_running_best": float(np.mean(aucs)),
        "avg_time_to_target": float(np.mean(ttts)),
        "_raw_best_metrics": best_metrics,
    }


def run_b10() -> dict:
    print("=" * 60)
    print("BENCHMARK B10: Full RF-2 Ablation Study")
    print("=" * 60)

    conditions = [
        ("A: Full RF-2",                    "contextual_policy", False),
        ("B: RF-2 no active sampling",       "contextual_policy", True),
        ("C: RF-1 trajectory_memory",        "trajectory_memory", False),
        ("D: Random (no learning)",          "random",            False),
    ]

    all_results = {}
    raw = {}

    for label, cond, disable_samp in conditions:
        r = run_condition(label, cond, disable_samp)
        all_results[label] = r
        raw[label] = r.pop("_raw_best_metrics")

    # Print summary table
    print("\n--- ABLATION SUMMARY TABLE ---")
    header = f"{'Condition':<30} {'AvgBest':>9} {'StdBest':>9} {'AUC':>9} {'TTT':>6}  {'Cohen-d vs A':>14}"
    print(header)
    print("-" * len(header))

    baseline_raw = raw["A: Full RF-2"]
    for label, r in all_results.items():
        d = cohens_d(baseline_raw, raw[label]) if label != "A: Full RF-2" else 0.0
        print(
            f"  {label:<28} {r['avg_best_metric']:>9.4f} {r['std_best_metric']:>9.4f} "
            f"{r['avg_auc_running_best']:>9.4f} {r['avg_time_to_target']:>6.1f}  {d:>+14.3f}"
        )

    print("\n(Cohen's d > 0 means Full RF-2 outperforms the ablation.)")
    return all_results


if __name__ == "__main__":
    run_b10()
