"""Phase 7 Benchmark: Adaptive Trajectory ECRM Evaluation.

Evaluates hierarchical context backoff against flat ECRM, static trajectory memory,
and no-memory baselines under a fixed research budget (fixed generations/seeds).

Tracks:
- Performance (best metric mean, std)
- Research Efficiency (RE)
- Search Efficiency (SE)
- Failure Repetition Rate (FRR)
- Negative Transfer Rate (NTR)
- Memory Utility (MU)
- Context level distribution (L3, L2, L1, L0)
- Backoff frequency
- Failure-blocking events
- Memory-influenced decisions

Writes machine-readable output to `phase7_benchmark_result.json`.
Does NOT overwrite historical benchmark artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchforge.benchmarks.tasks import digits_task
from researchforge.benchmarks.rde_bench import run_condition, BenchSummary, CONDITIONS


def run_phase7_benchmark(
    seeds: list[int] = (0, 1, 2),
    n_generations: int = 15,
) -> dict:
    task = digits_task(seed=0)
    print(f"Running Phase 7 Benchmark on '{task.name}' across seeds {seeds}, n_gen={n_generations}...")
    t0 = time.time()

    results: dict[str, BenchSummary] = {}
    for cond in CONDITIONS:
        print(f"  Evaluating condition: {cond}...")
        results[cond] = run_condition(task, cond, list(seeds), n_generations)

    # Compute Memory Utility relative to no_memory
    nomem_best = results["no_memory"].best_metric_mean
    for m_cond in ("full", "trajectory_memory", "adaptive_trajectory"):
        if m_cond in results:
            results[m_cond].memory_utility = results[m_cond].best_metric_mean - nomem_best

    elapsed = time.time() - t0

    # Build machine-readable summary
    summary_data = {
        "timestamp": time.time(),
        "task_name": task.name,
        "n_generations": n_generations,
        "seeds": list(seeds),
        "elapsed_seconds": elapsed,
        "conditions": {},
    }

    for cond_name, s in results.items():
        summary_data["conditions"][cond_name] = {
            "best_metric_mean": s.best_metric_mean,
            "best_metric_std": s.best_metric_std,
            "research_efficiency": s.research_efficiency,
            "search_efficiency": s.search_efficiency,
            "failure_repetition_rate": s.failure_repetition_rate,
            "negative_transfer_rate": s.negative_transfer_rate,
            "memory_utility": s.memory_utility,
            "context_level_distribution": s.context_level_distribution,
            "backoff_frequency": s.backoff_frequency,
            "failure_blocking_events": s.failure_blocking_events,
            "memory_influenced_decisions": s.memory_influenced_decisions,
        }

    out_path = PROJECT_ROOT / "phase7_benchmark_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nWrote benchmark results to {out_path}")
    print("\n=== Phase 7 Benchmark Summary ===")
    print(f"{'Condition':<22}{'Best':>8}{'RE':>8}{'SE':>6}{'FRR':>7}{'NTR':>7}{'MU':>9}{'Backoff%':>10}")
    for cond_name in CONDITIONS:
        s = results[cond_name]
        mu_str = f"{s.memory_utility:+.4f}" if cond_name in ("full", "trajectory_memory", "adaptive_trajectory") else "--"
        bk_str = f"{s.backoff_frequency * 100:.1f}%" if cond_name == "adaptive_trajectory" else "--"
        print(f"{cond_name:<22}{s.best_metric_mean:>8.4f}{s.research_efficiency:>8.4f}"
              f"{s.search_efficiency:>6d}{s.failure_repetition_rate:>7.2f}"
              f"{s.negative_transfer_rate:>7.2f}{mu_str:>9}{bk_str:>10}")

    if "adaptive_trajectory" in results:
        at = results["adaptive_trajectory"]
        print("\n--- Adaptive Trajectory Telemetry ---")
        print(f"Context Level Distribution: {at.context_level_distribution}")
        print(f"Backoff Frequency: {at.backoff_frequency * 100:.2f}%")
        print(f"Failure-blocking events: {at.failure_blocking_events}")
        print(f"Memory-influenced decisions: {at.memory_influenced_decisions}")

    return summary_data


if __name__ == "__main__":
    # Use 2 seeds and 10 generations for rapid deterministic benchmark execution
    run_phase7_benchmark(seeds=[0, 1], n_generations=10)
