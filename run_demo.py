"""End-to-end demo: builds the two RDE-Bench tasks, runs the full ablation
ladder (full system / no-memory / random search, design doc Sec. 6/8) with
several seeds each, prints a comparison report, and dumps raw results +
best-so-far curves to demo_results.json.

Usage:
    python run_demo.py                  # default: 5 seeds, 25 generations
    python run_demo.py --seeds 3 --generations 15   # faster, noisier
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")  # sklearn convergence warnings on tiny/noisy tasks

from researchforge.benchmarks.tasks import digits_task, synthetic_ecg_task
from researchforge.benchmarks.rde_bench import run_rde_bench, print_report


def main() -> None:
    parser = argparse.ArgumentParser(description="ResearchForge-ECRM RDE-Bench demo")
    parser.add_argument("--seeds", type=int, default=5, help="number of random seeds per condition")
    parser.add_argument("--generations", type=int, default=25, help="generations per run")
    parser.add_argument("--out", type=str, default="demo_results.json")
    args = parser.parse_args()

    tasks = [digits_task(seed=0), synthetic_ecg_task(seed=0)]
    seeds = list(range(args.seeds))

    print(f"Running RDE-Bench: {len(tasks)} tasks x 3 conditions x {len(seeds)} seeds "
          f"x {args.generations} generations...")
    report = run_rde_bench(tasks, seeds=seeds, n_generations=args.generations)
    print_report(report)

    serializable = {
        task_name: {
            cond_name: {
                "best_metric_mean": s.best_metric_mean,
                "best_metric_std": s.best_metric_std,
                "research_efficiency": s.research_efficiency,
                "search_efficiency": s.search_efficiency,
                "failure_repetition_rate": s.failure_repetition_rate,
                "negative_transfer_rate": s.negative_transfer_rate,
                "memory_utility": s.memory_utility,
                "memory_half_life_days": (
                    None if s.memory_half_life_days != s.memory_half_life_days  # NaN check
                    else s.memory_half_life_days),
                "curves": s.curves,
            }
            for cond_name, s in conds.items()
        }
        for task_name, conds in report.items()
    }
    out_data = {
        "metadata": {
            "execution_class": "END_TO_END_INTEGRATION_RUN",
            "evidence_status": os.environ.get("RF_EVIDENCE_STATUS", "EXECUTED"),
            "confirmatory": False,
            "provider": os.environ.get("RF_LLM_PROVIDER", "heuristic"),
        },
        "results": serializable
    }
    
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved detailed results (including per-seed best-so-far curves) to {args.out}")


if __name__ == "__main__":
    main()
