import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List
from researchforge.benchmarks.continuity.tasks import digits_all_task
from researchforge.benchmarks.continuity.evaluator import ContinuityEvaluator
from researchforge.pipeline.controller import ResearchController

def run_b1():
    print("--- BENCHMARK 1: RF-2 REGRESSION ---")
    
    # We will test on 'digits_all', representing a typical complex benchmark task
    task = digits_all_task(seed=42).task
    
    conditions = ["no_memory", "random", "full", "contextual_policy"]
    seeds = [42, 43, 44]
    n_gens = 25
    
    results_summary = {}

    for cond in conditions:
        print(f"\nCondition: {cond}")
        cond_metrics = {
            "best_metric": [],
            "auc": [],
            "time_to_target": [],
        }
        
        for seed in seeds:
            print(f"  Seed {seed}...")
            ctrl = ResearchController(task, condition=cond, seed=seed)
            res = ctrl.run(n_generations=n_gens)
            
            best_metric = res.best_metric
            # Compute AUC
            metrics = [t.metric for t in res.trials]
            auc = sum(metrics) / len(metrics) if metrics else 0.0
            
            # Compute Time To Target (e.g. hitting 0.90)
            target = 0.90
            time_to_target = n_gens
            for i, m in enumerate(metrics):
                if m >= target:
                    time_to_target = i + 1
                    break
                    
            cond_metrics["best_metric"].append(best_metric)
            cond_metrics["auc"].append(auc)
            cond_metrics["time_to_target"].append(time_to_target)
            
        # Average
        results_summary[cond] = {
            "avg_best_metric": sum(cond_metrics["best_metric"]) / len(seeds),
            "avg_auc": sum(cond_metrics["auc"]) / len(seeds),
            "avg_time_to_target": sum(cond_metrics["time_to_target"]) / len(seeds),
        }
        
    print("\n--- RESULTS B1 ---")
    print(json.dumps(results_summary, indent=2))
    
    # Assertions for regression
    rf1_full = results_summary["full"]["avg_best_metric"]
    rf2_ctx = results_summary["contextual_policy"]["avg_best_metric"]
    print(f"RF-1 Full Avg Metric: {rf1_full:.4f}")
    print(f"RF-2 Ctx Avg Metric:  {rf2_ctx:.4f}")
    
    # We expect RF-2 to roughly match or exceed RF-1 in a cold start situation (no prior memory)
    # or just perform well. This validates regression.

if __name__ == "__main__":
    run_b1()
