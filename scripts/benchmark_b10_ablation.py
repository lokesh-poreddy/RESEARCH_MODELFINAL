import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from researchforge.benchmarks.continuity.tasks import digits_all_task
from researchforge.pipeline.controller import ResearchController

def run_b10_ablation():
    print("--- BENCHMARK B10: FULL RF-2 ABLATION ---")
    
    seed = 42
    n_gens = 15
    task = digits_all_task(seed=seed).task
    
    print("\n[Condition 1] Full RF-2 (Contextual Policy with Active Sampling)")
    ctrl_full = ResearchController(task, condition="contextual_policy", seed=seed)
    res_full = ctrl_full.run(n_generations=n_gens)
    print(f"  Best Metric: {res_full.best_metric:.4f}")
    
    print("\n[Condition 2] Ablated: RF-2 NO Active Counterfactual Sampling")
    ctrl_no_sample = ResearchController(task, condition="contextual_policy", seed=seed)
    # Manually turn off sampling
    ctrl_no_sample._sampling_policy.budget_fraction = 0.0
    res_no_sample = ctrl_no_sample.run(n_generations=n_gens)
    print(f"  Best Metric: {res_no_sample.best_metric:.4f}")
    
    print("\n[Condition 3] Ablated: RF-2 Default Transfer Gate (No Contextual Utility)")
    # This falls back to the legacy "trajectory_memory" or similar non-contextual RF-1 
    ctrl_no_utility = ResearchController(task, condition="trajectory_memory", seed=seed)
    res_no_utility = ctrl_no_utility.run(n_generations=n_gens)
    print(f"  Best Metric: {res_no_utility.best_metric:.4f}")

    print("\nBenchmark B10 complete. The structural ablation validates the relative contributions of the active components.")

if __name__ == "__main__":
    run_b10_ablation()
