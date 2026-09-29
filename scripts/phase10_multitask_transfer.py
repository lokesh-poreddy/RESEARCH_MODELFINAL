"""Phase 10: Multi-Task Transfer Benchmark

Tests whether RF-2 policy can learn when prior research experience transfers 
beneficially, neutrally, or negatively across different research contexts, 
compared to the RF-1 (full) baseline.

The experiment runs two sequential tasks (Digits -> ECG) using shared memory 
and policy components, and compares performance on Task 2 to a cold-start 
baseline to measure transfer (positive or negative).
"""
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchforge.benchmarks.tasks import digits_task, synthetic_ecg_task
from researchforge.pipeline.controller import ResearchController, STRATEGIES
from researchforge.memory.ecrm import ECRM
from researchforge.policy.policy_learner import PolicyLearner
from researchforge.memory.posterior import PosteriorMemory
from researchforge.policy.contextual_bandit import ContextualBanditPolicy
from researchforge.policy.transfer_gate import TransferGate

def run_multitask_transfer(n_seeds=10, n_generations=15):
    print(f"Running Multi-Task Transfer Benchmark (N={n_seeds} seeds, {n_generations} gen/task)...")
    
    results = {
        "cold_start": {"ecg_bests": []},
        "rf1_transfer": {"ecg_bests": []},
        "rf2_transfer": {"ecg_bests": []}
    }
    
    for seed in range(n_seeds):
        # 1. Cold Start (ECG only, no prior memory)
        task_ecg = synthetic_ecg_task(seed=seed)
        ctrl_cold = ResearchController(task_ecg, condition="no_memory", seed=seed)
        run_cold = ctrl_cold.run(n_generations=n_generations)
        results["cold_start"]["ecg_bests"].append(run_cold.best_metric)
        
        # 2. RF-1 Transfer (Digits -> ECG)
        task_digits = digits_task(seed=seed)
        rng = random.Random(seed)
        
        ecrm = ECRM()
        policy_rf1 = PolicyLearner(STRATEGIES, rng=rng)
        
        ctrl_rf1_digits = ResearchController(
            task_digits, condition="full", seed=seed,
            ecrm=ecrm, policy_learner=policy_rf1
        )
        ctrl_rf1_digits.run(n_generations=n_generations)
        
        ctrl_rf1_ecg = ResearchController(
            task_ecg, condition="full", seed=seed,
            ecrm=ecrm, policy_learner=policy_rf1
        )
        run_rf1_ecg = ctrl_rf1_ecg.run(n_generations=n_generations)
        results["rf1_transfer"]["ecg_bests"].append(run_rf1_ecg.best_metric)
        
        # 3. RF-2 Transfer (Digits -> ECG)
        posterior = PosteriorMemory()
        bandit = ContextualBanditPolicy(actions=STRATEGIES, alpha=1.0, rng=rng)
        gate = TransferGate(g_min=0.15)
        
        ctrl_rf2_digits = ResearchController(
            task_digits, condition="contextual_policy", seed=seed,
            posterior_memory=posterior, contextual_bandit=bandit, transfer_gate=gate
        )
        ctrl_rf2_digits.run(n_generations=n_generations)
        
        ctrl_rf2_ecg = ResearchController(
            task_ecg, condition="contextual_policy", seed=seed,
            posterior_memory=posterior, contextual_bandit=bandit, transfer_gate=gate
        )
        run_rf2_ecg = ctrl_rf2_ecg.run(n_generations=n_generations)
        results["rf2_transfer"]["ecg_bests"].append(run_rf2_ecg.best_metric)
        
        print(f"Seed {seed}: Cold={run_cold.best_metric:.4f}, RF1={run_rf1_ecg.best_metric:.4f}, RF2={run_rf2_ecg.best_metric:.4f}")

    # Compute stats
    cold = np.array(results["cold_start"]["ecg_bests"])
    rf1 = np.array(results["rf1_transfer"]["ecg_bests"])
    rf2 = np.array(results["rf2_transfer"]["ecg_bests"])
    
    print("\n=== Multi-Task Transfer Results (ECG Task after Digits Task) ===")
    print(f"Cold Start (No Transfer) : {np.mean(cold):.4f} ± {np.std(cold, ddof=1):.4f}")
    print(f"RF-1 Transfer            : {np.mean(rf1):.4f} ± {np.std(rf1, ddof=1):.4f} (Diff: {np.mean(rf1 - cold):+.4f})")
    print(f"RF-2 Contextual Transfer : {np.mean(rf2):.4f} ± {np.std(rf2, ddof=1):.4f} (Diff: {np.mean(rf2 - cold):+.4f})")
    
    # Save results
    out_path = PROJECT_ROOT / "phase10_multitask_benchmark_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "cold_start": {"mean": float(np.mean(cold)), "std": float(np.std(cold, ddof=1)), "raw": cold.tolist()},
            "rf1_transfer": {"mean": float(np.mean(rf1)), "std": float(np.std(rf1, ddof=1)), "raw": rf1.tolist()},
            "rf2_transfer": {"mean": float(np.mean(rf2)), "std": float(np.std(rf2, ddof=1)), "raw": rf2.tolist()},
        }, f, indent=2)

if __name__ == "__main__":
    run_multitask_transfer(n_seeds=15, n_generations=20)
