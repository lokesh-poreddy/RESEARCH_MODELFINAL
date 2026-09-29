"""Phase 9 Statistical Validation: Contextual Policy vs RF-1 Full.

Computes paired t-tests, 95% Confidence Intervals, and Cohen's d effect sizes
to validate the statistical significance of the RF-2 upgrade.
"""
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchforge.benchmarks.tasks import digits_task
from researchforge.benchmarks.rde_bench import run_condition

def run_statistical_validation(n_seeds=30, n_generations=20):
    task = digits_task(seed=0)
    print(f"Running Phase 9 Statistical Validation (N={n_seeds} seeds, {n_generations} gen)...")
    
    seeds = list(range(n_seeds))
    
    # Run contextual_policy
    t0 = time.time()
    print("Evaluating 'contextual_policy'...")
    res_ctx = run_condition(task, "contextual_policy", seeds, n_generations)
    
    # Run full (baseline)
    print("Evaluating 'full' (RF-1 Baseline)...")
    res_full = run_condition(task, "full", seeds, n_generations)
    
    elapsed = time.time() - t0
    print(f"Data collection completed in {elapsed:.1f}s.")
    
    # Extract paired best metrics (we need the underlying data)
    # Since run_condition only returns the mean in the summary, we actually need the raw 
    # 'bests' array which isn't currently exposed in BenchSummary directly.
    # But wait, run_condition computes mean_best from a list of bests.
    # Since we need paired samples, let's run them seed by seed here to get paired vectors.
    pass

def run_paired_analysis(n_seeds=30, n_generations=20):
    from researchforge.benchmarks.tasks import digits_task
    from researchforge.pipeline.controller import ResearchController
    
    task = digits_task(seed=0)
    print(f"Running Paired Statistical Validation (N={n_seeds} seeds)...")
    
    ctx_bests = []
    full_bests = []
    
    for seed in range(n_seeds):
        # Contextual Policy
        ctrl_ctx = ResearchController(task, condition="contextual_policy", seed=seed)
        run_ctx = ctrl_ctx.run(n_generations=n_generations)
        ctx_bests.append(run_ctx.best_metric)
        
        # Full (RF-1)
        ctrl_full = ResearchController(task, condition="full", seed=seed)
        run_full = ctrl_full.run(n_generations=n_generations)
        full_bests.append(run_full.best_metric)
        
    ctx_arr = np.array(ctx_bests)
    full_arr = np.array(full_bests)
    
    diffs = ctx_arr - full_arr
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    
    # Paired t-test
    t_stat, p_val = stats.ttest_rel(ctx_arr, full_arr)
    
    # 95% CI for the mean difference
    n = len(diffs)
    se = std_diff / math.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n-1)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se
    
    # Cohen's d
    pooled_std = math.sqrt((np.var(ctx_arr, ddof=1) + np.var(full_arr, ddof=1)) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
    
    print("\n=== Statistical Validation Report ===")
    print(f"Metric: Best Validation Score (Target = {task.target_metric})")
    print(f"N = {n_seeds} independent seeds\n")
    
    print(f"RF-2 Contextual Policy : {np.mean(ctx_arr):.4f} ± {np.std(ctx_arr, ddof=1):.4f}")
    print(f"RF-1 Full (Baseline)   : {np.mean(full_arr):.4f} ± {np.std(full_arr, ddof=1):.4f}")
    print("-" * 40)
    print(f"Mean Difference        : {mean_diff:+.4f}")
    print(f"95% Confidence Interval: [{ci_lower:+.4f}, {ci_upper:+.4f}]")
    print(f"Paired t-statistic     : {t_stat:.4f}")
    print(f"p-value (2-tailed)     : {p_val:.4g}")
    print(f"Cohen's d (Effect Size): {cohens_d:.4f}")
    
    # Assess significance
    alpha = 0.05
    if p_val < alpha:
        print("\nConclusion: STATISTICALLY SIGNIFICANT difference.")
    else:
        print("\nConclusion: NOT statistically significant at alpha=0.05.")
        
    # Write report
    report = {
        "n_seeds": n_seeds,
        "n_generations": n_generations,
        "ctx_mean": float(np.mean(ctx_arr)),
        "full_mean": float(np.mean(full_arr)),
        "mean_diff": float(mean_diff),
        "ci_95": [float(ci_lower), float(ci_upper)],
        "p_value": float(p_val),
        "cohens_d": float(cohens_d)
    }
    
    out_path = PROJECT_ROOT / "phase9_statistical_validation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
if __name__ == "__main__":
    run_paired_analysis(n_seeds=30, n_generations=20)
