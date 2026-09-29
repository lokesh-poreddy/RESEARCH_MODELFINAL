"""RDE-Bench harness: runs the ablation ladder (full system vs no-memory vs
random search) across tasks and seeds, and computes the metrics defined in
ResearchForge-ECRM Sec. 6 / Sec. 8 (RE, SE, FRR, MU, NTR, Memory Half-life).
"""
from __future__ import annotations
import statistics
from dataclasses import dataclass, field
from typing import Dict, List

from ..pipeline.controller import ResearchController, RunResult, CONDITIONS
from .tasks import Task


@dataclass
class BenchSummary:
    task_name: str
    condition: str
    best_metric_mean: float
    best_metric_std: float
    research_efficiency: float       # RE: mean best metric per experiment
    search_efficiency: int           # SE: mean generation of first hit on target_metric
    failure_repetition_rate: float   # FRR
    negative_transfer_rate: float    # NTR
    memory_utility: float = 0.0      # MU: filled in for a memory condition, relative to 'no_memory'
    memory_half_life_days: float = float("nan")
    curves: List[List[float]] = field(default_factory=list)  # best-so-far per seed, incl. baseline
    context_level_distribution: Dict[int, int] = field(default_factory=dict)
    backoff_frequency: float = 0.0
    failure_blocking_events: int = 0
    memory_influenced_decisions: int = 0


def run_condition(task: Task, condition: str, seeds: List[int], n_generations: int) -> BenchSummary:
    bests, curves, fr_rates, ntr_rates, se_list = [], [], [], [], []
    half_life = float("nan")
    backoff_counts = 0
    failure_blocks = 0
    memory_influences = 0
    level_counts: Dict[int, int] = {3: 0, 2: 0, 1: 0, 0: 0}

    for seed in seeds:
        ctrl = ResearchController(task, condition=condition, seed=seed)
        run: RunResult = ctrl.run(n_generations=n_generations)
        bests.append(run.best_metric)
        curves.append([t.best_so_far for t in run.trials])

        # Signature matches exactly what has_similar_failure keys on
        # (strategy + model_type -- see ResearchController._mem_key), so FRR
        # measures repetition of the *specific* failures the memory check is
        # designed to catch, not just "this strategy name was ever tried again."
        seen_bad = set()
        n_repeat = 0
        for t in run.trials:
            sig = (t.strategy, t.model_type)
            if t.failure != "None":
                if sig in seen_bad:
                    n_repeat += 1
                seen_bad.add(sig)
            if t.context_level is not None:
                level_counts[t.context_level] = level_counts.get(t.context_level, 0) + 1
            if t.fallback_reason is not None and "insufficient_samples" in t.fallback_reason:
                backoff_counts += 1
            if t.memory_decision_contribution:
                memory_influences += 1
                contrib = t.memory_decision_contribution
                if contrib.get("audit_per_action", {}).get(t.strategy, {}).get("has_failed_majority"):
                    failure_blocks += 1
        fr_rates.append(n_repeat / len(run.trials) if run.trials else 0.0)

        used_mem = [t for t in run.trials if t.used_memory]
        neg = [t for t in used_mem if t.memory_negative_transfer]
        ntr_rates.append(len(neg) / len(used_mem) if used_mem else 0.0)

        hit = next((t.generation for t in run.trials if t.metric >= task.target_metric),
                   n_generations)
        se_list.append(hit)

        if condition == "full":
            half_life = run.memory_half_life_days

    mean_best = statistics.mean(bests)
    std_best = statistics.pstdev(bests) if len(bests) > 1 else 0.0
    re = mean_best / max(1, n_generations)
    total_context_queries = sum(level_counts.values())
    backoff_freq = backoff_counts / max(1, total_context_queries) if total_context_queries > 0 else 0.0

    return BenchSummary(
        task_name=task.name, condition=condition,
        best_metric_mean=mean_best, best_metric_std=std_best,
        research_efficiency=re, search_efficiency=int(round(statistics.mean(se_list))),
        failure_repetition_rate=statistics.mean(fr_rates),
        negative_transfer_rate=statistics.mean(ntr_rates),
        memory_half_life_days=half_life, curves=curves,
        context_level_distribution=level_counts,
        backoff_frequency=backoff_freq,
        failure_blocking_events=failure_blocks,
        memory_influenced_decisions=memory_influences,
    )


def run_rde_bench(tasks: List[Task], seeds: List[int] = (0, 1, 2),
                   n_generations: int = 25) -> Dict[str, Dict[str, BenchSummary]]:
    report: Dict[str, Dict[str, BenchSummary]] = {}
    for task in tasks:
        report[task.name] = {}
        for cond in CONDITIONS:
            report[task.name][cond] = run_condition(task, cond, list(seeds), n_generations)
        nomem = report[task.name]["no_memory"]
        for memory_cond in ("full", "trajectory_memory", "adaptive_trajectory", "contextual_policy"):
            if memory_cond in report[task.name]:
                report[task.name][memory_cond].memory_utility = (
                    report[task.name][memory_cond].best_metric_mean - nomem.best_metric_mean)
    return report


def print_report(report: Dict[str, Dict[str, BenchSummary]]) -> None:
    for task_name, conds in report.items():
        print(f"\n=== RDE-Bench: {task_name} ===")
        print(f"{'condition':<20}{'best':>8}{'RE':>8}{'SE':>6}{'FRR':>7}{'NTR':>7}{'MU':>9}")
        for cond_name in ("contextual_policy", "full", "trajectory_memory", "adaptive_trajectory", "no_memory", "random"):
            if cond_name not in conds:
                continue
            s = conds[cond_name]
            mu = f"{s.memory_utility:+.4f}" if cond_name in ("full", "trajectory_memory", "adaptive_trajectory", "contextual_policy") else "--"
            print(f"{cond_name:<20}{s.best_metric_mean:>8.4f}{s.research_efficiency:>8.4f}"
                  f"{s.search_efficiency:>6d}{s.failure_repetition_rate:>7.2f}"
                  f"{s.negative_transfer_rate:>7.2f}{mu:>9}")
        print("(best = mean best validation metric across seeds; RE = best/#experiments; "
              "SE = mean generation of first hit on target_metric; FRR/NTR are rates in [0,1]; "
              "MU is relative to no_memory)")
        print(f"analytic memory half-life (decay parameter): "
              f"{conds['full'].memory_half_life_days:.1f} days")

