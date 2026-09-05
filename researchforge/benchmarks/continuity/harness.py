"""researchforge/benchmarks/continuity/harness.py — Research Continuity Benchmark Execution Harness.

Phase 11 Scientific Execution Engine:
1. Executes multi-task sequences under controlled regimes (SAME_FAMILY, CROSS_FAMILY, UNRELATED).
2. Distinguishes COLD_START from NO_MEMORY and CONTINUOUS_EXPERIENCE.
3. Enforces cryptographic ExperiencePartition sealing and anti-future leakage.
4. Executes multiple task orderings (forward, reverse, cross_first) across seeds to eliminate order confounding.
5. Captures raw per-seed observations and computes learning-curve progression over experience count.
6. Separates benchmark bookkeeping/orchestration overhead from research metrics.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ...pipeline.controller import ResearchController, RunResult
from ...policy.evaluators import StandardBenchmarkEvaluator, StandardScientificEvaluator
from .evaluator import ContinuityEvaluator
from .models import (
    ContinuityArtifact,
    ContinuityCondition,
    ContinuityObservation,
    ExperiencePartition,
    FutureInformationLeakageError,
    TaskRegime,
    TransferClassification,
    TransferEvent,
)
from .tasks import ContinuityTask, get_ordered_tasks, get_pilot_task_sequence


class ContinuityHarness:
    """Orchestrates multi-task sequential continuity experiments with strict experimental controls."""

    def __init__(
        self,
        tasks: Optional[List[ContinuityTask]] = None,
        conditions: Optional[List[ContinuityCondition]] = None,
        orderings: Optional[List[str]] = None,
        seeds: Optional[List[int]] = None,
        n_generations: int = 15,
        population_size: int = 6,
        delta_threshold: float = 0.01,
        dispersion_threshold: float = 0.15,
    ) -> None:
        self.tasks = tasks or get_pilot_task_sequence(seed=0)
        self.conditions = conditions or [
            ContinuityCondition.COLD_START,
            ContinuityCondition.NO_MEMORY,
            ContinuityCondition.CONTINUOUS_EXPERIENCE,
        ]
        self.orderings = orderings or ["forward", "reverse", "cross_first"]
        self.seeds = seeds or [0, 1]
        self.n_generations = n_generations
        self.population_size = population_size

        self.evaluator = ContinuityEvaluator(
            delta_threshold=delta_threshold,
            dispersion_threshold=dispersion_threshold,
        )
        self.scientific_evaluator = StandardScientificEvaluator()
        self.benchmark_evaluator = StandardBenchmarkEvaluator()

    def run(self, output_path: Optional[Path | str] = None) -> ContinuityArtifact:
        """Executes the complete multi-condition, multi-ordering, multi-seed continuity protocol."""
        t_start = time.time()
        overhead_timings: Dict[str, float] = {
            "task_init_ms": 0.0,
            "experience_loading_ms": 0.0,
            "memory_reset_ms": 0.0,
            "partition_sealing_ms": 0.0,
            "bookkeeping_ms": 0.0,
            "artifact_writing_ms": 0.0,
        }

        # ── Step 1: Pre-calculate Controlled Within-Task Baselines ────────────────
        # Invariant (Correction 2): Transfer classification requires comparison against
        # the same task without prior transferred experience (COLD_START and NO_MEMORY).
        cold_start_baselines: Dict[Tuple[str, int], float] = {}
        no_memory_baselines: Dict[Tuple[str, int], float] = {}

        t_init_0 = time.time()
        for seed in self.seeds:
            for task in self.tasks:
                # Cold-start baseline for this task & seed
                ctrl_cs = ResearchController(
                    task.task,
                    condition="cold_start",
                    seed=seed,
                    population_size=self.population_size,
                )
                res_cs = ctrl_cs.run(n_generations=self.n_generations)
                cold_start_baselines[(task.task_id, seed)] = res_cs.best_metric

                # No-memory baseline for this task & seed
                ctrl_nm = ResearchController(
                    task.task,
                    condition="no_memory",
                    seed=seed,
                    population_size=self.population_size,
                )
                res_nm = ctrl_nm.run(n_generations=self.n_generations)
                no_memory_baselines[(task.task_id, seed)] = res_nm.best_metric

        overhead_timings["task_init_ms"] += (time.time() - t_init_0) * 1000.0

        # ── Step 2: Main Sequential Continuity Execution ─────────────────────────
        observations: List[ContinuityObservation] = []
        partitions: List[Dict[str, Any]] = []
        all_transfer_events: List[Dict[str, Any]] = []

        for cond in self.conditions:
            cond_str = cond.value.lower()
            for ordering in self.orderings:
                ordered_tasks = get_ordered_tasks(self.tasks, ordering)

                for seed in self.seeds:
                    # Initialize sequential memory chain for this (cond, ordering, seed)
                    t_mem_0 = time.time()
                    inherited_ecrm = None
                    inherited_traj = None
                    inherited_adaptive = None
                    inherited_policy = None
                    inherited_fails: Set[Tuple[str, str]] = set()
                    prior_task_count = 0
                    prior_experience_count = 0
                    overhead_timings["memory_reset_ms"] += (time.time() - t_mem_0) * 1000.0

                    for task_idx, cont_task in enumerate(ordered_tasks):
                        # Determine prior allowed tasks and future tasks for partition sealing
                        allowed_ids = [t.task_id for t in ordered_tasks[:task_idx]]
                        future_ids = [t.task_id for t in ordered_tasks[task_idx + 1:]]

                        # Handle COLD_START vs CONTINUOUS vs NO_MEMORY
                        t_load_0 = time.time()
                        if cond == ContinuityCondition.COLD_START:
                            # Fresh state for every task; previous task experience does not exist
                            ctrl = ResearchController(
                                cont_task.task,
                                condition="cold_start",
                                seed=seed,
                                population_size=self.population_size,
                            )
                            prior_task_count = 0
                            prior_experience_count = 0
                        elif cond == ContinuityCondition.NO_MEMORY:
                            # Memory contribution explicitly disabled
                            ctrl = ResearchController(
                                cont_task.task,
                                condition="no_memory",
                                seed=seed,
                                population_size=self.population_size,
                            )
                            prior_task_count = 0
                            prior_experience_count = 0
                        else:
                            # Continuous conditions inherit prior accumulated memory
                            ctrl = ResearchController(
                                cont_task.task,
                                condition=cond_str,
                                seed=seed,
                                population_size=self.population_size,
                                ecrm=inherited_ecrm,
                                trajectory_memory=inherited_traj,
                                adaptive_trajectory_memory=inherited_adaptive,
                                policy_learner=inherited_policy,
                                failed_signatures=inherited_fails,
                            )

                        overhead_timings["experience_loading_ms"] += (time.time() - t_load_0) * 1000.0

                        # Record memory state BEFORE task
                        mem_before_fp = ctrl.export_memory_fingerprint()

                        # Create and seal ExperiencePartition
                        t_part_0 = time.time()
                        partition = ExperiencePartition.create(
                            task_id=cont_task.task_id,
                            task_sequence_index=task_idx,
                            allowed_task_ids=allowed_ids,
                            prior_memory_fingerprint=mem_before_fp,
                            current_task_fingerprint=cont_task.fingerprint(),
                            future_task_ids=future_ids,
                        )
                        partitions.append(partition.to_dict())
                        overhead_timings["partition_sealing_ms"] += (time.time() - t_part_0) * 1000.0

                        # Execute research task under controlled generation budget
                        run_result: RunResult = ctrl.run(n_generations=self.n_generations)

                        # Record memory state AFTER task
                        mem_after_fp = ctrl.export_memory_fingerprint()

                        obs_prior_task_count = prior_task_count
                        obs_prior_experience_count = prior_experience_count

                        # If continuous, update inherited experience for next task
                        if cond not in (ContinuityCondition.COLD_START, ContinuityCondition.NO_MEMORY):
                            inherited_ecrm = ctrl.ecrm
                            inherited_traj = ctrl.trajectory_memory
                            inherited_adaptive = ctrl.adaptive_trajectory_memory
                            inherited_policy = ctrl.policy
                            inherited_fails = set(ctrl._failed_signatures)
                            inherited_mem_fp = mem_after_fp
                            prior_task_count += 1
                            prior_experience_count += len(run_result.trials)
                        else:
                            inherited_mem_fp = "none"

                        # ── Step 3: Compute Continuity Metrics & Audit Transfer ──
                        t_book_0 = time.time()
                        cs_baseline = cold_start_baselines.get((cont_task.task_id, seed), run_result.best_metric)
                        nm_baseline = no_memory_baselines.get((cont_task.task_id, seed), run_result.best_metric)

                        exp_gain = run_result.best_metric - cs_baseline
                        mem_utility = run_result.best_metric - nm_baseline

                        dq = self.evaluator.calculate_decision_quality(run_result.trials, inherited_fails)
                        redundant_exps = self.evaluator.calculate_redundant_experiments(run_result.trials)
                        port_div = self.evaluator.calculate_portfolio_diversity(run_result.trials)
                        pol_stab = self.evaluator.calculate_policy_stability(run_result.trials)
                        se = self.evaluator.calculate_search_efficiency(run_result.trials, cont_task.task.target_metric)
                        re = run_result.best_metric / max(1, len(run_result.trials))

                        # Repeated failures & negative transfer
                        used_mem_trials = [t for t in run_result.trials if t.used_memory]
                        neg_transfers = [t for t in used_mem_trials if t.memory_negative_transfer]
                        ntr = len(neg_transfers) / max(1, len(used_mem_trials)) if used_mem_trials else 0.0

                        seen_fails_in_run: Set[Tuple[str, str]] = set()
                        repeat_fails = 0
                        for t in run_result.trials:
                            sig = (t.strategy, t.model_type)
                            if t.failure != "None":
                                if sig in seen_fails_in_run or sig in inherited_fails:
                                    repeat_fails += 1
                                seen_fails_in_run.add(sig)
                        frr = repeat_fails / max(1, len(run_result.trials))

                        # Audit transfer events if memory from prior tasks was available
                        task_transfer_events: List[Dict[str, Any]] = []
                        if task_idx > 0 and cond not in (ContinuityCondition.COLD_START, ContinuityCondition.NO_MEMORY):
                            source_task_name = ordered_tasks[task_idx - 1].name
                            # Classify each trial that used prior memory
                            for t in used_mem_trials:
                                event = self.evaluator.classify_transfer(
                                    source_task=source_task_name,
                                    target_task=cont_task.name,
                                    strategy=t.strategy,
                                    model_type=t.model_type,
                                    experienced_result=t.metric,
                                    baseline_without_transfer=cs_baseline,
                                    sample_count=t.sample_count or 1,
                                    context_level=t.context_level,
                                    provenance_id=f"prov_xfer_{cont_task.task_id}_{t.generation}",
                                    rationale=f"Transfer from {source_task_name} in condition {cond.value}",
                                )
                                d_event = event.to_dict()
                                task_transfer_events.append(d_event)
                                all_transfer_events.append(d_event)

                        pos_count = sum(1 for e in task_transfer_events if e["transfer_classification"] == "POSITIVE")
                        neg_count = sum(1 for e in task_transfer_events if e["transfer_classification"] == "NEGATIVE")
                        succ_rate = pos_count / max(1, len(task_transfer_events)) if task_transfer_events else 0.0
                        harm_rate = neg_count / max(1, len(task_transfer_events)) if task_transfer_events else 0.0

                        obs = ContinuityObservation(
                            task_id=cont_task.task_id,
                            task_name=cont_task.name,
                            regime=cont_task.regime.value,
                            order_id=ordering,
                            condition=cond.value,
                            seed=seed,
                            task_sequence_index=task_idx,
                            prior_task_count=obs_prior_task_count,
                            prior_experience_count=obs_prior_experience_count,
                            best_metric=run_result.best_metric,
                            baseline_cold_start_metric=cs_baseline,
                            experience_gain=exp_gain,
                            research_efficiency=re,
                            search_efficiency=se,
                            failure_repetition_rate=frr,
                            negative_transfer_rate=ntr,
                            memory_utility=mem_utility,
                            decision_quality=dq,
                            successful_transfer_rate=succ_rate,
                            harmful_transfer_rate=harm_rate,
                            information_reuse=len(used_mem_trials),
                            redundant_experiments=redundant_exps,
                            cost_to_threshold=se,
                            policy_stability=pol_stab,
                            portfolio_diversity=port_div,
                            memory_fingerprint_before=mem_before_fp,
                            memory_fingerprint_after=mem_after_fp,
                            memory_fingerprint_inherited=inherited_mem_fp,
                            transfer_events=task_transfer_events,
                        )
                        observations.append(obs)
                        overhead_timings["bookkeeping_ms"] += (time.time() - t_book_0) * 1000.0

        # ── Step 4: Aggregate Learning-Curve Telemetry ────────────────────────────
        # Group observations by prior_task_count to evaluate monotonic/diminishing trends
        curve_by_count: Dict[int, List[ContinuityObservation]] = {}
        for o in observations:
            curve_by_count.setdefault(o.prior_task_count, []).append(o)

        learning_curve_telemetry: List[Dict[str, Any]] = []
        for p_count in sorted(curve_by_count.keys()):
            subset = curve_by_count[p_count]
            learning_curve_telemetry.append({
                "prior_task_count": p_count,
                "observation_count": len(subset),
                "sample_count": len(subset),
                "mean_best_metric": round(statistics.mean(s.best_metric for s in subset), 4),
                "mean_experience_gain": round(statistics.mean(s.experience_gain for s in subset), 4),
                "mean_decision_quality": round(statistics.mean(s.decision_quality for s in subset), 4),
                "mean_research_efficiency": round(statistics.mean(s.research_efficiency for s in subset), 6),
                "mean_failure_repetition_rate": round(statistics.mean(s.failure_repetition_rate for s in subset), 4),
                "mean_negative_transfer_rate": round(statistics.mean(s.negative_transfer_rate for s in subset), 4),
                "mean_portfolio_diversity": round(statistics.mean(s.portfolio_diversity for s in subset), 4),
            })

        # ── Step 5: Construct Canonical Continuity Artifact ──────────────────────
        t_art_0 = time.time()
        artifact = ContinuityArtifact(
            benchmark_version="1.0.0",
            code_revision="HEAD",
            rf_version="1.0.0-alpha.3",
            task_definitions=[t.to_dict() for t in self.tasks],
            task_fingerprints={t.task_id: t.fingerprint() for t in self.tasks},
            task_orderings=list(self.orderings),
            conditions=[c.value for c in self.conditions],
            seeds=list(self.seeds),
            budgets={"n_generations": self.n_generations, "population_size": self.population_size},
            policy_version="baseline_v1",
            memory_configuration={"adaptive_backoff": True, "min_context_samples": 3},
            experience_partitions=partitions,
            raw_observations=[o.to_dict() for o in observations],
            transfer_classifications=all_transfer_events,
            learning_curve_telemetry=learning_curve_telemetry,
            orchestration_overhead_ms={k: round(v, 2) for k, v in overhead_timings.items()},
        )
        artifact.artifact_fingerprint = artifact.compute_fingerprint()
        overhead_timings["artifact_writing_ms"] += (time.time() - t_art_0) * 1000.0

        # Save to disk if output path is provided
        if output_path is not None:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(artifact.to_dict(), f, indent=2)

        return artifact
