"""ResearchController: ties the Research Development Graph, Model Genomes,
ECRM, Policy Learner, Failure Diagnosis, and Discovery Pipeline into the
research loop described in ResearchForge-ECRM Sec. 1 / Sec. 9's architecture
diagrams:

    select branch -> synthesize genome -> run experiment -> diagnose
    -> update memory -> update policy -> repeat

RF-1.0.0-alpha.2.1 additions
------------------------------
The controller now uses TargetModelGenome (TMG) as the evolutionary object.
The population is List[TargetModelGenome]. Evaluators receive .to_model_genome()
for backward compatibility with sklearn_evaluator and capacity_bucket.

RSG wiring (alpha.2.1 — low-risk parameters only):
  - memory_config.decay_lambda, retention_threshold → ECRM initialization
  - execution_config.per_experiment_timeout_s → sandbox timeout
  - execution_config.execution_mode → use_sandbox determination

NOT wired in alpha.2.1 (requires ExperimentSpec fingerprinting first):
  - validity_config.n_permutations, significance_alpha (alpha.3)

Backward compatibility invariant (AD-013):
    rsg=None → EXACTLY the same execution as RF-1.0-alpha.1.
    No code path is altered; the rsg is stored only for provenance.
    rsg=RSG.default(condition) must produce a bitwise-identical trajectory
    (same seed → same trial sequence, same metrics, same trajectory hash).
    This is tested in test_genomes.py::test_rsg_none_behavioral_equivalence
    AND in the regression benchmark after the alpha.2.1 TMG migration.

Four `condition`s implement the RDE-Bench ablation ladder:

  - "full":              policy learner (bandit) + flat ECRM (memory.ecrm) +
                          a strategy/model-family-conditioned failure check
                          -- the original complete system (Sec. 6/8).
  - "trajectory_memory":  policy learner + memory.trajectory.TrajectoryMemory
                          instead of the flat ECRM.
  - "no_memory":         policy learner still adapts within the run but no
                          cross-experiment memory.
  - "random":            uniform-random strategy choice, no learning.
"""
from __future__ import annotations
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..rdg.graph import ResearchDevelopmentGraph
from ..genome.model_genome import ModelGenome
from ..genome.target_model_genome import TargetModelGenome  # alpha.2.1: TMG is now the evolutionary object
from ..genome.operators import STRATEGIES
from ..memory.ecrm import ECRM
from ..memory.trajectory import (
    TrajectoryMemory, TrajectoryRecord, capacity_bucket, generation_stage, new_trajectory_id,
)
from ..memory.adaptive_trajectory import (
    AdaptiveTrajectoryMemory,
    AdaptiveTrajectoryRecord,
    ContextualRetrievalResult,
    FailureCheckResult,
    new_adaptive_trajectory_id,
)
from ..policy.policy_learner import PolicyLearner
from ..diagnosis.failure_taxonomy import diagnose, ExperimentResult, FailureCategory
from ..evaluators.sklearn_evaluator import evaluate_genome
from .discovery import HeuristicSynthesizer, unit_test
from ..benchmarks.tasks import Task
from ..state.research_state import ResearchState

CONDITIONS = (
    "full", "trajectory_memory", "adaptive_trajectory", "no_memory", "random",
    "cold_start", "continuous_experience", "flat_ecrm",
)


@dataclass
class TrialRecord:
    generation: int
    strategy: str
    model_type: str
    metric: float
    best_so_far: float
    failure: str
    used_memory: bool
    memory_negative_transfer: bool
    genome_id: str
    context_level: Optional[int] = None
    requested_context_level: Optional[int] = None
    sample_count: Optional[int] = None
    fallback_reason: Optional[str] = None
    memory_decision_contribution: Optional[Dict[str, Any]] = None


@dataclass
class RunResult:
    task_name: str
    condition: str
    trials: List[TrialRecord] = field(default_factory=list)
    best_genome: Optional[TargetModelGenome] = None   # alpha.2.1: TMG (was ModelGenome)
    best_metric: float = 0.0
    rdg_stats: dict = field(default_factory=dict)
    memory_half_life_days: float = float("nan")
    trajectory_stats: Dict[str, int] = field(default_factory=dict)
    adaptive_trajectory_stats: Dict[str, Any] = field(default_factory=dict)
    wall_time_s: float = 0.0
    rsg_id: Optional[str] = None  # RF-1.0.0-alpha.2: set if RSG was provided
    states: List[ResearchState] = field(default_factory=list)  # alpha.2.1: ResearchState per generation


class ResearchController:
    def __init__(self, task: Task, condition: str = "full", seed: int = 0,
                 population_size: int = 6, initial_model_type: str = "LogisticRegression",
                 use_sandbox: bool = False, sandbox_timeout_s: float = 15.0,
                 rsg: Optional[Any] = None,
                 ecrm: Optional[ECRM] = None,
                 trajectory_memory: Optional[TrajectoryMemory] = None,
                 adaptive_trajectory_memory: Optional[AdaptiveTrajectoryMemory] = None,
                 policy_learner: Optional[PolicyLearner] = None,
                 failed_signatures: Optional[set] = None,
                 enable_dynamic_router: bool = False,
                 transfer_guard_mode: Optional[str] = None,
                 router: Optional[Any] = None,
                 transfer_guard: Optional[Any] = None):
        """Initialise the ResearchController.

        Parameters
        ----------
        rsg : ResearchSystemGenome | None
            Optional Research System Genome. When None (default), the controller
            uses exactly the same execution path as RF-1.0-alpha.1 — this is
            the backward compatibility guarantee (AD-013).
            When an RSG is provided, it is stored as self.rsg for provenance
            tracking (RunResult.rsg_id) but does NOT change any execution
            behaviour in alpha.2. Full RSG-driven execution wiring is
            scheduled for RF-1.0.0-alpha.3 (VRDEG integration).
        """
        if condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}")
        self.task = task
        self.condition = condition
        self.seed = seed
        self.rng = random.Random(seed)
        self.rdg = ResearchDevelopmentGraph()

        # RSG provenance + wiring (RF-1.0.0-alpha.2.1)
        self.rsg = rsg

        # ── RSG memory_config wiring (alpha.2.1, low-risk) ────────────────
        # When rsg is provided, its memory_config overrides the hardcoded
        # RF-0.x defaults. When rsg=None, exact same defaults as before.
        # NOT wired: validity_config params (require ExperimentSpec fingerprinting, alpha.3).
        if rsg is not None:
            _decay_lambda = rsg.memory_config.decay_lambda
            _retention_threshold = rsg.memory_config.retention_threshold
            _min_context_samples = getattr(rsg.memory_config, "min_context_samples", 3)
        else:
            _decay_lambda = 0.08        # RF-0.x hardcoded default
            _retention_threshold = 0.12 # RF-0.x hardcoded default
            _min_context_samples = 3

        self.ecrm = ecrm if ecrm is not None else ECRM(decay_lambda=_decay_lambda, retention_threshold=_retention_threshold)
        self.trajectory_memory = trajectory_memory if trajectory_memory is not None else TrajectoryMemory()
        self.adaptive_trajectory_memory = adaptive_trajectory_memory if adaptive_trajectory_memory is not None else AdaptiveTrajectoryMemory(min_context_samples=_min_context_samples)
        self.policy = policy_learner if policy_learner is not None else PolicyLearner(STRATEGIES, rng=self.rng)
        self.synth = HeuristicSynthesizer()
        self.population_size = population_size
        self.memory_enabled = condition in ("full", "trajectory_memory", "adaptive_trajectory", "continuous_experience", "cold_start", "flat_ecrm")
        self.use_flat_memory = condition in ("full", "flat_ecrm")
        self.use_trajectory_memory = condition == "trajectory_memory"
        self.use_adaptive_trajectory = condition in ("adaptive_trajectory", "continuous_experience", "cold_start")
        self.use_policy = condition in ("full", "trajectory_memory", "adaptive_trajectory", "no_memory", "continuous_experience", "cold_start", "flat_ecrm")
        self._failed_signatures = set(failed_signatures) if failed_signatures is not None else set()
        self._last_decision_metadata: Dict[str, Any] = {}

        # ── Phase 13 Opt-in Wiring (RF-1.0.0-beta.1) ─────────────────────
        self.enable_dynamic_router = enable_dynamic_router
        self.transfer_guard_mode = transfer_guard_mode
        self.router = router
        if self.enable_dynamic_router and self.router is None:
            from ..router.router import DynamicModeRouter
            self.router = DynamicModeRouter()

        self.transfer_guard = transfer_guard
        if self.transfer_guard_mode is not None and self.transfer_guard is None:
            from ..transfer.guard import TransferGuard
            from ..transfer.types import TransferGuardMode
            self.transfer_guard = TransferGuard(mode=TransferGuardMode(self.transfer_guard_mode.upper()))

        # ── RSG execution_config wiring (alpha.2.1) ───────────────────────
        # execution_config controls sandbox/resource policy, NOT scientific validity.
        # RSG.execution_config.execution_mode="sandboxed" overrides use_sandbox arg.
        # Mandatory safety (schema validate, genome safety_check, provenance) always runs.
        if rsg is not None and rsg.execution_config.execution_mode == "sandboxed":
            _use_sandbox = True
            _sandbox_timeout = rsg.execution_config.per_experiment_timeout_s
        else:
            _use_sandbox = use_sandbox
            _sandbox_timeout = sandbox_timeout_s

        self.use_sandbox = _use_sandbox
        self._sandbox = None
        if _use_sandbox:
            from ..safety.sandbox import SafeRunner, ResourceBudget
            self._sandbox = SafeRunner(ResourceBudget(per_experiment_timeout_s=_sandbox_timeout))

        self.problem = self.rdg.add_node(
            "Problem", f"Improve {task.metric_fn.__name__} on {task.name}")
        self.gap = self.rdg.add_node(
            "Gap", f"No model yet reaches target {task.target_metric} on {task.name}")
        self.rdg.add_edge(self.problem.id, self.gap.id, "identifies")

        # ── alpha.2.1: Population uses TargetModelGenome ──────────────────
        # Evaluators (sklearn_evaluator, capacity_bucket) still expect ModelGenome;
        # we bridge via .to_model_genome() in _run_experiment and _select_strategy.
        # This preserves bitwise-identical trajectories (regression-verified).
        base_mg = ModelGenome.default(initial_model_type, seed=seed)
        base = TargetModelGenome.from_model_genome(base_mg)
        base._score = -1.0
        self.population: List[TargetModelGenome] = [base]

    @property
    def use_memory(self) -> bool:
        return self.memory_enabled

    @property
    def memory_mode(self) -> str:
        if not self.memory_enabled:
            return "none"
        if self.use_adaptive_trajectory:
            return "adaptive"
        if self.use_trajectory_memory:
            return "trajectory"
        if self.use_flat_memory:
            return "ecrm"
        return "adaptive"

    @property
    def policy_learner(self) -> PolicyLearner:
        return self.policy

    # ------------------------------------------------------------------
    def _run_experiment(self, genome: TargetModelGenome) -> ExperimentResult:
        """Run evaluate_genome on this TMG genome.

        alpha.2.1 bridge: evaluator still expects ModelGenome; we convert via
        .to_model_genome(). This preserves bitwise-identical trajectories.

        Mandatory safety (always runs regardless of execution_mode):
          - genome.safety_check() — genome-level sanity
          - schema validation happens at TMG construction
          - result is an ExperimentResult (validated return type)
        """
        # Mandatory safety check (not skipped by trusted_offline)
        violations = genome.safety_check()
        if violations:
            return ExperimentResult(metric=0.0, success=False,
                                     exception=f"genome failed safety_check: {violations}",
                                     target=self.task.target_metric)
        # Bridge to evaluator (ModelGenome API)
        mg = genome.to_model_genome()
        if self._sandbox is not None:
            from ..safety.sandbox import SafetyStatus
            outcome = self._sandbox.run(
                evaluate_genome, mg, self.task.X_train, self.task.y_train,
                self.task.X_val, self.task.y_val, self.task.metric_fn,
                target=self.task.target_metric)
            if outcome.status == SafetyStatus.OK:
                return outcome.value
            return ExperimentResult(metric=0.0, success=False,
                                     exception=f"{outcome.status.value}: {outcome.error}",
                                     target=self.task.target_metric)
        return evaluate_genome(mg, self.task.X_train, self.task.y_train,
                                self.task.X_val, self.task.y_val,
                                self.task.metric_fn, target=self.task.target_metric)

    def _mem_key(self, strategy: str, model_type: str) -> str:
        """Compact (strategy, model_type, task) descriptor used as the ECRM
        text key. Deliberately terse rather than a full sentence: with the
        offline hashed bag-of-words embedding (memory/embeddings.py), a
        template sentence like "Improve on genome derived from X for Y via Z"
        shares 6+ boilerplate words across *every* record, which swamps the
        2-3 words that actually distinguish one context from another (cosine
        similarity stays >=0.7 even for unrelated strategy/model pairs). A
        bare `"{strategy} {model_type} {task}"` key gives clean, interpretable
        similarity: 1.0 for an exact repeat, ~0.67 for a one-token difference,
        0.0 for no overlap -- so `has_similar_failure`'s threshold actually
        means something. A production build using a real semantic encoder
        would not need this workaround; free-text hypotheses would already
        separate cleanly in embedding space."""
        return f"{strategy} {model_type} {self.task.name}"

    def _select_strategy(self, parent: TargetModelGenome) -> Tuple[str, bool]:
        if self.condition == "random":
            return self.policy.select_random(), False
        if self.condition == "no_memory":
            return self.policy.select_action(), False
        if self.condition == "trajectory_memory":
            parent_mg = parent.to_model_genome()  # bridge for capacity_bucket
            parent_bucket = capacity_bucket(parent_mg)
            multiplier = lambda a: 0.3 + 0.7 * self.trajectory_memory.contextual_success_rate(
                a, parent.model_type, parent_bucket)
            return self.policy.select_action(score_multiplier=multiplier), True
        if self.condition in ("adaptive_trajectory", "continuous_experience", "cold_start"):
            parent_mg = parent.to_model_genome()  # bridge for capacity_bucket
            parent_bucket = capacity_bucket(parent_mg)
            audit_info: Dict[str, Any] = {}
            best_action = None
            best_final_score = -float("inf")
            selected_retrieval: Optional[ContextualRetrievalResult] = None

            for a in self.policy.actions:
                reward = self.policy.q[a]
                bonus = self.policy.c * math.sqrt(
                    math.log(self.policy.total_trials + 1) / (1 + self.policy.times_tried[a])
                )
                baseline_score = reward + bonus

                retrieval = self.adaptive_trajectory_memory.contextual_success_rate(
                    a, parent.model_type, parent_bucket
                )
                effective_rate = (
                    retrieval.success_rate * retrieval.confidence
                    + self.adaptive_trajectory_memory.default_prior * (1.0 - retrieval.confidence)
                )
                multiplier = 0.3 + 0.7 * effective_rate

                fail_check = self.adaptive_trajectory_memory.similar_trajectory_recently_failed(
                    a, parent.model_type, parent_bucket, window=3
                )
                final_score = baseline_score * multiplier
                if fail_check.has_failed_majority:
                    final_score *= 0.5

                audit_info[a] = {
                    "baseline_score": round(baseline_score, 4),
                    "multiplier": round(multiplier, 4),
                    "context_level": retrieval.context_level,
                    "sample_count": retrieval.sample_count,
                    "evidence_sufficiency": retrieval.evidence_sufficiency,
                    "fallback_reason": retrieval.fallback_reason,
                    "confidence": retrieval.confidence,
                    "success_rate": round(retrieval.success_rate, 4),
                    "has_failed_majority": fail_check.has_failed_majority,
                    "final_score": round(final_score, 4),
                }

                if final_score > best_final_score:
                    best_final_score = final_score
                    best_action = a
                    selected_retrieval = retrieval

            best_mem_action = max(audit_info.keys(), key=lambda k: audit_info[k]["multiplier"])
            decision_contrib = {
                "memory_recommendation": best_mem_action,
                "baseline_strategy_score": audit_info[best_action]["baseline_score"],
                "final_strategy_score": audit_info[best_action]["final_score"],
                "audit_per_action": audit_info,
            }
            self._last_decision_metadata = {
                "context_level": selected_retrieval.context_level if selected_retrieval else 0,
                "requested_context_level": 3,
                "sample_count": selected_retrieval.sample_count if selected_retrieval else 0,
                "fallback_reason": selected_retrieval.fallback_reason if selected_retrieval else None,
                "memory_decision_contribution": decision_contrib,
            }
            return best_action, True
        # condition == "full"
        failure_check = lambda a: self.ecrm.has_similar_failure(
            self._mem_key(a, parent.model_type), threshold=0.9)
        return self.policy.select_action(failure_check=failure_check), True

    def _record_trial(self, result: RunResult, generation: int, strategy: str,
                       model_type: str, exp_result: ExperimentResult, best_metric: float,
                       failure: FailureCategory, used_memory: bool,
                       neg_transfer: bool, genome_id: str,
                       context_level: Optional[int] = None,
                       requested_context_level: Optional[int] = None,
                       sample_count: Optional[int] = None,
                       fallback_reason: Optional[str] = None,
                       memory_decision_contribution: Optional[Dict[str, Any]] = None) -> None:
        result.trials.append(TrialRecord(
            generation=generation, strategy=strategy, model_type=model_type,
            metric=exp_result.metric, best_so_far=best_metric, failure=failure.value,
            used_memory=used_memory, memory_negative_transfer=neg_transfer, genome_id=genome_id,
            context_level=context_level, requested_context_level=requested_context_level,
            sample_count=sample_count, fallback_reason=fallback_reason,
            memory_decision_contribution=memory_decision_contribution))


    # ------------------------------------------------------------------
    def run(self, n_generations: int = 25) -> RunResult:
        t0 = time.time()
        result = RunResult(task_name=self.task.name, condition=self.condition)

        # -- baseline: evaluate the seed genome before any search (generation -1)
        base = self.population[0]
        base_mg = base.to_model_genome()  # bridge for evaluator
        base_result = self._run_experiment(base)
        base_failure = diagnose(base_result)
        base._score = base_result.metric if base_result.success else -1.0
        best_metric = base._score
        best_genome = base if base_result.success else None

        hyp0 = self.rdg.add_node(
            "Hypothesis", f"Baseline {base.model_type} for {self.task.name}",
            attributes={"strategy": "baseline", "generation": -1})
        self.rdg.add_edge(self.gap.id, hyp0.id, "motivates")
        exp0 = self.rdg.add_node(
            "Experiment", f"Train/evaluate baseline {base.model_type}",
            attributes={"genome_id": base.tmg_id})
        self.rdg.add_edge(hyp0.id, exp0.id, "tested-by")
        finding0 = self.rdg.add_node(
            "Finding", f"metric={base_result.metric:.4f} failure={base_failure.value}",
            attributes={"metric": base_result.metric, "failure": base_failure.value})
        self.rdg.add_edge(exp0.id, finding0.id, "produces")
        if self.use_flat_memory:
            self.ecrm.store(text_summary=self._mem_key("baseline", base.model_type),
                             context={"task": self.task.name, "genome": base_mg.to_dict()},
                             outcome={"metric": base_result.metric, "success": base_result.success,
                                      "failure": base_failure.value},
                             strategy="baseline")
        elif self.use_trajectory_memory:
            base_bucket = capacity_bucket(base_mg)
            self.trajectory_memory.store(TrajectoryRecord(
                id=new_trajectory_id(), generation=-1, stage="baseline",
                problem_context=self.gap.content,
                parent_model_type=base.model_type, parent_capacity_bucket=base_bucket,
                strategy="baseline",
                child_model_type=base.model_type, child_capacity_bucket=base_bucket,
                metric=base_result.metric, success=base_result.success,
                failure=base_failure.value,
                hypothesis_id=hyp0.id, experiment_id=exp0.id, finding_id=finding0.id))
        elif self.use_adaptive_trajectory:
            base_bucket = capacity_bucket(base_mg)
            self.adaptive_trajectory_memory.store(AdaptiveTrajectoryRecord(
                id=new_adaptive_trajectory_id(), generation=-1, stage="baseline",
                problem_context=self.gap.content,
                parent_model_type=base.model_type, parent_capacity_bucket=base_bucket,
                strategy="baseline",
                child_model_type=base.model_type, child_capacity_bucket=base_bucket,
                metric=base_result.metric, success=base_result.success,
                failure=base_failure.value,
                hypothesis_id=hyp0.id, spec_id=exp0.id, run_id=f"run_baseline_{base.tmg_id}",
                finding_id=finding0.id, provenance_id=f"prov_baseline_{base.tmg_id}"))
        self._record_trial(result, -1, "baseline", base.model_type, base_result,
                            best_metric, base_failure, False, False, base.tmg_id)
        result.states.append(ResearchState.create(
            generation=-1,
            research_phase="exploration" if self.rsg is None else self.rsg.research_phase,
            active_rsg_id="" if self.rsg is None else self.rsg.rsg_id,
            active_rsg_fingerprint="" if self.rsg is None else self.rsg.fingerprint(),
            candidate_tmg_ids=[base.tmg_id],
            best_tmg_id=base.tmg_id if base_result.success else None,
            best_metric=best_metric,
            budget_remaining=n_generations,
            problem_id=self.problem.id,
            active_hypothesis_ids=[hyp0.id],
        ))

        # -- generational search loop
        for gen in range(n_generations):
            if self.router is not None:
                self.router.handle_budget_signal(remaining_budget=n_generations - gen, total_budget=n_generations)
            parent = self.rng.choice(self.population)
            hyp_text = f"Improve on genome derived from {parent.model_type} for {self.task.name}"
            strategy, used_memory = self._select_strategy(parent)

            hyp = self.rdg.add_node(
                "Hypothesis", f"{hyp_text} using strategy '{strategy}'",
                attributes={"strategy": strategy, "generation": gen})
            self.rdg.add_edge(self.gap.id, hyp.id, "motivates")

            # alpha.2.1: synthesizer still works with ModelGenome API;
            # convert parent TMG → ModelGenome for synthesis, then wrap back.
            parent_mg = parent.to_model_genome()
            child_mg = self.synth.synthesize(strategy, parent_mg, self.rng,
                                              [g.to_model_genome() for g in self.population])
            child_mg.generation = gen
            child = TargetModelGenome.from_model_genome(child_mg)
            valid = unit_test(child_mg)

            exp = self.rdg.add_node(
                "Experiment", f"Train/evaluate {child.model_type} (gen {gen})",
                attributes={"genome_id": child.tmg_id})
            self.rdg.add_edge(hyp.id, exp.id, "tested-by")

            if not valid:
                exp_result = ExperimentResult(metric=0.0, success=False,
                                               exception="invalid genome",
                                               target=self.task.target_metric)
            else:
                exp_result = self._run_experiment(child)

            failure = diagnose(exp_result)
            signature = (strategy, child.model_type)
            if failure != FailureCategory.NONE:
                self._failed_signatures.add(signature)

            finding = self.rdg.add_node(
                "Finding", f"metric={exp_result.metric:.4f} failure={failure.value}",
                attributes={"metric": exp_result.metric, "failure": failure.value})
            self.rdg.add_edge(exp.id, finding.id, "produces")
            if failure == FailureCategory.NONE:
                claim = self.rdg.add_node(
                    "Claim", f"Strategy '{strategy}' improved/held the {self.task.name} model")
                self.rdg.add_edge(finding.id, claim.id, "supports")

            neg_transfer = used_memory and exp_result.metric < best_metric - 0.05
            if neg_transfer and self.use_flat_memory:
                self.ecrm.flag_negative_transfer(strategy)

            if self.use_flat_memory:
                self.ecrm.store(
                    text_summary=self._mem_key(strategy, child.model_type),
                    context={"task": self.task.name, "genome": child_mg.to_dict()},
                    outcome={"metric": exp_result.metric,
                             "success": failure == FailureCategory.NONE,
                             "failure": failure.value},
                    strategy=strategy)
                if gen % 8 == 7:
                    self.ecrm.consolidate()
            elif self.use_trajectory_memory:
                self.trajectory_memory.store(TrajectoryRecord(
                    id=new_trajectory_id(), generation=gen,
                    stage=generation_stage(gen, n_generations),
                    problem_context=self.gap.content,
                    parent_model_type=parent.model_type,
                    parent_capacity_bucket=capacity_bucket(parent_mg),
                    strategy=strategy,
                    child_model_type=child.model_type,
                    child_capacity_bucket=capacity_bucket(child_mg),
                    metric=exp_result.metric,
                    success=(failure == FailureCategory.NONE),
                    failure=failure.value,
                    hypothesis_id=hyp.id, experiment_id=exp.id, finding_id=finding.id))
            elif self.use_adaptive_trajectory:
                self.adaptive_trajectory_memory.store(AdaptiveTrajectoryRecord(
                    id=new_adaptive_trajectory_id(), generation=gen,
                    stage=generation_stage(gen, n_generations),
                    problem_context=self.gap.content,
                    parent_model_type=parent.model_type,
                    parent_capacity_bucket=capacity_bucket(parent_mg),
                    strategy=strategy,
                    child_model_type=child.model_type,
                    child_capacity_bucket=capacity_bucket(child_mg),
                    metric=exp_result.metric,
                    success=(failure == FailureCategory.NONE),
                    failure=failure.value,
                    hypothesis_id=hyp.id, spec_id=exp.id, run_id=f"run_gen{gen}_{child.tmg_id}",
                    finding_id=finding.id, provenance_id=f"prov_gen{gen}_{child.tmg_id}"))

            if self.use_policy:
                self.policy.update(strategy, reward=exp_result.metric)

            child._score = exp_result.metric if valid else -1.0
            if valid and exp_result.metric > best_metric:
                best_metric = exp_result.metric
                best_genome = child

            self.population.append(child)
            self.population.sort(key=lambda g: -getattr(g, "_score", -1.0))
            if len(self.population) > self.population_size:
                self.population = self.population[: self.population_size]

            if self.use_adaptive_trajectory and self._last_decision_metadata:
                self._record_trial(
                    result, gen, strategy, child.model_type, exp_result,
                    best_metric, failure, used_memory, neg_transfer, child.tmg_id,
                    context_level=self._last_decision_metadata.get("context_level"),
                    requested_context_level=self._last_decision_metadata.get("requested_context_level"),
                    sample_count=self._last_decision_metadata.get("sample_count"),
                    fallback_reason=self._last_decision_metadata.get("fallback_reason"),
                    memory_decision_contribution=self._last_decision_metadata.get("memory_decision_contribution"),
                )
            else:
                self._record_trial(result, gen, strategy, child.model_type, exp_result,
                                    best_metric, failure, used_memory, neg_transfer, child.tmg_id)
            result.states.append(ResearchState.create(
                generation=gen,
                research_phase="exploration" if self.rsg is None else self.rsg.research_phase,
                active_rsg_id="" if self.rsg is None else self.rsg.rsg_id,
                active_rsg_fingerprint="" if self.rsg is None else self.rsg.fingerprint(),
                candidate_tmg_ids=[g.tmg_id for g in self.population],
                best_tmg_id=best_genome.tmg_id if best_genome else None,
                best_metric=best_metric,
                budget_remaining=n_generations - (gen + 1),
                problem_id=self.problem.id,
                active_hypothesis_ids=[hyp.id],
                unresolved_failure_ids=[f"{sig[0]}_{sig[1]}" for sig in self._failed_signatures],
            ))

        result.best_genome = best_genome
        result.best_metric = best_metric
        result.rdg_stats = self.rdg.stats()
        result.memory_half_life_days = (
            self.ecrm.memory_half_life_days() if self.use_flat_memory else float("nan"))
        result.trajectory_stats = (
            self.trajectory_memory.stats() if self.use_trajectory_memory else {})
        result.adaptive_trajectory_stats = (
            self.adaptive_trajectory_memory.stats() if self.use_adaptive_trajectory else {})
        result.wall_time_s = time.time() - t0
        result.rsg_id = self.rsg.rsg_id if self.rsg is not None else None
        return result

    def export_memory_fingerprint(self) -> str:
        """Computes a deterministic content-addressed hash of current memory state."""
        import hashlib
        if not self.use_memory:
            return "no_memory_disabled"
        ecrm_len = len(self.ecrm.records) if self.ecrm else 0
        traj_len = len(self.trajectory_memory.records) if self.trajectory_memory else 0
        adapt_len = len(self.adaptive_trajectory_memory.records) if self.adaptive_trajectory_memory else 0
        failed = sorted(list(str(s) for s in self._failed_signatures))
        parts = [
            f"ecrm:{ecrm_len}",
            f"traj:{traj_len}",
            f"adaptive:{adapt_len}",
            f"failed:{failed}",
        ]
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def reset_memory(self) -> None:
        """Explicitly reset all accumulated memory state for cold-start boundaries."""
        self.ecrm = ECRM()
        self.trajectory_memory = TrajectoryMemory()
        self.adaptive_trajectory_memory = AdaptiveTrajectoryMemory()
        self._failed_signatures = set()


