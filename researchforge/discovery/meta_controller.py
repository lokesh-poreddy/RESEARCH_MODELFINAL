"""researchforge/discovery/meta_controller.py — RF-3 Meta Controller.

Sits above the ResearchController. Maintains a population of OperatorGenomes,
evaluates them using real pilot runs via DSLCompiler + ResearchController,
and promotes high-utility operators into the core ContextualBandit arm space.

Evaluation Protocol:
  For each candidate OperatorGenome:
    1. Compile it into a callable via DSLCompiler.
    2. Register it globally via operators.register_dynamic_strategy().
    3. Run a short pilot ResearchController run on a held-out evaluation task,
       recording best_metric.
    4. Run a matched cold-start control run (no operator registration).
    5. utility = pilot_metric - control_metric  (empirical lift estimate).
  
  Only operators with utility > promotion_threshold are added to the bandit.
"""
from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional, Tuple

from researchforge.benchmarks.continuity.tasks import get_pilot_task_sequence
from researchforge.discovery.dsl_compiler import DSLCompiler, CompilationError
from researchforge.discovery.operator_genome import (
    OperatorGenome,
    get_seed_population,
)
from researchforge.genome.operators import register_dynamic_strategy, DYNAMIC_STRATEGIES
from researchforge.pipeline.controller import ResearchController

logger = logging.getLogger(__name__)


class MetaController:
    """Discovers and evaluates new mutation algorithms via evolutionary search."""

    def __init__(
        self,
        seed: int = 42,
        population_size: int = 5,
        n_pilot_gens: int = 8,
        promotion_threshold: float = 0.002,
    ) -> None:
        """
        Args:
            seed: Master RNG seed.
            population_size: Maximum operators kept across generations.
            n_pilot_gens: Generations per pilot evaluation run (keep small for speed).
            promotion_threshold: Minimum empirical lift required to promote an operator
                                 into the live bandit arm space.
        """
        self.seed = seed
        self.rng = random.Random(seed)
        self.population_size = population_size
        self.n_pilot_gens = n_pilot_gens
        self.promotion_threshold = promotion_threshold

        self.compiler = DSLCompiler(rng_seed=seed)

        # Initialise with the hand-crafted seed population
        self.operator_population: List[OperatorGenome] = get_seed_population()

        # Evaluation tasks (use first task from pilot sequence)
        pilot_tasks = get_pilot_task_sequence(seed=seed)
        self.eval_task = pilot_tasks[0].task if pilot_tasks else None

        # Operators that have been promoted into the live bandit
        self.promoted_operators: List[OperatorGenome] = []

    # ------------------------------------------------------------------
    # Core meta-generation loop
    # ------------------------------------------------------------------

    def run_meta_generation(self) -> List[OperatorGenome]:
        """Runs one generation of Algorithm Discovery.

        Returns:
            List of newly promoted OperatorGenomes (may be empty).
        """
        logger.info(
            "Meta-Generation started. Population: %d", len(self.operator_population)
        )

        # 1. Breed new candidates
        children: List[OperatorGenome] = []
        for op in self.operator_population:
            if self.rng.random() < 0.6:
                try:
                    child = op.mutate(self.rng)
                    children.append(child)
                except Exception as exc:
                    logger.debug("Mutation failed for '%s': %s", op.name, exc)

        candidates = self.operator_population + children

        # 2. Evaluate candidates
        for op in candidates:
            if op.n_evaluations == 0:  # Only evaluate fresh operators
                score = self._evaluate_operator(op)
                op.utility_score = score
                op.n_evaluations += 1

        # 3. Select top operators
        candidates.sort(key=lambda o: o.utility_score, reverse=True)
        self.operator_population = candidates[: self.population_size]

        # 4. Promote operators above the threshold into the live bandit
        newly_promoted: List[OperatorGenome] = []
        for op in self.operator_population:
            if op.utility_score >= self.promotion_threshold and op.name not in DYNAMIC_STRATEGIES:
                try:
                    compiled_fn = self.compiler.compile(op)
                    register_dynamic_strategy(op.name, compiled_fn)
                    self.promoted_operators.append(op)
                    newly_promoted.append(op)
                    logger.info(
                        "Promoted '%s' (utility=%.4f) into live bandit.", 
                        op.name, op.utility_score
                    )
                except CompilationError as exc:
                    logger.warning("Cannot promote '%s': %s", op.name, exc)

        # 5. Report
        print(f"\n--- Meta-Generation Complete (pop={len(self.operator_population)}) ---")
        for i, op in enumerate(self.operator_population):
            tag = "✓ promoted" if op in newly_promoted else ""
            print(
                f"  {i+1:2d}. {op.name[:40]:40s} [utility={op.utility_score:+.4f}] {tag}"
            )

        return newly_promoted

    # ------------------------------------------------------------------
    # Operator Evaluation
    # ------------------------------------------------------------------

    def _evaluate_operator(self, op: OperatorGenome) -> float:
        """Estimates the empirical utility of an operator via a real pilot run.

        Returns the delta: pilot_metric - control_metric.
        Falls back to a structural heuristic if the eval_task is unavailable.
        """
        if self.eval_task is None:
            return self._structural_heuristic(op)

        try:
            # Compile candidate
            compiled_fn = self.compiler.compile(op)
        except CompilationError as exc:
            logger.debug("Cannot compile '%s': %s", op.name, exc)
            return -0.01  # Penalise uncompilable operators

        try:
            # --- Pilot run (with new operator registered) ---
            register_dynamic_strategy(op.name, compiled_fn)
            ctrl_pilot = ResearchController(
                self.eval_task, condition="contextual_policy", seed=self.seed
            )
            ctrl_pilot._contextual_bandit.add_action(op.name)
            res_pilot = ctrl_pilot.run(n_generations=self.n_pilot_gens)

            # --- Control run (cold start, no new operator) ---
            ctrl_control = ResearchController(
                self.eval_task, condition="contextual_policy", seed=self.seed
            )
            res_control = ctrl_control.run(n_generations=self.n_pilot_gens)

            delta = res_pilot.best_metric - res_control.best_metric
            logger.debug(
                "Evaluated '%s': pilot=%.4f control=%.4f delta=%+.4f",
                op.name, res_pilot.best_metric, res_control.best_metric, delta,
            )
            return float(delta)

        except Exception as exc:
            logger.warning("Pilot evaluation failed for '%s': %s", op.name, exc)
            return 0.0

    def _structural_heuristic(self, op: OperatorGenome) -> float:
        """Fallback scoring heuristic when no eval task is available."""
        score = 0.0
        for prim in op.primitives:
            if prim.name in ("ExpandWidth", "AddLayer"):
                score += 0.02
            elif prim.name in ("PruneLowestMagnitude", "RemoveLayer"):
                score += 0.01
            elif prim.name in ("InsertResidual",):
                score += 0.005
            else:
                score += self.rng.gauss(0, 0.005)
        return score

    def get_population_summary(self) -> List[Dict]:
        return [op.to_dict() for op in self.operator_population]
