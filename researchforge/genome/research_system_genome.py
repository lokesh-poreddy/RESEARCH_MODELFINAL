"""ResearchSystemGenome (RSG): the versioned, typed genome that governs
*how* ResearchForge conducts research — its strategy, budget, memory
policy, validity gate configuration, and operator portfolio.

Conceptual distinction (AD-011)
---------------------------------
  RSG → evolves the research process/strategy
  TMG → evolves the target model/solution

This two-level evolutionary architecture allows ResearchForge to eventually
discover that the research strategy itself is suboptimal (not merely the
model), and adapt accordingly. This becomes foundational for RF-1.5/RF-3.0.

The RSG is a *versioned executable specification of the ResearchForge
research policy and operating constraints*. It is not a configuration dump.
Each RSG is an immutable research artifact: create evolved children via
``evolve()``, do not modify in place.

RSG operator namespace (AD-010)
---------------------------------
  RSG_EVOLUTION_OPERATORS — the set of operators in the research-strategy
  space. Deliberately distinct from TMG_OPERATORS (model-solution space) to
  avoid naming collisions when the research log records both.

  TMG operators: increase_capacity, change_family, crossover, …
  RSG operators: expand_budget, shift_phase, alter_memory_policy, …

researchforge_version vs schema_version (AD-012)
-------------------------------------------------
  schema_version : str
      Governs how this RSG dict should be parsed. Bumped when the RSG
      data model changes incompatibly. Current: "1.0".

  researchforge_version : str
      Records the RF software release that *created* this RSG. Distinct
      from schema_version because the same schema can be produced by
      multiple RF releases. Essential for reproducibility when ResearchForge
      starts evolving itself (RF-3.0+).
"""
from __future__ import annotations

import copy
import json
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .schema import (
    GENOME_SCHEMA_VERSION_RSG,
    deterministic_genome_id,
    genome_fingerprint,
    validate_genome,
)

# Import STRATEGIES so RSG.default() can reference the current TMG operator set
from .operators import STRATEGIES as _TMG_STRATEGIES

# --------------------------------------------------------------------------- #
# RSG operator namespace                                                       #
# --------------------------------------------------------------------------- #
RSG_EVOLUTION_OPERATORS: List[str] = [
    "init",                    # initial RSG construction
    "expand_budget",           # increase experiment_budget_per_hypothesis
    "contract_budget",         # decrease experiment_budget_per_hypothesis
    "shift_to_exploration",    # move research_phase toward exploration
    "shift_to_exploitation",   # move research_phase toward exploitation
    "alter_exploration_constant",  # change UCB exploration constant
    "alter_memory_decay",      # change memory config decay_lambda
    "reweight_operators",      # change operator prior weights
    "add_operator",            # add a TMG operator to the allowed portfolio
    "remove_operator",         # remove a TMG operator from the portfolio
    "crossover_policies",      # combine two RSG operator portfolios
    "legacy_import",           # created from condition string (backward compat)
]

# --------------------------------------------------------------------------- #
# JSON Schema                                                                  #
# --------------------------------------------------------------------------- #
GENOME_SCHEMA_RSG: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ResearchSystemGenome",
    "type": "object",
    "additionalProperties": False,
    "required": ["rsg_id", "schema_version", "research_phase",
                 "max_generations", "allowed_operators"],
    "properties": {
        "rsg_id": {"type": "string", "minLength": 1},
        "schema_version": {"type": "string", "enum": [GENOME_SCHEMA_VERSION_RSG]},
        "researchforge_version": {"type": "string"},
        "created_at": {"type": "number"},
        "operator": {"type": "string"},
        "parent_rsg_id": {"oneOf": [{"type": "null"}, {"type": "string"}]},
        "generation": {"type": "integer", "minimum": 0},
        "seed": {"type": "integer"},
        "research_phase": {"type": "string",
                           "enum": ["exploration", "exploitation", "transition"]},
        "hypothesis_budget": {"type": "integer", "minimum": 1},
        "experiment_budget_per_hypothesis": {"type": "integer", "minimum": 1},
        "max_generations": {"type": "integer", "minimum": 1},
        "plateau_patience": {"type": "integer", "minimum": 1},
        "target_metric": {"oneOf": [{"type": "null"}, {"type": "number"}]},
        "allowed_operators": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "operator_prior_weights": {
            "type": "object",
            "additionalProperties": {"type": "number"},
        },
        "memory_config": {"type": "object"},
        "validity_config": {"type": "object"},
        "retrieval_config": {"type": "object"},
        "execution_config": {"type": "object"},
        "operator_config": {"type": "object"},
        "termination_config": {"type": "object"},
    },
}


# --------------------------------------------------------------------------- #
# Sub-config dataclasses                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class ResearchMemoryConfig:
    """Memory sub-configuration for an RSG.

    memory_design : "flat_ecrm" | "trajectory" | "adaptive_trajectory" | "none"
        Selects the memory architecture. Maps to ResearchController's
        ``condition`` values: "full" → flat_ecrm, "trajectory_memory" →
        trajectory, "adaptive_trajectory" → adaptive_trajectory,
        "no_memory"/"random" → none.
    """
    memory_design: str = "flat_ecrm"
    decay_lambda: float = 0.08
    retention_threshold: float = 0.12
    consolidate_every_n: int = 8
    min_context_samples: int = 3

    _VALID_DESIGNS = frozenset(["flat_ecrm", "trajectory", "adaptive_trajectory", "none"])

    def __post_init__(self) -> None:
        if self.memory_design not in self._VALID_DESIGNS:
            raise ValueError(
                f"memory_design must be one of {sorted(self._VALID_DESIGNS)}, "
                f"got {self.memory_design!r}"
            )
        if self.min_context_samples < 1:
            raise ValueError(
                f"min_context_samples must be >= 1, got {self.min_context_samples}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_design": self.memory_design,
            "decay_lambda": self.decay_lambda,
            "retention_threshold": self.retention_threshold,
            "consolidate_every_n": self.consolidate_every_n,
            "min_context_samples": self.min_context_samples,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResearchMemoryConfig":
        return cls(
            memory_design=d.get("memory_design", "flat_ecrm"),
            decay_lambda=float(d.get("decay_lambda", 0.08)),
            retention_threshold=float(d.get("retention_threshold", 0.12)),
            consolidate_every_n=int(d.get("consolidate_every_n", 8)),
            min_context_samples=int(d.get("min_context_samples", 3)),
        )



@dataclass
class ResearchValidityConfig:
    """Validity gate sub-configuration for an RSG.

    paired : bool
        True (default) uses paired t-test for seed-matched RF-vs-RF
        comparisons. False uses Welch's unpaired t-test for genuinely
        independent samples. See reviewer note in RESEARCHFORGE_STATE.yaml.
    """
    enabled: bool = True
    n_permutations: int = 5
    significance_alpha: float = 0.05
    paired: bool = True        # True = paired t-test (default for RF-vs-RF)
    min_gap: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "n_permutations": self.n_permutations,
            "significance_alpha": self.significance_alpha,
            "paired": self.paired,
            "min_gap": self.min_gap,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResearchValidityConfig":
        return cls(
            enabled=bool(d.get("enabled", True)),
            n_permutations=int(d.get("n_permutations", 5)),
            significance_alpha=float(d.get("significance_alpha", 0.05)),
            paired=bool(d.get("paired", True)),
            min_gap=float(d.get("min_gap", 0.05)),
        )


@dataclass
class ResearchRetrievalConfig:
    """Literature retrieval sub-configuration for an RSG."""
    enabled: bool = True
    max_results_per_source: int = 5
    sources: List[str] = field(default_factory=lambda: ["github"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_results_per_source": self.max_results_per_source,
            "sources": list(self.sources),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResearchRetrievalConfig":
        return cls(
            enabled=bool(d.get("enabled", True)),
            max_results_per_source=int(d.get("max_results_per_source", 5)),
            sources=list(d.get("sources", ["github"])),
        )



@dataclass
class ExecutionConfig:
    """Execution resource and sandbox sub-configuration for an RSG.

    IMPORTANT: This is explicitly SEPARATE from ResearchValidityConfig.
    ValidityConfig contains scientific parameters (n_permutations, alpha).
    ExecutionConfig contains resource/sandbox policy.
    These are different concerns and must never be conflated.

    execution_mode : "trusted_offline" | "sandboxed"
        trusted_offline — runs directly in the calling process without the
            subprocess sandbox. Chosen for development/testing environments
            where the operator explicitly trusts the execution context.
            NOTE: trusted_offline does NOT skip:
                - tmg.safety_check() (genome-level sanity)
                - schema validation of all inputs/outputs
                - provenance capture (ExperimentRun record)
                - result validation
            Only the subprocess isolation (kill-switch) is omitted.
        sandboxed — runs in a forked subprocess with hard wall-clock timeout
            and process kill. Required for untrusted code or production.
    """
    execution_mode: str = "trusted_offline"   # "trusted_offline" | "sandboxed"
    per_experiment_timeout_s: float = 20.0
    max_total_experiments: int = 200
    max_wall_time_s: float = 3600.0
    # Future: cgroup limits, GPU quota, network policy, filesystem confinement

    _VALID_MODES = frozenset(["trusted_offline", "sandboxed"])

    def __post_init__(self) -> None:
        if self.execution_mode not in self._VALID_MODES:
            raise ValueError(
                f"execution_mode must be one of {sorted(self._VALID_MODES)}, "
                f"got {self.execution_mode!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_mode": self.execution_mode,
            "per_experiment_timeout_s": self.per_experiment_timeout_s,
            "max_total_experiments": self.max_total_experiments,
            "max_wall_time_s": self.max_wall_time_s,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExecutionConfig":
        return cls(
            execution_mode=d.get("execution_mode", "trusted_offline"),
            per_experiment_timeout_s=float(d.get("per_experiment_timeout_s", 20.0)),
            max_total_experiments=int(d.get("max_total_experiments", 200)),
            max_wall_time_s=float(d.get("max_wall_time_s", 3600.0)),
        )


@dataclass
class OperatorConfig:
    """TMG operator portfolio sub-configuration for an RSG.

    Controls which operators are currently active and their relative weights.
    Separate from the RSG-level allowed_operators list (which is the canonical
    source); this config holds derived metadata about the portfolio.
    """
    exploration_constant: float = 1.41   # UCB exploration constant (sqrt(2) default)
    min_portfolio_size: int = 2          # safety floor: always keep >=2 operators
    max_portfolio_size: int = 10         # ceiling on active operators

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exploration_constant": self.exploration_constant,
            "min_portfolio_size": self.min_portfolio_size,
            "max_portfolio_size": self.max_portfolio_size,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OperatorConfig":
        return cls(
            exploration_constant=float(d.get("exploration_constant", 1.41)),
            min_portfolio_size=int(d.get("min_portfolio_size", 2)),
            max_portfolio_size=int(d.get("max_portfolio_size", 10)),
        )


@dataclass
class TerminationConfig:
    """Termination criteria sub-configuration for an RSG."""
    plateau_patience: int = 8       # generations with no improvement before stopping
    target_metric: Optional[float] = None  # stop if this metric is reached
    min_generations: int = 3        # never stop before this many generations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plateau_patience": self.plateau_patience,
            "target_metric": self.target_metric,
            "min_generations": self.min_generations,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TerminationConfig":
        return cls(
            plateau_patience=int(d.get("plateau_patience", 8)),
            target_metric=d.get("target_metric"),
            min_generations=int(d.get("min_generations", 3)),
        )


# --------------------------------------------------------------------------- #
# ResearchSystemGenome                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class ResearchSystemGenome:
    """Versioned, executable specification of the ResearchForge research policy.

    Governs research strategy (exploration/exploitation), budget allocation,
    TMG operator portfolio, memory policy, validity gate configuration, and
    termination criteria.

    Immutability contract: use ``evolve()`` to create modified children.
    Do not mutate attributes in place — an RSG is a versioned research artifact.
    """

    # Research phase & budget
    research_phase: str                             # "exploration" | "exploitation" | "transition"
    max_generations: int                            # search horizon
    allowed_operators: List[str]                    # subset of TMG_OPERATORS
    operator_prior_weights: Dict[str, float]        # UCB initial weights per TMG operator
    memory_config: ResearchMemoryConfig
    validity_config: ResearchValidityConfig
    retrieval_config: ResearchRetrievalConfig
    execution_config: ExecutionConfig               # NEW alpha.2.1: resource/sandbox policy
    operator_config: OperatorConfig                 # NEW alpha.2.1: portfolio metadata
    termination_config: TerminationConfig           # NEW alpha.2.1: termination criteria

    hypothesis_budget: int = 3
    experiment_budget_per_hypothesis: int = 5
    # plateau_patience moved to termination_config; kept here for backward compat
    plateau_patience: int = 8
    target_metric: Optional[float] = None

    # Identity
    rsg_id: str = field(default_factory=lambda: f"rsg_{uuid.uuid4().hex[:10]}")
    schema_version: str = field(default=GENOME_SCHEMA_VERSION_RSG)
    created_at: float = field(default_factory=time.time)
    operator: str = "init"
    parent_rsg_id: Optional[str] = None
    generation: int = 0
    seed: int = 0

    # RF software version that created this RSG (AD-012)
    researchforge_version: str = "RF-1.0.0-alpha.2.1"

    # ------------------------------------------------------------------ #
    # Factories                                                            #
    # ------------------------------------------------------------------ #
    @classmethod
    def default(cls, condition: str = "full", seed: int = 0) -> "ResearchSystemGenome":
        """Create a canonical RSG matching the existing RF-1.0-alpha.1 controller defaults.

        The RSG produced by this factory is the explicit, versioned
        specification of what the controller does when rsg=None. This is the
        equivalence contract tested in test_genomes.py.

        Parameters
        ----------
        condition : "full" | "trajectory_memory" | "adaptive_trajectory" | "no_memory" | "random"
        seed : RNG seed for deterministic RSG identity
        """
        _memory_map = {
            "full": "flat_ecrm",
            "trajectory_memory": "trajectory",
            "adaptive_trajectory": "adaptive_trajectory",
            "no_memory": "none",
            "random": "none",
        }
        memory_design = _memory_map.get(condition, "flat_ecrm")
        memory_config = ResearchMemoryConfig(memory_design=memory_design)


        # All TMG operators allowed; equal prior weights
        operators = list(_TMG_STRATEGIES)
        weights = {op: 1.0 for op in operators}

        rsg_id = deterministic_genome_id(
            "rsg", [], "init", 0, seed,
            mutation_parameters={"condition": condition},
        )

        return cls(
            rsg_id=rsg_id,
            operator="init",
            generation=0,
            seed=seed,
            research_phase="exploration",
            hypothesis_budget=3,
            experiment_budget_per_hypothesis=5,
            max_generations=25,
            plateau_patience=8,
            target_metric=None,
            allowed_operators=operators,
            operator_prior_weights=weights,
            memory_config=memory_config,
            validity_config=ResearchValidityConfig(),
            retrieval_config=ResearchRetrievalConfig(),
            execution_config=ExecutionConfig(),
            operator_config=OperatorConfig(),
            termination_config=TerminationConfig(),
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResearchSystemGenome":
        """Deserialize from a dict, applying migration if needed."""
        from .migration import migrate_rsg
        d = migrate_rsg(dict(d))
        return cls(
            rsg_id=d.get("rsg_id", f"rsg_{uuid.uuid4().hex[:10]}"),
            schema_version=d.get("schema_version", GENOME_SCHEMA_VERSION_RSG),
            created_at=d.get("created_at", time.time()),
            operator=d.get("operator", "legacy_import"),
            parent_rsg_id=d.get("parent_rsg_id"),
            generation=int(d.get("generation", 0)),
            seed=int(d.get("seed", 0)),
            researchforge_version=d.get("researchforge_version", "RF-0.x"),
            research_phase=d.get("research_phase", "exploration"),
            hypothesis_budget=int(d.get("hypothesis_budget", 3)),
            experiment_budget_per_hypothesis=int(
                d.get("experiment_budget_per_hypothesis", 5)),
            max_generations=int(d.get("max_generations", 25)),
            plateau_patience=int(d.get("plateau_patience", 8)),
            target_metric=d.get("target_metric"),
            allowed_operators=list(d.get("allowed_operators", list(_TMG_STRATEGIES))),
            operator_prior_weights=dict(d.get("operator_prior_weights", {})),
            memory_config=ResearchMemoryConfig.from_dict(d.get("memory_config", {})),
            validity_config=ResearchValidityConfig.from_dict(d.get("validity_config", {})),
            retrieval_config=ResearchRetrievalConfig.from_dict(d.get("retrieval_config", {})),
            execution_config=ExecutionConfig.from_dict(d.get("execution_config", {})),
            operator_config=OperatorConfig.from_dict(d.get("operator_config", {})),
            termination_config=TerminationConfig.from_dict(d.get("termination_config", {})),
        )

    # ------------------------------------------------------------------ #
    # Serialization                                                        #
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rsg_id": self.rsg_id,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "operator": self.operator,
            "parent_rsg_id": self.parent_rsg_id,
            "generation": self.generation,
            "seed": self.seed,
            "researchforge_version": self.researchforge_version,
            "research_phase": self.research_phase,
            "hypothesis_budget": self.hypothesis_budget,
            "experiment_budget_per_hypothesis": self.experiment_budget_per_hypothesis,
            "max_generations": self.max_generations,
            "plateau_patience": self.plateau_patience,
            "target_metric": self.target_metric,
            "allowed_operators": list(self.allowed_operators),
            "operator_prior_weights": dict(self.operator_prior_weights),
            "memory_config": self.memory_config.to_dict(),
            "validity_config": self.validity_config.to_dict(),
            "retrieval_config": self.retrieval_config.to_dict(),
            "execution_config": self.execution_config.to_dict(),
            "operator_config": self.operator_config.to_dict(),
            "termination_config": self.termination_config.to_dict(),
        }

    def canonical_dict(self) -> Dict[str, Any]:
        """Canonical dict for fingerprinting — excludes volatile timestamps."""
        d = self.to_dict()
        d.pop("created_at", None)
        return d

    def to_json(self) -> str:
        """Deterministic JSON with sorted keys."""
        return json.dumps(self.to_dict(), sort_keys=True, default=str)

    def fingerprint(self) -> str:
        """sha256 fingerprint of the canonical RSG serialization."""
        return genome_fingerprint(self.canonical_dict())

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Validate against the strict RSG JSON Schema."""
        validate_genome(self.to_dict(), GENOME_SCHEMA_RSG)

    # ------------------------------------------------------------------ #
    # Evolution (immutability contract)                                    #
    # ------------------------------------------------------------------ #
    def evolve(
        self,
        operator: str,
        rng: Optional[random.Random] = None,
        child_index: int = 0,
        mutation_params: Optional[Dict[str, Any]] = None,
    ) -> "ResearchSystemGenome":
        """Create an evolved child RSG. Does NOT mutate self.

        Immutability contract: always returns a new RSG. The parent (self)
        is never modified.

        Parameters
        ----------
        operator : str — one of RSG_EVOLUTION_OPERATORS
        rng : random.Random — used only by stochastic operators; None = deterministic
        child_index : int — ordinal for sibling-collision-safe ID
        mutation_params : dict — operator-specific parameters
        """
        if operator not in RSG_EVOLUTION_OPERATORS:
            raise ValueError(
                f"Unknown RSG operator {operator!r}. "
                f"Valid operators: {RSG_EVOLUTION_OPERATORS}"
            )

        mp: Dict[str, Any] = mutation_params or {}
        child = copy.deepcopy(self)
        child.rsg_id = deterministic_genome_id(
            "rsg",
            [self.rsg_id],
            operator,
            self.generation + 1,
            self.seed,
            mutation_parameters=mp,
            child_index=child_index,
        )
        child.parent_rsg_id = self.rsg_id
        child.generation = self.generation + 1
        child.operator = operator
        child.created_at = time.time()

        # Apply operator-specific mutation
        if operator == "expand_budget":
            delta = int(mp.get("delta", 2))
            child.experiment_budget_per_hypothesis = min(
                child.experiment_budget_per_hypothesis + delta, 50
            )
        elif operator == "contract_budget":
            delta = int(mp.get("delta", 1))
            child.experiment_budget_per_hypothesis = max(
                child.experiment_budget_per_hypothesis - delta, 1
            )
        elif operator == "shift_to_exploration":
            child.research_phase = "exploration"
        elif operator == "shift_to_exploitation":
            child.research_phase = "exploitation"
            # Tighten memory retention when exploiting
            child.memory_config = copy.deepcopy(child.memory_config)
            child.memory_config.decay_lambda = max(
                0.02, child.memory_config.decay_lambda * 0.5
            )
        elif operator == "alter_exploration_constant":
            new_alpha = float(mp.get("significance_alpha", 0.05))
            child.validity_config = copy.deepcopy(child.validity_config)
            child.validity_config.significance_alpha = new_alpha
        elif operator == "alter_memory_decay":
            new_decay = float(mp.get("decay_lambda", 0.08))
            child.memory_config = copy.deepcopy(child.memory_config)
            child.memory_config.decay_lambda = max(0.001, min(1.0, new_decay))
        elif operator == "reweight_operators":
            weights = dict(mp.get("weights", {}))
            child.operator_prior_weights = dict(child.operator_prior_weights)
            for op_name, w in weights.items():
                if op_name in child.operator_prior_weights:
                    child.operator_prior_weights[op_name] = float(w)
        elif operator == "add_operator":
            op_name = str(mp.get("operator_name", ""))
            if op_name and op_name not in child.allowed_operators:
                child.allowed_operators = list(child.allowed_operators) + [op_name]
                child.operator_prior_weights = dict(child.operator_prior_weights)
                child.operator_prior_weights[op_name] = float(mp.get("weight", 1.0))
        elif operator == "remove_operator":
            op_name = str(mp.get("operator_name", ""))
            if op_name in child.allowed_operators and len(child.allowed_operators) > 1:
                child.allowed_operators = [
                    o for o in child.allowed_operators if o != op_name
                ]
                child.operator_prior_weights = {
                    k: v for k, v in child.operator_prior_weights.items()
                    if k != op_name
                }
        # "crossover_policies" and "legacy_import" are handled by factories.

        return child
