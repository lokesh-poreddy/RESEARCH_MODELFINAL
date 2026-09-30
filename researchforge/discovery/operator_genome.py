"""researchforge/discovery/operator_genome.py — RF-3 Algorithm Discovery.

Defines the Symbolic AST / DSL for mutation operators.
Instead of hardcoding 'increase_capacity' in Python, operators are expressed
as a composable sequence of typed primitives. Each OperatorGenome can be:
  - Mutated (parameter noise + structural add/remove)
  - Serialised to dict / JSON for logging
  - Compiled to a Python callable via DSLCompiler

Primitive vocabulary must stay in sync with DSLCompiler.primitive_handlers.
"""
from __future__ import annotations

import copy
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Canonical list of all primitives the DSLCompiler knows how to handle.
# Keep this in sync with DSLCompiler.primitive_handlers keys.
# ---------------------------------------------------------------------------
KNOWN_PRIMITIVES: List[str] = [
    "FindBottleneck",
    "ExpandWidth",
    "IdentifyDeepestPath",
    "InsertResidual",
    "PruneLowestMagnitude",
    "ReduceWidth",
    "AddLayer",
    "RemoveLayer",
]


@dataclass
class OperatorPrimitive:
    """A single primitive step in an algorithm."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    def mutate(self, rng: random.Random) -> OperatorPrimitive:
        """Returns a copy of this primitive with slightly noisy parameters."""
        new_params = copy.deepcopy(self.params)
        for k, v in new_params.items():
            if isinstance(v, float):
                # Multiplicative Gaussian noise, bounded away from 0
                noise = rng.gauss(0.0, 0.15)
                new_params[k] = max(0.05, v * (1.0 + noise))
            elif isinstance(v, int) and k != "stride_match":
                # Random walk
                new_params[k] = max(1, v + rng.choice([-1, 0, 1]))
            elif isinstance(v, bool):
                # Occasional bit-flip
                if rng.random() < 0.1:
                    new_params[k] = not v
        return OperatorPrimitive(name=self.name, params=new_params)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "params": copy.deepcopy(self.params)}


@dataclass
class OperatorGenome:
    """A full mutation operator, composed of a sequence of OperatorPrimitives."""

    name: str
    primitives: List[OperatorPrimitive]
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    utility_score: float = 0.0   # Empirically estimated average Delta from pilot runs
    n_evaluations: int = 0       # Number of times this operator has been scored

    def mutate(self, rng: random.Random) -> OperatorGenome:
        """Structural + parametric mutation, returning a new OperatorGenome child."""
        # Step 1: Parametric mutation of all primitives
        new_primitives = [p.mutate(rng) for p in self.primitives]

        # Step 2: Structural deletion (only if > 1 primitive remains)
        if len(new_primitives) > 1 and rng.random() < 0.20:
            del_idx = rng.randint(0, len(new_primitives) - 1)
            new_primitives.pop(del_idx)

        # Step 3: Structural insertion (from known primitive vocabulary)
        if rng.random() < 0.20:
            prim_name = rng.choice(KNOWN_PRIMITIVES)
            factor = round(rng.uniform(1.1, 2.0), 2)
            new_prim = OperatorPrimitive(name=prim_name, params={"factor": factor})
            ins_idx = rng.randint(0, len(new_primitives))
            new_primitives.insert(ins_idx, new_prim)

        suffix = rng.randint(100, 999)
        return OperatorGenome(
            name=f"Evolved_{self.name}_{suffix}",
            primitives=new_primitives,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "primitives": [p.to_dict() for p in self.primitives],
            "utility_score": round(self.utility_score, 6),
            "n_evaluations": self.n_evaluations,
        }


# ---------------------------------------------------------------------------
# Seed Population — canonical starting operators
# ---------------------------------------------------------------------------

def create_increase_capacity_operator() -> OperatorGenome:
    return OperatorGenome(
        name="increase_capacity_dsl",
        primitives=[
            OperatorPrimitive("FindBottleneck", {"metric": "activation_variance"}),
            OperatorPrimitive("ExpandWidth", {"factor": 1.5}),
        ],
    )


def create_add_skip_connection_operator() -> OperatorGenome:
    return OperatorGenome(
        name="add_skip_connection_dsl",
        primitives=[
            OperatorPrimitive("IdentifyDeepestPath", {}),
            OperatorPrimitive("InsertResidual", {"stride_match": True}),
        ],
    )


def create_reduce_and_add_operator() -> OperatorGenome:
    """Prune the smallest layer, then add a new layer of moderate size."""
    return OperatorGenome(
        name="prune_and_grow_dsl",
        primitives=[
            OperatorPrimitive("PruneLowestMagnitude", {}),
            OperatorPrimitive("AddLayer", {"units": 64}),
        ],
    )


def get_seed_population() -> List[OperatorGenome]:
    """Returns the initial seed population of hand-crafted OperatorGenomes."""
    return [
        create_increase_capacity_operator(),
        create_add_skip_connection_operator(),
        create_reduce_and_add_operator(),
    ]
