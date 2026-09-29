"""researchforge/discovery/operator_genome.py — RF-3 Algorithm Discovery.

Defines the Symbolic AST / DSL for mutation operators.
Instead of hardcoding 'increase_capacity' in Python, it is expressed as a 
composable graph of primitive operations.
"""
from __future__ import annotations

import random
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class OperatorPrimitive:
    """A single primitive step in an algorithm."""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    
    def mutate(self, rng: random.Random) -> OperatorPrimitive:
        """Slightly mutates the parameters of this primitive."""
        new_params = dict(self.params)
        for k, v in new_params.items():
            if isinstance(v, float):
                # Small Gaussian noise
                new_params[k] = max(0.1, v + rng.gauss(0, 0.2 * v))
            elif isinstance(v, int):
                # Random walk
                step = rng.choice([-1, 1])
                new_params[k] = max(1, v + step)
        return OperatorPrimitive(name=self.name, params=new_params)

@dataclass
class OperatorGenome:
    """A full mutation operator, composed of a sequence of primitives."""
    name: str
    primitives: List[OperatorPrimitive]
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    utility_score: float = 0.0 # Empirically estimated Delta
    
    def mutate(self, rng: random.Random) -> OperatorGenome:
        """Evolves the algorithm structurally."""
        new_primitives = [p.mutate(rng) for p in self.primitives]
        
        # Structural mutations: Add or remove primitives
        if rng.random() < 0.2 and len(new_primitives) > 1:
            # Delete random primitive
            idx = rng.randint(0, len(new_primitives)-1)
            new_primitives.pop(idx)
            
        if rng.random() < 0.2:
            # Insert random primitive (mock list for now)
            prims = ["FindBottleneck", "ExpandWidth", "InsertResidual", "PruneLowestMagnitude"]
            p = OperatorPrimitive(name=rng.choice(prims), params={"factor": rng.uniform(1.1, 2.0)})
            idx = rng.randint(0, len(new_primitives))
            new_primitives.insert(idx, p)
            
        return OperatorGenome(
            name=f"Evolved_{self.name}_{rng.randint(100,999)}",
            primitives=new_primitives
        )
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "primitives": [{"name": p.name, "params": p.params} for p in self.primitives],
            "utility_score": self.utility_score
        }

# --- Base Operators (Seed Population) ---

def create_increase_capacity_operator() -> OperatorGenome:
    return OperatorGenome(
        name="increase_capacity_dsl",
        primitives=[
            OperatorPrimitive("FindBottleneck", {"metric": "activation_variance"}),
            OperatorPrimitive("ExpandWidth", {"factor": 1.5})
        ]
    )

def create_add_skip_connection_operator() -> OperatorGenome:
    return OperatorGenome(
        name="add_skip_connection_dsl",
        primitives=[
            OperatorPrimitive("IdentifyDeepestPath", {}),
            OperatorPrimitive("InsertResidual", {"stride_match": True})
        ]
    )
