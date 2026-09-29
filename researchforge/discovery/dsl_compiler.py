"""researchforge/discovery/dsl_compiler.py — RF-3 DSL Compiler.

Compiles an OperatorGenome (AST DSL) into an executable Python function 
that mutates a ModelGenome.
"""
from __future__ import annotations

import copy
import random
from typing import Callable, Any

from researchforge.genome.model_genome import ModelGenome
from researchforge.discovery.operator_genome import OperatorGenome, OperatorPrimitive

class CompilationError(Exception):
    """Raised when an OperatorGenome contains invalid/uncompilable semantics."""
    pass

class DSLCompiler:
    """Translates OperatorGenomes into executable mutation strategies."""
    
    def __init__(self, rng_seed: int = 42):
        self.rng = random.Random(rng_seed)
        
        # A registry mapping primitive names to actual Python mutation logic
        self.primitive_handlers = {
            "FindBottleneck": self._handle_find_bottleneck,
            "ExpandWidth": self._handle_expand_width,
            "IdentifyDeepestPath": self._handle_identify_deepest_path,
            "InsertResidual": self._handle_insert_residual,
            "PruneLowestMagnitude": self._handle_prune_lowest_magnitude
        }

    def compile(self, op: OperatorGenome) -> Callable[[ModelGenome], ModelGenome]:
        """
        Compiles an OperatorGenome into a callable function that takes a 
        parent ModelGenome and returns a mutated child ModelGenome.
        """
        # Validate first
        for prim in op.primitives:
            if prim.name not in self.primitive_handlers:
                raise CompilationError(f"Unknown primitive: {prim.name}")
        
        def compiled_strategy(parent: ModelGenome) -> ModelGenome:
            child = parent.clone()
            
            # Context state passed between primitives during execution
            # (e.g., FindBottleneck stores the layer index, ExpandWidth uses it)
            execution_context = {}
            
            for prim in op.primitives:
                handler = self.primitive_handlers[prim.name]
                try:
                    handler(child, prim.params, execution_context)
                except Exception as e:
                    # If a primitive fails to apply (e.g. tree too shallow to prune),
                    # we just skip it or log it. For this compiler, we fail softly
                    # so evolution doesn't completely crash.
                    pass 
                    
            return child
            
        return compiled_strategy

    # --- Primitive Handlers ---

    def _handle_find_bottleneck(self, genome: ModelGenome, params: dict, context: dict) -> None:
        """Identifies the narrowest layer or a layer with high activation variance."""
        arch = genome.architecture
        if genome.model_type == "mlp" and "hidden_layer_sizes" in arch:
            sizes = arch["hidden_layer_sizes"]
            if sizes:
                # Find index of minimum size
                min_idx = min(range(len(sizes)), key=sizes.__getitem__)
                context["target_layer_idx"] = min_idx

    def _handle_expand_width(self, genome: ModelGenome, params: dict, context: dict) -> None:
        """Multiplies the width of a targeted layer by `factor`."""
        factor = params.get("factor", 1.5)
        arch = genome.architecture
        
        if genome.model_type == "mlp" and "hidden_layer_sizes" in arch:
            idx = context.get("target_layer_idx")
            if idx is not None and 0 <= idx < len(arch["hidden_layer_sizes"]):
                old_size = arch["hidden_layer_sizes"][idx]
                arch["hidden_layer_sizes"][idx] = int(old_size * factor)

    def _handle_identify_deepest_path(self, genome: ModelGenome, params: dict, context: dict) -> None:
        """Placeholder for finding the longest path in a DAG architecture."""
        # For an MLP, this is trivial
        if genome.model_type == "mlp":
            context["path_start"] = 0
            context["path_end"] = len(genome.architecture.get("hidden_layer_sizes", [])) - 1

    def _handle_insert_residual(self, genome: ModelGenome, params: dict, context: dict) -> None:
        """Placeholder for inserting a skip connection."""
        pass # To be fully implemented for ResNet-style graphs
        
    def _handle_prune_lowest_magnitude(self, genome: ModelGenome, params: dict, context: dict) -> None:
        """Prunes the smallest layer if there are multiple."""
        arch = genome.architecture
        if genome.model_type == "mlp" and "hidden_layer_sizes" in arch:
            sizes = arch["hidden_layer_sizes"]
            if len(sizes) > 1:
                min_idx = min(range(len(sizes)), key=sizes.__getitem__)
                sizes.pop(min_idx)

# Example usage/test
if __name__ == "__main__":
    from researchforge.discovery.operator_genome import create_increase_capacity_operator
    compiler = DSLCompiler()
    op = create_increase_capacity_operator()
    
    compiled_func = compiler.compile(op)
    
    # Mock a base genome
    base_genome = ModelGenome(
        model_type="mlp",
        architecture={"hidden_layer_sizes": [64, 32, 64]},
        hyperparameters={"learning_rate": 0.001}
    )
    
    print(f"Base genome architecture: {base_genome.architecture}")
    child_genome = compiled_func(base_genome)
    print(f"Mutated genome architecture: {child_genome.architecture}")
