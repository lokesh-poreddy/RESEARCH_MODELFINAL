"""researchforge/discovery/dsl_compiler.py — RF-3 DSL Compiler.

Compiles an OperatorGenome (Symbolic AST DSL) into an executable Python function
that mutates a ModelGenome. Supports all model families present in operators.py
(MLPClassifier, RandomForestClassifier, SVC, LogisticRegression).

Design principles:
  - Fail-soft: if a primitive cannot apply cleanly (e.g. only one layer, cannot prune),
    execution continues with remaining primitives rather than crashing.
  - Stateless primitives: each handler receives (genome, params, execution_context)
    and may read/write execution_context to communicate with downstream primitives.
  - Safety: all architecture mutations are bounds-checked before application.
"""
from __future__ import annotations

import copy
import logging
import random
from typing import Any, Callable, Dict, Optional

from researchforge.genome.model_genome import ModelGenome
from researchforge.discovery.operator_genome import OperatorGenome, OperatorPrimitive

logger = logging.getLogger(__name__)

# Architecture safety bounds (mirrors model_genome.ModelGenome.safety_check)
_MIN_UNITS = 4
_MAX_UNITS = 1024
_MAX_LAYERS = 6
_MIN_LAYERS = 1


class CompilationError(Exception):
    """Raised when an OperatorGenome contains unknown or invalid primitives."""
    pass


class DSLCompiler:
    """Translates OperatorGenomes into executable mutation strategies.

    Usage:
        compiler = DSLCompiler()
        compiled_fn = compiler.compile(op_genome)
        child = compiled_fn(parent_genome)
    """

    def __init__(self, rng_seed: int = 42) -> None:
        self.rng = random.Random(rng_seed)

        # Registry: primitive name → handler(genome, params, ctx) → None
        self.primitive_handlers: Dict[str, Callable] = {
            "FindBottleneck": self._handle_find_bottleneck,
            "ExpandWidth": self._handle_expand_width,
            "ReduceWidth": self._handle_reduce_width,
            "IdentifyDeepestPath": self._handle_identify_deepest_path,
            "InsertResidual": self._handle_insert_residual,
            "PruneLowestMagnitude": self._handle_prune_lowest_magnitude,
            "AddLayer": self._handle_add_layer,
            "RemoveLayer": self._handle_remove_layer,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(self, op: OperatorGenome) -> Callable[[ModelGenome], ModelGenome]:
        """Compiles an OperatorGenome into a callable ModelGenome → ModelGenome function.

        Raises:
            CompilationError: if any primitive in op.primitives is unregistered.
        """
        unknown = [p.name for p in op.primitives if p.name not in self.primitive_handlers]
        if unknown:
            raise CompilationError(
                f"OperatorGenome '{op.name}' contains unknown primitives: {unknown}. "
                f"Registered: {sorted(self.primitive_handlers)}"
            )

        # Capture a snapshot of the primitives at compile time (immutable closure)
        primitives_snapshot = copy.deepcopy(op.primitives)
        handlers = self.primitive_handlers  # dict lookup is safe to share

        def compiled_strategy(parent: ModelGenome) -> ModelGenome:
            child = parent.clone()
            execution_context: Dict[str, Any] = {}

            for prim in primitives_snapshot:
                handler = handlers[prim.name]
                try:
                    handler(child, prim.params, execution_context)
                except Exception as exc:
                    # Log but continue — evolution must not crash on a bad primitive
                    logger.debug(
                        "Primitive '%s' failed on genome %s: %s",
                        prim.name, child.model_id, exc,
                    )

            return child

        return compiled_strategy

    # ------------------------------------------------------------------
    # Primitive Handlers
    # ------------------------------------------------------------------

    def _handle_find_bottleneck(
        self, genome: ModelGenome, params: Dict[str, Any], context: Dict[str, Any]
    ) -> None:
        """Stores the index of the narrowest hidden layer in execution_context."""
        arch = genome.architecture
        if genome.model_type == "mlp":
            sizes = arch.get("hidden_layer_sizes", [])
            if sizes:
                context["target_layer_idx"] = min(range(len(sizes)), key=lambda i: sizes[i])

    def _handle_expand_width(
        self, genome: ModelGenome, params: Dict[str, Any], context: Dict[str, Any]
    ) -> None:
        """Multiplies the width of the target layer by params['factor']."""
        factor = float(params.get("factor", 1.5))
        arch = genome.architecture
        if genome.model_type == "mlp":
            sizes = arch.get("hidden_layer_sizes", [])
            idx = context.get("target_layer_idx", len(sizes) - 1)
            if sizes and 0 <= idx < len(sizes):
                new_size = int(sizes[idx] * factor)
                sizes[idx] = int(min(max(new_size, _MIN_UNITS), _MAX_UNITS))

    def _handle_reduce_width(
        self, genome: ModelGenome, params: Dict[str, Any], context: Dict[str, Any]
    ) -> None:
        """Divides the width of the target layer by params['factor']."""
        factor = float(params.get("factor", 1.5))
        arch = genome.architecture
        if genome.model_type == "mlp":
            sizes = arch.get("hidden_layer_sizes", [])
            idx = context.get("target_layer_idx", 0)
            if sizes and 0 <= idx < len(sizes):
                new_size = int(sizes[idx] / factor)
                sizes[idx] = int(min(max(new_size, _MIN_UNITS), _MAX_UNITS))

    def _handle_identify_deepest_path(
        self, genome: ModelGenome, params: Dict[str, Any], context: Dict[str, Any]
    ) -> None:
        """For an MLP, stores path bounds in execution_context."""
        if genome.model_type == "mlp":
            n_layers = len(genome.architecture.get("hidden_layer_sizes", []))
            context["path_start"] = 0
            context["path_end"] = max(0, n_layers - 1)

    def _handle_insert_residual(
        self, genome: ModelGenome, params: Dict[str, Any], context: Dict[str, Any]
    ) -> None:
        """Placeholder: Inserts a skip-connection annotation in the architecture dict.

        For sklearn-backed genomes, this is a no-op but is preserved as a valid
        primitive so RF-3 operators can include it without crashing.
        """
        arch = genome.architecture
        arch.setdefault("skip_connections", [])
        start = int(context.get("path_start", 0))
        end = int(context.get("path_end", 0))
        if end > start:
            arch["skip_connections"].append((start, end))

    def _handle_prune_lowest_magnitude(
        self, genome: ModelGenome, params: Dict[str, Any], context: Dict[str, Any]
    ) -> None:
        """Removes the smallest hidden layer (requires at least 2 layers)."""
        arch = genome.architecture
        if genome.model_type == "mlp":
            sizes = arch.get("hidden_layer_sizes", [])
            if len(sizes) > _MIN_LAYERS:
                min_idx = min(range(len(sizes)), key=lambda i: sizes[i])
                sizes.pop(min_idx)

    def _handle_add_layer(
        self, genome: ModelGenome, params: Dict[str, Any], context: Dict[str, Any]
    ) -> None:
        """Inserts a new hidden layer of size params['units']."""
        arch = genome.architecture
        if genome.model_type == "mlp":
            sizes = arch.get("hidden_layer_sizes", [])
            if len(sizes) < _MAX_LAYERS:
                units = int(min(max(int(params.get("units", 64)), _MIN_UNITS), _MAX_UNITS))
                # Insert after the bottleneck layer if known, else append
                idx = context.get("target_layer_idx", len(sizes))
                sizes.insert(int(idx), units)

    def _handle_remove_layer(
        self, genome: ModelGenome, params: Dict[str, Any], context: Dict[str, Any]
    ) -> None:
        """Removes the last hidden layer (requires at least 2 layers)."""
        arch = genome.architecture
        if genome.model_type == "mlp":
            sizes = arch.get("hidden_layer_sizes", [])
            if len(sizes) > _MIN_LAYERS:
                sizes.pop()
