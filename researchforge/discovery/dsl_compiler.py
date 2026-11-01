"""researchforge/discovery/dsl_compiler.py — RF-3 DSL Compiler.

Compiles an OperatorGenome (Symbolic AST DSL) into an executable Python function
that mutates a ModelGenome or genome dictionary. Supports all model families
present in operators.py (MLPClassifier, RandomForestClassifier, SVC, LogisticRegression).

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
_MIN_UNITS = 8
_MAX_UNITS = 2048
_MAX_LAYERS = 10
_MIN_LAYERS = 1


class CompilationError(Exception):
    """Raised when an OperatorGenome contains critical compilation errors."""
    pass


class DSLCompiler:
    """Translates OperatorGenomes into executable mutation strategies.

    Usage:
        compiler = DSLCompiler()
        compiled_fn = compiler.compile(op_genome)
        child = compiled_fn(parent_genome)
    """

    _MIN_UNITS: int = _MIN_UNITS
    _MAX_UNITS: int = _MAX_UNITS
    _MAX_LAYERS: int = _MAX_LAYERS
    _MIN_LAYERS: int = _MIN_LAYERS

    def __init__(self, rng_seed: int = 42) -> None:
        self.rng = random.Random(rng_seed)

        # Registry: primitive name → handler(genome, params, ctx) → None
        self.primitive_handlers: Dict[str, Callable] = {
            "FindBottleneck": self._handle_find_bottleneck,
            "ExpandWidth": self._handle_expand_width,
            "ReduceWidth": self._handle_reduce_width,
            "IdentifyDeepestPath": self._handle_identify_deepest_path,
            "InsertResidual": self._handle_insert_residual,
            "AddSkipConnection": self._handle_insert_residual,
            "PruneLowestMagnitude": self._handle_prune_lowest_magnitude,
            "AddLayer": self._handle_add_layer,
            "RemoveLayer": self._handle_remove_layer,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(self, op: OperatorGenome) -> Callable[[Any], Any]:
        """Compiles an OperatorGenome into a callable ModelGenome/dict mutation function.

        Unknown primitives are skipped gracefully without crashing.
        """
        # Capture a snapshot of the primitives at compile time (immutable closure)
        primitives_snapshot = copy.deepcopy(op.primitives)
        handlers = self.primitive_handlers  # dict lookup is safe to share

        def compiled_strategy(parent: Any) -> Any:
            if hasattr(parent, "clone"):
                child = parent.clone()
            else:
                child = copy.deepcopy(parent)

            execution_context: Dict[str, Any] = {}

            for prim in primitives_snapshot:
                handler = handlers.get(prim.name)
                if handler is None:
                    # Skip unknown primitive gracefully
                    continue
                try:
                    handler(child, prim.params, execution_context)
                except Exception as exc:
                    # Log but continue — evolution must not crash on a bad primitive
                    logger.debug(
                        "Primitive '%s' failed on genome %s: %s",
                        prim.name, getattr(child, "model_id", "dict"), exc,
                    )

            return child

        return compiled_strategy

    # ------------------------------------------------------------------
    # Primitive Handlers
    # ------------------------------------------------------------------

    def _handle_find_bottleneck(
        self, genome: Any, params: Dict[str, Any], context: Dict[str, Any]
    ) -> None:
        """Stores the index of the narrowest hidden layer in execution_context."""
        if isinstance(genome, dict):
            units = genome.get("hidden_units", [])
            if units:
                context["target_layer_idx"] = min(range(len(units)), key=lambda i: units[i])
        else:
            arch = getattr(genome, "architecture", {})
            if getattr(genome, "model_type", None) == "mlp":
                sizes = arch.get("hidden_layer_sizes", [])
                if sizes:
                    context["target_layer_idx"] = min(range(len(sizes)), key=lambda i: sizes[i])

    def _handle_expand_width(
        self, genome: Any, params: Dict[str, Any], context: Dict[str, Any]
    ) -> None:
        """Expands the width of the target layer by params['factor']."""
        factor = float(params.get("factor", 1.5))
        if factor <= 0.0:
            factor = 1.5
        elif 0.0 < factor < 1.0:
            factor = 1.0 / factor

        if isinstance(genome, dict):
            units = genome.get("hidden_units", [])
            if units:
                idx = context.get("target_layer_idx", len(units) - 1)
                idx = max(0, min(idx, len(units) - 1))
                new_size = int(units[idx] * factor)
                units[idx] = int(min(max(new_size, self._MIN_UNITS), self._MAX_UNITS))
        else:
            arch = getattr(genome, "architecture", {})
            if getattr(genome, "model_type", None) == "mlp":
                sizes = arch.get("hidden_layer_sizes", [])
                if sizes:
                    idx = context.get("target_layer_idx", len(sizes) - 1)
                    idx = max(0, min(idx, len(sizes) - 1))
                    new_size = int(sizes[idx] * factor)
                    sizes[idx] = int(min(max(new_size, self._MIN_UNITS), self._MAX_UNITS))

    def _handle_reduce_width(
        self, genome: Any, params: Dict[str, Any], context: Dict[str, Any]
    ) -> None:
        """Reduces the width of the target layer by params['factor']."""
        factor = float(params.get("factor", 1.5))
        if factor <= 0.0:
            factor = 1.5
        elif factor > 1.0:
            factor = 1.0 / factor

        if isinstance(genome, dict):
            units = genome.get("hidden_units", [])
            if units:
                idx = context.get("target_layer_idx", 0)
                idx = max(0, min(idx, len(units) - 1))
                new_size = int(units[idx] * factor)
                units[idx] = int(min(max(new_size, self._MIN_UNITS), self._MAX_UNITS))
        else:
            arch = getattr(genome, "architecture", {})
            if getattr(genome, "model_type", None) == "mlp":
                sizes = arch.get("hidden_layer_sizes", [])
                if sizes:
                    idx = context.get("target_layer_idx", 0)
                    idx = max(0, min(idx, len(sizes) - 1))
                    new_size = int(sizes[idx] * factor)
                    sizes[idx] = int(min(max(new_size, self._MIN_UNITS), self._MAX_UNITS))

    def _handle_identify_deepest_path(
        self, genome: Any, params: Dict[str, Any], context: Dict[str, Any]
    ) -> None:
        """Stores path bounds in execution_context."""
        if isinstance(genome, dict):
            n_layers = len(genome.get("hidden_units", []))
            context["path_start"] = 0
            context["path_end"] = max(0, n_layers - 1)
        else:
            if getattr(genome, "model_type", None) == "mlp":
                arch = getattr(genome, "architecture", {})
                n_layers = len(arch.get("hidden_layer_sizes", []))
                context["path_start"] = 0
                context["path_end"] = max(0, n_layers - 1)

    def _handle_insert_residual(
        self, genome: Any, params: Dict[str, Any], context: Dict[str, Any]
    ) -> None:
        """Inserts a skip-connection annotation in the architecture or dict."""
        if isinstance(genome, dict):
            genome["skip_connections"] = True
        else:
            arch = getattr(genome, "architecture", {})
            arch.setdefault("skip_connections", [])
            start = int(context.get("path_start", 0))
            end = int(context.get("path_end", 0))
            if end > start:
                arch["skip_connections"].append((start, end))

    def _handle_prune_lowest_magnitude(
        self, genome: Any, params: Dict[str, Any], context: Dict[str, Any]
    ) -> None:
        """Removes the smallest hidden layer (requires at least _MIN_LAYERS + 1)."""
        if isinstance(genome, dict):
            units = genome.get("hidden_units", [])
            if len(units) > self._MIN_LAYERS:
                min_idx = min(range(len(units)), key=lambda i: units[i])
                units.pop(min_idx)
        else:
            arch = getattr(genome, "architecture", {})
            if getattr(genome, "model_type", None) == "mlp":
                sizes = arch.get("hidden_layer_sizes", [])
                if len(sizes) > self._MIN_LAYERS:
                    min_idx = min(range(len(sizes)), key=lambda i: sizes[i])
                    sizes.pop(min_idx)

    def _handle_add_layer(
        self, genome: Any, params: Dict[str, Any], context: Dict[str, Any]
    ) -> None:
        """Inserts a new hidden layer of size params['units']."""
        units_param = int(min(max(int(params.get("units", 64)), self._MIN_UNITS), self._MAX_UNITS))
        if isinstance(genome, dict):
            units = genome.setdefault("hidden_units", [])
            if len(units) < self._MAX_LAYERS:
                idx = context.get("target_layer_idx", len(units))
                idx = max(0, min(idx, len(units)))
                units.insert(int(idx), units_param)
        else:
            arch = getattr(genome, "architecture", {})
            if getattr(genome, "model_type", None) == "mlp":
                sizes = arch.get("hidden_layer_sizes", [])
                if len(sizes) < self._MAX_LAYERS:
                    idx = context.get("target_layer_idx", len(sizes))
                    idx = max(0, min(idx, len(sizes)))
                    sizes.insert(int(idx), units_param)

    def _handle_remove_layer(
        self, genome: Any, params: Dict[str, Any], context: Dict[str, Any]
    ) -> None:
        """Removes the last hidden layer (requires at least _MIN_LAYERS + 1)."""
        if isinstance(genome, dict):
            units = genome.get("hidden_units", [])
            if len(units) > self._MIN_LAYERS:
                units.pop()
        else:
            arch = getattr(genome, "architecture", {})
            if getattr(genome, "model_type", None) == "mlp":
                sizes = arch.get("hidden_layer_sizes", [])
                if len(sizes) > self._MIN_LAYERS:
                    sizes.pop()
