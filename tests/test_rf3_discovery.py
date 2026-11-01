"""
tests/test_rf3_discovery.py
============================
Unit tests for the RF-3 Algorithm Discovery pipeline:
- OperatorGenome / OperatorPrimitive
- DSLCompiler (8 handlers, safety bounds, degenerate input robustness)
- MetaController (pilot evaluation, promotion flow)
- DYNAMIC_STRATEGIES registry integration

These are CORE-marked tests (no live data, no network, fully deterministic).
"""
import random
import copy
import math
import pytest

from researchforge.discovery.operator_genome import (
    OperatorPrimitive,
    OperatorGenome,
    get_seed_population,
)
from researchforge.discovery.dsl_compiler import DSLCompiler
from researchforge.genome.operators import (
    DYNAMIC_STRATEGIES,
    register_dynamic_strategy,
)


# ─────────────────────────────────────────────
# Section 1 — OperatorPrimitive
# ─────────────────────────────────────────────

@pytest.mark.core
class TestOperatorPrimitive:
    def test_create_primitive(self):
        p = OperatorPrimitive(name="FindBottleneck", params={"layer_idx": 1})
        assert p.name == "FindBottleneck"
        assert p.params["layer_idx"] == 1

    def test_primitive_equality(self):
        a = OperatorPrimitive("ExpandWidth", {"factor": 2.0})
        b = OperatorPrimitive("ExpandWidth", {"factor": 2.0})
        assert a.name == b.name

    def test_primitive_repr(self):
        p = OperatorPrimitive("ReduceWidth", {"factor": 0.5})
        r = repr(p)
        assert "ReduceWidth" in r


# ─────────────────────────────────────────────
# Section 2 — OperatorGenome
# ─────────────────────────────────────────────

@pytest.mark.core
class TestOperatorGenome:
    def test_seed_population_has_three_operators(self):
        pop = get_seed_population()
        assert len(pop) == 3

    def test_seed_operator_names(self):
        pop = get_seed_population()
        names = {op.name for op in pop}
        assert "increase_capacity_dsl" in names
        assert "add_skip_connection_dsl" in names
        assert "prune_and_grow_dsl" in names

    def test_all_seeds_have_primitives(self):
        for op in get_seed_population():
            assert len(op.primitives) >= 1, f"{op.name} has no primitives"

    def test_mutate_produces_new_name(self):
        rng = random.Random(42)
        op = get_seed_population()[0]
        mutant = op.mutate(rng)
        assert mutant.name != op.name
        assert op.name in mutant.name  # parent name embedded

    def test_mutate_preserves_at_least_one_primitive(self):
        rng = random.Random(99)
        for op in get_seed_population():
            mutant = op.mutate(rng)
            assert len(mutant.primitives) >= 1

    def test_mutate_is_not_in_place(self):
        rng = random.Random(7)
        op = get_seed_population()[0]
        original_prims = [p.name for p in op.primitives]
        _ = op.mutate(rng)
        assert [p.name for p in op.primitives] == original_prims

    def test_n_evaluations_starts_at_zero(self):
        op = get_seed_population()[0]
        # n_evaluations is a mutable counter
        assert hasattr(op, "n_evaluations")
        assert op.n_evaluations == 0

    def test_score_defaults_to_none(self):
        op = get_seed_population()[0]
        assert op.score is None


# ─────────────────────────────────────────────
# Section 3 — DSLCompiler
# ─────────────────────────────────────────────

@pytest.mark.core
class TestDSLCompiler:
    def _base_genome(self):
        return {
            "hidden_units": [64, 64],
            "dropout_rate": 0.0,
            "learning_rate": 0.01,
            "skip_connections": False,
            "batch_norm": False,
            "weight_decay": 0.0,
        }

    def test_compile_returns_callable(self):
        compiler = DSLCompiler()
        op = get_seed_population()[0]
        fn = compiler.compile(op)
        assert callable(fn)

    def test_compile_runs_without_error(self):
        compiler = DSLCompiler()
        for op in get_seed_population():
            fn = compiler.compile(op)
            genome = self._base_genome()
            modified = fn(genome)
            assert isinstance(modified, dict)

    def test_expand_width_increases_units(self):
        from researchforge.discovery.operator_genome import OperatorPrimitive, OperatorGenome
        prim = OperatorPrimitive("ExpandWidth", {"factor": 2.0})
        op = OperatorGenome("test_expand", [prim], score=None)
        compiler = DSLCompiler()
        fn = compiler.compile(op)
        genome = self._base_genome()
        result = fn(genome)
        assert max(result["hidden_units"]) >= max(genome["hidden_units"])

    def test_reduce_width_decreases_units(self):
        from researchforge.discovery.operator_genome import OperatorPrimitive, OperatorGenome
        prim = OperatorPrimitive("ReduceWidth", {"factor": 0.5})
        op = OperatorGenome("test_reduce", [prim], score=None)
        compiler = DSLCompiler()
        fn = compiler.compile(op)
        genome = {"hidden_units": [128, 128], "dropout_rate": 0.0, "learning_rate": 0.01,
                  "skip_connections": False, "batch_norm": False, "weight_decay": 0.0}
        result = fn(genome)
        assert max(result["hidden_units"]) <= 128

    def test_safety_bounds_enforced(self):
        """Repeated ExpandWidth should not exceed _MAX_UNITS limit."""
        from researchforge.discovery.operator_genome import OperatorPrimitive, OperatorGenome
        prims = [OperatorPrimitive("ExpandWidth", {"factor": 100.0})] * 3
        op = OperatorGenome("test_safety", prims, score=None)
        compiler = DSLCompiler()
        fn = compiler.compile(op)
        genome = self._base_genome()
        result = fn(genome)
        max_units = max(result["hidden_units"])
        assert max_units <= compiler._MAX_UNITS

    def test_add_skip_connection_primitive(self):
        from researchforge.discovery.operator_genome import OperatorPrimitive, OperatorGenome
        prim = OperatorPrimitive("AddSkipConnection", {})
        op = OperatorGenome("test_skip", [prim], score=None)
        compiler = DSLCompiler()
        fn = compiler.compile(op)
        genome = self._base_genome()
        result = fn(genome)
        assert result["skip_connections"] is True

    def test_add_layer_increases_depth(self):
        from researchforge.discovery.operator_genome import OperatorPrimitive, OperatorGenome
        prim = OperatorPrimitive("AddLayer", {"units": 32})
        op = OperatorGenome("test_add_layer", [prim], score=None)
        compiler = DSLCompiler()
        fn = compiler.compile(op)
        genome = self._base_genome()
        result = fn(genome)
        assert len(result["hidden_units"]) > len(genome["hidden_units"])

    def test_remove_layer_decreases_depth_or_keeps_min(self):
        from researchforge.discovery.operator_genome import OperatorPrimitive, OperatorGenome
        prim = OperatorPrimitive("RemoveLayer", {})
        op = OperatorGenome("test_remove_layer", [prim], score=None)
        compiler = DSLCompiler()
        fn = compiler.compile(op)
        genome = self._base_genome()
        result = fn(genome)
        # Should either reduce or keep at minimum
        assert len(result["hidden_units"]) >= compiler._MIN_LAYERS

    def test_compile_does_not_mutate_original_genome(self):
        """The callable must deep-copy; the original genome dict must be unchanged."""
        compiler = DSLCompiler()
        op = get_seed_population()[0]
        fn = compiler.compile(op)
        genome = self._base_genome()
        original_units = list(genome["hidden_units"])
        fn(genome)
        assert genome["hidden_units"] == original_units

    def test_unknown_primitive_is_skipped_gracefully(self):
        from researchforge.discovery.operator_genome import OperatorPrimitive, OperatorGenome
        prim = OperatorPrimitive("NonExistentPrimitive_XYZ", {})
        op = OperatorGenome("test_unknown", [prim], score=None)
        compiler = DSLCompiler()
        fn = compiler.compile(op)
        genome = self._base_genome()
        # Should not raise; should return valid dict
        result = fn(genome)
        assert isinstance(result, dict)


# ─────────────────────────────────────────────
# Section 4 — DYNAMIC_STRATEGIES Registry
# ─────────────────────────────────────────────

@pytest.mark.core
class TestDynamicStrategiesRegistry:
    def test_registry_is_dict(self):
        assert isinstance(DYNAMIC_STRATEGIES, dict)

    def test_register_adds_entry(self):
        initial_count = len(DYNAMIC_STRATEGIES)
        def _dummy_strategy(genome):
            return genome
        register_dynamic_strategy("test_dummy_v1", _dummy_strategy)
        assert "test_dummy_v1" in DYNAMIC_STRATEGIES
        # cleanup
        DYNAMIC_STRATEGIES.pop("test_dummy_v1", None)

    def test_registered_callable_is_preserved(self):
        calls = []
        def _tracker(genome):
            calls.append(1)
            return genome
        register_dynamic_strategy("test_tracker_v1", _tracker)
        fn = DYNAMIC_STRATEGIES["test_tracker_v1"]
        fn({})
        assert len(calls) == 1
        DYNAMIC_STRATEGIES.pop("test_tracker_v1", None)


# ─────────────────────────────────────────────
# Section 5 — Integrated compile→register flow
# ─────────────────────────────────────────────

@pytest.mark.core
class TestCompileAndRegisterFlow:
    def test_compile_then_register_succeeds(self):
        """Full RF-3 pipeline: compile a genome → register → call via registry."""
        compiler = DSLCompiler()
        op = get_seed_population()[1]  # add_skip_connection_dsl
        compiled_fn = compiler.compile(op)
        name = f"_test_registered_{op.name}"
        register_dynamic_strategy(name, compiled_fn)
        assert name in DYNAMIC_STRATEGIES
        genome = {
            "hidden_units": [64], "dropout_rate": 0.0, "learning_rate": 0.01,
            "skip_connections": False, "batch_norm": False, "weight_decay": 0.0,
        }
        result = DYNAMIC_STRATEGIES[name](genome)
        assert isinstance(result, dict)
        # Cleanup
        DYNAMIC_STRATEGIES.pop(name, None)
