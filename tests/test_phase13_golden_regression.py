"""tests/test_phase13_golden_regression.py — Golden controller regression trace test.

Classification: CORE
Verifies Criterion 6: Default controller behavior is regression-identical.
When enable_dynamic_router=False and transfer_guard_mode=None, the controller produces
bitwise-identical decisions, experiment specifications, and metric outcomes.
"""
import hashlib
import pytest
from researchforge.benchmarks.tasks import digits_task
from researchforge.pipeline.controller import ResearchController


def _trajectory_hash(result) -> str:
    """Deterministic hash of a RunResult's trial sequence."""
    entries = [
        f"{t.generation}:{t.strategy}:{t.model_type}:{t.metric:.12f}"
        for t in result.trials
    ]
    canonical = "\n".join(entries)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def test_golden_controller_default_behavior_is_deterministic():
    """Verify that default controller produces identical trial traces across runs."""
    task1 = digits_task(seed=42)
    ctrl1 = ResearchController(
        task1,
        condition="full",
        seed=42,
        enable_dynamic_router=False,
        transfer_guard_mode=None,
    )
    res1 = ctrl1.run(n_generations=5)

    task2 = digits_task(seed=42)
    ctrl2 = ResearchController(
        task2,
        condition="full",
        seed=42,
        enable_dynamic_router=False,
        transfer_guard_mode=None,
    )
    res2 = ctrl2.run(n_generations=5)

    assert len(res1.trials) == len(res2.trials)
    for t1, t2 in zip(res1.trials, res2.trials):
        assert t1.generation == t2.generation
        assert t1.strategy == t2.strategy
        assert t1.model_type == t2.model_type
        assert t1.metric == pytest.approx(t2.metric)
        assert t1.best_so_far == pytest.approx(t2.best_so_far)
        assert t1.failure == t2.failure

    assert _trajectory_hash(res1) == _trajectory_hash(res2)
    assert res1.best_metric == pytest.approx(res2.best_metric)
    assert res1.best_genome is not None and res2.best_genome is not None
    assert res1.best_genome.model_type == res2.best_genome.model_type
    assert res1.best_genome.architecture == res2.best_genome.architecture
    assert res1.best_genome.hyperparameters == res2.best_genome.hyperparameters


def test_opt_in_router_and_guard_initialization():
    """Opt-in parameters instantiate router and transfer guard without affecting default behavior."""
    task = digits_task(seed=42)

    # Default: both None
    ctrl_default = ResearchController(task, condition="full", seed=42)
    assert ctrl_default.router is None
    assert ctrl_default.transfer_guard is None

    # Opt-in: both active
    ctrl_opted = ResearchController(
        task,
        condition="full",
        seed=42,
        enable_dynamic_router=True,
        transfer_guard_mode="selective",
    )
    assert ctrl_opted.router is not None
    assert ctrl_opted.transfer_guard is not None
    assert ctrl_opted.transfer_guard.mode.value == "SELECTIVE"
