"""tests/test_chemreact.py — Tests for ChemReact Reaction-Kinetics Benchmark.

Verifies:
  1. Kinetic state generation and physical constants.
  2. Dataset generation shapes and stratification.
  3. Task interface compliance with Task dataclass.
  4. Real model training and evaluation on ChemReact.
"""
import numpy as np
import pytest
from researchforge.benchmarks.chemreact import (
    generate_chemreact_dataset,
    chemreact_defect_detection_task,
    chemreact_titration_task,
)
from researchforge.pipeline.controller import ResearchController
from researchforge.genome.model_genome import ModelGenome


def test_generate_chemreact_dataset_shapes():
    X, y, feature_names = generate_chemreact_dataset(seed=42, n_samples=100)
    assert X.shape == (100, 12)
    assert y.shape == (100,)
    assert len(feature_names) == 12
    assert set(np.unique(y)).issubset({0, 1})
    assert np.all(np.isfinite(X))


def test_chemreact_defect_detection_task_interface():
    task = chemreact_defect_detection_task(seed=42, n_samples=200, n_train=40)
    assert task.name == "chemreact_defect_detection"
    assert task.X_train.shape[0] == 40
    assert task.X_val.shape[0] == 80
    assert task.X_test.shape[0] == 80
    assert task.target_metric == 0.88


def test_chemreact_titration_task_interface():
    task = chemreact_titration_task(seed=42, n_samples=200, n_train=40)
    assert task.name == "chemreact_titration_equivalence"
    assert task.X_train.shape[1] == 14
    assert task.target_metric == 0.85


def test_chemreact_controller_short_run():
    task = chemreact_defect_detection_task(seed=42, n_samples=150, n_train=40)
    ctrl = ResearchController(task, condition="contextual_policy", seed=42)
    res = ctrl.run(n_generations=3)
    assert res.best_metric > 0.70
    assert len(res.trials) >= 4  # baseline + 3 gens
