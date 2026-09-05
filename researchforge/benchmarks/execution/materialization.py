"""researchforge/benchmarks/execution/materialization.py — Deterministic Task Materializer.

Enforces:
1. Exact bitwise deterministic task dataset construction from TaskGenerationSpec.
2. Canonical content hashing over raw array byte buffers, shapes, dtypes, and splits.
3. Audit capability: independent double-materialization parity verification.
"""
from __future__ import annotations

import hashlib
from typing import Tuple
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ..cohort.models import TaskGenerationSpec
from ..tasks import Task, synthetic_ecg_task


def materialize_task(spec: TaskGenerationSpec) -> Task:
    """Materializes a Task object strictly conforming to the TaskGenerationSpec recipe."""
    seed = spec.dataset_seed

    if spec.task_id == "digits_0_4":
        data = load_digits()
        mask = data.target < 5
        X_sub, y_sub = data.data[mask], data.target[mask]
        X_train, X_rest, y_train, y_rest = train_test_split(
            X_sub, y_sub, train_size=spec.sample_count_train, random_state=seed, stratify=y_sub
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest
        )
        return Task(
            name=spec.task_id,
            description="5-class digit classification (0-4)",
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            metric_fn=accuracy_score,
            target_metric=spec.target_metric_threshold,
        )

    elif spec.task_id == "digits_5_9":
        data = load_digits()
        mask = data.target >= 5
        X_sub, y_sub = data.data[mask], data.target[mask] - 5
        X_train, X_rest, y_train, y_rest = train_test_split(
            X_sub, y_sub, train_size=spec.sample_count_train, random_state=seed, stratify=y_sub
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest
        )
        return Task(
            name=spec.task_id,
            description="5-class digit classification (5-9 shifted to 0-4)",
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            metric_fn=accuracy_score,
            target_metric=spec.target_metric_threshold,
        )

    elif spec.task_id == "digits_all_10":
        data = load_digits()
        X_all, y_all = data.data, data.target
        X_train, X_rest, y_train, y_rest = train_test_split(
            X_all, y_all, train_size=spec.sample_count_train, random_state=seed, stratify=y_all
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest
        )
        return Task(
            name=spec.task_id,
            description="Full 10-class handwritten digits classification",
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            metric_fn=accuracy_score,
            target_metric=spec.target_metric_threshold,
        )

    elif spec.task_id == "synthetic_ecg_lead1":
        t = synthetic_ecg_task(seed=seed, n_samples=700, noise=0.30)
        t.name = spec.task_id
        t.target_metric = spec.target_metric_threshold
        return t

    elif spec.task_id == "synthetic_ecg_lead2":
        rng = np.random.RandomState(seed + 99)
        t_space = np.linspace(0, 1, 96)

        def beat(centers, widths, heights, nz):
            sig = np.zeros_like(t_space)
            for c, w, h in zip(centers, widths, heights):
                sig = sig + h * np.exp(-((t_space - c) ** 2) / (2 * w ** 2))
            return sig + rng.normal(0, nz, size=t_space.shape)

        X, y = [], []
        n_samples = 700
        n_pos = n_samples // 2
        for _ in range(n_pos):
            X.append(beat(centers=[0.25, 0.55, 0.75], widths=[0.04, 0.045, 0.05], heights=[0.4, 1.1, 0.35], nz=0.35))
            y.append(0)
        for _ in range(n_samples - n_pos):
            jitter = rng.uniform(-0.03, 0.03)
            X.append(beat(centers=[0.25 + jitter, 0.55 + jitter, 0.75], widths=[0.05, 0.04, 0.06], heights=[0.5, 0.9, 0.4], nz=0.35))
            y.append(1)

        X_arr, y_arr = np.array(X), np.array(y)
        X_train, X_rest, y_train, y_rest = train_test_split(X_arr, y_arr, train_size=spec.sample_count_train, random_state=seed, stratify=y_arr)
        X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest)

        return Task(
            name=spec.task_id,
            description="Synthetic lead-2 beat classification with shifted morphology",
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            metric_fn=accuracy_score,
            target_metric=spec.target_metric_threshold,
        )

    elif spec.task_id == "xor_parity_8bit":
        rng = np.random.RandomState(seed + 777)
        n_samples = 600
        bits = rng.randint(0, 2, size=(n_samples, 8))
        y = (np.sum(bits, axis=1) % 2).astype(int)
        X = bits.astype(float) + rng.normal(0, 0.15, size=bits.shape)

        X_train, X_rest, y_train, y_rest = train_test_split(X, y, train_size=spec.sample_count_train, random_state=seed, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest)

        return Task(
            name=spec.task_id,
            description="8-bit noisy non-linear parity classification",
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            metric_fn=accuracy_score,
            target_metric=spec.target_metric_threshold,
        )

    raise ValueError(f"Unknown task specification ID '{spec.task_id}'")


def compute_canonical_task_content_hash(task: Task) -> str:
    """Computes a bitwise SHA-256 hash across array byte buffers, shapes, dtypes, labels, and target metric."""
    hasher = hashlib.sha256()
    hasher.update(task.name.encode("utf-8"))

    for arr in (task.X_train, task.y_train, task.X_val, task.y_val, task.X_test, task.y_test):
        arr_c = np.ascontiguousarray(arr)
        hasher.update(arr_c.dtype.str.encode("utf-8"))
        hasher.update(str(arr_c.shape).encode("utf-8"))
        hasher.update(arr_c.tobytes())

    hasher.update(f"{task.target_metric:.6f}".encode("utf-8"))
    return hasher.hexdigest()


def verify_deterministic_materialization(spec: TaskGenerationSpec) -> Tuple[bool, str, str]:
    """Materializes a task twice independently and verifies bitwise equality of the content hash."""
    t1 = materialize_task(spec)
    h1 = compute_canonical_task_content_hash(t1)

    t2 = materialize_task(spec)
    h2 = compute_canonical_task_content_hash(t2)

    return (h1 == h2, h1, h2)


from dataclasses import dataclass
from ..cohort.specification import create_canonical_cohort_spec


@dataclass(frozen=True)
class BenchmarkTaskMaterialization:
    task_id: str
    seed: int
    task: Task
    content_hash: str


def materialize_benchmark_task(task_id: str, seed: int = 0) -> BenchmarkTaskMaterialization:
    """Convenience helper to materialize a task from the canonical cohort specification."""
    import dataclasses
    cohort = create_canonical_cohort_spec()
    spec = cohort.tasks.get(task_id)
    if not spec:
        raise ValueError(f"Task '{task_id}' not found in cohort tasks.")
    # If a seed offset is needed, adjust the generation spec
    adjusted_spec = dataclasses.replace(spec, dataset_seed=spec.dataset_seed + seed)
    t = materialize_task(adjusted_spec)
    ch = compute_canonical_task_content_hash(t)
    return BenchmarkTaskMaterialization(
        task_id=task_id,
        seed=seed,
        task=t,
        content_hash=ch,
    )

