"""researchforge/benchmarks/continuity/tasks.py — Multi-regime task suites.

Phase 11 Task Regimes:
1. SAME_FAMILY: Digit recognition subsets with shared visual pixel features.
2. CROSS_FAMILY: Continuous waveform/time-series classification with differing morphology.
3. UNRELATED: Discrete tabular parity problem completely orthogonal to pixel or waveform statistics.

All tasks are deterministic, require zero network access, and use synthetic or bundled data.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ..tasks import Task, synthetic_ecg_task
from .models import TaskRegime


@dataclass(frozen=True)
class ContinuityTask:
    """A research task contextualized within a continuity regime."""
    task_id: str
    name: str
    regime: TaskRegime
    family: str
    task: Task
    description: str

    def fingerprint(self) -> str:
        """Content-addressed fingerprint of the task data and configuration."""
        raw = (
            f"{self.task_id}|{self.regime.value}|{self.family}|"
            f"{self.task.X_train.shape}|{float(np.sum(self.task.X_train)):.4f}|"
            f"{self.task.y_train.shape}|{float(np.sum(self.task.y_train)):.4f}|"
            f"{self.task.target_metric}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "regime": self.regime.value,
            "family": self.family,
            "description": self.description,
            "fingerprint": self.fingerprint(),
            "target_metric": self.task.target_metric,
            "train_samples": len(self.task.X_train),
            "val_samples": len(self.task.X_val),
            "test_samples": len(self.task.X_test),
        }


# ── 1. Same-Family Tasks (Digit Subsets) ──────────────────────────────────────

def digits_0_4_task(seed: int = 0, n_train: int = 40) -> ContinuityTask:
    """Handwritten digits subset for classes 0, 1, 2, 3, 4."""
    from sklearn.datasets import load_digits
    data = load_digits()
    mask = data.target < 5
    X_sub, y_sub = data.data[mask], data.target[mask]

    X_train, X_rest, y_train, y_rest = train_test_split(
        X_sub, y_sub, train_size=n_train, random_state=seed, stratify=y_sub
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest
    )

    t = Task(
        name="digits_0_4",
        description=f"5-class digit classification (0-4) with {n_train} training samples",
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        metric_fn=accuracy_score,
        target_metric=0.90,
    )
    return ContinuityTask(
        task_id="T_digits_0_4",
        name="Digits (0-4)",
        regime=TaskRegime.SAME_FAMILY,
        family="digits_vision",
        task=t,
        description="Low-digit recognition task sharing 8x8 pixel grayscale distribution",
    )


def digits_5_9_task(seed: int = 0, n_train: int = 40) -> ContinuityTask:
    """Handwritten digits subset for classes 5, 6, 7, 8, 9."""
    from sklearn.datasets import load_digits
    data = load_digits()
    mask = data.target >= 5
    X_sub, y_sub = data.data[mask], data.target[mask] - 5

    X_train, X_rest, y_train, y_rest = train_test_split(
        X_sub, y_sub, train_size=n_train, random_state=seed, stratify=y_sub
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest
    )

    t = Task(
        name="digits_5_9",
        description=f"5-class digit classification (5-9) with {n_train} training samples",
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        metric_fn=accuracy_score,
        target_metric=0.90,
    )
    return ContinuityTask(
        task_id="T_digits_5_9",
        name="Digits (5-9)",
        regime=TaskRegime.SAME_FAMILY,
        family="digits_vision",
        task=t,
        description="High-digit recognition task sharing 8x8 pixel grayscale distribution",
    )


def digits_all_task(seed: int = 0, n_train: int = 60) -> ContinuityTask:
    """Standard 10-class handwritten digits classification."""
    from ..tasks import digits_task
    t = digits_task(seed=seed, n_train=n_train)
    return ContinuityTask(
        task_id="T_digits_all",
        name="Digits (0-9 Full)",
        regime=TaskRegime.SAME_FAMILY,
        family="digits_vision",
        task=t,
        description="Full 10-class digit recognition task",
    )


# ── 2. Cross-Family Tasks (Continuous Signals / Biomedical Surrogates) ─────────

def synthetic_ecg_lead1_task(seed: int = 0) -> ContinuityTask:
    """Binary synthetic ECG classification (lead 1 surrogate)."""
    t = synthetic_ecg_task(seed=seed, n_samples=700, noise=0.30)
    return ContinuityTask(
        task_id="T_ecg_lead1",
        name="Synthetic ECG Lead-1",
        regime=TaskRegime.CROSS_FAMILY,
        family="signal_processing",
        task=t,
        description="Single-lead Gaussian bump ECG surrogate with standard noise",
    )


def synthetic_ecg_lead2_task(seed: int = 0) -> ContinuityTask:
    """Binary synthetic ECG classification with altered waveform morphology (lead 2 surrogate)."""
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

    X, y = np.array(X), np.array(y)
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, train_size=80, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest)

    t = Task(
        name="synthetic_ecg_lead2",
        description="Synthetic lead-2 beat classification with shifted morphology",
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        metric_fn=accuracy_score,
        target_metric=0.78,
    )
    return ContinuityTask(
        task_id="T_ecg_lead2",
        name="Synthetic ECG Lead-2",
        regime=TaskRegime.CROSS_FAMILY,
        family="signal_processing",
        task=t,
        description="Shifted morphology ECG surrogate representing cross-lead transfer",
    )


# ── 3. Unrelated / Mismatched Tasks (Tabular Parity) ───────────────────────────

def tabular_xor_parity_task(seed: int = 0) -> ContinuityTask:
    """Synthetic 4-bit parity classification (non-linear discrete logic)."""
    rng = np.random.RandomState(seed + 777)
    n_samples = 600
    # Generate 4-bit binary inputs with mild Gaussian noise
    bits = rng.randint(0, 2, size=(n_samples, 4))
    # Parity: 1 if odd sum, 0 if even sum
    y = (np.sum(bits, axis=1) % 2).astype(int)
    # Add continuous noise so models cannot simply do exact table lookups
    X = bits.astype(float) + rng.normal(0, 0.15, size=bits.shape)

    X_train, X_rest, y_train, y_rest = train_test_split(X, y, train_size=60, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest)

    t = Task(
        name="tabular_xor_parity",
        description="4-bit non-linear parity classification over noisy binary inputs",
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        metric_fn=accuracy_score,
        target_metric=0.85,
    )
    return ContinuityTask(
        task_id="T_xor_parity",
        name="Tabular XOR Parity",
        regime=TaskRegime.UNRELATED,
        family="discrete_logic",
        task=t,
        description="Mismatched non-linear discrete parity task orthogonal to vision and waveforms",
    )


# ── 4. Transfer Matrix Specific Tasks ──────────────────────────────────────────

def digits_shifted_task(seed: int = 0, n_train: int = 60) -> ContinuityTask:
    """High-overlap transfer candidate: digits with intensity/spatial perturbation."""
    from sklearn.datasets import load_digits
    import scipy.ndimage
    data = load_digits()
    X_orig, y = data.data, data.target
    
    rng = np.random.RandomState(seed + 101)
    X_shifted = np.zeros_like(X_orig)
    
    # Apply intensity variation, Gaussian noise, and mild spatial shift
    for i in range(len(X_orig)):
        img = X_orig[i].reshape(8, 8)
        
        # 1. Spatial displacement (mild)
        shift_r, shift_c = rng.uniform(-0.8, 0.8, size=2)
        img_shifted = scipy.ndimage.shift(img, [shift_r, shift_c], mode='nearest')
        
        # 2. Intensity variation
        scale = rng.uniform(0.7, 1.2)
        img_shifted = img_shifted * scale
        
        # 3. Gaussian perturbation
        img_shifted += rng.normal(0, 1.5, size=(8, 8))
        
        X_shifted[i] = np.clip(img_shifted.flatten(), 0, 16)

    X_train, X_rest, y_train, y_rest = train_test_split(
        X_shifted, y, train_size=n_train, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest
    )

    t = Task(
        name="digits_shifted",
        description=f"10-class digit classification with spatial/intensity perturbations ({n_train} training samples)",
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        metric_fn=accuracy_score,
        target_metric=0.85,
    )
    return ContinuityTask(
        task_id="T_digits_shifted",
        name="Digits (Shifted)",
        regime=TaskRegime.SAME_FAMILY,
        family="digits_vision_shifted",
        task=t,
        description="Perturbed load_digits for high-overlap transfer testing",
    )


def capacity_trap_task(seed: int = 0) -> ContinuityTask:
    """Negative-transfer candidate: designed to severely punish over-capacity."""
    from sklearn.datasets import make_classification
    
    # High-dimensional, low-sample, noisy, redundant
    X, y = make_classification(
        n_samples=600,
        n_features=96,
        n_informative=5,
        n_redundant=40,
        n_repeated=5,
        n_classes=2,
        flip_y=0.15,          # 15% label noise
        class_sep=0.5,        # highly overlapping classes
        random_state=seed + 202
    )
    
    # Very small training set (60 samples) vs high features (96)
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, train_size=60, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest)

    t = Task(
        name="capacity_trap",
        description="Noisy tabular dataset where adding model capacity degrades validation",
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        metric_fn=accuracy_score,
        target_metric=0.70,
    )
    return ContinuityTask(
        task_id="T_capacity_trap",
        name="Capacity Trap",
        regime=TaskRegime.UNRELATED,
        family="noisy_tabular",
        task=t,
        description="Controlled negative-transfer candidate",
    )


# ── Task Suites & Orderings ───────────────────────────────────────────────────

def get_pilot_task_sequence(seed: int = 0) -> List[ContinuityTask]:
    """Authoritative 3-task sequence for Phase 11 methodology pilot."""
    return [
        digits_0_4_task(seed=seed),          # T1: SAME_FAMILY (Digits 0-4)
        digits_5_9_task(seed=seed),          # T2: SAME_FAMILY (Digits 5-9)
        synthetic_ecg_lead1_task(seed=seed), # T3: CROSS_FAMILY (ECG Lead 1)
    ]


def get_full_task_sequence(seed: int = 0) -> List[ContinuityTask]:
    """Full 6-task sequence spanning all 3 regimes for subsequent scaling."""
    return [
        digits_0_4_task(seed=seed),
        digits_5_9_task(seed=seed),
        digits_all_task(seed=seed),
        synthetic_ecg_lead1_task(seed=seed),
        synthetic_ecg_lead2_task(seed=seed),
        tabular_xor_parity_task(seed=seed),
        digits_shifted_task(seed=seed),
        capacity_trap_task(seed=seed),
    ]


def get_ordered_tasks(tasks: List[ContinuityTask], ordering: str) -> List[ContinuityTask]:
    """Returns task sequence according to the specified permutation."""
    if ordering == "forward":
        return list(tasks)
    elif ordering == "reverse":
        return list(reversed(tasks))
    elif ordering == "cross_first":
        # Put the cross-family / unrelated task first to test reverse transfer
        if len(tasks) == 3:
            return [tasks[2], tasks[0], tasks[1]]
        elif len(tasks) >= 4:
            return [tasks[3], tasks[0], tasks[1], tasks[2]] + list(tasks[4:])
        return list(reversed(tasks))
    else:
        raise ValueError(f"Unknown ordering '{ordering}'. Supported: 'forward', 'reverse', 'cross_first'.")
