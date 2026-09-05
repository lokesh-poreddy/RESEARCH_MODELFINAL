"""researchforge/benchmarks/continuity — Research Continuity Benchmark package.

Phase 11: Testing the core hypothesis:
"Does accumulated research experience improve ResearchForge's later research decisions on subsequent problems?"
"""
from .evaluator import ContinuityEvaluator
from .harness import ContinuityHarness
from .models import (
    ContinuityArtifact,
    ContinuityCondition,
    ContinuityObservation,
    ExperiencePartition,
    FutureInformationLeakageError,
    TaskRegime,
    TransferClassification,
    TransferEvent,
)
from .tasks import (
    ContinuityTask,
    digits_0_4_task,
    digits_5_9_task,
    digits_all_task,
    get_full_task_sequence,
    get_ordered_tasks,
    get_pilot_task_sequence,
    synthetic_ecg_lead1_task,
    synthetic_ecg_lead2_task,
    tabular_xor_parity_task,
)

__all__ = [
    "ContinuityEvaluator",
    "ContinuityHarness",
    "ContinuityArtifact",
    "ContinuityCondition",
    "ContinuityObservation",
    "ExperiencePartition",
    "FutureInformationLeakageError",
    "TaskRegime",
    "TransferClassification",
    "TransferEvent",
    "ContinuityTask",
    "digits_0_4_task",
    "digits_5_9_task",
    "digits_all_task",
    "synthetic_ecg_lead1_task",
    "synthetic_ecg_lead2_task",
    "tabular_xor_parity_task",
    "get_ordered_tasks",
    "get_pilot_task_sequence",
    "get_full_task_sequence",
]
