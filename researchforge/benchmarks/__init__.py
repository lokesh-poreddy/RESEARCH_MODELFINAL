"""RDE-Bench: task definitions and the benchmark harness that computes
Research Efficiency, Search Efficiency, Failure Repetition Rate, Memory
Utility, Negative Transfer Rate, and Memory Half-life."""

from .tasks import Task, digits_task, synthetic_ecg_task
from .chemreact import chemreact_defect_detection_task, chemreact_titration_task

__all__ = [
    "Task",
    "digits_task",
    "synthetic_ecg_task",
    "chemreact_defect_detection_task",
    "chemreact_titration_task",
]
