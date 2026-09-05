"""Evidence- and Outcome-Conditioned Research Memory (ECRM): the write-manage-
read memory loop, scoring, and forgetting policy described in
ResearchForge-ECRM Sec. 1 and Sec. 4 of the technical report.

RF-1.0.0-alpha.2.1:
  - MemoryRecord: Class B versioned mutable memory envelope
  - ECRM: flat memory store with vector search adapter
  - TrajectoryMemory: capacity-aware contextual trajectory memory
"""
from .ecrm import ECRM
from .record import MemoryRecord, MEMORY_RECORD_SCHEMA
from .trajectory import TrajectoryMemory, TrajectoryRecord
from .adaptive_trajectory import (
    AdaptiveTrajectoryMemory,
    AdaptiveTrajectoryRecord,
    ContextualRetrievalResult,
    FailureCheckResult,
    new_adaptive_trajectory_id,
)

__all__ = [
    "ECRM",
    "MemoryRecord",
    "MEMORY_RECORD_SCHEMA",
    "TrajectoryMemory",
    "TrajectoryRecord",
    "AdaptiveTrajectoryMemory",
    "AdaptiveTrajectoryRecord",
    "ContextualRetrievalResult",
    "FailureCheckResult",
    "new_adaptive_trajectory_id",
]

