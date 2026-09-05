"""researchforge/benchmarks/execution — Benchmark Execution Protocol & Preflight Gates.

Phase 12B.2 Core Architecture:
- Manifest generator with two-tier transfer regimes (sequence vs effective)
- Bitwise deterministic task materialization and canonical content hashing
- 8-point preflight verification gate suite
- Cryptographic preflight attestation token
- Human authorization safety gate (machine verification != execution authorization)
- Append-only execution ledger and auditable artifact registry
"""
from __future__ import annotations

from .manifest import generate_execution_manifest
from .materialization import (
    BenchmarkTaskMaterialization,
    compute_canonical_task_content_hash,
    materialize_benchmark_task,
)
from .models import (
    ArtifactRegistryRecord,
    ExecutionManifest,
    ExecutionNotAuthorizedError,
    LedgerEntry,
    PreflightAttestation,
    PreflightCheckResult,
    PreflightGateError,
    PreflightValidationReport,
    RunManifestEntry,
    get_software_commit_info,
)
from .preflight import PreflightValidator
from .protocol import BenchmarkExecutionProtocol

__all__ = [
    "ArtifactRegistryRecord",
    "BenchmarkExecutionProtocol",
    "BenchmarkTaskMaterialization",
    "ExecutionManifest",
    "ExecutionNotAuthorizedError",
    "LedgerEntry",
    "PreflightAttestation",
    "PreflightCheckResult",
    "PreflightGateError",
    "PreflightValidationReport",
    "PreflightValidator",
    "RunManifestEntry",
    "compute_canonical_task_content_hash",
    "generate_execution_manifest",
    "get_software_commit_info",
    "materialize_benchmark_task",
]
