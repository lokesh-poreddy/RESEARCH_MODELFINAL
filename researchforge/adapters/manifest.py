"""researchforge/adapters/manifest.py — Canonical Adapter Manifest contract.

Phase 10 Invariants:
1. Hard distinction between implementation statuses:
   - IMPLEMENTED: Fully wired and functional in repository
   - PARTIAL: Partially functional or hybrid
   - DECLARATIVE: Contract/schema defined for format exchange without execution runtime
   - STUB: Interface mock/stub placeholder
   - UNAVAILABLE: Declared but dependencies/runtime currently missing
2. Never claim an adapter is implemented merely because a declaration or schema exists.
3. Every declaration exposes capabilities, interface versions, schemas, provenance,
   limitations, and configuration fingerprint.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..domain.base import DomainObject, _canonical_json


class ImplementationStatus(str, Enum):
    """Factual, unembellished implementation status of an adapter."""
    IMPLEMENTED = "IMPLEMENTED"
    PARTIAL = "PARTIAL"
    DECLARATIVE = "DECLARATIVE"
    STUB = "STUB"
    UNAVAILABLE = "UNAVAILABLE"


class AdapterCapability(str, Enum):
    """Core capabilities exposed through adapters."""
    RESEARCH_STATE = "RESEARCH_STATE"
    EXPERIMENTS = "EXPERIMENTS"
    EVIDENCE = "EVIDENCE"
    MEMORY = "MEMORY"
    VRDEG = "VRDEG"
    POLICY = "POLICY"
    PORTFOLIO = "PORTFOLIO"
    SATURATION = "SATURATION"
    PROVENANCE = "PROVENANCE"
    ARTIFACTS = "ARTIFACTS"
    BENCHMARKS = "BENCHMARKS"


@dataclass(frozen=True)
class AdapterDeclaration(DomainObject):
    """Explicit, inspectable capability declaration for an external or native adapter."""
    adapter_id: str
    name: str
    capability: AdapterCapability
    implementation_status: ImplementationStatus
    interface_version: str
    supported_operations: List[str]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    provenance_behavior: str
    reproducibility_characteristics: str
    external_dependencies: List[str]
    limitations: List[str]
    configuration_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.configuration_fingerprint:
            payload = {
                "adapter_id": self.adapter_id,
                "interface_version": self.interface_version,
                "capability": self.capability.value,
                "implementation_status": self.implementation_status.value,
                "supported_operations": sorted(self.supported_operations),
            }
            fp = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
            object.__setattr__(self, "configuration_fingerprint", fp)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AdapterDeclaration":
        data = dict(d)
        if "capability" in data and isinstance(data["capability"], str):
            data["capability"] = AdapterCapability(data["capability"])
        if "implementation_status" in data and isinstance(data["implementation_status"], str):
            data["implementation_status"] = ImplementationStatus(data["implementation_status"])
        return cls(**data)


@dataclass(frozen=True)
class CanonicalAdapterManifest(DomainObject):
    """The authoritative manifest of all declared and implemented adapters in ResearchForge."""
    manifest_version: str
    adapters: Dict[str, AdapterDeclaration]
    created_at: str = "2026-09-05T12:00:00Z"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_adapter(self, adapter_id: str) -> Optional[AdapterDeclaration]:
        return self.adapters.get(adapter_id)

    def list_adapters(
        self,
        status: Optional[ImplementationStatus] = None,
        capability: Optional[AdapterCapability] = None,
    ) -> List[AdapterDeclaration]:
        res = list(self.adapters.values())
        if status is not None:
            res = [a for a in res if a.implementation_status == status]
        if capability is not None:
            res = [a for a in res if a.capability == capability]
        return res

    @classmethod
    def default_manifest(cls) -> "CanonicalAdapterManifest":
        """Factory producing the factual, verified adapter manifest for RF-1.0.0-alpha.3."""
        adapters = {
            # 1. Python Native Adapter: Fully implemented internal execution & reflection
            "python_native": AdapterDeclaration(
                id="decl_python_native",
                schema_version="1.0",
                adapter_id="python_native",
                name="Python Native Substrate Adapter",
                capability=AdapterCapability.EXPERIMENTS,
                implementation_status=ImplementationStatus.IMPLEMENTED,
                interface_version="1.0.0",
                supported_operations=["execute_genome", "mutate_genome", "evaluate_metric", "diagnose_failure"],
                input_schema={"type": "object", "properties": {"genome_id": {"type": "string"}, "task_id": {"type": "string"}}},
                output_schema={"type": "object", "properties": {"metric": {"type": "number"}, "status": {"type": "string"}}},
                provenance_behavior="Captures execution PID, resource usage, execution spec ID, and run ID",
                reproducibility_characteristics="Deterministic given identical random seeds and platform runtime",
                external_dependencies=["scikit-learn", "numpy"],
                limitations=["In-process or local child process execution only"],
            ),

            # 2. MATLAB Bridge Adapter: Stub / file-exchange bridge only; not pretending to be full MATLAB runtime
            "matlab_bridge": AdapterDeclaration(
                id="decl_matlab_bridge",
                schema_version="1.0",
                adapter_id="matlab_bridge",
                name="MATLAB M-File / MAT Exchange Adapter",
                capability=AdapterCapability.EXPERIMENTS,
                implementation_status=ImplementationStatus.STUB,
                interface_version="0.1.0",
                supported_operations=["export_mfile", "parse_mat_struct"],
                input_schema={"type": "object", "properties": {"m_script": {"type": "string"}}},
                output_schema={"type": "object", "properties": {"output_vars": {"type": "object"}}},
                provenance_behavior="Records source m-file path, command string, and environment hash",
                reproducibility_characteristics="Subject to external MATLAB engine version and licensing",
                external_dependencies=["matlab_runtime (optional)"],
                limitations=["Stub placeholder; requires local MATLAB engine or octave CLI which is not bundled"],
            ),

            # 3. Literature Retrieval Adapter: Wired to OpenAlex, Semantic Scholar, arXiv in researchforge/retrieval/
            "literature_retrieval": AdapterDeclaration(
                id="decl_literature_retrieval",
                schema_version="1.0",
                adapter_id="literature_retrieval",
                name="Multi-Source Scholarly Retrieval Adapter",
                capability=AdapterCapability.EVIDENCE,
                implementation_status=ImplementationStatus.IMPLEMENTED,
                interface_version="1.0.0",
                supported_operations=["search_openalex", "search_semantic_scholar", "search_arxiv", "fetch_metadata"],
                input_schema={"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}}},
                output_schema={"type": "object", "properties": {"evidence_items": {"type": "array"}}},
                provenance_behavior="Attaches raw source URL, corpus snapshot timestamp, and content SHA-256 hash",
                reproducibility_characteristics="Deterministic for cached responses; remote endpoints subject to API drift",
                external_dependencies=["urllib", "requests/http"],
                limitations=["Requires active network access when query is not present in local cache"],
            ),

            # 4. External Runtime Sandbox: SafeRunner process isolation and resource governance
            "runtime_sandbox": AdapterDeclaration(
                id="decl_runtime_sandbox",
                schema_version="1.0",
                adapter_id="runtime_sandbox",
                name="SafeRunner Resource-Guarded Execution Sandbox",
                capability=AdapterCapability.EXPERIMENTS,
                implementation_status=ImplementationStatus.IMPLEMENTED,
                interface_version="1.0.0",
                supported_operations=["run_isolated", "enforce_timeout", "track_budget", "terminate_runaway"],
                input_schema={"type": "object", "properties": {"target_callable": {"type": "string"}, "timeout_s": {"type": "number"}}},
                output_schema={"type": "object", "properties": {"exit_code": {"type": "integer"}, "wall_time_s": {"type": "number"}, "peak_memory_mb": {"type": "number"}}},
                provenance_behavior="Records child process PID, OS signal status, and wall-clock duration",
                reproducibility_characteristics="Process-isolated with hard signal timeouts",
                external_dependencies=["multiprocessing", "resource"],
                limitations=["Linux/macOS POSIX signal enforcement; limited memory limit primitives on non-cgroup hosts"],
            ),

            # 5. Tensor & Data Framework Adapter: Declarative exchange contract
            "tensor_data_exchange": AdapterDeclaration(
                id="decl_tensor_data_exchange",
                schema_version="1.0",
                adapter_id="tensor_data_exchange",
                name="Tensor & Dataset Interchange Adapter",
                capability=AdapterCapability.BENCHMARKS,
                implementation_status=ImplementationStatus.DECLARATIVE,
                interface_version="0.2.0",
                supported_operations=["serialize_tensor_manifest", "validate_array_schema"],
                input_schema={"type": "object", "properties": {"tensor_format": {"type": "string", "enum": ["safetensors", "numpy_npz", "arrow"]}}},
                output_schema={"type": "object", "properties": {"manifest_hash": {"type": "string"}}},
                provenance_behavior="Content-addresses tensor metadata and column schemas without loading raw buffers",
                reproducibility_characteristics="Bitwise deterministic schema specification",
                external_dependencies=["numpy (optional)", "pyarrow (optional)"],
                limitations=["Declarative schema specification only; zero-copy tensor runtime planned for Phase 12"],
            ),
        }

        return cls(
            id="manifest_canonical_v1",
            schema_version="1.0",
            manifest_version="1.0.0",
            adapters=adapters,
            created_at="2026-09-05T12:00:00Z",
            metadata={"environment": "ResearchForge-ECRM RF-1.0.0-alpha.3"},
        )
