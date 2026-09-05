"""researchforge/artifacts/model.py — Canonical Artifact domain object.

RF-1.0.0-alpha.3 (Phase 8A): Identity-bearing research artifact contract.
Separates artifact metadata and cryptographic fingerprint from large raw byte contents.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..domain.base import DomainObject


@dataclass(frozen=True)
class Artifact(DomainObject):
    """Canonical metadata and cryptographic fingerprint for a stored research artifact."""
    artifact_type: str
    uri_or_path: str
    sha256: str
    size_bytes: int
    producer: str | None = None
    code_revision: str | None = None
    experiment_run_id: str | None = None
    provenance_id: str | None = None
    mime_type: str = "application/octet-stream"
    metadata: Dict[str, Any] | None = None
    created_at: float | None = None
    is_retired: bool = False

    @property
    def artifact_id(self) -> str:
        return self.id

    @property
    def uri(self) -> str:
        return self.uri_or_path

    @property
    def checksum_sha256(self) -> str:
        return self.sha256

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base["artifact_id"] = self.id
        base["uri"] = self.uri_or_path
        base["checksum_sha256"] = self.sha256
        base["is_retired"] = self.is_retired
        return base

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "Artifact":
        data = dict(obj)
        if "id" not in data and "artifact_id" in data:
            data["id"] = data.pop("artifact_id")
        else:
            data.pop("artifact_id", None)

        if "uri_or_path" not in data and "uri" in data:
            data["uri_or_path"] = data.pop("uri")
        else:
            data.pop("uri", None)

        if "sha256" not in data and "checksum_sha256" in data:
            data["sha256"] = data.pop("checksum_sha256")
        else:
            data.pop("checksum_sha256", None)

        data.pop("researchforge_version", None)
        return cls(**data)
