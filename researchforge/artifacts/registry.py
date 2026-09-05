"""researchforge/artifacts/registry.py — High-level ArtifactRegistry service.

RF-1.0.0-alpha.3 (Phase 8A): Coordinates metadata persistence and byte-level storage.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, Optional

from .model import Artifact
from .storage import ArtifactIntegrityError, ArtifactNotFoundError, ArtifactStorage, LocalArtifactStorage


class ArtifactRegistry:
    """Coordinates artifact byte persistence and metadata registration."""

    def __init__(
        self,
        storage: Optional[ArtifactStorage] = None,
        repository: Optional[Any] = None,
    ) -> None:
        self.storage = storage or LocalArtifactStorage()
        self.repository = repository

    def register_artifact(
        self,
        content: bytes,
        artifact_id: str,
        artifact_type: str,
        producer: Optional[str] = None,
        code_revision: Optional[str] = None,
        experiment_run_id: Optional[str] = None,
        provenance_id: Optional[str] = None,
        mime_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, Any]] = None,
        extension: str = "bin",
    ) -> Artifact:
        """Store bytes in artifact storage, compute cryptographic checksum, and create Artifact record."""
        sha256 = hashlib.sha256(content).hexdigest()
        size_bytes = len(content)

        uri = self.storage.store(artifact_id=artifact_id, content=content, extension=extension)

        artifact = Artifact(
            id=artifact_id,
            schema_version="1.0",
            artifact_type=artifact_type,
            uri_or_path=uri,
            sha256=sha256,
            size_bytes=size_bytes,
            producer=producer,
            code_revision=code_revision,
            experiment_run_id=experiment_run_id,
            provenance_id=provenance_id,
            mime_type=mime_type,
            metadata=metadata or {},
            created_at=time.time(),
        )

        if self.repository is not None:
            self.repository.save(artifact)

        return artifact

    def load_artifact_bytes(self, artifact_or_id: Artifact | str) -> bytes:
        """Retrieve artifact raw bytes and verify cryptographic integrity."""
        if isinstance(artifact_or_id, Artifact):
            artifact = artifact_or_id
        elif self.repository is not None:
            artifact = self.repository.get(artifact_or_id)
            if artifact is None:
                raise ArtifactNotFoundError(f"Artifact metadata with id '{artifact_or_id}' not found.")
        else:
            raise ValueError("Cannot resolve artifact ID without an active repository.")

        return self.storage.retrieve(
            uri_or_path=artifact.uri_or_path,
            expected_sha256=artifact.sha256,
        )

    def retire_artifact(self, artifact_or_id: Artifact | str, reason: str = "deprecated") -> Artifact:
        """Mark an artifact as retired in metadata while preserving underlying binary storage."""
        if isinstance(artifact_or_id, Artifact):
            artifact = artifact_or_id
        elif self.repository is not None:
            artifact = self.repository.get(artifact_or_id)
            if artifact is None:
                raise ArtifactNotFoundError(f"Artifact metadata with id '{artifact_or_id}' not found.")
        else:
            raise ValueError("Cannot resolve artifact ID without an active repository.")

        import dataclasses
        meta = dict(artifact.metadata or {})
        meta["retired_reason"] = reason
        meta["retired_at"] = time.time()
        updated = dataclasses.replace(artifact, is_retired=True, metadata=meta)
        if self.repository is not None:
            self.repository.save(updated)
        return updated
