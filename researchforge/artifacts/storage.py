"""researchforge/artifacts/storage.py — Artifact byte storage abstraction and local implementation.

RF-1.0.0-alpha.3 (Phase 8A): Enforces separation between relational metadata and raw file bytes.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional, Protocol


class ArtifactIntegrityError(Exception):
    """Raised when an artifact checksum mismatch is detected or data is corrupted."""
    pass


class ArtifactNotFoundError(Exception):
    """Raised when an artifact cannot be located at the requested path or URI."""
    pass


class ArtifactStorage(Protocol):
    """Abstract protocol for byte storage backends (local filesystem, S3, MinIO)."""

    def store(self, artifact_id: str, content: bytes, extension: str = "bin") -> str:
        """Persist raw content bytes and return a canonical URI or absolute path."""
        ...

    def retrieve(self, uri_or_path: str, expected_sha256: Optional[str] = None) -> bytes:
        """Retrieve content bytes and verify cryptographic integrity if checksum is provided."""
        ...

    def exists(self, uri_or_path: str) -> bool:
        """Check if an artifact exists at the given URI or path."""
        ...

    def delete(self, uri_or_path: str) -> bool:
        """Remove an artifact from storage. Returns True if deleted, False if not found."""
        ...


class LocalArtifactStorage:
    """Local filesystem implementation of ArtifactStorage."""

    def __init__(self, base_dir: Path | str = "./artifacts") -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, uri_or_path: str) -> Path:
        if uri_or_path.startswith("file://"):
            p = Path(uri_or_path[7:])
        else:
            p = Path(uri_or_path)
        if not p.is_absolute():
            p = self.base_dir / p
        return p

    def store(self, artifact_id: str, content: bytes, extension: str = "bin") -> str:
        # Partition into 2-level directory tree using prefix to avoid massive flat folders
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in artifact_id)
        prefix = safe_id[:2] if len(safe_id) >= 2 else "00"
        subdir = self.base_dir / prefix
        subdir.mkdir(parents=True, exist_ok=True)

        ext = extension.lstrip(".")
        filename = f"{safe_id}.{ext}" if ext else safe_id
        target_path = subdir / filename

        with open(target_path, "wb") as f:
            f.write(content)

        return f"file://{target_path.as_posix()}"

    def retrieve(self, uri_or_path: str, expected_sha256: Optional[str] = None) -> bytes:
        target_path = self._resolve_path(uri_or_path)
        if not target_path.exists():
            raise ArtifactNotFoundError(f"Artifact not found at {target_path}")

        with open(target_path, "rb") as f:
            content = f.read()

        if expected_sha256 is not None:
            actual_sha256 = hashlib.sha256(content).hexdigest()
            if actual_sha256.lower() != expected_sha256.lower():
                raise ArtifactIntegrityError(
                    f"Artifact hash mismatch at {target_path}! "
                    f"Expected {expected_sha256}, got {actual_sha256}."
                )

        return content

    def exists(self, uri_or_path: str) -> bool:
        try:
            target_path = self._resolve_path(uri_or_path)
            return target_path.exists()
        except Exception:
            return False

    def delete(self, uri_or_path: str) -> bool:
        try:
            target_path = self._resolve_path(uri_or_path)
            if target_path.exists():
                target_path.unlink()
                return True
            return False
        except Exception:
            return False
