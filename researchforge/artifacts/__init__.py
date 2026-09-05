"""researchforge/artifacts package — Production artifact registry and storage.

RF-1.0.0-alpha.3 (Phase 8A): Enforces separation of relational metadata and raw file bytes.
"""
from .model import Artifact
from .storage import (
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ArtifactStorage,
    LocalArtifactStorage,
)
from .registry import ArtifactRegistry

__all__ = [
    "Artifact",
    "ArtifactStorage",
    "LocalArtifactStorage",
    "ArtifactRegistry",
    "ArtifactIntegrityError",
    "ArtifactNotFoundError",
]
