from __future__ import annotations

import dataclasses
import json
import hashlib
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Type, TypeVar

T = TypeVar("T", bound="DomainObject")


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _as_primitive(obj: Any) -> Any:
    # Convert nested DomainObjects and enums to primitives for deterministic
    # serialization.
    if isinstance(obj, DomainObject):
        return obj.to_dict()
    if dataclasses.is_dataclass(obj):
        return {k: _as_primitive(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_as_primitive(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _as_primitive(v) for k, v in obj.items()}
    return obj


@dataclass(frozen=True)
class DomainObject:
    """Base class for canonical domain contracts.

    Provides:
    - `schema_version`: contract version
    - deterministic `to_dict` / `from_dict`
    - `fingerprint()` using SHA-256 of canonical JSON
    """

    id: str
    schema_version: str

    def to_dict(self) -> Dict[str, Any]:
        raw = dataclasses.asdict(self)
        return _as_primitive(raw)

    @classmethod
    def from_dict(cls: Type[T], obj: Dict[str, Any]) -> T:
        # Default naive constructor: subclasses may override for nested parsing
        return cls(**obj)  # type: ignore

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    def fingerprint(self) -> str:
        # Content-addressed fingerprint using schema_version and canonical JSON
        payload = {"schema_version": self.schema_version, "content": self.to_dict()}
        j = _canonical_json(payload)
        return hashlib.sha256(j.encode("utf-8")).hexdigest()


from pathlib import Path
from typing import Sequence, Optional


def compute_canonical_fingerprint(
    payload: Dict[str, Any],
    excluded_fields: Optional[Sequence[str]] = None,
) -> str:
    r"""Computes a deterministic SHA-256 fingerprint over canonical JSON.

    CRITICAL INVARIANT:
    Excludes self-referential fingerprint fields and explicitly passed fields to prevent
    hash circularity: H = SHA256(payload \ {self_fingerprint}).
    """
    default_excluded = frozenset(["fingerprint"])
    excl = default_excluded.union(excluded_fields or [])
    cleaned = {k: v for k, v in payload.items() if k not in excl}
    canonical_str = _canonical_json(_as_primitive(cleaned))
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


def compute_file_sha256(file_path: Path | str) -> str:
    """Computes bitwise SHA-256 over exact physical file bytes on disk."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found for SHA-256 calculation: {p}")
    return hashlib.sha256(p.read_bytes()).hexdigest()

