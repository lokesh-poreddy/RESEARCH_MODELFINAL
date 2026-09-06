"""Hugging Face provenance capture.

Every interaction with the HF API must produce a provenance record.
These records are evidence, not scientific truth.

Schema:
    provider:         "huggingface"
    model_id:         HF Hub model identifier
    model_revision:   Git SHA of the model revision (if available)
    dataset_id:       HF Hub dataset identifier (if applicable)
    dataset_revision: Git SHA of the dataset revision (if available)
    task:             Inference task type
    inference_provider: HF provider route (if available)
    request_timestamp: ISO-8601 UTC timestamp
    response_latency_s: Float seconds
    execution_mode:   "live" | "offline" | "mock"
    artifact_fingerprint: SHA-256 of response content (if applicable)
    config_fingerprint:   SHA-256 of config dict
    notes:            Free text (no credentials)
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _sha256_dict(d: Dict[str, Any]) -> str:
    """Deterministic SHA-256 of a JSON-serializable dict."""
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class HFProvenance:
    """Immutable provenance record for a single HF API interaction.

    Must NOT contain credential values, raw response bodies with PII,
    or any information not safe for inclusion in research artifacts.
    """
    provider: str = "huggingface"
    model_id: str = ""
    model_revision: str = ""
    dataset_id: str = ""
    dataset_revision: str = ""
    task: str = ""
    inference_provider: str = ""
    request_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    response_latency_s: float = 0.0
    execution_mode: str = "offline"   # "live" | "offline" | "mock"
    artifact_fingerprint: str = ""    # SHA-256 of response content
    config_fingerprint: str = ""      # SHA-256 of HFClientConfig.to_provenance_dict()
    success: bool = False
    error_code: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Canonical fingerprint of this provenance record."""
        return _sha256_dict(self.to_dict())

    @classmethod
    def build(
        cls,
        *,
        model_id: str,
        task: str,
        config_dict: Dict[str, Any],
        execution_mode: str = "offline",
        model_revision: str = "",
        dataset_id: str = "",
        dataset_revision: str = "",
        inference_provider: str = "",
        response_content: Optional[Any] = None,
        latency_s: float = 0.0,
        success: bool = False,
        error_code: str = "",
        notes: str = "",
    ) -> "HFProvenance":
        """Factory method with auto-computed fingerprints."""
        artifact_fp = ""
        if response_content is not None:
            try:
                content_str = json.dumps(response_content, sort_keys=True, ensure_ascii=True)
                artifact_fp = hashlib.sha256(content_str.encode("utf-8")).hexdigest()
            except (TypeError, ValueError):
                artifact_fp = hashlib.sha256(str(response_content).encode("utf-8")).hexdigest()

        config_fp = _sha256_dict(config_dict)

        return cls(
            provider="huggingface",
            model_id=model_id,
            model_revision=model_revision,
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            task=task,
            inference_provider=inference_provider,
            request_timestamp=datetime.now(timezone.utc).isoformat(),
            response_latency_s=latency_s,
            execution_mode=execution_mode,
            artifact_fingerprint=artifact_fp,
            config_fingerprint=config_fp,
            success=success,
            error_code=error_code,
            notes=notes,
        )
