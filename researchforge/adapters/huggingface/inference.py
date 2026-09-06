"""Hugging Face inference provider.

Connects to Hugging Face Inference API for:
- Text generation / chat completion
- Feature extraction (embeddings)
- Other supported inference tasks

Provider mode selection:
    offline: HF_TOKEN not present → raises HFTokenMissingError when called live
    live:    HF_TOKEN present → uses HF Inference API

Architecture position:
    LLMSynthesizer (future wiring)
        → HuggingFaceInferenceProvider.generate()
        → HFInferenceClient
        → Hugging Face Inference API

All results carry HFProvenance records.
No result should bypass ResearchPolicy or ECRM.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

from .client import HFClientConfig, HFInferenceClient
from .errors import (
    HFProviderError,
    HFTokenMissingError,
    HFInvalidResponseError,
    HFTaskUnsupportedError,
)
from .provenance import HFProvenance


# Supported inference tasks for validation
SUPPORTED_TASKS = {
    "text-generation",
    "text2text-generation",
    "conversational",
    "feature-extraction",
    "text-classification",
    "token-classification",
    "question-answering",
    "summarization",
    "translation",
    "fill-mask",
    "zero-shot-classification",
}


class InferenceResult:
    """Typed result from a Hugging Face inference call."""

    def __init__(
        self,
        text: str,
        raw: Any,
        model_id: str,
        task: str,
        latency_s: float,
        provenance: HFProvenance,
    ) -> None:
        self.text = text
        self.raw = raw
        self.model_id = model_id
        self.task = task
        self.latency_s = latency_s
        self.provenance = provenance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "model_id": self.model_id,
            "task": self.task,
            "latency_s": self.latency_s,
            "provenance": self.provenance.to_dict(),
        }


class HuggingFaceInferenceProvider:
    """Provider for HF-hosted model inference.

    Supports text generation (primary) and feature extraction.

    Offline mode (no HF_TOKEN):
        generate() → raises HFTokenMissingError
        embed()    → raises HFTokenMissingError
        Use mocks in tests (do not set HF_TOKEN in tests).
    """

    def __init__(self, config: Optional[HFClientConfig] = None) -> None:
        self.config = config or HFClientConfig.from_env()
        self._client: Optional[HFInferenceClient] = None

    def _get_client(self) -> HFInferenceClient:
        if self._client is None:
            self._client = HFInferenceClient(self.config)
        return self._client

    def generate(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        task: str = "text-generation",
    ) -> InferenceResult:
        """Generate text using the HF Inference API.

        In offline mode (no HF_TOKEN), raises HFTokenMissingError.

        Parameters
        ----------
        prompt : str
            Input text / instruction
        model_id : str, optional
            Override config model_id
        max_new_tokens : int, optional
            Override config max_new_tokens
        temperature : float, optional
            Override config temperature
        task : str
            Inference task type

        Returns
        -------
        InferenceResult with text and provenance
        """
        if not self.config.live_mode:
            raise HFTokenMissingError()

        _model_id = model_id or self.config.model_id
        _max_tokens = max_new_tokens or self.config.max_new_tokens
        _temperature = temperature if temperature is not None else self.config.temperature
        config_dict = self.config.to_provenance_dict()

        start = time.monotonic()

        try:
            client = self._get_client()

            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": _max_tokens,
                    "temperature": _temperature,
                    "return_full_text": False,
                },
            }

            raw = client.infer(_model_id, payload)
            latency = time.monotonic() - start

            # Parse response
            text = _extract_text(raw, task)

            prov = HFProvenance.build(
                model_id=_model_id,
                task=task,
                config_dict=config_dict,
                execution_mode="live",
                response_content={"length": len(text)},
                latency_s=latency,
                success=True,
                notes=f"Text generation: {len(prompt)} chars → {len(text)} chars",
            )
            return InferenceResult(
                text=text,
                raw=raw,
                model_id=_model_id,
                task=task,
                latency_s=latency,
                provenance=prov,
            )

        except HFProviderError:
            raise
        except Exception as exc:
            latency = time.monotonic() - start
            prov = HFProvenance.build(
                model_id=_model_id,
                task=task,
                config_dict=config_dict,
                execution_mode="live",
                latency_s=latency,
                success=False,
                error_code="NETWORK_ERROR",
                notes=f"HF Inference failed: {type(exc).__name__}",
            )
            raise HFProviderError(
                f"HF inference failed: {type(exc).__name__}",
                model_id=_model_id,
                task=task,
            )

    def embed(
        self,
        texts: List[str],
        model_id: Optional[str] = None,
    ) -> Tuple[List[List[float]], HFProvenance]:
        """Embed texts using HF feature-extraction models.

        In offline mode, raises HFTokenMissingError.
        Returns (embeddings, provenance).
        """
        if not self.config.live_mode:
            raise HFTokenMissingError()

        _model_id = model_id or "sentence-transformers/all-MiniLM-L6-v2"
        config_dict = self.config.to_provenance_dict()
        start = time.monotonic()

        try:
            client = self._get_client()
            raw = client.embed(_model_id, texts)
            latency = time.monotonic() - start

            # Parse embeddings
            embeddings = _parse_embeddings(raw, len(texts))

            prov = HFProvenance.build(
                model_id=_model_id,
                task="feature-extraction",
                config_dict=config_dict,
                execution_mode="live",
                response_content={"n_texts": len(texts), "dim": len(embeddings[0]) if embeddings else 0},
                latency_s=latency,
                success=True,
                notes=f"Embedded {len(texts)} texts using {_model_id!r}",
            )
            return embeddings, prov

        except HFProviderError:
            raise
        except Exception as exc:
            latency = time.monotonic() - start
            prov = HFProvenance.build(
                model_id=_model_id,
                task="feature-extraction",
                config_dict=config_dict,
                execution_mode="live",
                latency_s=latency,
                success=False,
                error_code="NETWORK_ERROR",
                notes=f"Embedding failed: {type(exc).__name__}",
            )
            raise HFProviderError(
                f"Embedding failed: {type(exc).__name__}",
                model_id=_model_id,
                task="feature-extraction",
            )

    def smoke_test(self) -> Dict[str, Any]:
        """Run a minimal live connectivity test.

        Returns a dict with connectivity info. Never prints credentials.
        Only runs if HF_TOKEN is set (live_mode=True).
        """
        if not self.config.live_mode:
            return {
                "status": "offline",
                "live_mode": False,
                "notes": "HF_TOKEN not set; offline mode only.",
            }

        try:
            result = self.generate(
                prompt="Hello, ResearchForge.",
                max_new_tokens=20,
                temperature=0.0,
            )
            return {
                "status": "success",
                "live_mode": True,
                "model_id": result.model_id,
                "latency_s": result.latency_s,
                "response_length": len(result.text),
                "provenance_fingerprint": result.provenance.fingerprint(),
                "execution_mode": result.provenance.execution_mode,
            }
        except HFProviderError as exc:
            return {
                "status": "failed",
                "live_mode": True,
                "error_code": exc.error_code,
                "error_type": type(exc).__name__,
            }


def _extract_text(raw: Any, task: str) -> str:
    """Parse generated text from HF Inference API response."""
    if isinstance(raw, list) and raw:
        item = raw[0]
        if isinstance(item, dict):
            return str(
                item.get("generated_text", "")
                or item.get("translation_text", "")
                or item.get("summary_text", "")
                or ""
            )
        return str(item)
    if isinstance(raw, dict):
        return str(
            raw.get("generated_text", "")
            or raw.get("text", "")
            or ""
        )
    return str(raw) if raw is not None else ""


def _parse_embeddings(raw: Any, n_expected: int) -> List[List[float]]:
    """Parse feature-extraction response into list of embedding vectors."""
    if isinstance(raw, list):
        if raw and isinstance(raw[0], (int, float)):
            # Single embedding
            return [raw]
        if raw and isinstance(raw[0], list):
            return raw
    raise HFInvalidResponseError(
        f"Unexpected embedding response format: {type(raw).__name__}",
        task="feature-extraction",
    )
