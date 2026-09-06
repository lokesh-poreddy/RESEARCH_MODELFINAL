"""Hugging Face inference provider.

Connects to Hugging Face Inference API for:
- Text generation / chat completion
- Feature extraction (embeddings)
- Other supported inference tasks

Provider mode selection:
    offline: HF_TOKEN not present → raises HFTokenMissingError when called live
    live:    HF_TOKEN present → uses HF Inference API
    gemini:  GEMINI_API_KEY present → uses Gemini as fallback LLM

Architecture position:
    LLMSynthesizer (future wiring)
        → HuggingFaceInferenceProvider.generate()
        → HFInferenceClient
        → Hugging Face Inference API / Gemini API (fallback)

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
    Falls back to Gemini API if GEMINI_API_KEY is set and HF fails.

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
        Falls back to Gemini if GEMINI_API_KEY present and HF fails.

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
            # Try Gemini fallback
            gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
            if gemini_key:
                return self._gemini_fallback(
                    prompt=prompt,
                    config_dict=config_dict,
                    original_model_id=_model_id,
                    task=task,
                    original_error=str(type(exc).__name__),
                )
            raise

    def _gemini_fallback(
        self,
        prompt: str,
        config_dict: Dict[str, Any],
        original_model_id: str,
        task: str,
        original_error: str,
    ) -> InferenceResult:
        """Use Gemini API as a fallback LLM if HF fails.

        GEMINI_API_KEY is read from environment only. Never logged.
        This fallback is explicitly provenance-recorded.
        """
        import json as _json
        import urllib.request

        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise HFProviderError(
                "HF inference failed and GEMINI_API_KEY is not set for fallback.",
                model_id=original_model_id,
                task=task,
            )

        start = time.monotonic()
        gemini_model = "gemini-1.5-flash"
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{gemini_model}:generateContent?key={api_key}"
        )

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
            },
        }

        body = _json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
        except Exception as exc2:
            latency = time.monotonic() - start
            prov = HFProvenance.build(
                model_id=f"gemini/{gemini_model}",
                task=task,
                config_dict=config_dict,
                execution_mode="live",
                inference_provider="gemini-fallback",
                latency_s=latency,
                success=False,
                error_code="NETWORK_ERROR",
                notes=(
                    f"Gemini fallback failed after HF error ({original_error}): "
                    f"{type(exc2).__name__}"
                ),
            )
            raise HFProviderError(
                f"Both HF ({original_error}) and Gemini fallback failed.",
                model_id=original_model_id,
                task=task,
            )

        latency = time.monotonic() - start

        # Extract text from Gemini response
        try:
            text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
        except (IndexError, KeyError, TypeError):
            text = str(data)

        prov = HFProvenance.build(
            model_id=f"gemini/{gemini_model}",
            task=task,
            config_dict=config_dict,
            execution_mode="live",
            inference_provider="gemini-fallback",
            response_content={"length": len(text)},
            latency_s=latency,
            success=True,
            notes=(
                f"Gemini fallback used after HF error ({original_error}). "
                f"Original HF model: {original_model_id!r}. "
                f"Fallback is explicitly recorded in provenance."
            ),
        )
        return InferenceResult(
            text=text,
            raw=data,
            model_id=f"gemini/{gemini_model}",
            task=task,
            latency_s=latency,
            provenance=prov,
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
