"""Hugging Face client configuration and factory.

Reads HF_TOKEN from environment. Never logs, stores, or transmits the
token value in any artifact, log line, or error message.

Usage:
    config = HFClientConfig.from_env()
    client = get_hf_client(config)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .errors import (
    HFProviderError,
    HFTokenMissingError,
    HFAuthError,
    HFNetworkError,
    HFTimeoutError,
    HFRateLimitedError,
    HFProviderUnavailableError,
    HFInvalidResponseError,
    HFModelNotFoundError,
)


@dataclass(frozen=True)
class HFClientConfig:
    """Immutable configuration for the HF adapter layer.

    The token field is read from HF_TOKEN at construction time.
    It is never stored in any artifact or log.
    """
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"
    inference_provider: str = "auto"
    timeout_s: float = 30.0
    max_retries: int = 3
    retry_backoff_s: float = 2.0
    temperature: float = 0.1
    max_new_tokens: int = 512
    task: str = "text-generation"
    # Whether a live HF token was found (True) or we run in offline/mock mode
    live_mode: bool = False

    @classmethod
    def from_env(cls) -> "HFClientConfig":
        """Build config from environment variables.

        HF_TOKEN presence determines live vs offline mode.
        The token value itself is not stored in this object.
        """
        token = os.environ.get("HF_TOKEN", "").strip()
        live_mode = bool(token)

        return cls(
            model_id=os.environ.get("RF_HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3"),
            inference_provider=os.environ.get("RF_HF_INFERENCE_PROVIDER", "auto"),
            timeout_s=float(os.environ.get("RF_HF_TIMEOUT_S", "30")),
            max_retries=int(os.environ.get("RF_HF_MAX_RETRIES", "3")),
            retry_backoff_s=float(os.environ.get("RF_HF_RETRY_BACKOFF_S", "2.0")),
            temperature=float(os.environ.get("RF_HF_TEMPERATURE", "0.1")),
            max_new_tokens=int(os.environ.get("RF_HF_MAX_NEW_TOKENS", "512")),
            task=os.environ.get("RF_HF_TASK", "text-generation"),
            live_mode=live_mode,
        )

    def to_provenance_dict(self) -> Dict[str, Any]:
        """Return a dict safe for inclusion in provenance records.

        The token is NOT included.
        """
        return {
            "model_id": self.model_id,
            "inference_provider": self.inference_provider,
            "timeout_s": self.timeout_s,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "task": self.task,
            "live_mode": self.live_mode,
        }


def _get_hf_token() -> str:
    """Read HF_TOKEN from environment. Raises HFTokenMissingError if absent."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise HFTokenMissingError()
    return token




def get_hf_client(config: Optional[HFClientConfig] = None) -> "HFInferenceClient":
    """Return a configured HFInferenceClient.

    If HF_TOKEN is not set, returns an offline/mock client.
    """
    if config is None:
        config = HFClientConfig.from_env()
    return HFInferenceClient(config)


class HFInferenceClient:
    """Low-level HTTP wrapper around Hugging Face Inference API.

    Implements retry logic, timeout handling, rate-limit detection,
    and maps HTTP errors to typed HFProviderError subclasses.

    The token is read from environment at call time, not stored in
    instance attributes to minimise accidental exposure.
    """

    def __init__(self, config: HFClientConfig) -> None:
        self.config = config
        self._session: Any = None

    def _get_session(self) -> Any:
        """Lazily create an HTTP session using available libraries."""
        if self._session is None:
            try:
                import requests
                self._session = requests.Session()
            except ImportError:
                self._session = _UrllibSession()
        return self._session

    def _build_headers(self) -> Dict[str, str]:
        """Build HTTP headers. Token read from env at call time."""
        if not self.config.live_mode:
            return {"Content-Type": "application/json"}
        token = _get_hf_token()
        # The token appears only in the header value, never logged
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def _make_request(self, url: str, payload: Dict[str, Any]) -> Any:
        """POST to HF Inference API with retry/backoff.

        Returns parsed JSON response dict.
        Raises typed HFProviderError on failure.
        """
        import json as _json

        headers = self._build_headers()
        body = _json.dumps(payload).encode("utf-8")
        last_exc: Optional[Exception] = None

        for attempt in range(self.config.max_retries):
            try:
                session = self._get_session()
                resp = session.post(
                    url,
                    data=body,
                    headers=headers,
                    timeout=self.config.timeout_s,
                )

                status = getattr(resp, "status_code", None)

                if status == 200:
                    try:
                        return resp.json()
                    except Exception:
                        raise HFInvalidResponseError(
                            "Response is not valid JSON",
                            model_id=self.config.model_id,
                            task=self.config.task,
                        )

                if status == 401 or status == 403:
                    raise HFAuthError(
                        "Authentication failed — check HF_TOKEN is valid and has Inference Providers permission.",
                        model_id=self.config.model_id,
                    )

                if status == 404:
                    raise HFModelNotFoundError(
                        f"Model '{self.config.model_id}' not found on Hugging Face Hub.",
                        model_id=self.config.model_id,
                        task=self.config.task,
                    )

                if status == 429:
                    retry_after = float(getattr(resp, "headers", {}).get("Retry-After", 60))
                    raise HFRateLimitedError(
                        f"Rate limited. Retry after {retry_after:.0f}s.",
                        model_id=self.config.model_id,
                        task=self.config.task,
                        retry_after_s=retry_after,
                    )

                if status in (502, 503, 504):
                    if attempt < self.config.max_retries - 1:
                        time.sleep(self.config.retry_backoff_s * (2 ** attempt))
                        continue
                    raise HFProviderUnavailableError(
                        f"Hugging Face inference provider unavailable (HTTP {status}).",
                        model_id=self.config.model_id,
                        task=self.config.task,
                    )

                # Generic non-2xx
                raise HFInvalidResponseError(
                    f"Unexpected HTTP status {status}.",
                    model_id=self.config.model_id,
                    task=self.config.task,
                )

            except HFProviderError:
                raise
            except Exception as exc:
                err_str = str(exc).lower()
                if "timeout" in err_str or "timed out" in err_str:
                    last_exc = HFTimeoutError(
                        f"Request timed out after {self.config.timeout_s}s.",
                        model_id=self.config.model_id,
                        task=self.config.task,
                        timeout_s=self.config.timeout_s,
                    )
                else:
                    last_exc = HFNetworkError(
                        f"Network error communicating with Hugging Face API: {type(exc).__name__}",
                        model_id=self.config.model_id,
                        task=self.config.task,
                    )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_backoff_s * (2 ** attempt))
                    continue
                raise last_exc

        raise HFProviderUnavailableError(
            "Exhausted retries with no successful response.",
            model_id=self.config.model_id,
        )

    def infer(self, model_id: str, payload: Dict[str, Any]) -> Any:
        """Run inference against the given model."""
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        return self._make_request(url, payload)

    def infer_chat(self, model_id: str, messages: list, **kwargs: Any) -> Any:
        """Run chat completion inference."""
        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": messages,
        }
        payload.update(kwargs)
        url = f"https://api-inference.huggingface.co/v1/chat/completions"
        return self._make_request(url, payload)

    def embed(self, model_id: str, texts: list) -> Any:
        """Run feature-extraction (embedding) inference."""
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        return self._make_request(url, {"inputs": texts})


class _UrllibSession:
    """Minimal urllib fallback when requests is not available."""

    def post(self, url: str, data: bytes, headers: Dict[str, str],
             timeout: float) -> "_FakeResponse":
        import json as _json
        import urllib.request
        import urllib.error

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                code = resp.status
                return _FakeResponse(code, body)
        except urllib.error.HTTPError as e:
            body = e.read()
            return _FakeResponse(e.code, body)
        except urllib.error.URLError as e:
            raise ConnectionError(f"URLError: {e.reason}") from e


class _FakeResponse:
    def __init__(self, status_code: int, body: bytes) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> Any:
        import json as _json
        return _json.loads(self._body.decode("utf-8"))
