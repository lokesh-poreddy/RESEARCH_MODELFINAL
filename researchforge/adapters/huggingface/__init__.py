"""Hugging Face provider adapter for ResearchForge-ECRM.

Adapter API version: 1
Schema version:      1

This module implements the provider-agnostic interfaces defined in
researchforge.adapters.providers using Hugging Face Hub and Inference
as the backend.

Secret management:
    HF_TOKEN is read exclusively from the environment.
    It is NEVER stored in source, artifacts, tests, or logs.
    If HF_TOKEN is unset, the provider degrades gracefully to an offline mock
    (returning deterministic dummy structures), ensuring the test suite and
    pipeline remain fully functional without credentials.

Usage:
    from researchforge.adapters.huggingface import (
        HuggingFaceInferenceProvider,
        HuggingFaceModelProvider,
        HuggingFaceDatasetProvider,
        HuggingFaceEmbeddingProvider,
        get_hf_client,
        HFProviderError,
    )
"""
from __future__ import annotations

from .errors import (                          # noqa: F401
    HFProviderError,
    HFTokenMissingError,
    HFAuthError,
    HFModelNotFoundError,
    HFModelNotSupportedError,
    HFProviderUnavailableError,
    HFRateLimitedError,
    HFTimeoutError,
    HFInvalidResponseError,
    HFTaskUnsupportedError,
    HFNetworkError,
)

from .client import get_hf_client, HFClientConfig  # noqa: F401

from .inference import HuggingFaceInferenceProvider    # noqa: F401
from .models import HuggingFaceModelProvider           # noqa: F401
from .datasets import HuggingFaceDatasetProvider       # noqa: F401
from .embeddings import HuggingFaceEmbeddingProvider   # noqa: F401
from .provenance import HFProvenance                   # noqa: F401

__all__ = [
    # Errors
    "HFProviderError",
    "HFTokenMissingError",
    "HFAuthError",
    "HFModelNotFoundError",
    "HFModelNotSupportedError",
    "HFProviderUnavailableError",
    "HFRateLimitedError",
    "HFTimeoutError",
    "HFInvalidResponseError",
    "HFTaskUnsupportedError",
    "HFNetworkError",
    # Client
    "get_hf_client",
    "HFClientConfig",
    # Providers
    "HuggingFaceInferenceProvider",
    "HuggingFaceModelProvider",
    "HuggingFaceDatasetProvider",
    "HuggingFaceEmbeddingProvider",
    # Provenance
    "HFProvenance",
]
