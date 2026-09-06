"""Hugging Face embedding provider.

Wraps HF feature-extraction inference behind the EmbeddingProvider interface.

Architecture:
    ECRM / memory layer
        → EmbeddingProvider (interface)
            → HuggingFaceEmbeddingProvider (this module) [live/offline]
            → LocalHashingProvider (default, from memory.embeddings) [always available]

Design decision:
    - Local hashing (HashingVectorizer 256-d) remains the DEFAULT embedder.
    - HF embeddings are opt-in and must not change ECRM baseline behavior.
    - The embedding dimension is declared at construction and stored in provenance.
    - Any switch of embedding model would invalidate prior similarity scores.
    - This is documented in the provenance record.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .client import HFClientConfig
from .errors import HFTokenMissingError, HFProviderError
from .provenance import HFProvenance


# Default embedding model for HF path
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBED_DIM = 384  # all-MiniLM-L6-v2 output dimension


class HuggingFaceEmbeddingProvider:
    """Embedding provider that uses HF Inference API for feature extraction.

    Important invariants:
    - ECRM uses local hashing by default (memory.embeddings.embed).
    - This provider is opt-in only. Do not inject it into ECRM without
      explicit configuration, as it changes similarity scores.
    - The dimension and model_id are recorded in every provenance record.
    - The provider must declare its dimension before use.

    Offline behavior (no HF_TOKEN):
        embed() falls back to local hashing, marks execution_mode="local_fallback".
    """

    def __init__(
        self,
        model_id: str = DEFAULT_EMBED_MODEL,
        config: Optional[HFClientConfig] = None,
        allow_local_fallback: bool = True,
    ) -> None:
        self.model_id = model_id
        self.config = config or HFClientConfig.from_env()
        self.allow_local_fallback = allow_local_fallback
        self._dim: Optional[int] = None

    @property
    def declared_dimension(self) -> Optional[int]:
        """Embedding dimension. None until first successful live call."""
        return self._dim

    def embed(
        self,
        texts: List[str],
    ) -> Tuple[np.ndarray, HFProvenance]:
        """Embed texts. Returns (array[n_texts, dim], provenance).

        In offline mode:
            - If allow_local_fallback=True, uses HashingVectorizer (256-d)
            - If allow_local_fallback=False, raises HFTokenMissingError

        In live mode:
            - Uses HF feature-extraction API
            - Updates self._dim from actual response
        """
        config_dict = self.config.to_provenance_dict()
        start = time.monotonic()

        if not self.config.live_mode:
            if not self.allow_local_fallback:
                raise HFTokenMissingError()
            # Fall back to local hashing
            from ...memory.embeddings import embed as local_embed
            arrays = np.stack([local_embed(t) for t in texts])
            latency = time.monotonic() - start

            prov = HFProvenance.build(
                model_id="local/hashingvectorizer-256",
                task="feature-extraction",
                config_dict=config_dict,
                execution_mode="local_fallback",
                response_content={"n_texts": len(texts), "dim": arrays.shape[1]},
                latency_s=latency,
                success=True,
                notes=(
                    "HF_TOKEN not set; using local HashingVectorizer 256-d fallback. "
                    "Embedding similarity scores differ from HF model. "
                    "Do not mix with HF embeddings in the same ECRM store."
                ),
            )
            if self._dim is None:
                self._dim = arrays.shape[1]
            return arrays, prov

        # Live mode: use HF inference
        try:
            from .inference import HuggingFaceInferenceProvider
            provider = HuggingFaceInferenceProvider(self.config)
            embeddings_list, prov = provider.embed(texts, model_id=self.model_id)

            arrays = np.array(embeddings_list, dtype=np.float32)
            if self._dim is None and arrays.ndim == 2:
                self._dim = arrays.shape[1]
            return arrays, prov

        except HFProviderError:
            if self.allow_local_fallback:
                from ...memory.embeddings import embed as local_embed
                arrays = np.stack([local_embed(t) for t in texts])
                latency = time.monotonic() - start
                prov = HFProvenance.build(
                    model_id="local/hashingvectorizer-256",
                    task="feature-extraction",
                    config_dict=config_dict,
                    execution_mode="local_fallback",
                    latency_s=latency,
                    success=True,
                    notes="HF embedding failed; using local HashingVectorizer fallback.",
                )
                if self._dim is None:
                    self._dim = arrays.shape[1]
                return arrays, prov
            raise

    def embed_single(self, text: str) -> Tuple[np.ndarray, HFProvenance]:
        """Embed a single text. Returns (1-d array, provenance)."""
        arrays, prov = self.embed([text])
        return arrays[0], prov
