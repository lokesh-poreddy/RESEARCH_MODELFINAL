"""Hugging Face model discovery provider.

Implements model search, metadata retrieval, and revision capture
via the Hugging Face Hub API.

All results are returned as typed dicts with provenance attached.
Results are evidence with provenance, not scientific ground truth.

Offline mode (no HF_TOKEN): returns empty results with execution_mode="offline".
Live mode (HF_TOKEN present): queries HF Hub API.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from .client import HFClientConfig, HFInferenceClient, get_hf_client
from .errors import (
    HFProviderError,
    HFNetworkError,
    HFInvalidResponseError,
)
from .provenance import HFProvenance


class ModelMetadata:
    """Typed model metadata from Hugging Face Hub."""

    def __init__(self, raw: Dict[str, Any]) -> None:
        self._raw = raw

    @property
    def model_id(self) -> str:
        return self._raw.get("modelId", self._raw.get("id", ""))

    @property
    def pipeline_tag(self) -> str:
        return self._raw.get("pipeline_tag", "")

    @property
    def tags(self) -> List[str]:
        return list(self._raw.get("tags", []))

    @property
    def downloads(self) -> int:
        return int(self._raw.get("downloads", 0))

    @property
    def likes(self) -> int:
        return int(self._raw.get("likes", 0))

    @property
    def revision(self) -> str:
        """Model git revision / SHA if available."""
        return self._raw.get("sha", self._raw.get("revision", ""))

    @property
    def library_name(self) -> str:
        return self._raw.get("library_name", "")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "pipeline_tag": self.pipeline_tag,
            "tags": self.tags,
            "downloads": self.downloads,
            "likes": self.likes,
            "revision": self.revision,
            "library_name": self.library_name,
        }


class HuggingFaceModelProvider:
    """Provider for HF model discovery and metadata retrieval.

    Uses the HF Hub REST API (/api/models endpoint).
    Results are wrapped with provenance metadata.

    Architecture position:
        ResearchController
            → HuggingFaceModelProvider.search()
            → [TMG candidate creation with hub_origin metadata]
    """

    HUB_API_BASE = "https://huggingface.co/api"

    def __init__(self, config: Optional[HFClientConfig] = None) -> None:
        self.config = config or HFClientConfig.from_env()

    def _get_session(self) -> Any:
        """Get HTTP session for Hub API calls."""
        try:
            import requests
            sess = requests.Session()
            if self.config.live_mode:
                from .client import _get_hf_token
                token = _get_hf_token()
                sess.headers["Authorization"] = f"Bearer {token}"
            return sess
        except ImportError:
            return None

    def search(
        self,
        task: str = "",
        query: str = "",
        limit: int = 10,
        sort: str = "downloads",
    ) -> Tuple[List[ModelMetadata], HFProvenance]:
        """Search Hugging Face Hub for models matching criteria.

        Returns (models, provenance). In offline mode, returns ([], provenance).

        Parameters
        ----------
        task : str
            Pipeline tag filter e.g. "text-classification", "text-generation"
        query : str
            Search query string
        limit : int
            Maximum number of results
        sort : str
            Sort field: "downloads" | "likes" | "lastModified"

        Returns
        -------
        Tuple of (list of ModelMetadata, HFProvenance)
        """
        start = time.monotonic()
        config_dict = self.config.to_provenance_dict()

        if not self.config.live_mode:
            prov = HFProvenance.build(
                model_id="",
                task=task or "model-search",
                config_dict=config_dict,
                execution_mode="offline",
                success=True,
                notes="Offline mode: HF_TOKEN not set, returning empty model list.",
            )
            return [], prov

        try:
            import urllib.request
            import urllib.parse
            import json as _json

            params: Dict[str, Any] = {"limit": str(limit), "sort": sort}
            if task:
                params["filter"] = task
            if query:
                params["search"] = query

            url = f"{self.HUB_API_BASE}/models?" + urllib.parse.urlencode(params)

            from .client import _get_hf_token
            token = _get_hf_token()
            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            }

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                data = _json.loads(resp.read().decode("utf-8"))

            latency = time.monotonic() - start
            models = [ModelMetadata(m) for m in (data if isinstance(data, list) else [])]

            prov = HFProvenance.build(
                model_id="",
                task=task or "model-search",
                config_dict=config_dict,
                execution_mode="live",
                response_content={"count": len(models)},
                latency_s=latency,
                success=True,
                notes=f"Searched models: task={task!r}, query={query!r}, limit={limit}",
            )
            return models, prov

        except HFProviderError:
            raise
        except Exception as exc:
            latency = time.monotonic() - start
            prov = HFProvenance.build(
                model_id="",
                task=task or "model-search",
                config_dict=config_dict,
                execution_mode="live",
                latency_s=latency,
                success=False,
                error_code="NETWORK_ERROR",
                notes=f"Model search failed: {type(exc).__name__}",
            )
            return [], prov

    def get_model_info(self, model_id: str) -> Tuple[Optional[ModelMetadata], HFProvenance]:
        """Retrieve metadata for a specific model.

        Returns (ModelMetadata, provenance) or (None, provenance) if not found.
        """
        start = time.monotonic()
        config_dict = self.config.to_provenance_dict()

        if not self.config.live_mode:
            prov = HFProvenance.build(
                model_id=model_id,
                task="model-info",
                config_dict=config_dict,
                execution_mode="offline",
                success=False,
                notes="Offline mode: model info not available without HF_TOKEN.",
            )
            return None, prov

        try:
            import urllib.request
            import json as _json

            url = f"{self.HUB_API_BASE}/models/{model_id}"
            from .client import _get_hf_token
            token = _get_hf_token()
            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            }
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                data = _json.loads(resp.read().decode("utf-8"))

            latency = time.monotonic() - start
            meta = ModelMetadata(data)
            prov = HFProvenance.build(
                model_id=model_id,
                task="model-info",
                config_dict=config_dict,
                execution_mode="live",
                model_revision=meta.revision,
                response_content=meta.to_dict(),
                latency_s=latency,
                success=True,
                notes=f"Retrieved model info for {model_id!r}",
            )
            return meta, prov

        except Exception as exc:
            latency = time.monotonic() - start
            prov = HFProvenance.build(
                model_id=model_id,
                task="model-info",
                config_dict=config_dict,
                execution_mode="live",
                latency_s=latency,
                success=False,
                error_code="NETWORK_ERROR",
                notes=f"Model info failed for {model_id!r}: {type(exc).__name__}",
            )
            return None, prov
