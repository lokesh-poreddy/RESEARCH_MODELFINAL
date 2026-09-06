"""Hugging Face dataset discovery provider.

Implements dataset search and metadata retrieval via the HF Hub API.
All results carry provenance records.

Offline mode: returns empty results.
Live mode: queries HF Hub datasets API.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from .client import HFClientConfig
from .errors import HFProviderError
from .provenance import HFProvenance


class DatasetMetadata:
    """Typed dataset metadata from HF Hub."""

    def __init__(self, raw: Dict[str, Any]) -> None:
        self._raw = raw

    @property
    def dataset_id(self) -> str:
        return self._raw.get("id", "")

    @property
    def tags(self) -> List[str]:
        return list(self._raw.get("tags", []))

    @property
    def task_categories(self) -> List[str]:
        return list(self._raw.get("task_categories", []))

    @property
    def downloads(self) -> int:
        return int(self._raw.get("downloads", 0))

    @property
    def likes(self) -> int:
        return int(self._raw.get("likes", 0))

    @property
    def revision(self) -> str:
        return self._raw.get("sha", self._raw.get("revision", ""))

    @property
    def description(self) -> str:
        return self._raw.get("description", "")[:500]  # Truncate for safety

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "tags": self.tags,
            "task_categories": self.task_categories,
            "downloads": self.downloads,
            "likes": self.likes,
            "revision": self.revision,
        }


class HuggingFaceDatasetProvider:
    """Provider for HF dataset discovery and metadata retrieval."""

    HUB_API_BASE = "https://huggingface.co/api"

    def __init__(self, config: Optional[HFClientConfig] = None) -> None:
        self.config = config or HFClientConfig.from_env()

    def search(
        self,
        task: str = "",
        query: str = "",
        limit: int = 10,
        sort: str = "downloads",
    ) -> Tuple[List[DatasetMetadata], HFProvenance]:
        """Search HF Hub for datasets.

        Returns (datasets, provenance). Offline: ([], provenance).
        """
        start = time.monotonic()
        config_dict = self.config.to_provenance_dict()

        if not self.config.live_mode:
            prov = HFProvenance.build(
                model_id="",
                task=task or "dataset-search",
                config_dict=config_dict,
                execution_mode="offline",
                success=True,
                notes="Offline mode: HF_TOKEN not set, returning empty dataset list.",
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

            url = f"{self.HUB_API_BASE}/datasets?" + urllib.parse.urlencode(params)
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
            datasets = [DatasetMetadata(d) for d in (data if isinstance(data, list) else [])]

            prov = HFProvenance.build(
                model_id="",
                task=task or "dataset-search",
                config_dict=config_dict,
                execution_mode="live",
                response_content={"count": len(datasets)},
                latency_s=latency,
                success=True,
                notes=f"Searched datasets: task={task!r}, query={query!r}, limit={limit}",
            )
            return datasets, prov

        except HFProviderError:
            raise
        except Exception as exc:
            latency = time.monotonic() - start
            prov = HFProvenance.build(
                model_id="",
                task=task or "dataset-search",
                config_dict=config_dict,
                execution_mode="live",
                latency_s=latency,
                success=False,
                error_code="NETWORK_ERROR",
                notes=f"Dataset search failed: {type(exc).__name__}",
            )
            return [], prov

    def get_dataset_info(self, dataset_id: str) -> Tuple[Optional[DatasetMetadata], HFProvenance]:
        """Retrieve metadata for a specific dataset."""
        start = time.monotonic()
        config_dict = self.config.to_provenance_dict()

        if not self.config.live_mode:
            prov = HFProvenance.build(
                model_id="",
                dataset_id=dataset_id,
                task="dataset-info",
                config_dict=config_dict,
                execution_mode="offline",
                success=False,
                notes="Offline mode: dataset info not available without HF_TOKEN.",
            )
            return None, prov

        try:
            import urllib.request
            import json as _json

            url = f"{self.HUB_API_BASE}/datasets/{dataset_id}"
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
            meta = DatasetMetadata(data)
            prov = HFProvenance.build(
                model_id="",
                dataset_id=dataset_id,
                dataset_revision=meta.revision,
                task="dataset-info",
                config_dict=config_dict,
                execution_mode="live",
                response_content=meta.to_dict(),
                latency_s=latency,
                success=True,
                notes=f"Retrieved dataset info for {dataset_id!r}",
            )
            return meta, prov

        except Exception as exc:
            latency = time.monotonic() - start
            prov = HFProvenance.build(
                model_id="",
                dataset_id=dataset_id,
                task="dataset-info",
                config_dict=config_dict,
                execution_mode="live",
                latency_s=latency,
                success=False,
                error_code="NETWORK_ERROR",
                notes=f"Dataset info failed for {dataset_id!r}: {type(exc).__name__}",
            )
            return None, prov
