import os
import pytest
from researchforge.adapters.huggingface.client import HFClientConfig
from researchforge.adapters.huggingface.inference import HuggingFaceInferenceProvider
from researchforge.adapters.huggingface.models import HuggingFaceModelProvider
from researchforge.adapters.huggingface.datasets import HuggingFaceDatasetProvider

LIVE = os.environ.get("RESEARCHFORGE_HF_LIVE_TEST", "0") == "1"
skip_unless_live = pytest.mark.skipif(not LIVE, reason="Live HF test requires RESEARCHFORGE_HF_LIVE_TEST=1")

@skip_unless_live
class TestLiveHFIntegration:
    """Live integration tests. Requires RESEARCHFORGE_HF_LIVE_TEST=1 and HF_TOKEN."""

    def test_live_smoke_test(self):
        cfg = HFClientConfig.from_env()
        assert cfg.live_mode, "HF_TOKEN must be set for live tests"
        provider = HuggingFaceInferenceProvider(cfg)
        result = provider.smoke_test()
        assert result["status"] == "success", f"Smoke test failed: {result}"
        assert result["live_mode"] is True

    def test_live_model_search(self):
        cfg = HFClientConfig.from_env()
        provider = HuggingFaceModelProvider(cfg)
        models, prov = provider.search(task="text-classification", limit=5)
        assert prov.success is True
        assert prov.execution_mode == "live"

    def test_live_dataset_search(self):
        cfg = HFClientConfig.from_env()
        provider = HuggingFaceDatasetProvider(cfg)
        datasets, prov = provider.search(limit=5)
        assert prov.success is True
        assert prov.execution_mode == "live"
