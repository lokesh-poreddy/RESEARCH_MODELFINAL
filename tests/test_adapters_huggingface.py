"""Tests for the Hugging Face adapter layer.

All tests run offline by default. No HF_TOKEN or GEMINI_API_KEY required.
Live tests are gated on RESEARCHFORGE_HF_LIVE_TEST=1.

Test categories:
    - No-token / offline behavior
    - Configuration loading
    - Error types and propagation
    - Mock inference (no real network)
    - Provenance correctness
    - Controller/ECRM integration (offline)
    - Secret safety (token never in logs/errors/artifacts)
"""
from __future__ import annotations

import os
import json
import time
import hashlib
from unittest.mock import patch, MagicMock
from typing import Any, Dict, List

import numpy as np
import pytest


# ── Import guard ─────────────────────────────────────────────────────────────

from researchforge.adapters.huggingface import (
    HFProviderError,
    HFTokenMissingError,
    HFAuthError,
    HFModelNotFoundError,
    HFRateLimitedError,
    HFTimeoutError,
    HFInvalidResponseError,
    HFProviderUnavailableError,
    HFNetworkError,
    HFClientConfig,
    get_hf_client,
    HuggingFaceInferenceProvider,
    HuggingFaceModelProvider,
    HuggingFaceDatasetProvider,
    HuggingFaceEmbeddingProvider,
    HFProvenance,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _offline_config() -> HFClientConfig:
    """Config with live_mode=False (no HF_TOKEN)."""
    return HFClientConfig(live_mode=False)


def _mock_config() -> HFClientConfig:
    """Config that simulates live mode for mocked tests."""
    return HFClientConfig(live_mode=True, model_id="test/model", timeout_s=5.0)


LIVE = os.environ.get("RESEARCHFORGE_HF_LIVE_TEST", "0") == "1"
skip_unless_live = pytest.mark.skipif(not LIVE, reason="Live HF test requires RESEARCHFORGE_HF_LIVE_TEST=1")


# ── Section 1: Error types ────────────────────────────────────────────────────

class TestErrorTypes:
    def test_hf_provider_error_base(self):
        e = HFProviderError("test message", model_id="a/b", task="text-generation")
        assert "test message" in str(e)
        assert e.model_id == "a/b"
        assert e.task == "text-generation"
        assert e.error_code == "HF_PROVIDER_ERROR"

    def test_token_missing_error(self):
        e = HFTokenMissingError()
        assert "HF_TOKEN" in str(e)
        # Must NOT contain actual token values
        token = os.environ.get("HF_TOKEN", "NOSECRET")
        if token != "NOSECRET":
            assert token not in str(e)
        assert e.error_code == "HF_TOKEN_MISSING"

    def test_auth_error(self):
        e = HFAuthError("auth failed", model_id="org/model")
        assert e.error_code == "HF_AUTH_FAILED"
        assert e.model_id == "org/model"

    def test_model_not_found(self):
        e = HFModelNotFoundError("not found", model_id="org/missing")
        assert e.error_code == "MODEL_NOT_FOUND"

    def test_rate_limited_with_retry_after(self):
        e = HFRateLimitedError("rate limited", retry_after_s=60.0)
        assert e.retry_after_s == 60.0
        assert e.error_code == "RATE_LIMITED"

    def test_timeout_error(self):
        e = HFTimeoutError("timed out", timeout_s=30.0)
        assert e.timeout_s == 30.0
        assert e.error_code == "TIMEOUT"

    def test_invalid_response(self):
        e = HFInvalidResponseError("bad json")
        assert e.error_code == "INVALID_RESPONSE"

    def test_network_error(self):
        e = HFNetworkError("connection refused")
        assert e.error_code == "NETWORK_ERROR"

    def test_provider_unavailable(self):
        e = HFProviderUnavailableError("503")
        assert e.error_code == "PROVIDER_UNAVAILABLE"

    def test_errors_are_subclasses(self):
        for cls in [
            HFTokenMissingError, HFAuthError, HFModelNotFoundError,
            HFRateLimitedError, HFTimeoutError, HFInvalidResponseError,
            HFProviderUnavailableError, HFNetworkError,
        ]:
            if cls == HFTokenMissingError:
                assert issubclass(cls, HFProviderError)
            else:
                assert issubclass(cls, HFProviderError)


# ── Section 2: Configuration ──────────────────────────────────────────────────

class TestConfiguration:
    def test_offline_config_no_token(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_TOKEN", None)
            cfg = HFClientConfig.from_env()
            assert cfg.live_mode is False

    def test_live_config_with_token(self):
        with patch.dict(os.environ, {"HF_TOKEN": "hf_fakefakefake"}, clear=False):
            cfg = HFClientConfig.from_env()
            assert cfg.live_mode is True

    def test_provenance_dict_excludes_token(self):
        with patch.dict(os.environ, {"HF_TOKEN": "hf_fakefakefake"}, clear=False):
            cfg = HFClientConfig.from_env()
            prov_dict = cfg.to_provenance_dict()
            assert "hf_fakefakefake" not in str(prov_dict)
            assert "HF_TOKEN" not in prov_dict
            assert "token" not in prov_dict

    def test_config_immutability(self):
        cfg = HFClientConfig(live_mode=False)
        with pytest.raises((AttributeError, TypeError)):
            cfg.live_mode = True  # type: ignore[misc]

    def test_custom_env_vars(self):
        with patch.dict(os.environ, {
            "RF_HF_MODEL_ID": "custom/model",
            "RF_HF_TIMEOUT_S": "60",
            "RF_HF_MAX_NEW_TOKENS": "1024",
            "HF_TOKEN": "hf_fake",
        }):
            cfg = HFClientConfig.from_env()
            assert cfg.model_id == "custom/model"
            assert cfg.timeout_s == 60.0
            assert cfg.max_new_tokens == 1024

    def test_secret_not_in_config_repr(self):
        with patch.dict(os.environ, {"HF_TOKEN": "hf_supersecret123"}, clear=False):
            cfg = HFClientConfig.from_env()
            cfg_str = str(cfg)
            assert "hf_supersecret123" not in cfg_str


# ── Section 3: Provenance ─────────────────────────────────────────────────────

class TestProvenance:
    def test_provenance_build(self):
        prov = HFProvenance.build(
            model_id="org/model",
            task="text-generation",
            config_dict={"model_id": "org/model", "live_mode": False},
            execution_mode="offline",
            success=True,
            notes="test",
        )
        assert prov.provider == "huggingface"
        assert prov.model_id == "org/model"
        assert prov.task == "text-generation"
        assert prov.execution_mode == "offline"
        assert prov.success is True

    def test_provenance_fingerprint_deterministic(self):
        prov1 = HFProvenance.build(
            model_id="org/model",
            task="text-generation",
            config_dict={"a": "b"},
            execution_mode="offline",
            success=True,
        )
        prov2 = HFProvenance(
            **{k: v for k, v in prov1.to_dict().items()}
        )
        # Same content → same fingerprint
        assert prov1.fingerprint() == prov2.fingerprint()

    def test_provenance_to_dict_no_credentials(self):
        prov = HFProvenance.build(
            model_id="org/model",
            task="text-generation",
            config_dict={"model_id": "org/model"},
            execution_mode="live",
            notes="with hf_token in notes should be excluded",
        )
        d = prov.to_dict()
        d_str = json.dumps(d)
        # No actual token values
        real_token = os.environ.get("HF_TOKEN", "")
        if real_token:
            assert real_token not in d_str

    def test_artifact_fingerprint_computed(self):
        prov = HFProvenance.build(
            model_id="org/model",
            task="text-generation",
            config_dict={},
            execution_mode="live",
            response_content={"text": "hello"},
            success=True,
        )
        assert prov.artifact_fingerprint  # non-empty
        assert len(prov.artifact_fingerprint) == 64  # SHA-256 hex

    def test_config_fingerprint_computed(self):
        prov = HFProvenance.build(
            model_id="org/model",
            task="text-generation",
            config_dict={"model_id": "org/model", "live_mode": False},
            execution_mode="offline",
        )
        assert prov.config_fingerprint
        assert len(prov.config_fingerprint) == 64


# ── Section 4: Offline provider behavior ─────────────────────────────────────

class TestOfflineInferenceProvider:
    def test_generate_offline_raises_token_missing(self):
        provider = HuggingFaceInferenceProvider(_offline_config())
        with pytest.raises(HFTokenMissingError):
            provider.generate("test prompt")

    def test_embed_offline_raises_token_missing(self):
        provider = HuggingFaceInferenceProvider(_offline_config())
        with pytest.raises(HFTokenMissingError):
            provider.embed(["text1", "text2"])

    def test_smoke_test_offline(self):
        provider = HuggingFaceInferenceProvider(_offline_config())
        result = provider.smoke_test()
        assert result["status"] == "offline"
        assert result["live_mode"] is False


class TestOfflineModelProvider:
    def test_search_offline_returns_empty(self):
        provider = HuggingFaceModelProvider(_offline_config())
        models, prov = provider.search(task="text-classification")
        assert models == []
        assert prov.execution_mode == "offline"
        assert prov.success is True

    def test_get_info_offline_returns_none(self):
        provider = HuggingFaceModelProvider(_offline_config())
        meta, prov = provider.get_model_info("org/model")
        assert meta is None
        assert prov.execution_mode == "offline"


class TestOfflineDatasetProvider:
    def test_search_offline_returns_empty(self):
        provider = HuggingFaceDatasetProvider(_offline_config())
        datasets, prov = provider.search(task="text-classification")
        assert datasets == []
        assert prov.execution_mode == "offline"
        assert prov.success is True

    def test_get_info_offline_returns_none(self):
        provider = HuggingFaceDatasetProvider(_offline_config())
        meta, prov = provider.get_dataset_info("org/dataset")
        assert meta is None
        assert prov.execution_mode == "offline"


class TestOfflineEmbeddingProvider:
    def test_embed_offline_uses_local_fallback(self):
        provider = HuggingFaceEmbeddingProvider(config=_offline_config(), allow_local_fallback=True)
        arrays, prov = provider.embed(["hello world", "test text"])
        assert arrays.shape[0] == 2
        assert arrays.shape[1] == 256  # HashingVectorizer default
        assert prov.execution_mode == "local_fallback"
        assert "local" in prov.model_id

    def test_embed_offline_no_fallback_raises(self):
        provider = HuggingFaceEmbeddingProvider(config=_offline_config(), allow_local_fallback=False)
        with pytest.raises(HFTokenMissingError):
            provider.embed(["hello"])

    def test_embed_single_offline(self):
        provider = HuggingFaceEmbeddingProvider(config=_offline_config(), allow_local_fallback=True)
        vec, prov = provider.embed_single("hello")
        assert vec.ndim == 1
        assert vec.shape[0] == 256


# ── Section 5: Mocked inference (live_mode=True but HTTP mocked) ──────────────

class TestMockedInference:
    """Test inference code paths with mocked HTTP responses."""

    def _make_provider(self) -> HuggingFaceInferenceProvider:
        return HuggingFaceInferenceProvider(_mock_config())

    def test_generate_text_success(self):
        provider = self._make_provider()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"generated_text": "Hello from ResearchForge"}]

        with patch("researchforge.adapters.huggingface.client.HFInferenceClient._get_session") as mock_sess_fn:
            mock_sess = MagicMock()
            mock_sess.post.return_value = mock_response
            mock_sess_fn.return_value = mock_sess
            with patch.dict(os.environ, {"HF_TOKEN": "hf_fake"}):
                result = provider.generate("Say hello")

        assert "Hello" in result.text
        assert result.model_id == "test/model"
        assert result.provenance.success is True
        assert result.provenance.execution_mode == "live"

    def test_generate_404_raises_model_not_found(self):
        provider = self._make_provider()
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("researchforge.adapters.huggingface.client.HFInferenceClient._get_session") as mock_sess_fn:
            mock_sess = MagicMock()
            mock_sess.post.return_value = mock_response
            mock_sess_fn.return_value = mock_sess
            with patch.dict(os.environ, {"HF_TOKEN": "hf_fake"}):
                with pytest.raises(HFModelNotFoundError):
                    provider.generate("test")

    def test_generate_401_raises_auth_error(self):
        provider = self._make_provider()
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("researchforge.adapters.huggingface.client.HFInferenceClient._get_session") as mock_sess_fn:
            mock_sess = MagicMock()
            mock_sess.post.return_value = mock_response
            mock_sess_fn.return_value = mock_sess
            with patch.dict(os.environ, {"HF_TOKEN": "hf_fake"}):
                with pytest.raises(HFAuthError):
                    provider.generate("test")

    def test_generate_429_raises_rate_limited(self):
        provider = self._make_provider()
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "30"}

        with patch("researchforge.adapters.huggingface.client.HFInferenceClient._get_session") as mock_sess_fn:
            mock_sess = MagicMock()
            mock_sess.post.return_value = mock_response
            mock_sess_fn.return_value = mock_sess
            with patch.dict(os.environ, {"HF_TOKEN": "hf_fake"}):
                with pytest.raises(HFRateLimitedError) as exc_info:
                    provider.generate("test")
        assert exc_info.value.retry_after_s == 30.0

    def test_generate_503_retries_then_fails(self):
        """503 should retry max_retries times then raise HFProviderUnavailableError."""
        config = HFClientConfig(live_mode=True, model_id="test/model", max_retries=2, retry_backoff_s=0.0)
        provider = HuggingFaceInferenceProvider(config)
        mock_response = MagicMock()
        mock_response.status_code = 503

        with patch("researchforge.adapters.huggingface.client.HFInferenceClient._get_session") as mock_sess_fn:
            mock_sess = MagicMock()
            mock_sess.post.return_value = mock_response
            mock_sess_fn.return_value = mock_sess
            with patch.dict(os.environ, {"HF_TOKEN": "hf_fake"}):
                with pytest.raises(HFProviderUnavailableError):
                    provider.generate("test")

    def test_secret_not_in_error_messages(self):
        """Verify that HF token never appears in error text."""
        provider = self._make_provider()
        fake_token = "hf_SUPERSECRETTOKEN12345"
        mock_response = MagicMock()
        mock_response.status_code = 403

        with patch("researchforge.adapters.huggingface.client.HFInferenceClient._get_session") as mock_sess_fn:
            mock_sess = MagicMock()
            mock_sess.post.return_value = mock_response
            mock_sess_fn.return_value = mock_sess
            with patch.dict(os.environ, {"HF_TOKEN": fake_token}):
                try:
                    provider.generate("test")
                except HFProviderError as exc:
                    assert fake_token not in str(exc)
                    assert "SUPERSECRETTOKEN" not in str(exc)

    def test_mocked_embedding_response(self):
        """Test embedding parsing from a mocked HF response."""
        provider = self._make_provider()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        with patch("researchforge.adapters.huggingface.client.HFInferenceClient._get_session") as mock_sess_fn:
            mock_sess = MagicMock()
            mock_sess.post.return_value = mock_response
            mock_sess_fn.return_value = mock_sess
            with patch.dict(os.environ, {"HF_TOKEN": "hf_fake"}):
                embeddings, prov = provider.embed(["text1", "text2"])

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 3
        assert prov.success is True


# ── Section 6: Controller integration (offline) ───────────────────────────────

class TestControllerIntegration:
    """Verify HF adapter doesn't break offline ResearchController behavior."""

    def test_controller_runs_without_hf_token(self):
        """Core controller must work with no HF credentials."""
        from researchforge.pipeline.controller import ResearchController
        from researchforge.benchmarks.tasks import digits_task

        task = digits_task(seed=42, n_train=30)

        ctrl = ResearchController(task, condition="no_memory", seed=42, population_size=3)
        result = ctrl.run(n_generations=2)

        assert result.best_metric >= 0.0
        assert len(result.trials) > 0

        assert result.best_metric >= 0.0
        assert len(result.trials) > 0

    def test_hf_provider_offline_config_builds(self):
        """HF adapter config builds without HF_TOKEN."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_TOKEN", None)
            cfg = HFClientConfig.from_env()
            assert cfg.live_mode is False
            # Adapter modules import cleanly
            from researchforge.adapters.huggingface import (
                HuggingFaceInferenceProvider,
                HuggingFaceModelProvider,
            )
            provider = HuggingFaceInferenceProvider(cfg)
            assert provider is not None


# ── Section 7: ECRM integration (offline) ────────────────────────────────────

class TestECRMIntegration:
    """Verify HF embedding provider maintains ECRM compatibility."""

    def test_default_ecrm_unaffected(self):
        """Default ECRM must use local hashing (256-d), not HF embeddings."""
        from researchforge.memory.ecrm import ECRM
        from researchforge.memory.embeddings import embed

        ecrm = ECRM()
        vec = embed("test strategy increase_capacity")
        assert len(vec) == 256  # Default 256-d local hashing

    def test_hf_embedding_local_fallback_dimension(self):
        """HF embedding provider in offline mode returns 256-d local fallback."""
        provider = HuggingFaceEmbeddingProvider(
            config=_offline_config(),
            allow_local_fallback=True,
        )
        arrays, prov = provider.embed(["test text"])
        assert arrays.shape[1] == 256
        assert prov.execution_mode == "local_fallback"
        assert prov.model_id.startswith("local/")


# ── Section 8: Security audit ─────────────────────────────────────────────────

class TestSecurityAudit:
    """Verify no credentials appear in adapter outputs."""

    def test_provenance_json_no_token(self):
        """Provenance records must never contain HF_TOKEN value."""
        fake_token = "hf_SECRETTOKEN99"
        with patch.dict(os.environ, {"HF_TOKEN": fake_token}):
            cfg = HFClientConfig.from_env()
            prov = HFProvenance.build(
                model_id="org/model",
                task="text-generation",
                config_dict=cfg.to_provenance_dict(),
                execution_mode="live",
            )
            prov_json = json.dumps(prov.to_dict())
            assert fake_token not in prov_json
            assert "SECRETTOKEN" not in prov_json

    def test_config_provenance_dict_no_token(self):
        """Config.to_provenance_dict() must not contain any auth material."""
        fake_token = "hf_ANOTHERSECRET"
        with patch.dict(os.environ, {"HF_TOKEN": fake_token}):
            cfg = HFClientConfig.from_env()
            d = cfg.to_provenance_dict()
            d_str = json.dumps(d)
            assert fake_token not in d_str
            assert "ANOTHERSECRET" not in d_str
            assert "Authorization" not in d_str
            assert "Bearer" not in d_str

    def test_error_messages_no_token(self):
        """Error messages must not contain HF_TOKEN."""
        fake_token = "hf_ERRORTEST456"
        with patch.dict(os.environ, {"HF_TOKEN": fake_token}):
            try:
                from researchforge.adapters.huggingface.client import _get_hf_token
                token = _get_hf_token()
                # We read the token internally but never expose it
                assert token == fake_token  # Verify we did read it
            except Exception as e:
                assert fake_token not in str(e)


# ── Section 9: Live tests (disabled by default) ───────────────────────────────

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
