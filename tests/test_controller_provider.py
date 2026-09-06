"""Tests for LLM Provider wiring in ResearchController."""
import os
from unittest.mock import patch

from researchforge.pipeline.controller import ResearchController
from researchforge.benchmarks.tasks import digits_task

def test_controller_default_is_heuristic():
    """By default, ResearchController uses HeuristicSynthesizer."""
    from researchforge.pipeline.discovery import HeuristicSynthesizer
    with patch.dict(os.environ, {}, clear=True):
        task = digits_task(seed=42, n_train=30)
        ctrl = ResearchController(task, condition="random")
        assert isinstance(ctrl.synth, HeuristicSynthesizer)

def test_controller_huggingface_provider_selection():
    """When RF_LLM_PROVIDER=huggingface, ResearchController uses LLMSynthesizer backed by HF."""
    from researchforge.pipeline.discovery import LLMSynthesizer
    from researchforge.adapters.huggingface.inference import HuggingFaceInferenceProvider
    
    with patch.dict(os.environ, {
        "RF_LLM_PROVIDER": "huggingface",
        "RF_HF_MODEL_ID": "test/model",
        "HF_TOKEN": "fake_token"
    }, clear=True):
        task = digits_task(seed=42, n_train=30)
        ctrl = ResearchController(task, condition="random")
        
        assert isinstance(ctrl.synth, LLMSynthesizer)
        assert isinstance(ctrl.synth.provider, HuggingFaceInferenceProvider)
        assert ctrl.synth.provider.config.model_id == "test/model"
        assert ctrl.synth.provider.config.live_mode is True
