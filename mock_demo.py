import sys
import json
from unittest.mock import patch
import run_demo

def mock_make_request(*args, **kwargs):
    # Mock response format expected by LLMSynthesizer
    mutation_response = {
        "strategy": "test_strategy",
        "base_genome_id": "test",
        "mutation": {
            "operator": "update_parameters",
            "parameters": {
                "hyperparameters": {"learning_rate_init": 0.005}
            }
        },
        "reason": "Test reason",
        "expected_effect": "Test effect"
    }
    class MockResponse:
        @property
        def text(self):
            # The client usually returns a string for text generation, or a list of dicts.
            # wait, client.infer for text-generation returns the API response.
            # The `generate` method expects an `LLMResult` with `.text`.
            # In `HuggingFaceInferenceProvider.generate`, it parses the response.
            return ""

    # Actually, we need to mock at the requests.Session.post level or client.infer level.
    # It's easier to mock client.infer to return a list of dicts with 'generated_text'.
    return [{"generated_text": json.dumps(mutation_response)}]

if __name__ == "__main__":
    with patch("researchforge.adapters.huggingface.client.HFInferenceClient.infer", side_effect=mock_make_request):
        # We need to set the sys.argv to pass args to run_demo.py
        sys.argv = ["run_demo.py", "--seeds", "1", "--generations", "2", "--out", "mock_demo_results.json"]
        run_demo.main()
