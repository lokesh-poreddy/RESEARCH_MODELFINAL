"""Tests for LLMSynthesizer mutation logic."""
import json
import random
from researchforge.pipeline.discovery import LLMSynthesizer
from researchforge.genome.model_genome import ModelGenome

class MockResponse:
    def __init__(self, text):
        self.text = text

class MockLLMProvider:
    def __init__(self, response_text):
        self.response_text = response_text
        self.last_prompt = None

    def generate(self, prompt: str) -> MockResponse:
        self.last_prompt = prompt
        return MockResponse(self.response_text)


def test_llm_synthesizer_mutation():
    """Prove that LLM generated JSON mutation intent correctly creates a ModelGenome."""
    # Define a valid JSON intent structure that the LLM is expected to return
    json_intent = json.dumps({
        "mutation": {
            "operator": "update_parameters",
            "parameters": {
                "hyperparameters": {"max_depth": 5, "n_estimators": 150}
            }
        },
        "rationale": "Testing mutation path."
    })
    
    provider = MockLLMProvider(response_text=f"```json\n{json_intent}\n```")
    synth = LLMSynthesizer(provider=provider)
    
    base_genome = ModelGenome(
        model_type="RandomForestClassifier",
        architecture={},
        hyperparameters={"max_depth": 3, "n_estimators": 100},
        model_id="base_1"
    )
    
    # Synthesize new genome
    new_genome = synth.synthesize("mutate", base_genome, random.Random(42), [])
    
    # Verify canonical mutation path was followed
    assert new_genome is not base_genome
    assert new_genome.model_type == "RandomForestClassifier"
    assert new_genome.hyperparameters["max_depth"] == 5
    assert new_genome.hyperparameters["n_estimators"] == 150
    assert new_genome.parent_ids == ["base_1"]
