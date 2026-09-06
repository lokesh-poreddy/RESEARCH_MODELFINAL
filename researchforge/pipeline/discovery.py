"""Algorithm-Discovery Pipeline: retrieve -> recombine -> synthesize ->
unit-test -> pilot -> validate (design doc Sec. 6 flowchart).

`Synthesizer` is a pluggable interface. `HeuristicSynthesizer` maps a chosen
strategy directly onto a genome-evolution operator -- deterministic, offline,
no network calls. `LLMSynthesizer` sketches how a real language model would
instead *propose* the mutation or a new architecture from natural language,
consistent with the design doc's own framing:

    "The LLM is not the research novelty. The LLM is replaceable."
    (source doc, Sec. "Why this is stronger than 'ResearchForge' alone")

It is deliberately left unimplemented in this offline reference build: there
are no API credentials configured in this sandbox, and wiring one up is a
config/credentials change, not an architecture change (see README's
"Extending toward the full design" section for exactly what that change looks
like).
"""
from __future__ import annotations
import random
from typing import List, Protocol

from ..genome.model_genome import ModelGenome, GENOME_SCHEMA
from ..genome.operators import apply_strategy


class Synthesizer(Protocol):
    def synthesize(self, strategy: str, base: ModelGenome, rng: random.Random,
                    population: List[ModelGenome]) -> ModelGenome: ...


class HeuristicSynthesizer:
    """Deterministic strategy -> genome-operator mapping. No network, no LLM."""

    def synthesize(self, strategy: str, base: ModelGenome, rng: random.Random,
                    population: List[ModelGenome]) -> ModelGenome:
        return apply_strategy(strategy, base, rng, population=population)


class LLMSynthesizer:
    """Uses an LLMProvider to propose genome mutations via structured JSON.
    
    Converts a mutation intent into a ModelGenome deterministically.
    Does not allow arbitrary Python execution.
    """

    def __init__(self, provider: 'LLMProvider'):
        self.provider = provider

    def synthesize(self, strategy: str, base: ModelGenome, rng: random.Random,
                    population: List[ModelGenome]) -> ModelGenome:
        import json
        import re

        prompt = f"""You are an automated machine learning research assistant.
Your task is to mutate the given base model genome according to the provided strategy.

Strategy: {strategy}
Base Genome: {base.to_json()}

You must output a structured mutation intent as a JSON object matching this schema:
{{
  "strategy": "{strategy}",
  "base_genome_id": "{base.model_id}",
  "mutation": {{
    "operator": "update_parameters",
    "parameters": {{
        "hyperparameters": {{"key": "new_value"}},
        "architecture": {{}},
        "data_pipeline": {{}}
    }}
  }},
  "reason": "Explain why this mutation aligns with the strategy.",
  "expected_effect": "Explain the expected outcome."
}}

Output ONLY the JSON object.
"""
        result = self.provider.generate(prompt)
        text = result.text
        
        # Extract JSON
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if not json_match:
            raise ValueError(f"INVALID_LLM_PROPOSAL: no JSON object found. Response: {text}")
            
        try:
            intent = json.loads(json_match.group(0))
        except json.JSONDecodeError as e:
            raise ValueError(f"INVALID_LLM_PROPOSAL: invalid JSON. Error: {e}")
            
        mutation = intent.get("mutation", {})
        operator = mutation.get("operator", "")
        params = mutation.get("parameters", {})
        
        # Deterministically convert intent to genome
        g = base.clone()
        g.parent_ids = [base.model_id]
        
        if operator == "update_parameters":
            if "hyperparameters" in params and isinstance(params["hyperparameters"], dict):
                g.hyperparameters.update(params["hyperparameters"])
            if "architecture" in params and isinstance(params["architecture"], dict):
                g.architecture.update(params["architecture"])
            if "data_pipeline" in params and isinstance(params["data_pipeline"], dict):
                g.data_pipeline.update(params["data_pipeline"])
        else:
            raise ValueError(f"INVALID_LLM_PROPOSAL: Unsupported operator '{operator}'.")
                
        return g


def unit_test(genome: ModelGenome) -> bool:
    """Sanity check: does the genome validate against the schema, pass the
    resource-bound safety check, and build a real, instantiable estimator?
    (design doc Sec. 6 'Unit Test & Sanity Check', extended with the
    safety_check() layer from genome.model_genome)."""
    try:
        genome.validate()
        if genome.safety_check():
            return False
        genome.build_estimator()
        return True
    except Exception:
        return False
