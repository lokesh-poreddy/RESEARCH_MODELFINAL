"""LLM Provider abstraction.

Defines the interface for text generation providers (e.g. HuggingFace).
"""
from typing import Optional, Protocol, Any

class InferenceResult(Protocol):
    text: str
    raw: Any
    model_id: str
    latency_s: float
    provenance: Any

class LLMProvider(Protocol):
    def generate(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        task: str = "text-generation",
    ) -> InferenceResult:
        """Generate text from a prompt."""
        ...
