"""
BaseAgent: shared LLM interface and utilities for all ResearchForge agents — v2.

v2 changes:
- InstructorConfig dataclass: per-agent role, retry, JSON requirement
- InstructorConfig dataclass: per-agent role, retry, JSON requirement
- retry-with-backoff wrapper (up to settings.llm_max_retries attempts)
- ManuscriptAgent mock returns structured text instead of leaking raw prompt
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config.settings import settings

logger = logging.getLogger(__name__)


# ── Per-agent instructor configuration ───────────────────────────────────────

@dataclass
class InstructorConfig:
    """Declare how an agent should be prompted and validated."""
    system_role: str = ""         # injected into system prompt
    max_retries: int = 3          # LLM call retries on transient failure
    json_required: bool = False   # enforce JSON parsing of every response
    temperature_override: Optional[float] = None  # None → use global setting


# ── Base class ────────────────────────────────────────────────────────────────

class BaseAgent:
    """
    Abstract base class for all ResearchForge-ECRM agents.
    Wraps LLM calls with consistent prompt formatting, retry logic,
    and provider routing (OpenAI / Anthropic / mock).
    """

    def __init__(
        self,
        name: str = "BaseAgent",
        instructor: Optional[InstructorConfig] = None,
    ):
        self.name = name
        self.instructor = instructor or InstructorConfig()
        self._client = None
        self._model: str = ""
        self._init_client()

    def _init_client(self) -> None:
        """Initialise LLM client based on settings."""
        provider = settings.llm_provider
        if provider == "openai":
            if not settings.openai_api_key:
                logger.info("[%s] No OpenAI key configured; using deterministic offline mode.", self.name)
                return
            try:
                import openai  # type: ignore
                self._client = openai.OpenAI(api_key=settings.openai_api_key)
                self._model = settings.openai_model
                logger.info("[%s] Using OpenAI (%s).", self.name, self._model)
            except Exception as exc:
                logger.warning("[%s] OpenAI unavailable: %s", self.name, exc)

        elif provider == "anthropic":
            if not settings.anthropic_api_key:
                logger.info("[%s] No Anthropic key configured; using deterministic offline mode.", self.name)
                return
            try:
                import anthropic  # type: ignore
                self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
                self._model = settings.anthropic_model
                logger.info("[%s] Using Anthropic (%s).", self.name, self._model)
            except Exception as exc:
                logger.warning("[%s] Anthropic unavailable: %s", self.name, exc)



        else:
            logger.info("[%s] Running in mock LLM mode.", self.name)

    # ── Public call interface ─────────────────────────────────────────────────

    def llm_call(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        """Call the configured LLM and return the text response.

        Applies up to ``settings.llm_max_retries`` retries with exponential
        back-off on transient errors before falling back to the mock.
        """
        if self._client is None:
            return self._mock_llm(system_prompt, user_prompt)

        max_tries = max(1, self.instructor.max_retries or settings.llm_max_retries)
        last_exc: Optional[Exception] = None

        for attempt in range(1, max_tries + 1):
            try:
                if settings.llm_provider == "openai":
                    return self._openai_call(system_prompt, user_prompt, json_mode)
                elif settings.llm_provider == "anthropic":
                    return self._anthropic_call(system_prompt, user_prompt)

            except Exception as exc:
                last_exc = exc
                wait = 2 ** (attempt - 1)  # 1s, 2s, 4s …
                logger.warning(
                    "[%s] LLM call failed (attempt %d/%d): %s – retrying in %ds.",
                    self.name, attempt, max_tries, exc, wait,
                )
                if attempt < max_tries:
                    time.sleep(wait)

        logger.error("[%s] All retries exhausted (%s) – using mock.", self.name, last_exc)
        return self._mock_llm(system_prompt, user_prompt)

    # ── Provider implementations ──────────────────────────────────────────────

    def _openai_call(self, system: str, user: str, json_mode: bool) -> str:
        temperature = (
            self.instructor.temperature_override
            if self.instructor.temperature_override is not None
            else settings.llm_temperature
        )
        kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": settings.llm_max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def _anthropic_call(self, system: str, user: str) -> str:
        temperature = (
            self.instructor.temperature_override
            if self.instructor.temperature_override is not None
            else settings.llm_temperature
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=settings.llm_max_tokens,
            system=system,
            temperature=temperature,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text



    # ── Deterministic offline mock ─────────────────────────────────────────────

    def _mock_llm(self, system: str, user: str) -> str:
        """Deterministic mock for offline testing.

        Each agent name returns a valid, structured response so that
        downstream parsers do not receive raw prompt text.
        """
        if self.name == "HypothesisAgent":
            return json.dumps({
                "hypothesis": (
                    "Test a regularized, context-aware model variant and "
                    "compare it against the current baseline."
                ),
                "rationale": (
                    "A bounded intervention permits reproducible evidence "
                    "and failure attribution."
                ),
            })
        if self.name == "AnalyzerAgent":
            return json.dumps({
                "finding": "The experiment produced a measured validation outcome.",
                "claim": (
                    "The intervention is retained only when its outcome is "
                    "supported by the recorded experiment."
                ),
                "supports_hypothesis": True,
                "failure_flags": ["None"],
            })
        if self.name == "ManuscriptAgent":
            return json.dumps({
                "title": "ResearchForge-ECRM v2: context-conditioned memory for iterative ML research",
                "abstract": (
                    "We describe a deterministic research-loop framework that "
                    "combines procedural memory, negative-evidence tracking, and "
                    "strategy-portfolio diversity to improve branch-selection "
                    "efficiency under a fixed compute budget."
                ),
                "sections": ["Introduction", "Method", "Results", "Conclusion"],
            })
        # Generic fallback — structured so parsers can handle it
        return json.dumps({
            "agent": self.name,
            "response": (
                "A new hypothesis based on the research context. "
                "[offline mock — configure an LLM provider for real output]"
            ),
        })

    # ── Utility ───────────────────────────────────────────────────────────────

    def parse_json_response(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response, handling markdown code fences."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("[%s] Failed to parse JSON response.", self.name)
            return {"raw": text}
