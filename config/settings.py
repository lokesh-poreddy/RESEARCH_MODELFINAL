"""
ResearchForge-ECRM Configuration Settings — v2
"""
from __future__ import annotations

LEGACY_STATUS = {
    "canonical": False,
    "replacement": "researchforge.config",
    "deprecated_since": "RF-1.0.0-alpha.2.1",
    "removal_target": None,  # preserved for legacy CLI (main.py)
    "cross_imports_allowed": False,  # researchforge/ must never import from here
}

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Project ──────────────────────────────────────────────────────────────
    project_name: str = "ResearchForge-ECRM"
    version: str = "2.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # ── LLM Providers ────────────────────────────────────────────────────────
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o"
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    # "openai" | "anthropic" | "mock"
    llm_provider: str = "openai"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.7
    llm_max_retries: int = 3              # retry-with-backoff on transient errors

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = "sqlite:///./researchforge.db"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_enabled: bool = False           # set True when Neo4j is available

    # ── Vector Store ─────────────────────────────────────────────────────────
    vector_store_path: str = "./data/vector_store"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # ── MLflow ───────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "sqlite:///./mlflow.db"
    mlflow_experiment_name: str = "ResearchForge-ECRM"

    # ── Policy / RL ───────────────────────────────────────────────────────────
    policy_type: str = "ucb"              # "ucb" | "thompson" | "rl"
    ucb_c: float = 1.41                   # UCB exploration constant
    rl_learning_rate: float = 0.1
    rl_gamma: float = 0.95
    failure_penalty_factor: float = 0.5

    # ── Memory ───────────────────────────────────────────────────────────────
    memory_retain_lambda: float = 0.01   # Forgetting rate
    memory_retain_threshold: float = 0.1
    memory_max_records: int = 10_000
    ntr_threshold: float = 0.3           # Negative Transfer Rate threshold

    # ── Evolution ────────────────────────────────────────────────────────────
    evolution_mutation_delta: float = 0.1
    evolution_population_size: int = 10
    evolution_plateau_threshold: int = 5  # experiments without improvement

    # ── Experiment Sandbox ────────────────────────────────────────────────────
    sandbox_timeout_seconds: int = 3600
    sandbox_max_gpu_hours: float = 10.0
    sandbox_workspace: str = "./sandbox_runs"

    # ── Retrieval APIs ───────────────────────────────────────────────────────
    openalex_email: Optional[str] = None
    semantic_scholar_api_key: Optional[str] = None
    github_token: Optional[str] = None

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_dir: str = "./data"
    models_dir: str = "./saved_models"
    logs_dir: str = "./logs"

    # ── v2 Promotion Gates ────────────────────────────────────────────────────
    v2_min_seeds: int = 5                 # minimum independent seeds before promotion
    v2_compute_budget_hours: float = 10.0  # total wall-clock budget per ablation
    v2_promotion_min_improvement: float = 0.01  # minimum delta over champion to promote

    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        for p in [self.data_dir, self.models_dir, self.logs_dir,
                  self.vector_store_path, self.sandbox_workspace]:
            Path(p).mkdir(parents=True, exist_ok=True)


# Singleton
settings = Settings()

