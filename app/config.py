"""
Central configuration for the RAG chatbot.
All settings are loaded from environment variables with sensible defaults.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    # ── Database ──────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ragbot"

    # ── Redis ─────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── LLM Provider (OpenAI) ─────────────────────────────────
    llm_provider: str = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-5-nano"

    # ── Embedding ─────────────────────────────────────────────
    embedding_model: str = "multi-qa-MiniLM-L6-cos-v1"
    embedding_dimension: int = 384

    # ── ASR (Speech-to-Text via AssemblyAI) ───────────────────
    assemblyai_api_key: str = ""

    # ── Confidence Thresholds ─────────────────────────────────
    asr_confidence_threshold: float = 0.70
    retrieval_confidence_low: float = 0.20
    retrieval_confidence_high: float = 0.70
    answer_match_threshold: float = 0.80
    language_detection_threshold: float = 0.85

    # ── Concurrency ──────────────────────────────────────────
    max_concurrent_requests: int = 50
    request_timeout_seconds: int = 30

    # ── File Storage ──────────────────────────────────────────
    upload_dir: str = "./uploads"
    max_upload_size_mb: int = 100

    # ── Logging ───────────────────────────────────────────────
    log_level: str = "INFO"

    # ── App ───────────────────────────────────────────────────
    app_name: str = "RAG Answering Service"
    app_version: str = "1.0.0"
    debug: bool = False

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
