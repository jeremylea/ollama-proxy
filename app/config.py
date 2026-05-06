"""
Configuration for the Ollama API proxy.

All settings are loaded from environment variables or a .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    # LiteLLM proxy configuration
    LITELLM_BASE_URL: str = os.getenv("LITELLM_BASE_URL", "http://localhost:4000")
    LITELLM_API_KEY: str = os.getenv("LITELLM_API_KEY", "")

    # Server configuration
    PORT: int = int(os.getenv("PORT", "11434"))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info").lower()

    # Request timeout in seconds for calls to LiteLLM proxy
    HTTP_TIMEOUT: float = float(os.getenv("HTTP_TIMEOUT", "300.0"))

    # Proxy version string returned by /api/version
    VERSION: str = "0.6.4"


settings = Settings()


def setup_logging() -> None:
    """Configure application-wide logging."""
    import logging

    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
