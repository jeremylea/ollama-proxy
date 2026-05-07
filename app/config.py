"""
Configuration for the Ollama API proxy.

All settings are loaded from environment variables or a .env file.
Model metadata is loaded from config.yaml at startup.
"""

import os
import logging
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


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


settings = Settings

# Model metadata loaded from config.yaml
MODEL_METADATA: Dict[str, Any] = {}


def load_model_metadata() -> None:
    """Load model metadata from config.yaml (or config.example.yaml as fallback).

    Mutates MODEL_METADATA in-place so that modules which already imported it
    (e.g. app.main) see the updated contents.
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(base_dir, "config.yaml")
    example_path = os.path.join(base_dir, "config.example.yaml")

    # Use config.yaml if it exists, otherwise fall back to config.example.yaml
    if not os.path.exists(config_path):
        if os.path.exists(example_path):
            config_path = example_path
            logger.info("Using config.example.yaml as config.yaml not found")
        else:
            logger.warning(
                "Neither config.yaml nor config.example.yaml found in %s", base_dir
            )
            MODEL_METADATA.clear()
            return

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        MODEL_METADATA.clear()
        MODEL_METADATA.update(config.get("models", {}) if config else {})
        logger.info(
            "Loaded metadata for %d models from %s",
            len(MODEL_METADATA),
            os.path.basename(config_path),
        )
    except Exception as e:
        logger.error("Failed to load %s: %s", os.path.basename(config_path), e)
        MODEL_METADATA.clear()


def setup_logging() -> None:
    """Configure application-wide logging."""
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # Load model metadata after logging is configured
    load_model_metadata()
