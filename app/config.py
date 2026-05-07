"""
Configuration for the Ollama API proxy.

All settings are loaded from environment variables or a .env file.
Model metadata is loaded from config.yaml at startup, then optionally
enriched with information from the Hugging Face API.
"""

import os
import logging
from typing import Any, Dict, List, Optional

import httpx
import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Hugging Face API base URL
HF_API_BASE = os.getenv("HF_API_BASE", "https://huggingface.co/api/models")

# Timeout for individual Hugging Face API requests (seconds)
HF_REQUEST_TIMEOUT = float(os.getenv("HF_REQUEST_TIMEOUT", "10.0"))

# Disable HF metadata enrichment (set to "0" or "false" to skip)
HF_METADATA_ENABLED = os.getenv("HF_METADATA_ENABLED", "1").lower() not in ("0", "false", "off")


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

    For synchronous use only.  Call ``enrich_model_metadata_from_hf()``
    afterwards (in the app lifespan) to fetch Hugging Face data.
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
            "Loaded metadata for %d models from %s: %s",
            len(MODEL_METADATA),
            os.path.basename(config_path),
            ", ".join(sorted(MODEL_METADATA.keys())),
        )
    except Exception as e:
        logger.error("Failed to load %s: %s", os.path.basename(config_path), e)
        MODEL_METADATA.clear()


async def enrich_model_metadata_from_hf(
    litellm_client: httpx.AsyncClient,
) -> None:
    """Enrich MODEL_METADATA with information from the Hugging Face API.

    For each model entry in MODEL_METADATA, this function resolves a Hugging
    Face model ID using the following priority:

      1. Explicit ``hf_id`` in config.yaml (override)
      2. Derived from LiteLLM model info (strip ``openai/`` prefix)

    It then attempts to fetch metadata from
    ``https://huggingface.co/api/models/<hf_id>``.  On success, selected
    fields (size, family, parameter_size, context_length, quantization_level,
    model_info) are merged into the existing entry, with the YAML values
    acting as fallback when the API field is missing.

    Failures (network errors, 404, timeouts, malformed responses) are logged
    at WARNING level and the model falls back to its YAML-only metadata.

    Parameters
    ----------
    litellm_client :
        The shared HTTP client configured for the LiteLLM proxy (used to
        look up model info for deriving ``hf_id``).
    """
    if not HF_METADATA_ENABLED:
        logger.info("Hugging Face metadata enrichment disabled (HF_METADATA_ENABLED=0)")
        return

    if not MODEL_METADATA:
        logger.debug("No models loaded from YAML — skipping HF enrichment")
        return

    # Step 1: Resolve hf_id for each model
    models_with_hf: Dict[str, str] = {}
    for model_name, meta in MODEL_METADATA.items():
        hf_id = await _resolve_hf_id(model_name, meta, litellm_client)
        if hf_id:
            models_with_hf[model_name] = hf_id

    if not models_with_hf:
        logger.info("No resolvable hf_id for any model — skipping HF enrichment")
        return

    logger.info(
        "Resolving metadata from Hugging Face for %d models: %s",
        len(models_with_hf),
        ", ".join(sorted(models_with_hf.keys())),
    )

    # Step 2: Fetch from Hugging Face API
    async with httpx.AsyncClient(timeout=HF_REQUEST_TIMEOUT) as hf_client:
        for model_name, hf_id in models_with_hf.items():
            await _enrich_single_model(hf_client, model_name, hf_id)


async def _resolve_hf_id(
    model_name: str,
    meta: Dict[str, Any],
    litellm_client: httpx.AsyncClient,
) -> Optional[str]:
    """Resolve the Hugging Face model ID for a given model.

    Priority:
      1. Explicit ``hf_id`` in YAML config (override)
      2. Derived from LiteLLM model info (strip ``openai/`` prefix)
    """
    # 1. Explicit override in config.yaml
    explicit_hf_id = meta.get("hf_id")
    if explicit_hf_id:
        logger.info(
            "  %s: using explicit hf_id=%r from config.yaml",
            model_name,
            explicit_hf_id,
        )
        return explicit_hf_id

    # 2. Derive from LiteLLM model info
    model_key = await _get_litellm_model_key(litellm_client, model_name)
    if model_key:
        hf_id = model_key.removeprefix("openai/")
        logger.info(
            "  %s: derived hf_id=%r from LiteLLM model_info.key=%r",
            model_name,
            hf_id,
            model_key,
        )
        return hf_id

    logger.debug("  %s: no hf_id available (skipping HF enrichment)", model_name)
    return None


async def _get_litellm_model_key(
    client: httpx.AsyncClient, model_name: str
) -> Optional[str]:
    """Look up the underlying model key from LiteLLM's model info endpoint.

    Returns the model key (e.g. ``openai/Qwen/Qwen3-Coder-480B-A35B-Instruct``)
    or ``None`` on failure.
    """
    try:
        response = await client.get("/v1/model/info", params={"model": model_name})
    except (httpx.NetworkError, httpx.TimeoutException) as e:
        logger.debug("LiteLLM model info lookup failed for %s: %s", model_name, e)
        return None

    if response.status_code != 200:
        logger.debug(
            "LiteLLM model info returned %d for %s", response.status_code, model_name
        )
        return None

    try:
        data = response.json()
    except Exception:
        return None

    # LiteLLM returns {"data": [{"model_info": {"key": "openai/..."}}]}
    # or {"data": [{"key": "openai/..."}]}
    items = data.get("data", [])
    if items and isinstance(items[0], dict):
        entry = items[0]
        model_info = entry.get("model_info", {})
        if isinstance(model_info, dict):
            key = model_info.get("key")
            if key:
                return str(key)
        # Fallback: key might be at the top level of the entry
        key = entry.get("key")
        if key:
            return str(key)

    return None


async def _enrich_single_model(
    client: httpx.AsyncClient, model_name: str, hf_id: str
) -> None:
    """Fetch and merge metadata for a single model from Hugging Face."""
    url = f"{HF_API_BASE}/{hf_id}"
    try:
        response = await client.get(url)
    except httpx.NetworkError as e:
        logger.warning(
            "HF network error for %s (hf_id=%s): %s — falling back to YAML",
            model_name,
            hf_id,
            e,
        )
        return
    except httpx.TimeoutException as e:
        logger.warning(
            "HF timeout for %s (hf_id=%s): %s — falling back to YAML",
            model_name,
            hf_id,
            e,
        )
        return

    if response.status_code == 404:
        logger.warning(
            "HF model not found for %s (hf_id=%s) — falling back to YAML",
            model_name,
            hf_id,
        )
        return
    if response.status_code != 200:
        logger.warning(
            "HF API returned %d for %s (hf_id=%s) — falling back to YAML",
            response.status_code,
            model_name,
            hf_id,
        )
        return

    try:
        hf_data = response.json()
    except Exception as e:
        logger.warning(
            "Failed to parse HF response for %s (hf_id=%s): %s — falling back to YAML",
            model_name,
            hf_id,
            e,
        )
        return

    meta = MODEL_METADATA[model_name]

    # Map HF fields into our metadata structure
    hf_size = hf_data.get("size")
    hf_pipeline_tag = hf_data.get("pipeline_tag")
    hf_config = hf_data.get("config", {}) or {}
    hf_tags = hf_data.get("tags", [])
    hf_card_data = hf_data.get("cardData", {}) or {}

    # --- size (model weight file size in bytes) ---
    if hf_size is not None:
        meta.setdefault("size", hf_size)
        logger.debug("  %s: size <- %d (HF)", model_name, hf_size)

    # --- family (inferred from pipeline_tag or tags) ---
    if "family" not in meta or not meta["family"]:
        family = _infer_family_from_hf(hf_pipeline_tag, hf_tags)
        if family:
            meta["family"] = family
            logger.debug("  %s: family <- %r (HF)", model_name, family)

    # --- parameter_size (from cardData or config) ---
    if "parameter_size" not in meta or not meta["parameter_size"]:
        param_size = _infer_parameter_size(hf_card_data, hf_config)
        if param_size:
            meta["parameter_size"] = param_size
            logger.debug("  %s: parameter_size <- %r (HF)", model_name, param_size)

    # --- context_length ---
    if "context_length" not in meta or not meta["context_length"]:
        ctx_len = _infer_context_length(hf_config)
        if ctx_len is not None:
            meta["context_length"] = ctx_len
            logger.debug("  %s: context_length <- %d (HF)", model_name, ctx_len)

    # --- quantization_level (from tags) ---
    if "quantization_level" not in meta or not meta["quantization_level"]:
        quant = _infer_quantization(hf_tags)
        if quant:
            meta["quantization_level"] = quant
            logger.debug("  %s: quantization_level <- %r (HF)", model_name, quant)

    # --- model_info (rich key-value from HF) ---
    if "model_info" not in meta or not meta["model_info"]:
        model_info = _build_model_info(hf_data, hf_config, hf_card_data)
        if model_info:
            meta["model_info"] = model_info

    logger.info("Enriched metadata for %s from Hugging Face (hf_id=%s)", model_name, hf_id)


def _infer_family_from_hf(
    pipeline_tag: Optional[str], tags: List[str]
) -> Optional[str]:
    """Infer model family from Hugging Face pipeline_tag or tags."""
    if pipeline_tag:
        tag_lower = pipeline_tag.lower()
        if "text-generation" in tag_lower:
            # Try to infer from tags
            tags_lower = [t.lower() for t in tags]
            for keyword, family in [
                ("llama", "llama"),
                ("mistral", "mistral"),
                ("qwen", "qwen2"),
                ("gemma", "gemma2"),
                ("phi", "phi"),
            ]:
                if any(keyword in t for t in tags_lower):
                    return family
            return "unknown"
    return None


def _infer_parameter_size(
    card_data: Dict[str, Any], config: Dict[str, Any]
) -> Optional[str]:
    """Infer parameter size string from Hugging Face cardData or config."""
    # Try cardData first (model_name, library_name often hint size)
    param_count = card_data.get("tokenizer_config", {}).get("max_position_embeddings")
    if param_count:
        return _format_param_size(param_count)

    # Try config.torch_dtype or config.vocab_size as hints
    torch_dtype = config.get("torch_dtype", "")
    if torch_dtype:
        return torch_dtype.replace("torch.", "")

    return None


def _format_param_size(count: int) -> str:
    """Format a parameter count into a human-readable string like '7B'."""
    if count >= 1_000_000_000_000:
        return f"{count // 1_000_000_000_000}T"
    if count >= 1_000_000_000:
        return f"{count // 1_000_000_000}B"
    if count >= 1_000_000:
        return f"{count // 1_000_000}M"
    return str(count)


def _infer_context_length(config: Dict[str, Any]) -> Optional[int]:
    """Infer context length from Hugging Face model config."""
    # Common keys across architectures
    for key in (
        "max_position_embeddings",
        "context_length",
        "n_ctx",
        "max_sequence_length",
    ):
        val = config.get(key)
        if val and isinstance(val, int):
            return val
    return None


def _infer_quantization(tags: List[str]) -> Optional[str]:
    """Infer quantization level from Hugging Face tags."""
    tags_lower = [t.lower() for t in tags]
    for quant in (
        "fp8", "fp4", "nf4",
        "q8_0", "q5_0", "q5_1", "q4_0", "q4_1", "q3_k", "q2_k",
        "int8", "int4",
    ):
        if any(quant in t for t in tags_lower):
            return quant.upper()
    return None


def _build_model_info(
    hf_data: Dict[str, Any],
    config: Dict[str, Any],
    card_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a model_info dict from Hugging Face response data."""
    info: Dict[str, Any] = {}

    # Basic model info
    if hf_data.get("id"):
        info["general.name"] = hf_data["id"]
    if hf_data.get("likes"):
        info["general.likes"] = hf_data["likes"]
    if hf_data.get("downloads"):
        info["general.downloads"] = hf_data["downloads"]
    if hf_data.get("pipeline_tag"):
        info["general.pipeline_tag"] = hf_data["pipeline_tag"]
    if hf_data.get("size"):
        info["general.size"] = hf_data["size"]

    # Config details
    for key in ("vocab_size", "hidden_size", "intermediate_size",
                 "num_hidden_layers", "num_attention_heads",
                 "max_position_embeddings", "torch_dtype"):
        val = config.get(key)
        if val is not None:
            info[f"config.{key}"] = val

    return info if info else None


def setup_logging() -> None:
    """Configure application-wide logging."""
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


async def initialize_model_metadata(
    litellm_client: httpx.AsyncClient,
) -> None:
    """Load model metadata from YAML and enrich with Hugging Face API data.

    This is the main entry point called from the app lifespan handler.
    It first loads the YAML configuration, then attempts to enrich each
    model with data from Hugging Face.

    Parameters
    ----------
    litellm_client :
        The shared HTTP client configured for the LiteLLM proxy (used to
        look up model info for deriving ``hf_id``).
    """
    load_model_metadata()
    await enrich_model_metadata_from_hf(litellm_client)
