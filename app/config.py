"""
Configuration for the Ollama API proxy.

All settings are loaded from environment variables or a .env file.
Model metadata is loaded from config.yaml at startup, then optionally
enriched with information from the Hugging Face API.
"""

import asyncio
import os
import logging
import re
from typing import Any, Dict, List, Optional

import httpx
import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Hugging Face API base URL
HF_API_BASE = os.getenv("HF_API_BASE", "https://huggingface.co/api/models")
# Raw file base (for fetching tokenizer_config.json etc.)
HF_RAW_BASE = os.getenv("HF_RAW_BASE", "https://huggingface.co")

# Timeout for individual Hugging Face API requests (seconds)
HF_REQUEST_TIMEOUT = float(os.getenv("HF_REQUEST_TIMEOUT", "10.0"))

# Disable HF metadata enrichment (set to "0" or "false" to skip)
HF_METADATA_ENABLED = os.getenv("HF_METADATA_ENABLED", "1").lower() not in ("0", "false", "off")

# Known model family patterns for inference from HF model ID / tags
_FAMILY_PATTERNS: List[tuple[str, str]] = [
    (r"qwen", "qwen2"),
    (r"llama", "llama"),
    (r"mistral|mixtral", "mistral"),
    (r"gemma", "gemma2"),
    (r"phi", "phi"),
    (r"deepseek", "deepseek"),
    (r"command", "command-r"),
    (r"falcon", "falcon"),
    (r"yi\b", "yi"),
    (r"internlm", "internlm"),
]

# Known stop tokens by family (fallback when tokenizer_config lacks them)
_FAMILY_STOP_TOKENS: Dict[str, List[str]] = {
    "qwen2": ["<|im_start|>", "<|im_end|>"],
    "llama": ["<|eot_id|>", "<|end_of_text|>"],
    "mistral": ["</s>", "[/INST]"],
    "gemma2": ["<end_of_turn>", "<eos>"],
    "phi": ["<|end|>"],
    "deepseek": ["<|end_of_sentence|>"],
    "command-r": ["<|END_OF_TURN_TOKEN|>"],
}


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
    Face model ID and fetches:

      - Main model metadata (``/api/models/<hf_id>``)
      - ``tokenizer_config.json`` for chat_template and stop tokens
      - ``config.json`` for architecture details and context length

    Failures are logged at WARNING level; models fall back to YAML-only metadata.
    """
    if not HF_METADATA_ENABLED:
        logger.info("Hugging Face metadata enrichment disabled (HF_METADATA_ENABLED=0)")
        return

    if not MODEL_METADATA:
        logger.debug("No models loaded from YAML — skipping HF enrichment")
        return

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

    async with httpx.AsyncClient(timeout=HF_REQUEST_TIMEOUT) as hf_client:
        tasks = [
            _enrich_single_model(hf_client, model_name, hf_id)
            for model_name, hf_id in models_with_hf.items()
        ]
        if tasks:
            await asyncio.gather(*tasks)


async def _resolve_hf_id(
    model_name: str,
    meta: Dict[str, Any],
    litellm_client: httpx.AsyncClient,
) -> Optional[str]:
    """Resolve the Hugging Face model ID for a given model."""
    explicit_hf_id = meta.get("hf_id")
    if explicit_hf_id:
        logger.info(
            "  %s: using explicit hf_id=%r from config.yaml",
            model_name,
            explicit_hf_id,
        )
        return explicit_hf_id

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
    """Look up the underlying model key from LiteLLM's model info endpoint."""
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

    items = data.get("data", [])
    if items and isinstance(items[0], dict):
        entry = items[0]
        model_info = entry.get("model_info", {})
        if isinstance(model_info, dict):
            key = model_info.get("key")
            if key:
                return str(key)
        key = entry.get("key")
        if key:
            return str(key)

    return None


async def _fetch_hf_json(
    client: httpx.AsyncClient, url: str, label: str
) -> Optional[Dict[str, Any]]:
    """Fetch a JSON resource from Hugging Face, returning None on any failure."""
    try:
        response = await client.get(url)
    except (httpx.NetworkError, httpx.TimeoutException) as e:
        logger.debug("HF fetch failed for %s: %s", label, e)
        return None

    if response.status_code != 200:
        logger.debug("HF fetch returned %d for %s", response.status_code, label)
        return None

    try:
        return response.json()
    except Exception as e:
        logger.debug("Failed to parse HF JSON for %s: %s", label, e)
        return None


async def _enrich_single_model(
    client: httpx.AsyncClient, model_name: str, hf_id: str
) -> None:
    """Fetch and merge metadata for a single model from Hugging Face.

    Fetches three sources:
      1. /api/models/<hf_id>         — tags, pipeline_tag, siblings list, size
      2. resolve/main/config.json    — architecture, context_length, vocab_size
      3. resolve/main/tokenizer_config.json — chat_template, stop tokens, eos_token
    """
    # 1. Main model API
    api_url = f"{HF_API_BASE}/{hf_id}"
    hf_data = await _fetch_hf_json(client, api_url, f"{model_name}/api")
    if hf_data is None:
        logger.warning(
            "HF API fetch failed for %s (hf_id=%s) — falling back to YAML",
            model_name,
            hf_id,
        )
        return

    # 2. config.json (architecture details)
    config_url = f"{HF_RAW_BASE}/{hf_id}/resolve/main/config.json"
    hf_config = await _fetch_hf_json(client, config_url, f"{model_name}/config.json") or {}

    # 3. tokenizer_config.json (chat template, stop tokens)
    tok_url = f"{HF_RAW_BASE}/{hf_id}/resolve/main/tokenizer_config.json"
    hf_tok_config = await _fetch_hf_json(client, tok_url, f"{model_name}/tokenizer_config.json") or {}

    meta = MODEL_METADATA[model_name]
    hf_tags = hf_data.get("tags", []) or []
    hf_card_data = hf_data.get("cardData", {}) or {}

    # --- size ---
    hf_size = hf_data.get("size")
    if hf_size is not None:
        meta.setdefault("size", hf_size)

    # --- family ---
    if not meta.get("family"):
        family = _infer_family(hf_id, hf_data.get("pipeline_tag"), hf_tags)
        if family:
            meta["family"] = family
            logger.debug("  %s: family <- %r", model_name, family)

    # --- parameter_size ---
    if not meta.get("parameter_size"):
        param_size = _infer_parameter_size(hf_id, hf_card_data, hf_config)
        if param_size:
            meta["parameter_size"] = param_size
            logger.debug("  %s: parameter_size <- %r", model_name, param_size)

    # --- context_length ---
    if not meta.get("context_length"):
        ctx_len = _infer_context_length(hf_config, hf_tok_config)
        if ctx_len is not None:
            meta["context_length"] = ctx_len
            logger.debug("  %s: context_length <- %d", model_name, ctx_len)

    # --- quantization_level ---
    if not meta.get("quantization_level"):
        quant = _infer_quantization(hf_tags)
        if quant:
            meta["quantization_level"] = quant
            logger.debug("  %s: quantization_level <- %r", model_name, quant)

    # --- chat template (from tokenizer_config.json) ---
    if not meta.get("template"):
        template = _extract_chat_template(hf_tok_config)
        if template:
            meta["template"] = template
            logger.debug("  %s: template extracted from tokenizer_config.json", model_name)

    # --- stop tokens ---
    if not meta.get("stop_tokens"):
        stop_tokens = _extract_stop_tokens(hf_tok_config, meta.get("family"))
        if stop_tokens:
            meta["stop_tokens"] = stop_tokens
            logger.debug("  %s: stop_tokens <- %r", model_name, stop_tokens)

    # --- model_info (rich key-value for api/show) ---
    if not meta.get("model_info"):
        model_info = _build_model_info(hf_data, hf_config, hf_tok_config, hf_card_data)
        if model_info:
            meta["model_info"] = model_info

    logger.info("Enriched metadata for %s from Hugging Face (hf_id=%s)", model_name, hf_id)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _infer_family(
    hf_id: str,
    pipeline_tag: Optional[str],
    tags: List[str],
) -> Optional[str]:
    """Infer model family from hf_id, pipeline_tag, or tags."""
    search_text = " ".join([hf_id.lower()] + [t.lower() for t in tags])
    for pattern, family in _FAMILY_PATTERNS:
        if re.search(pattern, search_text):
            return family
    return None


def _infer_parameter_size(
    hf_id: str,
    card_data: Dict[str, Any],
    config: Dict[str, Any],
) -> Optional[str]:
    """Infer parameter size from hf_id pattern, cardData, or config."""
    # Most HF model IDs encode size directly, e.g. "Qwen3-Coder-480B-A35B" or "27B"
    # Match patterns like 480B, 27B, 7B, 235B, 1.5B, 0.5B, 72B
    # If there's an A-suffix (MoE active params), prefer the total
    # e.g. "480B-A35B" → "480B"
    total = re.search(r"(\d+(?:\.\d+)?)[Bb]-[Aa]\d", hf_id)
    if total:
        return total.group(1) + "B"
    match = re.search(r"(\d+(?:\.\d+)?)[Bb]", hf_id)
    if match:
        return match.group(1) + "B"

    # cardData model_name sometimes has it
    card_name = card_data.get("model_name", "")
    match = re.search(r"(\d+(?:\.\d+)?)[Bb]", str(card_name))
    if match:
        return match.group(0).upper()

    # Fallback: derive from num_parameters in config
    num_params = config.get("num_parameters")
    if num_params and isinstance(num_params, int):
        # For MoE models, report active parameters instead of total.
        # num_experts * num_experts_active gives the sparsity factor.
        num_experts = config.get("num_experts")
        num_experts_active = config.get("num_experts_active")
        if num_experts and num_experts_active:
            # Active params ≈ total / num_experts * num_experts_active
            active = int(num_params / num_experts * num_experts_active)
            return _format_param_size(active)
        return _format_param_size(num_params)

    return None


def _format_param_size(count: int) -> str:
    """Format a parameter count into a human-readable string like '7B'."""
    if count >= 1_000_000_000_000:
        return f"{count / 1_000_000_000_000:.0f}T"
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.0f}B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.0f}M"
    return str(count)


def _infer_context_length(
    config: Dict[str, Any],
    tok_config: Dict[str, Any],
) -> Optional[int]:
    """Infer context length from config.json or tokenizer_config.json."""
    for key in (
        "max_position_embeddings",
        "context_length",
        "n_ctx",
        "max_sequence_length",
        "seq_length",
    ):
        val = config.get(key)
        if val and isinstance(val, int) and val > 512:
            return val

    # tokenizer_config sometimes has model_max_length
    val = tok_config.get("model_max_length")
    if val and isinstance(val, int) and val > 512 and val < 10_000_000:
        return val

    return None


def _infer_quantization(tags: List[str]) -> Optional[str]:
    """Infer quantization level from Hugging Face tags."""
    tags_lower = [t.lower() for t in tags]
    for quant in (
        "fp8", "fp4", "nf4",
        "q8_0", "q5_0", "q5_1", "q4_0", "q4_1", "q3_k", "q2_k",
        "int8", "int4", "awq", "gptq",
    ):
        if any(quant in t for t in tags_lower):
            return quant.upper()
    return None


def _extract_chat_template(tok_config: Dict[str, Any]) -> Optional[str]:
    """Extract and convert the Jinja chat template into an Ollama-style Go template.

    tokenizer_config.json stores the template as a Jinja2 string (or a dict of
    named templates).  Ollama uses Go templates, but most clients that read
    api/show just want to see *something* here for format detection — the actual
    rendering is done server-side by vLLM/LiteLLM.  We store the raw Jinja
    string under ``jinja_template`` and derive a simplified Go template for
    ``template`` that correctly reflects the token format (ChatML vs Llama3 etc.).
    """
    raw = tok_config.get("chat_template")
    if not raw:
        return None

    # If it's a dict of named templates, prefer "default" or the first entry
    if isinstance(raw, dict):
        raw = raw.get("default") or next(iter(raw.values()), None)
    if not raw or not isinstance(raw, str):
        return None

    # Detect format from the Jinja template content and map to a Go template.
    if "▌" in raw:
        # ChatML format (Qwen, many others)
        return (
            "{{ if .System }}▌system\n{{ .System }}\n▌user\n{{ end }}"
            "{{ range .Messages }}▌{{ .Role }}\n{{ .Content }}\n▌assistant\n{{ end }}"
        )
    if "<|start_header_id|>" in raw:
        # Llama-3 format
        return (
            "{{ if .System }}<|start_header_id|>system<|end_header_id|>\n\n"
            "{{ .System }}<|eot_id|>{{ end }}"
            "{{ range .Messages }}<|start_header_id|>{{ .Role }}<|end_header_id|>\n\n"
            "{{ .Content }}<|eot_id|>{{ end }}"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    if "[INST]" in raw:
        # Mistral/Llama-2 format
        return (
            "{{ if .System }}[INST] {{ .System }} [/INST]\n{{ end }}"
            "{{ range .Messages }}{{ if eq .Role \"user\" }}[INST] {{ .Content }} [/INST]\n"
            "{{ else }}{{ .Content }}\n{{ end }}{{ end }}"
        )
    if "<|user|>" in raw:
        # Phi-3 / Zephyr format
        return (
            "{{ if .System }}<|system|>\n{{ .System }}<|end|>\n{{ end }}"
            "{{ range .Messages }}<|{{ .Role }}|>\n{{ .Content }}<|end|>\n{{ end }}"
            "<|assistant|>\n"
        )
    if "<start_of_turn>" in raw:
        # Gemma format
        return (
            "{{ range .Messages }}<start_of_turn>{{ .Role }}\n"
            "{{ .Content }}<end_of_turn>\n{{ end }}"
            "<start_of_turn>model\n"
        )

    # Unknown format — return a generic placeholder so callers know a template exists
    return "{{ range .Messages }}{{ .Role }}: {{ .Content }}\n{{ end }}"


def _extract_stop_tokens(
    tok_config: Dict[str, Any],
    family: Optional[str],
) -> List[str]:
    """Extract stop/EOS tokens from tokenizer_config.json.

    Checks eos_token, added_tokens_decoder for common stop token patterns,
    and falls back to known per-family defaults.
    """
    stop: List[str] = []

    eos = tok_config.get("eos_token")
    if isinstance(eos, str) and eos:
        stop.append(eos)
    elif isinstance(eos, dict):
        content = eos.get("content")
        if content:
            stop.append(content)

    # additional_special_tokens sometimes lists stop tokens explicitly
    for tok in tok_config.get("additional_special_tokens", []):
        if isinstance(tok, str) and any(
            kw in tok.lower()
            for kw in ("end", "eos", "stop", "eot", "im_end", "eot_id")
        ):
            if tok not in stop:
                stop.append(tok)

    # Fallback to family defaults if we found nothing useful
    if not stop and family:
        stop = _FAMILY_STOP_TOKENS.get(family, [])

    return stop


def _build_model_info(
    hf_data: Dict[str, Any],
    config: Dict[str, Any],
    tok_config: Dict[str, Any],
    card_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a model_info dict from all available Hugging Face data.

    Uses architecture-namespaced keys mirroring the GGUF convention that
    Ollama's api/show response uses (e.g. ``qwen2.context_length``).
    """
    info: Dict[str, Any] = {}

    # Determine architecture prefix from config.json
    arch = config.get("model_type", "")
    # Normalise: "qwen2_moe" -> "qwen2", "llama" -> "llama" etc.
    arch_prefix = re.sub(r"[_-]moe$|[_-]vl$|[_-]audio$", "", arch).lower() if arch else ""

    # General fields
    if hf_data.get("id"):
        info["general.name"] = hf_data["id"]
    if hf_data.get("pipeline_tag"):
        info["general.pipeline_tag"] = hf_data["pipeline_tag"]

    # Architecture-namespaced context / embedding dims
    ctx_len = _infer_context_length(config, tok_config)
    if ctx_len and arch_prefix:
        info[f"{arch_prefix}.context_length"] = ctx_len

    for src_key, dst_suffix in (
        ("hidden_size", "embedding_length"),
        ("num_hidden_layers", "block_count"),
        ("num_attention_heads", "attention.head_count"),
        ("num_key_value_heads", "attention.head_count_kv"),
        ("intermediate_size", "feed_forward_length"),
        ("vocab_size", "vocab_size"),
    ):
        val = config.get(src_key)
        if val is not None and arch_prefix:
            info[f"{arch_prefix}.{dst_suffix}"] = val
        elif val is not None:
            info[f"config.{src_key}"] = val

    # Parameter count
    num_params = config.get("num_parameters")
    if num_params:
        info["general.parameter_count"] = num_params

    # Torch dtype
    dtype = config.get("torch_dtype")
    if dtype:
        info["general.file_type"] = dtype

    # Raw Jinja template stored for reference (not used by Ollama directly)
    raw_template = tok_config.get("chat_template")
    if isinstance(raw_template, str) and raw_template:
        info["tokenizer.chat_template"] = raw_template
    elif isinstance(raw_template, dict):
        # Store the default variant
        default = raw_template.get("default") or next(iter(raw_template.values()), None)
        if default:
            info["tokenizer.chat_template"] = default

    # EOS / BOS tokens
    for key in ("eos_token", "bos_token", "pad_token"):
        val = tok_config.get(key)
        if isinstance(val, str) and val:
            info[f"tokenizer.{key}"] = val
        elif isinstance(val, dict):
            content = val.get("content")
            if content:
                info[f"tokenizer.{key}"] = content

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
    """Load model metadata from YAML and enrich with Hugging Face API data."""
    load_model_metadata()
    await enrich_model_metadata_from_hf(litellm_client)
