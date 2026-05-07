"""
Tests for Hugging Face metadata enrichment in app/config.py.
"""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure HF enrichment is disabled by default in tests
os.environ["HF_METADATA_ENABLED"] = "0"

from app.config import (
    MODEL_METADATA,
    load_model_metadata,
    enrich_model_metadata_from_hf,
    _enrich_single_model,
    _resolve_hf_id,
    _get_litellm_model_key,
    _infer_family_from_hf,
    _infer_context_length,
    _infer_quantization,
    _build_model_info,
    HF_METADATA_ENABLED,
)


class TestHFEnrichmentDisabled:
    """Test that HF enrichment is properly disabled in test environment."""

    def test_hf_metadata_disabled_in_tests(self):
        """HF_METADATA_ENABLED should be False when env var is '0'."""
        assert HF_METADATA_ENABLED is False

    @pytest.mark.asyncio
    async def test_enrich_noop_when_disabled(self):
        """enrich_model_metadata_from_hf should return immediately when disabled."""
        MODEL_METADATA.clear()
        MODEL_METADATA.update({"test-model": {"family": "llama", "hf_id": "test/model"}})
        # Should complete without any network calls
        mock_litellm_client = AsyncMock()
        await enrich_model_metadata_from_hf(mock_litellm_client)
        # Metadata should be unchanged
        assert MODEL_METADATA["test-model"]["family"] == "llama"
        MODEL_METADATA.clear()


class TestInferFamilyFromHF:
    """Test _infer_family_from_hf helper."""

    def test_llama_family(self):
        result = _infer_family_from_hf("text-generation", ["llama", "llama-3"])
        assert result == "llama"

    def test_mistral_family(self):
        result = _infer_family_from_hf("text-generation", ["mistral", "mistral-large"])
        assert result == "mistral"

    def test_qwen_family(self):
        result = _infer_family_from_hf("text-generation", ["qwen", "qwen2"])
        assert result == "qwen2"

    def test_gemma_family(self):
        result = _infer_family_from_hf("text-generation", ["gemma", "gemma-2"])
        assert result == "gemma2"

    def test_phi_family(self):
        result = _infer_family_from_hf("text-generation", ["phi", "phi-3"])
        assert result == "phi"

    def test_unknown_family(self):
        result = _infer_family_from_hf("text-generation", ["some-model"])
        assert result == "unknown"

    def test_non_text_generation_pipeline(self):
        result = _infer_family_from_hf("text-classification", ["llama"])
        assert result is None

    def test_no_pipeline_tag(self):
        result = _infer_family_from_hf(None, ["llama"])
        assert result is None


class TestInferContextLength:
    """Test _infer_context_length helper."""

    def test_max_position_embeddings(self):
        config = {"max_position_embeddings": 4096}
        assert _infer_context_length(config) == 4096

    def test_context_length_key(self):
        config = {"context_length": 8192}
        assert _infer_context_length(config) == 8192

    def test_n_ctx_key(self):
        config = {"n_ctx": 2048}
        assert _infer_context_length(config) == 2048

    def test_max_sequence_length_key(self):
        config = {"max_sequence_length": 16384}
        assert _infer_context_length(config) == 16384

    def test_no_matching_key(self):
        config = {"other_key": 123}
        assert _infer_context_length(config) is None

    def test_empty_config(self):
        assert _infer_context_length({}) is None

    def test_non_int_value(self):
        config = {"max_position_embeddings": "4096"}
        assert _infer_context_length(config) is None


class TestInferQuantization:
    """Test _infer_quantization helper."""

    def test_fp8(self):
        assert _infer_quantization(["fp8", "text-generation"]) == "FP8"

    def test_fp4(self):
        assert _infer_quantization(["fp4"]) == "FP4"

    def test_nf4(self):
        assert _infer_quantization(["bitsandbytes-nf4"]) == "NF4"

    def test_q4_0(self):
        assert _infer_quantization(["q4_0", "gguf"]) == "Q4_0"

    def test_int8(self):
        assert _infer_quantization(["int8"]) == "INT8"

    def test_no_quantization_tag(self):
        assert _infer_quantization(["text-generation", "pytorch"]) is None

    def test_empty_tags(self):
        assert _infer_quantization([]) is None


class TestBuildModelInfo:
    """Test _build_model_info helper."""

    def test_basic_info(self):
        hf_data = {"id": "test/model", "likes": 100, "downloads": 5000,
                    "pipeline_tag": "text-generation", "size": 1000000}
        config = {"vocab_size": 32000, "hidden_size": 4096}
        card_data = {}
        result = _build_model_info(hf_data, config, card_data)
        assert result["general.name"] == "test/model"
        assert result["general.likes"] == 100
        assert result["general.downloads"] == 5000
        assert result["general.pipeline_tag"] == "text-generation"
        assert result["general.size"] == 1000000
        assert result["config.vocab_size"] == 32000
        assert result["config.hidden_size"] == 4096

    def test_empty_data(self):
        result = _build_model_info({}, {}, {})
        assert result is None

    def test_partial_data(self):
        hf_data = {"id": "test/model"}
        config = {}
        card_data = {}
        result = _build_model_info(hf_data, config, card_data)
        assert result["general.name"] == "test/model"
        assert "general.likes" not in result


class TestEnrichSingleModel:
    """Test _enrich_single_model with mocked HTTP responses."""

    @pytest.mark.asyncio
    async def test_successful_enrichment(self):
        MODEL_METADATA.clear()
        MODEL_METADATA["test-model"] = {
            "family": "llama",
            "hf_id": "test/model",
            "format": "gguf",
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test/model",
            "size": 5000000000,
            "pipeline_tag": "text-generation",
            "tags": ["llama", "fp8"],
            "config": {
                "max_position_embeddings": 8192,
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
            },
            "cardData": {},
            "likes": 50,
            "downloads": 1000,
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        await _enrich_single_model(mock_client, "test-model", "test/model")

        # YAML values should be preserved (setdefault behavior)
        assert MODEL_METADATA["test-model"]["family"] == "llama"
        assert MODEL_METADATA["test-model"]["format"] == "gguf"
        # HF values should be merged in
        assert MODEL_METADATA["test-model"]["size"] == 5000000000
        assert MODEL_METADATA["test-model"]["context_length"] == 8192
        assert MODEL_METADATA["test-model"]["quantization_level"] == "FP8"
        assert "model_info" in MODEL_METADATA["test-model"]
        MODEL_METADATA.clear()

    @pytest.mark.asyncio
    async def test_404_fallback(self):
        MODEL_METADATA.clear()
        MODEL_METADATA["test-model"] = {
            "family": "llama",
            "hf_id": "nonexistent/model",
            "format": "gguf",
        }

        mock_response = MagicMock()
        mock_response.status_code = 404

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        await _enrich_single_model(mock_client, "test-model", "nonexistent/model")

        # Should fall back to YAML values unchanged
        assert MODEL_METADATA["test-model"]["family"] == "llama"
        assert "size" not in MODEL_METADATA["test-model"]
        MODEL_METADATA.clear()

    @pytest.mark.asyncio
    async def test_network_error_fallback(self):
        MODEL_METADATA.clear()
        MODEL_METADATA["test-model"] = {
            "family": "llama",
            "hf_id": "test/model",
            "format": "gguf",
        }

        import httpx
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.NetworkError("Connection refused"))

        await _enrich_single_model(mock_client, "test-model", "test/model")

        # Should fall back to YAML values unchanged
        assert MODEL_METADATA["test-model"]["family"] == "llama"
        assert "size" not in MODEL_METADATA["test-model"]
        MODEL_METADATA.clear()

    @pytest.mark.asyncio
    async def test_timeout_fallback(self):
        MODEL_METADATA.clear()
        MODEL_METADATA["test-model"] = {
            "family": "llama",
            "hf_id": "test/model",
            "format": "gguf",
        }

        import httpx
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))

        await _enrich_single_model(mock_client, "test-model", "test/model")

        assert MODEL_METADATA["test-model"]["family"] == "llama"
        assert "size" not in MODEL_METADATA["test-model"]
        MODEL_METADATA.clear()

    @pytest.mark.asyncio
    async def test_yaml_values_preserved_over_hf(self):
        """YAML values should take precedence when already set (setdefault)."""
        MODEL_METADATA.clear()
        MODEL_METADATA["test-model"] = {
            "family": "llama",
            "parameter_size": "70B",
            "quantization_level": "Q4_0",
            "context_length": 4096,
            "size": 1234567890,
            "hf_id": "test/model",
            "format": "gguf",
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test/model",
            "size": 9999999999,
            "pipeline_tag": "text-generation",
            "tags": ["mistral", "fp8"],
            "config": {
                "max_position_embeddings": 32768,
            },
            "cardData": {},
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        await _enrich_single_model(mock_client, "test-model", "test/model")

        # All YAML values should be preserved
        assert MODEL_METADATA["test-model"]["family"] == "llama"
        assert MODEL_METADATA["test-model"]["parameter_size"] == "70B"
        assert MODEL_METADATA["test-model"]["quantization_level"] == "Q4_0"
        assert MODEL_METADATA["test-model"]["context_length"] == 4096
        assert MODEL_METADATA["test-model"]["size"] == 1234567890
        MODEL_METADATA.clear()

    @pytest.mark.asyncio
    async def test_hf_fills_missing_fields(self):
        """HF should fill in fields that are missing from YAML."""
        MODEL_METADATA.clear()
        MODEL_METADATA["test-model"] = {
            "family": "",  # Empty family
            "parameter_size": "",  # Empty parameter_size
            "context_length": None,  # None context_length
            "quantization_level": "",  # Empty quantization
            "model_info": {},  # Empty model_info
            "hf_id": "test/model",
            "format": "gguf",
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test/model",
            "size": 5000000000,
            "pipeline_tag": "text-generation",
            "tags": ["qwen", "fp8"],
            "config": {
                "max_position_embeddings": 131072,
                "hidden_size": 5120,
                "num_hidden_layers": 60,
                "num_attention_heads": 40,
            },
            "cardData": {},
            "likes": 200,
            "downloads": 5000,
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        await _enrich_single_model(mock_client, "test-model", "test/model")

        # HF should fill in the empty/None fields
        assert MODEL_METADATA["test-model"]["family"] == "qwen2"
        assert MODEL_METADATA["test-model"]["context_length"] == 131072
        assert MODEL_METADATA["test-model"]["quantization_level"] == "FP8"
        assert MODEL_METADATA["test-model"]["size"] == 5000000000
        assert len(MODEL_METADATA["test-model"]["model_info"]) > 0
        MODEL_METADATA.clear()


class TestEnrichModelMetadataFromHF:
    """Test the top-level enrich_model_metadata_from_hf function."""

    @pytest.mark.asyncio
    async def test_skips_when_no_metadata(self):
        MODEL_METADATA.clear()
        # Temporarily enable HF enrichment
        import app.config
        original = app.config.HF_METADATA_ENABLED
        app.config.HF_METADATA_ENABLED = True
        try:
            mock_litellm_client = AsyncMock()
            await enrich_model_metadata_from_hf(mock_litellm_client)
        finally:
            app.config.HF_METADATA_ENABLED = original
        assert len(MODEL_METADATA) == 0

    @pytest.mark.asyncio
    async def test_skips_when_no_hf_id(self):
        """When no explicit hf_id and LiteLLM lookup fails, skip enrichment."""
        MODEL_METADATA.clear()
        MODEL_METADATA["test-model"] = {"family": "llama", "format": "gguf"}
        import app.config
        original = app.config.HF_METADATA_ENABLED
        app.config.HF_METADATA_ENABLED = True
        try:
            mock_litellm_client = AsyncMock()
            # LiteLLM lookup fails, so no hf_id is resolved
            mock_litellm_client.get = AsyncMock(return_value=MagicMock(status_code=404))
            await enrich_model_metadata_from_hf(mock_litellm_client)
        finally:
            app.config.HF_METADATA_ENABLED = original
        # Should not have added any fields since HF enrichment was skipped
        assert "size" not in MODEL_METADATA["test-model"]
        MODEL_METADATA.clear()

    @pytest.mark.asyncio
    async def test_enriches_models_with_hf_id(self):
        MODEL_METADATA.clear()
        MODEL_METADATA["test-model"] = {
            "family": "llama",
            "hf_id": "test/model",
            "format": "gguf",
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test/model",
            "size": 5000000000,
            "pipeline_tag": "text-generation",
            "tags": ["llama"],
            "config": {"max_position_embeddings": 8192},
            "cardData": {},
        }

        import app.config
        original = app.config.HF_METADATA_ENABLED
        app.config.HF_METADATA_ENABLED = True

        with patch("app.config.httpx") as mock_httpx:
            mock_hf_client = AsyncMock()
            mock_hf_client.get = AsyncMock(return_value=mock_response)
            mock_hf_client.__aenter__ = AsyncMock(return_value=mock_hf_client)
            mock_hf_client.__aexit__ = AsyncMock(return_value=None)
            mock_httpx.AsyncClient.return_value = mock_hf_client

            mock_litellm_client = AsyncMock()
            mock_litellm_client.get = AsyncMock(
                side_effect=Exception("should not be called — explicit hf_id")
            )

            try:
                await enrich_model_metadata_from_hf(mock_litellm_client)
            finally:
                app.config.HF_METADATA_ENABLED = original

        assert MODEL_METADATA["test-model"]["size"] == 5000000000
        assert MODEL_METADATA["test-model"]["context_length"] == 8192
        MODEL_METADATA.clear()


class TestResolveHFId:
    """Test _resolve_hf_id helper."""

    @pytest.mark.asyncio
    async def test_explicit_hf_id_takes_precedence(self):
        """Explicit hf_id in YAML should skip LiteLLM lookup."""
        meta = {"hf_id": "explicit/model", "family": "llama"}
        mock_litellm_client = AsyncMock()
        mock_litellm_client.get = AsyncMock(side_effect=Exception("should not be called"))

        result = await _resolve_hf_id("test-model", meta, mock_litellm_client)
        assert result == "explicit/model"

    @pytest.mark.asyncio
    async def test_derives_from_litellm_when_no_explicit(self):
        """When no explicit hf_id, derive from LiteLLM model info."""
        meta = {"family": "llama"}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"model_info": {"key": "openai/Qwen/Qwen3-Coder"}}]
        }
        mock_litellm_client = AsyncMock()
        mock_litellm_client.get = AsyncMock(return_value=mock_response)

        result = await _resolve_hf_id("qwen3-coder", meta, mock_litellm_client)
        assert result == "Qwen/Qwen3-Coder"

    @pytest.mark.asyncio
    async def test_litellm_key_without_openai_prefix(self):
        """When LiteLLM key has no openai/ prefix, pass through as-is."""
        meta = {"family": "llama"}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"model_info": {"key": "Qwen/Qwen3-Coder"}}]
        }
        mock_litellm_client = AsyncMock()
        mock_litellm_client.get = AsyncMock(return_value=mock_response)

        result = await _resolve_hf_id("qwen3-coder", meta, mock_litellm_client)
        assert result == "Qwen/Qwen3-Coder"

    @pytest.mark.asyncio
    async def test_litellm_key_at_top_level(self):
        """When key is at top level of data entry (not in model_info)."""
        meta = {"family": "llama"}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"key": "openai/meta-llama/Llama-3.1-8B"}]
        }
        mock_litellm_client = AsyncMock()
        mock_litellm_client.get = AsyncMock(return_value=mock_response)

        result = await _resolve_hf_id("llama-3.1-8b", meta, mock_litellm_client)
        assert result == "meta-llama/Llama-3.1-8B"

    @pytest.mark.asyncio
    async def test_returns_none_when_litellm_fails(self):
        """When LiteLLM lookup fails, return None."""
        import httpx
        meta = {"family": "llama"}
        mock_litellm_client = AsyncMock()
        mock_litellm_client.get = AsyncMock(
            side_effect=httpx.NetworkError("Connection refused")
        )

        result = await _resolve_hf_id("test-model", meta, mock_litellm_client)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_litellm_404(self):
        """When LiteLLM returns 404, return None."""
        meta = {"family": "llama"}
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_litellm_client = AsyncMock()
        mock_litellm_client.get = AsyncMock(return_value=mock_response)

        result = await _resolve_hf_id("test-model", meta, mock_litellm_client)
        assert result is None


class TestGetLiteLLMModelKey:
    """Test _get_litellm_model_key helper."""

    @pytest.mark.asyncio
    async def test_success_with_model_info_key(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"model_info": {"key": "openai/Qwen/Qwen3-Coder"}}]
        }
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await _get_litellm_model_key(mock_client, "qwen3-coder")
        assert result == "openai/Qwen/Qwen3-Coder"
        mock_client.get.assert_called_once_with("/v1/model/info", params={"model": "qwen3-coder"})

    @pytest.mark.asyncio
    async def test_success_with_top_level_key(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"key": "openai/meta-llama/Llama-3.1-8B"}]
        }
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await _get_litellm_model_key(mock_client, "llama-3.1-8b")
        assert result == "openai/meta-llama/Llama-3.1-8B"

    @pytest.mark.asyncio
    async def test_network_error(self):
        import httpx
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.NetworkError("Connection refused"))

        result = await _get_litellm_model_key(mock_client, "test-model")
        assert result is None

    @pytest.mark.asyncio
    async def test_timeout(self):
        import httpx
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))

        result = await _get_litellm_model_key(mock_client, "test-model")
        assert result is None

    @pytest.mark.asyncio
    async def test_non_200_response(self):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await _get_litellm_model_key(mock_client, "test-model")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_data(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await _get_litellm_model_key(mock_client, "test-model")
        assert result is None

    @pytest.mark.asyncio
    async def test_malformed_json(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await _get_litellm_model_key(mock_client, "test-model")
        assert result is None
