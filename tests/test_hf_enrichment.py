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
    _fetch_hf_json,
    _infer_family,
    _infer_parameter_size,
    _infer_context_length,
    _infer_quantization,
    _extract_chat_template,
    _extract_stop_tokens,
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


class TestInferFamily:
    """Test _infer_family helper."""

    def test_llama_family(self):
        result = _infer_family("meta-llama/Llama-3", "text-generation", ["llama", "llama-3"])
        assert result == "llama"

    def test_mistral_family(self):
        result = _infer_family("mistralai/Mistral-Large", "text-generation", ["mistral", "mistral-large"])
        assert result == "mistral"

    def test_qwen_family(self):
        result = _infer_family("Qwen/Qwen2", "text-generation", ["qwen", "qwen2"])
        assert result == "qwen2"

    def test_gemma_family(self):
        result = _infer_family("google/gemma-2", "text-generation", ["gemma", "gemma-2"])
        assert result == "gemma2"

    def test_phi_family(self):
        result = _infer_family("microsoft/phi-3", "text-generation", ["phi", "phi-3"])
        assert result == "phi"

    def test_deepseek_family(self):
        result = _infer_family("deepseek-ai/DeepSeek-R1", "text-generation", ["deepseek"])
        assert result == "deepseek"

    def test_mixtral_family(self):
        result = _infer_family("mistralai/Mixtral-8x7B", "text-generation", ["mixtral"])
        assert result == "mistral"

    def test_unknown_family(self):
        result = _infer_family("some-org/some-model", "text-generation", ["some-model"])
        assert result is None

    def test_infer_from_tags_only(self):
        """Family should be inferred from tags when hf_id doesn't match."""
        result = _infer_family("some-org/model", "text-generation", ["llama"])
        assert result == "llama"


class TestInferContextLength:
    """Test _infer_context_length helper."""

    def test_max_position_embeddings(self):
        config = {"max_position_embeddings": 4096}
        assert _infer_context_length(config, {}) == 4096

    def test_context_length_key(self):
        config = {"context_length": 8192}
        assert _infer_context_length(config, {}) == 8192

    def test_n_ctx_key(self):
        config = {"n_ctx": 2048}
        assert _infer_context_length(config, {}) == 2048

    def test_max_sequence_length_key(self):
        config = {"max_sequence_length": 16384}
        assert _infer_context_length(config, {}) == 16384

    def test_seq_length_key(self):
        config = {"seq_length": 32768}
        assert _infer_context_length(config, {}) == 32768

    def test_model_max_length_from_tok_config(self):
        config = {}
        tok_config = {"model_max_length": 131072}
        assert _infer_context_length(config, tok_config) == 131072

    def test_no_matching_key(self):
        config = {"other_key": 123}
        assert _infer_context_length(config, {}) is None

    def test_empty_config(self):
        assert _infer_context_length({}, {}) is None

    def test_non_int_value(self):
        config = {"max_position_embeddings": "4096"}
        assert _infer_context_length(config, {}) is None

    def test_small_value_ignored(self):
        """Values <= 512 are ignored as they are not realistic context lengths."""
        config = {"max_position_embeddings": 256}
        assert _infer_context_length(config, {}) is None

    def test_tok_config_huge_value_ignored(self):
        """model_max_length > 10M is ignored (likely a placeholder)."""
        tok_config = {"model_max_length": 100_000_000}
        assert _infer_context_length({}, tok_config) is None


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

    def test_awq(self):
        assert _infer_quantization(["awq", "text-generation"]) == "AWQ"

    def test_gptq(self):
        assert _infer_quantization(["gptq", "text-generation"]) == "GPTQ"

    def test_no_quantization_tag(self):
        assert _infer_quantization(["text-generation", "pytorch"]) is None

    def test_empty_tags(self):
        assert _infer_quantization([]) is None


class TestBuildModelInfo:
    """Test _build_model_info helper."""

    def test_basic_info(self):
        hf_data = {"id": "test/model", "pipeline_tag": "text-generation"}
        config = {
            "model_type": "llama",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "max_position_embeddings": 4096,
        }
        tok_config = {"eos_token": "</s>", "bos_token": "<s>"}
        card_data = {}
        result = _build_model_info(hf_data, config, tok_config, card_data)
        assert result["general.name"] == "test/model"
        assert result["general.pipeline_tag"] == "text-generation"
        assert result["llama.vocab_size"] == 32000
        assert result["llama.embedding_length"] == 4096
        assert result["llama.block_count"] == 32
        assert result["llama.context_length"] == 4096
        assert result["tokenizer.eos_token"] == "</s>"
        assert result["tokenizer.bos_token"] == "<s>"

    def test_empty_data(self):
        result = _build_model_info({}, {}, {}, {})
        assert result is None

    def test_partial_data(self):
        hf_data = {"id": "test/model"}
        config = {}
        tok_config = {}
        card_data = {}
        result = _build_model_info(hf_data, config, tok_config, card_data)
        assert result["general.name"] == "test/model"
        assert "general.pipeline_tag" not in result

    def test_arch_normalization(self):
        """qwen2_moe should be normalized to qwen2 prefix."""
        config = {
            "model_type": "qwen2_moe",
            "hidden_size": 5120,
            "max_position_embeddings": 131072,
        }
        result = _build_model_info({}, config, {}, {})
        assert result["qwen2.embedding_length"] == 5120
        assert result["qwen2.context_length"] == 131072

    def test_chat_template_stored(self):
        """Raw Jinja chat template should be stored in tokenizer.chat_template."""
        tok_config = {"chat_template": "{{ prompt }}{% for m in messages %}{{ m.content }}{% endfor %}"}
        result = _build_model_info({}, {}, tok_config, {})
        assert "tokenizer.chat_template" in result

    def test_chat_template_dict(self):
        """Dict chat_template should store the default variant."""
        tok_config = {"chat_template": {"default": "default template", "other": "other template"}}
        result = _build_model_info({}, {}, tok_config, {})
        assert result["tokenizer.chat_template"] == "default template"


class TestEnrichSingleModel:
    """Test _enrich_single_model with mocked HTTP responses."""

    def _make_mock_client(self, api_data=None, config_data=None, tok_data=None, api_error=None):
        """Helper to create a mock client that returns different responses for 3 URLs."""
        responses = []
        if api_error:
            responses.append(api_error)
        elif api_data is not None:
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = api_data
            responses.append(resp)
        else:
            resp = MagicMock()
            resp.status_code = 404
            responses.append(resp)

        if config_data is not None:
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = config_data
            responses.append(resp)
        else:
            resp = MagicMock()
            resp.status_code = 404
            responses.append(resp)

        if tok_data is not None:
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = tok_data
            responses.append(resp)
        else:
            resp = MagicMock()
            resp.status_code = 404
            responses.append(resp)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=responses)
        return mock_client

    @pytest.mark.asyncio
    async def test_successful_enrichment(self):
        MODEL_METADATA.clear()
        MODEL_METADATA["test-model"] = {
            "family": "llama",
            "hf_id": "test/model",
            "format": "gguf",
        }

        mock_client = self._make_mock_client(
            api_data={
                "id": "test/model",
                "size": 5000000000,
                "pipeline_tag": "text-generation",
                "tags": ["llama", "fp8"],
                "cardData": {},
            },
            config_data={
                "model_type": "llama",
                "max_position_embeddings": 8192,
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
            },
            tok_data={"eos_token": "</s>"},
        )

        await _enrich_single_model(mock_client, "test-model", "test/model")

        assert MODEL_METADATA["test-model"]["family"] == "llama"
        assert MODEL_METADATA["test-model"]["format"] == "gguf"
        assert MODEL_METADATA["test-model"]["size"] == 5000000000
        assert MODEL_METADATA["test-model"]["context_length"] == 8192
        assert MODEL_METADATA["test-model"]["quantization_level"] == "FP8"
        assert "model_info" in MODEL_METADATA["test-model"]
        MODEL_METADATA.clear()

    @pytest.mark.asyncio
    async def test_api_fetch_fails_fallback(self):
        """When the main HF API fetch fails, fall back to YAML."""
        MODEL_METADATA.clear()
        MODEL_METADATA["test-model"] = {
            "family": "llama",
            "hf_id": "nonexistent/model",
            "format": "gguf",
        }

        mock_client = self._make_mock_client(api_data=None)

        await _enrich_single_model(mock_client, "test-model", "nonexistent/model")

        assert MODEL_METADATA["test-model"]["family"] == "llama"
        assert "size" not in MODEL_METADATA["test-model"]
        MODEL_METADATA.clear()

    @pytest.mark.asyncio
    async def test_network_error_fallback(self):
        import httpx
        MODEL_METADATA.clear()
        MODEL_METADATA["test-model"] = {
            "family": "llama",
            "hf_id": "test/model",
            "format": "gguf",
        }

        mock_client = self._make_mock_client(api_error=httpx.NetworkError("Connection refused"))

        await _enrich_single_model(mock_client, "test-model", "test/model")

        assert MODEL_METADATA["test-model"]["family"] == "llama"
        assert "size" not in MODEL_METADATA["test-model"]
        MODEL_METADATA.clear()

    @pytest.mark.asyncio
    async def test_timeout_fallback(self):
        import httpx
        MODEL_METADATA.clear()
        MODEL_METADATA["test-model"] = {
            "family": "llama",
            "hf_id": "test/model",
            "format": "gguf",
        }

        mock_client = self._make_mock_client(api_error=httpx.TimeoutException("Timeout"))

        await _enrich_single_model(mock_client, "test-model", "test/model")

        assert MODEL_METADATA["test-model"]["family"] == "llama"
        assert "size" not in MODEL_METADATA["test-model"]
        MODEL_METADATA.clear()

    @pytest.mark.asyncio
    async def test_yaml_values_preserved_over_hf(self):
        """YAML values should take precedence when already set."""
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

        mock_client = self._make_mock_client(
            api_data={
                "id": "test/model",
                "size": 9999999999,
                "pipeline_tag": "text-generation",
                "tags": ["mistral", "fp8"],
                "cardData": {},
            },
            config_data={"max_position_embeddings": 32768},
            tok_data={},
        )

        await _enrich_single_model(mock_client, "test-model", "test/model")

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
            "family": "",
            "parameter_size": "",
            "context_length": None,
            "quantization_level": "",
            "model_info": {},
            "hf_id": "Qwen/Qwen2-72B",
            "format": "gguf",
        }

        mock_client = self._make_mock_client(
            api_data={
                "id": "Qwen/Qwen2-72B",
                "size": 5000000000,
                "pipeline_tag": "text-generation",
                "tags": ["qwen", "fp8"],
                "cardData": {},
            },
            config_data={
                "model_type": "qwen2",
                "max_position_embeddings": 131072,
                "hidden_size": 5120,
                "num_hidden_layers": 60,
                "num_attention_heads": 40,
            },
            tok_data={"eos_token": "</s>"},
        )

        await _enrich_single_model(mock_client, "test-model", "Qwen/Qwen2-72B")

        assert MODEL_METADATA["test-model"]["family"] == "qwen2"
        assert MODEL_METADATA["test-model"]["parameter_size"] == "72B"
        assert MODEL_METADATA["test-model"]["context_length"] == 131072
        assert MODEL_METADATA["test-model"]["quantization_level"] == "FP8"
        assert MODEL_METADATA["test-model"]["size"] == 5000000000
        assert len(MODEL_METADATA["test-model"]["model_info"]) > 0
        MODEL_METADATA.clear()

    @pytest.mark.asyncio
    async def test_template_extracted_from_tok_config(self):
        """Chat template should be extracted from tokenizer_config.json."""
        MODEL_METADATA.clear()
        MODEL_METADATA["test-model"] = {
            "family": "qwen2",
            "hf_id": "Qwen/Qwen2",
            "format": "gguf",
        }

        mock_client = self._make_mock_client(
            api_data={
                "id": "Qwen/Qwen2",
                "size": 1000,
                "pipeline_tag": "text-generation",
                "tags": ["qwen"],
                "cardData": {},
            },
            config_data={"model_type": "qwen2", "max_position_embeddings": 4096},
            tok_data={"chat_template": "▌system\n{{ system }}\n▌user\n{{ prompt }}\n▌assistant\n"},
        )

        await _enrich_single_model(mock_client, "test-model", "Qwen/Qwen2")

        assert "template" in MODEL_METADATA["test-model"]
        assert "▌" in MODEL_METADATA["test-model"]["template"]
        MODEL_METADATA.clear()

    @pytest.mark.asyncio
    async def test_stop_tokens_extracted(self):
        """Stop tokens should be extracted from tokenizer_config.json."""
        MODEL_METADATA.clear()
        MODEL_METADATA["test-model"] = {
            "family": "llama",
            "hf_id": "meta-llama/Llama-3",
            "format": "gguf",
        }

        mock_client = self._make_mock_client(
            api_data={
                "id": "meta-llama/Llama-3",
                "size": 1000,
                "pipeline_tag": "text-generation",
                "tags": ["llama"],
                "cardData": {},
            },
            config_data={"model_type": "llama", "max_position_embeddings": 8192},
            tok_data={
                "eos_token": "<|end_of_text|>",
                "additional_special_tokens": ["<|eot_id|>", "<|start_header_id|>"],
            },
        )

        await _enrich_single_model(mock_client, "test-model", "meta-llama/Llama-3")

        assert "stop_tokens" in MODEL_METADATA["test-model"]
        assert "<|end_of_text|>" in MODEL_METADATA["test-model"]["stop_tokens"]
        MODEL_METADATA.clear()


class TestEnrichModelMetadataFromHF:
    """Test the top-level enrich_model_metadata_from_hf function."""

    @pytest.mark.asyncio
    async def test_skips_when_no_metadata(self):
        MODEL_METADATA.clear()
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
            mock_litellm_client.get = AsyncMock(return_value=MagicMock(status_code=404))
            await enrich_model_metadata_from_hf(mock_litellm_client)
        finally:
            app.config.HF_METADATA_ENABLED = original
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

        # Mock responses for 3 fetches: API, config.json, tokenizer_config.json
        api_resp = MagicMock()
        api_resp.status_code = 200
        api_resp.json.return_value = {
            "id": "test/model",
            "size": 5000000000,
            "pipeline_tag": "text-generation",
            "tags": ["llama"],
            "cardData": {},
        }
        config_resp = MagicMock()
        config_resp.status_code = 200
        config_resp.json.return_value = {
            "model_type": "llama",
            "max_position_embeddings": 8192,
        }
        tok_resp = MagicMock()
        tok_resp.status_code = 200
        tok_resp.json.return_value = {}

        import app.config
        original = app.config.HF_METADATA_ENABLED
        app.config.HF_METADATA_ENABLED = True

        with patch("app.config.httpx") as mock_httpx:
            mock_hf_client = AsyncMock()
            mock_hf_client.get = AsyncMock(side_effect=[api_resp, config_resp, tok_resp])
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


class TestFetchHFJson:
    """Test _fetch_hf_json helper."""

    @pytest.mark.asyncio
    async def test_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"key": "value"}
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await _fetch_hf_json(mock_client, "https://example.com/data.json", "test")
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_network_error(self):
        import httpx
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.NetworkError("Connection refused"))

        result = await _fetch_hf_json(mock_client, "https://example.com/data.json", "test")
        assert result is None

    @pytest.mark.asyncio
    async def test_timeout(self):
        import httpx
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))

        result = await _fetch_hf_json(mock_client, "https://example.com/data.json", "test")
        assert result is None

    @pytest.mark.asyncio
    async def test_non_200(self):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await _fetch_hf_json(mock_client, "https://example.com/data.json", "test")
        assert result is None

    @pytest.mark.asyncio
    async def test_malformed_json(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await _fetch_hf_json(mock_client, "https://example.com/data.json", "test")
        assert result is None


class TestExtractChatTemplate:
    """Test _extract_chat_template helper."""

    def test_no_template(self):
        assert _extract_chat_template({}) is None

    def test_empty_template(self):
        assert _extract_chat_template({"chat_template": ""}) is None

    def test_chatml_format(self):
        tok_config = {"chat_template": "▌system\n{{ system }}\n▌user\n{{ prompt }}\n▌assistant\n"}
        result = _extract_chat_template(tok_config)
        assert result is not None
        assert "▌" in result

    def test_llama3_format(self):
        tok_config = {"chat_template": "<|start_header_id|>system<|end_header_id|>\n\n{{ system }}<|eot_id|>"}
        result = _extract_chat_template(tok_config)
        assert result is not None
        assert "<|start_header_id|>" in result

    def test_mistral_format(self):
        tok_config = {"chat_template": "[INST] {{ prompt }} [/INST]"}
        result = _extract_chat_template(tok_config)
        assert result is not None
        assert "[INST]" in result

    def test_phi3_format(self):
        tok_config = {"chat_template": "<|user|>\n{{ prompt }}<|end|>\n<|assistant|>\n"}
        result = _extract_chat_template(tok_config)
        assert result is not None
        assert "<|system|>" in result
        assert "<|assistant|>" in result

    def test_gemma_format(self):
        tok_config = {"chat_template": "<start_of_turn>user\n{{ prompt }}<end_of_turn>\n<start_of_turn>model\n"}
        result = _extract_chat_template(tok_config)
        assert result is not None
        assert "<start_of_turn>" in result

    def test_unknown_format(self):
        tok_config = {"chat_template": "Some unknown template format"}
        result = _extract_chat_template(tok_config)
        assert result is not None
        assert "{{ .Role }}" in result

    def test_dict_template_default(self):
        tok_config = {"chat_template": {"default": "▌system\n{{ system }}", "other": "other"}}
        result = _extract_chat_template(tok_config)
        assert result is not None
        assert "▌" in result


class TestExtractStopTokens:
    """Test _extract_stop_tokens helper."""

    def test_eos_token_string(self):
        tok_config = {"eos_token": "</s>"}
        result = _extract_stop_tokens(tok_config, None)
        assert result == ["</s>"]

    def test_eos_token_dict(self):
        tok_config = {"eos_token": {"content": "<|end_of_text|>"}}
        result = _extract_stop_tokens(tok_config, None)
        assert result == ["<|end_of_text|>"]

    def test_additional_special_tokens(self):
        tok_config = {
            "eos_token": "</s>",
            "additional_special_tokens": ["<|eot_id|>", "<|start_header_id|>"],
        }
        result = _extract_stop_tokens(tok_config, None)
        assert "</s>" in result
        assert "<|eot_id|>" in result

    def test_family_fallback_qwen(self):
        tok_config = {}
        result = _extract_stop_tokens(tok_config, "qwen2")
        assert len(result) == 2

    def test_family_fallback_llama(self):
        tok_config = {}
        result = _extract_stop_tokens(tok_config, "llama")
        assert "<|eot_id|>" in result

    def test_family_fallback_mistral(self):
        tok_config = {}
        result = _extract_stop_tokens(tok_config, "mistral")
        assert "</s>" in result

    def test_no_family_no_tokens(self):
        tok_config = {}
        result = _extract_stop_tokens(tok_config, None)
        assert result == []

    def test_no_family_unknown(self):
        tok_config = {}
        result = _extract_stop_tokens(tok_config, "unknown")
        assert result == []


class TestInferParameterSize:
    """Test _infer_parameter_size helper."""

    def test_from_hf_id_simple(self):
        result = _infer_parameter_size("Qwen/Qwen2-72B", {}, {})
        assert result == "72B"

    def test_from_hf_id_moe_total(self):
        """MoE models like 480B-A35B should return total params (480B)."""
        result = _infer_parameter_size("Qwen/Qwen3-Coder-480B-A35B-Instruct", {}, {})
        assert result == "480B"

    def test_from_hf_id_decimal(self):
        result = _infer_parameter_size("meta-llama/Llama-3.1-8B", {}, {})
        assert result == "8B"

    def test_from_hf_id_small(self):
        result = _infer_parameter_size("google/gemma-2-2B", {}, {})
        assert result == "2B"

    def test_from_card_data(self):
        result = _infer_parameter_size("some/model", {"model_name": "Llama-3-70B"}, {})
        assert result == "70B"

    def test_from_config_num_parameters(self):
        result = _infer_parameter_size("some/model", {}, {"num_parameters": 8000000000})
        assert result == "8B"

    def test_no_size_info(self):
        result = _infer_parameter_size("some/model", {}, {})
        assert result is None

    def test_hf_id_takes_precedence_over_card(self):
        """hf_id pattern should take precedence over cardData."""
        result = _infer_parameter_size("Qwen/Qwen2-72B", {"model_name": "Llama-3-8B"}, {})
        assert result == "72B"

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
