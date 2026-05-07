"""
Tests for the /api/tags endpoint and model list transformation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@patch("app.main.get_http_client")
def test_tags_endpoint(mock_get_client, test_client):
    """Test /api/tags fetches and transforms models from LiteLLM."""
    import app.main
    original = app.main.MODEL_METADATA.copy()
    app.main.MODEL_METADATA.clear()
    app.main.MODEL_METADATA.update({
        "gpt-4o": {"family": "gpt", "format": "gguf", "size": 1000,
                    "parameter_size": "1.8B", "quantization_level": "Q4_0"},
        "claude-3-5-sonnet-20241022": {"family": "claude", "format": "gguf", "size": 2000,
                    "parameter_size": "15B", "quantization_level": "Q4_0"},
    })
    try:
        _test_tags_endpoint_impl(mock_get_client, test_client)
    finally:
        app.main.MODEL_METADATA.clear()
        app.main.MODEL_METADATA.update(original)


def _test_tags_endpoint_impl(mock_get_client, test_client):
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {
                "id": "gpt-4o",
                "object": "model",
                "created": 1700000000,
                "owned_by": "openai",
            },
            {
                "id": "claude-3-5-sonnet-20241022",
                "object": "model",
                "created": 1699000000,
                "owned_by": "anthropic",
            },
        ],
    }
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    response = test_client.get("/api/tags")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) == 2

    # Ollama uses model:tag format, so names include ':latest'
    names = [m["name"] for m in data["models"]]
    assert "gpt-4o:latest" in names
    assert "claude-3-5-sonnet-20241022:latest" in names

    # Check model structure matches Ollama spec
    for model in data["models"]:
        assert "name" in model
        assert "model" in model  # Ollama spec requires both 'name' and 'model'
        assert model["name"] == model["model"]  # Both should be identical
        assert "modified_at" in model
        assert "size" in model
        assert "digest" in model
        assert "details" in model
        assert "family" in model["details"]


@patch("app.main.get_http_client")
def test_tags_empty_models(mock_get_client, test_client):
    """Test /api/tags with empty model list."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": []}
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    response = test_client.get("/api/tags")
    assert response.status_code == 200
    data = response.json()
    assert data["models"] == []


@patch("app.main.get_http_client")
def test_tags_litellm_error(mock_get_client, test_client):
    """Test /api/tags when LiteLLM returns non-200."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    response = test_client.get("/api/tags")
    # Should return empty list on error (not crash)
    assert response.status_code == 200
    data = response.json()
    assert data["models"] == []


@patch("app.main.get_http_client")
def test_tags_model_families(mock_get_client, test_client):
    """Test that model families come from config.yaml metadata."""
    import app.main
    original = app.main.MODEL_METADATA.copy()
    app.main.MODEL_METADATA.clear()
    app.main.MODEL_METADATA.update({
        "gpt-4o": {"family": "gpt", "format": "gguf", "size": 0,
                    "parameter_size": "1.8B", "quantization_level": "Q4_0"},
        "claude-3-opus": {"family": "claude", "format": "gguf", "size": 0,
                    "parameter_size": "20B", "quantization_level": "Q4_0"},
        "gemini-pro": {"family": "gemini", "format": "gguf", "size": 0,
                    "parameter_size": "10B", "quantization_level": "Q4_0"},
        "llama3.1": {"family": "llama", "format": "gguf", "size": 0,
                    "parameter_size": "8B", "quantization_level": "Q4_0"},
        "mistral-large": {"family": "mistral", "format": "gguf", "size": 0,
                    "parameter_size": "12B", "quantization_level": "Q4_0"},
    })
    try:
        _test_tags_model_families_impl(mock_get_client, test_client)
    finally:
        app.main.MODEL_METADATA.clear()
        app.main.MODEL_METADATA.update(original)


def _test_tags_model_families_impl(mock_get_client, test_client):
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"id": "gpt-4o", "object": "model", "created": 1700000000},
            {"id": "claude-3-opus", "object": "model", "created": 1700000000},
            {"id": "gemini-pro", "object": "model", "created": 1700000000},
            {"id": "llama3.1", "object": "model", "created": 1700000000},
            {"id": "mistral-large", "object": "model", "created": 1700000000},
            {"id": "some-unknown-model", "object": "model", "created": 1700000000},
        ],
    }
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    response = test_client.get("/api/tags")
    assert response.status_code == 200
    data = response.json()

    # "some-unknown-model" is not in config, so it should be filtered out
    assert len(data["models"]) == 5

    # Ollama uses model:tag format, so names include ':latest'
    families = {m["name"]: m["details"]["family"] for m in data["models"]}
    assert families["gpt-4o:latest"] == "gpt"
    assert families["claude-3-opus:latest"] == "claude"
    assert families["gemini-pro:latest"] == "gemini"
    assert families["llama3.1:latest"] == "llama"
    assert families["mistral-large:latest"] == "mistral"
