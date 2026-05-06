"""
Tests for the /api/tags endpoint and model list transformation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@patch("app.main.get_http_client")
def test_tags_endpoint(mock_get_client, test_client):
    """Test /api/tags fetches and transforms models from LiteLLM."""
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

    names = [m["name"] for m in data["models"]]
    assert "gpt-4o" in names
    assert "claude-3-5-sonnet-20241022" in names

    # Check model structure
    for model in data["models"]:
        assert "name" in model
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
    """Test that model families are correctly inferred."""
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

    families = {m["name"]: m["details"]["family"] for m in data["models"]}
    assert families["gpt-4o"] == "gpt"
    assert families["claude-3-opus"] == "claude"
    assert families["gemini-pro"] == "gemini"
    assert families["llama3.1"] == "llama"
    assert families["mistral-large"] == "mistral"
    assert families["some-unknown-model"] == "unknown"
