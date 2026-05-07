"""
Tests for the main API endpoints of the Ollama API proxy.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import ConnectError, TimeoutException


def test_root_endpoint(test_client):
    """Test the root endpoint."""
    response = test_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data


def test_root_head(test_client):
    """Test the HEAD root endpoint."""
    response = test_client.head("/")
    assert response.status_code == 200


def test_version_endpoint(test_client):
    """Test the version endpoint."""
    response = test_client.get("/api/version")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data


def test_ps_endpoint(test_client):
    """Test the /api/ps endpoint."""
    response = test_client.get("/api/ps")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)


def test_show_endpoint(test_client):
    """Test the POST /api/show endpoint."""
    response = test_client.post("/api/show", json={"model": "gpt-4o"})
    assert response.status_code == 200
    data = response.json()
    assert "modelfile" in data


def test_show_endpoint_empty_json_body(test_client):
    """Test the POST /api/show endpoint with an empty JSON body (Copilot sends this)."""
    response = test_client.post(
        "/api/show",
        content=b"{}",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "modelfile" in data


def test_show_endpoint_no_model_field(test_client):
    """Test the POST /api/show endpoint when model field is missing."""
    response = test_client.post("/api/show", json={})
    assert response.status_code == 200
    data = response.json()
    assert "modelfile" in data


@pytest.mark.parametrize("endpoint,method,body", [
    ("/api/create", "post", {"model": "test"}),
    ("/api/copy", "post", {"source": "src", "destination": "dst"}),
    ("/api/pull", "post", {"model": "test"}),
    ("/api/push", "post", {"model": "test"}),
])
def test_unsupported_post_endpoints(test_client, endpoint, method, body):
    """Test unsupported POST endpoints return 501."""
    client_method = getattr(test_client, method)
    response = client_method(endpoint, json=body)
    assert response.status_code == 501
    assert "detail" in response.json()


def test_unsupported_delete_endpoint(test_client):
    """Test unsupported DELETE endpoint returns 501."""
    response = test_client.request("DELETE", "/api/delete", json={"model": "test"})
    assert response.status_code == 501
    assert "detail" in response.json()


@patch("app.main.get_http_client")
def test_generate_non_streaming(mock_get_client, test_client):
    """Test the generate endpoint with stream=false."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {"content": "This is a test response.", "role": "assistant"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10},
    }
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    response = test_client.post("/api/generate", json={
        "model": "gpt-4o",
        "prompt": "Hello",
        "stream": False,
    })

    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "gpt-4o"
    assert data["response"] == "This is a test response."
    assert data["done"] is True
    assert data["prompt_eval_count"] == 5
    assert data["eval_count"] == 10

    # Verify the request sent to LiteLLM
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args
    assert call_kwargs[0][0] == "/v1/chat/completions"
    body = call_kwargs[1]["json"]
    assert body["model"] == "gpt-4o"
    assert len(body["messages"]) == 1
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"] == "Hello"


@patch("app.main.get_http_client")
def test_generate_with_system(mock_get_client, test_client):
    """Test generate with a system prompt."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "OK", "role": "assistant"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    response = test_client.post("/api/generate", json={
        "model": "gpt-4o",
        "prompt": "Hi",
        "system": "You are helpful.",
        "stream": False,
    })

    assert response.status_code == 200
    call_kwargs = mock_client.post.call_args
    body = call_kwargs[1]["json"]
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "You are helpful."


@patch("app.main.get_http_client")
def test_generate_empty_prompt(mock_get_client, test_client):
    """Test generate with empty prompt returns load response."""
    mock_client = AsyncMock()
    mock_get_client.return_value = mock_client

    response = test_client.post("/api/generate", json={
        "model": "gpt-4o",
        "prompt": "",
        "stream": False,
    })

    assert response.status_code == 200
    data = response.json()
    assert data["done"] is True
    assert data["done_reason"] == "load"
    assert data["response"] == ""
    # Should NOT call LiteLLM
    mock_client.post.assert_not_called()


@patch("app.main.get_http_client")
def test_chat_non_streaming(mock_get_client, test_client):
    """Test the chat endpoint with stream=false."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {"content": "Hello! How can I help?", "role": "assistant"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 8, "completion_tokens": 6},
    }
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    response = test_client.post("/api/chat", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi there"}],
        "stream": False,
    })

    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "gpt-4o"
    assert data["message"]["role"] == "assistant"
    assert data["message"]["content"] == "Hello! How can I help?"
    assert data["done"] is True
    assert data["prompt_eval_count"] == 8
    assert data["eval_count"] == 6


@patch("app.main.get_http_client")
def test_chat_empty_messages(mock_get_client, test_client):
    """Test chat with empty messages returns load response."""
    mock_client = AsyncMock()
    mock_get_client.return_value = mock_client

    response = test_client.post("/api/chat", json={
        "model": "gpt-4o",
        "messages": [],
        "stream": False,
    })

    assert response.status_code == 200
    data = response.json()
    assert data["done"] is True
    assert data["done_reason"] == "load"
    mock_client.post.assert_not_called()


@patch("app.main.get_http_client")
def test_embed_endpoint(mock_get_client, test_client):
    """Test the embed endpoint."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3]},
        ],
        "usage": {"prompt_tokens": 3},
    }
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    response = test_client.post("/api/embed", json={
        "model": "text-embedding-3-small",
        "input": "Hello world",
    })

    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "text-embedding-3-small"
    assert len(data["embeddings"]) == 1
    assert data["embeddings"][0] == [0.1, 0.2, 0.3]


@patch("app.main.get_http_client")
def test_generate_backend_unavailable(mock_get_client, test_client):
    """Test generate when LiteLLM proxy is unreachable."""
    mock_client = AsyncMock()
    mock_client.post.side_effect = ConnectError("Connection refused")
    mock_get_client.return_value = mock_client

    response = test_client.post("/api/generate", json={
        "model": "gpt-4o",
        "prompt": "Hello",
        "stream": False,
    })

    assert response.status_code == 502
    assert "Backend proxy unavailable" in response.json()["detail"]


@patch("app.main.get_http_client")
def test_chat_backend_unavailable(mock_get_client, test_client):
    """Test chat when LiteLLM proxy is unreachable."""
    mock_client = AsyncMock()
    mock_client.post.side_effect = ConnectError("Connection refused")
    mock_get_client.return_value = mock_client

    response = test_client.post("/api/chat", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
    })

    assert response.status_code == 502
    assert "Backend proxy unavailable" in response.json()["detail"]


@patch("app.main.get_http_client")
def test_generate_timeout(mock_get_client, test_client):
    """Test generate when LiteLLM proxy times out."""
    mock_client = AsyncMock()
    mock_client.post.side_effect = TimeoutException("Timeout")
    mock_get_client.return_value = mock_client

    response = test_client.post("/api/generate", json={
        "model": "gpt-4o",
        "prompt": "Hello",
        "stream": False,
    })

    assert response.status_code == 504
    assert "Backend proxy timeout" in response.json()["detail"]


@patch("app.main.get_http_client")
def test_generate_litellm_error(mock_get_client, test_client):
    """Test generate when LiteLLM returns an error."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.json.return_value = {
        "error": {"message": "Invalid model", "type": "invalid_request_error"}
    }
    mock_response.text = ""
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    response = test_client.post("/api/generate", json={
        "model": "bad-model",
        "prompt": "Hello",
        "stream": False,
    })

    assert response.status_code == 400
    assert "Invalid model" in response.json()["detail"]


@patch("app.main.get_http_client")
def test_tags_backend_unavailable(mock_get_client, test_client):
    """Test /api/tags when LiteLLM proxy is unreachable."""
    mock_client = AsyncMock()
    mock_client.get.side_effect = ConnectError("Connection refused")
    mock_get_client.return_value = mock_client

    response = test_client.get("/api/tags")
    assert response.status_code == 502
    assert "Backend proxy unavailable" in response.json()["detail"]


@patch("app.main.get_http_client")
def test_embed_backend_unavailable(mock_get_client, test_client):
    """Test /api/embed when LiteLLM proxy is unreachable."""
    mock_client = AsyncMock()
    mock_client.post.side_effect = ConnectError("Connection refused")
    mock_get_client.return_value = mock_client

    response = test_client.post("/api/embed", json={
        "model": "text-embedding-3-small",
        "input": "Hello",
    })

    assert response.status_code == 502
    assert "Backend proxy unavailable" in response.json()["detail"]


# ── OpenAI-compatible /v1/ pass-through endpoints ───────────────────────────


@patch("app.main.get_http_client")
def test_v1_chat_completions_non_streaming(mock_get_client, test_client):
    """Test /v1/chat/completions pass-through with stream=false."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "chatcmpl-123",
        "choices": [{
            "message": {"content": "Hello!", "role": "assistant"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }
    mock_client.send = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    response = test_client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
    })

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "chatcmpl-123"
    assert data["choices"][0]["message"]["content"] == "Hello!"

    # Verify send was called with stream=False
    mock_client.send.assert_called_once()
    call_args = mock_client.send.call_args
    assert call_args[1]["stream"] is False


@patch("app.main.get_http_client")
def test_v1_chat_completions_streaming(mock_get_client, test_client):
    """Test /v1/chat/completions pass-through with stream=true."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200

    async def _aiter_lines():
        for line in [
            "data: {\"id\":\"chatcmpl-123\",\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}",
            "data: [DONE]",
        ]:
            yield line

    mock_response.aiter_lines = _aiter_lines
    mock_response.aclose = AsyncMock()
    mock_client.send = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    response = test_client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    })

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


@patch("app.main.get_http_client")
def test_v1_chat_completions_backend_unavailable(mock_get_client, test_client):
    """Test /v1/chat/completions when LiteLLM is unreachable."""
    mock_client = AsyncMock()
    mock_client.send.side_effect = ConnectError("Connection refused")
    mock_get_client.return_value = mock_client

    response = test_client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
    })

    assert response.status_code == 502
    assert "Backend proxy unavailable" in response.json()["detail"]


@patch("app.main.get_http_client")
def test_v1_models(mock_get_client, test_client):
    """Test /v1/models pass-through."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"id": "gpt-4o", "object": "model", "created": 1234567890},
        ]
    }
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    response = test_client.get("/v1/models")

    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "gpt-4o"

    mock_client.get.assert_called_once_with("/v1/models")


@patch("app.main.get_http_client")
def test_v1_models_backend_unavailable(mock_get_client, test_client):
    """Test /v1/models when LiteLLM is unreachable."""
    mock_client = AsyncMock()
    mock_client.get.side_effect = ConnectError("Connection refused")
    mock_get_client.return_value = mock_client

    response = test_client.get("/v1/models")

    assert response.status_code == 502
    assert "Backend proxy unavailable" in response.json()["detail"]


@patch("app.main.get_http_client")
def test_v1_embeddings(mock_get_client, test_client):
    """Test /v1/embeddings pass-through."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
        "usage": {"prompt_tokens": 3},
    }
    mock_client.send = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    response = test_client.post("/v1/embeddings", json={
        "model": "text-embedding-3-small",
        "input": "Hello world",
    })

    assert response.status_code == 200
    data = response.json()
    assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]

    mock_client.send.assert_called_once()


@patch("app.main.get_http_client")
def test_v1_embeddings_backend_unavailable(mock_get_client, test_client):
    """Test /v1/embeddings when LiteLLM is unreachable."""
    mock_client = AsyncMock()
    mock_client.send.side_effect = ConnectError("Connection refused")
    mock_get_client.return_value = mock_client

    response = test_client.post("/v1/embeddings", json={
        "model": "text-embedding-3-small",
        "input": "Hello",
    })

    assert response.status_code == 502
    assert "Backend proxy unavailable" in response.json()["detail"]
