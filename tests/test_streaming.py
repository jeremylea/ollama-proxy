"""
Tests for the streaming functionality.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_sse_lines(contents, usage=None):
    """Generate OpenAI-style SSE lines for a list of content chunks."""
    lines = []
    for content in contents:
        data = {
            "choices": [{
                "delta": {"content": content, "role": "assistant"},
            }],
        }
        lines.append(f"data: {json.dumps(data)}")
    if usage:
        lines.append(f"data: {json.dumps({'usage': usage})}")
    lines.append("data: [DONE]")
    return lines


@patch("app.main.get_http_client")
def test_generate_streaming(mock_get_client, test_client):
    """Test the generate endpoint with streaming."""
    mock_client = AsyncMock()

    # Build mock streaming response
    sse_lines = _make_sse_lines(["Hello", " world", "!"], usage={"prompt_tokens": 3, "completion_tokens": 3})

    mock_response = MagicMock()
    mock_response.status_code = 200

    async def aiter_lines():
        for line in sse_lines:
            yield line

    mock_response.aiter_lines = aiter_lines
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    response = test_client.post("/api/generate", json={
        "model": "gpt-4o",
        "prompt": "Say hello",
        "stream": True,
    })

    assert response.status_code == 200

    chunks = [line for line in response.iter_lines() if line.strip()]
    # 3 content chunks + 1 final done chunk
    assert len(chunks) >= 4

    intermediate = []
    final = None
    for chunk_str in chunks:
        data = json.loads(chunk_str)
        if data.get("done"):
            final = data
        else:
            intermediate.append(data)

    assert len(intermediate) == 3
    assert intermediate[0]["response"] == "Hello"
    assert intermediate[1]["response"] == " world"
    assert intermediate[2]["response"] == "!"
    assert all(c["model"] == "gpt-4o" for c in intermediate)

    assert final is not None
    assert final["done"] is True
    assert final["done_reason"] == "stop"
    assert final["prompt_eval_count"] == 3
    assert final["eval_count"] == 3


@patch("app.main.get_http_client")
def test_chat_streaming(mock_get_client, test_client):
    """Test the chat endpoint with streaming."""
    mock_client = AsyncMock()

    sse_lines = _make_sse_lines(["I'm ", "doing ", "well!"], usage={"prompt_tokens": 4, "completion_tokens": 3})

    mock_response = MagicMock()
    mock_response.status_code = 200

    async def aiter_lines():
        for line in sse_lines:
            yield line

    mock_response.aiter_lines = aiter_lines
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    response = test_client.post("/api/chat", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "How are you?"}],
        "stream": True,
    })

    assert response.status_code == 200

    chunks = [line for line in response.iter_lines() if line.strip()]
    assert len(chunks) >= 4

    intermediate = []
    final = None
    for chunk_str in chunks:
        data = json.loads(chunk_str)
        if data.get("done"):
            final = data
        else:
            intermediate.append(data)

    assert len(intermediate) == 3
    assert intermediate[0]["message"]["content"] == "I'm "
    assert intermediate[1]["message"]["content"] == "doing "
    assert intermediate[2]["message"]["content"] == "well!"
    assert all(c["message"]["role"] == "assistant" for c in intermediate)

    assert final is not None
    assert final["done"] is True
    assert final["done_reason"] == "stop"
    assert final["prompt_eval_count"] == 4
    assert final["eval_count"] == 3


@patch("app.main.get_http_client")
def test_generate_streaming_no_usage(mock_get_client, test_client):
    """Test generate streaming when no usage data is provided."""
    mock_client = AsyncMock()

    sse_lines = _make_sse_lines(["Hello", "!"])

    mock_response = MagicMock()
    mock_response.status_code = 200

    async def aiter_lines():
        for line in sse_lines:
            yield line

    mock_response.aiter_lines = aiter_lines
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    response = test_client.post("/api/generate", json={
        "model": "gpt-4o",
        "prompt": "Hi",
        "stream": True,
    })

    assert response.status_code == 200
    chunks = [line for line in response.iter_lines() if line.strip()]
    final = None
    for chunk_str in chunks:
        data = json.loads(chunk_str)
        if data.get("done"):
            final = data
    assert final is not None
    assert final["done"] is True


@patch("app.main.get_http_client")
def test_chat_streaming_with_finish_reason(mock_get_client, test_client):
    """Test chat streaming captures finish_reason from the last choice."""
    mock_client = AsyncMock()

    lines = []
    chunk1 = json.dumps({"choices": [{"delta": {"content": "OK", "role": "assistant"}}]})
    lines.append(f"data: {chunk1}")
    chunk2 = json.dumps({"choices": [{"finish_reason": "length"}], "usage": {"prompt_tokens": 2, "completion_tokens": 1}})
    lines.append(f"data: {chunk2}")
    lines.append("data: [DONE]")

    mock_response = MagicMock()
    mock_response.status_code = 200

    async def aiter_lines():
        for line in lines:
            yield line

    mock_response.aiter_lines = aiter_lines
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    response = test_client.post("/api/chat", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    })

    assert response.status_code == 200
    chunks = [line for line in response.iter_lines() if line.strip()]
    final = None
    for chunk_str in chunks:
        data = json.loads(chunk_str)
        if data.get("done"):
            final = data
    assert final is not None
    assert final["done_reason"] == "length"
