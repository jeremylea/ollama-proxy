"""
Ollama API Proxy Server

Implements the Ollama API specification and routes all requests to a
configurable LiteLLM proxy server via HTTP.
"""

import hashlib
import json
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator, Dict, Any, List, Optional, Union

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from app.config import settings, setup_logging, MODEL_METADATA, initialize_model_metadata
from app.models import (
    GenerateRequest,
    GenerateResponse,
    ChatRequest,
    ChatMessage,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ShowModelRequest,
    ShowModelResponse,
    ListTagsResponse,
    ModelInfo,
    ModelDetails,
    PsResponse,
    VersionResponse,
    CreateModelRequest,
    CopyModelRequest,
    DeleteModelRequest,
    PullModelRequest,
    PushModelRequest,
)

logger = logging.getLogger(__name__)

# ── Global HTTP client (initialized in lifespan) ────────────────────────────

_http_client: Optional[httpx.AsyncClient] = None


async def get_http_client() -> httpx.AsyncClient:
    """Return the shared async HTTP client for LiteLLM proxy calls.

    The client is initialized once during app startup in the lifespan handler,
    so this function simply returns the pre-created instance.
    """
    assert _http_client is not None and not _http_client.is_closed, (
        "HTTP client not initialized — call outside of app lifespan?"
    )
    return _http_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global _http_client
    setup_logging()
    logger.info("Starting Ollama API proxy")
    logger.info("LiteLLM base URL: %s", settings.LITELLM_BASE_URL)

    # Initialize the shared HTTP client once before serving traffic
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if settings.LITELLM_API_KEY:
        headers["Authorization"] = f"Bearer {settings.LITELLM_API_KEY}"
    _http_client = httpx.AsyncClient(
        base_url=settings.LITELLM_BASE_URL,
        headers=headers,
        timeout=httpx.Timeout(settings.HTTP_TIMEOUT, connect=10.0),
    )

    # Load model metadata from YAML and enrich with Hugging Face API
    await initialize_model_metadata(_http_client)

    yield

    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        logger.info("HTTP client closed")


# ── FastAPI Application ─────────────────────────────────────────────────────

app = FastAPI(
    title="Ollama API Proxy",
    description="Ollama-compatible proxy routing to a LiteLLM backend",
    version=settings.VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request Transformation Helpers ──────────────────────────────────────────

def map_options_to_openai(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Map Ollama-style options to OpenAI-compatible parameters.

    Ollama options use keys like ``num_ctx`` and ``repeat_penalty``.
    OpenAI uses ``max_tokens``, ``frequency_penalty``, etc.
    We pass through the common keys and translate the well-known ones.
    """
    if not options:
        return {}

    param_map = {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "max_tokens": "max_tokens",
        "num_predict": "max_tokens",
        "repeat_penalty": "frequency_penalty",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
        "seed": "seed",
        "stop": "stop",
    }

    ignored_keys = {
        "num_ctx", "num_thread", "num_keep", "tfs_z", "typical_p",
        "mirostat", "mirostat_tau", "mirostat_eta", "penalize_newline"
    }

    result: Dict[str, Any] = {}
    for ollama_key, openai_key in param_map.items():
        if ollama_key in options and options[ollama_key] is not None and result.get(openai_key) is None:
            result[openai_key] = options[ollama_key]

    # Pass through any extra options that might be understood by LiteLLM
    for key, value in options.items():
        if key not in param_map and key not in ignored_keys:
            result[key] = value
        elif key in ignored_keys:
            logger.debug("Ignoring unsupported Ollama option: %s", key)

    return result


def build_openai_messages_generate(
    prompt: str, system: Optional[str]
) -> List[Dict[str, str]]:
    """Build OpenAI messages array for a generate request."""
    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def build_openai_messages_chat(
    messages: List[ChatMessage],
) -> List[Dict[str, Any]]:
    """Build OpenAI messages array from Ollama chat messages."""
    result: List[Dict[str, Any]] = []
    for msg in messages:
        entry: Dict[str, Any] = {"role": msg.role}
        if msg.images:
            content_list: List[Dict[str, Any]] = []
            if msg.content:
                content_list.append({"type": "text", "text": msg.content})
            for img in msg.images:
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                })
            entry["content"] = content_list
        else:
            entry["content"] = msg.content

        if msg.tool_calls:
            entry["tool_calls"] = msg.tool_calls
        result.append(entry)
    return result


def build_openai_format(format_val: Optional[Union[str, Dict[str, Any]]]) -> Optional[Any]:
    """Convert Ollama format parameter to OpenAI response_format."""
    if format_val is None:
        return None
    if isinstance(format_val, str) and format_val == "json":
        return {"type": "json_object"}
    if isinstance(format_val, dict):
        return {"type": "json_schema", "json_schema": format_val}
    return None


# ── Response Transformation Helpers ─────────────────────────────────────────

def now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def extract_usage_nanos(usage: Optional[Dict[str, Any]]) -> Dict[str, Optional[int]]:
    """Extract timing/count fields from an OpenAI usage object.

    Returns nanosecond-based durations (best-effort estimates).
    """
    if not usage:
        return {
            "prompt_eval_count": None,
            "eval_count": None,
            "prompt_eval_duration": None,
            "eval_duration": None,
            "total_duration": None,
            "load_duration": None,
        }

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")

    return {
        "prompt_eval_count": prompt_tokens,
        "eval_count": completion_tokens,
        # OpenAI doesn't expose per-phase durations, so we leave these as None
        "prompt_eval_duration": None,
        "eval_duration": None,
        "total_duration": None,
        "load_duration": None,
    }


def transform_chat_message(message: Dict[str, Any]) -> ChatMessage:
    """Transform an OpenAI message object to an Ollama ChatMessage."""
    return ChatMessage(
        role=message.get("role", "assistant"),
        content=message.get("content", ""),
        tool_calls=message.get("tool_calls"),
    )


# ── Streaming Helpers ───────────────────────────────────────────────────────

async def stream_generate_transform(
    http_client: httpx.AsyncClient,
    response: httpx.Response,
    model: str,
) -> AsyncGenerator[str, None]:
    """Transform OpenAI SSE stream into Ollama generate SSE format."""
    created_at = now_iso()
    start_time = time.monotonic()
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    final_usage: Optional[Dict[str, Any]] = None

    try:
        async for line in response.aiter_lines():
            line = line.strip()
            if not line:
                continue
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                # Emit final chunk
                elapsed_ns = int((time.monotonic() - start_time) * 1e9)
                chunk = {
                    "model": model,
                    "created_at": created_at,
                    "response": "",
                    "done": True,
                    "done_reason": "stop",
                    "total_duration": elapsed_ns,
                    "load_duration": 0,
                    "prompt_eval_count": prompt_tokens,
                    "eval_count": completion_tokens,
                }
                yield json.dumps(chunk) + "\n"
                return

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            # Capture usage only once from OpenAI's final chunk
            if "usage" in data:
                final_usage = data["usage"]
                prompt_tokens = data["usage"].get("prompt_tokens")
                completion_tokens = data["usage"].get("completion_tokens")

            choices = data.get("choices")
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content:
                yield json.dumps({
                    "model": model,
                    "created_at": created_at,
                    "response": content,
                    "done": False,
                }) + "\n"

        # Fallback final chunk if [DONE] was not received
        elapsed_ns = int((time.monotonic() - start_time) * 1e9)
        yield json.dumps({
            "model": model,
            "created_at": created_at,
            "response": "",
            "done": True,
            "done_reason": "stop",
            "total_duration": elapsed_ns,
            "load_duration": 0,
            "prompt_eval_count": prompt_tokens,
            "eval_count": completion_tokens,
        }) + "\n"
    finally:
        await response.aclose()


async def stream_chat_transform(
    http_client: httpx.AsyncClient,
    response: httpx.Response,
    model: str,
) -> AsyncGenerator[str, None]:
    """Transform OpenAI SSE stream into Ollama chat SSE format."""
    created_at = now_iso()
    start_time = time.monotonic()
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    final_usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None

    try:
        async for line in response.aiter_lines():
            line = line.strip()
            if not line:
                continue
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                # Emit final chunk
                elapsed_ns = int((time.monotonic() - start_time) * 1e9)
                chunk: Dict[str, Any] = {
                    "model": model,
                    "created_at": created_at,
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": finish_reason or "stop",
                    "total_duration": elapsed_ns,
                    "load_duration": 0,
                    "prompt_eval_count": prompt_tokens,
                    "eval_count": completion_tokens,
                }
                yield json.dumps(chunk) + "\n"
                return

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            # Capture usage from the authoritative backend response
            if "usage" in data:
                final_usage = data["usage"]
                prompt_tokens = data["usage"].get("prompt_tokens")
                completion_tokens = data["usage"].get("completion_tokens")

            # Capture finish_reason
            choices = data.get("choices")
            if choices and choices[0].get("finish_reason"):
                finish_reason = choices[0]["finish_reason"]

            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            content = delta.get("content", "")
            role = delta.get("role", "assistant")
            tool_calls = delta.get("tool_calls")

            message: Dict[str, Any] = {"role": role, "content": content}
            if tool_calls:
                message["tool_calls"] = tool_calls

            if content or tool_calls:
                yield json.dumps({
                    "model": model,
                    "created_at": created_at,
                    "message": message,
                    "done": False,
                }) + "\n"

        # Fallback final chunk
        elapsed_ns = int((time.monotonic() - start_time) * 1e9)
        yield json.dumps({
            "model": model,
            "created_at": created_at,
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": finish_reason or "stop",
            "total_duration": elapsed_ns,
            "load_duration": 0,
            "prompt_eval_count": prompt_tokens,
            "eval_count": completion_tokens,
        }) + "\n"
    finally:
        await response.aclose()

# ── Model Metadata Helper ───────────────────────────────────────────────────

def get_model_metadata(model_id: str) -> Dict[str, Any]:
    """Get metadata for a model from config.yaml with sensible defaults."""
    if model_id in MODEL_METADATA:
        return MODEL_METADATA[model_id]

    # Infer defaults for unknown models
    name_lower = model_id.lower()
    if "gpt" in name_lower:
        family = "gpt"
    elif "claude" in name_lower:
        family = "claude"
    elif "gemini" in name_lower:
        family = "gemini"
    elif "llama" in name_lower:
        family = "llama"
    elif "mistral" in name_lower:
        family = "mistral"
    elif "qwen" in name_lower:
        family = "qwen2"
    else:
        family = "unknown"

    return {
        "family": family,
        "parameter_size": "8B",
        "quantization_level": "Q4_0",
        "format": "gguf",
        "size": 0,
        "context_length": 2048,
        "parameters": "",
        "template": "",
        "model_info": {},
        "capabilities": ["completion"],
    }


# ── LiteLLM /v1/models -> Ollama /api/tags transformation ──────────────────

def transform_litellm_models(data: Dict[str, Any]) -> ListTagsResponse:
    """Transform LiteLLM /v1/models response to Ollama /api/tags format.

    Only includes models that are present in both the LiteLLM response and
    config.yaml (MODEL_METADATA). Models from LiteLLM without a config entry
    are silently dropped.
    
    Each model name includes the ':latest' tag suffix to match Ollama's
    naming convention (e.g. 'llama3.2:latest').
    """
    models: List[ModelInfo] = []
    for m in data.get("data", []):
        id_ = m.get("id", "")

        # Skip models not defined in config.yaml
        if id_ not in MODEL_METADATA:
            logger.debug("Skipping LiteLLM model %r (not in config.yaml)", id_)
            continue

        metadata = MODEL_METADATA[id_]
        
        # Ollama uses model:tag format, where tag defaults to 'latest'
        model_name = f"{id_}:latest"
        digest = "sha256:" + hashlib.sha256(model_name.encode()).hexdigest()

        created_val = m.get("created")
        if created_val is None:
            created_val = 0

        model_info = ModelInfo(
            name=model_name,
            model=model_name,
            modified_at=datetime.fromtimestamp(
                created_val, tz=timezone.utc
            ).isoformat(),
            size=metadata.get("size", 0),
            digest=digest,
            details=ModelDetails(
                format=metadata.get("format", "gguf"),
                family=metadata["family"],
                families=[metadata["family"]],
                parameter_size=metadata.get("parameter_size", "8B"),
                quantization_level=metadata.get("quantization_level", "Q4_0"),
            ),
        )
        models.append(model_info)
    return ListTagsResponse(models=models)


# ── Error Handling Helper ───────────────────────────────────────────────────

def handle_litellm_error(response: httpx.Response) -> None:
    """Raise an HTTPException based on a LiteLLM error response."""
    status_code = response.status_code
    try:
        error_data = response.json()
    except Exception:
        error_data = {}

    detail = error_data.get("error", {})
    if isinstance(detail, dict):
        message = detail.get("message", response.text)
    else:
        message = str(detail) or response.text

    logger.error("LiteLLM error %d: %s", status_code, message)
    raise HTTPException(status_code=status_code, detail=message)


# ── API Endpoints ───────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Ollama API proxy"}


@app.head("/")
async def root_head():
    """HEAD health check."""
    return JSONResponse(status_code=200, content={})


@app.get("/api/version", response_model=VersionResponse)
async def version():
    """Return the Ollama version."""
    return VersionResponse(version=settings.VERSION)


@app.get("/api/tags", response_model=ListTagsResponse)
async def list_tags():
    """List available models by fetching from LiteLLM /v1/models."""
    client = await get_http_client()
    try:
        response = await client.get("/v1/models")
        if response.status_code == 200:
            data = response.json()
            return transform_litellm_models(data)
        # Surface client errors (4xx) to the caller; gracefully fallback on 5xx
        if 400 <= response.status_code < 500:
            logger.error("LiteLLM models endpoint returned %d", response.status_code)
            raise HTTPException(
                status_code=response.status_code,
                detail=f"LiteLLM models endpoint returned {response.status_code}",
            )
        logger.warning("Failed to fetch models from LiteLLM: %d", response.status_code)
        return ListTagsResponse(models=[])
    except httpx.ConnectError as e:
        logger.error("Cannot connect to LiteLLM proxy: %s", e)
        raise HTTPException(
            status_code=502,
            detail="Backend proxy unavailable",
        )
    except httpx.TimeoutException as e:
        logger.error("Timeout connecting to LiteLLM proxy: %s", e)
        raise HTTPException(
            status_code=504,
            detail="Backend proxy timeout",
        )


@app.post("/api/generate")
async def generate(request: GenerateRequest):
    """Generate a completion (Ollama /api/generate endpoint).

    Transforms the request to OpenAI chat completions format and routes
    to the LiteLLM proxy.
    """
    logger.info("Generate request: model=%s, stream=%s", request.model, request.stream)

    # Handle empty prompt (model load check)
    if not request.prompt:
        logger.info("Empty prompt for model %s – returning load response", request.model)
        return GenerateResponse(
            model=request.model,
            created_at=now_iso(),
            response="",
            done=True,
            done_reason="load",
        )

    client = await get_http_client()
    messages = build_openai_messages_generate(request.prompt, request.system)
    openai_params = map_options_to_openai(request.options)
    fmt = build_openai_format(request.format)

    body: Dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        **openai_params,
    }
    if fmt:
        body["response_format"] = fmt

    try:
        if request.stream:
            response = await client.post(
                "/v1/chat/completions",
                json=body,
                stream=True,
            )
            if response.status_code != 200:
                await handle_litellm_error(response)
            return StreamingResponse(
                stream_generate_transform(client, response, request.model),
                media_type="text/event-stream",
            )
        else:
            response = await client.post(
                "/v1/chat/completions",
                json=body,
            )
            if response.status_code != 200:
                await handle_litellm_error(response)
            data = response.json()
            usage_info = extract_usage_nanos(data.get("usage"))
            choice = data["choices"][0]
            content = choice["message"].get("content", "")
            return GenerateResponse(
                model=request.model,
                created_at=now_iso(),
                response=content,
                done=True,
                done_reason=choice.get("finish_reason"),
                **usage_info,
            )
    except httpx.ConnectError as e:
        logger.error("Cannot connect to LiteLLM proxy: %s", e)
        raise HTTPException(status_code=502, detail="Backend proxy unavailable")
    except httpx.TimeoutException as e:
        logger.error("Timeout connecting to LiteLLM proxy: %s", e)
        raise HTTPException(status_code=504, detail="Backend proxy timeout")


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat completion (Ollama /api/chat endpoint).

    Transforms the request to OpenAI chat completions format and routes
    to the LiteLLM proxy.
    """
    logger.info("Chat request: model=%s, stream=%s, messages=%d",
                 request.model, request.stream, len(request.messages))

    # Handle empty messages (model load check)
    if not request.messages:
        logger.info("Empty messages for model %s – returning load response", request.model)
        return ChatResponse(
            model=request.model,
            created_at=now_iso(),
            message=ChatMessage(role="assistant", content=""),
            done=True,
            done_reason="load",
        )

    client = await get_http_client()
    messages = build_openai_messages_chat(request.messages)
    openai_params = map_options_to_openai(request.options)
    fmt = build_openai_format(request.format)

    body: Dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        **openai_params,
    }
    if fmt:
        body["response_format"] = fmt
    if request.tools:
        body["tools"] = request.tools

    try:
        if request.stream:
            response = await client.post(
                "/v1/chat/completions",
                json=body,
                stream=True,
            )
            if response.status_code != 200:
                await handle_litellm_error(response)
            return StreamingResponse(
                stream_chat_transform(client, response, request.model),
                media_type="text/event-stream",
            )
        else:
            response = await client.post(
                "/v1/chat/completions",
                json=body,
            )
            if response.status_code != 200:
                await handle_litellm_error(response)
            data = response.json()
            usage_info = extract_usage_nanos(data.get("usage"))
            choice = data["choices"][0]
            msg = transform_chat_message(choice["message"])
            return ChatResponse(
                model=request.model,
                created_at=now_iso(),
                message=msg,
                done=True,
                done_reason=choice.get("finish_reason"),
                **usage_info,
            )
    except httpx.ConnectError as e:
        logger.error("Cannot connect to LiteLLM proxy: %s", e)
        raise HTTPException(status_code=502, detail="Backend proxy unavailable")
    except httpx.TimeoutException as e:
        logger.error("Timeout connecting to LiteLLM proxy: %s", e)
        raise HTTPException(status_code=504, detail="Backend proxy timeout")


@app.post("/api/embed", response_model=EmbeddingResponse)
async def embed(request: EmbeddingRequest):
    """Generate embeddings (Ollama /api/embed endpoint)."""
    logger.info("Embed request: model=%s", request.model)

    client = await get_http_client()
    input_list: List[str] = (
        [request.input] if isinstance(request.input, str) else request.input
    )

    body: Dict[str, Any] = {
        "model": request.model,
        "input": input_list,
    }
    if request.truncate is not None:
        body["truncate"] = request.truncate

    try:
        response = await client.post("/v1/embeddings", json=body)
        if response.status_code != 200:
            await handle_litellm_error(response)
        data = response.json()
        embeddings = [item["embedding"] for item in data.get("data", [])]
        usage_info = extract_usage_nanos(data.get("usage"))
        return EmbeddingResponse(
            model=request.model,
            embeddings=embeddings,
            prompt_eval_count=usage_info.get("prompt_eval_count"),
        )
    except httpx.ConnectError as e:
        logger.error("Cannot connect to LiteLLM proxy: %s", e)
        raise HTTPException(status_code=502, detail="Backend proxy unavailable")
    except httpx.TimeoutException as e:
        logger.error("Timeout connecting to LiteLLM proxy: %s", e)
        raise HTTPException(status_code=504, detail="Backend proxy timeout")


@app.post("/api/show", response_model=ShowModelResponse)
async def show(request: ShowModelRequest):
    """Show model information with metadata from config.yaml."""
    logger.info("Show request: model=%s", request.model)

    metadata = get_model_metadata(request.model)

    return ShowModelResponse(
        modelfile=f"FROM {request.model}\n",
        parameters=metadata.get("parameters", ""),
        template=metadata.get("template", ""),
        license="",
        details=ModelDetails(
            format=metadata.get("format", "gguf"),
            family=metadata["family"],
            families=[metadata["family"]],
            parameter_size=metadata.get("parameter_size", "8B"),
            quantization_level=metadata.get("quantization_level", "Q4_0"),
        ),
        model_info=metadata.get("model_info", {}),
        capabilities=metadata.get("capabilities", ["completion"]),
    )


@app.get("/api/ps", response_model=PsResponse)
async def ps():
    """List running models.

    Returns empty list since we don't manage local model loading.
    """
    return PsResponse(models=[])


# ── Unsupported Endpoints (501 Not Implemented) ─────────────────────────────

@app.post("/api/create")
async def create_model(request: CreateModelRequest):
    """Create a model – not supported."""
    logger.warning("POST /api/create is not supported")
    raise HTTPException(
        status_code=501,
        detail="Creating models is not supported in this proxy implementation",
    )


@app.post("/api/copy")
async def copy_model(request: CopyModelRequest):
    """Copy a model – not supported."""
    logger.warning("POST /api/copy is not supported")
    raise HTTPException(
        status_code=501,
        detail="Copying models is not supported in this proxy implementation",
    )


@app.delete("/api/delete")
async def delete_model(request: DeleteModelRequest):
    """Delete a model – not supported."""
    logger.warning("DELETE /api/delete is not supported")
    raise HTTPException(
        status_code=501,
        detail="Deleting models is not supported in this proxy implementation",
    )


@app.post("/api/pull")
async def pull_model(request: PullModelRequest):
    """Pull a model – not supported."""
    logger.warning("POST /api/pull is not supported")
    raise HTTPException(
        status_code=501,
        detail="Pulling models is not supported in this proxy implementation",
    )


@app.post("/api/push")
async def push_model(request: PushModelRequest):
    """Push a model – not supported."""
    logger.warning("POST /api/push is not supported")
    raise HTTPException(
        status_code=501,
        detail="Pushing models is not supported in this proxy implementation",
    )
