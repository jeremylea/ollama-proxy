"""
Pydantic models for the Ollama API proxy.

Matches the Ollama API specification:
https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union


# ── Request Models ──────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    """Request model for POST /api/generate"""
    model: str
    prompt: str = ""
    suffix: Optional[str] = None
    images: Optional[List[str]] = None
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    stream: bool = True
    raw: bool = False
    keep_alive: Optional[str] = None
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)
    format: Optional[Union[str, Dict[str, Any]]] = None


class ChatMessage(BaseModel):
    """A single chat message."""
    role: str
    content: str = ""
    images: Optional[List[str]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatRequest(BaseModel):
    """Request model for POST /api/chat"""
    model: str
    messages: List[ChatMessage] = []
    stream: bool = True
    tools: Optional[List[Dict[str, Any]]] = None
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)
    format: Optional[Union[str, Dict[str, Any]]] = None
    keep_alive: Optional[str] = None


class EmbeddingRequest(BaseModel):
    """Request model for POST /api/embed"""
    model: str
    input: Union[str, List[str]]
    truncate: Optional[bool] = None
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)
    keep_alive: Optional[str] = None


class ShowModelRequest(BaseModel):
    """Request model for POST /api/show"""
    model: str = ""
    verbose: Optional[bool] = False


class CreateModelRequest(BaseModel):
    """Request model for POST /api/create (stub)"""
    model: str
    path: Optional[str] = None
    modelfile: Optional[str] = None
    stream: Optional[bool] = False


class CopyModelRequest(BaseModel):
    """Request model for POST /api/copy (stub)"""
    source: str
    destination: str


class DeleteModelRequest(BaseModel):
    """Request model for DELETE /api/delete (stub)"""
    model: str


class PullModelRequest(BaseModel):
    """Request model for POST /api/pull (stub)"""
    model: str
    insecure: Optional[bool] = False
    stream: Optional[bool] = False


class PushModelRequest(BaseModel):
    """Request model for POST /api/push (stub)"""
    model: str
    insecure: Optional[bool] = False
    stream: Optional[bool] = False


# ── Response Models ─────────────────────────────────────────────────────────

class GenerateResponse(BaseModel):
    """Response model for POST /api/generate"""
    model: str
    created_at: str
    response: str = ""
    done: bool = False
    done_reason: Optional[str] = None
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class ChatResponse(BaseModel):
    """Response model for POST /api/chat"""
    model: str
    created_at: str
    message: ChatMessage
    done: bool = False
    done_reason: Optional[str] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class ModelDetails(BaseModel):
    """Model technical details."""
    parent_model: Optional[str] = None
    format: Optional[str] = None
    family: Optional[str] = None
    families: Optional[List[str]] = None
    parameter_size: Optional[str] = None
    quantization_level: Optional[str] = None


class ModelInfo(BaseModel):
    """Model information for /api/tags response."""
    name: str
    modified_at: str
    size: int
    digest: str
    details: ModelDetails


class ListTagsResponse(BaseModel):
    """Response model for GET /api/tags"""
    models: List[ModelInfo] = []


class EmbeddingResponse(BaseModel):
    """Response model for POST /api/embed"""
    model: str
    embeddings: List[List[float]]
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None


class ShowModelResponse(BaseModel):
    """Response model for POST /api/show"""
    modelfile: Optional[str] = None
    license: Optional[str] = None
    parameters: Optional[str] = None
    template: Optional[str] = None
    details: Optional[ModelDetails] = None
    model_info: Optional[Dict[str, Any]] = None
    capabilities: Optional[List[str]] = None


class PsModelInfo(BaseModel):
    """Model info for /api/ps response."""
    name: str
    model: str
    size: int
    digest: str
    details: ModelDetails
    expires_at: Optional[str] = None
    size_vram: Optional[int] = None


class PsResponse(BaseModel):
    """Response model for GET /api/ps"""
    models: List[PsModelInfo] = []


class VersionResponse(BaseModel):
    """Response model for GET /api/version"""
    version: str
