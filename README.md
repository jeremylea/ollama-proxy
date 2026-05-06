# Ollama API Proxy

A lightweight proxy server that implements the [Ollama API specification](https://github.com/ollama/ollama/blob/main/docs/api.md) and routes all requests to a configurable LiteLLM proxy backend. Designed as a drop-in replacement for Ollama, primarily targeting GitHub Copilot in VSCode.

## Architecture

```
GitHub Copilot (VSCode)  →  Ollama Proxy (this)  →  LiteLLM Proxy Server
     (Ollama API)                (transforms)            (OpenAI API)
```

The proxy:
- Accepts Ollama-compatible requests on port 11434
- Transforms them to OpenAI-compatible format
- Forwards to a LiteLLM proxy server
- Transforms responses back to Ollama format
- Supports streaming (SSE) for both `/api/generate` and `/api/chat`

## Prerequisites

- Python 3.8+
- A running LiteLLM proxy server (e.g., `litellm --host 0.0.0.0 --port 4000`)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

## Configuration

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# LiteLLM Proxy Configuration
LITELLM_BASE_URL=http://localhost:4000
LITELLM_API_KEY=your_api_key_here

# Server Configuration
PORT=11434
HOST=0.0.0.0
LOG_LEVEL=info
HTTP_TIMEOUT=300
```

## Usage

```bash
# Start the server
python run.py
```

The server will be available at `http://localhost:11434`.

## API Endpoints

### Implemented

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET/HEAD | Health check |
| `/api/version` | GET | Return version info |
| `/api/tags` | GET | List models (from LiteLLM `/v1/models`) |
| `/api/generate` | POST | Generate completions |
| `/api/chat` | POST | Chat completions |
| `/api/embed` | POST | Generate embeddings |
| `/api/show` | POST | Show model info |
| `/api/ps` | GET | List running models |

### Not Implemented (501)

| Endpoint | Method |
|----------|--------|
| `/api/create` | POST |
| `/api/copy` | POST |
| `/api/delete` | DELETE |
| `/api/pull` | POST |
| `/api/push` | POST |

## Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_api_endpoints.py -v
```

## Project Structure

```
ollama-proxy/
├── .env.example          # Environment variable template
├── requirements.txt       # Production dependencies
├── requirements-dev.txt   # Development dependencies
├── pytest.ini            # Pytest configuration
├── run.py                # Entry point
├── app/
│   ├── __init__.py
│   ├── config.py         # Configuration from environment
│   ├── main.py           # FastAPI application and endpoints
│   └── models.py         # Pydantic request/response models
└── tests/
    ├── conftest.py       # Pytest fixtures
    ├── test_api_endpoints.py
    └── test_streaming.py
```

## License

MIT
