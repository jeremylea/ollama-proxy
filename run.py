"""Entry point for the Ollama API proxy server."""

import uvicorn
from app.config import settings

if __name__ == "__main__":
    print(f"Starting Ollama API proxy on {settings.HOST}:{settings.PORT}")
    print(f"LiteLLM backend: {settings.LITELLM_BASE_URL}")

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL,
    )
