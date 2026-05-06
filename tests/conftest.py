"""
Pytest configuration file with fixtures for testing.
"""

import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

# Add the parent directory to the path so we can import the app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app


@pytest.fixture
def test_client():
    """Create a FastAPI test client for the app."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def mock_httpx_response():
    """Factory fixture to create mock httpx responses."""
    def _create(status_code=200, json_data=None, stream=False):
        response = MagicMock(spec=MagicMock)
        response.status_code = status_code
        response.json.return_value = json_data or {}
        response.aiter_lines = AsyncMock(return_value=iter([]))
        return response
    return _create


@pytest.fixture
def mock_httpx_client(mock_httpx_response):
    """Mock the httpx.AsyncClient used by the app."""
    mock_client = AsyncMock()
    mock_client.is_closed = False
    mock_client.aclose = AsyncMock()
    return mock_client
