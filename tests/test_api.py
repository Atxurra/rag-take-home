from fastapi.testclient import TestClient
import pytest
import os
from datetime import datetime
from unittest.mock import AsyncMock, patch
from app.main import app, rag_service
from app.rag_service import RAGService

# Initialize the FastAPI test client
client = TestClient(app)

# Fixtures for setup and teardown
@pytest.fixture(autouse=True)
async def setup_rag_service():
    """Fixture to mock RAGService dependencies and reset state before each test."""
    # Mock environment variables
    with patch.dict(os.environ, {
        "LANGFUSE_PUBLIC_KEY": "test_public_key",
        "LANGFUSE_SECRET_KEY": "test_secret_key",
        "LANGFUSE_HOST": "https://test.langfuse.com",
        "ANSWER_MODE": "strict"
    }):
        # Mock Langfuse client
        with patch("app.main.Langfuse") as mock_langfuse:
            mock_langfuse_instance = mock_langfuse.return_value
            mock_langfuse_instance.start_as_current_span = AsyncMock()
            mock_langfuse_instance.flush = AsyncMock()

            # Mock RAGService methods
            with patch.object(RAGService, "ingest_documents", new=AsyncMock()) as mock_ingest:
                with patch.object(RAGService, "query", new=AsyncMock()) as mock_query:
                    # Set up mock return values
                    mock_ingest.return_value = {"status": "Documents ingested successfully"}
                    mock_query.return_value = {
                        "answer": "A function is a reusable block of code in Python.",
                        "sources": [
                            {"book": "python_doc.pdf", "page": 10, "text": "Functions are defined using def."}
                        ]
                    }
                    rag_service.last_ingested = datetime.utcnow()
                    yield

@pytest.mark.asyncio
async def test_health():
    """Test the /health endpoint returns status, last_ingested, and message."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OK"
    assert isinstance(data["last_ingested"], str)
    assert "message" in data
    assert "System is ready" in data["message"]
    assert "strict mode" in data["message"].lower()

@pytest.mark.asyncio
async def test_ingest_success():
    """Test the /ingest endpoint triggers ingestion and returns success."""
    response = client.post("/ingest")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Documents ingested successfully"
    assert rag_service.ingest_documents.called  # Verify the method was called

@pytest.mark.asyncio
async def test_ingest_failure():
    """Test the /ingest endpoint handles ingestion failure."""
    with patch.object(RAGService, "ingest_documents", side_effect=Exception("Ingestion error")):
        response = client.post("/ingest")
        assert response.status_code == 500
        assert "Ingestion failed" in response.json()["detail"]

@pytest.mark.asyncio
async def test_query_valid():
    """Test /query returns a valid answer and sources after ingestion."""
    response = client.post("/query", json={"question": "What is a function in Python?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["answer"] == "A function is a reusable block of code in Python."
    assert "sources" in data
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) == 1
    assert data["sources"][0]["book"] == "python_doc.pdf"
    assert data["sources"][0]["page"] == 10
    assert "text" in data["sources"][0]
    assert rag_service.query.called  # Verify the method was called

@pytest.mark.asyncio
async def test_query_invalid_payload():
    """Test /query returns 422 for missing question field."""
    response = client.post("/query", json={})
    assert response.status_code == 422
    assert "question" in response.json()["detail"][0]["loc"]

@pytest.mark.asyncio
async def test_query_empty_question():
    """Test /query handles empty question gracefully."""
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert isinstance(data["sources"], list)
    assert rag_service.query.called

@pytest.mark.asyncio
async def test_query_failure():
    """Test /query handles processing failure."""
    with patch.object(RAGService, "query", side_effect=Exception("Query error")):
        response = client.post("/query", json={"question": "What is a function in Python?"})
        assert response.status_code == 500
        assert "Query failed" in response.json()["detail"]