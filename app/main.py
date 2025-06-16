import logging
import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langfuse import Langfuse
from contextlib import asynccontextmanager

from .rag_service import RAGService
from .schemas import QueryRequest, QueryResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
)

pdf_dir = "/app/docs"

ANSWER_MODE = os.getenv("ANSWER_MODE", "strict")

rag_service = RAGService(
    pdf_paths=[os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")],
    langfuse=langfuse,
    answer_mode=ANSWER_MODE
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Running initial document ingestion on startup")
    await rag_service.ingest_documents()
    yield

app = FastAPI(title="RAG Microservice", lifespan=lifespan)

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Returns service status, last ingestion time, and current answer mode.
    """
    logger.info("Health check requested")
    last_ingested = (
        rag_service.last_ingested.isoformat() + "Z"
        if rag_service.last_ingested else "Never"
    )
    return {
        "status": "OK",
        "last_ingested": last_ingested,
        "message": (
            f"System is ready to answer questions based on the provided documents. "
            f"Currently in {rag_service.answer_mode} mode."
        )
    }

@app.post("/ingest")
async def ingest_documents():
    """
    Endpoint to manually trigger document ingestion and re-indexing.
    Returns a success message if ingestion completes successfully.
    """
    logger.info("Ingest endpoint called")
    try:
        await rag_service.ingest_documents()
        logger.info("Ingestion completed successfully")
        return {"status": "Documents ingested successfully"}
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    
@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Endpoint to answer user questions using the RAG pipeline.
    Accepts a question and returns an answer with supporting sources.
    """
    logger.info(f"Query endpoint called with question: {request.question}")
    try:
        result = await rag_service.query(request.question)
        logger.info("Query processed successfully")
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")