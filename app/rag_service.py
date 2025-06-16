import logging
from typing import List
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbedding
from langchain_community.vectorstores import FAISS
from langfuse import Langfuse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RAGService:
    """
    Retrieval-Augmented Generation (RAG) service for document ingestion, indexing, and question answering.
    Handles PDF loading, chunking, embedding, vector storage, and LLM-based answer synthesis with observability via Langfuse.
    """
    def __init__(
        self,
        pdf_paths: List[str],
        langfuse: Langfuse
    ):
        """
        Initialize the RAGService.

        Args:
            pdf_paths (List[str]): List of PDF file paths to ingest.
            langfuse (Langfuse): Langfuse client for observability and tracing.
            answer_mode (str, optional): Answering mode ('strict' or 'flexible'). Defaults to 'strict'.
        """
        self.pdf_paths = pdf_paths
        self.langfuse = langfuse
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.last_ingested: datetime | None = None

    async def ingest_documents(self):
        """
        Ingests and indexes documents by loading PDFs, splitting into chunks, generating embeddings,
        and storing them in a vector store. Traces each step with Langfuse and updates the last ingestion time.
        Returns a status message on success.
        """
        with self.langfuse.start_as_current_span(name="ingest_documents") as root_span:
            try:
                with self.langfuse.start_as_current_span(name="load_pdfs") as load_pdfs_span:
                    logger.info("Loading PDFs")
                    documents = []
                    for pdf_path in self.pdf_paths:
                        logger.info(f"Loading PDF: {pdf_path}")
                        loader = PyPDFLoader(pdf_path)
                        docs = loader.load()
                        documents.extend(docs)
                    # Log input and output to Langfuse
                    load_pdfs_span.update(
                        input=self.pdf_paths,
                        output={"document_count": len(documents)}
                    )

                with self.langfuse.start_as_current_span(name="split_documents") as split_span:
                    logger.info("Splitting documents into chunks")
                    chunks = self.text_splitter.split_documents(documents)
                    # Log input and output to Langfuse
                    split_span.update(
                        input=[doc.page_content for doc in documents],
                        output={"chunk_count": len(chunks)}
                    )

                with self.langfuse.start_as_current_span(name="embed_and_index") as embed_span:
                    logger.info("Generating embeddings and indexing")
                    chunk_texts = [doc.page_content for doc in chunks]
                    self.vector_store = FAISS.from_documents(chunks, self.embeddings)
                    # Log input and output to Langfuse
                    embed_span.update(
                        input=chunk_texts,
                        output={"index_size": len(chunk_texts)}
                    )

                try:
                    self.langfuse.flush()
                    logger.info("Langfuse events successfully flushed")
                    self.last_ingested = datetime.utcnow()
                    return {"status": "Documents ingested and traced successfully"}
                except Exception as flush_error:
                    logger.error(f"Langfuse flush failed: {str(flush_error)}")
                    raise Exception(f"Ingestion completed but Langfuse tracing failed: {str(flush_error)}")

            except Exception as e:
                logger.error(f"Ingestion error: {str(e)}")
                # Log error to Langfuse
                root_span.update(
                    output={"error": str(e)},
                    level="ERROR"
                )
                raise
            finally:
                pass