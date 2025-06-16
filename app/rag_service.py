import logging
from typing import List
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langfuse import Langfuse
from langchain_core.messages import SystemMessage, HumanMessage
import tiktoken
from .prompt_templates import STRICT_PROMPT, FLEXIBLE_PROMPT

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
        langfuse: Langfuse,
        answer_mode: str = "strict"
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
        self.answer_mode = answer_mode
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
    
    async def query(self, question: str):
        """
        Answers a user question using the RAG pipeline.
        Performs similarity search, synthesizes an answer with an LLM, and returns the answer with supporting sources.
        All steps are traced with Langfuse.

        Args:
            question (str): The user's question.
        Returns:
            dict: Dictionary with 'answer' and 'sources' (list of book/page/text dicts).
        """
        if not self.vector_store:
            raise Exception("Vector store not initialized. Please ingest documents first.")
        
        # Start only one query_documents span per query
        with self.langfuse.start_as_current_span(name="query_documents") as root_span:
            try:
                with self.langfuse.start_as_current_span(name="similarity_search") as sim_span:
                    logger.info("Performing similarity search")
                    results = self.vector_store.similarity_search(question, k=3)
                    sim_span.update(
                        input=question,
                        output=[{
                            "book": doc.metadata.get("source", "Unknown"),
                            "page": doc.metadata.get("page", "N/A"),
                            "text": doc.page_content
                        } for doc in results]
                    )

                formatted_context = "\n".join([
                    f"[{doc.metadata.get('source', 'Unknown')}, page {doc.metadata.get('page', 'N/A')}]: {doc.page_content}"
                    for doc in results
                ])

                with self.langfuse.start_as_current_span(name="llm_response") as llm_span:
                    logger.info("Generating answer using LLM")
                    retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
                    llm = ChatOpenAI(temperature=0)
                    if self.answer_mode == "strict":
                        prompt_template = PromptTemplate(
                            input_variables=["context", "question"],
                            template=STRICT_PROMPT
                        )
                    else:
                        prompt_template = PromptTemplate(
                            input_variables=["context", "question"],
                            template=FLEXIBLE_PROMPT
                        )
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        retriever=retriever,
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": prompt_template}
                    )
                    prompt_text = prompt_template.format(context=formatted_context, question=question)
                    model_name = llm.model_name or "unknown"
                    response = qa_chain({"context": formatted_context, "query": question})
                    answer = response["result"]
                    source_docs = response["source_documents"]
                    # Token usage (may not be visible in Langfuse UI)
                    input_tokens = llm.get_num_tokens(prompt_text)
                    output_tokens = llm.get_num_tokens(answer)
                    llm_span.update(
                        input={"context": formatted_context, "question": question},
                        output=answer,
                        model=model_name,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens
                    )

                sources = []
                for doc in source_docs:
                    sources.append({
                        "book": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", None),
                        "text": doc.page_content
                    })

                root_span.update(
                    output={"answer": answer, "sources": sources}
                )
                return {"answer": answer, "sources": sources}

            except Exception as e:
                logger.error(f"Query error: {str(e)}")
                root_span.update(
                    output={"error": str(e)},
                    level="ERROR"
                )
                raise