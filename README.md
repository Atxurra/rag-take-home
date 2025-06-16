# RAG Microservice with Langfuse Observability

## Overview
This project is a Python-based Retrieval-Augmented Generation (RAG) microservice that ingests PDF documents, answers user queries using an LLM (OpenAI), and provides full observability via Langfuse. It is designed for modularity, traceability, and easy deployment with Docker Compose.

## Features
- Ingests and indexes PDFs (e.g., Think Python, PEP 8) on startup and on demand
- Answers questions using only the provided documents (strict mode) or with general knowledge fallback (flexible mode)
- Returns answers with cited sources (book and page)
- Full observability with Langfuse spans for ingestion, embedding, search, and LLM calls
- Structured logging and health endpoint

## Requirements
- Docker & Docker Compose
- OpenAI API key
- Langfuse API keys (public & secret)

## Setup & Usage

### 1. Clone the repository
```sh
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Add your PDFs to the `docs/` folder
Place your source PDFs (e.g., `think_python.pdf`, `pep8.pdf`) in the `docs/` directory.

### 3. Create a `.env` file in the root directory
```env
OPENAI_API_KEY=sk-...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://us.cloud.langfuse.com
ANSWER_MODE=strict  # or flexible (Optional, defaults to strict)
```

### 4. Build and run the service (Docker Compose)
To build and start the API service:
```sh
docker-compose up --build
```
- The API will be available at [http://localhost:8000](http://localhost:8000)
- The service will ingest and index all PDFs in `docs/` on startup.

### 5. API Endpoints
- `GET /health` — Service status, last ingestion, mode
- `POST /ingest` — Re-ingest and re-index documents
- `POST /query` — Ask a question
  Example request:
  ```json
  { "question": "What is a function in Python?" }
  ```
  Example response:
  ```json
  {
    "answer": "According to 'Think Python', page 49: ...",
    "sources": [
      { "book": "Think Python", "page": 49, "text": "..." }
    ]
  }
  ```

### 6. Run the test suite (inside Docker)
To run all unit tests and see coverage:
```sh
docker compose run --rm rag-service pytest tests/test_api.py
```
- Coverage reports will be generated in the `htmlcov/` directory.

### 7. (Optional) Run tests locally (without Docker)
If you have Python 3.9+ and all dependencies installed:
```sh
pip install -r requirements.txt
pytest tests/
```

## Project Structure

```
app/
  main.py
  rag_service.py
  ingest_service.py
  prompt_templates.py
  schemas.py
  ...
docs/
  think_python.pdf
  pep8.pdf
Dockerfile
requirements.txt
docker-compose.yml
README.md
```

## Notes
- By default, the service runs in strict mode (answers only from provided documents). Set `ANSWER_MODE=flexible` in `.env` to allow general knowledge fallback.
- All API keys and secrets must be set in `.env` (never commit secrets to git).
- Langfuse observability is enabled for all major operations.
- To stop the service, press `Ctrl+C` or run `docker-compose down`.

## License
MIT
