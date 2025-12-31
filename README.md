start the app use

# Winery AI Chatbot

A Concirage FastAPI Chatbot backend that provides a multi-tenant, vector-search-enabled chatbot service. The project integrates Milvus for vector storage, PostgreSQL for relational data and OpenAI for embeddings and LLM responses.

## Features

- Multi-tenant document storage (PostgreSQL + Milvus)
- Document embedding generation via OpenAI
- Pre-prompts per tenant
- Dialog tree (state machine) storage per tenant
- Simple REST API for documents, preprompts, and dialog trees

## Project layout

- `main.py` — FastAPI application and route definitions (entrypoint).
- `ai/openai_integration.py` — OpenAI client wrappers: embedding and completion helpers (sync and streaming variants).
- `config/settings.py` — Application configuration (Pydantic settings) and connection helpers for PostgreSQL and Milvus.
- `core/database.py` — Database manager that initializes Milvus & PostgreSQL and provides Milvus store helpers.
- `database/database_function.py` — Small helper to get a PostgreSQL connection from the pool.
- `endpoints/api_endpoints.py` — Core business logic for documents, preprompts, and dialog trees.
- `models/pydantic_models.py` — Request/response Pydantic models used by FastAPI.
- `lifecycle/app_lifecycle.py` — FastAPI lifespan context: initializes DBs at startup and cleans up on shutdown.
- `utils/utility_function.py` — Utility helpers for documents and metadata parsing.
- `utils/logger.py` — Central logger configuration used across modules.
- `docker-compose.yml` — Optional compose file (if provided) to run dependencies like Postgres and Milvus.

## Requirements

This project expects a Python 3.11+ environment. Use the existing `requirements.txt` (if present) or install dependencies shown below.

Install dependencies (example using pip):

```bash
python -m pip install -r requirements.txt
```

If you do not have `requirements.txt`, here are the likely required packages:

```bash
python -m pip install fastapi uvicorn openai pymilvus psycopg_pool pydantic pydantic-settings python-dotenv structlog
```

## Configuration

Most settings are loaded from environment variables or a `.env` file via `config/settings.py` using Pydantic Settings.

Key environment variables (defaults live in `config/settings.py`):

- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` — PostgreSQL connection
- `MILVUS_HOST`, `MILVUS_PORT` — Milvus vector database
- `OPENAI_API_KEY` — OpenAI API key (required for embeddings and answers)
- `.env` file is supported and loaded automatically via `python-dotenv`.

Example `.env` (not committed):

```ini
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=winery-bot
POSTGRES_USER=postgres
POSTGRES_PASSWORD=yourpassword
MILVUS_HOST=localhost
MILVUS_PORT=19530
OPENAI_API_KEY=sk-...
```

## Database initialization

On startup the app ensures required PostgreSQL tables and Milvus collections exist. The logic lives in `core/database.py` and runs during FastAPI lifespan from `lifecycle/app_lifecycle.py`.

Ensure PostgreSQL and Milvus instances are reachable before starting the app.

## Running the application

Run locally with Uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

If you use `docker-compose.yml`, start supporting services (Postgres, Milvus) and then run the app.

## API overview

Base URL: `/`

- `GET /` — health/info endpoint
- `GET /api/v1/documents` — list documents (optional `tenant_id` query)
- `POST /api/v1/documents` — create document: expects `title`, `content`, `tenant_id`, `metadata`
- `GET /api/v1/preprompts` — list tenant preprompts (optional `tenant_id`)
- `POST /api/v1/preprompts` — create tenant preprompt
- `PUT /api/v1/preprompts/{preprompt_id}` — update preprompt
- `DELETE /api/v1/preprompts/{preprompt_id}` — delete preprompt
- `GET /api/v1/dialog_trees` — list dialog trees (optional `tenant_id`)
- `POST /api/v1/dialog_trees` — create dialog tree
- `PUT /api/v1/dialog_trees/{dialog_tree_id}` — update dialog tree
- `DELETE /api/v1/dialog_trees/{dialog_tree_id}` — delete dialog tree

OpenAPI docs are available at `/docs` when the app is running.

## How the main components work

- `main.py` — Registers FastAPI app, routes, and uses `lifespan` from `lifecycle/app_lifecycle.py` to initialize DBs at startup. Routes are thin and forward requests to handlers in `endpoints/api_endpoints.py`.

- `endpoints/api_endpoints.py` — Implements business logic:

  - Document creation: generates an embedding via `ai/openai_integration.py`, stores vector in Milvus (via `core/database.py` store), and persists document metadata/content in PostgreSQL.
  - Document retrieval: tries Milvus first, falls back to PostgreSQL.
  - Pre-prompt and dialog-tree CRUD: stored in PostgreSQL. Dialog trees store FSM definitions as JSONB.

- `ai/openai_integration.py` — Wraps OpenAI usage:

  - `generate_openai_embedding(text, cache_key=None)` — creates embeddings with caching and truncation safeguards.
  - `generate_openai_answer(...)` and history-aware variants — create chat completions (including streaming generator).
  - Handles API errors and maps them to FastAPI `HTTPException`.

- `core/database.py` — Responsible for Milvus and PostgreSQL initialization, schema creation for collections/tables, and Milvus store helpers (`MilvusStore`) used to upsert and search document vectors.

- `config/settings.py` — Centralized settings built with Pydantic Settings; constructs a `psycopg_pool.ConnectionPool` (`pg_pool`) and provides Milvus helpers.

- `utils/utility_function.py` — Helper functions used to safely read document fields and parse metadata from PostgreSQL or Milvus.

- `utils/logger.py` — Central logger used by modules to keep consistent formatting.

## Example: Add a document (curl)

```bash
curl -X POST http://localhost:8000/api/v1/documents \
	-H "Content-Type: application/json" \
	-d '{"title":"Wine Guide","content":"Tasting notes...","tenant_id":"tenant123","metadata":{}}'
```

## Development notes and tips

- Ensure `OPENAI_API_KEY` is set to use embeddings and LLM features.
- The code expects embedding vector length 1536 in one place — confirm your chosen embedding model matches that size.
- Milvus collection names in code: `KnowledgeBase`, `Users`, `UserChats` (case-sensitive depending on Milvus).
- The `core/database.py` file contains SQL DDL executed at startup to create tables if missing.

## Contributing

1. Open an issue describing the change.
2. Create a feature branch, add tests, and submit a PR.

## License

Specify your license here.

---

If you want, I can also:

- generate a `requirements.txt` capturing current dependencies,
- add example `.env.example`, or
- create a small docker-compose override to run Postgres+Milvus locally.
