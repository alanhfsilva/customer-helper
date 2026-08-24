# Customer Helper System

An AI-powered customer-support assistant that answers user questions using a company knowledge base. It retrieves relevant documents, generates grounded answers with citations, applies quality and safety guardrails, and escalates to a human when confidence is low.

## What It Does

- **RAG pipeline** — retrieves context from a knowledge base using hybrid search (semantic vectors + BM25 keyword), then generates grounded, cited answers
- **Guardrails** — input/output moderation, grounding verification, PII detection and redaction, and automatic escalation for sensitive topics
- **Human-in-the-loop** — low-confidence answers are flagged for agent review instead of being served directly
- **Feedback loop** — captures thumbs up/down, agent edits, and resolution signals; approved corrections feed back into the evaluation golden set
- **Evaluation harness** — automated accuracy gate that blocks releases if answer quality drops below configured thresholds
- **Online metrics** — tracks deflection rate, escalation rate, latency, and cost per request

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| API framework | FastAPI + uvicorn |
| Configuration | pydantic-settings + YAML config files |
| LLM | OpenAI API (GPT-4o) via a protocol-based client wrapper |
| Vector store | PostgreSQL 16 + pgvector (with in-memory store for tests) |
| Search | Hybrid retrieval — semantic kNN + BM25 keyword with Reciprocal Rank Fusion |
| Ingestion | Structure-aware document chunking with heading extraction and overlap |
| Infrastructure | AWS ECS Fargate, RDS, Secrets Manager, CloudWatch (Terraform) |
| CI | GitHub Actions (ruff, mypy, pytest, eval gate) |
| Container | Docker + docker-compose |

## Requirements

- **Python 3.11+**
- **Docker** and **Docker Compose** (for local development with PostgreSQL + pgvector)
- **OpenAI API key** (for production; tests run offline with `FakeLLMClient`)
- **Terraform >= 1.5** (only for AWS deployment)

## Project Structure

```
├── app/
│   ├── api/            # FastAPI routes, auth, rate limiting, validation
│   ├── llm/            # OpenAI client wrapper + FakeLLMClient
│   ├── retrieval/      # hybrid retriever, keyword index, vector store, fusion
│   ├── generation/     # prompt assembly, answer generation, citation parsing
│   ├── guardrails/     # moderation, grounding, PII, escalation pipeline
│   ├── feedback/       # feedback store, metrics collector, golden-set export
│   ├── orchestrator.py # RAG pipeline tying retrieval → generation → guardrails
│   ├── models.py       # domain models (Chunk, RetrievedChunk, Document)
│   └── settings.py     # typed config loader from YAML
├── ingestion/          # document connectors, chunker, normalizer, pipeline
├── eval/               # evaluation harness, golden dataset, rubric, CI gate
├── config/             # models.yaml, retrieval.yaml, thresholds.yaml, prompts/
├── infra/              # Terraform modules (ECS, RDS, Secrets, Monitoring)
├── tests/              # unit + integration + guardrail + eval test suites
├── Dockerfile
├── docker-compose.yml
└── Makefile
```

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/alanhfsilva/customer-helper.git
cd customer-helper
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Run the tests (no API key needed)

```bash
make test
```

This runs linting (ruff), type checking (mypy), and the full test suite (pytest). All 189 tests run offline using `FakeLLMClient`.

Individual targets:

```bash
make lint        # ruff check
make typecheck   # mypy
make unittest    # pytest
make eval        # evaluation harness with golden dataset
```

### 3. Run locally with Docker

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` and set your OpenAI API key:

```
SA_OPENAI_API_KEY=sk-...
```

Start the app and PostgreSQL with pgvector:

```bash
make dev
```

This runs `docker compose up --build`, starting:
- **app** on `http://localhost:8000`
- **db** (PostgreSQL 16 + pgvector) on `localhost:5432`

### 4. Verify it's running

```bash
curl http://localhost:8000/healthz
```

Expected: `{"status": "ok"}`

## API Endpoints

All endpoints except `/healthz` and `/metrics` require an `X-API-Key` header.

### POST /chat

Send a question and receive a grounded answer with citations.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"message": "How do I get a refund?"}'
```

Response:

```json
{
  "conversation_id": "uuid",
  "answer": "Refunds are processed within 5-10 business days...",
  "citations": [{"title": "Billing", "source_uri": "/billing", "chunk_ids": ["billing:0"]}],
  "status": "answered",
  "grounding_score": 1.0,
  "confidence": 0.92,
  "needs_human": false,
  "usage": {"prompt_tokens": 150, "completion_tokens": 40, "cost_usd": 0.0008},
  "request_id": "uuid",
  "latency_ms": 1200
}
```

Multi-turn conversations — pass `conversation_id` and `history`:

```json
{
  "conversation_id": "existing-conv-id",
  "message": "What about international orders?",
  "history": [
    {"role": "user", "content": "How do I get a refund?"},
    {"role": "assistant", "content": "Refunds are processed within 5-10 business days."}
  ]
}
```

### POST /feedback

Record user or agent feedback for a request.

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "request_id": "uuid-from-chat-response",
    "signal": "thumbs_up"
  }'
```

Signals: `thumbs_up`, `thumbs_down`, `agent_edit`, `resolved`, `escalated`.

For agent edits, include the corrected answer:

```json
{
  "request_id": "uuid",
  "signal": "agent_edit",
  "corrected_answer": "The correct refund policy is...",
  "agent_id": "agent-42"
}
```

### GET /metrics

Online metrics (no auth required).

```bash
curl http://localhost:8000/metrics
```

Returns: `total_requests`, `answered`, `abstained`, `escalated`, `blocked`, `deflection_rate`, `escalation_rate`, `avg_latency_ms`, `avg_cost_usd`.

### GET /healthz

Liveness check (no auth required).

## Ingestion

Ingest documents into the knowledge base:

```bash
make ingest
```

Or directly:

```bash
python -m ingestion.run
```

Place source documents in `tests/fixtures/corpus/` (Markdown format). The ingestion pipeline:
1. Reads and normalizes documents
2. Chunks with structure-aware splitting (headings, overlap)
3. Embeds chunks via the configured embedding model
4. Upserts to the vector store with content-hash deduplication (re-runs produce zero new embeddings)

## Evaluation

Run the evaluation harness against the golden dataset:

```bash
make eval
```

This scores the system on accuracy, faithfulness, citation validity, and retrieval quality (Recall@5, MRR) against 10 golden questions. The process exits with code 1 if scores fall below the thresholds in `config/thresholds.yaml`. CI runs this as a release gate.

## Configuration

All thresholds, model selections, and prompts are config-driven (no hard-coded values):

| File | Purpose |
|---|---|
| `config/models.yaml` | Chat model, embedding model, moderation model |
| `config/retrieval.yaml` | Top-k, score threshold, hybrid alpha, confidence floor |
| `config/ingestion.yaml` | Chunk size, overlap, embedding batch size |
| `config/thresholds.yaml` | Accuracy floor, faithfulness floor, latency SLO, grounding floor |
| `config/prompts/system.md` | System prompt template |
| `config/prompts/query_condenser.md` | Multi-turn query rewrite prompt |
| `config/prompts/judge.md` | Eval judge prompt with rubric |

## AWS Deployment

Infrastructure is defined in Terraform modules under `infra/`:

| Module | Resources |
|---|---|
| `modules/ecs` | ECS Fargate cluster, task definition, ALB, security groups, IAM roles |
| `modules/rds` | RDS PostgreSQL 16 with pgvector, encrypted storage, backups |
| `modules/secrets` | Secrets Manager for API key and database URL |
| `modules/monitoring` | CloudWatch alarms (p95 latency, 5xx errors, CPU), dashboard |

Environment configs in `infra/envs/dev/` and `infra/envs/prod/`.

```bash
cd infra/envs/prod
terraform init
terraform plan -var="container_image=<ECR_URI>" \
               -var="vpc_id=<VPC_ID>" \
               -var="subnet_ids=[\"<SUBNET_A>\",\"<SUBNET_B>\"]" \
               -var="certificate_arn=<CERT_ARN>"
terraform apply
```

See [`infra/RUNBOOK.md`](infra/RUNBOOK.md) for deploy, rollback, key rotation, re-index, incident response, and kill-switch procedures.
