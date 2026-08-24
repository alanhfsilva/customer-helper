# Technical Specification — AI-Powered Customer-Support Assistant

**Status:** Authoritative build spec. AI build agents implement against this document.
**Companion docs:** [`ROADMAP.md`](./ROADMAP.md) (phases & timeline) · [`AGENTS.md`](./AGENTS.md) (how agents work in this repo)

> **How to read this doc.** Sections 1–4 define *what the system is*. Sections 5–13 define *each component* with contracts and acceptance criteria. Section 14 is the file-by-file build order agents follow. Anything marked **[MUST]** is a hard requirement checked in review; **[SHOULD]** is a strong default an agent may deviate from only with a recorded reason.

---

## 1. Problem statement & scope

Build a production customer-support assistant that answers user questions using the company's own knowledge base. Given a question (and optional conversation history), the system retrieves the most relevant knowledge, generates a grounded answer with citations, applies quality/safety guardrails, and either returns the answer or escalates to a human. Answer accuracy is measured by an automated evaluation harness before every release.

**In scope:** RAG pipeline, prompt design, guardrails, evaluation harness, a REST API, AWS deployment, observability, and a human-in-the-loop workflow.

**Out of scope (v1):** Fine-tuning custom models, voice/telephony, multilingual beyond the primary language (design must not preclude it), and autonomous actions that mutate customer accounts (the assistant answers and drafts; it does not take account actions).

**Primary users:** End customers (directly or via an agent-assist surface) and support agents who review low-confidence drafts.

---

## 2. Non-functional requirements (SLOs & budgets)

| Requirement | Target |
|---|---|
| p95 end-to-end latency (non-streaming) | ≤ 4.0 s **[MUST]**; first streamed token ≤ 1.5 s **[SHOULD]** |
| Retrieval quality on golden set | Recall@5 ≥ 0.85, MRR ≥ 0.7 **[MUST]** |
| Answer accuracy (judge score) on golden set | ≥ configured floor (start 0.80) **[MUST gate]** |
| Faithfulness / groundedness | ≥ 0.95 of served answers supported by cited context **[MUST]** |
| Availability | 99.5% monthly **[SHOULD]** |
| Cost per resolved conversation | ≤ configured budget; reported per build **[MUST report]** |
| PII in logs | Zero un-redacted PII **[MUST]** |

All thresholds live in `config/thresholds.yaml` and are the single source of truth for gates.

---

## 3. High-level architecture

```
                         ┌───────────────────────────────────────────────┐
   User / Agent UI  ───▶ │  API layer (FastAPI)   POST /chat  /healthz    │
                         │   - auth, rate limit, request validation       │
                         └───────────────┬───────────────────────────────┘
                                         │
                                         ▼
                         ┌───────────────────────────────────────────────┐
                         │  Orchestrator (RAG pipeline)                   │
                         │   1. moderate input                            │
                         │   2. rewrite/condense query (multi-turn)       │
                         │   3. retrieve (hybrid) + rerank                │
                         │   4. assemble context (token budget)           │
                         │   5. generate answer (OpenAI chat)             │
                         │   6. guardrails: grounding, moderation, PII    │
                         │   7. decide: answer | abstain+escalate         │
                         └──────┬───────────────────────┬────────────────┘
                                │                        │
                    ┌───────────▼─────────┐   ┌──────────▼───────────┐
                    │ Vector store +      │   │ OpenAI API           │
                    │ keyword index       │   │ (chat + embeddings   │
                    │ (chunks + metadata) │   │  + moderation)       │
                    └───────────▲─────────┘   └──────────────────────┘
                                │
                    ┌───────────┴─────────┐
                    │ Ingestion pipeline  │  (offline, scheduled)
                    │ sources→clean→chunk │
                    │ →embed→upsert       │
                    └─────────────────────┘

   Cross-cutting:  config · secrets · structured logging/tracing · cost metering · eval harness
```

**Two independent entry points:**
1. **Serving path** (online): the API + orchestrator answering questions.
2. **Ingestion path** (offline/batch): building and refreshing the index.

They share only the vector store and the config. Keep them decoupled.

---

## 4. Technology choices

| Concern | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ **[MUST]** | Type hints required; `mypy`/`pyright` in CI |
| API framework | FastAPI + Uvicorn **[SHOULD]** | Async, streaming, OpenAPI for free |
| LLM provider | OpenAI API **[MUST]** | Chat, embeddings, and moderation endpoints |
| Model IDs | **Config, not hard-coded** **[MUST]** | Pin exact model IDs in `config/models.yaml`; see §5 |
| Vector store | Pluggable behind an interface **[MUST]** | Default: pgvector (Postgres) for portability; OpenSearch/managed vector DB acceptable. One implementation ships in v1 |
| Keyword search | BM25 / Postgres FTS or OpenSearch **[SHOULD]** | For hybrid retrieval |
| Infra | AWS via IaC (Terraform or AWS CDK) **[MUST]** | See §12 |
| Compute | ECS/Fargate behind ALB **[SHOULD]** (Lambda+API Gateway acceptable) | Decision recorded in `infra/README.md` |
| Packaging | Docker **[MUST]** | Same image local and prod |
| Config/secrets | pydantic-settings + AWS Secrets Manager/SSM **[MUST]** | No secrets in repo |

**Model configuration contract** (`config/models.yaml`): exactly one place defines `chat_model`, `chat_model_fallback` (cheaper/faster for routing), `embedding_model`, `embedding_dimensions`, and `moderation_model`. Code reads these through the settings module and **never** hard-codes a model string. Choose current OpenAI models at build time (a capable GPT-class chat model and a `text-embedding-3`-class embedding model) and pin the exact IDs. Provider calls go through one thin client wrapper (§5) so the rest of the code is model-agnostic.

---

## 5. OpenAI client wrapper

Create `app/llm/client.py` exposing a single interface used everywhere. **[MUST]** No other module imports the OpenAI SDK directly.

```python
class LLMClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def chat(self, messages: list[Message], *, stream: bool = False,
             max_tokens: int | None = None, temperature: float | None = None,
             response_format: dict | None = None) -> ChatResult: ...
    def moderate(self, text: str) -> ModerationResult: ...
```

Requirements:
- **Retries with exponential backoff + jitter** on 429/5xx; respect `Retry-After`. **[MUST]**
- **Timeouts** on every call. **[MUST]**
- **Token & cost accounting:** every call records prompt/completion tokens and computed cost, tagged with a `request_id`. **[MUST]**
- **Deterministic test mode:** a `FakeLLMClient` returning canned responses so unit tests and CI run without network/keys. **[MUST]**
- Batching for embeddings; concurrency-limited. **[SHOULD]**

---

## 6. Data model

### 6.1 Document & chunk (ingestion)
```
Document:
  id: str                # stable hash of source URI
  source_uri: str        # canonical URL / path
  title: str
  source_type: enum      # help_article | doc | policy | ticket | faq
  content_raw: str
  content_hash: str      # sha256 of normalized content (idempotency)
  metadata: dict         # product area, tags, last_updated, visibility
  fetched_at: datetime

Chunk:
  id: str                # {document_id}:{ordinal}
  document_id: str
  ordinal: int
  text: str
  heading_path: list[str]   # e.g. ["Billing","Refunds"]
  token_count: int
  embedding: vector(dims)   # dims from config
  metadata: dict            # inherits doc metadata + chunk-local (heading, url#anchor)
```

### 6.2 Serving objects
```
ChatRequest:
  conversation_id: str | null
  message: str                 # required, 1..4000 chars
  history: list[Turn] | null   # optional prior turns
  metadata: dict | null        # channel, user segment (no PII required)

RetrievedChunk:
  chunk_id, document_id, text, score, source_uri, title, heading_path

Answer (response envelope):
  conversation_id: str
  answer: str                  # may be empty if abstaining
  citations: list[{title, source_uri, chunk_ids}]
  status: enum                 # answered | abstained | escalated | blocked
  grounding_score: float       # 0..1
  confidence: float            # 0..1
  needs_human: bool
  usage: {prompt_tokens, completion_tokens, cost_usd}
  request_id: str
  latency_ms: int
```

`status` semantics: `answered` (grounded answer served), `abstained` (insufficient context; safe fallback + escalate), `escalated` (routed to human by policy), `blocked` (moderation blocked input/output).

---

## 7. Ingestion pipeline (`ingestion/`)

**Trigger:** CLI (`python -m ingestion.run`) and a scheduled job in prod. Must be **idempotent** and **re-runnable**. **[MUST]**

Stages:
1. **Fetch/load** from a source connector implementing `SourceConnector.iter_documents() -> Iterable[Document]`. v1 ships one connector (Markdown/HTML help-center export); the interface makes adding tickets/FAQ later trivial. **[MUST]**
2. **Normalize/clean:** strip nav/boilerplate, keep heading structure, resolve to a canonical URL, record `content_hash`. **[MUST]**
3. **Chunk:** structure-aware splitting that respects headings; target ~300–500 tokens with ~15% overlap (values in `config/ingestion.yaml`). Carry `heading_path` and `source_uri#anchor`. **[SHOULD]**
4. **Embed:** batched calls via `LLMClient.embed`; skip chunks whose parent doc `content_hash` is unchanged (dedupe against store). **[MUST]**
5. **Upsert** chunks + metadata into the vector store transactionally; deleted/updated docs reconcile (no orphan chunks). **[MUST]**
6. **Report:** emit an ingestion report (doc/chunk counts, tokens, cost, skipped-unchanged) to `artifacts/ingestion/`. **[MUST]**

**Acceptance:** Re-running with no source changes performs zero embedding calls and leaves the store byte-identical. Changing one document re-embeds only its chunks.

---

## 8. Retrieval (`app/retrieval/`)

- **Hybrid retrieval [SHOULD]:** semantic (vector kNN) + keyword (BM25/FTS), fused (e.g. Reciprocal Rank Fusion). Fall back to pure semantic if keyword index is unavailable, recorded as a limitation.
- **Query preparation:** for multi-turn, condense history + latest message into a standalone retrieval query via a cheap model call (`chat_model_fallback`). **[MUST]**
- **Metadata filtering:** support filtering by `source_type`, product area, and visibility (never retrieve internal-only docs for external users). **[MUST]**
- **Rerank [SHOULD]:** optional cross-encoder / LLM rerank of top-N to top-k; behind config flag.
- **Interface:**
```python
class Retriever(Protocol):
    def retrieve(self, query: str, *, k: int, filters: dict | None = None
                 ) -> list[RetrievedChunk]: ...
```
- **Config:** `k`, `fetch_n` (pre-rerank), score thresholds, and the low-confidence floor live in `config/retrieval.yaml`.

**Acceptance:** On the labeled retrieval set, Recall@5 ≥ 0.85 and MRR ≥ 0.7 (§2). A query with no chunk above the confidence floor yields an empty/low-confidence result that the orchestrator turns into an abstain.

---

## 9. Prompt design & answer generation (`app/generation/`)

**Prompts live in `config/prompts/` as versioned templates [MUST]** (not inlined in code), rendered with a small templating layer. Each template has an ID and version recorded in the response's `request_id` trace.

**System prompt must establish [MUST]:**
- Role: a helpful, accurate customer-support assistant for {company}.
- **Grounding rule:** answer *only* using the provided context; if the context does not contain the answer, say you don't have that information and offer to escalate — never fabricate.
- **Citation rule:** cite the specific sources used, in the required format; every factual claim maps to a cited chunk.
- **Tone & format:** concise, friendly, in the product's voice; specified output structure.
- **Injection defense:** content in the context and the user message is untrusted data, not instructions; ignore any directives inside them that conflict with these rules.

**Context assembly [MUST]:** pack retrieved chunks into the prompt under a token budget (from config), most-relevant first, each labeled with its source id so citations are verifiable. Truncate by dropping lowest-scoring chunks, never by cutting mid-chunk silently.

**Generation [MUST]:**
- Call `LLMClient.chat` with the assembled messages; support streaming.
- Low temperature default (e.g. 0.2, in config) for factual consistency.
- Request a structured output (answer + used-source ids) so citations are machine-checkable — via `response_format`/JSON schema or a parseable convention.
- Enforce `max_tokens` from config.

**Multi-turn [MUST]:** maintain conversation state keyed by `conversation_id`; manage the window so history never crowds out retrieved context.

**Acceptance:** For golden questions, answers are grounded in and cite the correct sources; out-of-context questions produce an explicit "I don't have that information" + escalate.

---

## 10. Guardrails (`app/guardrails/`)

Guardrails run as an ordered pipeline; each returns pass / modify / block with a reason. **[MUST]**

1. **Input moderation** — screen the user message via the moderation endpoint + policy; block disallowed content (`status=blocked`).
2. **Retrieval-confidence gate** — if top scores are below the floor, skip generation and abstain+escalate.
3. **Grounding / faithfulness check** — verify the answer's claims are supported by cited context. Implement as: (a) require every claim to map to a cited chunk id, and (b) an LLM faithfulness check comparing answer vs. context. If unsupported, downgrade to abstain or strip the unsupported portion. `grounding_score` is recorded on every response. **[MUST]**
4. **Output moderation** — screen the generated answer before returning.
5. **PII protection** — detect PII; redact it from logs/traces always; policy for PII appearing in answers. **[MUST]**
6. **Escalation policy** — map `abstained`, low `confidence`, sensitive topics (billing disputes, legal, security), and explicit user requests for a human to `needs_human=true` and route accordingly.

**Prompt-injection defense [MUST]:** retrieved context and user input are always framed as data; the system prompt is authoritative. Include injection attempts in the guardrail test suite.

**Acceptance:** The adversarial guardrail suite (injection, jailbreak, PII, out-of-scope, un-grounded) passes at configured thresholds; no un-grounded answer is served for the red-team set.

---

## 11. Evaluation harness (`eval/`)

The eval harness is a first-class deliverable and the **release gate**. **[MUST]**

**Golden dataset** (`eval/datasets/golden.jsonl`, versioned): each record has `question`, optional `history`, `reference_answer`, `expected_source_ids`, `category`, and `should_abstain` (bool for out-of-scope items).

**Metrics [MUST]:**
- *Retrieval:* Recall@k, MRR against `expected_source_ids`.
- *Answer accuracy/correctness:* LLM-as-judge scored against `reference_answer` with a written rubric (`eval/rubric.md`); supplement with similarity/exact-match where applicable.
- *Faithfulness/groundedness:* is the answer supported by retrieved context.
- *Citation validity:* do cited sources exist and support the answer.
- *Refusal correctness:* does the assistant abstain exactly on `should_abstain` items (precision/recall of abstention).
- *Latency & cost:* p50/p95 latency and cost per question.

**Judge discipline [MUST]:** the LLM judge uses a fixed rubric and a pinned model; a labeled sample is periodically checked against human judgment to validate the judge. Judge prompts are versioned in `config/prompts/`.

**Runner [MUST]:** `python -m eval.run --dataset golden` produces `artifacts/eval/<run_id>/report.md` (+ machine-readable `report.json`) with per-metric scores, per-category breakdown, and a diff vs. the previous run.

**CI gate [MUST]:** on the release branch, the harness runs against the golden set; if `answer_accuracy < thresholds.answer_accuracy_floor` (or faithfulness/recall below floors), the build **fails**. This is what "measure answer accuracy before release" means concretely.

---

## 12. API layer (`app/api/`)

Endpoints:
- `POST /chat` — body = `ChatRequest`; returns `Answer` (streaming and non-streaming variants). **[MUST]**
- `GET /healthz` — liveness/readiness (checks vector store + OpenAI reachability). **[MUST]**
- `POST /feedback` — record thumbs/agent-edit/resolution for a `request_id`. **[SHOULD]**
- `GET /metrics` — Prometheus-style metrics (or push to CloudWatch). **[SHOULD]**

Requirements:
- **Auth** on `/chat` and `/feedback` (API key/JWT). **[MUST]**
- **Request validation** via pydantic; reject oversized/malformed input. **[MUST]**
- **Rate limiting** per caller. **[MUST]**
- **Structured JSON logs** with `request_id`, latency, token/cost, status, grounding/confidence — PII-redacted. **[MUST]**
- OpenAPI docs auto-generated.

---

## 13. Infrastructure & operations (`infra/`)

- **IaC** (Terraform or CDK) provisions: the container service (ECS/Fargate + ALB, or Lambda + API Gateway), the vector store (RDS Postgres w/ pgvector or managed equivalent), Secrets Manager/SSM, IAM roles (least privilege), logging/metrics, and alarms. **[MUST]**
- **Environments:** `dev` and `prod` at minimum, from the same modules. **[MUST]**
- **Secrets:** OpenAI key + DB creds in Secrets Manager; injected at runtime; **never in the image or repo**. **[MUST]**
- **Observability:** dashboards for latency, error rate, cost/day, abstain & escalation rate, retrieval scores; alarms on p95 latency, error rate, daily spend, and abstain-rate spikes. **[MUST]**
- **Cost controls:** response/embedding caching where safe, `max_tokens` caps, cheap-model routing for query condensation/rerank, and a spend alarm that can trip a kill-switch flag. **[SHOULD]**
- **Runbook** (`infra/RUNBOOK.md`): deploy, rollback, key rotation, re-index, incident response, kill-switch. **[MUST]**

---

## 14. Build order for agents (execution plan)

Agents implement in this order; each step is a self-contained unit with the acceptance check that closes it. Do not start a step until its predecessor's acceptance passes. See `AGENTS.md` for working rules and definition-of-done.

1. **Repo scaffold & config** — directory layout, `pydantic-settings`, `config/*.yaml`, `config/prompts/`, Docker, `docker compose`, CI (lint/type/test). *Accept:* app boots, `/healthz` stub green, CI passes.
2. **LLM client wrapper (§5)** with `FakeLLMClient`. *Accept:* unit tests pass offline; retries/timeouts/cost-accounting covered.
3. **Data model & vector-store interface (§6, §8)** with one concrete store (pgvector). *Accept:* upsert + kNN round-trip test green.
4. **Ingestion pipeline (§7)** + one source connector + ingestion report. *Accept:* idempotency test (re-run = zero embeds), report produced.
5. **Retrieval (§8)** hybrid + filters + query condensation. *Accept:* Recall@5 / MRR on labeled retrieval set meet §2.
6. **Generation & prompts (§9)** + structured citations + streaming. *Accept:* end-to-end grounded, cited answer for real questions.
7. **`POST /chat` API (§12)** wiring orchestrator, auth, validation, logging. *Accept:* API returns valid `Answer` envelope; logs PII-redacted.
8. **Guardrails (§10)** pipeline incl. grounding check, moderation, PII, abstain/escalate. *Accept:* guardrail suite green; no un-grounded answer for red-team set.
9. **Evaluation harness (§11)** + golden dataset + report + **CI accuracy gate**. *Accept:* eval report generated; sub-threshold build fails.
10. **Infra & deploy (§13)** to dev then prod, observability, runbook. *Accept:* live behind auth within SLOs; alarms active; rollback rehearsed.
11. **Feedback loop & live workflow (Roadmap Phase 6)** — `/feedback`, human-in-the-loop drafting, online metrics. *Accept:* real-user channel serving; feedback flows into golden set.

---

## 15. Repository layout

```
support-assistant/
├── app/
│   ├── api/            # FastAPI routes, auth, validation
│   ├── llm/            # OpenAI client wrapper (+ FakeLLMClient)
│   ├── retrieval/      # retriever, hybrid fusion, filters, rerank
│   ├── generation/     # prompt assembly, answer generation, streaming
│   ├── guardrails/     # moderation, grounding, PII, escalation
│   ├── orchestrator.py # the RAG pipeline tying it together
│   └── settings.py     # typed config loader
├── ingestion/          # connectors, clean, chunk, embed, upsert, run.py
├── eval/               # harness, datasets/golden.jsonl, rubric.md, run.py
├── config/             # models.yaml, retrieval.yaml, ingestion.yaml,
│                       # thresholds.yaml, prompts/*.md
├── infra/              # Terraform/CDK, RUNBOOK.md, README.md
├── tests/              # unit + integration + guardrail suites
├── artifacts/          # ingestion/ and eval/ reports (gitignored outputs)
├── docs/               # this spec, roadmap, agents guide
├── Dockerfile
├── docker-compose.yml
└── Makefile            # dev, test, lint, ingest, eval targets
```

---

## 16. Testing strategy

- **Unit:** every module, offline, using `FakeLLMClient` and an in-memory/ephemeral store. **[MUST]**
- **Integration:** ingestion→retrieval→generation happy path against a small fixture corpus. **[MUST]**
- **Guardrail suite:** adversarial fixtures (injection, jailbreak, PII, out-of-scope). **[MUST]**
- **Eval:** golden-set run wired as the release gate. **[MUST]**
- **No live-key requirement in CI:** network calls stubbed; a separate, opt-in smoke test may hit the real API. **[MUST]**

---

## 17. Open decisions to record (not blockers)

Agents pick a sensible default and record the choice in the relevant `README.md`:
- Compute target: ECS/Fargate vs. Lambda.
- Vector store: pgvector vs. managed vector DB / OpenSearch.
- Live channel in Phase 6: web widget vs. help-desk integration vs. Slack.
- Reranker: enabled by default or behind a flag.

Each decision is a lightweight ADR in `docs/adr/`.
