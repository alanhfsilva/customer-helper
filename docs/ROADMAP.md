# Roadmap — AI-Powered Customer-Support Assistant

**Project:** A production customer-support assistant built on the OpenAI API, taking a user question and returning a grounded, accurate, guardrailed answer drawn from the company's own knowledge base (help center, docs, policies, past tickets).

**Stack:** Python · OpenAI API · REST APIs · AWS
**Core capabilities:** Retrieval-Augmented Generation (RAG) · Prompt engineering · Response-quality guardrails · Offline + online evaluation

This roadmap is the *what and when*. The companion [`TECHNICAL_SPEC.md`](./TECHNICAL_SPEC.md) is the *how* — it is the authoritative source that AI build agents execute against. [`AGENTS.md`](./AGENTS.md) tells agents how to work in this repo.

---

## Guiding principles

1. **Ship a thin vertical slice first.** A single-question → grounded-answer path that works end to end beats a half-built pipeline of everything.
2. **Evaluation is a feature, not an afterthought.** The eval harness lands before the assistant is exposed to real users, and answer accuracy is measured *before* every release.
3. **Grounded or abstain.** The assistant answers only from retrieved context. When context is insufficient, it says so and escalates rather than inventing an answer.
4. **Everything configurable is config, not code.** Model IDs, prompts, thresholds, and chunking parameters live in versioned config so they can change without a code deploy and are pinned for reproducibility.
5. **Cost and latency are product requirements.** Every phase carries a p95 latency and cost-per-resolved-conversation budget.

---

## Milestones at a glance

| Phase | Name | Outcome | Target gate |
|------|------|---------|-------------|
| 0 | Foundations | Repo, config, CI, secrets, skeleton API | Dev can run the app locally and in a dev AWS account |
| 1 | Ingestion & Index | Knowledge base chunked, embedded, searchable | Retrieval returns relevant chunks for a labeled query set |
| 2 | Answering (RAG) | Grounded answer generation with citations | End-to-end question → cited answer works |
| 3 | Guardrails | Safety, grounding, PII, refusal/abstain behavior | Guardrail test suite green |
| 4 | Evaluation | Offline eval harness + accuracy gate in CI | Accuracy metrics reported per build; release gate enforced |
| 5 | Productionize | AWS deploy, observability, cost controls | Live behind auth, monitored, within SLOs |
| 6 | Live workflow & iterate | Real-user channel, feedback loop, online metrics | Serving real users; weekly quality review |

---

## Phase 0 — Foundations

**Goal:** A running skeleton that everything else hangs off.

**Deliverables**
- Repository scaffold matching the layout in the technical spec (`app/`, `ingestion/`, `eval/`, `infra/`, `tests/`).
- Central `config/` with model IDs, thresholds, and prompt templates; loaded through a typed settings module.
- Secrets handling (OpenAI key, datastore creds) via AWS Secrets Manager / SSM — never in the repo.
- `POST /chat` and `GET /healthz` FastAPI endpoints returning stubbed responses.
- CI pipeline: lint, type-check, unit tests on every PR.
- Local dev via `docker compose` (app + vector store).

**Exit criteria**
- `make dev` (or documented equivalent) boots the app locally; `/healthz` is green.
- CI passes on a trivial PR.
- A stubbed `/chat` call returns a valid response envelope.

---

## Phase 1 — Ingestion & Index

**Goal:** Turn the raw knowledge base into a searchable vector index.

**Deliverables**
- Source connectors for the initial corpus (start with help-center articles / Markdown/HTML docs; design for adding ticket history later).
- Normalization + cleaning (strip boilerplate, preserve headings, capture source URL and metadata).
- Chunking strategy (structure-aware, with overlap) governed by config.
- Embedding generation via the OpenAI embeddings model (batched, retried, deduplicated).
- Vector store populated with chunks + metadata; hybrid (semantic + keyword) retrieval available.
- Idempotent, re-runnable ingestion with content hashing so unchanged docs aren't re-embedded.

**Exit criteria**
- Full corpus ingested; row counts and token/cost report produced.
- Retrieval smoke test: for a hand-labeled set of ~30 queries, the correct source document appears in top-k for an agreed majority (e.g. Recall@5 ≥ 0.8).

---

## Phase 2 — Answering (RAG)

**Goal:** A grounded answer with citations, end to end.

**Deliverables**
- Retrieval → context assembly → prompt → OpenAI chat completion pipeline.
- Prompt design: system prompt establishing role, grounding rule, citation format, and abstain behavior.
- Answer envelope: answer text, cited sources (with URLs), confidence/grounding signal, and a "needs human" flag.
- Streaming responses for the API.
- Conversation state (multi-turn) with context-window management.

**Exit criteria**
- `POST /chat` returns a grounded, cited answer for real questions against the ingested corpus.
- Manual review of ~20 answers shows citations that actually support the claims.

---

## Phase 3 — Guardrails

**Goal:** The assistant is safe, grounded, and knows its limits.

**Deliverables**
- **Grounding/faithfulness check:** detect answers not supported by retrieved context; downgrade or abstain.
- **Abstain & escalate:** when retrieval confidence is low or the topic is out of scope, return a safe fallback and route to a human.
- **Input/output moderation:** screen for disallowed content via the moderation endpoint and policy rules.
- **PII handling:** detect and redact PII in logs; policy for PII in prompts.
- **Prompt-injection resistance:** treat retrieved content and user input as untrusted; instructions in documents must not override the system prompt.
- **Rate limiting & abuse protection** at the API edge.

**Exit criteria**
- Guardrail test suite (adversarial prompts, injection attempts, out-of-scope questions, PII) passes at the agreed thresholds.
- No un-grounded answer escapes for the red-team question set.

---

## Phase 4 — Evaluation

**Goal:** Measure answer accuracy before release — and keep measuring.

**Deliverables**
- **Golden dataset:** curated question/reference-answer/expected-source set, versioned in the repo.
- **Offline eval harness** scoring: retrieval quality (Recall@k, MRR), answer accuracy/correctness, faithfulness/groundedness, citation validity, refusal correctness, latency, and cost.
- **LLM-as-judge** scoring with a rubric, plus exact/similarity checks where applicable; spot-checked against human labels.
- **Regression gate in CI:** a build that drops below thresholds on the golden set fails the release.
- Eval report artifact (Markdown/HTML) per run with per-metric breakdown and diffs vs. the previous run.

**Exit criteria**
- Eval runs in CI and on demand; a report is produced.
- Release gate is wired: accuracy below the configured floor blocks the release.

---

## Phase 5 — Productionize

**Goal:** Run it on AWS, safely and observably.

**Deliverables**
- Infrastructure-as-code (AWS) for the app, vector store, secrets, and networking.
- Deploy target (containerized service on ECS/Fargate or Lambda behind API Gateway — chosen in the spec) with autoscaling.
- Authentication/authorization on the API.
- Observability: structured logs, traces, per-request token/cost metrics, dashboards, and alerts on latency, error rate, cost, and abstain rate.
- Cost controls: caching, max-token caps, model routing (cheap model first where adequate), spend alarms.
- Runbook: rollback, key rotation, re-index procedure, incident response.

**Exit criteria**
- Service is live in a production AWS account behind auth, within p95 latency and cost SLOs.
- Dashboards and alerts are active; a rollback has been rehearsed.

---

## Phase 6 — Live workflow & iterate

**Goal:** Serve real users and close the quality loop.

**Deliverables**
- Integration with the real support channel (web widget, help-desk tool, or Slack — decided with stakeholders).
- Human-in-the-loop: confident answers are served directly; low-confidence ones are drafted for an agent to approve/edit.
- Feedback capture (thumbs, agent edits, resolution outcome) flowing back into the golden dataset and prompt/retrieval tuning.
- Online metrics: deflection/resolution rate, escalation rate, CSAT, average handle time, cost per conversation.
- Weekly quality review ritual using eval + online metrics to drive the next iteration.

**Exit criteria**
- Assistant is answering real user questions in the chosen channel.
- Feedback loop demonstrably improves the golden set and metrics over at least one iteration.

---

## Cross-cutting workstreams (run through every phase)

- **Security & privacy:** least-privilege IAM, encrypted data at rest/in transit, PII minimization, data-retention policy.
- **Cost management:** token accounting from day one; every phase reports spend.
- **Documentation:** keep the spec and this roadmap in sync; update the runbook as production behavior is learned.
- **Testing:** unit + integration + eval; no release without the accuracy gate.

---

## Suggested sequencing & dependencies

```
Phase 0 ─┬─> Phase 1 ─> Phase 2 ─> Phase 3 ─┐
         │                                   ├─> Phase 5 ─> Phase 6
         └────────────> Phase 4 <────────────┘
```

Phase 4 (evaluation) starts as soon as Phase 2 produces answers and must be complete before Phase 5 exposes the assistant to users. Phase 3 and Phase 4 can proceed largely in parallel once answering works.

---

## Definition of done for the whole project

A live customer-support assistant that (1) retrieves from the company knowledge base, (2) generates grounded, cited answers, (3) refuses/escalates when it shouldn't answer, (4) has its answer accuracy measured before every release by an automated gate, and (5) runs on AWS within defined latency, cost, and quality SLOs while serving real users through a real support channel.
