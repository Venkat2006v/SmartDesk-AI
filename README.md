# SmartDesk AI — IT & HR Operations Agent

A multi-agent, RAG-grounded helpdesk assistant that answers IT and HR policy
questions from a hybrid knowledge base, creates support tickets with mandatory
human-in-the-loop (HITL) confirmation, and checks the status of existing
tickets — all routed by a Supervisor agent built on LangGraph.

## Capabilities

1. **Knowledge base Q&A (RAG)** — hybrid dense + sparse retrieval (Qdrant)
   grounds answers in retrieved documents across two specialist agents (IT, HR).
2. **Ticket creation** — LLM extracts fields from natural language, validates
   email, confirms with the user (HITL) before writing, then creates a ticket
   via the configured ticketing client.
3. **Ticket status check** — look up existing tickets for a user by email,
   handling zero / one / many results cleanly.
4. **Graceful escalation** — when retrieval confidence is below threshold the
   agent admits it doesn't know and suggests opening a ticket.

A Supervisor agent uses the LLM to classify each query into one of five routes:
`it_kb` · `hr_kb` · `create_ticket` · `ticket_status` · `off_topic`.

## Stack

| Layer | Choice |
|---|---|
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph` |
| Vector store | [Qdrant](https://qdrant.tech) — local embedded mode (no Docker needed) |
| Dense embeddings | OpenAI `text-embedding-3-small` or fastembed (local, free) |
| Sparse embeddings | fastembed BM25 (`Qdrant/bm25`) — hybrid search bonus |
| LLM | OpenAI (`gpt-4o-mini` default) or Anthropic |
| Ticketing | Mock in-memory client (swap `TICKETING_PROVIDER=real` for Jira/etc.) |
| UI (bonus) | Gradio 6 chat interface |

## Project layout

```
SmartDesk-AI/
├── docs/                          # Architecture + design-decision log
├── data/
│   ├── knowledge_base/
│   │   ├── it_docs/               # LLM-generated IT Q&A docs (JSON)
│   │   └── hr_docs/               # HuggingFace HR dataset + synthetic docs
│   └── processed/
│       └── vector_index/          # Qdrant local store (git-ignored)
├── src/smartdesk/
│   ├── config.py                  # All settings via .env
│   ├── main.py                    # CLI entry point
│   ├── orchestrator/
│   │   ├── graph.py               # LangGraph StateGraph — build_orchestrator()
│   │   └── state.py               # AgentState TypedDict + Route literal
│   ├── agents/
│   │   ├── _llm.py                # Shared call_llm() dispatcher (OpenAI/Anthropic)
│   │   ├── supervisor.py          # LLM-based 5-way router
│   │   ├── it_knowledge_agent.py  # RAG → grounded answer or escalate
│   │   ├── hr_knowledge_agent.py  # same, domain="hr"
│   │   ├── ticket_creation_agent.py  # field extraction → HITL → create
│   │   └── ticket_status_agent.py    # email lookup → format response
│   ├── rag/
│   │   ├── ingestion.py           # HuggingFace + local file loaders + chunker
│   │   ├── embeddings.py          # make_dense_embedding / make_sparse_embedding
│   │   ├── vector_store.py        # Qdrant local client, add_documents, hybrid search
│   │   └── retriever.py           # retrieve() + decide_escalation()
│   ├── tools/
│   │   ├── hitl.py                # confirm_action() — CLI y/N gate
│   │   └── ticketing/
│   │       ├── base.py            # TicketingClient ABC + Ticket TypedDict
│   │       ├── mock_client.py     # In-memory mock (fully implemented)
│   │       └── ticketing_client.py  # RealTicketingClient stub (your integration)
│   ├── guardrails/
│   │   ├── grounding.py           # build_grounded_prompt + check_grounding
│   │   └── validation.py          # is_valid_email + with_retry decorator
│   ├── evaluation/
│   │   └── eval_pipeline.py       # TODO: RAG metrics (faithfulness, precision)
│   └── ui/
│       └── app.py                 # Gradio 6 chat UI with multi-turn HITL
├── scripts/
│   ├── generate_synthetic_docs.py # LLM-generated IT + HR Q&A docs
│   └── build_knowledge_base.py    # Ingest → chunk → embed → upsert to Qdrant
└── tests/
    ├── conftest.py
    ├── test_retriever.py
    ├── test_supervisor_routing.py
    ├── test_ticket_creation_agent.py
    ├── test_ticket_status_agent.py
    └── test_guardrails.py
```

## Getting started

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### 2. Configure

```bash
cp .env.example .env
```

Open `.env` and fill in at minimum:

```env
LLM_PROVIDER=openai
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini

EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small

# Leave TICKETING_PROVIDER=mock for development
```

### 3. Build the knowledge base

```bash
# Generate synthetic IT + HR documents via LLM
python scripts/generate_synthetic_docs.py

# Ingest HuggingFace HR dataset + synthetic docs → embed → store in Qdrant
python scripts/build_knowledge_base.py
```

This populates `data/processed/vector_index/` (local Qdrant, no Docker needed).

### 4. Run

**CLI:**
```bash
python -m smartdesk.main
```

**Gradio UI (bonus):**
```bash
pip install gradio
python src/smartdesk/ui/app.py
```

**Tests (no API keys needed — all mocked):**
```bash
pytest tests/ -v
```

## Knowledge base design (Option C)

The KB combines two sources deliberately:

- **Real data** — [`strova-ai/hr-policies-qa-dataset`](https://huggingface.co/datasets/strova-ai/hr-policies-qa-dataset) from HuggingFace
- **Synthetic data** — 12 IT + 12 HR Q&A docs generated by the LLM via `generate_synthetic_docs.py`

Certain topics are **intentionally left uncovered** (AWS EC2 provisioning, Kubernetes, GDPR deletion, executive equity) so the escalation path is always testable.

## Hybrid search

Every chunk is indexed with **both** a dense vector (semantic meaning) and a sparse vector (keyword/BM25). At query time:

1. Sparse prefetch — keyword match catches IT acronyms (MFA, VPN, TOTP, SSO) precisely
2. Dense rerank — semantic similarity re-orders the candidates

The system degrades gracefully to dense-only if sparse embeddings are not configured.

## HITL flow

```
User: "Create a ticket — my laptop won't connect to VPN"
  │
  ▼
Supervisor → create_ticket
  │
  ▼
ticket_creation_agent:
  1. LLM extracts summary + description from query
  2. Validates email (from session or query)
  3. ⚠ HITL gate — prints proposed ticket, waits for y/N
  4. On yes → MockTicketingClient.create_ticket() → MOCK-1
  5. Returns ticket ID + confirmation message
```

Ticket creation is **never** skipped past the HITL gate.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `LLM_API_KEY` | — | API key for LLM provider |
| `LLM_MODEL` | `gpt-4o-mini` | Model name |
| `EMBEDDING_PROVIDER` | `openai` | `openai` or `fastembed` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Dense embedding model |
| `SPARSE_EMBEDDING_MODEL` | `Qdrant/bm25` | fastembed sparse model |
| `VECTOR_INDEX_DIR` | `./data/processed/vector_index` | Qdrant local path |
| `QDRANT_COLLECTION_NAME` | `smartdesk_kb` | Collection name |
| `CONFIDENCE_THRESHOLD` | `0.0` | Escalation threshold (0–1) |
| `TICKETING_PROVIDER` | `mock` | `mock` or `real` |
| `HITL_MODE` | `cli` | `cli` (stdin) or `ui` (Gradio) |

## Status

| Component | Status |
|---|---|
| RAG pipeline (ingest → embed → retrieve) | ✅ Implemented |
| Hybrid search (dense + sparse, Qdrant local) | ✅ Implemented |
| Supervisor routing (LLM-based) | ✅ Implemented |
| IT / HR knowledge agents | ✅ Implemented |
| Ticket creation agent + HITL | ✅ Implemented |
| Ticket status agent | ✅ Implemented |
| LangGraph orchestrator | ✅ Implemented |
| CLI entry point | ✅ Implemented |
| Grounding guardrail | ✅ Implemented |
| Gradio UI (bonus) | ✅ Implemented |
| Evaluation pipeline | 🔜 Next |
