# SmartDesk AI — IT & HR Operations Agent

A multi-agent, RAG-grounded helpdesk assistant that answers IT and HR policy
questions from a hybrid knowledge base, creates support tickets with mandatory
human-in-the-loop (HITL) confirmation, and checks the status of existing
tickets — all routed by a Supervisor agent built on LangGraph.

## Capabilities

1. **Knowledge base Q&A (RAG)** — hybrid dense + sparse retrieval (Qdrant)
   grounds answers in retrieved documents across two specialist agents (IT, HR)
   and a combined synthesizer when a query spans both domains.
2. **Ticket creation** — LLM extracts fields from natural language, validates
   email, confirms with the user (HITL) before writing, then creates a ticket
   via the configured ticketing client.
3. **Ticket status check** — look up existing tickets for a user by email,
   handling zero / one / many results cleanly.
4. **Graceful escalation** — when retrieval confidence is below threshold the
   agent admits it doesn't know, shows the confidence score, and invites the
   user to open a ticket.

A Supervisor agent uses the LLM to classify each query into one of six routes:
`it_kb` · `hr_kb` · `combined_kb` · `create_ticket` · `ticket_status` · `off_topic`.

## Stack

| Layer | Choice |
|---|---|
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph` |
| Vector store | [Qdrant](https://qdrant.tech) — local embedded mode (no Docker needed) |
| Dense embeddings | OpenAI `text-embedding-3-small` or fastembed (local, free) |
| Sparse embeddings | fastembed BM25 (`Qdrant/bm25`) — hybrid search bonus |
| LLM | OpenAI (`gpt-4o-mini` default) or Anthropic |
| Ticketing | Mock in-memory client (swap `TICKETING_PROVIDER=jira` for live Jira) |
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
│   │   ├── supervisor.py          # LLM-based 6-way router
│   │   ├── it_knowledge_agent.py  # RAG → grounded answer or escalate
│   │   ├── hr_knowledge_agent.py  # same, domain="hr"
│   │   ├── combined_knowledge_agent.py  # IT + HR dual retrieval → synthesizer
│   │   ├── ticket_creation_agent.py     # field extraction → HITL → create
│   │   └── ticket_status_agent.py       # email lookup → format response
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
│   │       └── ticketing_client.py  # RealTicketingClient — Jira-backed live integration
│   ├── guardrails/
│   │   ├── grounding.py           # build_grounded_prompt + check_grounding
│   │   └── validation.py          # is_valid_email + with_retry decorator
│   ├── evaluation/
│   │   └── eval_pipeline.py       # RAG eval harness — faithfulness, precision, relevance (pending)
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

---

## Sample queries — test the full system

The scenarios below cover every branch of the agent graph. Run them in the CLI
(`python -m smartdesk.main`) or the Gradio UI.

---

### 1. IT knowledge base — direct answer

```
you> How do I connect to the VPN from my laptop?
```

**Route:** `supervisor → it_kb`

SmartDesk retrieves from the IT knowledge base and returns a numbered,
step-by-step answer with bold key terms. The response footer confirms what
was used:

```
---
*Sources: VPN Setup Guide · Confidence: High (89%)*
```

Other IT queries to try:
- `"Walk me through setting up MFA / TOTP on my phone"`
- `"My laptop won't connect to the office Wi-Fi"`
- `"How do I request access to a new software tool?"`
- `"What's the process for resetting my SSO password?"`

---

### 2. HR knowledge base — direct answer

```
you> How many PTO days do I get per year? Does unused time carry over?
```

**Route:** `supervisor → hr_kb`

SmartDesk retrieves from the HR knowledge base (HuggingFace
`strova-ai/hr-policies-qa-dataset` + synthetic HR docs) and returns the policy.

```
---
*Sources: Pto Policy, Hr Policies Qa Dataset · Confidence: High (83%)*
```

Other HR queries to try:
- `"When is the benefits open enrollment window?"`
- `"How do I submit an expense reimbursement?"`
- `"What's the remote work policy?"`
- `"How does the performance review cycle work?"`

---

### 3. Combined IT + HR — synthesizer agent

```
you> I'm a new hire — what are the IT onboarding steps I need to complete,
     and what HR enrollment deadlines should I be aware of?
```

**Route:** `supervisor → combined_kb`

This is the only query type that activates both specialist agents in a single
turn. `combined_knowledge_node` runs `retrieve(domain="it")` and
`retrieve(domain="hr")` independently, merges the chunk pools, and asks the
LLM to synthesize a unified response organized in two labelled sections:

```
**IT:**
1. Submit a laptop request to IT via the equipment portal...
2. Enroll in **MFA** through the **SSO portal** before your start date...
3. Connect to **VPN** using GlobalProtect with your SSO credentials...

**HR:**
Benefits enrollment opens within the first **30 days** of your start date...
Submit your **direct deposit form** via the HR portal before your first paycheck...

---
*Sources: Laptop Setup Guide, Onboarding Checklist, Benefits Enrollment · IT Confidence: High (85%) · HR Confidence: Medium (71%)*
```

The supervisor routes to `combined_kb` only when the query explicitly mentions
both IT and HR topics. Single-domain questions still route to `it_kb` or `hr_kb`
directly.

> **Implementation note:** `combined_knowledge_agent.py` handles all three
> sub-cases: both domains answer confidently (full synthesis), one domain
> escalates (partial answer + escalation note for the gap), or both escalate
> (suggest a ticket).

---

### 4. Hybrid search — sparse + dense retrieval

```
you> What is TOTP and how do I enroll in MFA?
```

**Route:** `supervisor → it_kb`

This query exercises the hybrid retrieval path in Qdrant:

- **Sparse (BM25)** prefetch catches "TOTP" and "MFA" as exact token matches —
  acronyms that dense embeddings can miss on short queries.
- **Dense (semantic)** rerank surfaces related setup docs even when the user's
  phrasing differs from the document wording.

Both signals are fused via Qdrant's Query API (sparse prefetch → dense rerank).
The `[it_kb]` log line shows the number of chunks retrieved and the top score.

Other queries that stress-test sparse vs dense:
- `"SSO login keeps failing"` — exact acronym, strong BM25 signal
- `"LDAP directory sync errors"` — less common term, tests dense fallback
- `"My MFA TOTP code is rejected at the SSO portal"` — all signals together

---

### 5. Escalation + ticket creation

A two-turn flow. First, ask something outside the knowledge base:

**Turn 1 — triggers escalation:**
```
you> How do I provision EC2 instances for my team's project?
```

**Route:** `supervisor → it_kb` → `should_escalate=True`
(EC2 provisioning is intentionally absent from IT docs — confidence will be well below `CONFIDENCE_THRESHOLD=0.4`)

```
SmartDesk: I searched our IT knowledge base but couldn't find a confident
answer (relevance: 18%).

This topic may not be fully documented yet, or it may need specialist input.

To open a support request, just say:
  "Create a ticket — [brief description of your issue]"
and I'll take care of the rest.
```

**Turn 2 — trigger ticket creation:**
```
you> Create a ticket — need EC2 provisioning access for the ML project
```

**Route:** `supervisor → create_ticket`

SmartDesk:
1. LLM extracts `summary`, `description`, `category` (IT Support), `priority` (High) from your message
2. Asks for your **email address** (first time only — cached for the rest of the session)
3. Shows the proposed ticket and waits for **HITL confirmation**:

```
[HITL] Proposed ticket:
  Summary     : Need EC2 provisioning access for the ML project
  Description : User needs EC2 provisioning access for the ML project.
  Category    : IT Support
  Priority    : High
  Requester   : you@company.com

Proceed? [y/N]: y
```

**Mock output** (default, `TICKETING_PROVIDER=mock`):
```
SmartDesk: Ticket created successfully.
  ID       : MOCK-1
  Summary  : Need EC2 provisioning access for the ML project
  Category : IT Support | Priority: High
```

**Jira output** (`TICKETING_PROVIDER=jira` + credentials in `.env`):
```
SmartDesk: Ticket created successfully.
  ID   : SD-42
  View : https://yoursite.atlassian.net/browse/SD-42
```

The Jira client stores the requester email as a label (`requester:you_at_company_com`)
so tickets can later be retrieved by email using JQL — no custom Jira field required.

---

### 6. Ticket status check

```
you> What tickets do I have open?
```

**Route:** `supervisor → ticket_status`

SmartDesk reads the email from the session cache (set when the ticket was created)
or asks for it if not yet provided. It then calls `get_tickets_by_email()`:

- **Mock:** returns in-memory tickets created this session
- **Jira:** runs `JQL: project = "SD" AND labels = "requester:<email>" ORDER BY created DESC`

Sample output (after the escalation + ticket creation example above):
```
SmartDesk: You have 1 ticket on file for you@company.com:

  • MOCK-1 — Need EC2 provisioning access for the ML project  [To Do]
```

Zero tickets: `"No open tickets found for you@company.com."`

---

### Intentionally uncovered topics (force escalation)

These topics are deliberately absent from both knowledge bases so the escalation
path is always testable.

| Query | Route | Why it escalates |
|---|---|---|
| `"How do I provision EC2 / Kubernetes?"` | `it_kb → escalate` | Not in IT docs |
| `"Process a GDPR data deletion request"` | `it_kb → escalate` | Not in IT docs |
| `"Tell me about executive equity and vesting"` | `hr_kb → escalate` | Not in HR docs |
| `"What HIPAA training is required for my role?"` | `hr_kb → escalate` | Not in HR docs |
| `"How does parental leave work?"` | `hr_kb → escalate` | Not in HR docs |

---

## Knowledge base design (Option C)

The KB combines two sources deliberately:

- **Real data** — [`strova-ai/hr-policies-qa-dataset`](https://huggingface.co/datasets/strova-ai/hr-policies-qa-dataset) from HuggingFace
- **Synthetic data** — 12 IT + 12 HR Q&A docs generated by the LLM via `generate_synthetic_docs.py`

Certain topics are **intentionally left uncovered** (see table above) so the
escalation path is always exercisable during a demo.

## Hybrid search

Every chunk is indexed with **both** a dense vector (semantic meaning) and a sparse
vector (keyword/BM25). At query time:

1. Sparse prefetch — keyword match catches IT acronyms (MFA, VPN, TOTP, SSO) precisely
2. Dense rerank — semantic similarity re-orders the candidates

The system degrades gracefully to dense-only if sparse embeddings are not configured.

This satisfies the **hybrid search bonus rubric item**.

## HITL flow

```
User: "Create a ticket — my laptop won't connect to VPN"
  │
  ▼
Supervisor → create_ticket
  │
  ▼
ticket_creation_agent:
  1. LLM extracts summary + description + category + priority from query
  2. Validates email (from session cache or extracted from query)
  3. ⚠ HITL gate — prints proposed ticket, waits for y/N
  4. On yes → ticketing_client.create_ticket() → MOCK-1 (or SD-42 for Jira)
  5. Returns ticket ID + confirmation message
```

Ticket creation is **never** skipped past the HITL gate. There is no code path
that calls `create_ticket()` before `confirm_action()` returns `True`.

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
| `CONFIDENCE_THRESHOLD` | `0.4` | Escalation threshold (0–1) |
| `TICKETING_PROVIDER` | `mock` | `mock` or `jira` |
| `HITL_MODE` | `cli` | `cli` (stdin) or `ui` (Gradio) |
| `JIRA_EMAIL` | — | Atlassian account email (Jira only) |

## Status

| Component | Status |
|---|---|
| RAG pipeline (ingest → embed → retrieve) | ✅ Implemented |
| Hybrid search (dense + sparse, Qdrant local) | ✅ Implemented |
| Supervisor routing (LLM-based, 6-way) | ✅ Implemented |
| IT / HR knowledge agents | ✅ Implemented |
| Combined IT+HR synthesizer agent | ✅ Implemented |
| Ticket creation agent + HITL | ✅ Implemented |
| Jira live integration | ✅ Implemented |
| Ticket status agent | ✅ Implemented |
| LangGraph orchestrator | ✅ Implemented |
| CLI entry point | ✅ Implemented |
| Grounding guardrail + response enhancement | ✅ Implemented |
| Gradio UI (bonus) | ✅ Implemented |
| Evaluation pipeline | 🔜 Next milestone |
