# Architecture

## High-level flow

```
                              ┌─────────────────────┐
                              │        User          │
                              │  (CLI or Gradio UI)  │
                              └──────────┬───────────┘
                                         │ query
                                         ▼
                              ┌─────────────────────┐
                              │   Supervisor Agent   │
                              │  LLM classifies →    │
                              │  sets state["route"] │
                              └──────────┬───────────┘
   ┌──────┬──────────┬───────────────────┼────────────────┬──────────────┐
   ▼      ▼          ▼                   ▼                ▼              ▼
it_kb  hr_kb  combined_kb        create_ticket    ticket_status   off_topic
  │      │         │                    │                │            │
  │      │    (both IT+HR               │                │            │
  │      │    retrieval +               │                │            │
  │      │    synthesizer)              │                │            │
  └──────┴─────────┴──────┐    ┌────────────┐  ┌──────────────────┐  │
                           │    │ HITL gate  │  │ Ticketing client │  │
  ┌────────────────────┐   │    │ confirm_   │  │ Mock → Jira API  │  │
  │ Qdrant local store │   │    │ action()   │  │                  │  │
  │ hybrid dense+sparse│   │    └─────┬──────┘  └──────────────────┘  │
  │ named vectors      │   │          │ confirmed                       │
  └────────────────────┘   │          ▼                                 │
                           │  ┌──────────────────┐                      │
                           │  │ create_ticket()  │                      │
                           │  │ → ticket_id      │                      │
                           │  └──────────────────┘                      │
                           └─────────────────────────── all → [END] ◄───┘
                                                                ▼
                                                  state["response"] → user
```

---

## Components

### Orchestrator (`orchestrator/`)

**`state.py`** defines `AgentState`, the single TypedDict that flows through every
LangGraph node. Key fields:

| Field | Set by | Purpose |
|---|---|---|
| `query` | `main.py` | Raw user input |
| `email` | `main.py` / ticket agents | User email for ticket ops |
| `route` | supervisor | Which node LangGraph dispatches to |
| `retrieved_chunks` | KB agents | List of `RetrievedChunk` (text, source, score) |
| `confidence_score` | KB agents / retriever | Top retrieval score 0–1 |
| `should_escalate` | KB agents | True when KB can't answer confidently |
| `ticket_id` | ticket creation | ID of created ticket |
| `response` | every agent | Final text shown to user |

**`graph.py`** — `build_orchestrator()` compiles the LangGraph `StateGraph` once
at startup. `run_once(graph, state)` invokes it per turn.

---

### Supervisor (`agents/supervisor.py`)

Zero-shot LLM classifier. Sends the query to the LLM with a prompt listing six
valid route names. LLM responds with exactly one. Falls back to `off_topic` on
any error — the pipeline never crashes on a routing failure.

| Route | Trigger |
|---|---|
| `it_kb` | IT questions: VPN, MFA, SSO, software, hardware, network |
| `hr_kb` | HR questions: PTO, benefits, policies, onboarding |
| `combined_kb` | Query explicitly spans both IT and HR domains |
| `create_ticket` | Explicit ticket creation request |
| `ticket_status` | Check/list existing tickets |
| `off_topic` | Anything outside IT/HR scope |

---

### IT / HR Knowledge Agents

Both agents follow the same RAG flow, scoped to their domain:

```
retrieve(query, domain="it" or "hr")
  ├─ query_dense_embedding(query)        → 1536-dim float vector (OpenAI)
  ├─ query_sparse_embedding(query)       → {indices, values} (fastembed BM25)
  └─ Qdrant Query API:
       prefetch(sparse, limit=top_k×3)   → keyword candidates
       rerank with dense vector          → top_k semantically best chunks

decide_escalation(chunks)
  ├─ no chunks → escalate
  └─ top_score < CONFIDENCE_THRESHOLD → escalate

if escalate:
  └─ "I don't have enough information — consider creating a ticket"

else:
  └─ build_grounded_prompt(query, chunks)
  └─ call_llm(GROUNDED_SYSTEM_INSTRUCTIONS, prompt)   → answer
  └─ check_grounding(answer, chunks)                  → warn if low overlap
```

---

### Combined Knowledge Agent (`agents/combined_knowledge_agent.py`)

Activated only when the supervisor routes to `combined_kb` — i.e. the query explicitly
spans both IT and HR topics (e.g. "new hire IT setup AND HR enrollment deadlines").

```
retrieve(query, domain="it")    → it_chunks
retrieve(query, domain="hr")    → hr_chunks
decide_escalation on each domain independently
  both confident  → merge chunks → call_llm(_COMBINED_SYSTEM_INSTRUCTIONS) → synthesized answer
  one escalates   → answer from confident domain + escalation note for the other
  both escalate   → "couldn't find confident answers in either KB — create a ticket"

Response format:
  **IT:** <numbered steps or answer>
  **HR:** <numbered steps or answer>
  ---
  *Sources: ... · IT Confidence: High (85%) · HR Confidence: Medium (62%)*
```

---

### Ticket Creation Agent (`agents/ticket_creation_agent.py`)

```
LLM extracts {summary, description, category, priority} from query (JSON)
Validate email (is_valid_email) — from state or extracted via LLM
⚠ HITL gate: confirm_action(summary, description)
  ├─ CLI:  blocks on input("Proceed? [y/N]")
  └─ UI:   returns confirmation message; next turn is the answer
On confirm:
  └─ ticketing_client.create_ticket()  [@with_retry(max_attempts=3)]
  └─ state["ticket_id"] = ticket["id"]
On deny:
  └─ "Ticket creation cancelled"
```

HITL is **mandatory and cannot be bypassed** — there is no code path that calls
`create_ticket()` before `confirm_action()` returns `True`.

---

### Ticket Status Agent (`agents/ticket_status_agent.py`)

```
Resolve email → state["email"] or LLM extraction from query
Validate email (is_valid_email)
get_tickets_by_email(email)
Format response:
  0 tickets → "No open tickets for <email>"
  1 ticket  → ID, summary, status, URL
  N tickets → numbered list with status per ticket
```

---

### RAG Pipeline (`rag/`)

| File | Responsibility |
|---|---|
| `ingestion.py` | HuggingFace HR loader, local JSON loader, sliding-window chunker |
| `embeddings.py` | `make_dense_embedding` / `query_dense_embedding` (OpenAI or fastembed); `make_sparse_embedding` / `query_sparse_embedding` (fastembed BM25); module-level model cache |
| `vector_store.py` | Qdrant local client; `add_documents()` upserts dense + sparse named vectors in batches of 100; `similarity_search()` sparse prefetch → dense rerank; `persist()` / `load()` are no-ops (local mode auto-persists) |
| `retriever.py` | `retrieve()` singleton wrapper; `decide_escalation()` threshold strategy with LLM self-assessment path commented in for easy switching |

---

### Tools (`tools/`)

**`hitl.py`** — mandatory HITL gate. CLI: `input()`. UI: raises so Gradio
handles confirmation as a multi-turn exchange (controlled by `HITL_MODE` env var).

**`ticketing/`**:
- `base.py` — `TicketingClient` ABC + `Ticket` TypedDict
- `mock_client.py` — in-memory mock, sequential IDs (`MOCK-1, MOCK-2, ...`)
- `ticketing_client.py` — `RealTicketingClient` — Jira-backed live integration (labels email as `requester:<email>`, uses JQL for lookup)
- `__init__.py` — `get_ticketing_client()` singleton factory

---

### Guardrails (`guardrails/`)

**`grounding.py`**:
- `GROUNDED_SYSTEM_INSTRUCTIONS` — tells the LLM to answer only from context
- `build_grounded_prompt()` — formats chunks + query into a structured block
- `check_grounding()` — keyword overlap sanity check; warns but does not block

**`validation.py`**:
- `is_valid_email()` — regex check used by both ticket agents
- `with_retry()` — decorator applied to `create_ticket()` for API resilience

---

### Shared LLM adapter (`agents/_llm.py`)

`call_llm(system, user, temperature)` dispatches to OpenAI or Anthropic based
on `settings.llm_provider`. All agent LLM calls go through this single function.

---

## The hardest design decision: escalation

The spec identifies this as the central design call. Three options evaluated:

1. **Score threshold** *(implemented)* — `escalate = top_score < 0.4`. Fast, no
   extra API call. Brittle if embeddings are poorly calibrated.
2. **LLM self-assessment** *(documented + commented-out in `retriever.py`)* — ask
   the LLM "can you answer this from the context?". More accurate, costs one
   extra call per turn.
3. **Hybrid** — threshold for speed, LLM check for borderline cases.

See `docs/DESIGN_DECISIONS.md §6` for the full trade-off analysis.
