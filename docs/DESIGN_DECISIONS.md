# Design Decisions

This document records the key architectural and implementation choices made for
SmartDesk AI, and the reasoning behind each. Every decision below was made
deliberately — this is not a default scaffold.

---

## 1. Orchestration framework — LangGraph

**Choice:** LangGraph `StateGraph`

**Why:**
- Explicit state-machine model maps directly onto the problem: each query takes
  exactly one path through a fixed set of nodes (supervisor → specialist → END).
- `AgentState` is a plain `TypedDict` — every field is visible and type-checked,
  with no hidden message-list state.
- Conditional edges make routing logic readable: the supervisor sets
  `state["route"]` and LangGraph dispatches to the matching node.
- Lighter dependency footprint than Google ADK and no GCP lock-in.
- Framework-agnostic agent nodes: the node functions are plain Python; swapping
  the graph runner wouldn't require rewriting any agent logic.

**Rejected alternatives:**
- *CrewAI* — role-based abstraction suits creative pipelines but adds
  unnecessary indirection for a well-defined 5-route dispatch problem.
- *Raw function-calling* — viable but loses the explicit state graph, making
  escalation and HITL harder to reason about and test.

---

## 2. LLM provider — OpenAI (default), Anthropic (alternative)

**Choice:** `LLM_PROVIDER=openai`, model `gpt-4o-mini` by default.

**Why:**
- Strong price/performance ratio for the four LLM calls per turn: routing
  (supervisor), grounded answer (KB agent), field extraction (ticket creation),
  and optional email extraction.
- The `agents/_llm.py` adapter makes the provider fully swappable: setting
  `LLM_PROVIDER=anthropic` requires no code changes.
- Temperature 0.0 for deterministic tasks (routing, JSON extraction);
  0.2 for answer generation — documented at each call site.

---

## 3. Vector store — Qdrant (local embedded mode)

**Choice:** `QdrantClient(path=settings.vector_index_dir)` — no Docker or
server required for development.

**Why:**
- Qdrant is the only mainstream vector DB with native named vectors: a single
  point carries both a `"dense"` float vector and a `"sparse"` SparseVector,
  queried independently via the Query API. This enables true hybrid search
  without a separate keyword index.
- Local embedded mode removes all infrastructure friction for development and
  the capstone demo.
- Upgrading to Qdrant Cloud requires a single line change in `get_vector_store()`:
  replace `QdrantClient(path=...)` with `QdrantClient(url=..., api_key=...)`.

---

## 4. Embeddings — hybrid dense + sparse

**Dense:** OpenAI `text-embedding-3-small` (default) or fastembed
`BAAI/bge-small-en-v1.5` (free, local, no API key — set `EMBEDDING_PROVIDER=fastembed`).

**Sparse:** fastembed `SparseTextEmbedding` with `Qdrant/bm25` (default BM25).

**Why hybrid:**

| Query type | Dense | Sparse (BM25) |
|---|---|---|
| "How do I reset my password?" | ✅ semantic match | ✅ keyword match |
| "What is TOTP?" | ✅ knows acronym context | ✅ exact token match |
| "MFA enrollment steps" | ⚠ may miss short query | ✅ exact acronym hit |

IT support queries are acronym-heavy (MFA, VPN, SSO, TOTP, LDAP). BM25
catches exact keyword matches that dense embeddings can miss on short queries.
The hybrid path (sparse prefetch → dense rerank) combines both signals.
The system degrades gracefully to dense-only if sparse raises `NotImplementedError`.

This satisfies the **hybrid search bonus rubric item**.

---

## 5. Knowledge base sourcing — Option C (real + synthetic)

**IT docs:** 12 LLM-generated Q&A documents covering: VPN setup, MFA/TOTP
enrollment, SSO, password reset, laptop setup, Wi-Fi, printer, software
installs, access requests, cloud tools, email, and security policies.

**HR docs:** `strova-ai/hr-policies-qa-dataset` from HuggingFace (real data)
+ 12 LLM-generated Q&A documents covering: PTO, benefits enrollment,
onboarding, offboarding, payroll, performance reviews, remote work, expense
reimbursement, code of conduct, leave of absence, training, and hiring.

**Deliberate gaps (untrained topics — for escalation testing):**
- IT: AWS EC2 provisioning, Kubernetes deployments, GDPR data deletion requests
- HR: parental leave specifics, HIPAA training requirements, executive equity plans

These gaps ensure the escalation path is testable — the agent must create a
ticket rather than hallucinate when these topics are queried.

---

## 6. Confidence / escalation strategy — retrieval score threshold

**Implemented:** score threshold with LLM self-assessment as a documented alternative.

```python
should_escalate = (len(chunks) == 0) OR (chunks[0]["score"] < CONFIDENCE_THRESHOLD)
confidence      = chunks[0]["score"]   # Qdrant cosine similarity, 0.0–1.0
```

**Default threshold:** `CONFIDENCE_THRESHOLD=0.4`
- Below 0.4: top chunk is weakly related → escalate, suggest ticket.
- Above 0.4: top chunk is a reasonable match → generate grounded answer.

**Failure modes acknowledged:**
- *Too low (0.0):* agent answers everything including borderline queries —
  the spec's "answers everything" common pitfall. Avoided by setting 0.4.
- *Too high (0.7+):* agent escalates too aggressively, even for well-covered
  topics — the "escalates everything" pitfall.

**LLM self-assessment path** is implemented but commented out in
`rag/retriever.py`. To switch: uncomment the block and set
`CONFIDENCE_THRESHOLD=0.0`. Costs one extra API call per query but is more
semantically aware. Trade-off: latency and cost vs. accuracy on edge cases.

**Note — escalation threshold vs. response confidence label:**
The escalation decision (`should_escalate`) and the confidence label shown in
the response footer (`High / Medium / Low`) are two separate mechanisms:

| Score range | Escalation decision | Response footer label |
|---|---|---|
| < 0.4 | Escalate — no answer, suggest ticket | — (never reached) |
| 0.4 – 0.7 | Answer | Medium |
| ≥ 0.7 | Answer | High |

A "Medium (52%)" label in a response means the retrieval score was above the
escalation threshold (so answering is correct) but below the High band. It is
a transparency signal to the user — not an indication that the answer should
have been withheld. Escalation is triggered exclusively by the threshold check;
the label is informational only.

---

## 7. Ticketing — MockTicketingClient + Jira live integration

**Default:** `MockTicketingClient` — fully in-memory, sequential IDs
(`MOCK-1`, `MOCK-2`, ...), no credentials required. Used for development and
the capstone demo (the FAQ explicitly allows mocking with clearly documented
dummy data).

**Live Jira integration:** `RealTicketingClient` is fully implemented in
`tools/ticketing/ticketing_client.py`. Activate by setting:
```env
TICKETING_PROVIDER=jira
JIRA_EMAIL=you@atlassian.com
TICKETING_API_KEY=<api-token>
TICKETING_BASE_URL=https://yoursite.atlassian.net
TICKETING_PROJECT_KEY=SD
```

**Email tracking in Jira:** Jira labels can't contain `@` or `.`, so requester
email is encoded as `requester:you_at_company_com` on every created issue. Ticket
lookup uses JQL `project=SD AND labels="requester:<encoded>"` — no custom field
needed, works on the free tier.

---

## 8. HITL — mandatory, never skipped

**Implementation:** `tools/hitl.py::confirm_action()` is called inside
`ticket_creation_agent.py` before every `create_ticket()` call. There is no
code path that creates a ticket silently.

**CLI mode:** blocks on `input("Proceed? [y/N]")` — user must type `y`/`yes`.

**Gradio UI mode:** handled as a multi-turn exchange — the agent returns a
confirmation prompt, and the next user message is interpreted as the answer.
Controlled by `HITL_MODE=ui` (set automatically by `ui/app.py`).

**Why mandatory:** ticket creation is a write to an external system — a
side-effecting, externally-visible action. This mirrors the broader principle:
always confirm before irreversible or external actions.

---

## 9. Multi-agent IT/HR split — separate specialist nodes

**Choice:** two separate agent nodes (`it_knowledge_node`, `hr_knowledge_node`)
rather than one generic KB agent.

**Why:**
- Domain isolation: each agent passes its own `domain=` filter to Qdrant,
  ensuring IT queries never retrieve HR chunks and vice versa.
- Independent escalation messages tailored per domain.
- Satisfies the **multi-agent IT/HR split bonus rubric item** explicitly.
- Easy to extend: domain-specific system prompts, retrieval parameters, or
  different LLM models per domain can be added without touching the other agent.

---

## 10. Evaluation pipeline

**Implemented:** `evaluation/eval_pipeline.py` + `scripts/run_evaluation.py`

**Approach:** LLM-as-judge (no Ragas dependency). Uses the same `call_llm()`
adapter already wired throughout the system — the same provider and model
configured in `.env`.

**Three metric categories:**

| Category | Metrics | How |
|---|---|---|
| Routing accuracy | % correct routes | Compare supervisor output to expected_route |
| Escalation accuracy | Precision / Recall / F1 | Compare should_escalate to CONFIDENCE_THRESHOLD decision |
| RAG quality | Faithfulness, Answer Relevance | LLM judge scores each 0–5, normalized to 0–1 |

**Test suite:** 20 cases across all routes and escalation scenarios. A 6-case
`MINIMAL_TEST_SUITE` enables fast smoke tests without LLM judge calls.

**Why LLM-as-judge over Ragas:**
- Ragas requires a separate evaluation LLM and dataset format overhead
- Our `call_llm()` adapter already handles both OpenAI and Anthropic — the judge
  uses the same provider with no additional setup
- Faithfulness judge checks if the answer uses only retrieved context (aligns
  directly with the grounding guardrail we already enforce)
- Relevance judge checks if the answer addresses the question (catches hallucinations
  that technically overlap with context but don't answer the query)

**LLM judge prompts** use a 0–5 integer scale, then normalize to 0–1. Temperature
is set to 0.0 for deterministic scores.

**Usage:**
```bash
# Full suite with LLM judges
python scripts/run_evaluation.py

# Fast structural check (routing + escalation only, zero API calls beyond the KB)
python scripts/run_evaluation.py --skip-llm-judges

# Six-case smoke test
python scripts/run_evaluation.py --suite minimal --skip-llm-judges

# Save results to JSON for tracking over time
python scripts/run_evaluation.py --output eval_results.json
```

---

## 11. Observability — LangSmith tracing

**Why LangSmith over Prometheus/Grafana:**
Prometheus + Grafana excel at infrastructure metrics (request rate, CPU, pod health).
LangSmith is purpose-built for the AI-agent layer — it understands LLM calls, token
counts, retrieved context, and multi-step agent traces. Both can coexist in a production
deployment, but for an AI agent the LLM-layer observability is more actionable.

**What gets traced per query:**

```
graph.invoke()                            ← LangGraph auto-traces this
  └─ supervisor_node                      ← node span (input: query, output: route)
  └─ it_knowledge_node                    ← node span (input: query, output: response)
       └─ call_llm [run_type="llm"]       ← @traceable span
            └─ openai.chat.completions    ← wrap_openai captures token counts
```

**Three integration points in the code:**

| Layer | How | File |
|---|---|---|
| LangGraph graph run | Auto-traced when env vars set | `orchestrator/graph.py` (no changes needed) |
| `call_llm()` | `@traceable(name="call_llm", run_type="llm")` | `agents/_llm.py` |
| OpenAI / Anthropic SDK | `wrap_openai()` / `wrap_anthropic()` client wrappers | `agents/_llm.py` |

The `@traceable` decorator and SDK wrappers are **no-ops** when `LANGCHAIN_TRACING_V2`
is not set — zero runtime cost in environments without LangSmith configured.

**Setup (2 minutes):**
1. Sign up free at https://smith.langchain.com
2. Settings → API Keys → Create API Key
3. Add to `.env`:
```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=smartdesk-ai
```
4. Run the app — every query now appears as a trace in the LangSmith UI.

**Startup indicator:** `main.py` prints tracing status on launch:
```
[tracing] LangSmith ENABLED — project: 'smartdesk-ai'
          View traces → https://smith.langchain.com
```

---

## 12. Bonus items summary

| Item | Status | Location |
|---|---|---|
| Hybrid search (dense + sparse) | ✅ Done | `rag/embeddings.py`, `rag/vector_store.py` |
| Multi-agent IT/HR split | ✅ Done | `agents/it_knowledge_agent.py`, `hr_knowledge_agent.py` |
| Combined IT+HR synthesizer | ✅ Done | `agents/combined_knowledge_agent.py` |
| Jira live integration | ✅ Done | `tools/ticketing/ticketing_client.py` |
| Response enhancement (citations + confidence) | ✅ Done | KB agent nodes + `GROUNDED_SYSTEM_INSTRUCTIONS` |
| Gradio UI | ✅ Done | `ui/app.py` |
| Evaluation pipeline (LLM-as-judge) | ✅ Done | `evaluation/eval_pipeline.py`, `scripts/run_evaluation.py` |
| LangSmith observability | ✅ Done | `agents/_llm.py`, `config.py`, `.env.example` |
| Caching | ❌ Not implemented | — |
| Cross-session memory | ❌ Not implemented | — |
