# Architecture

## High-level flow

```
                         ┌─────────────────────┐
                         │        User          │
                         └──────────┬───────────┘
                                    │ query (+ email, for ticket ops)
                                    ▼
                         ┌─────────────────────┐
                         │   Supervisor Agent    │
                         │  (classify -> route)  │
                         └──────────┬───────────┘
                  ┌───────────┬─────┴──────┬───────────────┐
                  ▼           ▼            ▼               ▼
          ┌─────────────┐ ┌─────────┐ ┌───────────────┐ ┌───────────────┐
          │ IT Knowledge│ │HR Know- │ │Ticket Creation │ │Ticket Status  │
          │   Agent     │ │ledge Ag.│ │    Agent       │ │    Agent      │
          │  (RAG)      │ │ (RAG)   │ │ (HITL confirm) │ │               │
          └──────┬──────┘ └────┬────┘ └───────┬───────┘ └───────┬───────┘
                 │             │              │                  │
                 ▼             ▼              ▼                  ▼
          ┌─────────────────────────┐  ┌─────────────────────────────┐
          │   Vector store (RAG)     │  │   Ticketing client            │
          │  retriever -> escalation │  │  (Mock now, real API later)   │
          └─────────────────────────┘  └─────────────────────────────┘
```

## Components

- **Supervisor agent** (`agents/supervisor.py`) — classifies the incoming
  query into a `Route` (`it_kb`, `hr_kb`, `create_ticket`, `ticket_status`,
  `off_topic`) and dispatches to the matching node. TODO: replace the
  hardcoded stub with real classification logic (rules, LLM call, or both).

- **IT / HR Knowledge agents** (`agents/it_knowledge_agent.py`,
  `agents/hr_knowledge_agent.py`) — deliberately split into two agents
  (rather than one generic KB agent) so the project demonstrates genuine
  multi-agent specialization. Each retrieves from its own domain-filtered
  slice of the vector store, builds a grounded prompt
  (`guardrails/grounding.py`), and only answers from retrieved content.

- **Ticket Creation agent** (`agents/ticket_creation_agent.py`) — collects
  the minimum required fields (email, summary, description), and **must**
  get explicit human confirmation (`tools/hitl.py`) before calling the
  ticketing client. Never create a ticket silently.

- **Ticket Status agent** (`agents/ticket_status_agent.py`) — looks up
  tickets by email via the ticketing client and reports status. Must
  handle zero, one, and multiple results distinctly.

- **RAG pipeline** (`rag/`) — `ingestion.py` (load + chunk) →
  `embeddings.py` (dense + sparse) → `vector_store.py` (Qdrant — hybrid
  index + similarity search) → `retriever.py` (retrieve + decide whether
  to escalate).

- **Orchestrator** (`orchestrator/`) — `state.py` defines the shared
  `AgentState` passed between nodes; `graph.py` wires the agents into a
  LangGraph `StateGraph` (chosen framework — see
  `docs/DESIGN_DECISIONS.md`; a hand-rolled dispatcher works too if you'd
  rather skip the dependency).

- **Tools** (`tools/`) — `hitl.py` (confirmation gate),
  `ticketing/` (abstract `TicketingClient` + a working `MockTicketingClient`
  + a `RealTicketingClient` stub for Jira/Asana/etc.).

- **Guardrails** (`guardrails/`) — `grounding.py` (build grounded prompts,
  check answers don't drift from retrieved content), `validation.py`
  (email validation, retry-with-backoff decorator — both fully implemented
  already).

## The hardest design decision: confidence / escalation

When should the knowledge agents say "I don't know, let me create a
ticket for you" instead of answering? Three options, all valid:

1. **Retrieval-score threshold** — escalate if the top similarity score
   is below some cutoff. Simple, fast, but picking the right threshold is
   guesswork until you've seen real query/KB behavior.
2. **LLM self-assessment** — ask the LLM "can you answer this confidently
   from the provided context?" after retrieval. More adaptive, costs an
   extra call, and the LLM can be overconfident.
3. **Hybrid** — combine both, and/or add a HITL confirmation step before
   actually escalating to a ticket ("I couldn't find a confident answer —
   want me to open a ticket?").

This is intentionally left as a `TODO` in `rag/retriever.py::decide_escalation`.
Pick one, document your reasoning in `docs/DESIGN_DECISIONS.md`, and only
revisit it once you have a real knowledge base to test against (including
the deliberately-uncovered topics — see `data/README.md`).
