# SmartDesk AI — IT & HR Operations Agent

A multi-agent, RAG-grounded helpdesk assistant that answers IT and HR policy
questions from a knowledge base, creates support tickets with mandatory
human-in-the-loop (HITL) confirmation, and checks the status of existing
tickets — all routed through a Supervisor agent.

This repository is a **scaffold**: the architecture, module boundaries, and
function signatures are in place, but the core logic is intentionally left
as `TODO`s for you to implement. That's the point of the exercise — see
`docs/ARCHITECTURE.md` and `docs/DESIGN_DECISIONS.md` before you start.

## Capabilities

1. **Knowledge base Q&A (RAG)** — answer IT and HR questions grounded only
   in retrieved documents, split across two specialist agents (IT, HR).
2. **Ticket creation** — collect required fields, confirm with the user
   (HITL) before writing, then create a ticket via a ticketing client.
3. **Ticket status check** — look up existing ticket(s) for a user by email
   and report status, handling zero/one/many results.

A Supervisor agent classifies each incoming query and routes it to the
right specialist agent.

## Project layout

```
SmartDesk-AI/
├── docs/                    # Architecture + your own design-decision log
├── data/                    # Raw sources, processed chunks, knowledge base
├── src/smartdesk/
│   ├── orchestrator/        # Multi-agent graph + shared state definition
│   ├── agents/               # Supervisor, IT/HR knowledge, ticket agents
│   ├── rag/                  # Ingestion, embeddings, vector store, retriever
│   ├── tools/                 # Ticketing client(s) + HITL confirmation
│   ├── guardrails/           # Grounding checks, validation, retry helper
│   ├── evaluation/           # Eval harness (bonus)
│   └── ui/                    # Optional chat UI (bonus)
├── scripts/                  # One-off CLI scripts (build KB, gen synthetic docs)
└── tests/                     # Pytest suite (some tests fully runnable, others skipped TODOs)
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your API keys
python -m smartdesk.main
```

Until you implement the `TODO`s, `python -m smartdesk.main` will run but
the orchestrator / agents will raise `NotImplementedError` where logic is
missing. The `MockTicketingClient` and validation/retry helpers are fully
wired up already so you have a working baseline to build against.

## Where to start

See `ROADMAP.md` for a suggested build order, and `docs/ARCHITECTURE.md`
for the system diagram and the open design questions (especially the
confidence/escalation decision — it's the hardest part of this project).

## Status

🚧 Scaffold only. Logic not yet implemented — see `TODO` markers throughout
`src/smartdesk/`.
