# Design decisions log

Use this file to record the choices you make and why. Keep entries short —
a sentence or two of reasoning is enough. This is also useful interview
material later ("why did you choose X over Y").

## Framework

- **Orchestration framework:** LangGraph
- **Why:** Explicit state-machine graph gives direct control over state
  transitions and maps cleanly onto the `AgentState` TypedDict already in
  `orchestrator/state.py`. Far fewer dependencies than Google ADK (~6 vs.
  ~45) and doesn't lock the project into the GCP/Gemini stack. ADK was the
  other finalist — it has a built-in eval CLI, OpenTelemetry
  instrumentation, and the A2A cross-framework protocol, but optimizes for
  "build fast and deploy to GCP" rather than fine-grained control.

## LLM

- **Provider/model:** TODO
- **Why:**

## Vector store

- **Backend:** Qdrant
- **Why:** Native support for both sparse and dense vectors on the same
  point, with a Query API that lets you prefetch with one and
  rerank/fuse with the other (RRF / RSF / DBSF). Directly satisfies the
  "hybrid search" bonus rubric item instead of requiring a hand-rolled
  fusion layer. Free to self-host via Docker, or run with no server at
  all via qdrant-client's embedded local mode.

## Embeddings

- **Provider/model (dense):** TODO
- **Sparse model (for hybrid search):** TODO — e.g. fastembed's bundled
  BM25 / SPLADE++ / miniCOIL sparse models
- **Why:**

## Ticketing

- **Provider:** TODO (mock / Jira / Asana / Notion / Linear / GitHub Issues)
- **Why:**
- Setup requirements (keys, base URLs, IDs) for Jira/Asana/Notion are
  documented in `.env.example`. No real keys are needed until you've
  picked exactly one and reached that build step — mock requires nothing.

## Escalation strategy

- **Approach:** TODO (threshold / LLM self-assessment / hybrid)
- **Threshold/parameters (if applicable):**
- **Why:**

## Knowledge base sourcing

- **IT docs source:** TODO
- **HR docs source:** TODO
- **Deliberate gaps (topics intentionally left uncovered, to test escalation):**

## Bonus items attempted

- [ ] Hybrid search
- [ ] Evaluation pipeline (LangSmith / Ragas / DeepEval)
- [ ] Deployed UI
- [ ] Caching
- [ ] Cross-session memory
