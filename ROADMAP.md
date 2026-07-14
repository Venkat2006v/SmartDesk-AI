# Suggested build order

This mirrors the build order suggested in the capstone guide. You don't have
to follow it exactly, but each step is meant to keep you with a runnable
system as you go.

- [ ] **1. Knowledge base** — Source IT + HR docs (Hugging Face dataset,
      Kaggle, or LLM-generated synthetic Q&A). Aim for 30–50+ chunks per
      domain. Leave a few topics deliberately uncovered so escalation is
      testable later. See `data/README.md`.
- [ ] **2. Ingestion + chunking** — `src/smartdesk/rag/ingestion.py`.
- [ ] **3. Embeddings + vector store** — `src/smartdesk/rag/embeddings.py`,
      `src/smartdesk/rag/vector_store.py`. Pick one vector DB (FAISS/Chroma/
      Pinecone/Qdrant/pgvector) and record the choice in
      `docs/DESIGN_DECISIONS.md`.
- [ ] **4. Retriever + escalation logic** — `src/smartdesk/rag/retriever.py`.
      This is the hardest design call in the project. Decide: retrieval-score
      threshold, LLM self-assessment, or a hybrid — then implement
      `decide_escalation`.
- [ ] **5. IT + HR knowledge agents** — wire retrieval into grounded prompts
      (`src/smartdesk/guardrails/grounding.py`) and generate answers.
- [ ] **6. Supervisor / router** — `src/smartdesk/agents/supervisor.py`,
      replacing the hardcoded stub classification with real routing logic
      (rule-based, LLM-based, or both).
- [ ] **7. Ticket creation agent** — `src/smartdesk/agents/
      ticket_creation_agent.py` + `src/smartdesk/tools/hitl.py`. Confirm
      before writing. Use `MockTicketingClient` first, swap to a real
      ticketing API later if you want full credit on that path.
- [ ] **8. Ticket status agent** — `src/smartdesk/agents/
      ticket_status_agent.py`. Handle zero/one/multiple ticket results.
- [ ] **9. Orchestrator graph** — `src/smartdesk/orchestrator/graph.py`.
      Wire the agents together with your framework of choice (LangGraph,
      CrewAI, AutoGen, raw function-calling, or custom).
- [ ] **10. Error handling + guardrails pass** — retries, malformed input,
      empty KB results, ticketing API failures.
- [ ] **11. Tests** — fill in the skipped tests in `tests/`, add more as
      needed.
- [ ] **12. Bonus (optional)** — hybrid search, evaluation pipeline
      (`src/smartdesk/evaluation/eval_pipeline.py`), deployed UI
      (`src/smartdesk/ui/app.py`), caching, cross-session memory.
