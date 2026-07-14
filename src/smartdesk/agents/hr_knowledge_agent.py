"""HR Knowledge Agent — RAG-grounded answers over the HR knowledge base.

Identical flow to it_knowledge_agent.py, scoped to domain="hr".

Flow:
  1. retrieve(query, domain="hr")           → chunks
  2. decide_escalation(query, chunks)        → (should_escalate, confidence)
  3a. Answerable: build prompt → call LLM → check_grounding → state["response"]
  3b. Escalate:   honest "I don't know" + should_escalate=True in state
"""

from __future__ import annotations

from smartdesk.agents._llm import call_llm
from smartdesk.guardrails.grounding import (
    GROUNDED_SYSTEM_INSTRUCTIONS,
    build_grounded_prompt,
    check_grounding,
)
from smartdesk.orchestrator.state import AgentState
from smartdesk.rag.retriever import decide_escalation, retrieve

_ESCALATION_MESSAGE = (
    "I searched our HR knowledge base but couldn't find a confident answer to your question. "
    "This topic may not yet be documented, or it may require personalised guidance from HR. "
    "I recommend creating a support ticket so an HR team member can follow up with you."
)


def hr_knowledge_node(state: AgentState) -> AgentState:
    """LangGraph node: retrieve HR docs, generate grounded answer or escalate."""
    query = state.get("query", "")
    print(f"[hr_kb] Retrieving for: {query!r}")

    # 1. Retrieve
    chunks = retrieve(query, domain="hr")
    print(f"[hr_kb] Retrieved {len(chunks)} chunks "
          f"(top score: {chunks[0]['score']:.3f})" if chunks else "[hr_kb] No chunks found")

    # 2. Escalation decision
    should_escalate, confidence = decide_escalation(query, chunks)
    print(f"[hr_kb] escalate={should_escalate}, confidence={confidence:.3f}")

    if should_escalate:
        return {
            **state,
            "retrieved_chunks": chunks,
            "confidence_score": confidence,
            "should_escalate": True,
            "response": _ESCALATION_MESSAGE,
        }

    # 3. Build grounded LLM answer
    user_prompt = build_grounded_prompt(query, chunks)
    answer = call_llm(system=GROUNDED_SYSTEM_INSTRUCTIONS, user=user_prompt, temperature=0.2)

    # Optional grounding check — warn but don't block
    if not check_grounding(answer, chunks):
        print("[hr_kb] ⚠ Grounding check failed — answer may contain hallucinated content")

    return {
        **state,
        "retrieved_chunks": chunks,
        "confidence_score": confidence,
        "should_escalate": False,
        "response": answer,
    }
