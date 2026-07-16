"""IT Knowledge Agent — RAG-grounded answers over the IT knowledge base.

Flow:
  1. retrieve(query, domain="it")          → chunks
  2. decide_escalation(query, chunks)       → (should_escalate, confidence)
  3a. Answerable: build prompt → call LLM → check_grounding → state["response"]
  3b. Escalate:   honest "I don't know" + should_escalate=True in state

Response enhancement (state["response"])
-----------------------------------------
For successful answers the response is enriched with:
  - LLM-generated grounded answer (formatted via GROUNDED_SYSTEM_INSTRUCTIONS:
    numbered steps, bold key terms, concise 3–6 step answers)
  - Source citations derived from chunk["source"] metadata
  - Confidence label (High / Medium) based on the top retrieval score

For escalations it includes the confidence score and a specific instruction for
how to trigger ticket creation on the next turn, so the user knows exactly what
to say rather than hitting a dead end.
"""

from __future__ import annotations

import os
from typing import List

from smartdesk._log import vprint
from smartdesk.agents._llm import call_llm
from smartdesk.guardrails.grounding import (
    GROUNDED_SYSTEM_INSTRUCTIONS,
    build_grounded_prompt,
    check_grounding,
)
from smartdesk.orchestrator.state import AgentState, RetrievedChunk
from smartdesk.rag.retriever import decide_escalation, retrieve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _source_label(source: str) -> str:
    """Convert a path like 'it_docs/vpn_setup_guide.json' → 'VPN Setup Guide'."""
    stem = os.path.splitext(os.path.basename(source))[0]
    return stem.replace("_", " ").title()


def _confidence_label(score: float) -> str:
    if score >= 0.7:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"


def _build_citation_footer(chunks: List[RetrievedChunk], confidence: float) -> str:
    """Return a markdown footer with unique source names and a confidence label.

    Example:
        ---
        *Sources: VPN Setup Guide, SSO Documentation · Confidence: High (87%)*
    """
    seen: dict[str, None] = {}
    for chunk in chunks:
        label = _source_label(chunk["source"])
        seen[label] = None
    sources = ", ".join(seen.keys())
    label = _confidence_label(confidence)
    return f"\n\n---\n*Sources: {sources} · Confidence: {label} ({confidence:.0%})*"


def _escalation_message(confidence: float) -> str:
    """Escalation response with confidence score and clear CTA to create a ticket."""
    return (
        f"I searched our IT knowledge base but couldn't find a confident answer "
        f"(relevance: {confidence:.0%}).\n\n"
        "This topic may not be fully documented yet, or it may need specialist input.\n\n"
        "To open a support request, just say:\n"
        "  **\"Create a ticket — [brief description of your issue]\"**\n"
        "and I'll take care of the rest."
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def it_knowledge_node(state: AgentState) -> AgentState:
    """LangGraph node: retrieve IT docs, generate grounded answer or escalate."""
    query = state.get("query", "")
    vprint(f"[it_kb] Retrieving for: {query!r}")

    # 1. Retrieve
    chunks = retrieve(query, domain="it")
    vprint(f"[it_kb] Retrieved {len(chunks)} chunks "
           f"(top score: {chunks[0]['score']:.3f})" if chunks else "[it_kb] No chunks found")

    # 2. Escalation decision
    should_escalate, confidence = decide_escalation(query, chunks)
    vprint(f"[it_kb] escalate={should_escalate}, confidence={confidence:.3f}")

    if should_escalate:
        return {
            **state,
            "retrieved_chunks": chunks,
            "confidence_score": confidence,
            "should_escalate": True,
            "response": _escalation_message(confidence),
        }

    # 3. Build grounded LLM answer
    user_prompt = build_grounded_prompt(query, chunks)
    answer = call_llm(system=GROUNDED_SYSTEM_INSTRUCTIONS, user=user_prompt, temperature=0.2)

    # Optional grounding check — warn but don't block
    if not check_grounding(answer, chunks):
        vprint("[it_kb] ⚠ Grounding check failed — answer may contain hallucinated content")

    # 4. Append source citations + confidence label  ← this is what enhances state["response"]
    enhanced_response = answer + _build_citation_footer(chunks, confidence)

    return {
        **state,
        "retrieved_chunks": chunks,
        "confidence_score": confidence,
        "should_escalate": False,
        "response": enhanced_response,
    }
