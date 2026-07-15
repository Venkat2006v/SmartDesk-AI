"""Combined Knowledge Agent — synthesizes answers that span both IT and HR domains.

Activated when the supervisor classifies the query as `combined_kb`, i.e. the user
is asking about BOTH IT and HR topics in the same message (e.g. "what are the IT
setup steps and what HR enrollment deadlines do I need to know as a new hire?").

Flow:
  1. retrieve(query, domain="it")   → it_chunks
  2. retrieve(query, domain="hr")   → hr_chunks
  3. decide_escalation on each domain independently
  4a. Both answerable → merge chunks → call LLM with combined context → synthesize
  4b. One domain escalates → answer from the confident domain + note on the other
  4c. Both escalate → suggest creating a ticket

The synthesizer node runs both KB agents in a single Python call (sequential retrieval)
rather than two separate LangGraph nodes, keeping the graph topology simple.

Response enhancement (state["response"])
-----------------------------------------
Same footer pattern as the single-domain KB agents:
  *Sources: <IT source>, <HR source> · IT Confidence: High (85%) · HR Confidence: Medium (62%)*
"""

from __future__ import annotations

import os
from typing import List, Tuple

from smartdesk.agents._llm import call_llm
from smartdesk.guardrails.grounding import (
    build_grounded_prompt,
    check_grounding,
)
from smartdesk.orchestrator.state import AgentState, RetrievedChunk
from smartdesk.rag.retriever import decide_escalation, retrieve


# ---------------------------------------------------------------------------
# System instructions for synthesized cross-domain answers
# ---------------------------------------------------------------------------

_COMBINED_SYSTEM_INSTRUCTIONS = (
    "You are SmartDesk AI, a helpful IT and HR helpdesk assistant. "
    "The user's question spans both IT and HR topics. "
    "Answer BOTH parts using ONLY the context provided below. "
    "Structure your response in two clearly labelled sections:\n"
    "  **IT:** (answer the IT part)\n"
    "  **HR:** (answer the HR part)\n"
    "Use numbered steps for procedures. Bold key terms and UI labels. "
    "Keep each section concise — 3–5 steps or sentences. "
    "If the context does not cover one part, say so in that section rather than guessing. "
    "Do NOT speculate or use knowledge outside the provided context."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _source_label(source: str) -> str:
    stem = os.path.splitext(os.path.basename(source))[0]
    return stem.replace("_", " ").title()


def _confidence_label(score: float) -> str:
    if score >= 0.7:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"


def _build_combined_footer(
    it_chunks: List[RetrievedChunk],
    hr_chunks: List[RetrievedChunk],
    it_conf: float,
    hr_conf: float,
) -> str:
    """Build a footer citing sources and confidence from both domains."""
    seen: dict[str, None] = {}
    for chunk in it_chunks + hr_chunks:
        seen[_source_label(chunk["source"])] = None
    sources = ", ".join(seen.keys())
    return (
        f"\n\n---\n"
        f"*Sources: {sources} · "
        f"IT Confidence: {_confidence_label(it_conf)} ({it_conf:.0%}) · "
        f"HR Confidence: {_confidence_label(hr_conf)} ({hr_conf:.0%})*"
    )


def _escalation_note(domain: str, confidence: float) -> str:
    label = domain.upper()
    return (
        f"\n\n> **{label}:** I couldn't find a confident answer in the {label} knowledge base "
        f"(relevance: {confidence:.0%}). "
        f"To get help from the {label} team, say: "
        f"**\"Create a ticket — [describe your {label.lower()} issue]\"**"
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def combined_knowledge_node(state: AgentState) -> AgentState:
    """LangGraph node: retrieve from both IT and HR, synthesize a unified answer."""
    query = state.get("query", "")
    print(f"[combined_kb] Retrieving from both IT and HR for: {query!r}")

    # 1. Retrieve from both domains independently
    it_chunks = retrieve(query, domain="it")
    hr_chunks = retrieve(query, domain="hr")

    it_count = f"{len(it_chunks)} chunks (top: {it_chunks[0]['score']:.3f})" if it_chunks else "0 chunks"
    hr_count = f"{len(hr_chunks)} chunks (top: {hr_chunks[0]['score']:.3f})" if hr_chunks else "0 chunks"
    print(f"[combined_kb] IT: {it_count} | HR: {hr_count}")

    # 2. Escalation decision per domain
    it_escalate, it_conf = decide_escalation(query, it_chunks)
    hr_escalate, hr_conf = decide_escalation(query, hr_chunks)
    print(f"[combined_kb] IT escalate={it_escalate} ({it_conf:.3f}) | HR escalate={hr_escalate} ({hr_conf:.3f})")

    # 3a. Both domains have no confident answer → suggest ticket
    if it_escalate and hr_escalate:
        response = (
            f"I searched both the IT and HR knowledge bases but couldn't find confident answers "
            f"(IT relevance: {it_conf:.0%}, HR relevance: {hr_conf:.0%}).\n\n"
            "These topics may not yet be documented or may need specialist input.\n\n"
            "To open a support request, say:\n"
            "  **\"Create a ticket — [describe your question]\"**"
        )
        return {
            **state,
            "retrieved_chunks": it_chunks + hr_chunks,
            "confidence_score": max(it_conf, hr_conf),
            "should_escalate": True,
            "response": response,
        }

    # 3b. One domain escalates — answer from the confident side, note gap on the other
    if it_escalate or hr_escalate:
        # Use whichever domain has a confident answer
        answerable_chunks = hr_chunks if it_escalate else it_chunks
        user_prompt = build_grounded_prompt(query, answerable_chunks)
        answer = call_llm(
            system=_COMBINED_SYSTEM_INSTRUCTIONS,
            user=user_prompt,
            temperature=0.2,
        )
        # Append escalation note for the domain that failed
        if it_escalate:
            answer += _escalation_note("IT", it_conf)
        else:
            answer += _escalation_note("HR", hr_conf)

        answer += _build_combined_footer(it_chunks, hr_chunks, it_conf, hr_conf)
        return {
            **state,
            "retrieved_chunks": it_chunks + hr_chunks,
            "confidence_score": max(it_conf, hr_conf),
            "should_escalate": False,
            "response": answer,
        }

    # 3c. Both domains have confident answers → merge and synthesize
    all_chunks = it_chunks + hr_chunks
    user_prompt = build_grounded_prompt(query, all_chunks)
    answer = call_llm(
        system=_COMBINED_SYSTEM_INSTRUCTIONS,
        user=user_prompt,
        temperature=0.2,
    )

    if not check_grounding(answer, all_chunks):
        print("[combined_kb] ⚠ Grounding check failed — answer may contain hallucinated content")

    enhanced_response = answer + _build_combined_footer(it_chunks, hr_chunks, it_conf, hr_conf)

    return {
        **state,
        "retrieved_chunks": all_chunks,
        "confidence_score": (it_conf + hr_conf) / 2,
        "should_escalate": False,
        "response": enhanced_response,
    }
