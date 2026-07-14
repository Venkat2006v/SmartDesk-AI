"""Retrieval + escalation decision.

Two public functions that the agent nodes call:

    chunks = retrieve(query, domain="it")
    should_escalate, confidence = decide_escalation(query, chunks)

Design choice for decide_escalation — THRESHOLD strategy (default):
  confidence  = top retrieval score (0–1, cosine similarity)
  should_escalate = confidence < confidence_threshold  OR  no chunks found

  Set CONFIDENCE_THRESHOLD in .env:
    0.0  → never escalate on score alone (only escalate if zero chunks returned)
    0.5  → escalate when top chunk score < 0.5 (a good production starting point)
    0.7  → aggressive escalation; only answer when highly confident

Alternative — LLM SELF-ASSESSMENT (commented out below):
  Ask the LLM "Can you answer this from the context? Yes/No".
  More accurate for semantic misses, but adds one extra LLM call per query.
  Document your final choice in docs/DESIGN_DECISIONS.md.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from smartdesk.config import settings
from smartdesk.orchestrator.state import RetrievedChunk
from smartdesk.rag.vector_store import VectorStore, get_vector_store

# ---------------------------------------------------------------------------
# Singleton vector store — loaded once per process, reused for all queries
# ---------------------------------------------------------------------------
_store: Optional[VectorStore] = None


def _get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = get_vector_store()
    return _store


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    domain: str,
    top_k: Optional[int] = None,
) -> List[RetrievedChunk]:
    """Run a hybrid dense+sparse search and return the top matching chunks.

    Args:
        query:  User query string.
        domain: "it" or "hr" — restricts retrieval to one knowledge domain.
        top_k:  Number of chunks to return. Defaults to settings.retrieval_top_k.

    Returns:
        List of RetrievedChunk dicts, highest score first.
        Returns an empty list if the KB has no match.
    """
    k = top_k if top_k is not None else settings.retrieval_top_k
    store = _get_store()
    return store.similarity_search(query=query, domain=domain, top_k=k)


def decide_escalation(
    query: str,
    retrieved: List[RetrievedChunk],
    confidence_threshold: Optional[float] = None,
) -> Tuple[bool, float]:
    """Decide whether the retrieved context is sufficient to answer the query.

    Strategy: THRESHOLD
      confidence  = top chunk score (0.0 if no chunks were returned)
      escalate    = no chunks found  OR  top score < threshold

    Args:
        query:                The original user query (used by LLM path if enabled).
        retrieved:            Chunks returned by retrieve().
        confidence_threshold: Override settings.confidence_threshold for this call.

    Returns:
        (should_escalate: bool, confidence_score: float)
        Callers should set AgentState["should_escalate"] and ["confidence_score"].
    """
    threshold = (
        confidence_threshold
        if confidence_threshold is not None
        else settings.confidence_threshold
    )

    # No chunks → definitely escalate
    if not retrieved:
        return True, 0.0

    top_score = retrieved[0]["score"]
    should_escalate = top_score < threshold

    return should_escalate, top_score

    # -----------------------------------------------------------------------
    # ALTERNATIVE: LLM self-assessment (uncomment to switch strategies)
    # More accurate for semantic edge cases; costs one extra LLM call per query.
    # -----------------------------------------------------------------------
    # from openai import OpenAI
    # context = "\n\n".join(chunk["text"] for chunk in retrieved[:3])
    # client = OpenAI(api_key=settings.llm_api_key)
    # resp = client.chat.completions.create(
    #     model=settings.llm_model or "gpt-4o-mini",
    #     messages=[
    #         {"role": "system", "content": (
    #             "You are an IT/HR helpdesk assistant. "
    #             "Answer ONLY with 'yes' or 'no'."
    #         )},
    #         {"role": "user", "content": (
    #             f"Context:\n{context}\n\n"
    #             f"Question: {query}\n\n"
    #             "Can you fully answer this question using ONLY the context above?"
    #         )},
    #     ],
    #     max_tokens=3,
    #     temperature=0,
    # )
    # answer = resp.choices[0].message.content.strip().lower()
    # can_answer = answer.startswith("yes")
    # return (not can_answer), top_score
