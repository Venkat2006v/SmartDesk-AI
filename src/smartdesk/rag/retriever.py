"""Retrieval + the escalation decision.

This module contains the hardest design call in the whole project: when
should the system answer from the knowledge base vs. admit it doesn't know
and escalate (route to ticket creation)? See docs/ARCHITECTURE.md for the
three candidate strategies (threshold / LLM self-assessment / hybrid).

TODO: implement both functions below. They're kept separate from
vector_store.py so you can unit-test the escalation logic independently of
whatever vector DB you're using (see tests/test_retriever.py).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from smartdesk.orchestrator.state import RetrievedChunk


def retrieve(
    query: str, domain: str, top_k: Optional[int] = None
) -> List[RetrievedChunk]:
    """Retrieve the top_k chunks for `query`, scoped to `domain` ("it"|"hr").

    TODO: implement. Should embed `query` both ways — embeddings.embed_query
    for dense, embeddings.embed_sparse_query for sparse — then call
    vector_store.get_vector_store().similarity_search(...), which performs
    the Qdrant hybrid prefetch+fusion internally. Falls back to
    config.settings.retrieval_top_k if top_k is None.
    """
    raise NotImplementedError("TODO: implement retrieve")


def decide_escalation(
    query: str,
    retrieved: List[RetrievedChunk],
    confidence_threshold: Optional[float] = None,
) -> Tuple[bool, float]:
    """Decide whether to escalate (route to ticket creation) instead of
    answering from `retrieved`.

    Returns (should_escalate, confidence_score).

    TODO: implement one of:
    - Threshold: should_escalate = top score < confidence_threshold
      (falls back to config.settings.confidence_threshold if None).
    - LLM self-assessment: ask the LLM to rate its own confidence given
      the retrieved context, parse a score/boolean from the response.
    - Hybrid: combine both, optionally gate the final escalation on a
      HITL confirmation (see tools/hitl.py) before creating a ticket.

    Document which approach you pick in docs/DESIGN_DECISIONS.md.
    """
    raise NotImplementedError("TODO: implement decide_escalation")
