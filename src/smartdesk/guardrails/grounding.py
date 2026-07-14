"""Grounding helpers: keep answers tied to retrieved content only.

The brief is explicit: answers must come only from retrieved knowledge-base
content, never the model's training memory. If the KB doesn't have it, the
agent should say so honestly rather than guess.

TODO: implement both functions.
"""

from __future__ import annotations

from typing import List

from smartdesk.orchestrator.state import RetrievedChunk

GROUNDED_SYSTEM_INSTRUCTIONS = """\
You are a helpdesk assistant. Answer ONLY using the provided context.
If the context does not contain a confident answer, say you don't know -
do not guess or use outside knowledge.
"""


def build_grounded_prompt(query: str, retrieved_chunks: List[RetrievedChunk]) -> str:
    """Build the prompt sent to the LLM, embedding retrieved_chunks as
    context and instructing the model not to answer outside of it.

    TODO: implement. Keep GROUNDED_SYSTEM_INSTRUCTIONS (or your own
    rewording of it) as part of the prompt.
    """
    raise NotImplementedError("TODO: implement build_grounded_prompt")


def check_grounding(answer: str, retrieved_chunks: List[RetrievedChunk]) -> bool:
    """Sanity-check that `answer` doesn't appear to drift from
    retrieved_chunks (e.g. via keyword overlap, an NLI model, or asking the
    LLM to self-check). Return True if the answer looks grounded.

    TODO: implement. This is optional polish, but useful for the
    evaluation pipeline (evaluation/eval_pipeline.py).
    """
    raise NotImplementedError("TODO: implement check_grounding")
