"""Grounding helpers — keep LLM answers anchored to retrieved context.

build_grounded_prompt: formats chunks + query into an LLM prompt.
check_grounding:       lightweight sanity check that the answer
                       doesn't hallucinate beyond the retrieved text.
"""

from __future__ import annotations

from typing import List

from smartdesk.orchestrator.state import RetrievedChunk

GROUNDED_SYSTEM_INSTRUCTIONS = (
    "You are SmartDesk AI, a helpful IT and HR helpdesk assistant. "
    "Answer the user's question using ONLY the context provided below. "
    "Be concise and direct. "
    "If the context does not contain enough information to answer confidently, "
    "say: \"I don't have enough information in our knowledge base to answer that. "
    "I recommend creating a support ticket so the team can assist you.\""
    " Do NOT guess, speculate, or use knowledge outside the provided context."
)


def build_grounded_prompt(query: str, retrieved_chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks + user query into a single user-turn message.

    The caller passes this as the `user` argument to call_llm().
    The system prompt (GROUNDED_SYSTEM_INSTRUCTIONS) is passed separately.

    Format:
        --- CONTEXT ---
        [Source: <source> | Score: <score>]
        <text>
        ...
        --- END CONTEXT ---

        Question: <query>
    """
    if not retrieved_chunks:
        context_block = "(No relevant documents found in the knowledge base.)"
    else:
        lines = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            lines.append(
                f"[{i}] Source: {chunk['source']} | Relevance: {chunk['score']:.2f}\n"
                f"{chunk['text'].strip()}"
            )
        context_block = "\n\n".join(lines)

    return (
        f"--- CONTEXT ---\n"
        f"{context_block}\n"
        f"--- END CONTEXT ---\n\n"
        f"Question: {query}"
    )


def check_grounding(answer: str, retrieved_chunks: List[RetrievedChunk]) -> bool:
    """Lightweight keyword-overlap grounding check.

    Strategy: extract meaningful content words from the retrieved chunks,
    then verify the answer shares at least some overlap. Returns False
    only when the answer is suspiciously long with zero overlap (likely
    hallucination); returns True in all other cases to avoid false positives.

    For production, replace with an NLI model or LLM self-check call.
    See docs/DESIGN_DECISIONS.md for upgrade options.
    """
    if not retrieved_chunks:
        # No context was provided — trust the agent's "I don't know" response
        return True

    # Combine all chunk text and normalize
    corpus = " ".join(chunk["text"] for chunk in retrieved_chunks).lower()
    corpus_words = set(_tokenize(corpus))

    answer_words = set(_tokenize(answer.lower()))

    # Remove very common stop words from both sets
    corpus_meaningful = corpus_words - _STOP_WORDS
    answer_meaningful = answer_words - _STOP_WORDS

    if not answer_meaningful:
        return True  # Too short to evaluate

    overlap = corpus_meaningful & answer_meaningful
    overlap_ratio = len(overlap) / len(answer_meaningful)

    # Flag as ungrounded only when overlap is very low AND answer is long
    # (conservative: avoids false positives on valid paraphrasing)
    is_long_answer = len(answer_meaningful) > 20
    return not (is_long_answer and overlap_ratio < 0.10)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Split text into alpha-only tokens of length >= 3."""
    import re
    return re.findall(r"[a-z]{3,}", text)


_STOP_WORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "with",
    "this", "that", "have", "from", "they", "will", "your", "our", "has",
    "was", "been", "its", "also", "more", "than", "then", "when", "into",
    "their", "there", "about", "which", "would", "could", "should", "please",
    "use", "may", "any", "who", "how", "what", "why", "where", "each",
    "per", "via", "using", "used", "provided", "available", "following",
}
