"""Evaluation harness for the SmartDesk AI RAG pipeline.

Status: pending implementation (next milestone after core agent completion).

Planned metrics:
  - Faithfulness       — does the answer only use retrieved context?
  - Context Precision  — are retrieved chunks relevant to the question?
  - Answer Relevance   — does the answer address the question?

Planned approach: Ragas (https://docs.ragas.io) or DeepEval.
The run_evaluation() signature below is the intended public interface — the
eval script and CI step will call this once it is implemented.

See docs/DESIGN_DECISIONS.md §10 for current status.
"""

from __future__ import annotations
from typing import Any, Dict, List


def run_evaluation(test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run test_cases through the orchestrator and return RAG quality metrics.

    Args:
        test_cases: List of dicts with keys:
            - "query":    str   — the user question
            - "answer":   str   — reference / expected answer
            - "contexts": list  — ground-truth relevant passages (optional)

    Returns:
        Dict with metric names as keys and float scores as values.
        e.g. {"faithfulness": 0.92, "context_precision": 0.87, "answer_relevance": 0.91}
    """
    raise NotImplementedError(
        "Evaluation pipeline not yet implemented. "
        "Planned: Ragas metrics (faithfulness, context_precision, answer_relevance). "
        "See docs/DESIGN_DECISIONS.md §10 for the implementation plan."
    )
