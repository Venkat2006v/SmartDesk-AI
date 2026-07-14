"""Evaluation harness (bonus item).

TODO: implement, if you want to go after the evaluation bonus points.
Options: LangSmith, Ragas, DeepEval, or a hand-rolled comparison against
expected answers/routes.

Suggested shape for a test case:
    {"query": "...", "expected_route": "it_kb", "expected_answer_contains": "..."}

run_evaluation should run each case through the orchestrator and report
per-case pass/fail plus aggregate metrics (e.g. routing accuracy, answer
relevance/groundedness if using Ragas/DeepEval).
"""

from __future__ import annotations

from typing import Any, Dict, List


def run_evaluation(test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run `test_cases` through the orchestrator and return a results dict.

    TODO: implement.
    """
    raise NotImplementedError("TODO: implement evaluation pipeline")
