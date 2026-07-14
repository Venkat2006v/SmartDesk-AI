"""TODO: implement these once rag/retriever.py is implemented."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO: implement once retrieve() is implemented")
def test_retrieve_returns_top_k_chunks_for_domain() -> None:
    raise NotImplementedError


@pytest.mark.skip(reason="TODO: implement once decide_escalation() is implemented")
def test_decide_escalation_flags_low_confidence_queries() -> None:
    # Use a deliberately-uncovered topic (see data/README.md) and assert
    # should_escalate is True.
    raise NotImplementedError
