"""TODO: implement these once agents/ticket_creation_agent.py is implemented."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO: implement once ticket_creation_node() is implemented")
def test_ticket_is_not_created_without_hitl_confirmation(mock_ticketing_client) -> None:
    # Critical case: simulate the user declining confirmation and assert
    # mock_ticketing_client has zero tickets afterward.
    raise NotImplementedError


@pytest.mark.skip(reason="TODO: implement once ticket_creation_node() is implemented")
def test_ticket_is_created_after_hitl_confirmation(mock_ticketing_client) -> None:
    raise NotImplementedError
