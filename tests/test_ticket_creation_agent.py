"""Tests for agents/ticket_creation_agent.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from smartdesk.orchestrator.state import AgentState
from smartdesk.tools.ticketing.base import Ticket
from smartdesk.tools.ticketing.mock_client import MockTicketingClient


def _mock_ticket(tid: str = "MOCK-1", email: str = "venkat@company.com") -> Ticket:
    return Ticket(
        id=tid,
        email=email,
        summary="Laptop can't connect to VPN",
        description="User's laptop won't connect to the corporate VPN.",
        status="open",
        url=f"https://mock-ticketing.local/tickets/{tid}",
    )


class TestTicketCreationNode:
    def test_ticket_not_created_without_hitl_confirmation(self, mock_ticketing_client) -> None:
        """HITL denial → no ticket created, ticket_id absent, response mentions cancel."""
        state = AgentState(
            query="My laptop won't connect to VPN. Please create a ticket.",
            email="venkat@company.com",
        )

        with (
            patch(
                "smartdesk.agents.ticket_creation_agent.call_llm",
                return_value='{"summary": "VPN issue", "description": "Cannot connect to VPN."}',
            ),
            patch("smartdesk.agents.ticket_creation_agent.confirm_action", return_value=False),
            patch(
                "smartdesk.agents.ticket_creation_agent.get_ticketing_client",
                return_value=mock_ticketing_client,
            ),
        ):
            from smartdesk.agents.ticket_creation_agent import ticket_creation_node
            result = ticket_creation_node(state)

        assert result.get("ticket_id") is None
        assert result.get("ticket_confirmed") is False
        assert "cancel" in result.get("response", "").lower()
        # Sanity: mock client was never asked to create anything
        assert len(mock_ticketing_client._tickets) == 0

    def test_ticket_created_after_hitl_confirmation(self, mock_ticketing_client) -> None:
        """HITL approval → ticket_id is set, response contains the ticket ID."""
        state = AgentState(
            query="My laptop won't connect to VPN. Please create a ticket.",
            email="venkat@company.com",
        )

        with (
            patch(
                "smartdesk.agents.ticket_creation_agent.call_llm",
                return_value='{"summary": "VPN issue", "description": "Cannot connect to VPN."}',
            ),
            patch("smartdesk.agents.ticket_creation_agent.confirm_action", return_value=True),
            patch(
                "smartdesk.agents.ticket_creation_agent.get_ticketing_client",
                return_value=mock_ticketing_client,
            ),
        ):
            from smartdesk.agents.ticket_creation_agent import ticket_creation_node
            result = ticket_creation_node(state)

        assert result.get("ticket_confirmed") is True
        ticket_id = result.get("ticket_id")
        assert ticket_id is not None
        assert ticket_id in result.get("response", "")
        assert len(mock_ticketing_client._tickets) == 1

    def test_invalid_email_returns_email_prompt(self) -> None:
        """Invalid email → no HITL gate triggered, response asks for valid email."""
        state = AgentState(query="Create a ticket for VPN issues", email="not-an-email")

        with (
            patch(
                "smartdesk.agents.ticket_creation_agent.call_llm",
                side_effect=[
                    '{"summary": "VPN issue", "description": "Details."}',
                    "",  # email extraction returns nothing
                ],
            ),
            patch("smartdesk.agents.ticket_creation_agent.confirm_action") as mock_confirm,
        ):
            from smartdesk.agents.ticket_creation_agent import ticket_creation_node
            result = ticket_creation_node(state)

        mock_confirm.assert_not_called()
        assert result.get("ticket_id") is None
        assert "email" in result.get("response", "").lower()

    def test_empty_query_returns_detail_prompt(self) -> None:
        """No extractable summary → response asks for more detail (no HITL)."""
        state = AgentState(query="help", email="venkat@company.com")

        with (
            patch(
                "smartdesk.agents.ticket_creation_agent.call_llm",
                return_value='{"summary": "", "description": ""}',
            ),
            patch("smartdesk.agents.ticket_creation_agent.confirm_action") as mock_confirm,
        ):
            from smartdesk.agents.ticket_creation_agent import ticket_creation_node
            result = ticket_creation_node(state)

        mock_confirm.assert_not_called()
        assert result.get("ticket_id") is None
        assert "detail" in result.get("response", "").lower()
