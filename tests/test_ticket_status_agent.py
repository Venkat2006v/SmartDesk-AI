"""Tests for agents/ticket_status_agent.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from smartdesk.orchestrator.state import AgentState
from smartdesk.tools.ticketing.base import Ticket
from smartdesk.tools.ticketing.mock_client import MockTicketingClient


def _make_ticket(tid: str, email: str, summary: str) -> Ticket:
    return Ticket(
        id=tid,
        email=email,
        summary=summary,
        description=f"Description for {summary}",
        status="open",
        url=f"https://mock-ticketing.local/tickets/{tid}",
    )


class TestTicketStatusNode:
    def test_ticket_status_handles_zero_tickets(self, mock_ticketing_client) -> None:
        """No tickets for email → response says no tickets found."""
        state = AgentState(query="What tickets do I have?", email="venkat@company.com")

        with patch(
            "smartdesk.agents.ticket_status_agent.get_ticketing_client",
            return_value=mock_ticketing_client,
        ):
            from smartdesk.agents.ticket_status_agent import ticket_status_node
            result = ticket_status_node(state)

        response = result.get("response", "")
        assert "no" in response.lower() or "0" in response
        assert "venkat@company.com" in response

    def test_ticket_status_handles_single_ticket(self, mock_ticketing_client) -> None:
        """One ticket → response mentions the ticket ID."""
        mock_ticketing_client._tickets["MOCK-1"] = _make_ticket(
            "MOCK-1", "venkat@company.com", "VPN not working"
        )
        state = AgentState(query="Show my tickets", email="venkat@company.com")

        with patch(
            "smartdesk.agents.ticket_status_agent.get_ticketing_client",
            return_value=mock_ticketing_client,
        ):
            from smartdesk.agents.ticket_status_agent import ticket_status_node
            result = ticket_status_node(state)

        response = result.get("response", "")
        assert "MOCK-1" in response
        assert "VPN not working" in response

    def test_ticket_status_handles_multiple_tickets(self, mock_ticketing_client) -> None:
        """Multiple tickets → all IDs appear in response."""
        for i, summary in enumerate(["VPN issue", "MFA broken", "Printer offline"], start=1):
            mock_ticketing_client._tickets[f"MOCK-{i}"] = _make_ticket(
                f"MOCK-{i}", "venkat@company.com", summary
            )

        state = AgentState(query="List all my tickets", email="venkat@company.com")

        with patch(
            "smartdesk.agents.ticket_status_agent.get_ticketing_client",
            return_value=mock_ticketing_client,
        ):
            from smartdesk.agents.ticket_status_agent import ticket_status_node
            result = ticket_status_node(state)

        response = result.get("response", "")
        assert "MOCK-1" in response
        assert "MOCK-2" in response
        assert "MOCK-3" in response

    def test_invalid_email_returns_prompt(self) -> None:
        """No valid email in state or query → response asks for email."""
        state = AgentState(query="Show my tickets", email="")

        with patch(
            "smartdesk.agents.ticket_status_agent.call_llm",
            return_value="",  # email extraction returns nothing
        ):
            from smartdesk.agents.ticket_status_agent import ticket_status_node
            result = ticket_status_node(state)

        assert "email" in result.get("response", "").lower()

    def test_email_resolved_from_state(self, mock_ticketing_client) -> None:
        """Email already in state["email"] — no LLM extraction needed."""
        mock_ticketing_client._tickets["MOCK-5"] = _make_ticket(
            "MOCK-5", "alice@corp.com", "Keyboard issue"
        )
        state = AgentState(query="check my tickets", email="alice@corp.com")

        with patch(
            "smartdesk.agents.ticket_status_agent.get_ticketing_client",
            return_value=mock_ticketing_client,
        ):
            from smartdesk.agents.ticket_status_agent import ticket_status_node
            result = ticket_status_node(state)

        assert result.get("email") == "alice@corp.com"
        assert "MOCK-5" in result.get("response", "")
