"""Ticket Status Agent — looks up existing tickets for a user by email.

TODO: implement the flow:
1. Ensure state["email"] is present and valid (guardrails/validation.py
   is_valid_email). If missing, ask the user for it.
2. Call your TicketingClient.get_tickets_by_email(email).
3. Handle three distinct cases in the response:
   - Zero tickets: tell the user clearly, no tickets found.
   - One ticket: report its status directly.
   - Multiple tickets: summarize all of them (e.g. a short list with id +
     status), don't just report the first.
4. Set state["response"] accordingly.
"""

from __future__ import annotations

from smartdesk.orchestrator.state import AgentState


def ticket_status_node(state: AgentState) -> AgentState:
    """Orchestrator-facing node for ticket status lookups."""
    raise NotImplementedError("TODO: implement ticket status agent")
