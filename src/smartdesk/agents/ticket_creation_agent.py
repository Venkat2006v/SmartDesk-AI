"""Ticket Creation Agent — collects fields, confirms with the user (HITL),
then creates a ticket via a TicketingClient.

IMPORTANT (per the project brief): the agent must NOT create tickets
silently. Always confirm with the user before calling create_ticket.

TODO: implement the flow:
1. Ensure required fields are present: email, summary, description.
   - If state["email"] is missing/invalid (see guardrails/validation.py
     is_valid_email), ask the user for it before proceeding.
   - Derive a summary/description from state["query"] (and conversation
     context, if you're tracking any) or ask clarifying questions.
2. Call tools.hitl.confirm_action(summary) and only proceed if it returns
   True. If False, set a response acknowledging the cancellation and stop.
3. On confirmation, call your TicketingClient.create_ticket(...) (start
   with MockTicketingClient — see tools/ticketing/mock_client.py) and set
   state["ticket_id"] + a confirmation message in state["response"].
4. Handle ticketing client errors (see guardrails/validation.py
   with_retry) rather than letting exceptions propagate.
"""

from __future__ import annotations

from smartdesk.orchestrator.state import AgentState


def ticket_creation_node(state: AgentState) -> AgentState:
    """Orchestrator-facing node for ticket creation, with mandatory HITL."""
    raise NotImplementedError("TODO: implement ticket creation agent (with HITL)")
