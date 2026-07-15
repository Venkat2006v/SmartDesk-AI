"""Ticket Status Agent — look up tickets by email address.

Flow:
  1. Get email from state["email"] or extract from query via LLM.
  2. Validate the email with is_valid_email().
  3. Call ticketing_client.get_tickets_by_email(email).
  4. Format a clear response for 0 / 1 / many tickets.
"""

from __future__ import annotations

from smartdesk.agents._llm import call_llm
from smartdesk.guardrails.validation import is_valid_email
from smartdesk.orchestrator.state import AgentState
from smartdesk.tools.ticketing import get_ticketing_client

_EXTRACT_EMAIL_SYSTEM = """You are a helpdesk assistant. Extract the email address from the user's message.
Respond with ONLY the email address, nothing else.
If no email is found, respond with an empty string "".
"""


_CLOSURE_KEYWORDS = {"close", "closed", "cancel", "cancelled", "resolve", "resolved", "reopen", "update"}


def _is_closure_request(query: str) -> bool:
    """Return True if the query is asking to close/cancel/resolve a ticket."""
    words = set(query.lower().split())
    return bool(words & _CLOSURE_KEYWORDS)


def ticket_status_node(state: AgentState) -> AgentState:
    """LangGraph node: look up tickets and format a response."""
    query = state.get("query", "")

    # ── Closure / update requests — not supported via this interface ───────
    if _is_closure_request(query):
        return {
            **state,
            "response": (
                "Closing, cancelling, or updating tickets isn't something I can do directly — "
                "that requires action from your IT support team or manager.\n\n"
                "Here's what you can do:\n"
                "1. **Contact your IT support team** and reference the ticket ID.\n"
                "2. **Reply to the ticket email notification** you received when it was created.\n"
                "3. **Visit the Jira portal** linked in your ticket confirmation to update it yourself.\n\n"
                "To look up your open tickets and get the ticket ID, say: "
                "**\"What tickets do I have open?\"**"
            ),
        }

    # ── Step 1: Resolve email ──────────────────────────────────────────────
    email = state.get("email") or ""

    if not is_valid_email(email):
        # Try to extract email from the query itself
        email = _extract_email_from_query(query)

    if not is_valid_email(email):
        return {
            **state,
            "pending_action": "ticket_status",
            "response": (
                "I need your email address to look up your tickets. "
                "Please provide it and I'll check right away."
            ),
        }

    # ── Step 2: Fetch tickets ──────────────────────────────────────────────
    client = get_ticketing_client()

    try:
        tickets = client.get_tickets_by_email(email)
    except Exception as exc:
        return {
            **state,
            "email": email,
            "response": (
                f"I ran into an error looking up tickets for {email}: {exc}. "
                "Please try again or contact the helpdesk directly."
            ),
        }

    # ── Step 3: Format response ────────────────────────────────────────────
    response = _format_tickets(email, tickets)
    print(f"[ticket_status] Found {len(tickets)} ticket(s) for {email!r}")

    return {**state, "email": email, "response": response}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_tickets(email: str, tickets: list) -> str:
    """Build a human-readable ticket list."""
    if not tickets:
        return (
            f"I found no open tickets for **{email}**. "
            "If you expected to see a ticket here, double-check your email address "
            "or create a new ticket."
        )

    if len(tickets) == 1:
        t = tickets[0]
        return (
            f"You have **1 ticket** on file for {email}:\n\n"
            f"  • **{t['id']}** — {t['summary']}\n"
            f"    Status : {t['status']}\n"
            f"    URL    : {t['url']}"
        )

    lines = [f"You have **{len(tickets)} tickets** on file for {email}:\n"]
    for t in tickets:
        lines.append(
            f"  • **{t['id']}** [{t['status']}] — {t['summary']}\n"
            f"    {t['url']}"
        )
    return "\n".join(lines)


def _extract_email_from_query(query: str) -> str:
    """Use LLM to pull an email address out of the query text."""
    try:
        result = call_llm(system=_EXTRACT_EMAIL_SYSTEM, user=query, temperature=0.0)
        candidate = result.strip()
        return candidate if is_valid_email(candidate) else ""
    except Exception:
        return ""
