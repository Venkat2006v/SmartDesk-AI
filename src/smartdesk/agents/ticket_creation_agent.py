"""Ticket Creation Agent — collect fields, HITL confirm, create ticket.

MANDATORY: Never create a ticket without explicit user confirmation (HITL).
This is a hard requirement and is enforced by confirm_action().

Flow:
  1. Extract ticket fields (summary, description, category, priority) from
     the query via LLM if not already in state.
  2. Get / validate email from state["email"] or extract from query.
  3. HITL gate: confirm_action(summary, description) — abort if denied.
  4. ticketing_client.create_ticket(email, summary, description)
  5. Set state["ticket_id"] and state["response"].
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ValidationError

from smartdesk.agents._llm import call_llm, call_llm_json
from smartdesk.guardrails.validation import is_valid_email, with_retry
from smartdesk.orchestrator.state import AgentState
from smartdesk.tools.hitl import confirm_action
from smartdesk.tools.ticketing import get_ticketing_client


# ---------------------------------------------------------------------------
# Structured output schema (Pydantic)
# ---------------------------------------------------------------------------

class TicketFields(BaseModel):
    """Schema for LLM-extracted ticket fields.

    Using Pydantic here gives us:
    - Type coercion (LLM returns string priority → validated against Literal)
    - Clear error messages if the LLM returns an unexpected value
    - Zero manual field validation in _extract_ticket_fields()
    """
    summary: str
    description: str
    category: Literal[
        "IT Support", "HR Request", "Access Request", "Hardware", "Software", "Other"
    ] = "IT Support"
    priority: Literal["Low", "Medium", "High", "Critical"] = "Medium"

# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = """You are a helpdesk assistant that extracts ticket fields from user messages.
Given the user's message, extract:
  - summary:     A concise one-line title for the ticket (max 80 chars)
  - description: A full description of the issue or request
  - category:    One of: "IT Support" | "HR Request" | "Access Request" | "Hardware" | "Software" | "Other"
  - priority:    One of: "Low" | "Medium" | "High" | "Critical"
                 (Critical = system down / blocking work; High = significant impact;
                  Medium = inconvenient but workaround exists; Low = minor / informational)

Respond with a JSON object containing exactly these four keys:
  "summary", "description", "category", "priority"

If a field cannot be determined from the message, use sensible defaults:
  category → "IT Support", priority → "Medium"
"""

_EXTRACT_EMAIL_SYSTEM = """You are a helpdesk assistant. Extract the email address from the user's message.
Respond with ONLY the email address, nothing else.
If no email is found, respond with an empty string "".
"""


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def ticket_creation_node(state: AgentState) -> AgentState:
    """LangGraph node: extract fields → HITL confirm → create ticket."""
    query = state.get("query", "")

    # ── Step 1: Extract summary + description + category + priority ─────────
    summary = state.get("ticket_summary") or ""
    description = state.get("ticket_description") or ""
    category = ""
    priority = ""

    if not summary or not description:
        summary, description, category, priority = _extract_ticket_fields(query)

    if not summary:
        return {
            **state,
            "response": (
                "I need a bit more detail to create a ticket. "
                "Could you describe the issue or request you'd like to raise?"
            ),
        }

    # ── Step 2: Get + validate email ──────────────────────────────────────
    email = state.get("email") or ""

    if not email:
        email = _extract_email_from_query(query)

    if not is_valid_email(email):
        # Persist the already-extracted ticket fields so that on the next turn
        # (when the user provides their email) we don't re-extract from scratch
        # and lose the summary/description.
        return {
            **state,
            "ticket_summary": summary,
            "ticket_description": description,
            "response": (
                "To create this ticket I need your email address. "
                "Please reply with your email and I'll take care of the rest.\n\n"
                f"*(Ticket queued: {summary})*"
            ),
        }

    # ── Step 3: HITL confirmation ──────────────────────────────────────────
    hitl_summary = f"Create ticket: {summary}"
    hitl_detail = (
        f"{description}\n\n"
        f"Category : {category or 'IT Support'}\n"
        f"Priority : {priority or 'Medium'}"
    )
    print(f"[ticket_creation] Requesting HITL confirmation for: {summary!r}")
    confirmed = confirm_action(summary=hitl_summary, description=hitl_detail)

    if not confirmed:
        return {
            **state,
            "ticket_confirmed": False,
            "response": "Ticket creation cancelled. Let me know if you need anything else.",
        }

    # ── Step 4: Create ticket (with retry) ────────────────────────────────
    client = get_ticketing_client()

    # Enrich description with category + priority for the ticketing system
    full_description = (
        f"{description or summary}\n\n"
        f"Category: {category or 'IT Support'}\n"
        f"Priority: {priority or 'Medium'}"
    )

    @with_retry(max_attempts=3, backoff_seconds=1.0)
    def _create():
        return client.create_ticket(
            email=email,
            summary=summary,
            description=full_description,
        )

    try:
        ticket = _create()
        print(f"[ticket_creation] Created ticket {ticket['id']!r}")
    except Exception as exc:
        return {
            **state,
            "ticket_confirmed": True,
            "response": (
                f"I confirmed the ticket but ran into an error creating it: {exc}. "
                "Please try again or contact the helpdesk directly."
            ),
        }

    # ── Step 5: Return success state ──────────────────────────────────────
    return {
        **state,
        "ticket_confirmed": True,
        "ticket_id": ticket["id"],
        "ticket_summary": summary,
        "ticket_description": full_description,
        "email": email,
        "response": (
            f"✓ Ticket **{ticket['id']}** created successfully!\n"
            f"  Summary  : {ticket['summary']}\n"
            f"  Category : {category or 'IT Support'}\n"
            f"  Priority : {priority or 'Medium'}\n"
            f"  Status   : {ticket['status']}\n"
            f"  URL      : {ticket['url']}\n"
            f"A confirmation has been logged to {ticket['email']}."
        ),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_ticket_fields(query: str) -> tuple[str, str, str, str]:
    """Use LLM to extract summary, description, category, and priority.

    Uses call_llm_json() which:
    - For OpenAI: sets response_format={"type": "json_object"} — the API
      guarantees valid JSON, eliminating markdown stripping and json.loads errors.
    - For Anthropic: falls back to prompt-based JSON + safe parsing.

    The parsed dict is then validated against the TicketFields Pydantic schema,
    which enforces the exact category and priority enum values and provides
    clear error messages for unexpected LLM output.
    """
    try:
        data = call_llm_json(system=_EXTRACT_SYSTEM, user=query)
        fields = TicketFields(**data)
        return (
            fields.summary.strip(),
            fields.description.strip(),
            fields.category,
            fields.priority,
        )
    except ValidationError as exc:
        # Pydantic caught an invalid category or priority — use defaults
        print(f"[ticket_creation] Schema validation failed: {exc!r} — applying defaults")
        # Still extract what we can from the raw dict
        summary = str(data.get("summary", query[:80])).strip()  # type: ignore[possibly-undefined]
        description = str(data.get("description", query)).strip()
        return summary, description, "IT Support", "Medium"
    except Exception as exc:
        print(f"[ticket_creation] Field extraction failed: {exc!r} — using query as summary")
        return query[:80].strip(), query.strip(), "IT Support", "Medium"


def _extract_email_from_query(query: str) -> str:
    """Use LLM to pull an email address out of the query text.

    Returns the email only if it passes is_valid_email(); otherwise returns "".
    This guards against the LLM returning prose like "No email found" instead
    of the empty string instructed in the system prompt.
    """
    try:
        result = call_llm(system=_EXTRACT_EMAIL_SYSTEM, user=query, temperature=0.0)
        candidate = result.strip()
        return candidate if is_valid_email(candidate) else ""
    except Exception:
        return ""
