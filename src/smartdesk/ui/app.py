"""Gradio chat UI — bonus deliverable.

Run with:
    pip install gradio
    python src/smartdesk/ui/app.py

OR after `pip install -e .`:
    python -m smartdesk.ui.app

HITL in Gradio:
  Because Gradio is async and can't block on input(), HITL confirmation
  is handled as a multi-turn exchange:
    1. ticket_creation_node detects we are in UI mode (HITL_MODE=ui)
       and sets state["response"] = "Shall I create this ticket? (yes/no)"
       WITHOUT calling confirm_action().
    2. The user replies "yes" or "no".
    3. The next invocation sees state["ticket_confirmed"]=True/False
       (set by the UI layer below) and proceeds accordingly.

  This is implemented via gr.State() tracking a pending_ticket dict.
"""

from __future__ import annotations

import os

os.environ.setdefault("HITL_MODE", "ui")  # switch all agents to UI mode

try:
    import gradio as gr
except ImportError:
    raise SystemExit(
        "Gradio is not installed. Run: pip install gradio\n"
        "Or install the full extras: pip install -e '.[ui]'"
    )

from smartdesk.orchestrator.graph import build_orchestrator, run_once

# Build the graph once at module load
print("[ui] Building agent graph...")
_graph = build_orchestrator()
print("[ui] Ready.")


# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------

def _respond(
    message: str,
    history: list,
    session_email: str,
    pending_ticket: dict | None,
    session_ticket_summary: str | None,
    session_ticket_description: str | None,
    session_pending_action: str | None,
    session_last_kb_query: str | None,
) -> tuple[str, list, str, dict | None, str | None, str | None, str | None, str | None]:
    """Process one user turn.

    Returns:
        (cleared_input, history, email, pending_ticket,
         ticket_summary, ticket_description, pending_action, last_kb_query)
    """
    message = message.strip()
    if not message:
        return "", history, session_email, pending_ticket, session_ticket_summary, session_ticket_description, session_pending_action, session_last_kb_query

    # ── HITL confirmation turn ──────────────────────────────────────────────
    if pending_ticket:
        answer = message.lower()
        if answer in {"yes", "y", "confirm", "ok", "yeah", "yep"}:
            from smartdesk.tools.ticketing import get_ticketing_client
            from smartdesk.guardrails.validation import with_retry

            email = pending_ticket.get("email", session_email or "")
            summary = pending_ticket.get("summary", "")
            description = pending_ticket.get("description", "")

            client = get_ticketing_client()
            try:
                @with_retry(max_attempts=3, backoff_seconds=1.0)
                def _create():
                    return client.create_ticket(email=email, summary=summary, description=description)

                ticket = _create()
                bot_msg = (
                    f"✓ Ticket **{ticket['id']}** created!\n"
                    f"  Summary: {ticket['summary']}\n"
                    f"  Status : {ticket['status']}\n"
                    f"  URL    : {ticket['url']}"
                )
            except Exception as exc:
                bot_msg = f"Failed to create ticket: {exc}. Please try again."

            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": bot_msg})
            # Clear ticket fields after creation
            return "", history, email or session_email, None, None, None, None, session_last_kb_query

        else:
            bot_msg = "Ticket creation cancelled. Let me know if you need anything else."
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": bot_msg})
            return "", history, session_email, None, None, None, None, session_last_kb_query

    # ── Normal query turn ────────────────────────────────────────────────────
    # Enrich vague follow-ups and ticket requests with conversation context.
    _FOLLOWUP_WORDS = {"still", "not working", "didn't work", "doesn't work",
                       "same issue", "same problem", "not fixed", "still failing"}
    _VAGUE_TICKET   = {"this issue", "the issue", "the problem", "this problem",
                       "the above", "for this", "about this"}
    _TICKET_WORDS   = {"ticket", "create", "raise", "open", "log"}
    msg_lower = message.lower()

    query = message
    if session_last_kb_query:
        is_vague_followup = any(w in msg_lower for w in _FOLLOWUP_WORDS)
        is_vague_ticket   = (any(ref in msg_lower for ref in _VAGUE_TICKET)
                             and any(kw in msg_lower for kw in _TICKET_WORDS))
        if is_vague_followup or is_vague_ticket:
            query = f"{message}\n\n[Context: the user was previously asking about: {session_last_kb_query}]"

    initial_state = {
        "query": query,
        "email": session_email or None,
        "ticket_summary": session_ticket_summary or None,
        "ticket_description": session_ticket_description or None,
        "pending_action": session_pending_action or None,
    }

    try:
        result = run_once(_graph, initial_state)
    except Exception as exc:
        bot_msg = f"Sorry, I ran into an error: {exc}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": bot_msg})
        return "", history, session_email, None, session_ticket_summary, session_ticket_description, session_pending_action

    # Cache session state from result
    resolved_email = result.get("email") or session_email
    new_ticket_summary = result.get("ticket_summary") or session_ticket_summary
    new_ticket_description = result.get("ticket_description") or session_ticket_description
    new_pending_action = result.get("pending_action") or None

    # Track last successfully answered KB query for follow-up context injection
    new_last_kb_query = session_last_kb_query
    route = result.get("route", "")
    if route in ("it_kb", "hr_kb", "combined_kb") and not result.get("should_escalate"):
        new_last_kb_query = message  # original message, not enriched query

    # Clear ticket fields after successful creation
    if result.get("ticket_id"):
        new_ticket_summary = None
        new_ticket_description = None

    response = result.get("response", "[no response]")

    # ── Detect HITL-pending state from ticket creation ───────────────────────
    # Only show confirmation prompt when we have BOTH a ticket summary AND an email.
    # If email is missing, the agent already asked for it — wait for the next turn.
    new_pending: dict | None = None
    if (
        result.get("route") == "create_ticket"
        and not result.get("ticket_id")
        and new_ticket_summary
        and resolved_email  # must have email before offering yes/no
    ):
        summary = new_ticket_summary
        description = new_ticket_description or ""
        response += (
            f"\n\n📋 **Proposed ticket:**\n"
            f"  Summary     : {summary}\n"
            f"  For email   : {resolved_email}\n"
            f"  Description : {description[:120]}{'...' if len(description) > 120 else ''}\n\n"
            f"Shall I create this ticket? Type **yes** to confirm or **no** to cancel."
        )
        new_pending = {"summary": summary, "description": description, "email": resolved_email}
        # Don't need to persist these in state — they're in pending_ticket now
        new_ticket_summary = None
        new_ticket_description = None

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return "", history, resolved_email, new_pending, new_ticket_summary, new_ticket_description, new_pending_action, new_last_kb_query


# ---------------------------------------------------------------------------
# Gradio UI layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="SmartDesk AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🖥️ SmartDesk AI\n"
        "**IT & HR Helpdesk Assistant** — Ask about VPN, MFA, PTO, benefits, "
        "or create/check support tickets."
    )

    chatbot = gr.Chatbot(label="SmartDesk AI", height=480, type="messages")
    msg_box = gr.Textbox(
        label="Your question",
        placeholder="e.g. How do I set up MFA?",
        show_label=False,
    )
    send_btn = gr.Button("Send", variant="primary")

    # Hidden state — mirrors CLI session vars
    email_state = gr.State("")
    pending_state = gr.State(None)
    ticket_summary_state = gr.State(None)
    ticket_description_state = gr.State(None)
    pending_action_state = gr.State(None)
    last_kb_query_state = gr.State(None)  # last successfully answered KB query (for follow-up context)

    _inputs  = [msg_box, chatbot, email_state, pending_state, ticket_summary_state, ticket_description_state, pending_action_state, last_kb_query_state]
    _outputs = [msg_box, chatbot, email_state, pending_state, ticket_summary_state, ticket_description_state, pending_action_state, last_kb_query_state]

    send_btn.click(_respond, inputs=_inputs, outputs=_outputs)
    msg_box.submit(_respond, inputs=_inputs, outputs=_outputs)

    gr.Examples(
        examples=[
            "How do I connect to the corporate VPN?",
            "Walk me through setting up MFA with TOTP.",
            "How many PTO days do I get per year?",
            "When is benefits open enrollment?",
            "Create a ticket — my laptop won't connect to Wi-Fi.",
            "What tickets do I have open?",
        ],
        inputs=msg_box,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        share=False,
    )
