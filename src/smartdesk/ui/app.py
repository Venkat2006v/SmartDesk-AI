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
) -> tuple[str, list, str, dict | None]:
    """Process one user turn and return (response, updated_history, email, pending_ticket)."""
    message = message.strip()
    if not message:
        return "", history, session_email, pending_ticket

    # ── HITL confirmation turn ──────────────────────────────────────────────
    if pending_ticket:
        answer = message.lower()
        if answer in {"yes", "y", "confirm", "ok", "yeah", "yep"}:
            # Replay ticket creation with ticket_confirmed=True
            from smartdesk.tools.ticketing import get_ticketing_client
            from smartdesk.guardrails.validation import is_valid_email, with_retry

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
            return "", history, email or session_email, None  # clear pending

        else:
            # User declined
            bot_msg = "Ticket creation cancelled. Let me know if you need anything else."
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": bot_msg})
            return "", history, session_email, None

    # ── Normal query turn ────────────────────────────────────────────────────
    initial_state = {
        "query": message,
        "email": session_email or None,
    }

    try:
        result = run_once(_graph, initial_state)
    except Exception as exc:
        bot_msg = f"Sorry, I ran into an error: {exc}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": bot_msg})
        return "", history, session_email, None

    # Update session email if resolved this turn
    resolved_email = result.get("email") or session_email

    response = result.get("response", "[no response]")

    # ── Detect HITL-pending state from ticket creation ───────────────────────
    # In UI mode, ticket_creation_node returns a confirmation-request message.
    # Detect it by checking if route was create_ticket and no ticket_id yet.
    new_pending: dict | None = None
    if (
        result.get("route") == "create_ticket"
        and not result.get("ticket_id")
        and result.get("ticket_summary")
    ):
        # Append a HITL confirmation prompt to the response
        summary = result.get("ticket_summary", "")
        description = result.get("ticket_description", "")
        email_for_ticket = result.get("email") or resolved_email or ""
        response += (
            f"\n\n📋 **Proposed ticket:**\n"
            f"  Summary     : {summary}\n"
            f"  For email   : {email_for_ticket}\n"
            f"  Description : {description[:120]}{'...' if len(description) > 120 else ''}\n\n"
            f"Shall I create this ticket? Type **yes** to confirm or **no** to cancel."
        )
        new_pending = {"summary": summary, "description": description, "email": email_for_ticket}

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return "", history, resolved_email, new_pending


# ---------------------------------------------------------------------------
# Gradio UI layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="SmartDesk AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🖥️ SmartDesk AI\n"
        "**IT & HR Helpdesk Assistant** — Ask about VPN, MFA, PTO, benefits, "
        "or create/check support tickets."
    )

    with gr.Row():
        email_box = gr.Textbox(
            label="Your email (required for ticket operations)",
            placeholder="you@company.com",
            scale=3,
        )

    chatbot = gr.Chatbot(label="SmartDesk AI", height=480, type="messages")
    msg_box = gr.Textbox(
        label="Your question",
        placeholder="e.g. How do I set up MFA?",
        show_label=False,
    )
    send_btn = gr.Button("Send", variant="primary")

    # Hidden state for multi-turn HITL
    pending_state = gr.State(None)

    def _submit(message, history, email, pending):
        return _respond(message, history, email, pending)

    send_btn.click(
        _submit,
        inputs=[msg_box, chatbot, email_box, pending_state],
        outputs=[msg_box, chatbot, email_box, pending_state],
    )
    msg_box.submit(
        _submit,
        inputs=[msg_box, chatbot, email_box, pending_state],
        outputs=[msg_box, chatbot, email_box, pending_state],
    )

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
    demo.launch(show_error=True)
