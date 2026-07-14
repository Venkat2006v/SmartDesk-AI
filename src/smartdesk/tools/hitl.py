"""Human-in-the-loop (HITL) confirmation gate.

Called by ticket_creation_agent BEFORE any write operation.
Never skip this — it is a hard requirement of the capstone rubric.

CLI mode:  prompts the user via stdin (input()).
UI mode:   raise NotImplementedError — Gradio handles confirmation
           as a multi-turn exchange inside ui/app.py.

To add a new frontend, extend _confirm_ui() and set HITL_MODE.
"""

from __future__ import annotations

import os

# Reads from env so the UI layer can override without code changes.
# "cli" (default) | "ui" (Gradio — confirmation handled externally)
_HITL_MODE = os.getenv("HITL_MODE", "cli").lower()


def confirm_action(summary: str, description: str = "") -> bool:
    """Ask the user to confirm a write action. Return True if confirmed.

    Args:
        summary:     One-line description of the action (shown to user).
        description: Optional longer detail shown before the prompt.
    """
    if _HITL_MODE == "cli":
        return _confirm_cli(summary, description)
    elif _HITL_MODE == "ui":
        # In UI mode the Gradio app handles multi-turn confirmation.
        # ticket_creation_node checks state["ticket_confirmed"] instead.
        raise NotImplementedError(
            "HITL in UI mode is managed by ui/app.py — "
            "do not call confirm_action() directly from agent nodes in UI mode."
        )
    else:
        raise ValueError(f"Unknown HITL_MODE: {_HITL_MODE!r}. Use 'cli' or 'ui'.")


def _confirm_cli(summary: str, description: str) -> bool:
    """Print the proposed action and prompt y/N on stdin."""
    print("\n" + "─" * 60)
    print("⚠  ACTION REQUIRES YOUR CONFIRMATION")
    print("─" * 60)
    print(f"  Action  : {summary}")
    if description:
        # Wrap description at 56 chars for readability
        import textwrap
        wrapped = textwrap.fill(description, width=56, initial_indent="  Details : ", subsequent_indent="            ")
        print(wrapped)
    print("─" * 60)

    try:
        answer = input("  Proceed? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nConfirmation cancelled.")
        return False

    confirmed = answer in {"y", "yes"}
    if confirmed:
        print("  ✓ Confirmed.\n")
    else:
        print("  ✗ Cancelled.\n")
    return confirmed
