"""Human-in-the-loop confirmation gate.

Used by ticket_creation_agent.py before any write/side-effecting action.
Per the project brief: the agent must not create tickets silently.

TODO: implement. In a CLI context this can just be an input() prompt; in
a UI context (ui/app.py) it would be a button/confirmation dialog. Keep
the function signature simple so either context can call it.
"""

from __future__ import annotations


def confirm_action(summary: str) -> bool:
    """Ask the user to confirm `summary` before proceeding.

    Return True if confirmed, False otherwise.

    TODO: implement (e.g. input(f"Create ticket: {summary}? [y/N] ")).
    """
    raise NotImplementedError("TODO: implement HITL confirmation")
