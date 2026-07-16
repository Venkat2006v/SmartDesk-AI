"""CLI entry point.

Run with:
    cd SmartDesk-AI
    python -m smartdesk.main

Or after `pip install -e .`:
    smartdesk

Prerequisites:
  1. Fill in .env (LLM_API_KEY, EMBEDDING_PROVIDER, etc.)
  2. Build the knowledge base:  python scripts/build_knowledge_base.py
  3. Then run this file.

Session flow:
  - Email is NOT collected upfront. Per spec, email is only needed for ticket
    operations (creation / status). The ticket agents request it when triggered
    and cache it in state["email"] for the rest of the session.
  - Each user turn invokes the full LangGraph pipeline.
  - Type 'exit' or Ctrl-C to quit.
"""

from __future__ import annotations

import sys


def _print_banner() -> None:
    print("=" * 60)
    print("  SmartDesk AI — IT & HR Helpdesk Assistant")
    print("  Type 'exit' to quit | 'help' for example queries")
    print("=" * 60)


def _print_help() -> None:
    print(
        "\nExample queries:"
        "\n  IT  : How do I connect to the VPN?"
        "\n  IT  : Walk me through setting up MFA."
        "\n  HR  : How many PTO days do I get per year?"
        "\n  HR  : When is benefits open enrollment?"
        "\n  TKT : Create a ticket — my laptop won't connect to Wi-Fi."
        "\n  TKT : What tickets do I have open? (uses your session email)"
        "\n"
    )


def main() -> None:
    _print_banner()

    # ── Lazy import — keeps startup fast if running --help ──────────────────
    try:
        from smartdesk.orchestrator.graph import build_orchestrator, run_once
    except ImportError as exc:
        print(f"[error] Missing dependency: {exc}")
        print("Run: pip install -e '.[dev]'")
        sys.exit(1)

    # ── Build graph once ────────────────────────────────────────────────────
    print("\n[startup] Loading knowledge base and building agent graph...")
    try:
        graph = build_orchestrator()
    except Exception as exc:
        print(f"[error] Failed to build orchestrator: {exc}")
        print("Make sure you have run: python scripts/build_knowledge_base.py")
        sys.exit(1)
    print("[startup] Ready.\n")

    # ── LangSmith tracing status ────────────────────────────────────────────
    from smartdesk.config import settings
    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        print(
            f"[tracing] LangSmith ENABLED — project: {settings.langchain_project!r}\n"
            f"          View traces → https://smith.langchain.com\n"
        )
    else:
        print(
            "[tracing] LangSmith disabled "
            "(set LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY in .env to enable)\n"
        )

    # ── Session state — persisted across turns ───────────────────────────────
    # Email: not collected upfront; ticket agents request it when needed.
    # Ticket summary/description: cached when the agent asks for email mid-flow
    # so the next turn (user replies with their email) can resume from the
    # already-extracted fields without re-parsing the original request.
    session_email: str | None = None
    session_ticket_summary: str | None = None
    session_ticket_description: str | None = None
    session_pending_action: str | None = None
    # Last successfully answered KB query — used to inject context into vague
    # follow-up messages ("still not working") and vague ticket requests
    # ("create a ticket for this issue") so the supervisor and ticket agent
    # can understand what "this" refers to.
    session_last_kb_query: str | None = None

    # ── Main REPL loop ──────────────────────────────────────────────────────
    print()
    while True:
        try:
            raw = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break

        if not raw:
            continue
        if raw.lower() in {"exit", "quit", "q"}:
            print("bye.")
            break
        if raw.lower() in {"help", "?"}:
            _print_help()
            continue

        # Enrich vague follow-ups and ticket requests with conversation context.
        # Covers two cases:
        #   "I'm still not able to fix it"   → supervisor can't route without context
        #   "Create a ticket for this issue" → ticket agent can't extract a summary
        _FOLLOWUP_WORDS = {"still", "not working", "didn't work", "doesn't work",
                           "same issue", "same problem", "not fixed", "still failing"}
        _VAGUE_TICKET   = {"this issue", "the issue", "the problem", "this problem",
                           "the above", "for this", "about this"}
        _TICKET_WORDS   = {"ticket", "create", "raise", "open", "log"}
        raw_lower = raw.lower()

        query = raw
        if session_last_kb_query:
            is_vague_followup = any(w in raw_lower for w in _FOLLOWUP_WORDS)
            is_vague_ticket   = (any(ref in raw_lower for ref in _VAGUE_TICKET)
                                 and any(kw in raw_lower for kw in _TICKET_WORDS))
            if is_vague_followup or is_vague_ticket:
                query = f"{raw}\n\n[Context: the user was previously asking about: {session_last_kb_query}]"

        # Build initial state for this turn. Pass cached session values so
        # agents don't re-ask for things already collected this session.
        initial_state = {
            "query": query,
            "email": session_email,
            "ticket_summary": session_ticket_summary,
            "ticket_description": session_ticket_description,
            "pending_action": session_pending_action,
        }

        try:
            result = run_once(graph, initial_state)
        except KeyboardInterrupt:
            print("\n[interrupted]")
            continue
        except Exception as exc:
            print(f"[error] {exc}")
            continue

        # Track last successfully answered KB query for follow-up context injection
        route = result.get("route", "")
        if route in ("it_kb", "hr_kb", "combined_kb") and not result.get("should_escalate"):
            session_last_kb_query = raw  # store the original user query (not enriched)

        # Cache validated values for the rest of the session
        if result.get("email"):
            session_email = result["email"]
        # Persist pending ticket fields (agent may have extracted them but
        # not yet created the ticket — e.g. waiting for the user's email)
        if result.get("ticket_summary"):
            session_ticket_summary = result["ticket_summary"]
        if result.get("ticket_description"):
            session_ticket_description = result["ticket_description"]
        # Track what the agent is waiting for (e.g. email for ticket lookup)
        session_pending_action = result.get("pending_action") or None
        # Clear pending ticket once it's been successfully created
        if result.get("ticket_id"):
            session_ticket_summary = None
            session_ticket_description = None

        # Print response
        response = result.get("response", "[no response]")
        print(f"\nSmartDesk: {response}\n")


if __name__ == "__main__":
    main()
