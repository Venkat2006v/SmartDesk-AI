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

    # ── Session email — collected lazily by ticket agents when needed ────────
    # Per spec: "when it can't answer, collect the employee's email + issue
    # summary + description". Email is not required for KB queries, so we do
    # NOT ask upfront. The ticket_creation_agent and ticket_status_agent will
    # request it the first time a ticket operation is triggered, then we cache
    # it in session_email so they don't re-ask on subsequent turns.
    session_email: str | None = None

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

        # Build initial state for this turn. Pass cached email so ticket agents
        # don't re-ask once the user has provided it earlier in the session.
        initial_state = {
            "query": raw,
            "email": session_email,
        }

        try:
            result = run_once(graph, initial_state)
        except KeyboardInterrupt:
            print("\n[interrupted]")
            continue
        except Exception as exc:
            print(f"[error] {exc}")
            continue

        # Cache email for the rest of the session once an agent validates it
        if result.get("email"):
            session_email = result["email"]

        # Print response
        response = result.get("response", "[no response]")
        print(f"\nSmartDesk: {response}\n")


if __name__ == "__main__":
    main()
