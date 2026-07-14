"""CLI entry point.

Run with: python -m smartdesk.main

TODO: once orchestrator/graph.py is implemented, wire this loop up to call
it per turn (build the AgentState, run the graph, print the response, and
carry over conversation/session state turn to turn).
"""

from __future__ import annotations


def main() -> None:
    print("SmartDesk AI — type 'exit' to quit.")
    print("[TODO] wire this up to src/smartdesk/orchestrator/graph.py")

    while True:
        try:
            query = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break

        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        # TODO: replace with a real call once the orchestrator exists, e.g.:
        #   from smartdesk.orchestrator.graph import build_orchestrator, run_once
        #   graph = build_orchestrator()
        #   state = run_once(graph, {"query": query})
        #   print(state.get("response"))
        print("[TODO] no orchestrator wired up yet — see graph.py")


if __name__ == "__main__":
    main()
