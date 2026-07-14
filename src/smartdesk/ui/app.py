"""Optional deployed UI (bonus item). Not required for the core rubric.

TODO (optional): build a small chat UI, e.g. with Gradio:

    import gradio as gr
    from smartdesk.orchestrator.graph import build_orchestrator, run_once

    graph = build_orchestrator()

    def respond(message, history):
        state = run_once(graph, {"query": message})
        return state.get("response", "")

    demo = gr.ChatInterface(respond, title="SmartDesk AI")

    if __name__ == "__main__":
        demo.launch()

Swap in Streamlit/FastAPI+frontend if you'd rather. The orchestrator
interface (build_orchestrator/run_once) is intentionally UI-agnostic so
main.py (CLI) and this module can both call into it the same way.
"""

from __future__ import annotations

# TODO: implement the UI entry point described above.
