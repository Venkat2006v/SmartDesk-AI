"""Orchestrator: wires the Supervisor + specialist agents together using
LangGraph (see docs/DESIGN_DECISIONS.md).

TODO: implement. Rough shape:

    from langgraph.graph import StateGraph, END
    from smartdesk.orchestrator.state import AgentState
    from smartdesk.agents.supervisor import supervisor_node
    from smartdesk.agents.it_knowledge_agent import it_knowledge_node
    from smartdesk.agents.hr_knowledge_agent import hr_knowledge_node
    from smartdesk.agents.ticket_creation_agent import ticket_creation_node
    from smartdesk.agents.ticket_status_agent import ticket_status_node

    def build_orchestrator():
        graph = StateGraph(AgentState)
        graph.add_node("supervisor", supervisor_node)
        graph.add_node("it_kb", it_knowledge_node)
        graph.add_node("hr_kb", hr_knowledge_node)
        graph.add_node("create_ticket", ticket_creation_node)
        graph.add_node("ticket_status", ticket_status_node)

        graph.set_entry_point("supervisor")
        graph.add_conditional_edges(
            "supervisor",
            lambda state: state["route"],
            {
                "it_kb": "it_kb",
                "hr_kb": "hr_kb",
                "create_ticket": "create_ticket",
                "ticket_status": "ticket_status",
                "off_topic": END,
            },
        )
        for node in ("it_kb", "hr_kb", "create_ticket", "ticket_status"):
            graph.add_edge(node, END)

        return graph.compile()

    def run_once(graph, initial_state):
        return graph.invoke(initial_state)

If LangGraph's checkpointing/streaming features end up being more
friction than they're worth during development, a plain if/elif dispatch
on state["route"] is equally valid for the rubric — the bonus is clear
multi-agent boundaries, not framework sophistication. Keep that as a
fallback.
"""

from __future__ import annotations

from typing import Any

from smartdesk.orchestrator.state import AgentState


def build_orchestrator() -> Any:
    """Construct and return the compiled LangGraph graph.

    TODO: implement (see module docstring).
    """
    raise NotImplementedError("TODO: build the orchestrator graph")


def run_once(graph: Any, initial_state: AgentState) -> AgentState:
    """Run a single turn through the compiled graph and return the final
    state (graph.invoke(initial_state) for LangGraph).

    TODO: implement.
    """
    raise NotImplementedError("TODO: run the orchestrator for one turn")
