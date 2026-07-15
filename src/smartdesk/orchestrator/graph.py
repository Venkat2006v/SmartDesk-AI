"""LangGraph orchestrator — wires the six agent nodes into a StateMachine.

Graph topology:

    [START]
       │
   supervisor          ← classifies query → sets state["route"]
       │
   ┌───┴──────────────────────────────────────────┐
   │         conditional edges on route           │
   ▼                                              ▼
it_kb  hr_kb  combined_kb  create_ticket  ticket_status  off_topic
   │      │        │             │               │            │
   └──────┴────────┴─────────────┴───────────────┴────────────┘
                                 │
                               [END]

combined_kb runs both IT and HR retrievals in a single node and synthesizes
a unified answer — used only when the query explicitly spans both domains.

Usage:
    from smartdesk.orchestrator.graph import build_orchestrator, run_once

    graph = build_orchestrator()          # call once at startup
    result = run_once(graph, {"query": "How do I reset my VPN?"})
    print(result["response"])
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from smartdesk.agents._llm import call_llm
from smartdesk.agents.combined_knowledge_agent import combined_knowledge_node
from smartdesk.agents.hr_knowledge_agent import hr_knowledge_node
from smartdesk.agents.it_knowledge_agent import it_knowledge_node
from smartdesk.agents.supervisor import supervisor_node
from smartdesk.agents.ticket_creation_agent import ticket_creation_node
from smartdesk.agents.ticket_status_agent import ticket_status_node
from smartdesk.orchestrator.state import AgentState

# ---------------------------------------------------------------------------
# Off-topic node — LLM-generated response
# ---------------------------------------------------------------------------

_OFF_TOPIC_SYSTEM = (
    "You are SmartDesk AI, a friendly IT and HR helpdesk assistant. "
    "The user has sent a message that is not an IT or HR support question. "
    "Respond naturally and warmly — acknowledge greetings, thank-yous, and "
    "farewells in a friendly way. "
    "If the message is genuinely off-topic (not a greeting or social exchange), "
    "politely explain that you specialise in IT and HR support and briefly mention "
    "what you can help with: VPN, MFA, password resets, PTO, benefits, and support tickets. "
    "Keep your reply concise — one or two sentences."
)


def _off_topic_node(state: AgentState) -> AgentState:
    query = state.get("query", "")
    try:
        response = call_llm(system=_OFF_TOPIC_SYSTEM, user=query, temperature=0.4)
    except Exception:
        response = (
            "I'm SmartDesk AI, here to help with IT and HR topics. "
            "Feel free to ask about VPN, MFA, PTO, benefits, or support tickets!"
        )
    return {**state, "response": response}


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------

def _route(state: AgentState) -> str:
    """Read state['route'] set by supervisor_node and return the target node name."""
    return state.get("route", "off_topic")


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_orchestrator() -> Any:
    """Construct and compile the LangGraph StateGraph.

    Returns the compiled graph object. Call this once at application startup
    and reuse the returned object for every query invocation.
    """
    graph: StateGraph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("it_kb", it_knowledge_node)
    graph.add_node("hr_kb", hr_knowledge_node)
    graph.add_node("combined_kb", combined_knowledge_node)
    graph.add_node("create_ticket", ticket_creation_node)
    graph.add_node("ticket_status", ticket_status_node)
    graph.add_node("off_topic", _off_topic_node)

    # Entry point
    graph.set_entry_point("supervisor")

    # Supervisor → agent nodes (conditional on state["route"])
    graph.add_conditional_edges(
        "supervisor",
        _route,
        {
            "it_kb": "it_kb",
            "hr_kb": "hr_kb",
            "combined_kb": "combined_kb",
            "create_ticket": "create_ticket",
            "ticket_status": "ticket_status",
            "off_topic": "off_topic",
        },
    )

    # All agent nodes → END (single-turn; each node sets state["response"])
    for node_name in ("it_kb", "hr_kb", "combined_kb", "create_ticket", "ticket_status", "off_topic"):
        graph.add_edge(node_name, END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_once(graph: Any, initial_state: AgentState) -> AgentState:
    """Run the graph for a single query and return the final state.

    Args:
        graph:         Compiled graph from build_orchestrator().
        initial_state: Must include at least {"query": str}.
                       Optionally include {"email": str} for ticket flows.

    Returns:
        Final AgentState with state["response"] set.
    """
    result = graph.invoke(initial_state)
    return result
