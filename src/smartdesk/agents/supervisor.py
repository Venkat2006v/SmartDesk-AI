"""Supervisor agent: classifies the incoming query and sets state["route"].

This is the entry point of the multi-agent system. Everything downstream
depends on this classification being reasonably accurate.

TODO: replace `classify` with real logic. Options:
- Rule-based (keyword/regex matching for "ticket", "status", "password",
  "PTO", etc.) — fast, transparent, brittle.
- LLM-based (ask the model to pick one of the Route values) — more robust
  to phrasing, costs a call, needs structured output parsing.
- Hybrid — cheap rules for obvious cases, LLM fallback for ambiguous ones.

The stub below hardcodes a single route so the rest of the pipeline is
runnable end-to-end while you build out the other agents. Replace it.
"""

from __future__ import annotations

from smartdesk.orchestrator.state import AgentState, Route


def classify(state: AgentState) -> Route:
    """Return the route for this query.

    TODO: implement real classification. Currently hardcoded to "it_kb"
    purely so callers have a runnable default during development.
    """
    # TODO: remove this hardcoded stub once real classification is in place.
    return "it_kb"


def supervisor_node(state: AgentState) -> AgentState:
    """Orchestrator-facing node: sets state["route"] and returns state."""
    route = classify(state)
    state["route"] = route
    return state
