"""IT Knowledge Agent — answers IT questions grounded in the IT knowledge base.

Deliberately a separate agent from hr_knowledge_agent.py (rather than one
generic "knowledge agent" with a domain parameter) so the project
demonstrates genuine multi-agent specialization, per the capstone's
"multi-agent IT/HR split" bonus item. The two agents can share helper
functions (e.g. via rag/retriever.py) without being the same agent.

TODO: implement the flow:
1. retrieve(state["query"], domain="it") -> retrieved chunks
2. decide_escalation(...) -> should this be answered or routed to a ticket?
3. If answerable: build_grounded_prompt(...) -> call the LLM -> set
   state["response"]
4. If not: set state["should_escalate"] = True and a helpful response
   explaining you couldn't find a confident answer (the orchestrator can
   then offer to create a ticket).
"""

from __future__ import annotations

from smartdesk.orchestrator.state import AgentState


def it_knowledge_node(state: AgentState) -> AgentState:
    """Orchestrator-facing node for IT knowledge-base queries."""
    raise NotImplementedError("TODO: implement IT knowledge agent (RAG)")
