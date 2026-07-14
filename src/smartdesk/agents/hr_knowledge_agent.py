"""HR Knowledge Agent — answers HR questions grounded in the HR knowledge base.

See it_knowledge_agent.py for the rationale behind keeping IT and HR as
separate agents rather than one generic knowledge agent. Same flow as the
IT agent, just scoped to domain="hr".

TODO: implement the flow:
1. retrieve(state["query"], domain="hr") -> retrieved chunks
2. decide_escalation(...) -> answer vs. escalate
3. If answerable: build_grounded_prompt(...) -> call the LLM -> set
   state["response"]
4. If not: set state["should_escalate"] = True + an honest "I don't know"
   style response.
"""

from __future__ import annotations

from smartdesk.orchestrator.state import AgentState


def hr_knowledge_node(state: AgentState) -> AgentState:
    """Orchestrator-facing node for HR knowledge-base queries."""
    raise NotImplementedError("TODO: implement HR knowledge agent (RAG)")
