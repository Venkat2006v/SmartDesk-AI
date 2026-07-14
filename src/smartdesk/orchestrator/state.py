"""Shared state passed between agents/nodes.

Framework-neutral on purpose: this is a plain TypedDict so it works whether
you adopt LangGraph's StateGraph (which expects a state schema like this),
CrewAI, or a hand-rolled dispatcher. Don't feel obligated to keep every
field — extend or trim as your design evolves, just keep it documented.
"""

from __future__ import annotations

from typing import List, Literal, Optional, TypedDict

try:
    from typing import NotRequired
except ImportError:  # Python < 3.11
    from typing_extensions import NotRequired  # type: ignore

Route = Literal["it_kb", "hr_kb", "create_ticket", "ticket_status", "off_topic"]


class RetrievedChunk(TypedDict):
    text: str
    source: str
    score: float


class AgentState(TypedDict):
    # Input
    query: str
    email: NotRequired[Optional[str]]

    # Routing
    route: NotRequired[Optional[Route]]

    # RAG
    retrieved_chunks: NotRequired[List[RetrievedChunk]]
    confidence_score: NotRequired[float]
    should_escalate: NotRequired[bool]

    # Ticket creation
    ticket_summary: NotRequired[Optional[str]]
    ticket_description: NotRequired[Optional[str]]
    ticket_confirmed: NotRequired[bool]
    ticket_id: NotRequired[Optional[str]]

    # Output
    response: NotRequired[str]
