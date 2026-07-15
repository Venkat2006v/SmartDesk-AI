"""Shared state passed between agents/nodes."""

from __future__ import annotations

from typing import List, Literal, Optional, TypedDict

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired  # type: ignore

Route = Literal["it_kb", "hr_kb", "combined_kb", "create_ticket", "ticket_status", "off_topic"]


class RetrievedChunk(TypedDict):
    text: str
    source: str
    score: float


class AgentState(TypedDict):
    query: str
    email: NotRequired[Optional[str]]
    route: NotRequired[Optional[Route]]
    retrieved_chunks: NotRequired[List[RetrievedChunk]]
    confidence_score: NotRequired[float]
    should_escalate: NotRequired[bool]
    ticket_summary: NotRequired[Optional[str]]
    ticket_description: NotRequired[Optional[str]]
    ticket_confirmed: NotRequired[bool]
    ticket_id: NotRequired[Optional[str]]
    response: NotRequired[str]
