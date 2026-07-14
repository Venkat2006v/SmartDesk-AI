"""Abstract ticketing client interface.

Both MockTicketingClient and RealTicketingClient implement this, so the
agents (ticket_creation_agent.py, ticket_status_agent.py) never need to
know which concrete backend is in use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, TypedDict


class Ticket(TypedDict):
    id: str
    email: str
    summary: str
    description: str
    status: str  # e.g. "open" | "in_progress" | "closed"
    url: str


class TicketingClient(ABC):
    @abstractmethod
    def create_ticket(self, email: str, summary: str, description: str) -> Ticket:
        """Create a new ticket. Implementations should raise on failure
        rather than returning a partial/invalid Ticket."""
        raise NotImplementedError

    @abstractmethod
    def get_tickets_by_email(self, email: str) -> List[Ticket]:
        """Return all tickets associated with `email` (possibly empty)."""
        raise NotImplementedError
