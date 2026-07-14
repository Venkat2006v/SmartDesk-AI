"""Mock ticketing client — fully implemented.

Use this to build/test the full pipeline without real API credentials.
Swap in RealTicketingClient when you're ready for the real integration.
"""

from __future__ import annotations
import itertools
from typing import Dict, List
from smartdesk.tools.ticketing.base import Ticket, TicketingClient


class MockTicketingClient(TicketingClient):
    """In-memory client with sequential IDs (MOCK-1, MOCK-2, ...)."""

    def __init__(self) -> None:
        self._tickets: Dict[str, Ticket] = {}
        self._id_counter = itertools.count(1)

    def create_ticket(self, email: str, summary: str, description: str) -> Ticket:
        tid = f"MOCK-{next(self._id_counter)}"
        ticket: Ticket = {
            "id": tid, "email": email, "summary": summary,
            "description": description, "status": "open",
            "url": f"https://mock-ticketing.local/tickets/{tid}",
        }
        self._tickets[tid] = ticket
        return ticket

    def get_tickets_by_email(self, email: str) -> List[Ticket]:
        needle = email.strip().lower()
        return [t for t in self._tickets.values() if t["email"].strip().lower() == needle]
