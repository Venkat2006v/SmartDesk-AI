"""Mock ticketing client — fully implemented.

This is intentionally NOT a TODO: the project brief explicitly sanctions
mocking the ticketing API with realistic dummy data if a live integration
doesn't come together in time. Use this to build/test the rest of the
pipeline end-to-end, and optionally swap in RealTicketingClient later for
extra credit.
"""

from __future__ import annotations

import itertools
from typing import Dict, List

from smartdesk.tools.ticketing.base import Ticket, TicketingClient


class MockTicketingClient(TicketingClient):
    """In-memory ticketing client with sequential mock IDs (e.g. MOCK-1)."""

    def __init__(self) -> None:
        self._tickets: Dict[str, Ticket] = {}
        self._id_counter = itertools.count(1)

    def create_ticket(self, email: str, summary: str, description: str) -> Ticket:
        ticket_id = f"MOCK-{next(self._id_counter)}"
        ticket: Ticket = {
            "id": ticket_id,
            "email": email,
            "summary": summary,
            "description": description,
            "status": "open",
            "url": f"https://mock-ticketing.local/tickets/{ticket_id}",
        }
        self._tickets[ticket_id] = ticket
        return ticket

    def get_tickets_by_email(self, email: str) -> List[Ticket]:
        needle = email.strip().lower()
        return [t for t in self._tickets.values() if t["email"].strip().lower() == needle]
