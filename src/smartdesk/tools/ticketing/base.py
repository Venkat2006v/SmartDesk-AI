"""Abstract ticketing client interface."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, TypedDict


class Ticket(TypedDict):
    id: str
    email: str
    summary: str
    description: str
    status: str
    url: str


class TicketingClient(ABC):
    @abstractmethod
    def create_ticket(self, email: str, summary: str, description: str) -> Ticket:
        raise NotImplementedError

    @abstractmethod
    def get_tickets_by_email(self, email: str) -> List[Ticket]:
        raise NotImplementedError
