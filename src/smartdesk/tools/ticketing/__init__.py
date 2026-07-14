"""Ticketing client factory.

Usage:
    from smartdesk.tools.ticketing import get_ticketing_client
    client = get_ticketing_client()
    ticket = client.create_ticket(email, summary, description)

Provider is controlled by TICKETING_PROVIDER in .env:
    "mock"  (default) — MockTicketingClient, no credentials needed
    "jira"            — RealTicketingClient (Jira), needs TICKETING_API_KEY etc.
"""

from __future__ import annotations

from smartdesk.config import settings
from smartdesk.tools.ticketing.base import TicketingClient

_client: TicketingClient | None = None


def get_ticketing_client() -> TicketingClient:
    """Return the singleton ticketing client for this process."""
    global _client
    if _client is None:
        provider = settings.ticketing_provider.lower()
        if provider == "mock":
            from smartdesk.tools.ticketing.mock_client import MockTicketingClient
            _client = MockTicketingClient()
        elif provider == "jira":
            from smartdesk.tools.ticketing.ticketing_client import RealTicketingClient
            _client = RealTicketingClient(
                api_key=settings.ticketing_api_key,
                base_url=settings.ticketing_base_url,
                project_key=settings.ticketing_project_key,
            )
        else:
            raise ValueError(
                f"Unknown TICKETING_PROVIDER: {provider!r}. "
                "Set to 'mock' or 'jira' in .env."
            )
    return _client
