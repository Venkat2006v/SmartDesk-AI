"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from smartdesk.tools.ticketing.mock_client import MockTicketingClient


@pytest.fixture
def mock_ticketing_client() -> MockTicketingClient:
    return MockTicketingClient()
