"""Real ticketing client — implement for full credit on the integration.

TODO: implement against your chosen tool (Jira / Asana / Notion / etc.).

Jira sketch:
    from jira import JIRA
    self._client = JIRA(server=base_url, basic_auth=(email, api_token))
    issue = self._client.create_issue(project=project_key,
        summary=summary, description=description, issuetype={"name": "Task"})

GitHub Issues sketch (no SDK):
    POST https://api.github.com/repos/{owner}/{repo}/issues
    {"title": summary, "body": description}

Map the provider's response onto the Ticket TypedDict in base.py.
Credentials come from config.settings (ticketing_api_key, ticketing_base_url, etc.).
"""

from __future__ import annotations
from typing import List
from smartdesk.tools.ticketing.base import Ticket, TicketingClient


class RealTicketingClient(TicketingClient):
    def __init__(self, api_key: str, base_url: str, project_key: str) -> None:
        raise NotImplementedError("TODO: initialize the real ticketing SDK/client")

    def create_ticket(self, email: str, summary: str, description: str) -> Ticket:
        raise NotImplementedError("TODO: implement create_ticket")

    def get_tickets_by_email(self, email: str) -> List[Ticket]:
        raise NotImplementedError("TODO: implement get_tickets_by_email")
