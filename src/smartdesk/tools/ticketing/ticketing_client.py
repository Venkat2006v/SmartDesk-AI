"""Real ticketing client — implement this for full credit on the ticketing
integration (the brief allows the mock as a fallback, but a real
integration is worth more).

TODO: implement against whichever tool you pick (Jira / Asana / Notion /
Linear / GitHub Issues). Sketch for Jira:

    from jira import JIRA
    self._client = JIRA(server=base_url, basic_auth=(email, api_token))
    self._client.create_issue(project=project_key, summary=..., description=...,
                               issuetype={"name": "Task"})

Sketch for GitHub Issues (via REST, no SDK needed):

    POST https://api.github.com/repos/{owner}/{repo}/issues
    {"title": summary, "body": description}

Whatever you choose, map its response fields onto the `Ticket` TypedDict
in base.py so the agents don't need to know which provider is behind this.
"""

from __future__ import annotations

from typing import List

from smartdesk.tools.ticketing.base import Ticket, TicketingClient


class RealTicketingClient(TicketingClient):
    def __init__(self, api_key: str, base_url: str, project_key: str) -> None:
        # TODO: initialize the real SDK/HTTP client here.
        raise NotImplementedError("TODO: implement real ticketing client init")

    def create_ticket(self, email: str, summary: str, description: str) -> Ticket:
        raise NotImplementedError("TODO: implement real create_ticket")

    def get_tickets_by_email(self, email: str) -> List[Ticket]:
        raise NotImplementedError("TODO: implement real get_tickets_by_email")
