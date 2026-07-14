"""Jira ticketing client — live integration.

Setup (5 minutes):
  1. Free Atlassian account → https://www.atlassian.com/try/cloud/signup
  2. Create a Jira Software project (e.g. key "SD" for SmartDesk)
  3. Generate an API token → https://id.atlassian.com/manage-profile/security/api-tokens
  4. Fill in .env:
       TICKETING_PROVIDER=jira
       TICKETING_BASE_URL=https://<your-site>.atlassian.net
       TICKETING_API_KEY=<api-token>
       JIRA_EMAIL=<your-atlassian-account-email>
       TICKETING_PROJECT_KEY=SD

Install the SDK:
  pip install jira

How requester email is tracked:
  - Stored as a label: "requester:<email>" on every created issue.
  - get_tickets_by_email() uses JQL:
      project=SD AND labels = "requester:venkat@company.com"
  This avoids needing a custom Jira field and works on free tier.
"""

from __future__ import annotations

from typing import List

from smartdesk.tools.ticketing.base import Ticket, TicketingClient


class RealTicketingClient(TicketingClient):
    """Jira-backed ticketing client.

    One instance is created by get_ticketing_client() and reused
    for the lifetime of the process.
    """

    def __init__(self, api_key: str, base_url: str, project_key: str) -> None:
        try:
            from jira import JIRA
        except ImportError:
            raise ImportError(
                "jira package not installed. Run: pip install jira"
            )

        from smartdesk.config import settings

        if not api_key:
            raise ValueError("TICKETING_API_KEY is required for Jira integration.")
        if not base_url:
            raise ValueError("TICKETING_BASE_URL is required (e.g. https://yoursite.atlassian.net).")
        if not settings.jira_email:
            raise ValueError("JIRA_EMAIL is required — your Atlassian account email.")
        if not project_key:
            raise ValueError("TICKETING_PROJECT_KEY is required (e.g. SD).")

        self._project_key = project_key
        self._base_url = base_url.rstrip("/")

        self._jira = JIRA(
            server=self._base_url,
            basic_auth=(settings.jira_email, api_key),
        )

        # Verify credentials on startup — fail fast rather than at first ticket
        self._jira.myself()
        print(f"[jira] Connected to {self._base_url} (project: {project_key})")

    # ------------------------------------------------------------------
    # TicketingClient interface
    # ------------------------------------------------------------------

    def create_ticket(self, email: str, summary: str, description: str) -> Ticket:
        """Create a Jira issue and return it as a Ticket TypedDict.

        The requester email is stored as a label so it can be retrieved
        later by get_tickets_by_email().
        """
        label = _email_label(email)

        issue = self._jira.create_issue(
            project=self._project_key,
            summary=summary,
            description=description,
            issuetype={"name": "Task"},
            labels=[label],
        )

        return Ticket(
            id=issue.key,
            email=email,
            summary=summary,
            description=description,
            status=str(issue.fields.status),
            url=f"{self._base_url}/browse/{issue.key}",
        )

    def get_tickets_by_email(self, email: str) -> List[Ticket]:
        """Search Jira for issues labelled with the requester's email.

        Uses JQL:  project=<KEY> AND labels = "requester:<email>"
        Returns an empty list if none found.
        """
        label = _email_label(email)
        jql = f'project = "{self._project_key}" AND labels = "{label}" ORDER BY created DESC'

        issues = self._jira.search_issues(jql, maxResults=50)

        tickets: List[Ticket] = []
        for issue in issues:
            tickets.append(
                Ticket(
                    id=issue.key,
                    email=email,
                    summary=str(issue.fields.summary),
                    description=str(issue.fields.description or ""),
                    status=str(issue.fields.status),
                    url=f"{self._base_url}/browse/{issue.key}",
                )
            )
        return tickets


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _email_label(email: str) -> str:
    """Convert email to a Jira-safe label.

    Jira labels can't contain '@' or '.', so we encode them.
    e.g. venkat@company.com → requester:venkat_at_company_com
    """
    safe = email.strip().lower().replace("@", "_at_").replace(".", "_")
    return f"requester:{safe}"
