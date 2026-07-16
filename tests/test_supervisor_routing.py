"""Tests for agents/supervisor.py — classify() routing."""

from __future__ import annotations

from unittest.mock import patch

from smartdesk.orchestrator.state import AgentState


class TestSupervisorClassify:
    def _classify(self, llm_response: str, query: str) -> str:
        """Mock the LLM response and run classify()."""
        with patch("smartdesk.agents.supervisor.call_llm", return_value=llm_response):
            from smartdesk.agents import supervisor
            state = AgentState(query=query)
            return supervisor.classify(state)

    def test_classify_routes_it_kb(self) -> None:
        assert self._classify("it_kb", "How do I reset my VPN password?") == "it_kb"

    def test_classify_routes_hr_kb(self) -> None:
        assert self._classify("hr_kb", "How many PTO days do I get?") == "hr_kb"

    def test_classify_routes_create_ticket(self) -> None:
        assert self._classify("create_ticket", "Open a ticket for my broken laptop") == "create_ticket"

    def test_classify_routes_ticket_status(self) -> None:
        assert self._classify("ticket_status", "What tickets do I have open?") == "ticket_status"

    def test_classify_routes_off_topic(self) -> None:
        assert self._classify("off_topic", "What is the weather today?") == "off_topic"

    def test_classify_invalid_llm_response_defaults_to_off_topic(self) -> None:
        """LLM returns unexpected text → graceful fallback to off_topic, no crash."""
        assert self._classify("I cannot determine this", "some query") == "off_topic"

    def test_classify_llm_error_defaults_to_off_topic(self) -> None:
        """LLM raises an exception → off_topic, never propagates error."""
        with patch("smartdesk.agents.supervisor.call_llm", side_effect=RuntimeError("API down")):
            from smartdesk.agents import supervisor
            route = supervisor.classify(AgentState(query="anything"))
        assert route == "off_topic"

    def test_classify_routes_ticket_status_queries(self) -> None:
        route = self._classify("ticket_status", "Can you show me all my support tickets?")
        assert route == "ticket_status"

    def test_classify_routes_hr_vs_it_queries(self) -> None:
        assert self._classify("it_kb", "I can't connect with MFA on my phone.") == "it_kb"
        assert self._classify("hr_kb", "How do I enroll in health benefits?") == "hr_kb"
