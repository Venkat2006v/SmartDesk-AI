"""Supervisor agent — routes the user query to the right agent node.

classify() uses the LLM to map a free-text query onto one of six routes:
  it_kb         — IT question (VPN, MFA, software, hardware, network...)
  hr_kb         — HR question (PTO, benefits, policies, onboarding...)
  combined_kb   — Query explicitly spans BOTH IT and HR domains
  create_ticket — User explicitly wants to open/create a support ticket
  ticket_status — User wants to check status/list of existing tickets
  off_topic     — Anything else (greet, chat, out-of-scope topic)

The LLM is asked to respond with EXACTLY ONE route keyword so we can
parse it without regex gymnastics. Temperature is 0 for determinism.
"""

from __future__ import annotations

from smartdesk.agents._llm import call_llm
from smartdesk.orchestrator.state import AgentState, Route

_VALID_ROUTES: set[Route] = {
    "it_kb", "hr_kb", "combined_kb", "create_ticket", "ticket_status", "off_topic"
}

_SYSTEM_PROMPT = """You are a query router for SmartDesk AI, an IT and HR helpdesk assistant.

Classify the user's query into EXACTLY ONE of these categories:
  it_kb         - IT support questions: VPN, MFA, SSO, TOTP, software installs, hardware, \
network issues, passwords, access requests, laptop setup, Wi-Fi, printers, cloud tools
  hr_kb         - HR questions: PTO, time off, benefits, health insurance, onboarding, \
offboarding, payroll, company policies, performance reviews, equity, parental leave
  combined_kb   - Query EXPLICITLY mentions BOTH IT and HR topics in the same question, \
e.g. "laptop setup AND benefits enrollment", "new hire IT access AND HR policies", \
"onboarding steps for IT equipment AND PTO policy"
  create_ticket - User wants to CREATE or OPEN a new support ticket / helpdesk request
  ticket_status - User wants to CHECK, VIEW, or LIST their existing tickets
  off_topic     - Anything not related to IT or HR support (small talk, general questions, etc.)

Rules:
- Respond with ONLY the category name, nothing else.
- No punctuation, no explanation, no quotes.
- Use combined_kb ONLY when the query clearly asks about both IT and HR topics together.
- For single-domain questions (even if the topic is borderline), prefer it_kb or hr_kb.
- If ambiguous between it_kb/hr_kb and create_ticket, choose create_ticket only when \
the user explicitly says "create", "open", "submit", "raise", or "file" a ticket.
- Prefer it_kb or hr_kb when the user is asking a question, even if they mention a problem.
"""


def classify(state: AgentState) -> Route:
    """Call the LLM to classify state["query"] into a Route."""
    query = state.get("query", "").strip()
    if not query:
        return "off_topic"

    try:
        raw = call_llm(
            system=_SYSTEM_PROMPT,
            user=query,
            temperature=0.0,
        )
        route = raw.strip().lower().replace("-", "_")

        # Validate — fall back to off_topic if LLM returns something unexpected
        if route in _VALID_ROUTES:
            return route  # type: ignore[return-value]

        # Handle partial matches (e.g. "it" → "it_kb")
        for valid in _VALID_ROUTES:
            if route in valid or valid in route:
                return valid  # type: ignore[return-value]

        return "off_topic"

    except Exception as exc:
        # Never crash the pipeline — default to off_topic on LLM failure
        print(f"[supervisor] classify() error: {exc!r} — defaulting to off_topic")
        return "off_topic"


def supervisor_node(state: AgentState) -> AgentState:
    """LangGraph node: classify query and set state['route']."""
    route = classify(state)
    print(f"[supervisor] route → {route!r}")
    return {**state, "route": route}
