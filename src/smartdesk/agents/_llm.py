"""Shared LLM call helper used by all agent nodes.

Single-turn: call_llm(system, user) → assistant response string.
Dispatches on settings.llm_provider ("openai" | "anthropic").

LangSmith tracing
-----------------
call_llm() is decorated with @traceable so every invocation appears as a
named span in LangSmith under the parent LangGraph run:

    graph.invoke()                          ← LangGraph auto-trace
      └─ supervisor_node                    ← node span
      └─ it_knowledge_node                  ← node span
           └─ call_llm [run_type="llm"]     ← this decorator
                └─ openai.chat.completions  ← wrap_openai captures token counts

The @traceable decorator is a no-op when LANGCHAIN_TRACING_V2 is not set,
so there is zero runtime cost in environments without LangSmith configured.

Keep this module small — it is intentionally a thin adapter layer.
Business logic belongs in the individual agent modules.
"""

from __future__ import annotations

from smartdesk.config import settings

# ---------------------------------------------------------------------------
# LangSmith tracing setup
# ---------------------------------------------------------------------------
# Import traceable and SDK wrappers. Both are no-ops when LANGCHAIN_TRACING_V2
# is unset, so this import is safe in all environments.

try:
    from langsmith import traceable
    from langsmith.wrappers import wrap_anthropic, wrap_openai
    _LANGSMITH_AVAILABLE = True
except ImportError:
    # Graceful fallback — tracing simply disabled if langsmith isn't installed
    _LANGSMITH_AVAILABLE = False

    def traceable(**_kw):  # type: ignore[misc]
        def decorator(fn):
            return fn
        return decorator

    def wrap_openai(client):  # type: ignore[misc]
        return client

    def wrap_anthropic(client):  # type: ignore[misc]
        return client


# ---------------------------------------------------------------------------
# Cached SDK clients (wrapped once at first use)
# ---------------------------------------------------------------------------
# Clients are created lazily so config is read after load_dotenv() runs.
# wrap_openai / wrap_anthropic patch the client to emit LangSmith spans that
# include model name, token counts (prompt + completion), and call latency.

_openai_client = None
_anthropic_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        _openai_client = wrap_openai(OpenAI(api_key=settings.llm_api_key))
    return _openai_client


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        _anthropic_client = wrap_anthropic(anthropic.Anthropic(api_key=settings.llm_api_key))
    return _anthropic_client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@traceable(name="call_llm", run_type="llm")
def call_llm(system: str, user: str, temperature: float = 0.2) -> str:
    """Run a single-turn LLM completion and return the response text.

    Args:
        system:      System prompt (instructions / persona).
        user:        User-turn content (query + context, etc.).
        temperature: Sampling temperature. Use 0.0 for deterministic
                     tasks (routing, extraction); 0.2 for answers.

    Returns:
        The assistant's response as a plain string.

    Raises:
        ValueError: If settings.llm_provider is not supported.
        ImportError: If the required SDK is not installed.

    LangSmith:
        When LANGCHAIN_TRACING_V2=true this call appears as a child span of
        the current LangGraph node with the system prompt, user prompt,
        response text, and (via wrap_openai/wrap_anthropic) token counts.
    """
    provider = settings.llm_provider.lower()

    if provider == "openai":
        return _call_openai(system, user, temperature)
    elif provider == "anthropic":
        return _call_anthropic(system, user, temperature)
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: {provider!r}. "
            "Set LLM_PROVIDER to 'openai' or 'anthropic' in .env."
        )


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _call_openai(system: str, user: str, temperature: float) -> str:
    client = _get_openai_client()
    model = settings.llm_model or "gpt-4o-mini"

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def _call_anthropic(system: str, user: str, temperature: float) -> str:
    client = _get_anthropic_client()
    model = settings.llm_model or "claude-haiku-4-5-20251001"

    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=temperature,
    )
    return resp.content[0].text.strip()
