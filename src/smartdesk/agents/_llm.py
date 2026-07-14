"""Shared LLM call helper used by all agent nodes.

Single-turn: call_llm(system, user) → assistant response string.
Dispatches on settings.llm_provider ("openai" | "anthropic").

Keep this module small — it is intentionally a thin adapter layer.
Business logic belongs in the individual agent modules.
"""

from __future__ import annotations

from smartdesk.config import settings


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
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    client = OpenAI(api_key=settings.llm_api_key)
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
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")

    client = anthropic.Anthropic(api_key=settings.llm_api_key)
    model = settings.llm_model or "claude-haiku-4-5-20251001"

    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=temperature,
    )
    return resp.content[0].text.strip()
