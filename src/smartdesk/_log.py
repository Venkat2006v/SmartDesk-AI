"""Lightweight verbose-print helper.

Agent debug output is gated behind this so it can be suppressed by setting
SMARTDESK_VERBOSE=false in the environment or .env file.
"""

from __future__ import annotations

from smartdesk.config import settings


def vprint(*args, **kwargs) -> None:
    """Print only when verbose mode is enabled (SMARTDESK_VERBOSE != false)."""
    if settings.verbose:
        print(*args, **kwargs)
