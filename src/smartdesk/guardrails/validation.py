"""Validation + reliability helpers — fully implemented.

These are generic enough that there's no real design decision to make, so
unlike the rest of the scaffold they're not left as TODOs. Use
with_retry() around any flaky external call (LLM, vector store, ticketing
API) per the project's error-handling rubric item.
"""

from __future__ import annotations

import functools
import re
import time
from typing import Callable, TypeVar

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

T = TypeVar("T")


def is_valid_email(value: object) -> bool:
    """Return True if `value` looks like a valid email address."""
    if not isinstance(value, str) or not value.strip():
        return False
    return bool(_EMAIL_RE.match(value.strip()))


def with_retry(
    max_attempts: int = 3, backoff_seconds: float = 1.0
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator: retry a flaky function up to max_attempts times, with
    linear backoff between attempts. Re-raises the last exception if all
    attempts fail.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - intentional broad catch for retry
                    last_exc = exc
                    if attempt < max_attempts:
                        time.sleep(backoff_seconds * attempt)
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator
