"""Validation + reliability helpers — fully implemented."""

from __future__ import annotations
import functools
import re
import time
from typing import Callable, TypeVar

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
T = TypeVar("T")


def is_valid_email(value: object) -> bool:
    """Return True if value looks like a valid email address."""
    if not isinstance(value, str) or not value.strip():
        return False
    return bool(_EMAIL_RE.match(value.strip()))


def with_retry(
    max_attempts: int = 3, backoff_seconds: float = 1.0
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator: retry a flaky function with linear backoff."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        time.sleep(backoff_seconds * attempt)
            assert last_exc is not None
            raise last_exc
        return wrapper
    return decorator
