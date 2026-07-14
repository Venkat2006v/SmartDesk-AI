"""Fully implemented — proves validation helpers work out of the box."""

from __future__ import annotations
from smartdesk.guardrails.validation import is_valid_email


def test_is_valid_email_accepts_valid_address() -> None:
    assert is_valid_email("user@example.com") is True


def test_is_valid_email_rejects_garbage() -> None:
    assert is_valid_email("not-an-email") is False
    assert is_valid_email("missing-domain@") is False


def test_is_valid_email_rejects_empty_or_none() -> None:
    assert is_valid_email("") is False
    assert is_valid_email(None) is False
