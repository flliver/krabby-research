"""Unit tests for CI Cognito password rotation helpers (no AWS)."""
from __future__ import annotations

import string

from krabby_fleet_config.rotate_ci_password import _SYMBOLS, generate_password


def test_generate_password_meets_cognito_default_policy():
    for _ in range(20):
        password = generate_password()
        assert len(password) == 32
        assert any(c.islower() for c in password)
        assert any(c.isupper() for c in password)
        assert any(c.isdigit() for c in password)
        assert any(c in _SYMBOLS for c in password)
        assert all(c in string.ascii_letters + string.digits + _SYMBOLS for c in password)


def test_generate_password_unique():
    samples = {generate_password() for _ in range(10)}
    assert len(samples) == 10
