"""Bench E2E: skip locally; pass/fail in CI when BENCH_E2E=1 or GitHub Actions."""
from __future__ import annotations

import os

import pytest

from krabby_fleet_config import (
    bench_e2e_requested,
    ci_cognito_password,
    ci_operator_access_token,
    ci_operator_cognito,
    load_fleet_config,
)


def pytest_configure(config: pytest.Config) -> None:
    if not bench_e2e_requested():
        return
    try:
        cfg = load_fleet_config()
        ci_cognito_password()
        if not cfg.service_url or not cfg.cognito_user_pool_id or not cfg.cognito_app_client_id:
            raise RuntimeError("fleet.toml missing fleet URLs or Cognito IDs")
        if not cfg.portal_url:
            raise RuntimeError("fleet.toml missing portal URL")
        ci_operator_access_token()
    except Exception as exc:
        raise pytest.UsageError(f"Bench E2E prerequisites: {exc}") from exc
    if not cfg.ci_operator_username:
        raise pytest.UsageError("[ci].operator_username is required in fleet.toml")
    for key, value in cfg.as_env().items():
        os.environ.setdefault(key, value)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if bench_e2e_requested():
        return
    reason = "Bench E2E disabled (set BENCH_E2E=1 or run in GitHub Actions)"
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        item.add_marker(skip)


@pytest.fixture
def operator_token() -> str:
    return ci_operator_access_token()


@pytest.fixture
def operator_session():
    return ci_operator_cognito()
