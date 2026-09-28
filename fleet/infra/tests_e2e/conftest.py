"""Bench control-plane E2E: skip locally; pass/fail when BENCH_E2E=1 or GitHub Actions."""
from __future__ import annotations

import os

import pytest

from krabby_fleet_config import bench_e2e_requested, load_fleet_config


def pytest_configure(config: pytest.Config) -> None:
    if not bench_e2e_requested():
        return
    try:
        cfg = load_fleet_config()
    except Exception as exc:
        raise pytest.UsageError(f"Bench E2E prerequisites: {exc}") from exc
    for key, value in cfg.as_env().items():
        os.environ.setdefault(key, value)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if bench_e2e_requested():
        return
    reason = "Bench E2E disabled (set BENCH_E2E=1 or run in GitHub Actions)"
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        item.add_marker(skip)
