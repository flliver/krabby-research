"""Pytest configuration and hooks."""

import pytest


def pytest_ignore_collect(collection_path, config):
    """Skip collecting Jetson test files when running Isaac Sim tests."""
    # Skip jetson test files when running with isaacsim marker
    marker_expr = config.getoption("-m", default=None)
    if marker_expr and "isaacsim" in marker_expr:
        if "jetson" in str(collection_path).lower():
            return True
    return None

