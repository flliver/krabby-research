"""Resolve fleet E2E env vars from committed config with env overrides."""
from __future__ import annotations

import os
from functools import lru_cache

from krabby_fleet_config.loader import FleetConfig, load_fleet_config


@lru_cache(maxsize=1)
def get_fleet_config() -> FleetConfig | None:
    try:
        return load_fleet_config()
    except (FileNotFoundError, ValueError):
        return None


def _env(name: str, fallback: str = "") -> str:
    explicit = os.environ.get(name, "").strip()
    if explicit:
        return explicit
    cfg = get_fleet_config()
    if cfg is not None:
        return cfg.as_env().get(name, fallback)
    return fallback


AWS_REGION = _env("AWS_REGION", "us-east-1")
FLEET_SERVICE_URL = _env("FLEET_SERVICE_URL", "").rstrip("/")
FLEET_PORTAL_URL = _env("FLEET_PORTAL_URL", "").rstrip("/")
COGNITO_USER_POOL_ID = _env("COGNITO_USER_POOL_ID", "")
COGNITO_APP_CLIENT_ID = _env("COGNITO_APP_CLIENT_ID", "")
BENCH_THING_NAME = _env("BENCH_THING_NAME", "bench-krabby-ci")

FLEET_E2E_CONFIGURED = bool(
    FLEET_SERVICE_URL and COGNITO_USER_POOL_ID and COGNITO_APP_CLIENT_ID
)
