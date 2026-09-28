"""Fleet env for service E2E (committed fleet.toml + optional env overrides)."""
from krabby_fleet_config.e2e_env import (
    AWS_REGION,
    BENCH_THING_NAME,
    COGNITO_APP_CLIENT_ID,
    COGNITO_USER_POOL_ID,
    FLEET_E2E_CONFIGURED,
    FLEET_PORTAL_URL,
    FLEET_SERVICE_URL,
)

__all__ = [
    "AWS_REGION",
    "BENCH_THING_NAME",
    "COGNITO_APP_CLIENT_ID",
    "COGNITO_USER_POOL_ID",
    "FLEET_E2E_CONFIGURED",
    "FLEET_PORTAL_URL",
    "FLEET_SERVICE_URL",
]
