"""Runtime settings: Cognito + IoT ATS + TURN, read from SSM / env at startup.

The EC2 instance this service runs on has its own IAM role (granted by
`FleetServiceStack`) with read access to SSM parameters -- no static AWS
credentials needed. Env var overrides exist for local dev/tests so nothing
here requires real AWS access to run. TURN auth secret is written into
`/etc/krabby-fleet/service.env` by deploy-fleet-service.sh (from Secrets
Manager), loaded via systemd EnvironmentFile.

Region must be set explicitly via ``AWS_REGION`` or ``AWS_DEFAULT_REGION``
(deploy writes both into ``service.env``). No silent default region.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

# Must match fleet/infra/fleet_service_stack.py parameter names -- keep in sync.
COGNITO_USER_POOL_ID_PARAM = "/krabby/fleet/cognito-user-pool-id"
COGNITO_APP_CLIENT_ID_PARAM = "/krabby/fleet/cognito-app-client-id"
IOT_ATS_ENDPOINT_PARAM = "/krabby/fleet/iot-ats-endpoint"

DEFAULT_TURN_TTL_SECS = 3600


def aws_region() -> str:
    """Return the configured AWS region, or raise if unset."""
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        raise RuntimeError(
            "AWS region is not set; export AWS_REGION or AWS_DEFAULT_REGION "
            "(deploy-fleet-service.sh writes both into /etc/krabby-fleet/service.env)"
        )
    return region


# Back-compat: ``from krabby_fleet_service._config import AWS_REGION`` still
# works (resolves via aws_region() on access).
def __getattr__(name: str):
    if name == "AWS_REGION":
        return aws_region()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass(frozen=True)
class Settings:
    aws_region: str
    cognito_user_pool_id: str
    cognito_app_client_id: str
    iot_ats_endpoint: str
    turn_auth_secret: str = ""
    turn_host: str = ""
    turn_ttl_secs: int = DEFAULT_TURN_TTL_SECS

    @property
    def cognito_issuer(self) -> str:
        return f"https://cognito-idp.{self.aws_region}.amazonaws.com/{self.cognito_user_pool_id}"

    @property
    def cognito_jwks_url(self) -> str:
        return f"{self.cognito_issuer}/.well-known/jwks.json"


def _ssm_parameter(name: str) -> str:
    import boto3

    client = boto3.client("ssm", region_name=aws_region())
    return client.get_parameter(Name=name)["Parameter"]["Value"]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    region = aws_region()
    user_pool_id = os.environ.get("KRABBY_FLEET_COGNITO_USER_POOL_ID") or _ssm_parameter(
        COGNITO_USER_POOL_ID_PARAM
    )
    app_client_id = os.environ.get("KRABBY_FLEET_COGNITO_APP_CLIENT_ID") or _ssm_parameter(
        COGNITO_APP_CLIENT_ID_PARAM
    )
    iot_ats = os.environ.get("KRABBY_FLEET_IOT_ATS_ENDPOINT") or _ssm_parameter(
        IOT_ATS_ENDPOINT_PARAM
    )
    return Settings(
        aws_region=region,
        cognito_user_pool_id=user_pool_id,
        cognito_app_client_id=app_client_id,
        iot_ats_endpoint=iot_ats,
        turn_auth_secret=os.environ.get("KRABBY_FLEET_TURN_AUTH_SECRET", ""),
        turn_host=os.environ.get("KRABBY_FLEET_TURN_HOST", ""),
        turn_ttl_secs=_env_int("KRABBY_FLEET_TURN_TTL_SECS", DEFAULT_TURN_TTL_SECS),
    )
