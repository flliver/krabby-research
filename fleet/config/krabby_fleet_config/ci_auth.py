"""Authenticate the persistent CI Cognito operator (fleet.toml + env secret)."""
from __future__ import annotations

from functools import lru_cache

from krabby_fleet_config.loader import ci_cognito_password, load_fleet_config


@lru_cache(maxsize=1)
def ci_operator_cognito():
    from pycognito import Cognito

    cfg = load_fleet_config()
    if not cfg.ci_operator_username:
        raise RuntimeError("[ci].operator_username is required in fleet.toml")
    user = Cognito(
        cfg.cognito_user_pool_id,
        cfg.cognito_app_client_id,
        username=cfg.ci_operator_username,
    )
    user.authenticate(password=ci_cognito_password())
    return user


def ci_operator_access_token() -> str:
    return ci_operator_cognito().access_token
