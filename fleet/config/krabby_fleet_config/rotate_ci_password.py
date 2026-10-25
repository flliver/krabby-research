"""Rotate the persistent CI Cognito operator password.

Used by ``.github/workflows/fleet-ci-rotate-cognito.yml``. Generates a password
that satisfies Cognito's default policy, sets it permanently via
``AdminSetUserPassword``, verifies USER_SRP_AUTH, and writes the new value to a
caller-supplied file (never prints it). Rollback restores the previous password
from ``COGNITO_CI_PASSWORD`` when GitHub secret update fails.
"""
from __future__ import annotations

import argparse
import secrets
import string
import sys
from pathlib import Path

from krabby_fleet_config.loader import (
    CI_COGNITO_PASSWORD_ENV,
    ci_cognito_password,
    load_fleet_config,
)

# Cognito default policy: min 8; upper, lower, digit, symbol required.
_SYMBOLS = "!@#$%^&*-_"
_ALPHABET = string.ascii_letters + string.digits + _SYMBOLS
_PASSWORD_LENGTH = 32


def generate_password(length: int = _PASSWORD_LENGTH) -> str:
    if length < 8:
        raise ValueError("password length must be >= 8")
    while True:
        password = "".join(secrets.choice(_ALPHABET) for _ in range(length))
        if (
            any(c.islower() for c in password)
            and any(c.isupper() for c in password)
            and any(c.isdigit() for c in password)
            and any(c in _SYMBOLS for c in password)
        ):
            return password


def _admin_set_password(user_pool_id: str, username: str, password: str) -> None:
    import boto3

    boto3.client("cognito-idp").admin_set_user_password(
        UserPoolId=user_pool_id,
        Username=username,
        Password=password,
        Permanent=True,
    )


def _verify_srp(
    user_pool_id: str,
    app_client_id: str,
    username: str,
    password: str,
) -> None:
    from pycognito import Cognito

    user = Cognito(user_pool_id, app_client_id, username=username)
    user.authenticate(password=password)


def rotate_and_write(password_path: Path) -> None:
    cfg = load_fleet_config()
    if not cfg.ci_operator_username:
        raise RuntimeError("[ci].operator_username is required in fleet.toml")

    previous = ci_cognito_password()
    new_password = generate_password()

    _admin_set_password(
        cfg.cognito_user_pool_id, cfg.ci_operator_username, new_password
    )
    try:
        _verify_srp(
            cfg.cognito_user_pool_id,
            cfg.cognito_app_client_id,
            cfg.ci_operator_username,
            new_password,
        )
    except Exception:
        _admin_set_password(
            cfg.cognito_user_pool_id, cfg.ci_operator_username, previous
        )
        raise

    password_path.write_text(new_password, encoding="utf-8")
    password_path.chmod(0o600)


def restore_from_env() -> None:
    """Set Cognito password back to ``COGNITO_CI_PASSWORD`` (pre-rotation value)."""
    cfg = load_fleet_config()
    if not cfg.ci_operator_username:
        raise RuntimeError("[ci].operator_username is required in fleet.toml")
    previous = ci_cognito_password()
    _admin_set_password(
        cfg.cognito_user_pool_id, cfg.ci_operator_username, previous
    )
    _verify_srp(
        cfg.cognito_user_pool_id,
        cfg.cognito_app_client_id,
        cfg.ci_operator_username,
        previous,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--write-password",
        type=Path,
        metavar="PATH",
        help="Rotate Cognito password and write the new value to PATH",
    )
    group.add_argument(
        "--restore-from-env",
        action="store_true",
        help=f"Restore Cognito password from {CI_COGNITO_PASSWORD_ENV}",
    )
    args = parser.parse_args(argv)

    if args.restore_from_env:
        restore_from_env()
        print("Restored Cognito CI password from env", file=sys.stderr)
        return 0

    rotate_and_write(args.write_password)
    print(
        f"Rotated Cognito password for {load_fleet_config().ci_operator_username}; "
        f"wrote new value to {args.write_password}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
