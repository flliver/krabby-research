"""Tests for krabby_fleet_config loader."""
from __future__ import annotations

from pathlib import Path

import pytest

from krabby_fleet_config.loader import FleetConfig, load_fleet_config


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_load_fleet_config_derives_urls(tmp_path):
    path = _write(
        tmp_path / "fleet.toml",
        """
        [aws]
        region = "us-east-2"

        [fleet]
        domain = "fleet.example.com"

        [cognito]
        user_pool_id = "us-east-2_ABC"
        client_id = "client123"

        [iot]
        thing_type = "Krab"
        device_policy = "KrabDevicePolicy"

        [bench]
        thing_name = "bench-krabby-ci"

        [ci]
        operator_username = "ci@example.com"
        github_actions_role_arn = "arn:aws:iam::123456789012:role/krabby-fleet-ci"

        [deploy]
        hosted_zone_name = "example.com"
        github_owner = "acme"
        github_repo = "krabby-research"
        github_branch = "main"
        github_oidc_provider_arn = "arn:aws:iam::123456789012:oidc-provider/token.actions.githubusercontent.com"
        cdk_bootstrap_qualifier = "hnb659fds"
        """,
    )
    cfg = load_fleet_config(path)
    assert cfg.aws_region == "us-east-2"
    assert cfg.portal_url == "https://fleet.example.com"
    assert cfg.service_url == "https://fleet.example.com/api"
    assert cfg.cognito_issuer.endswith("/us-east-2_ABC")
    assert cfg.bench_thing_name == "bench-krabby-ci"
    assert cfg.ci_operator_username == "ci@example.com"
    assert cfg.ci_github_actions_role_arn == "arn:aws:iam::123456789012:role/krabby-fleet-ci"
    assert cfg.deploy_hosted_zone_name == "example.com"
    assert cfg.cdk_context_args() == [
        "-c", "domainName=fleet.example.com",
        "-c", "hostedZoneName=example.com",
        "-c", "githubOwner=acme",
        "-c", "githubRepo=krabby-research",
        "-c", "githubBranch=main",
        "-c", "githubOidcProviderArn=arn:aws:iam::123456789012:oidc-provider/token.actions.githubusercontent.com",
        "-c", "cdkBootstrapQualifier=hnb659fds",
    ]
    env = cfg.as_env()
    assert env["FLEET_CDK_DOMAIN_NAME"] == "fleet.example.com"
    assert env["FLEET_CDK_BOOTSTRAP_QUALIFIER"] == "hnb659fds"


def test_as_env(tmp_path):
    path = _write(
        tmp_path / "fleet.toml",
        """
        [aws]
        region = "us-east-1"

        [fleet]
        domain = "f.example.com"

        [cognito]
        user_pool_id = "pool"
        client_id = "client"

        [iot]
        """,
    )
    cfg = load_fleet_config(path)
    env = cfg.as_env()
    assert env["FLEET_SERVICE_URL"] == "https://f.example.com/api"
    assert env["COGNITO_USER_POOL_ID"] == "pool"


def test_missing_cognito_fails(tmp_path):
    path = _write(
        tmp_path / "fleet.toml",
        """
        [aws]
        region = "us-east-1"

        [fleet]
        domain = "f.example.com"

        [cognito]
        user_pool_id = ""
        client_id = ""

        [iot]
        """,
    )
    with pytest.raises(ValueError, match="user_pool_id"):
        load_fleet_config(path)
