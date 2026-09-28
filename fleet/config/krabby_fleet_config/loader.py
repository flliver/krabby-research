"""Resolve and parse the committed fleet config file.

Search order for ``fleet.toml``:

1. ``KRABBY_FLEET_CONFIG`` env (explicit path)
2. Walk parents from cwd for ``fleet/config/fleet.toml`` (repo checkout / CI)
3. ``fleet.toml`` next to this package (``pip install -e fleet/config``)
4. ``~/.config/krabby-fleet/config.toml`` (operator machine override)

Secrets (CI Cognito password, etc.) are **not** in the TOML — read via env helpers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib  # type: ignore[import]
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

_PACKAGE_DIR = Path(__file__).resolve().parent
_BUNDLED_CONFIG = _PACKAGE_DIR.parent / "fleet.toml"
_OPERATOR_OVERRIDE = Path.home() / ".config" / "krabby-fleet" / "config.toml"

CI_COGNITO_PASSWORD_ENV = "COGNITO_CI_PASSWORD"


@dataclass(frozen=True)
class FleetConfig:
    aws_region: str
    domain: str
    service_url: str
    portal_url: str
    cognito_user_pool_id: str
    cognito_app_client_id: str
    iot_thing_type: str
    iot_device_policy: str
    bench_thing_name: str
    ci_operator_username: str = ""
    ci_github_actions_role_arn: str = ""
    deploy_hosted_zone_name: str = ""
    deploy_github_owner: str = ""
    deploy_github_repo: str = ""
    deploy_github_branch: str = ""
    deploy_github_oidc_provider_arn: str = ""
    deploy_cdk_bootstrap_qualifier: str = ""

    @property
    def cognito_issuer(self) -> str:
        return (
            f"https://cognito-idp.{self.aws_region}.amazonaws.com/"
            f"{self.cognito_user_pool_id}"
        )

    def cdk_context_args(self) -> list[str]:
        """``cdk -c key=value`` args for ControlPlaneStack + FleetServiceStack."""
        required = {
            "domainName": self.domain,
            "hostedZoneName": self.deploy_hosted_zone_name,
            "githubOwner": self.deploy_github_owner,
            "githubRepo": self.deploy_github_repo,
            "githubBranch": self.deploy_github_branch,
            "githubOidcProviderArn": self.deploy_github_oidc_provider_arn,
            "cdkBootstrapQualifier": self.deploy_cdk_bootstrap_qualifier,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(
                "fleet.toml [deploy] (and [fleet].domain) incomplete for CDK; "
                f"missing: {', '.join(missing)}"
            )
        args: list[str] = []
        for key, value in required.items():
            args.extend(["-c", f"{key}={value}"])
        return args

    def as_env(self) -> dict[str, str]:
        """Non-secret keys as env-style names for pytest / shell export."""
        out = {
            "AWS_REGION": self.aws_region,
            "AWS_DEFAULT_REGION": self.aws_region,
            "FLEET_SERVICE_URL": self.service_url,
            "FLEET_PORTAL_URL": self.portal_url,
            "COGNITO_USER_POOL_ID": self.cognito_user_pool_id,
            "COGNITO_APP_CLIENT_ID": self.cognito_app_client_id,
            "BENCH_THING_NAME": self.bench_thing_name,
            "KRABBY_FLEET_COGNITO_USER_POOL_ID": self.cognito_user_pool_id,
            "KRABBY_FLEET_COGNITO_APP_CLIENT_ID": self.cognito_app_client_id,
        }
        if self.ci_operator_username:
            out["COGNITO_CI_USERNAME"] = self.ci_operator_username
        if self.ci_github_actions_role_arn:
            out["FLEET_CI_ROLE_ARN"] = self.ci_github_actions_role_arn
        if self.deploy_hosted_zone_name:
            out["FLEET_CDK_HOSTED_ZONE_NAME"] = self.deploy_hosted_zone_name
            out["FLEET_CDK_GITHUB_OWNER"] = self.deploy_github_owner
            out["FLEET_CDK_GITHUB_REPO"] = self.deploy_github_repo
            out["FLEET_CDK_GITHUB_BRANCH"] = self.deploy_github_branch
            out["FLEET_CDK_GITHUB_OIDC_PROVIDER_ARN"] = (
                self.deploy_github_oidc_provider_arn
            )
            out["FLEET_CDK_BOOTSTRAP_QUALIFIER"] = (
                self.deploy_cdk_bootstrap_qualifier
            )
            out["FLEET_CDK_DOMAIN_NAME"] = self.domain
        return out


def find_fleet_config_path() -> Path | None:
    explicit = os.environ.get("KRABBY_FLEET_CONFIG", "").strip()
    if explicit:
        return Path(explicit).expanduser()

    for parent in [Path.cwd(), *_PACKAGE_DIR.parents]:
        candidate = parent / "fleet" / "config" / "fleet.toml"
        if candidate.is_file():
            return candidate
        if parent.name == "fleet" and (parent / "config" / "fleet.toml").is_file():
            return parent / "config" / "fleet.toml"

    if _BUNDLED_CONFIG.is_file():
        return _BUNDLED_CONFIG

    if _OPERATOR_OVERRIDE.is_file():
        return _OPERATOR_OVERRIDE

    return None


def _require_str(raw: dict, section: str, key: str, path: Path) -> str:
    block = raw.get(section)
    if not isinstance(block, dict):
        raise ValueError(f"{path}: missing [{section}] section")
    value = block.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: [{section}] {key} is required and must be non-empty")
    return value.strip()


def _optional_str(raw: dict, section: str, key: str) -> str:
    block = raw.get(section)
    if not isinstance(block, dict):
        return ""
    value = block.get(key)
    if value is None:
        return ""
    return str(value).strip()


def load_fleet_config(path: Path | None = None) -> FleetConfig:
    config_path = path or find_fleet_config_path()
    if config_path is None:
        raise FileNotFoundError(
            "fleet config not found. Set KRABBY_FLEET_CONFIG, run from a repo "
            "checkout with fleet/config/fleet.toml, pip install -e fleet/config, "
            "or create ~/.config/krabby-fleet/config.toml"
        )

    raw = tomllib.loads(config_path.read_text())
    aws_region = _require_str(raw, "aws", "region", config_path)
    domain = _require_str(raw, "fleet", "domain", config_path)

    fleet = raw.get("fleet")
    if not isinstance(fleet, dict):
        raise ValueError(f"{config_path}: missing [fleet] section")

    portal_url = str(fleet.get("portal_url") or f"https://{domain}").rstrip("/")
    service_url = str(fleet.get("service_url") or f"{portal_url}/api").rstrip("/")

    cognito_user_pool_id = _require_str(raw, "cognito", "user_pool_id", config_path)
    cognito_app_client_id = _require_str(raw, "cognito", "client_id", config_path)

    iot = raw.get("iot")
    if not isinstance(iot, dict):
        raise ValueError(f"{config_path}: missing [iot] section")
    iot_thing_type = str(iot.get("thing_type") or "Krab").strip()
    iot_device_policy = str(iot.get("device_policy") or "KrabDevicePolicy").strip()

    bench_thing_name = _optional_str(raw, "bench", "thing_name") or "bench-krabby-ci"
    ci_operator_username = _optional_str(raw, "ci", "operator_username")
    ci_github_actions_role_arn = _optional_str(raw, "ci", "github_actions_role_arn")
    deploy_hosted_zone_name = _optional_str(raw, "deploy", "hosted_zone_name")
    deploy_github_owner = _optional_str(raw, "deploy", "github_owner")
    deploy_github_repo = _optional_str(raw, "deploy", "github_repo")
    deploy_github_branch = _optional_str(raw, "deploy", "github_branch")
    deploy_github_oidc_provider_arn = _optional_str(
        raw, "deploy", "github_oidc_provider_arn"
    )
    deploy_cdk_bootstrap_qualifier = _optional_str(
        raw, "deploy", "cdk_bootstrap_qualifier"
    )

    return FleetConfig(
        aws_region=aws_region,
        domain=domain,
        service_url=service_url,
        portal_url=portal_url,
        cognito_user_pool_id=cognito_user_pool_id,
        cognito_app_client_id=cognito_app_client_id,
        iot_thing_type=iot_thing_type,
        iot_device_policy=iot_device_policy,
        bench_thing_name=bench_thing_name,
        ci_operator_username=ci_operator_username,
        ci_github_actions_role_arn=ci_github_actions_role_arn,
        deploy_hosted_zone_name=deploy_hosted_zone_name,
        deploy_github_owner=deploy_github_owner,
        deploy_github_repo=deploy_github_repo,
        deploy_github_branch=deploy_github_branch,
        deploy_github_oidc_provider_arn=deploy_github_oidc_provider_arn,
        deploy_cdk_bootstrap_qualifier=deploy_cdk_bootstrap_qualifier,
    )


def ci_cognito_password() -> str:
    """CI-only secret from GitHub Actions (or local export for manual runs)."""
    value = os.environ.get(CI_COGNITO_PASSWORD_ENV, "").strip()
    if not value:
        raise RuntimeError(
            f"{CI_COGNITO_PASSWORD_ENV} is not set (GitHub secret / local export)"
        )
    return value
