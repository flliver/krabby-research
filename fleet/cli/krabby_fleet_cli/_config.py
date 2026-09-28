"""~/.config/krabby-fleet/config.toml loading — delegates to shared fleet config."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from krabby_fleet_config import load_fleet_config as _load_shared
except ImportError:
    _load_shared = None  # type: ignore[assignment,misc]

try:
    import tomllib  # type: ignore[import]
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

CONFIG_PATH = Path.home() / ".config" / "krabby-fleet" / "config.toml"


@dataclass(frozen=True)
class Config:
    service_url: str
    cognito_user_pool_id: str
    cognito_client_id: str
    default_ssh_user: str = "operator"
    # Optional override; default is service_url with a trailing /api stripped.
    portal_url: str = ""


def portal_base_url(config: Config) -> str:
    """Public portal origin for browser routes (teleop, devices UI)."""
    if config.portal_url:
        return config.portal_url.rstrip("/")
    service = config.service_url.rstrip("/")
    if service.endswith("/api"):
        return service[: -len("/api")] or service
    return service


def _from_shared() -> Config:
    shared = _load_shared()
    return Config(
        service_url=shared.service_url,
        cognito_user_pool_id=shared.cognito_user_pool_id,
        cognito_client_id=shared.cognito_app_client_id,
        portal_url=shared.portal_url,
    )


def _from_legacy_toml(path: Path) -> Config:
    raw = tomllib.loads(path.read_text())
    try:
        fleet = raw["fleet"]
        cognito = raw["cognito"]
    except KeyError as exc:
        print(f"error: {path} missing required section {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        service_url = fleet["service_url"]
        user_pool_id = cognito["user_pool_id"]
        client_id = cognito["client_id"]
    except KeyError as exc:
        print(f"error: {path} missing required key {exc}", file=sys.stderr)
        sys.exit(1)

    return Config(
        service_url=service_url.rstrip("/"),
        cognito_user_pool_id=user_pool_id,
        cognito_client_id=client_id,
        default_ssh_user=raw.get("ssh", {}).get("default_user", "operator"),
        portal_url=str(fleet.get("portal_url") or "").rstrip("/"),
    )


def load_config(path: Path = CONFIG_PATH) -> Config:
    if _load_shared is not None and path == CONFIG_PATH:
        try:
            return _from_shared()
        except FileNotFoundError:
            pass

    if not path.exists():
        print(
            f"error: fleet config not found.\n\n"
            "Install shared config: pip install -e fleet/config\n"
            "Or create ~/.config/krabby-fleet/config.toml — see fleet/config/README.md\n",
            file=sys.stderr,
        )
        sys.exit(1)

    return _from_legacy_toml(path)
