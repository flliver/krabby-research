"""Config dataclass + TOML loading from /etc/krabby-bench/config.toml."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib  # type: ignore[import]
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError as exc:
        raise ImportError("Install tomli on Python < 3.11: pip install tomli") from exc

CONFIG_PATH = Path("/etc/krabby-bench/config.toml")
STATE_PATH = Path("/var/lib/krabby-bench/state.json")


@dataclass
class EcrConfig:
    repo: str = "public.ecr.aws/t7t7b3i3/krabby-locomotion"
    tag: str = "mainline-latest"
    poll_interval: int = 60


@dataclass
class SmokeConfig:
    firmware_channel: str = "release/0.2.9"
    run_hal_check: bool = False


@dataclass
class AlertConfig:
    mode: str = "email"  # "email" | "github" | "both"
    dedup_window: int = 3600


@dataclass
class SmtpConfig:
    host: str = field(default_factory=lambda: os.environ.get("BENCH_SMTP_HOST", ""))
    port: int = field(default_factory=lambda: int(os.environ.get("BENCH_SMTP_PORT", "587")))
    user: str = field(default_factory=lambda: os.environ.get("BENCH_SMTP_USER", ""))
    password: str = field(default_factory=lambda: os.environ.get("BENCH_SMTP_PASSWORD", ""))
    from_addr: str = field(default_factory=lambda: os.environ.get("BENCH_SMTP_FROM", ""))
    to_addr: str = field(default_factory=lambda: os.environ.get("BENCH_SMTP_TO", ""))


@dataclass
class GithubConfig:
    repo: str = field(default_factory=lambda: os.environ.get("BENCH_GITHUB_REPO", ""))
    token: str = field(default_factory=lambda: os.environ.get("BENCH_GITHUB_TOKEN", ""))


@dataclass
class Config:
    ecr: EcrConfig = field(default_factory=EcrConfig)
    smoke: SmokeConfig = field(default_factory=SmokeConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    smtp: SmtpConfig = field(default_factory=SmtpConfig)
    github: GithubConfig = field(default_factory=GithubConfig)
    state_path: Path = field(default_factory=lambda: STATE_PATH)


def _overlay(dc, raw: dict):
    """Return a new dataclass instance with fields overridden from raw."""
    known = {f.name for f in dc.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return dc.__class__(**{**dc.__dict__, **{k: v for k, v in raw.items() if k in known}})


def load_config(path: Path = CONFIG_PATH) -> Config:
    if not path.exists():
        return Config()
    raw = tomllib.loads(path.read_text())
    cfg = Config()
    if "ecr" in raw:
        cfg.ecr = _overlay(cfg.ecr, raw["ecr"])
    if "smoke" in raw:
        cfg.smoke = _overlay(cfg.smoke, raw["smoke"])
    if "alert" in raw:
        cfg.alert = _overlay(cfg.alert, raw["alert"])
    if "smtp" in raw:
        smtp_raw = {k if k != "from" else "from_addr": v for k, v in raw["smtp"].items()}
        smtp_raw = {k if k != "to" else "to_addr": v for k, v in smtp_raw.items()}
        cfg.smtp = _overlay(cfg.smtp, smtp_raw)
    if "github" in raw:
        cfg.github = _overlay(cfg.github, raw["github"])
    if "state_path" in raw:
        cfg.state_path = Path(raw["state_path"])
    return cfg
