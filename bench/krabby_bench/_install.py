"""Bootstrap the krabby-bench systemd service on a fresh Jetson."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from krabby_bench._config import EcrConfig, SmokeConfig

_UNIT_PATH = Path("/etc/systemd/system/krabby-bench.service")
_CONF_DIR = Path("/etc/krabby-bench")
_STATE_DIR = Path("/var/lib/krabby-bench")
_SMTP_ENV = _CONF_DIR / "smtp.env"
_CONFIG_TOML = _CONF_DIR / "config.toml"

_SECRET_VARS = [
    "BENCH_SMTP_HOST",
    "BENCH_SMTP_PORT",
    "BENCH_SMTP_USER",
    "BENCH_SMTP_PASSWORD",
    "BENCH_SMTP_FROM",
    "BENCH_SMTP_TO",
    "BENCH_GITHUB_TOKEN",
]

_UNIT_TEMPLATE = """\
[Unit]
Description=Krabby bench watchdog
After=network-online.target

[Service]
User=krabby
EnvironmentFile=-/etc/krabby-bench/smtp.env
ExecStart={bin} --config /etc/krabby-bench/config.toml
Restart=on-failure

[Install]
WantedBy=multi-user.target
"""

_CONFIG_TEMPLATE = """\
[ecr]
repo = "{ecr_repo}"
tag = "{ecr_tag}"
poll_interval = 60

[smoke]
firmware_channel = "{firmware_channel}"

[alert]
mode = "{mode}"

[github]
repo = "{github_repo}"
"""


def install(
    ecr_repo: str = EcrConfig.repo,
    ecr_tag: str = EcrConfig.tag,
    firmware_channel: str = SmokeConfig.firmware_channel,
    mode: str = "both",
    github_repo: str = "",
) -> None:
    if os.geteuid() != 0:
        print("error: krabby-bench install must be run as root (sudo)", file=sys.stderr)
        sys.exit(1)

    bin_path = shutil.which("krabby-bench") or "/home/krabby/.local/bin/krabby-bench"

    # Directories
    _CONF_DIR.mkdir(parents=True, exist_ok=True)
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(["chown", "krabby:krabby", str(_STATE_DIR)], check=True)

    # Secrets env file — written only if at least one var is set
    secret_pairs = [(v, os.environ.get(v, "")) for v in _SECRET_VARS]
    if any(val for _, val in secret_pairs):
        lines = [f"{name}={val}\n" for name, val in secret_pairs if val]
        _SMTP_ENV.write_text("".join(lines))
        _SMTP_ENV.chmod(0o600)
        print(f"  wrote {_SMTP_ENV}")
    else:
        print("  no BENCH_SMTP_* / BENCH_GITHUB_TOKEN env vars found — skipping smtp.env")

    # config.toml
    _CONFIG_TOML.write_text(_CONFIG_TEMPLATE.format(
        ecr_repo=ecr_repo,
        ecr_tag=ecr_tag,
        firmware_channel=firmware_channel,
        mode=mode,
        github_repo=github_repo,
    ))
    print(f"  wrote {_CONFIG_TOML}")

    # systemd unit
    _UNIT_PATH.write_text(_UNIT_TEMPLATE.format(bin=bin_path))
    print(f"  wrote {_UNIT_PATH}")

    subprocess.run(["systemctl", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "enable", "--now", "krabby-bench"], check=True)
    print("  krabby-bench.service enabled and started")
    print("\nDone. Monitor with: journalctl -fu krabby-bench")
