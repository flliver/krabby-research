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
_SECRETS_ENC = _CONF_DIR / "secrets.enc"

_LEGACY_SECRET_VARS = [
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
{env_file}ExecStart={bin} --config /etc/krabby-bench/config.toml
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

_SSM_CONFIG_SECTION = """\

[ssm]
prefix = "{ssm_prefix}"
"""


def install(
    ecr_repo: str = EcrConfig.repo,
    ecr_tag: str = EcrConfig.tag,
    firmware_channel: str = SmokeConfig.firmware_channel,
    mode: str = "both",
    github_repo: str = "",
    ssm_prefix: str = "",
) -> None:
    if os.geteuid() != 0:
        print("error: krabby-bench install must be run as root (sudo)", file=sys.stderr)
        sys.exit(1)

    bin_path = shutil.which("krabby-bench") or "/home/krabby/.local/bin/krabby-bench"

    # Directories
    _CONF_DIR.mkdir(parents=True, exist_ok=True)
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(["chown", "krabby:krabby", str(_STATE_DIR)], check=True)

    aws_key_id = os.environ.get("BENCH_AWS_KEY_ID", "")
    aws_key_secret = os.environ.get("BENCH_AWS_SECRET_KEY", "")

    use_ssm = bool(ssm_prefix and aws_key_id and aws_key_secret)

    if use_ssm:
        # SSM mode: encrypt the device IAM key; no plaintext credentials on disk
        _preflight_ssm(ssm_prefix, aws_key_id, aws_key_secret)
        from krabby_bench._secrets import write_device_key
        write_device_key(aws_key_id, aws_key_secret)
        subprocess.run(["chown", "krabby:krabby", str(_CONF_DIR / "secrets.enc"),
                        str(_CONF_DIR / "salt")], check=True)
        print(f"  wrote encrypted device key to {_SECRETS_ENC}")
        # Remove legacy smtp.env if it exists from a prior install
        if _SMTP_ENV.exists():
            _SMTP_ENV.unlink()
            print(f"  removed {_SMTP_ENV} (superseded by SSM)")
    elif ssm_prefix:
        print("  warning: --ssm-prefix given but BENCH_AWS_KEY_ID/BENCH_AWS_SECRET_KEY not set "
              "— falling back to legacy mode; SSM prefix will not be written to config")
    else:
        # Legacy mode: write smtp.env from env vars
        secret_pairs = [(v, os.environ.get(v, "")) for v in _LEGACY_SECRET_VARS]
        if any(val for _, val in secret_pairs):
            lines = [f"{name}={val}\n" for name, val in secret_pairs if val]
            _SMTP_ENV.write_text("".join(lines))
            _SMTP_ENV.chmod(0o600)
            print(f"  wrote {_SMTP_ENV}")
        else:
            print("  no BENCH_SMTP_* / BENCH_GITHUB_TOKEN env vars found — skipping smtp.env")

    # config.toml
    toml = _CONFIG_TEMPLATE.format(
        ecr_repo=ecr_repo,
        ecr_tag=ecr_tag,
        firmware_channel=firmware_channel,
        mode=mode,
        github_repo=github_repo,
    )
    if use_ssm:
        toml += _SSM_CONFIG_SECTION.format(ssm_prefix=ssm_prefix)
    _CONFIG_TOML.write_text(toml)
    print(f"  wrote {_CONFIG_TOML}")

    # systemd unit — EnvironmentFile only needed in legacy mode
    env_file_line = "EnvironmentFile=-/etc/krabby-bench/smtp.env\n" if not use_ssm else ""
    _UNIT_PATH.write_text(_UNIT_TEMPLATE.format(bin=bin_path, env_file=env_file_line))
    print(f"  wrote {_UNIT_PATH}")

    subprocess.run(["systemctl", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "enable", "--now", "krabby-bench"], check=True)
    print("  krabby-bench.service enabled and started")
    print("\nDone. Monitor with: journalctl -fu krabby-bench")


def _preflight_ssm(prefix: str, key_id: str, key_secret: str) -> None:
    """Warn if SSM params are unreachable; does not abort install."""
    try:
        import boto3
        client = boto3.client(
            "ssm",
            aws_access_key_id=key_id,
            aws_secret_access_key=key_secret,
            region_name="us-east-1",
        )
        resp = client.get_parameters_by_path(Path=prefix, WithDecryption=False, MaxResults=1)
        if not resp.get("Parameters"):
            print(f"  warning: no SSM params found under {prefix!r} — "
                  "service will start without alert credentials until params are populated")
    except Exception as exc:
        print(f"  warning: SSM preflight failed ({exc}) — "
              "service will start without alert credentials if SSM is unreachable")
