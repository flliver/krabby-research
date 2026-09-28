"""Shared AWS IoT Core helpers for `krabby enroll` / `krabby agent`.

Device identity (cert, private key, Amazon Root CA, ATS endpoint, thing name)
lives under `/etc/krabby/iot/` — a fixed, root-owned location rather than the
invoking user's home (contrast `_state.py`'s SUDO_USER-aware home). Unlike the
locomotion image ref (a per-user "what did I last install" preference),
device credentials are fleet infrastructure: `krabby enroll` already needs
root to install packages and write systemd units, `krabby-agent.service` runs
as root so it can read the key (0600) without a user/group dance, and a
single, well-known path means `krabby agent` never has to guess where enroll
put things.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

IOT_DIR = Path("/etc/krabby/iot")
CERT_PATH = IOT_DIR / "device.pem.crt"
KEY_PATH = IOT_DIR / "private.pem.key"
ROOT_CA_PATH = IOT_DIR / "AmazonRootCA1.pem"
CONFIG_PATH = IOT_DIR / "config.json"

# Amazon Trust Services root, required for IoT Core's ATS endpoints (the
# legacy VeriSign root is being phased out). Fetched fresh at enroll time
# rather than bundled in the package, so a stale copy shipped in an old
# `krabby-launcher` release can never cause a cert-chain failure — enroll is
# already online (AWS API calls) so one more HTTPS GET costs nothing extra.
AMAZON_ROOT_CA_URL = "https://www.amazontrust.com/repository/AmazonRootCA1.pem"


def fetch_amazon_root_ca() -> bytes:
    with urllib.request.urlopen(AMAZON_ROOT_CA_URL, timeout=15) as resp:
        return resp.read()


def write_identity(thing_name: str, endpoint: str, cert_pem: bytes, key_pem: bytes, root_ca_pem: bytes) -> None:
    """Persist device identity to IOT_DIR. Called once, by `krabby enroll`."""
    IOT_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(IOT_DIR, 0o700)

    CERT_PATH.write_bytes(cert_pem)
    ROOT_CA_PATH.write_bytes(root_ca_pem)
    CONFIG_PATH.write_text(json.dumps({"thing_name": thing_name, "endpoint": endpoint}, indent=2))

    # Private key never leaves the device and is never world/group readable.
    KEY_PATH.write_bytes(key_pem)
    os.chmod(KEY_PATH, 0o600)


def load_config() -> dict:
    """Read back what `krabby enroll` wrote. No AWS creds needed — agent only reads local files."""
    if not CONFIG_PATH.exists():
        print(f"[err] {CONFIG_PATH} not found — run `krabby enroll` first", file=sys.stderr)
        sys.exit(1)
    try:
        return json.loads(CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[err] cannot read {CONFIG_PATH}: {exc}", file=sys.stderr)
        sys.exit(1)


def identity_files_present() -> bool:
    return CERT_PATH.exists() and KEY_PATH.exists() and ROOT_CA_PATH.exists() and CONFIG_PATH.exists()


def build_mqtt_connection(thing_name: str, endpoint: str) -> Any:
    """mTLS MQTT connection to IoT Core, cert/key/CA read from IOT_DIR.

    Uses the AWS IoT Device SDK v2's `mtls_from_path` builder. The underlying
    CRT MQTT client reconnects automatically with exponential backoff on any
    connection drop — that's a built-in property of the client returned here,
    not something bolted on in this repo; callers just `connect()` once and
    let the client keep retrying for the life of the process.
    """
    from awscrt import io as crt_io
    from awsiot import mqtt_connection_builder

    event_loop_group = crt_io.EventLoopGroup(1)
    host_resolver = crt_io.DefaultHostResolver(event_loop_group)
    client_bootstrap = crt_io.ClientBootstrap(event_loop_group, host_resolver)

    def _on_connection_interrupted(connection, error, **kwargs):
        print(f"[!]   MQTT connection interrupted: {error} — reconnecting ...")

    def _on_connection_resumed(connection, return_code, session_present, **kwargs):
        print(f"[ok]  MQTT connection resumed (return_code={return_code}, session_present={session_present})")

    return mqtt_connection_builder.mtls_from_path(
        endpoint=endpoint,
        cert_filepath=str(CERT_PATH),
        pri_key_filepath=str(KEY_PATH),
        ca_filepath=str(ROOT_CA_PATH),
        client_bootstrap=client_bootstrap,
        client_id=thing_name,
        clean_session=False,
        keep_alive_secs=30,
        on_connection_interrupted=_on_connection_interrupted,
        on_connection_resumed=_on_connection_resumed,
    )


# --- krabby-agent.service -------------------------------------------------
#
# Separate unit from krabby-locomotion.service (installed by `krabby install`
# / `_host.py`): locomotion owns the app container, agent owns the MQTT
# connection. Same idempotent write-if-changed + daemon-reload + enable
# pattern as `_host.py`'s `_boot_service_unit` / `_ensure_boot_service`
# (see `_BOOT_SERVICE_PATH` there), duplicated rather than shared because the
# two units differ in owning user, dependency ordering, and restart policy.

AGENT_SERVICE_PATH = Path("/etc/systemd/system/krabby-agent.service")
AGENT_SERVICE_NAME = "krabby-agent.service"


def _agent_service_unit(krabby_bin: str) -> str:
    """systemd unit that runs `krabby agent` — the always-on MQTT client.

    Runs as root (not the invoking user, unlike krabby-locomotion.service):
    it reads the 0600 private key in IOT_DIR and spawns localproxy against
    localhost:22, neither of which benefit from running as an unprivileged
    user here.
    """
    return f"""\
[Unit]
Description=Krabby fleet agent (AWS IoT Core MQTT client)
Documentation=https://github.com/flliver/krabby-research
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=300
StartLimitBurst=10

[Service]
Type=simple
User=root
ExecStart={krabby_bin} agent
Restart=always
RestartSec=5
TimeoutStopSec=20

[Install]
WantedBy=multi-user.target
"""


def ensure_agent_service() -> bool:
    """Install + enable krabby-agent.service. Does not start it (caller verifies connect first)."""
    import shutil

    if not shutil.which("systemctl"):
        print("[skip] systemctl not found — skipping krabby-agent service install (not a systemd host?)")
        return True

    krabby_bin = shutil.which("krabby") or "/usr/local/bin/krabby"
    unit = _agent_service_unit(krabby_bin)

    if AGENT_SERVICE_PATH.exists() and AGENT_SERVICE_PATH.read_text() == unit:
        print(f"[ok]  {AGENT_SERVICE_NAME} already in place: {AGENT_SERVICE_PATH}")
    else:
        try:
            AGENT_SERVICE_PATH.write_text(unit)
        except PermissionError:
            print(f"[err] cannot write {AGENT_SERVICE_PATH} — run with sudo", file=sys.stderr)
            return False
        subprocess.run(["systemctl", "daemon-reload"])
        print(f"[+]   wrote {AGENT_SERVICE_PATH}")

    ret = subprocess.run(["systemctl", "enable", AGENT_SERVICE_NAME]).returncode
    if ret != 0:
        print(f"[err] systemctl enable {AGENT_SERVICE_NAME} failed (exit {ret})", file=sys.stderr)
        return False
    print(f"[ok]  {AGENT_SERVICE_NAME} enabled; will start on next boot "
          f"(or `sudo systemctl start {AGENT_SERVICE_NAME}` now).")
    return True
