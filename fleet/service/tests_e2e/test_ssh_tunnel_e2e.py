"""SSH tunnel and device API E2E: CI operator against deployed fleet + bench.

Uses the persistent CI Cognito operator from ``fleet/config/fleet.toml`` and
``COGNITO_CI_PASSWORD`` (not scratch users). Skipped locally unless
``BENCH_E2E=1`` or GitHub Actions; when enabled, missing config, auth, tools,
or bench → **fail** (no skip).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterator

import boto3
import pytest
import requests
from pycognito import Cognito

from tests_e2e._fleet_env import (
    AWS_REGION,
    BENCH_THING_NAME,
    COGNITO_APP_CLIENT_ID,
    COGNITO_USER_POOL_ID,
    FLEET_SERVICE_URL,
)

BENCH_SSH_USER = os.environ.get("BENCH_SSH_USER", "operator")

_CLI_BIN = "krabby-fleet"
_LOCALPROXY_BIN = "localproxy"

_CLI_CONFIG_PATH = Path.home() / ".config" / "krabby-fleet" / "config.toml"
_CLI_SESSION_PATH = Path.home() / ".config" / "krabby-fleet" / "session.json"


def test_open_and_close_tunnel_happy_path(operator_token: str):
    resp = requests.post(
        f"{FLEET_SERVICE_URL}/devices/{BENCH_THING_NAME}/ssh-tunnel",
        headers={"Authorization": f"Bearer {operator_token}"},
        timeout=30,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["tunnelId"]
    assert body["sourceAccessToken"]
    assert body["region"]

    close_resp = requests.delete(
        f"{FLEET_SERVICE_URL}/devices/{BENCH_THING_NAME}/ssh-tunnel/{body['tunnelId']}",
        headers={"Authorization": f"Bearer {operator_token}"},
        timeout=30,
    )
    assert close_resp.status_code == 204


def test_get_devices_lists_bench_robot(operator_token: str):
    resp = requests.get(
        f"{FLEET_SERVICE_URL}/devices",
        headers={"Authorization": f"Bearer {operator_token}"},
        timeout=30,
    )
    assert resp.status_code == 200, resp.text
    thing_names = {device["thingName"] for device in resp.json()}
    assert BENCH_THING_NAME in thing_names


def test_get_device_returns_bench_shadow(operator_token: str):
    resp = requests.get(
        f"{FLEET_SERVICE_URL}/devices/{BENCH_THING_NAME}",
        headers={"Authorization": f"Bearer {operator_token}"},
        timeout=30,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["thingName"] == BENCH_THING_NAME
    assert "connected" in body
    assert isinstance(body.get("reported"), dict)
    reported = body["reported"]
    assert reported.get("timestamp"), "expected recent shadow reported.timestamp from bench agent"
    assert time.time() - int(reported["timestamp"]) < 300, (
        "shadow reported.timestamp should be within the last 5 minutes"
    )


@pytest.fixture
def cli_operator_session(operator_session: Cognito) -> Iterator[None]:
    """Real ~/.config/krabby-fleet/{config.toml,session.json} for krabby-fleet CLI."""
    config_backup = _CLI_CONFIG_PATH.read_text() if _CLI_CONFIG_PATH.exists() else None
    session_backup = _CLI_SESSION_PATH.read_text() if _CLI_SESSION_PATH.exists() else None
    _CLI_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    _CLI_CONFIG_PATH.write_text(
        "[fleet]\n"
        f'service_url = "{FLEET_SERVICE_URL}"\n\n'
        "[cognito]\n"
        f'user_pool_id = "{COGNITO_USER_POOL_ID}"\n'
        f'client_id = "{COGNITO_APP_CLIENT_ID}"\n\n'
        "[ssh]\n"
        f'default_user = "{BENCH_SSH_USER}"\n'
    )
    _CLI_SESSION_PATH.write_text(
        json.dumps(
            {
                "access_token": operator_session.access_token,
                "id_token": operator_session.id_token,
                "refresh_token": operator_session.refresh_token,
            }
        )
    )
    _CLI_SESSION_PATH.chmod(0o600)

    try:
        yield
    finally:
        if config_backup is not None:
            _CLI_CONFIG_PATH.write_text(config_backup)
        else:
            _CLI_CONFIG_PATH.unlink(missing_ok=True)
        if session_backup is not None:
            _CLI_SESSION_PATH.write_text(session_backup)
        else:
            _CLI_SESSION_PATH.unlink(missing_ok=True)


def test_krabby_fleet_ssh_runs_command_end_to_end(cli_operator_session: None):
    assert shutil.which(_CLI_BIN), f"{_CLI_BIN} not on PATH (required for bench E2E)"
    assert shutil.which(_LOCALPROXY_BIN), f"{_LOCALPROXY_BIN} not on PATH (required for bench E2E)"

    result = subprocess.run(
        [_CLI_BIN, "ssh", BENCH_THING_NAME],
        input="echo hello\nexit\n",
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "hello" in result.stdout, result.stdout + result.stderr

    match = re.search(r"tunnel (\S+) open", result.stdout)
    assert match, f"couldn't find tunnel ID in CLI output:\n{result.stdout}"
    tunnel_id = match.group(1)

    tunneling = boto3.client("iotsecuretunneling", region_name=AWS_REGION)
    with pytest.raises(tunneling.exceptions.ResourceNotFoundException):
        tunneling.describe_tunnel(tunnelId=tunnel_id)
