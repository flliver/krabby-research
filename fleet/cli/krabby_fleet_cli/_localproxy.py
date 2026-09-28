"""Source-mode localproxy: listens locally, forwards through an open Secure Tunnel."""
from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import time

_LOCALPROXY_BIN = "localproxy"
_READY_TIMEOUT_SECS = 10.0


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def spawn_source_proxy(source_access_token: str, region: str, local_port: int) -> subprocess.Popen:
    if not shutil.which(_LOCALPROXY_BIN):
        print(
            f"error: {_LOCALPROXY_BIN} not installed -- install aws-iot-securetunneling-localproxy",
            file=sys.stderr,
        )
        sys.exit(1)
    return subprocess.Popen(
        [
            _LOCALPROXY_BIN,
            "-s",
            str(local_port),
            "-b",
            "127.0.0.1",
            "-t",
            source_access_token,
            "-r",
            region,
            "-c",
            "/etc/ssl/certs",
        ]
    )


def wait_until_ready(
    local_port: int, proc: subprocess.Popen, timeout: float = _READY_TIMEOUT_SECS
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            print(f"error: localproxy exited early (code {proc.returncode})", file=sys.stderr)
            sys.exit(1)
        try:
            with socket.create_connection(("127.0.0.1", local_port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    print("error: timed out waiting for localproxy to start listening", file=sys.stderr)
    proc.terminate()
    sys.exit(1)
