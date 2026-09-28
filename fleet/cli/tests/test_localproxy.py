"""Source-mode localproxy helpers: no real localproxy binary needed."""
from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import pytest

from krabby_fleet_cli._localproxy import free_local_port, spawn_source_proxy, wait_until_ready


def test_free_local_port_is_bindable():
    port = free_local_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port))  # raises OSError if the port weren't actually free


def test_spawn_source_proxy_missing_binary_exits():
    with patch("krabby_fleet_cli._localproxy.shutil.which", return_value=None):
        with pytest.raises(SystemExit):
            spawn_source_proxy("token", "us-east-1", 12345)


def test_spawn_source_proxy_invokes_with_source_flag():
    fake_proc = MagicMock()
    with patch("krabby_fleet_cli._localproxy.shutil.which", return_value="/usr/bin/localproxy"), \
         patch("krabby_fleet_cli._localproxy.subprocess.Popen", return_value=fake_proc) as popen:
        result = spawn_source_proxy("src-token", "us-east-1", 54321)

    assert result is fake_proc
    args, _ = popen.call_args
    assert args[0] == [
        "localproxy",
        "-s",
        "54321",
        "-b",
        "127.0.0.1",
        "-t",
        "src-token",
        "-r",
        "us-east-1",
        "-c",
        "/etc/ssl/certs",
    ]


def test_wait_until_ready_succeeds_once_port_is_listening():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        fake_proc = MagicMock()
        fake_proc.poll.return_value = None  # still running
        wait_until_ready(port, fake_proc, timeout=2.0)  # does not raise
    finally:
        server.close()


def test_wait_until_ready_exits_if_process_dies_early():
    fake_proc = MagicMock()
    fake_proc.poll.return_value = 1  # already exited
    fake_proc.returncode = 1
    with pytest.raises(SystemExit):
        wait_until_ready(54322, fake_proc, timeout=2.0)
