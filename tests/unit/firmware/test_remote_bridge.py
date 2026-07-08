"""Tests for the ssh-managed remote bridge path of the bench GUI.

Covers both halves of `python -m firmware.gui --remote <host>`:
- firmware/gui/remote.py — the launcher that spawns one ssh child carrying the
  port forward and the bridge, probes readiness, and tears it down.
- firmware/tools/serial_tcp_bridge.py — the bridge-side pidfile takeover (no
  more hunting stale bridges with `kill`) and the stdin deadman switch.

The bridge script is not an installed package (it runs by path on the bench
host), so it is loaded here by file path.
"""
import importlib.util
import io
import os
import re
import socket
import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest

import firmware.gui.remote as remote
from firmware.gui.__main__ import build_parser
from firmware.gui.remote import (
    DEFAULT_REMOTE_DIR,
    RemoteBridge,
    RemoteBridgeError,
    free_local_port,
)

FIRMWARE_DIR = Path(__file__).resolve().parents[3] / "firmware"
BRIDGE_PATH = FIRMWARE_DIR / "tools" / "serial_tcp_bridge.py"


@pytest.fixture(scope="module")
def bridge_mod():
    spec = importlib.util.spec_from_file_location("serial_tcp_bridge", BRIDGE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- launcher

def test_ssh_command_shape():
    rb = RemoteBridge("krabby-orin", serial_dev="/dev/ttyACM1", bridge_port=6000,
                      remote_dir="~/somewhere/firmware")
    cmd = rb.ssh_command(12345)
    assert cmd[0] == "ssh"
    assert "12345:localhost:6000" in cmd
    assert "ExitOnForwardFailure=yes" in cmd
    assert "BatchMode=yes" in cmd
    assert "krabby-orin" in cmd
    remote_cmd = cmd[-1]
    assert "~/somewhere/firmware/tools/serial_tcp_bridge.py" in remote_cmd
    assert "--serial /dev/ttyACM1" in remote_cmd
    assert "--port 6000" in remote_cmd
    assert "--exit-on-stdin-close" in remote_cmd


def test_free_local_port_is_bindable():
    port = free_local_port()
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))  # must be free right after


class StubProc:
    def __init__(self, returncode=None, stdout=b""):
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(stdout)
        self.returncode = returncode
        self.wait_calls = []

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        return self.returncode or 0


def test_stop_closes_deadman_stdin_and_reaps():
    rb = RemoteBridge("host")
    proc = StubProc()
    rb._proc = proc
    rb.stop()
    assert proc.stdin.closed
    assert proc.wait_calls
    rb.stop()  # second stop is a no-op
    assert len(proc.wait_calls) == 1


def test_start_reports_ssh_death_with_sync_remote_hint(monkeypatch):
    proc = StubProc(returncode=255)
    monkeypatch.setattr(remote.subprocess, "Popen", lambda *a, **k: proc)
    rb = RemoteBridge("krabby-orin")
    with pytest.raises(RemoteBridgeError, match="sync-remote"):
        rb.start(timeout=5.0)
    assert proc.stdin.closed  # failure path still tears down


def test_start_times_out_when_bridge_never_answers(monkeypatch):
    proc = StubProc(returncode=None)
    monkeypatch.setattr(remote.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(RemoteBridge, "_probe", lambda self: False)
    rb = RemoteBridge("krabby-orin")
    with pytest.raises(RemoteBridgeError, match="did not answer"):
        rb.start(timeout=0.5)
    assert proc.stdin.closed


def test_probe_deferred_until_listening_banner(monkeypatch):
    # A probe before the remote bridge listens is refused at the remote end and
    # makes ssh print "channel N: open failed: connect failed" — so no probe may
    # fire until the bridge's "listening on" banner arrives (or the grace expires).
    proc = StubProc(returncode=None)  # no banner ever
    monkeypatch.setattr(remote.subprocess, "Popen", lambda *a, **k: proc)
    probes = []
    monkeypatch.setattr(RemoteBridge, "_probe", lambda self: probes.append(1) or True)
    monkeypatch.setattr(RemoteBridge, "PROBE_GRACE", 60.0)
    rb = RemoteBridge("krabby-orin")
    with pytest.raises(RemoteBridgeError, match="did not answer"):
        rb.start(timeout=0.8)
    assert not probes, "probed the tunnel before the bridge announced readiness"


def test_probe_fires_once_banner_seen(monkeypatch, capsys):
    proc = StubProc(returncode=None,
                    stdout=b"[bridge] serial /dev/ttyUSB0@115200 open\n"
                           b"[bridge] listening on 0.0.0.0:5331 (one client at a time)\n")
    monkeypatch.setattr(remote.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(RemoteBridge, "_probe", lambda self: True)
    monkeypatch.setattr(RemoteBridge, "PROBE_GRACE", 60.0)
    rb = RemoteBridge("krabby-orin")
    assert rb.start(timeout=5.0).startswith("socket://localhost:")
    # bridge log lines are still echoed to the terminal
    assert "listening on 0.0.0.0:5331" in capsys.readouterr().out
    # the device the bridge actually opened is captured (it may glob a fallback
    # differing from the requested serial_dev)
    assert rb.resolved_serial == "/dev/ttyUSB0"


def test_probe_grace_fallback_for_stale_bridge_banner(monkeypatch):
    # A stale remote bridge whose log text differs must still come up: after
    # PROBE_GRACE with no banner, blind probing resumes (old behavior).
    proc = StubProc(returncode=None, stdout=b"[bridge] some unrecognized banner\n")
    monkeypatch.setattr(remote.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(RemoteBridge, "_probe", lambda self: True)
    monkeypatch.setattr(RemoteBridge, "PROBE_GRACE", 0.0)
    rb = RemoteBridge("krabby-orin")
    assert rb.start(timeout=5.0).startswith("socket://localhost:")


def test_probe_true_when_connection_is_held_open():
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        held = []
        threading.Thread(target=lambda: held.append(srv.accept()), daemon=True).start()
        rb = RemoteBridge("host")
        rb.local_port = srv.getsockname()[1]
        assert rb._probe() is True


def test_probe_false_when_upstream_closes_immediately():
    # Mimics ssh accepting locally but failing to reach the remote bridge: the
    # local connection gets torn down right after connect.
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)

        def accept_and_close():
            conn, _ = srv.accept()
            conn.close()

        threading.Thread(target=accept_and_close, daemon=True).start()
        rb = RemoteBridge("host")
        rb.local_port = srv.getsockname()[1]
        assert rb._probe() is False


def test_probe_false_when_nothing_listens():
    rb = RemoteBridge("host")
    rb.local_port = free_local_port()
    assert rb._probe() is False


# ---------------------------------------------------------------- GUI parser

def test_gui_parser_remote_defaults():
    args = build_parser().parse_args(["--remote", "krabby-orin"])
    assert args.remote == "krabby-orin"
    assert args.serial == "/dev/ttyACM0"
    assert args.bridge_port == 5331
    assert args.remote_dir == "~/krabby-fw-bench/firmware"
    assert args.port is None


def test_makefile_remote_dir_matches_launcher_default():
    # sync-remote pushes to Makefile REMOTE_DIR; --remote launches the bridge from
    # DEFAULT_REMOTE_DIR. If they diverge, every launch runs stale bridge code —
    # the exact failure RemoteBridgeError's sync-remote hint warns about.
    makefile = (FIRMWARE_DIR / "Makefile").read_text()
    m = re.search(r"^REMOTE_DIR\s*\?=\s*(\S+)", makefile, re.MULTILINE)
    assert m, "REMOTE_DIR not found in firmware/Makefile"
    assert m.group(1).rstrip("/") == DEFAULT_REMOTE_DIR.rstrip("/")


# ---------------------------------------------------------------- bridge side

def test_takeover_kills_stale_bridge(bridge_mod, tmp_path):
    stale = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        pidfile = tmp_path / "krabby-serial-bridge-5331.pid"
        pidfile.write_text(str(stale.pid))
        bridge_mod.takeover_pidfile(5331, piddir=str(tmp_path))
        stale.wait(timeout=5)  # SIGTERM'd by the takeover
        assert stale.poll() is not None
        assert pidfile.read_text() == str(os.getpid())
    finally:
        if stale.poll() is None:
            stale.kill()


def test_takeover_tolerates_garbage_and_dead_pids(bridge_mod, tmp_path):
    pidfile = tmp_path / "krabby-serial-bridge-5331.pid"
    pidfile.write_text("not-a-pid")
    bridge_mod.takeover_pidfile(5331, piddir=str(tmp_path))
    assert pidfile.read_text() == str(os.getpid())

    # A pid that no longer exists: reuse a just-reaped child's pid.
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait(timeout=10)
    pidfile.write_text(str(dead.pid))
    bridge_mod.takeover_pidfile(5331, piddir=str(tmp_path))
    assert pidfile.read_text() == str(os.getpid())


def _fake_proc(tmp_path, port, inode, holder_pid):
    """Minimal /proc tree: one LISTEN row on `port`, one process holding its socket inode."""
    net = tmp_path / "net"
    net.mkdir()
    hexport = f"{port:04X}"
    (net / "tcp").write_text(
        "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt"
        "   uid  timeout inode\n"
        f"   0: 00000000:{hexport} 00000000:0000 0A 00000000:00000000 00:00000000"
        f" 00000000  1000        0 {inode} 1 0000000000000000 100 0 0 10 0\n"
    )
    fd_dir = tmp_path / str(holder_pid) / "fd"
    fd_dir.mkdir(parents=True)
    os.symlink(f"socket:[{inode}]", fd_dir / "3")  # dangling target, readlink still works


def test_find_port_listener_maps_port_to_pid(bridge_mod, tmp_path):
    _fake_proc(tmp_path, port=5331, inode=99999, holder_pid=4242)
    assert bridge_mod.find_port_listener(5331, proc_root=str(tmp_path)) == 4242
    assert bridge_mod.find_port_listener(6000, proc_root=str(tmp_path)) is None


def test_find_port_listener_no_proc_is_none(bridge_mod, tmp_path):
    assert bridge_mod.find_port_listener(5331, proc_root=str(tmp_path / "nope")) is None


def test_bind_or_evict_binds_free_port(bridge_mod):
    port = free_local_port()
    with socket.socket() as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        bridge_mod.bind_or_evict(srv, port, attempts=2, wait=0.01)
        assert srv.getsockname()[1] == port


def test_bind_or_evict_gives_up_on_unkillable_holder(bridge_mod):
    # Holder is this very process, which find_port_listener refuses to target
    # (and /proc is absent on macOS anyway) — so eviction must fail loudly.
    with socket.socket() as holder:
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind(("0.0.0.0", 0))
        holder.listen(1)
        port = holder.getsockname()[1]
        with socket.socket() as srv, pytest.raises(OSError, match="still in use"):
            bridge_mod.bind_or_evict(srv, port, attempts=2, wait=0.01)


@pytest.mark.skipif(sys.platform != "linux", reason="needs a real /proc")
def test_bind_or_evict_evicts_foreign_holder(bridge_mod):
    port = free_local_port()
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import socket,sys,time; s=socket.socket();"
         "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1);"
         f"s.bind(('0.0.0.0', {port})); s.listen(1); print('ready', flush=True);"
         "time.sleep(60)"],
        stdout=subprocess.PIPE,
    )
    try:
        assert holder.stdout.readline().strip() == b"ready"
        with socket.socket() as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            bridge_mod.bind_or_evict(srv, port, attempts=10, wait=0.3)
            assert srv.getsockname()[1] == port
        holder.wait(timeout=5)  # evicted
    finally:
        if holder.poll() is None:
            holder.kill()


def test_watch_stdin_exits_on_eof(bridge_mod, monkeypatch):
    monkeypatch.setattr(bridge_mod.sys, "stdin",
                        types.SimpleNamespace(buffer=io.BytesIO(b"held open for a while")))
    exit_codes = []
    monkeypatch.setattr(bridge_mod.os, "_exit", lambda code: exit_codes.append(code))
    bridge_mod.watch_stdin_and_exit()
    assert exit_codes == [0]
