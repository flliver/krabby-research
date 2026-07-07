"""Launch the serial<->TCP bridge on a remote bench host and tunnel to it.

Backs `python -m firmware.gui --remote krabby-orin`, replacing the manual
two-terminal dance (one `ssh -L ... 'python3 .../serial_tcp_bridge.py ...'`,
then the GUI in another window). A single ssh child process carries both the
port forward and the remote bridge, and its stdin pipe doubles as a deadman
switch: when this process exits — cleanly or by crash — the pipe closes, the
bridge sees stdin EOF and exits, and the remote serial port is freed for
flashing (`make -C firmware flash-remote`). The bridge also takes over from a
stale instance on its TCP port, so re-running the GUI after a `sync-remote`
always gets fresh bridge code with no manual `kill`.
"""
from __future__ import annotations

import re
import select
import socket
import subprocess
import sys
import threading
import time

DEFAULT_BRIDGE_PORT = 5331
DEFAULT_REMOTE_DIR = "~/krabby-fw-bench/firmware"  # firmware/Makefile REMOTE_DIR (sync-remote target)


class RemoteBridgeError(RuntimeError):
    """The ssh/bridge child failed to start or never became reachable."""


def free_local_port() -> int:
    """Ask the OS for an unused local TCP port (racy in theory, fine in practice)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class RemoteBridge:
    """One ssh child = tunnel + remote bridge, lifetime tied to this process."""

    # Probing the tunnel before the remote bridge listens makes ssh print a scary
    # (but harmless) "channel N: open failed: connect failed" — so the first probe
    # waits for the bridge's "listening on" banner. If the banner never comes (a
    # stale bridge whose log text differs), fall back to blind probing after this.
    PROBE_GRACE = 4.0

    def __init__(self, host: str, serial_dev: str = "/dev/ttyACM0",
                 bridge_port: int = DEFAULT_BRIDGE_PORT,
                 remote_dir: str = DEFAULT_REMOTE_DIR):
        self.host = host
        self.serial_dev = serial_dev
        self.bridge_port = bridge_port
        self.remote_dir = remote_dir
        self.local_port: int | None = None
        # The device the bridge actually opened — may differ from serial_dev: the
        # bridge's resolve_device() globs for any ttyUSB*/ttyACM* when the requested
        # path is absent (CH340 re-enumeration self-heal). Parsed from its
        # "[bridge] serial <dev>@<baud> open" log line; None until seen.
        self.resolved_serial: str | None = None
        self._proc: subprocess.Popen | None = None
        self._listening = threading.Event()

    def ssh_command(self, local_port: int) -> list[str]:
        # remote_dir may contain ~, which the remote shell expands — don't quote it.
        remote_cmd = (
            f"python3 -u {self.remote_dir}/tools/serial_tcp_bridge.py"
            f" --serial {self.serial_dev} --port {self.bridge_port} --exit-on-stdin-close"
        )
        return [
            "ssh",
            "-L", f"{local_port}:localhost:{self.bridge_port}",
            "-o", "BatchMode=yes",            # fail fast instead of hanging on a password prompt
            "-o", "ExitOnForwardFailure=yes",
            self.host,
            remote_cmd,
        ]

    def start(self, timeout: float = 15.0) -> str:
        """Spawn ssh+bridge; return the pyserial URL once the bridge answers."""
        self.local_port = free_local_port()
        # stdin must be a pipe we hold open: the bridge exits on stdin EOF, so the
        # remote side cannot outlive us. stdout is piped through _pump_stdout so we
        # can spot the bridge's "listening on" banner (readiness signal) while still
        # echoing [bridge] logs to the terminal; stderr inherits so ssh errors stay
        # visible next to the GUI's output.
        self._proc = subprocess.Popen(self.ssh_command(self.local_port),
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        threading.Thread(target=self._pump_stdout, args=(self._proc,), daemon=True).start()
        try:
            self._wait_ready(timeout)
        except Exception:
            self.stop()
            raise
        return f"socket://localhost:{self.local_port}"

    def _pump_stdout(self, proc: subprocess.Popen):
        """Echo the ssh child's stdout (the remote bridge's log lines) and flag
        readiness when the bridge announces it is accepting connections."""
        try:
            for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace")
                sys.stdout.write(line)
                sys.stdout.flush()
                if (m := re.search(r"\[bridge\] serial (\S+)@\d+ open", line)):
                    self.resolved_serial = m.group(1)
                if "listening on" in line:
                    self._listening.set()
        except ValueError:
            pass  # pipe closed during teardown

    def _wait_ready(self, timeout: float):
        started = time.monotonic()
        deadline = started + timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RemoteBridgeError(
                    f"ssh to {self.host} exited with status {self._proc.returncode} — see the"
                    " output above. If the bridge rejected an unknown argument, its code on"
                    f" the remote is stale: run `make -C firmware sync-remote REMOTE={self.host}`."
                )
            # Don't poke the tunnel until the bridge is listening: a premature probe
            # is refused at the remote end and ssh logs "channel N: open failed".
            ready_to_probe = (self._listening.is_set()
                              or time.monotonic() - started > self.PROBE_GRACE)
            if ready_to_probe and self._probe():
                return
            time.sleep(0.3)
        raise RemoteBridgeError(
            f"bridge on {self.host} did not answer on port {self.bridge_port} within"
            f" {timeout:.0f}s (is {self.serial_dev} present on the remote?)"
        )

    def _probe(self) -> bool:
        # ssh accepts on the local port as soon as it starts, whether or not the
        # remote bridge is listening yet — a dead remote end shows up as the socket
        # closing right away. So the readiness signal is "connected and stayed open
        # for a beat"; early bytes (telemetry already streaming) also mean success,
        # while a readable EOF means ssh couldn't reach the bridge.
        try:
            s = socket.create_connection(("127.0.0.1", self.local_port), timeout=1.0)
        except OSError:
            return False
        try:
            readable, _, _ = select.select([s], [], [], 0.5)
            if not readable:
                return True
            return s.recv(1, socket.MSG_PEEK) != b""
        except OSError:
            return False
        finally:
            s.close()

    def stop(self):
        """Close the deadman stdin (bridge exits remotely), then reap/kill ssh."""
        if not (proc := self._proc):
            return
        self._proc = None
        if proc.stdin:
            try:
                proc.stdin.close()
            except OSError:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
