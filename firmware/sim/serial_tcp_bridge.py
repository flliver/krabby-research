"""Bridge host pyserial to simulated USB ``Serial`` via localhost TCP (all OSes).

Use when POSIX PTY is unavailable (native Windows). pyserial opens the client
side with ``serial.serial_for_url("socket://127.0.0.1:<port>")`` or set
``KRABBY_MCU_PORT`` to that URL.
"""

from __future__ import annotations

import select
import socket
import sys
import threading
from typing import TYPE_CHECKING

from firmware.sim.serial_usb_relay import relay_usb_serial_once

if TYPE_CHECKING:
    from firmware.sim.host_bridge import HostActuatorBridge


class SimMcuTcpBridge:
    """Expose ``Serial`` (USB) as a ``socket://`` URL pyserial can open."""

    def __init__(
        self,
        bridge: HostActuatorBridge,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self._bridge = bridge
        self._host = host
        self._port = port
        self.port_url: str = ""
        self._listen_sock: socket.socket | None = None
        self._client_sock: socket.socket | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> str:
        listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen.bind((self._host, self._port))
        listen.listen(1)
        listen.settimeout(0.5)
        self._listen_sock = listen
        bound_host, bound_port = listen.getsockname()
        self.port_url = f"socket://{bound_host}:{bound_port}"
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._relay_loop,
            name="sim-mcu-tcp",
            daemon=True,
        )
        self._thread.start()
        return self.port_url

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        for sock in (self._client_sock, self._listen_sock):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self._client_sock = None
        self._listen_sock = None
        self.port_url = ""

    def _accept_client(self) -> None:
        assert self._listen_sock is not None
        while not self._stop.is_set() and self._client_sock is None:
            try:
                client, _addr = self._listen_sock.accept()
            except (TimeoutError, socket.timeout):
                continue
            except OSError:
                break
            client.setblocking(True)
            self._client_sock = client

    def _relay_loop(self) -> None:
        self._accept_client()
        client = self._client_sock
        if client is None:
            return
        bridge = self._bridge
        while not self._stop.is_set():
            readable, _, _ = select.select([client], [], [], 0.05)
            try:
                if client in readable:
                    relay_usb_serial_once(
                        bridge,
                        read_client=lambda n: client.recv(n),
                        write_client=client.sendall,
                    )
                else:
                    relay_usb_serial_once(
                        bridge,
                        read_client=lambda _n: b"",
                        write_client=client.sendall,
                    )
            except OSError:
                break

    def __enter__(self) -> SimMcuTcpBridge:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def main() -> int:
    from firmware.sim.host_bridge import HostLibraryNotBuiltError, load_host_bridge

    try:
        bridge = load_host_bridge()
    except HostLibraryNotBuiltError as exc:
        print(exc, file=sys.stderr)
        return 1
    bridge.reset()
    bridge.firmware_setup()
    with SimMcuTcpBridge(bridge) as tcp:
        print(tcp.port_url, flush=True)
        print("Set KRABBY_MCU_PORT to the URL above (or pass port= to KrabbyMCUSDK).", flush=True)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
