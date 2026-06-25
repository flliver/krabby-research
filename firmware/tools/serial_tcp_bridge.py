#!/usr/bin/env python3
"""Raw TCP <-> serial bridge: lets the GUI/SDK run on a different host than the MCU.

Run on the bench (where the MCU is plugged in):
    python3 firmware/tools/serial_tcp_bridge.py --serial /dev/ttyUSB0 --port 5331
Then on the Mac (native, no X11):
    testenv/bin/python -m firmware.gui --port socket://<bench-ip>:5331

pyserial's `socket://host:port` URL is a *raw* TCP transport (not RFC2217), so this
bridge just shuttles bytes in both directions. The real serial port stays open across
client reconnects, so the board isn't reset every time the GUI reconnects.
"""
import argparse
import selectors
import socket

import serial


def main():
    ap = argparse.ArgumentParser(description="Raw TCP<->serial bridge")
    ap.add_argument("--serial", default="/dev/ttyUSB0", help="serial device of the MCU")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--port", type=int, default=5331, help="TCP port to listen on")
    a = ap.parse_args()

    ser = serial.Serial()
    ser.port = a.serial
    ser.baudrate = a.baud
    ser.timeout = 0
    ser.dtr = False  # avoid resetting the board on open
    ser.open()
    print(f"[bridge] serial {a.serial}@{a.baud} open", flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", a.port))
    srv.listen(1)
    print(f"[bridge] listening on 0.0.0.0:{a.port} (one client at a time)", flush=True)

    while True:
        cli, addr = srv.accept()
        print(f"[bridge] client {addr} connected", flush=True)
        # Keep the socket BLOCKING — select() tells us when it's readable, so recv()
        # won't block; a non-blocking socket would instead raise EAGAIN (Errno 11) on
        # recv/sendall, which our except below would misread as a disconnect.
        sel = selectors.DefaultSelector()
        sel.register(cli, selectors.EVENT_READ, "sock")
        sel.register(ser.fileno(), selectors.EVENT_READ, "ser")
        try:
            while True:
                for key, _ in sel.select(timeout=1.0):
                    if key.data == "sock":
                        data = cli.recv(4096)
                        if not data:
                            raise ConnectionError("client closed")
                        ser.write(data)
                    else:
                        n = ser.in_waiting
                        if n:
                            cli.sendall(ser.read(n))
        except (ConnectionError, OSError) as e:
            print(f"[bridge] client {addr} gone ({e})", flush=True)
        finally:
            sel.close()
            cli.close()


if __name__ == "__main__":
    main()
