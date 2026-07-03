#!/usr/bin/env python3
"""Raw TCP <-> serial bridge: lets the GUI/SDK run on a different host than the MCU.

Preferred: let the GUI launch and manage this for you (tunnel + bridge + teardown):
    testenv/bin/python -m firmware.gui --remote krabby-orin

Manual: run on the bench (where the MCU is plugged in):
    python3 firmware/tools/serial_tcp_bridge.py --serial /dev/ttyUSB0 --port 5331
Then on the Mac (native, no X11):
    testenv/bin/python -m firmware.gui --port socket://<bench-ip>:5331

A starting bridge takes over from a previous instance on the same TCP port (pidfile
under /tmp), so you never have to hunt down and `kill` a stale one by hand. With
--exit-on-stdin-close it also exits as soon as stdin hits EOF — the GUI's --remote
mode holds stdin open over ssh, so closing the GUI (or losing the ssh connection)
frees the serial port for avrdude/flash-remote.

pyserial's `socket://host:port` URL is a *raw* TCP transport (not RFC2217), so this
bridge just shuttles bytes in both directions. The real serial port stays open across
client reconnects, so the board isn't reset every time the GUI reconnects.

The USB serial adapter (CH340) can re-enumerate under load (e.g. /dev/ttyUSB0 ->
/dev/ttyUSB1), which kills the original file descriptor with EIO. The bridge resolves
the device by glob and reopens it on any serial error, so a re-enumeration self-heals.
"""
import argparse
import errno
import glob
import os
import selectors
import signal
import socket
import sys
import threading
import time

import serial


def resolve_device(want):
    """Prefer the requested path; if it's gone (re-enumeration), take the first ttyUSB/ttyACM."""
    if os.path.exists(want):
        return want
    cands = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    return cands[0] if cands else want


def open_serial(want, baud):
    dev = resolve_device(want)
    ser = serial.Serial()
    ser.port = dev
    ser.baudrate = baud
    ser.timeout = 0
    ser.dtr = False  # avoid resetting the board on open
    ser.open()
    print(f"[bridge] serial {dev}@{baud} open", flush=True)
    return ser


def watch_stdin_and_exit():
    """Exit the whole process when stdin hits EOF (--exit-on-stdin-close).

    The GUI's --remote launcher holds our stdin open through ssh; EOF means the
    launcher is gone (clean close OR crash), so release the serial port and die.
    os._exit skips cleanup on purpose — the OS closes the serial fd and the
    listening socket, and there is nothing else to unwind.
    """
    try:
        while sys.stdin.buffer.read(4096):
            pass
    except OSError:
        pass
    print("[bridge] stdin closed — exiting", flush=True)
    os._exit(0)


def takeover_pidfile(port, piddir="/tmp"):
    """Kill any previous bridge on this TCP port and record ourselves in its place.

    Keyed by TCP port so intentionally-parallel bridges on different ports coexist.
    A stale entry (pid already dead, or not ours to signal) is simply overwritten.
    """
    pidfile = os.path.join(piddir, f"krabby-serial-bridge-{port}.pid")
    try:
        with open(pidfile) as f:
            stale = int(f.read().strip())
    except (OSError, ValueError):
        stale = None
    if stale and stale != os.getpid():
        try:
            os.kill(stale, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            stale = None
        if stale:
            print(f"[bridge] taking over from previous bridge pid {stale}", flush=True)
            for _ in range(30):  # give it up to ~3s to release the port + serial device
                try:
                    os.kill(stale, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.1)
            else:
                try:
                    os.kill(stale, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    with open(pidfile, "w") as f:
        f.write(str(os.getpid()))
    return pidfile


def find_port_listener(port, proc_root="/proc"):
    """Linux-only: pid of whatever is LISTENing on TCP `port`, or None.

    Covers holders the pidfile can't see (e.g. a bridge started before pidfiles
    existed, or one launched by hand). Maps /proc/net/tcp{,6} LISTEN rows to a
    socket inode, then scans /proc/*/fd for the process holding that inode —
    no lsof/fuser/psutil needed on the bench host.
    """
    inodes = set()
    for name in ("net/tcp", "net/tcp6"):
        try:
            with open(os.path.join(proc_root, name)) as f:
                rows = f.readlines()[1:]
        except OSError:
            continue
        for row in rows:
            parts = row.split()
            if len(parts) < 10:
                continue
            local_addr, state, inode = parts[1], parts[3], parts[9]
            if state == "0A" and int(local_addr.rsplit(":", 1)[1], 16) == port:
                inodes.add(f"socket:[{inode}]")
    if not inodes:
        return None
    for pid in os.listdir(proc_root):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        fd_dir = os.path.join(proc_root, pid, "fd")
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                if os.readlink(os.path.join(fd_dir, fd)) in inodes:
                    return int(pid)
            except OSError:
                continue
    return None


def bind_or_evict(srv, port, attempts=10, wait=0.5):
    """bind(), evicting whatever holds the port if the first try fails.

    The pidfile takeover handles well-behaved predecessors; this is the net for
    everyone else. SIGTERM first, SIGKILL on the later attempts.
    """
    for attempt in range(attempts):
        try:
            srv.bind(("0.0.0.0", port))
            return
        except OSError as e:
            if e.errno != errno.EADDRINUSE:
                raise
        if holder := find_port_listener(port):
            sig = signal.SIGTERM if attempt < attempts // 2 else signal.SIGKILL
            print(f"[bridge] port {port} held by pid {holder} — sending {sig.name}", flush=True)
            try:
                os.kill(holder, sig)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(wait)
    raise OSError(f"port {port} still in use after {attempts} eviction attempts; "
                  f"free it manually (fuser -k {port}/tcp)")


def main():
    ap = argparse.ArgumentParser(description="Raw TCP<->serial bridge")
    ap.add_argument("--serial", default="/dev/ttyUSB0", help="serial device of the MCU")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--port", type=int, default=5331, help="TCP port to listen on")
    ap.add_argument("--exit-on-stdin-close", action="store_true",
                    help="exit when stdin hits EOF (used by `firmware.gui --remote`; "
                         "do NOT combine with nohup, whose /dev/null stdin is instant EOF)")
    a = ap.parse_args()

    takeover_pidfile(a.port)
    if a.exit_on_stdin_close:
        threading.Thread(target=watch_stdin_and_exit, daemon=True).start()

    # Claim the TCP port before the serial device: evicting a previous bridge
    # here also makes it release the serial port before we open it.
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    bind_or_evict(srv, a.port)
    srv.listen(1)

    ser = open_serial(a.serial, a.baud)
    print(f"[bridge] listening on 0.0.0.0:{a.port} (one client at a time)", flush=True)

    while True:
        cli, addr = srv.accept()
        print(f"[bridge] client {addr} connected", flush=True)
        # Keep the socket BLOCKING — select() tells us when it's readable, so recv()
        # won't block; a non-blocking socket would instead raise EAGAIN (Errno 11) on
        # recv/sendall, which we'd misread as a disconnect.
        sel = selectors.DefaultSelector()
        sel.register(cli, selectors.EVENT_READ, "sock")
        sel.register(ser.fileno(), selectors.EVENT_READ, "ser")
        reopen_serial = False
        try:
            while True:
                for key, _ in sel.select(timeout=1.0):
                    if key.data == "sock":
                        data = cli.recv(4096)
                        if not data:
                            raise ConnectionError("client closed")
                        try:
                            ser.write(data)
                        except (serial.SerialException, OSError):
                            reopen_serial = True
                            raise
                    else:
                        try:
                            n = ser.in_waiting
                            chunk = ser.read(n) if n else b""
                        except (serial.SerialException, OSError):
                            reopen_serial = True
                            raise
                        if chunk:
                            cli.sendall(chunk)
        except Exception as e:
            print(f"[bridge] client {addr} gone ({e})", flush=True)
        finally:
            sel.close()
            try:
                cli.close()
            except OSError:
                pass

        if reopen_serial:
            print("[bridge] serial error — reopening device", flush=True)
            try:
                ser.close()
            except OSError:
                pass
            while True:
                try:
                    ser = open_serial(a.serial, a.baud)
                    break
                except Exception as oe:  # device not back yet
                    print(f"[bridge] reopen failed ({oe}); retry in 1s", flush=True)
                    time.sleep(1)


if __name__ == "__main__":
    main()
