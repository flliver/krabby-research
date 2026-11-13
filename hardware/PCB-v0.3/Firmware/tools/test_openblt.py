"""Destructive, single-board OpenBLT bench suite. Requires installed Krabby OpenBLT.

Erases/replaces the application over CAN, issues ST-Link resets, and leaves app A
running. Never installs/reflashes OpenBLT itself. Physical power cuts are NOT tested.
"""
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import time

import serial
from openblt_canable import APP_BASE, FLASH_END, MCU_FLASH_END, CONFIG_BASE, CONFIG_SIZE, ROOT, Canable, load_image, positive, update

from bench_tools import programmer_cli
IMAGES = ROOT / "bench/openblt"
BENCH = IMAGES / "config_page"
RESULTS = []


def record(name, **detail):
    RESULTS.append(dict(test=name, result="PASS", **detail))
    (BENCH / "results.json").write_text(json.dumps(RESULTS, indent=2)+"\n")
    print("PASS:", name, detail, flush=True)


def stlink(*args):
    result = subprocess.run([programmer_cli(), "-c", "port=SWD", "mode=HOTPLUG", *args],
                            capture_output=True, timeout=15)
    text = result.stdout.decode(errors="replace") + result.stderr.decode(errors="replace")
    if result.returncode or "Error:" in text:
        raise RuntimeError(text)
    return text


def reset():
    stlink("-rst")


def usb_status(expected):
    deadline = time.monotonic()+8
    last = ""
    while time.monotonic() < deadline:
        try:
            with serial.Serial("COM3", 115200, timeout=0.3, write_timeout=1) as port:
                port.reset_input_buffer()
                port.write(b"status\r")
                last = port.read(320).decode(errors="replace")
            if f"blink_ms={expected} " in last and "VTOR=08003000" in last and "node=0" in last:
                return last.strip()
        except serial.SerialException:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"Application USB identity not verified: {last!r}")


def rejected(adapter, packet, name):
    reply = adapter.request(packet)
    if not reply or reply[0] != 0xFE:
        raise RuntimeError(f"{name}: expected XCP error, got {reply!r}")
    record(name, response=reply.hex())


def main():
    BENCH.mkdir(parents=True, exist_ok=True)
    a = IMAGES / "app_250ms.srec"
    b = IMAGES / "app_125ms.srec"
    base, image = load_image(a)
    adapter = Canable()
    try:
        adapter.connect()
        config = adapter.read(CONFIG_BASE, CONFIG_SIZE)
        (BENCH / "config_before.bin").write_bytes(config)
        record("Configuration page baseline", sha256=hashlib.sha256(config).hexdigest(), non_erased=any(v != 255 for v in config))
        boot = adapter.read(0x08000000, APP_BASE-0x08000000)
        boot_hash = hashlib.sha256(boot).hexdigest()
        record("Connected to physical OpenBLT node 0", boot_sha256=boot_hash)

        def guard():
            adapter.connect()
            current = hashlib.sha256(adapter.read(0x08000000, len(boot))).hexdigest()
            if current != boot_hash:
                raise RuntimeError("Bootloader bytes changed")
            if adapter.read(CONFIG_BASE, CONFIG_SIZE) != config:
                raise RuntimeError("Persistent configuration page changed")

        for path, interval in ((a, 250), (b, 125), (a, 250)):
            update(adapter, path)
            status = usb_status(interval)
            record(f"CAN update and USB application identity {interval} ms", status=status)
            # The next connection must activate OpenBLT from the running application.
            guard()
            record("Application XCP re-entry and unchanged bootloader", version_ms=interval)

        # Non-destructive target-side negative requests, bypassing frontend validation.
        positive(adapter.request(b"\xd2"))
        set_boot = b"\xf6\0\0\0" + struct.pack("<I", 0x08000000)
        positive(adapter.request(set_boot))
        rejected(adapter, b"\xd1\0\0\0"+struct.pack("<I", 4096), "Reject bootloader erase")
        positive(adapter.request(set_boot))
        rejected(adapter, b"\xd0\x01\x00", "Reject bootloader write")
        rejected(adapter, b"\xf6\0\0\0"+struct.pack("<I", MCU_FLASH_END), "Reject out-of-range MTA")
        positive(adapter.request(b"\xf6\0\0\0"+struct.pack("<I", MCU_FLASH_END-1)))
        rejected(adapter, b"\xf5\x02", "Reject UPLOAD across flash end")
        rejected(adapter, b"\xd1\0\0\0"+struct.pack("<I", 2), "Reject erase across flash end")
        for addr, count in ((CONFIG_BASE, CONFIG_SIZE), (FLASH_END-1, 2)):
            positive(adapter.request(b"\xf6\0\0\0"+struct.pack("<I", addr)))
            rejected(adapter, b"\xd1\0\0\0"+struct.pack("<I", count), "Reject configuration/cross-boundary erase")
            positive(adapter.request(b"\xf6\0\0\0"+struct.pack("<I", addr)))
            rejected(adapter, b"\xd0\x02\x00\x00", "Reject configuration/cross-boundary write")
        for packet in (b"\xff", b"\xf6", b"\xd0\x05\x01", b"\xd1", b"\xc9\x01", b"\x01"):
            rejected(adapter, packet, "Reject malformed/unsupported "+packet.hex())
        adapter.send(b"")
        if adapter.receive(0.15) is not None:
            raise RuntimeError("Unexpected reply to empty CAN frame")
        adapter.connect()
        record("Empty CAN frame does not crash bootloader")
        adapter.send(b"\xff\0", identifier=0x668)
        if adapter.receive(0.2) is not None:
            raise RuntimeError("Node responded to another node's identifier")
        record("Wrong node identifier ignored")
        guard()
        record("Bootloader unchanged after negative requests")

        # Host loss without finalization, followed by a real MCU reset and CAN recovery.
        for offset in (0, 1024, len(image)//2, len(image)):
            result = update(adapter, a, start=False, interrupt_after=offset)
            if result is not False:
                raise RuntimeError("Interruption injector did not run")
            reset()
            time.sleep(2.3)  # invalid image must remain recoverable beyond entry window
            adapter.connect()
            if adapter.read(APP_BASE, 8) != b"\xff"*8:
                raise RuntimeError("Interrupted image acquired valid initial vectors")
            record("Interrupted transfer remains in bootloader after reset", offset=offset)
            update(adapter, b, start=False)
            guard()
            record("CAN-only recovery and complete readback", interrupted_offset=offset)

        # Trigger reset during an active transfer, not by disconnecting the bus.
        commands = 0
        fired = False
        def interrupt_transfer(packet):
            nonlocal commands, fired
            if packet[0] in (0xC9, 0xD0):
                commands += 1
            if commands == 50 and not fired:
                fired = True
                reset()
        try:
            update(adapter, a, start=False, fault=interrupt_transfer)
        except RuntimeError as error:
            if not fired:
                raise
            record("ST-Link reset during active CAN transfer detected", error=str(error))
        else:
            raise RuntimeError("Reset injection unexpectedly completed update")
        adapter.connect()
        update(adapter, a, start=False)
        guard()
        record("CAN-only recovery after reset during transfer")

        # Exercise all allowed application pages while preserving configuration.
        full = bytearray((i*37+11) & 255 for i in range(FLASH_END-APP_BASE))
        full[:len(image)] = image
        full_path = BENCH / "full_application_region.bin"
        full_path.write_bytes(full)
        update(adapter, full_path)
        record("Full 112 KiB application region programmed and read back over CAN",
               bytes=len(full), status=usb_status(250))
        guard()
        for index in range(3):
            update(adapter, b if index % 2 == 0 else a, start=False)
        record("Three additional alternating updates with full CAN readback")
        guard()
        update(adapter, a)
        record("Final application A restored over CAN", status=usb_status(250))

        guard()
        record("Configuration page preserved through all updates and resets", sha256=hashlib.sha256(config).hexdigest())

        adapter.request(b"\xcf", 0.1)
        usb_status(250)

        # Check physical MCU pin state, not just absence of motor-control calls.
        text = stlink("-r32", "0x48000414", "4", "-r32", "0x40012C44", "4")
        odr = int(re.search(r"0x48000414\s*:\s*([0-9A-Fa-f]+)", text)[1], 16)
        bdtr = int(re.search(r"0x40012C44\s*:\s*([0-9A-Fa-f]+)", text)[1], 16)
        if odr & ((1<<10)|(1<<14)|(1<<15)) or bdtr & (1<<15):
            raise RuntimeError("Motor-safe register checks failed")
        record("Motor sleeps LOW and TIM1 MOE disabled", GPIOB_ODR=hex(odr), TIM1_BDTR=hex(bdtr))
    finally:
        adapter.close()
    print(f"PASS: {len(RESULTS)} physical bootloader checks; see bench/openblt/config_page/results.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware', action='store_true', required=True,
                        help='Explicitly allow application replacement and ST-Link operations on the node-0 fixture')
    parser.parse_args()
    main()
