#!/usr/bin/env python3
"""Bench test: Krabby USB CDC <-> normaldotcom CANable2.

Install: python -m pip install pyserial
Run:     python tools/test_can_bridge.py --stm32 COM3 --adapter COM7
Close serial terminals first. Requires a powered transceiver and terminated bus.
Tests are sequential; this does not stress the receive queue or test overflow.
"""

import argparse
from contextlib import ExitStack
import sys
import time

import serial


LENGTHS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64)


class Terminal:
    def __init__(self, port, verbose):
        self.port = port
        self.verbose = verbose
        self.pending = bytearray()

    def send(self, command):
        if self.verbose:
            print(f"  {self.port.port} > {command}")
        self.port.write((command + "\r").encode("ascii"))

    def expect(self, predicate, timeout):
        deadline = time.monotonic() + timeout
        seen = []
        while time.monotonic() < deadline:
            # Read one byte so returning a match leaves following lines intact.
            chunk = self.port.read(1)
            for byte in chunk:
                if byte in (10, 13):
                    if not self.pending:
                        continue
                    line = self.pending.decode("ascii", errors="replace")
                    self.pending.clear()
                    seen.append(line)
                    if self.verbose:
                        print(f"  {self.port.port} < {line}")
                    # Never accept a good frame while silently ignoring overflow.
                    if line.startswith(("ERR", "CAN overflow:")):
                        raise RuntimeError(f"{self.port.port}: {line}")
                    if predicate(line):
                        return line
                    if line.startswith("RX "):
                        raise RuntimeError(f"Unexpected CAN frame on {self.port.port}: {line}")
                else:
                    self.pending.append(byte)
        raise RuntimeError(
            f"Timeout on {self.port.port}; received {seen!r}; partial={bytes(self.pending)!r}"
        )


def status(stm32, timeout):
    stm32.send("can status")
    first = stm32.expect(lambda line: line.startswith("CAN clk="), timeout)
    second = stm32.expect(lambda line: line.startswith("irq="), timeout)
    third = stm32.expect(lambda line: line.startswith("CCCR="), timeout)
    print(f"  {first}\n  {second}\n  {third}")
    fields = dict(item.split("=", 1) for item in (first + " " + second + " " + third).split() if "=" in item)
    return fields


def verify_timing(fields, nominal, data):
    clock = int(fields["clk"])
    n = int(fields["NBTP"], 16)
    d = int(fields["DBTP"], 16)
    n_div = (((n >> 16) & 511) + 1) * (1 + ((n >> 8) & 255) + 1 + (n & 127) + 1)
    d_prescaler = ((d >> 16) & 31) + 1
    d_div = d_prescaler * (1 + ((d >> 8) & 31) + 1 + ((d >> 4) & 15) + 1)
    if clock != nominal * n_div or clock != data * d_div:
        raise RuntimeError(f"Firmware bitrate mismatch: {clock/n_div:g}/{clock/d_div:g}; "
                           f"requested {nominal}/{data}. Flash matching firmware first.")
    if d & (1 << 23) and d_prescaler > 2:
        raise RuntimeError("Invalid firmware timing: TDC enabled with data prescaler > 2")


def require_clean(fields, label):
    if any(fields.get(key) != "0" for key in ("INIT", "BO", "EP", "TEC", "REC")):
        raise RuntimeError(f"CAN error counters/state after {label}")
    # PEA/PED are sticky: catch errors even if retries reduce TEC back to zero.
    if int(fields["IR"], 16) & 0x00600000:
        raise RuntimeError(f"Latched CAN protocol error after {label}")


def run(args):
    if args.stm32.casefold() == args.adapter.casefold():
        raise ValueError("STM32 and adapter ports must be different")
    with ExitStack() as stack:
        def connect(name):
            port = stack.enter_context(serial.Serial(
                name, 115200, timeout=0.05, write_timeout=1,
            ))
            port.dtr = True
            return Terminal(port, args.verbose)

        stm32 = connect(args.stm32)
        adapter = connect(args.adapter)
        try:
            time.sleep(0.3)
            for terminal in (stm32, adapter):
                terminal.port.reset_input_buffer()
            # Configuration commands have no ACK in this adapter firmware.
            nominal_command = {500000: "S6", 1000000: "S8"}[args.nominal]
            data_command = {2000000: "Y2", 5000000: "Y5"}[args.data]
            print(f"Requested nominal={args.nominal} data={args.data}; repeat={args.repeat}")
            for command in ("C", nominal_command, data_command, "M0", "A1"):
                adapter.send(command)
                time.sleep(0.15)
            adapter.send("V")
            version = adapter.expect(lambda line: bool(line), args.timeout)
            print(f"Adapter: {version}")
            stm32.send("hello")
            stm32.expect(lambda line: line.startswith("Hello from Krabby v0.3"), args.timeout)
            print("Initial STM32 status:")
            initial = status(stm32, args.timeout)
            verify_timing(initial, args.nominal, args.data)
            require_clean(initial, "startup; reset STM32 before a new run")
            adapter.send("O")
            time.sleep(0.15)

            passed = 0
            identifier = 0x200
            for iteration in range(args.repeat):
                print(f"Round {iteration + 1}/{args.repeat}")
                for mode, slcan, command, sizes in (
                    ("CAN", "t", "tx", (0, 4, 8)),
                    ("FD+BRS", "b", "txfd", LENGTHS),
                    ("FD", "d", "txfdn", LENGTHS),
                ):
                    if args.mode != "all" and slcan != {"classic": "t", "fd": "d", "brs": "b"}[args.mode]:
                        continue
                    for extended in (False, True):
                        for length in sizes:
                            identifier = 0x200 + (identifier - 0x200 + 1) % 0x500
                            # Keep diagnostic traffic away from application and
                            # OpenBLT service identifiers as the repeat count grows.
                            while any(base <= identifier < base + 8 for base in (0x500, 0x580, 0x600, 0x667)):
                                identifier = 0x200 + (identifier - 0x200 + 1) % 0x500
                            can_id = identifier | (0x1000000 if extended else 0)
                            payload = bytes((i + iteration) % 256 for i in range(length))
                            dlc = LENGTHS.index(length)
                            prefix = slcan.upper() if extended else slcan
                            id_text = f"{can_id:08X}" if extended else f"{can_id:03X}"
                            wire = f"{prefix}{id_text}{dlc:X}{payload.hex().upper()}"
                            expected = (
                                f"RX {'EXT' if extended else 'STD'} {can_id:08X} "
                                f"{'CAN' if mode == 'CAN' else 'FD'} DATA "
                                f"BRS={int(mode == 'FD+BRS')} DLC={dlc:X} len={length}"
                                + "".join(f" {byte:02X}" for byte in payload)
                            )
                            label = f"{mode} {'EXT' if extended else 'STD'} len={length}"
                            print(f"Testing {label}", flush=True)
                            adapter.send(wire)
                            stm32.expect(lambda line: line == expected, args.timeout)
                            passed += 1
                            print("  PASS adapter -> STM32")
                            if args.check_status:
                                check_counters(stm32, args.timeout, f"adapter -> STM32 {label}")

                            tx_command = command + ("x" if extended else "")
                            stm32.send(f"{tx_command} {can_id:X}" + "".join(
                                f" {byte:02X}" for byte in payload
                            ))
                            stm32.expect(lambda line: line.startswith("OK TX queued"), args.timeout)
                            # Only the independently received adapter frame counts as delivery.
                            adapter.expect(lambda line: line == wire, args.timeout)
                            passed += 1
                            print("  PASS STM32 -> adapter")
                            if args.check_status:
                                check_counters(stm32, args.timeout, f"STM32 -> adapter {label}")

            print("Final STM32 status:")
            final = status(stm32, args.timeout)
            for field in ("dropped", "hwloss", "rxerr"):
                if final.get(field) != initial.get(field):
                    raise RuntimeError(f"{field} increased during testing")
            require_clean(final, "final check")
            print(f"PASS: {passed} frame deliveries verified")
        except Exception:
            print("Test failed. STM32 diagnostic snapshot:", file=sys.stderr)
            try:
                status(stm32, args.timeout)
            except Exception as error:
                print(f"Could not read status: {error}", file=sys.stderr)
            raise
        finally:
            # Close CAN before closing serial handles, including on Ctrl+C.
            try:
                adapter.send("C")
                time.sleep(0.1)
            except serial.SerialException:
                pass


def check_counters(stm32, timeout, label):
    fields = status(stm32, timeout)
    require_clean(fields, label)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stm32", default="COM3")
    parser.add_argument("--adapter", default="COM7")
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--nominal", type=int, choices=(500000, 1000000), default=500000)
    parser.add_argument("--data", type=int, choices=(2000000, 5000000), default=2000000)
    parser.add_argument("--repeat", type=int, default=1, help="Repeat matrix with varying payloads")
    parser.add_argument("--mode", choices=("all", "classic", "fd", "brs"), default="all",
                        help="Test classic CAN, FD without BRS, FD with BRS, or all")
    parser.add_argument("--check-status", action="store_true",
                        help="Check CAN error counters after every delivered frame; fail on any error")
    args = parser.parse_args()
    if args.timeout <= 0 or args.repeat <= 0:
        parser.error("--timeout and --repeat must be positive")
    try:
        run(args)
    except KeyboardInterrupt:
        print("Stopped by user.", file=sys.stderr)
        return 130
    except (serial.SerialException, RuntimeError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
