"""Windows OpenBLT frontend: official LibOpenBLT -> localhost XCP -> CANable2.

Requires 64-bit Python and pyserial. No adapter reflash or custom target protocol.
The upstream TCP transport has a 32-bit counter followed by an XCP packet, without
a length field. The relay uses the upstream loader's command lengths, reads exact
stream lengths, and rejects unsupported opcodes rather than guessing boundaries.
"""
import argparse
import ctypes as C
from pathlib import Path
import socket
import struct
import threading
import time

import serial

ROOT = Path(__file__).resolve().parents[1]
APP_BASE = 0x08003000
MCU_FLASH_END = 0x08020000
CONFIG_BASE = 0x0801F000
CONFIG_SIZE = 0x1000
FLASH_END = CONFIG_BASE  # Exclusive application limit; configuration is never uploaded.
LENGTHS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64)


class Canable:
    def __init__(self, port="COM7", node=0):
        self.port = serial.Serial(port, 115200, timeout=0.01, write_timeout=1)
        self.port.dtr = True
        self.node = node
        self.pending = bytearray()
        for command in ("C", "S8", "Y5", "M0", "A1", "O"):
            self.port.write((command + "\r").encode())
            time.sleep(0.1)
        self.port.reset_input_buffer()

    def close(self):
        if not self.port.is_open:
            return
        self.port.write(b"C\r")
        self.port.close()

    def send(self, data, identifier=None):
        if len(data) > 64:
            raise ValueError("XCP packet exceeds CAN FD payload")
        size = next(n for n in LENGTHS if n >= len(data))
        dlc = LENGTHS.index(size)
        identifier = 0x667 + self.node if identifier is None else identifier
        wire = f"b{identifier:03X}{dlc:X}{data.ljust(size, bytes([0])).hex().upper()}\r"
        self.port.write(wire.encode())

    def receive(self, timeout=1, identifier_filter=None):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            byte = self.port.read(1)
            if not byte:
                continue
            if byte in (b"\r", b"\n"):
                line = bytes(self.pending)
                self.pending.clear()
                if not line or line[:1] not in (b"b", b"d", b"t", b"B", b"D", b"T"):
                    continue
                width = 8 if line[:1].isupper() else 3
                try:
                    identifier = int(line[1:1+width], 16)
                    dlc = int(line[1+width:2+width], 16)
                    data = bytes.fromhex(line[2+width:].decode())
                except ValueError as error:
                    raise RuntimeError(f"Malformed adapter frame: {line!r}") from error
                if len(data) != LENGTHS[dlc]:
                    raise RuntimeError(f"Adapter DLC mismatch: {line!r}")
                if identifier == (0x7E1 + self.node if identifier_filter is None else identifier_filter):
                    return data
            else:
                self.pending.extend(byte)
                if len(self.pending) > 512:
                    raise RuntimeError("Adapter serial record overflow")
        return None

    def request(self, data, timeout=1):
        self.send(data)
        return self.receive(timeout)

    def connect(self, timeout=4):
        self.port.reset_input_buffer()
        self.pending.clear()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            reply = self.request(b"\xff\x00", 0.15)
            if reply and len(reply) >= 8 and reply[0] == 0xFF:
                return reply
        raise RuntimeError(f"No OpenBLT response for node {self.node}")

    def read(self, address, length):
        reply = self.request(b"\xf6\0\0\0" + struct.pack("<I", address))
        positive(reply)
        result = bytearray()
        while len(result) < length:
            count = min(63, length - len(result))
            reply = self.request(bytes((0xF5, count)))
            positive(reply)
            if len(reply) < count + 1:
                raise RuntimeError("Truncated XCP UPLOAD response")
            result.extend(reply[1:count+1])
        return bytes(result)


def positive(reply):
    if not reply or reply[0] != 0xFF:
        raise RuntimeError(f"XCP error/timeout: {reply.hex() if reply else 'no response'}")


def recv_exact(conn, size):
    data = bytearray()
    while len(data) < size:
        part = conn.recv(size-len(data))
        if not part:
            raise EOFError
        data.extend(part)
    return bytes(data)


class Relay:
    def __init__(self, adapter, fault=None):
        self.adapter = adapter
        self.server = socket.socket()
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(1)
        self.server.settimeout(0.1)
        self.port = self.server.getsockname()[1]
        self.stopping = False
        self.error = None
        self.client = None
        self.finalized = False
        self.fault = fault
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        try:
            while not self.stopping:
                try:
                    conn, _ = self.server.accept()
                except socket.timeout:
                    continue
                self.client = conn
                with conn:
                    conn.settimeout(15)
                    while not self.stopping:
                        try:
                            counter = recv_exact(conn, 4)
                            command = recv_exact(conn, 1)
                        except EOFError:
                            break
                        opcode = command[0]
                        sizes = {0xFF: 2, 0xFE: 1, 0xFD: 1, 0xFC: 1, 0xFA: 2,
                                 0xF6: 8, 0xF5: 2, 0xD2: 1, 0xD1: 8, 0xCF: 1,
                                 0xC9: 64, 0xF1: 3}
                        if opcode == 0xD0:
                            command += recv_exact(conn, 1)
                            size = command[1] + 2
                        elif opcode in sizes:
                            size = sizes[opcode]
                        else:
                            raise RuntimeError(f"Unsupported upstream XCP opcode {opcode:02X}")
                        command += recv_exact(conn, size-len(command))
                        if self.fault:
                            self.fault(command)
                        reply = self.adapter.request(command, 0.1 if opcode == 0xFF else 2)
                        if reply:
                            if command == b"\xd0\x00":
                                self.finalized = reply[0] == 0xFF
                            conn.sendall(counter + reply)
                self.client = None
        except (OSError, EOFError) as error:
            if not self.stopping:
                self.error = error
        except Exception as error:
            self.error = error

    def close(self):
        self.stopping = True
        if self.client:
            try:
                self.client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.thread.join(3)
        self.server.close()


class Settings(C.Structure):
    _fields_ = [(name, C.c_uint16) for name in ("t1", "t3", "t4", "t5", "t6", "t7")] + [
        ("key", C.c_char_p), ("mode", C.c_uint8), ("bypass", C.c_uint8)]


class Network(C.Structure):
    _fields_ = [("address", C.c_char_p), ("port", C.c_uint16)]


def library():
    path = ROOT / "vendor/openblt/Host/libopenblt.dll"
    if C.sizeof(C.c_void_p) != 8:
        raise RuntimeError('The Windows LibOpenBLT backend requires 64-bit Python')
    if not path.is_file():
        raise RuntimeError('Install the 64-bit upstream libopenblt.dll at ' + str(path) +
                           '; see README.md, CAN firmware uploader dependency')
    lib = C.CDLL(str(path))
    lib.BltSessionInit.argtypes = [C.c_uint32, C.c_void_p, C.c_uint32, C.c_void_p]
    lib.BltSessionInit.restype = None
    for name in ("BltSessionTerminate", "BltSessionStop"):
        getattr(lib, name).restype = None
    lib.BltSessionClearMemory.argtypes = [C.c_uint32, C.c_uint32]
    for name in ("BltSessionWriteData", "BltSessionReadData"):
        getattr(lib, name).argtypes = [C.c_uint32, C.c_uint32, C.c_void_p]
    return lib


def load_image(path):
    path = Path(path)
    if path.suffix.lower() == ".bin":
        base, data = APP_BASE, path.read_bytes()
    else:
        memory = {}
        for line in path.read_text().splitlines():
            if not line.startswith(("S1", "S2", "S3")):
                continue
            record = bytes.fromhex(line[2:])
            if len(record) != record[0] + 1 or sum(record) & 255 != 255:
                raise ValueError("Invalid S-record count/checksum")
            width = int(line[1]) + 1
            address = int.from_bytes(record[1:1+width], "big")
            for offset, byte in enumerate(record[1+width:-1]):
                if address + offset in memory:
                    raise ValueError("Overlapping S-record data")
                if not APP_BASE <= address + offset < FLASH_END:
                    raise ValueError("Image includes protected/out-of-range flash")
                memory[address+offset] = byte
        if not memory:
            raise ValueError("No image data")
        base = min(memory)
        data = bytes(memory.get(addr, 255) for addr in range(base, max(memory)+1))
    if base != APP_BASE or len(data) < 0x1DC or base+len(data) > FLASH_END:
        raise ValueError("Expected a complete application image at 0x08003000")
    sp, reset = struct.unpack_from("<II", data)
    if not 0x20000000 < sp <= 0x20020000 or not reset & 1 or not base <= (reset & ~1) < base+len(data):
        raise ValueError("Invalid application vectors")
    return base, data


def update(adapter, path, start=True, interrupt_after=None, fault=None):
    base, data = load_image(path)
    adapter.connect()
    relay = Relay(adapter, fault)
    lib = library()
    settings = Settings(1000, 2000, 10000, 1000, 200, 2000, None, 0, 1)
    network = Network(b"127.0.0.1", relay.port)
    lib.BltSessionInit(0, C.byref(settings), 3, C.byref(network))
    def checked(code, operation):
        if code:
            raise RuntimeError(f"LibOpenBLT {operation} failed ({code}); relay={relay.error}")
    try:
        checked(lib.BltSessionStart(), "connect")
        checked(lib.BltSessionClearMemory(base, len(data)), "erase")
        for offset in range(0, len(data), 1024):
            if interrupt_after is not None and offset >= interrupt_after:
                # Deliberately do NOT finalize partial firmware. Terminate only.
                return False
            chunk = data[offset:offset+1024]
            buffer = C.create_string_buffer(chunk)
            checked(lib.BltSessionWriteData(base+offset, len(chunk), buffer), "program")
        if interrupt_after is not None:
            return False  # simulated host loss immediately before finalization
        lib.BltSessionStop()  # flushes deferred first block; stays in OpenBLT
        if not relay.finalized:
            raise RuntimeError(f"Target did not acknowledge finalization; relay={relay.error}")
    finally:
        # Terminate releases transport without finalizing an interrupted image.
        lib.BltSessionTerminate()
        relay.close()
    adapter.connect()
    actual = adapter.read(base, len(data))
    expected = bytearray(data)
    checksum = -sum(struct.unpack_from("<7I", data)) & 0xFFFFFFFF
    struct.pack_into("<I", expected, 0x1D8, checksum)
    if actual != expected:
        index = next(i for i, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1])
        raise RuntimeError(f"Full-image CAN readback mismatch at {base+index:08X}")
    print(f"PASS: {len(data)} application bytes read back over CAN, including OpenBLT signature")
    if start:
        adapter.request(b"\xcf", 0.1)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--port", default="COM7")
    parser.add_argument("--node", type=int, choices=range(8), default=0)
    parser.add_argument("--stay", action="store_true")
    args = parser.parse_args()
    adapter = Canable(args.port, args.node)
    try:
        update(adapter, args.image, start=not args.stay)
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
