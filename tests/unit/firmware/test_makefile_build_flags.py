"""Regression test for the front<->follower comms bug (M17 Task 1 §2.1).

Root cause: the leader Mega forwards ~200-byte telemetry lines from each follower
on Serial1/Serial2 while servicing USB + the actuator update. That needs a 256-byte
serial RX buffer. CI (.github/workflows/publish-firmware.yml) and install.py's
platform.local.txt both pass `-DSERIAL_RX_BUFFER_SIZE=256`, but the Makefile did
not -- so `make compile/upload-firmware` could emit a binary whose RX buffer fell
back to the AVR core default on a host whose core version defaults below 256,
dropping the middle of forwarded lines.

The fix bakes the define into the Makefile's BUILD_PROPS unconditionally, so every
local build matches CI. Pre-fix, BUILD_PROPS was empty for the default build and
only set KRABBY_PIN_REV otherwise, never the buffer size (and the Makefile's
default PIN_REV=2 disagreed with board_pins.h's built-in default of 3) -> these
assertions fail; post-fix they pass.

We assert against the *actual* compile command `make` would run (`make -n`), not a
regex on the Makefile text, so the test tracks real build behavior across edits.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

FIRMWARE_DIR = Path(__file__).resolve().parents[3] / "firmware"

pytestmark = pytest.mark.skipif(
    shutil.which("make") is None, reason="`make` not available on PATH"
)


def _compile_command(pin_rev: str | None = None) -> str:
    """Return the arduino-cli compile line `make` would run for compile-firmware.

    `make -n` prints commands without executing them, so this needs neither
    arduino-cli nor a board attached.
    """
    cmd = ["make", "-n", "compile-firmware"]
    if pin_rev is not None:
        cmd.append(f"PIN_REV={pin_rev}")
    out = subprocess.run(
        cmd, cwd=FIRMWARE_DIR, capture_output=True, text=True, check=True
    ).stdout
    for line in out.splitlines():
        if "arduino-cli compile" in line:
            return line
    raise AssertionError(f"no compile line in `make -n compile-firmware` output:\n{out}")


class TestSerialRxBufferFlag:
    def test_default_build_passes_256_buffer(self):
        line = _compile_command()
        assert "-DSERIAL_RX_BUFFER_SIZE=256" in line

    def test_pin_rev_override_still_passes_256_buffer(self):
        # The original bug: only the non-default PIN_REV path set BUILD_PROPS, and
        # even then it set KRABBY_PIN_REV only -- never the buffer size.
        line = _compile_command(pin_rev="1")
        assert "-DSERIAL_RX_BUFFER_SIZE=256" in line

    def test_buffer_flag_on_both_c_and_cpp(self):
        # The core's HardwareSerialN.cpp (C++) holds the ring buffer; the define
        # must reach both compilers to match CI and install.py exactly.
        line = _compile_command()
        assert "compiler.cpp.extra_flags" in line
        assert "compiler.c.extra_flags" in line
        assert line.count("-DSERIAL_RX_BUFFER_SIZE=256") == 2

    def test_pin_rev_still_selected(self):
        # The buffer fix must not drop KRABBY_PIN_REV selection.
        assert "-DKRABBY_PIN_REV=1" in _compile_command(pin_rev="1")
        assert "-DKRABBY_PIN_REV=3" in _compile_command(pin_rev="3")
