"""Regression guard for the floating-RX flood-starvation fix (M17 Task 1, AC 1d).

Root cause (see firmware/COMMS_DEBUG.md): a board drains its main serial channel
with an unbounded `while (mainSerial->available())` *before* the actuator update.
A disconnected/dangling RX line floats, EMI induces a continuous phantom-byte
stream, the drain never exits, and the board starves ("no response").

Fix (ported from the m17 branch, arduino.ino):
  1. pull the follower RX pins (and RX0) up so a disconnected line idles high,
     not into noise;
  2. bound every drain (RX_DRAIN_BUDGET / FWD_DRAIN_BUDGET) so no channel can
     starve the rest;
  3. discard unknown bytes singly and forward only printable-ASCII lines, so
     line noise costs neither blocking reads nor heap Strings.

These assertions FAIL on the pre-fix source (no pull-ups, no budget, a bare
`while (mainSerial->available())`) and PASS post-fix — the same shape as
test_makefile_build_flags.py. We assert against the firmware source because the
fix is AVR C++ with no host-side unit harness; the live before/after (0/N -> 8/8
config reads on the m17 bench) is the documented manual repro in COMMS_DEBUG.md.
"""
import re
from pathlib import Path

import pytest

ARDUINO_INO = Path(__file__).resolve().parents[3] / "firmware" / "arduino" / "arduino.ino"


@pytest.fixture(scope="module")
def src() -> str:
    return ARDUINO_INO.read_text()


class TestFloatingRxFix:
    def test_follower_rx_pins_pulled_up(self, src):
        # A disconnected uplink must idle HIGH (driven leader TX still overrides),
        # not float as an EMI antenna.
        assert re.search(r"pinMode\(\s*SERIAL_LEFT_RX\s*,\s*INPUT_PULLUP\s*\)", src), \
            "SERIAL_LEFT_RX must be INPUT_PULLUP"
        assert re.search(r"pinMode\(\s*SERIAL_RIGHT_RX\s*,\s*INPUT_PULLUP\s*\)", src), \
            "SERIAL_RIGHT_RX must be INPUT_PULLUP"

    def test_rx_drain_budget_defined(self, src):
        assert re.search(r"RX_DRAIN_BUDGET\s*=\s*\d+", src), \
            "RX_DRAIN_BUDGET cap must be defined"

    def test_drains_are_bounded(self, src):
        # The loop() main-channel drain must cap lines per pass, so a flooding
        # channel can't starve the actuator update. (On the m17 branch this also
        # covers processConfig(); that arrives with the SET/GET port.)
        bounded = re.findall(r"while\s*\([^)]*available\(\)[^)]*rxBudget--\s*>\s*0", src)
        assert len(bounded) >= 1, f"expected >=1 budget-bounded drain, found {len(bounded)}"

    def test_no_bare_unbounded_mainserial_drain(self, src):
        # The exact pre-fix bug: `while (mainSerial->available())` with no budget.
        # The real drain must be the bounded form (caught above).
        bare = re.findall(r"while\s*\(\s*mainSerial->available\(\)\s*\)", src)
        assert not bare, f"unbounded mainSerial drain reintroduced ({len(bare)} found)"

    def test_forward_drain_bounded(self, src):
        # Leader-side twin of the same bug: forwardFullLines() drains the follower
        # ports (Serial1/Serial2), which float on a follower-less bench. Motor EMI
        # bursts beat the weak pullup and an unbounded `while (from->available())`
        # captured loop() — runaway FLHY, bench 2026-07-03. Must stay budgeted.
        bare = re.findall(r"while\s*\(\s*from->available\(\)\s*\)", src)
        assert not bare, f"unbounded from->available() drain reintroduced ({len(bare)} found)"
        assert re.search(r"FWD_DRAIN_BUDGET\s*=\s*\d+", src), \
            "FWD_DRAIN_BUDGET cap must be defined"
        bounded = re.findall(r"while\s*\(\s*budget--\s*>\s*0\s*&&\s*from->available\(\)", src)
        assert len(bounded) >= 2, \
            f"both forwardFullLines drains (main + overrun-discard) must be budgeted, found {len(bounded)}"

    def test_rx0_pulled_up(self, src):
        # The USB serial chip tri-states RX0 when motor EMI knocks it off the bus
        # (observed re-enumerating mid-session); a floating RX0 feeds garbage straight
        # into the command dispatcher. Pull it up like the follower RX pins.
        assert re.search(r"pinMode\(\s*0\s*,\s*INPUT_PULLUP\s*\)", src), \
            "RX0 (pin 0) must be INPUT_PULLUP so a dead USB chip reads as idle"

    def test_unknown_bytes_discarded_without_line_drain(self, src):
        # Unknown bytes are line noise, not commands: readStringUntil() on them costs
        # a 50 ms block + a heap String per call, which stalled loop() for seconds and
        # fragmented the heap under continuous garbage. Must be a single-byte read().
        unknown_line_drain = re.search(
            r"//\s*Unknown[^\n]*\n(?:[^\n]*\n){0,8}?\s*mainSerial->readStringUntil", src)
        assert not unknown_line_drain, \
            "unknown-command path must not readStringUntil() (blocking + String churn)"
        assert "mainSerial->read();" in src, \
            "unknown-command path must discard a single byte via read()"

    def test_forward_filters_nonprintable_lines(self, src):
        # Second half of the same bench failure: EMI framing garbage that *does*
        # reach a newline must not be forwarded upstream — each junk println blocks
        # on the saturated USB TX buffer, slowing loop() until queued jog heartbeats
        # outlive the operator's release (motor stops late, then not at all).
        assert re.search(r"lineIsPrintable\s*\(", src), \
            "forwardFullLines must gate forwarding on lineIsPrintable()"
        assert re.search(r"if\s*\(\s*\*partialPos\s*>\s*0\s*&&\s*lineIsPrintable\(", src), \
            "only clean printable-ASCII complete lines may be forwarded"
