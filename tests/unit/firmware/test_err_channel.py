"""ERR channel (M17 Task 1 §5) parsing and fix-instruction translation (Task 4 §5)."""
import re
from pathlib import Path

from firmware.krabby_mcu import _FIX_INSTRUCTIONS, KrabbyMCUSDK, parse_err_line

FIRMWARE_DIR = Path(__file__).resolve().parents[3] / "firmware" / "arduino"


class TestParseErrLine:
    def test_joint_error(self):
        assert parse_err_line("ERR RLHL pot_value_invalid") == ("RLHL", "pot_value_invalid")

    def test_not_err_line(self):
        assert parse_err_line("FRONT; FLHY 0.5 500 0 0 0 0 0 0 1 2") is None
        assert parse_err_line("CAL FLHY FAIL no_stop") is None

    def test_wrong_token_count_is_rejected(self):
        assert parse_err_line("ERR RLHL") is None
        assert parse_err_line("ERR RLHL pot_value_invalid extra") is None


class TestErrDispatch:
    def test_err_line_is_recorded(self, bare_sdk):
        bare_sdk._on_err_line("ERR FRKL pot_value_invalid")
        assert list(bare_sdk.errors) == [("FRKL", "pot_value_invalid")]

    def test_malformed_err_line_is_ignored(self, bare_sdk):
        bare_sdk._on_err_line("ERR FRKL")
        assert not bare_sdk.errors


class TestExplainFailures:
    def test_known_code_names_the_joint(self):
        [msg] = KrabbyMCUSDK.explain_failures([("MLKL", "pot_value_invalid")])
        assert msg == "Check potentiometer wiring on MLKL (3-wire harness: VCC, signal, GND)."

    def test_unknown_code_is_surfaced_not_dropped(self):
        assert KrabbyMCUSDK.explain_failures([("FLHY", "mystery")]) == ["Unknown failure on FLHY: mystery"]

    def test_every_code_the_firmware_emits_has_instructions(self):
        src = "".join(p.read_text() for p in FIRMWARE_DIR.rglob("*.[hc]*") if p.suffix in (".h", ".cpp", ".ino"))
        emitted = set(re.findall(r'printErr\([^,]+,[^,]+,\s*"(\w+)"\)', src))
        assert "pot_value_invalid" in emitted
        assert emitted <= set(_FIX_INSTRUCTIONS)
