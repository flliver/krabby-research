"""Unit tests for the M17 Task 1 §5 ERR telemetry channel (SDK side).

Covers the wire parser, the reader-loop dispatch, the bounded ring, and the
optional callback. The firmware emit primitive (emitError) and the per-fault
throttle land with the emitting tasks (2-4); Task 1 owns the channel + SDK plumbing.
"""
import threading
import time

from firmware.krabby_mcu import (
    KrabbyMCUSDK,
    ErrorEvent,
    ERROR_CODES,
    parse_err_line,
    _ERROR_RING_MAX,
    _FIX_INSTRUCTIONS,
)


# --- parse_err_line -------------------------------------------------------

class TestParseErrLine:
    def test_joint_error(self):
        assert parse_err_line("ERR FRKL motor_did_not_move") == (
            "FRKL", "motor_did_not_move")

    def test_system_token(self):
        assert parse_err_line("ERR system current_sense_no_signal") == (
            "system", "current_sense_no_signal")

    def test_extra_whitespace_collapses(self):
        assert parse_err_line("ERR  FRKL   hall_drift") == ("FRKL", "hall_drift")

    def test_too_few_tokens_returns_none(self):
        assert parse_err_line("ERR onlyonetoken") is None

    def test_too_many_tokens_returns_none(self):
        # exactly two tokens after ERR per §5; three is malformed
        assert parse_err_line("ERR FRKL motor_did_not_move extra") is None

    def test_non_err_line_returns_none(self):
        assert parse_err_line("VER 1.0 main abc") is None
        assert parse_err_line("GET role FRONT") is None
        assert parse_err_line("FRONT; FLHY 0.5") is None
        assert parse_err_line("") is None

    def test_err_without_trailing_space_not_matched(self):
        # "ERROR ..." must not be mistaken for an ERR line
        assert parse_err_line("ERROR something happened") is None


# --- vocabulary -----------------------------------------------------------

class TestErrorVocabulary:
    def test_canonical_codes_present(self):
        # the eight §5 codes Tasks 2/4 emit
        for code in (
            "motor_did_not_move", "motor_jammed", "pot_value_invalid",
            "hall_no_edges", "hall_drift", "not_calibrated",
            "current_sense_no_signal", "current_sense_no_spike",
            "sensor_type_mismatch",
        ):
            assert code in ERROR_CODES

    def test_vocabulary_is_reference_only_not_enforced(self):
        # an unknown code still parses — the wire is self-describing
        assert parse_err_line("ERR FRKL brand_new_code") == ("FRKL", "brand_new_code")


# --- _record_error: ring + callback (no thread) ---------------------------

class TestRecordError:
    def test_appends_event_with_timestamp(self):
        sdk = KrabbyMCUSDK(port="fake")
        sdk._record_error("ERR FRKL motor_did_not_move")
        events = sdk.get_errors()
        assert len(events) == 1
        ev = events[0]
        assert isinstance(ev, ErrorEvent)
        assert (ev.token, ev.code) == ("FRKL", "motor_did_not_move")
        assert isinstance(ev.ts, float)

    def test_malformed_line_dropped(self):
        sdk = KrabbyMCUSDK(port="fake")
        sdk._record_error("ERR nope")
        sdk._record_error("not an err line")
        assert sdk.get_errors() == []

    def test_callback_invoked(self):
        sdk = KrabbyMCUSDK(port="fake")
        seen = []
        sdk.on_error(seen.append)
        sdk._record_error("ERR system current_sense_no_spike")
        assert len(seen) == 1
        assert (seen[0].token, seen[0].code) == ("system", "current_sense_no_spike")

    def test_callback_exception_isolated(self):
        sdk = KrabbyMCUSDK(port="fake")

        def boom(_event):
            raise RuntimeError("callback blew up")

        sdk.on_error(boom)
        # must not raise, and the event is still recorded
        sdk._record_error("ERR FRKL motor_jammed")
        assert len(sdk.get_errors()) == 1

    def test_on_error_none_clears_callback(self):
        sdk = KrabbyMCUSDK(port="fake")
        seen = []
        sdk.on_error(seen.append)
        sdk.on_error(None)
        sdk._record_error("ERR FRKL hall_no_edges")
        assert seen == []
        assert len(sdk.get_errors()) == 1

    def test_clear_errors(self):
        sdk = KrabbyMCUSDK(port="fake")
        sdk._record_error("ERR FRKL motor_did_not_move")
        sdk.clear_errors()
        assert sdk.get_errors() == []

    def test_ring_bounded_to_capacity(self):
        sdk = KrabbyMCUSDK(port="fake")
        for i in range(_ERROR_RING_MAX + 25):
            sdk._record_error(f"ERR J{i % 18} pot_value_invalid")
        events = sdk.get_errors()
        assert len(events) == _ERROR_RING_MAX
        # oldest dropped: the last event is the most recent one pushed
        assert events[-1].token == f"J{(_ERROR_RING_MAX + 24) % 18}"


# --- reader-loop dispatch (threaded, fake serial) -------------------------

class _FakeSerial:
    """Minimal serial stand-in: yields queued byte lines, then idles on b''."""
    def __init__(self, lines):
        self._lines = list(lines)
        self.is_open = True

    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        time.sleep(0.005)
        return b""


class TestReaderLoopDispatch:
    def test_err_lines_routed_to_ring(self):
        sdk = KrabbyMCUSDK(port="fake")
        sdk.ser = _FakeSerial([
            b"ERR FRKL motor_did_not_move\n",
            b"Krabby Ready role=FRONT\n",       # non-ERR, harmless info line
            b"ERR onlyonetoken\n",              # malformed -> dropped
            b"ERR system current_sense_no_spike\n",
        ])
        sdk.running = True
        t = threading.Thread(target=sdk._reader_loop, daemon=True)
        t.start()

        deadline = time.time() + 1.0
        while time.time() < deadline and len(sdk.get_errors()) < 2:
            time.sleep(0.01)
        sdk.running = False
        t.join(timeout=1.0)

        pairs = [(e.token, e.code) for e in sdk.get_errors()]
        assert pairs == [
            ("FRKL", "motor_did_not_move"),
            ("system", "current_sense_no_spike"),
        ]


class TestExplainFailures:
    """Task 4 §5 / 4f: reason-code → operator fix-instruction translation."""

    def test_translates_error_events(self):
        errs = [ErrorEvent("FLHL", "motor_did_not_move", 1.0),
                ErrorEvent("FRKL", "pot_value_invalid", 2.0)]
        out = KrabbyMCUSDK.explain_failures(errs)
        assert len(out) == 2
        assert "FLHL" in out[0] and "motor power" in out[0]
        assert "FRKL" in out[1] and "potentiometer" in out[1]

    def test_accepts_bare_tuples(self):
        out = KrabbyMCUSDK.explain_failures([("MLKL", "hall_drift")])
        assert out == [_FIX_INSTRUCTIONS["hall_drift"].format(joint="MLKL")]

    def test_unknown_code_is_surfaced_not_dropped(self):
        out = KrabbyMCUSDK.explain_failures([("RRHY", "brand_new_code")])
        assert out == ["Unknown failure on RRHY: brand_new_code"]

    def test_current_sense_codes_present(self):
        for code in ("current_sense_no_signal", "current_sense_no_spike"):
            out = KrabbyMCUSDK.explain_failures([("FLKL", code)])
            assert "FLKL" in out[0] and "current" in out[0].lower()


class TestVocabParity:
    """4a/4h: the canonical vocabulary and the fix-instruction table stay in lockstep."""

    def test_every_code_has_a_fix_instruction(self):
        missing = ERROR_CODES - set(_FIX_INSTRUCTIONS)
        assert not missing, f"codes with no fix instruction: {missing}"

    def test_every_fix_instruction_is_a_known_code(self):
        extra = set(_FIX_INSTRUCTIONS) - ERROR_CODES
        assert not extra, f"fix instructions for unknown codes: {extra}"

    def test_task4_required_codes_present(self):
        for code in ("current_sense_no_signal", "current_sense_no_spike",
                     "not_in_starting_pose", "not_calibrated"):
            assert code in ERROR_CODES
