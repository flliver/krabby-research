import json
import sys

from firmware.bench_tests import __main__ as bench_cli
from firmware.bench_tests.task4 import (
    _boot_id_is_listed,
    Evidence,
    POWERING_DOWN,
    SerialTrace,
    battery_from_line,
    journal_has_entries,
    current_daemon_log_pass,
    parse_power_line,
    production_pass,
    previous_shutdown_log_pass,
    soft_cut_pass,
)
from firmware.interfaces.battery_telemetry import (
    PackVoltageRegion,
    PowerSafetyState,
)


def _line(volts=26.5, region=5, state=0, valid=1):
    return (
        "FRONT; BATT "
        f"{volts} 0.0 0.0 0.0 13.2 13.3 0 {valid} 1 {region} {state}"
    )


class FakeSerial:
    def __init__(self, lines):
        self.lines = [f"{line}\n".encode() for line in lines]

    def readline(self):
        return self.lines.pop(0) if self.lines else b""


def test_power_parser_accepts_only_exact_v1_messages():
    assert parse_power_line(POWERING_DOWN) == (
        "POWERING_DOWN",
        "under_voltage_soft",
    )
    assert parse_power_line("PWR 1 SHUTDOWN_ACK") == ("SHUTDOWN_ACK", "")
    assert parse_power_line("PWR 1 RESUMING voltage_recovered") == (
        "RESUMING",
        "voltage_recovered",
    )
    assert parse_power_line("PWR 1 EMERGENCY_SHUTDOWN over_voltage") == (
        "EMERGENCY_SHUTDOWN",
        "over_voltage",
    )
    assert parse_power_line("PWR 2 SHUTDOWN_ACK") is None
    assert parse_power_line("PWR 1 SHUTDOWN_ACK extra") is None
    assert parse_power_line("noise PWR 1 SHUTDOWN_ACK") is None


def test_battery_parser_uses_current_numeric_wire_format():
    battery = battery_from_line(_line(23.8, 2, 1))

    assert battery.pack_volts == 23.8
    assert battery.pack_region is PackVoltageRegion.SOFT_CUT
    assert battery.power_safety_state is PowerSafetyState.POWERING_DOWN


def test_serial_trace_counts_four_consecutive_soft_frames_and_logs_transitions(tmp_path):
    lines = [
        _line(23.8, 2, 0),
        _line(23.8, 2, 0),
        _line(23.8, 2, 0),
        POWERING_DOWN,
        _line(23.8, 2, 1),
    ]
    evidence = Evidence(str(tmp_path / "results.json"))
    trace = SerialTrace(FakeSerial(lines), evidence)

    events = [trace.next() for _ in lines]

    assert events[-1].soft_streak == 4
    assert trace.power_messages == [POWERING_DOWN]
    records = [json.loads(line) for line in evidence.events_path.read_text().splitlines()]
    assert any(record["kind"] == "pwr" for record in records)
    assert any(
        record["kind"] == "batt_transition" and record["state"] == 1
        for record in records
    )


def test_non_soft_or_invalid_frame_resets_soft_streak(tmp_path):
    evidence = Evidence(str(tmp_path / "results.json"))
    trace = SerialTrace(
        FakeSerial(
            [
                _line(23.8, 2, 0),
                _line(23.8, 2, 0),
                _line(24.6, 1, 0),
                _line(23.8, 2, 0, valid=0),
            ]
        ),
        evidence,
    )

    assert trace.next().soft_streak == 1
    assert trace.next().soft_streak == 2
    assert trace.next().soft_streak == 0
    assert trace.next().soft_streak == 0


def test_evidence_persists_checkpoints_and_distinguishes_open_work(tmp_path):
    path = tmp_path / "task4.json"
    evidence = Evidence(str(path))
    evidence.checkpoint("instrumented.warn", True, voltage=24.6)
    evidence.phase("instrumented", "FAIL", "hard cut not run")

    saved = json.loads(path.read_text())
    assert saved["checkpoints"]["instrumented.warn"]["status"] == "PASS"
    assert saved["phases"]["instrumented"]["status"] == "FAIL"
    assert any("belly-down" in item for item in saved["deferred"])
    assert any("stopped-Orin" in item for item in saved["deviations"])
    assert evidence.events_path.exists()
    event = json.loads(evidence.events_path.read_text().splitlines()[0])
    assert event["attempt_id"] == evidence.attempt_id
    report = evidence.report_path.read_text()
    assert "**DEFERRED**" in report
    assert "**DEVIATION**" in report


def test_cli_routes_task4_to_the_stateful_runner(monkeypatch):
    called = {}

    def fake_run(port, remote, results):
        called.update(port=port, remote=remote, results=results)
        return 7

    monkeypatch.setattr(bench_cli.task4, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench",
            "run",
            "--suite",
            "task4",
            "--port",
            "/dev/cu.test",
            "--remote",
            "sanjay@orin",
        ],
    )

    assert bench_cli.main() == 7
    assert called == {
        "port": "/dev/cu.test",
        "remote": "sanjay@orin",
        "results": "bench-results-task4.json",
    }


def test_cli_requires_remote_for_task4(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["bench", "run", "--suite", "task4", "--port", "/dev/cu.test"],
    )

    assert bench_cli.main() == 2
    assert "--remote" in capsys.readouterr().err


def test_soft_cut_verdict_rejects_duplicate_message_and_wrong_clock():
    values = {
        "soft_streak": 4,
        "powering_down_count": 1,
        "cutoff_seconds": 15.0,
        "actuators_held": True,
        "relay_confirmed": True,
        "recovery_seconds": 29.5,
    }

    assert soft_cut_pass(values, acknowledged=True)
    assert not soft_cut_pass({**values, "powering_down_count": 2}, acknowledged=True)
    assert not soft_cut_pass({**values, "cutoff_seconds": 60.0}, acknowledged=True)
    assert soft_cut_pass({**values, "cutoff_seconds": 60.0}, acknowledged=False)


def test_production_verdict_cannot_pass_no_ack_or_incomplete_restart():
    values = {
        "low_battery_oled": True,
        "ssh_unreachable": True,
        "relay_off": True,
        "relay_off_seconds": 16.0,
        "relay_on": True,
        "relay_on_seconds": 29.0,
        "ssh_returned": True,
        "journal_ok": True,
        "boot_changed": True,
        "production_ready": True,
        "clean_shutdown": True,
    }

    assert production_pass(values)
    assert production_pass({**values, "relay_on_seconds": 34.9})
    assert not production_pass({**values, "relay_off_seconds": 60.0})
    assert not production_pass({**values, "relay_on_seconds": 35.1})
    assert not production_pass({**values, "production_ready": False})
    assert not production_pass({**values, "boot_changed": False})


def test_empty_previous_boot_journal_is_not_evidence():
    assert not journal_has_entries("")
    assert not journal_has_entries("-- No entries --")
    assert journal_has_entries("systemd[1]: Finished krabby-poweroff.service")


def test_shutdown_ack_and_current_readiness_come_from_different_boot_logs():
    previous = "Power-down received; host poweroff requested; acking"
    current = "Orin power daemon armed\nMCU connected successfully"

    assert previous_shutdown_log_pass(previous)
    assert not current_daemon_log_pass(previous)
    assert current_daemon_log_pass(current)
    assert not previous_shutdown_log_pass(current)


def test_previous_boot_list_is_bound_to_the_tested_boot_id():
    boot_id = "386acfa0-b1b8-4f52-9b55-6cac04bedf5d"
    output = "-1 386acfa0b1b84f529b556cac04bedf5d Sun 2026-08-30"

    assert _boot_id_is_listed(boot_id, output)
    assert not _boot_id_is_listed("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", output)
