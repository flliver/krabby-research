"""Stateful M16 Task 4 power-safety bench procedure.

The production Orin and a Mac cannot own the Mega's USB serial port at the same
time.  This runner therefore guides one production-owned phase, then opens one
long-lived Mac serial session for all instrumented state-machine checks.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from serial.tools import list_ports

import serial
from firmware.krabby_mcu import DEFAULT_BAUD
from firmware.interfaces.telemetry_frame import TelemetryFrame
from firmware.interfaces.battery_telemetry import (
    BatteryTelemetry,
    PackVoltageRegion,
    PowerSafetyState,
)
from firmware.interfaces.power_messages import PowerMessage


SAFE_VOLTS = 26.5
WARN_VOLTS = 24.6
SOFT_CUT_VOLTS = 23.8
HARD_CUT_VOLTS = 22.0
OVER_VOLT_VOLTS = 29.8

POWERING_DOWN = "PWR 1 POWERING_DOWN under_voltage_soft"
SHUTDOWN_ACK = "PWR 1 SHUTDOWN_ACK"
RESUMING = "PWR 1 RESUMING voltage_recovered"
OVER_VOLTAGE = "PWR 1 EMERGENCY_SHUTDOWN over_voltage"

PHASES = (
    ("static", "Firmware identity and native power-policy verification"),
    ("preflight", "Wiring, monitor, and production-host preflight"),
    ("production", "Production Orin shutdown and automatic recovery"),
    ("instrumented", "Single-session MCU state-machine characterization"),
    ("restore", "Restore production wiring and record open M16 work"),
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def parse_power_line(raw: str) -> Optional[tuple[str, str]]:
    """Expose the production power-message parser's typed result for evidence."""
    message = PowerMessage.parse(raw)
    if message is None:
        return None
    reason = message.reason.value if message.reason is not None else ""
    return message.type.value, reason


def battery_from_line(raw: str) -> Optional[BatteryTelemetry]:
    return TelemetryFrame.parse_line(raw).battery


def soft_cut_pass(values: dict, acknowledged: bool) -> bool:
    lower, upper = (12.0, 18.0) if acknowledged else (57.0, 63.0)
    return bool(
        values["soft_streak"] == 4
        and values["powering_down_count"] == 1
        and lower <= values["cutoff_seconds"] <= upper
        and values["actuators_held"]
        and values["relay_confirmed"]
        and values["recovery_seconds"] <= 31.0
    )


def production_pass(values: dict) -> bool:
    return bool(
        values["low_battery_oled"]
        and values["ssh_unreachable"]
        and values["relay_off"]
        and values["relay_off_seconds"] <= 30.0
        and values["relay_on"]
        and values["relay_on_seconds"] <= 35.0
        and values["ssh_returned"]
        and values["journal_ok"]
        and values["boot_changed"]
        and values["production_ready"]
        and values["clean_shutdown"]
    )


def journal_has_entries(output: str) -> bool:
    return bool(output.strip()) and "No entries" not in output


def previous_shutdown_log_pass(output: str) -> bool:
    return journal_has_entries(output) and "host poweroff requested; acking" in output


def current_daemon_log_pass(output: str) -> bool:
    return (
        "Orin power daemon armed" in output
        and "MCU connected successfully" in output
    )


def _result_paths(results_path: str) -> tuple[Path, Path, Path]:
    summary = Path(results_path)
    stem = summary.with_suffix("")
    return summary, Path(f"{stem}.events.jsonl"), Path(f"{stem}.md")


def port_identity(port: str) -> dict:
    """Return stable USB identity fields when pyserial exposes them."""
    match = next((item for item in list_ports.comports() if item.device == port), None)
    if match is None:
        return {"device": port}
    return {
        "device": match.device,
        "vid": match.vid,
        "pid": match.pid,
        "serial_number": match.serial_number,
        "location": match.location,
        "manufacturer": match.manufacturer,
        "product": match.product,
    }


class Evidence:
    """Durable results plus an append-only event stream."""

    def __init__(self, results_path: str):
        self.summary_path, self.events_path, self.report_path = _result_paths(
            results_path
        )
        self.started_monotonic = time.monotonic()
        self.has_previous = self.summary_path.exists()
        self.attempt_id = uuid.uuid4().hex[:12]
        self.data = {
            "schema_version": 1,
            "suite": "task4",
            "updated": utc_now(),
            "phases": {},
            "deferred": [
                "4c belly-down parking",
                "4h real-pack full-load threshold validation",
            ],
            "deviations": [
                "4i stopped-Orin-telemetry detection is not implemented",
                "over-voltage wire token is EMERGENCY_SHUTDOWN",
                "manual POWERING_DOWN reason was removed",
                "soft-cut OLED remains visible between 10 s refreshes",
                "threshold constants live in power_safety_constants.h",
            ],
            "acceptance": {
                "4a": {"status": "NOT RUN", "evidence": "native policy tests"},
                "4b": {"status": "DEVIATION", "evidence": "implemented wire contract"},
                "4c": {"status": "PARTIAL", "evidence": "belly-down deferred"},
                "4d": {"status": "NOT RUN", "evidence": "HardCut scenario"},
                "4e": {"status": "NOT RUN", "evidence": "OverVolt scenario"},
                "4f": {"status": "PARTIAL", "evidence": "cadence and recovery only"},
                "4g": {"status": "DEVIATION", "evidence": "persistent SoftCut panel"},
                "4h": {"status": "DEFERRED", "evidence": "full-load pack validation"},
                "4i": {"status": "DEVIATION", "evidence": "no stopped-telemetry detector"},
                "4j": {"status": "NOT RUN", "evidence": "production daemon scenario"},
                "4k": {"status": "NOT RUN", "evidence": "PSU timing scenarios"},
            },
        }
        if self.summary_path.exists():
            try:
                previous = json.loads(self.summary_path.read_text())
                if previous.get("suite") == "task4":
                    self.data.update(previous)
            except (OSError, json.JSONDecodeError):
                pass
        self._save()

    def fresh_attempt(self) -> None:
        self.data["phases"] = {}
        self.data["checkpoints"] = {}
        for criterion in ("4a", "4d", "4e", "4j", "4k"):
            self.data["acceptance"][criterion]["status"] = "NOT RUN"
        self._save()

    def event(self, kind: str, detail: str, **values) -> dict:
        record = {
            "at": utc_now(),
            "elapsed_s": round(time.monotonic() - self.started_monotonic, 3),
            "attempt_id": self.attempt_id,
            "kind": kind,
            "detail": detail,
            **values,
        }
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def phase(self, key: str, status: str, detail: str, **values) -> None:
        self.data["phases"][key] = {
            "status": status,
            "detail": detail,
            "at": utc_now(),
            **values,
        }
        self.event("phase", f"{key}: {status}", status=status, **values)
        self._save()

    def status(self, key: str) -> Optional[str]:
        return self.data.get("phases", {}).get(key, {}).get("status")

    def checkpoint(self, key: str, passed: bool, **values) -> None:
        checkpoints = self.data.setdefault("checkpoints", {})
        checkpoints[key] = {
            "status": "PASS" if passed else "FAIL",
            "at": utc_now(),
            **values,
        }
        self.event("checkpoint", key, passed=passed, **values)
        self._save()

    def clear_checkpoints(self, prefix: str) -> None:
        checkpoints = self.data.setdefault("checkpoints", {})
        self.data["checkpoints"] = {
            key: value
            for key, value in checkpoints.items()
            if not key.startswith(prefix)
        }
        self._save()

    def criterion(self, ac: str, status: str, detail: str) -> None:
        self.data["acceptance"][ac] = {"status": status, "evidence": detail}
        self._save()

    def _save(self) -> None:
        self.data["updated"] = utc_now()
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.summary_path.with_suffix(self.summary_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.data, indent=2) + "\n")
        os.replace(temporary, self.summary_path)
        self._write_report()

    def _write_report(self) -> None:
        lines = [
            "# M16 Task 4 bench evidence",
            "",
            f"Updated: `{self.data['updated']}`",
            "",
            "| Phase | Status | Evidence |",
            "|---|---|---|",
        ]
        for key, title in PHASES:
            result = self.data.get("phases", {}).get(key, {})
            detail = str(result.get("detail", "not run")).replace("\n", "<br>")
            lines.append(
                f"| {title} | {result.get('status', 'NOT RUN')} | {detail} |"
            )
        lines.extend(["", "## Instrumented checkpoints", ""])
        lines.extend(["| Check | Status | Measurements |", "|---|---|---|"])
        for key, result in self.data.get("checkpoints", {}).items():
            measurements = ", ".join(
                f"{name}={value}"
                for name, value in result.items()
                if name not in ("status", "at")
            )
            lines.append(f"| {key} | {result['status']} | {measurements} |")
        lines.extend(["", "## Grant acceptance accounting", ""])
        lines.extend(["| AC | Status | Evidence |", "|---|---|---|"])
        for ac, result in self.data["acceptance"].items():
            lines.append(f"| {ac} | {result['status']} | {result['evidence']} |")
        lines.extend(["", "## Deferred", ""])
        lines.extend(f"- **DEFERRED** — {item}" for item in self.data["deferred"])
        lines.extend(["", "## Deviations", ""])
        lines.extend(f"- **DEVIATION** — {item}" for item in self.data["deviations"])
        lines.append("")
        self.report_path.write_text("\n".join(lines))


def _yes(prompt: str) -> bool:
    return input(f"  {prompt} [y/N] > ").strip().lower().startswith("y")


def _number(prompt: str) -> float:
    while True:
        try:
            return float(input(f"  {prompt} > ").strip())
        except ValueError:
            print("  enter a numeric value")


def _confirm_safe_voltage() -> float:
    while True:
        measured = _number("verified Pack VBUS after restoring the safe setpoint (V)")
        if 26.4 < measured < 26.6:
            return measured
        print("  restore and verify approximately 26.5 V before continuing")


def _ssh(remote: str, command: str) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["ssh", remote, command],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )
        return result.returncode == 0, result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _local(command: list[str], timeout: float = 120.0) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
            cwd=REPO_ROOT,
        )
        return result.returncode == 0, result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _remote_host(remote: str) -> str:
    return remote.rsplit("@", 1)[-1]


def _boot_id_is_listed(boot_id: str, output: str) -> bool:
    return boot_id.strip().replace("-", "") in output.replace("-", "")


def _wait_for_ssh(host: str, reachable: bool, timeout: float) -> Optional[float]:
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        try:
            with socket.create_connection((host, 22), timeout=1.0):
                is_up = True
        except OSError:
            is_up = False
        if is_up is reachable:
            return time.monotonic() - started
        time.sleep(1.0)
    return None


@dataclass
class SerialEvent:
    raw: str
    at: float
    battery: Optional[BatteryTelemetry] = None
    power: Optional[tuple[str, str]] = None
    soft_streak: int = 0


class SerialTrace:
    """Observe one serial connection without resetting between scenarios."""

    def __init__(self, serial_port, evidence: Evidence):
        self.serial = serial_port
        self.evidence = evidence
        self.started = time.monotonic()
        self.last_battery_signature = None
        self.soft_streak = 0
        self.power_messages: list[str] = []

    def next(self) -> Optional[SerialEvent]:
        raw = self.serial.readline().decode("utf-8", errors="replace").strip()
        if not raw:
            return None
        at = time.monotonic()
        power = parse_power_line(raw)
        battery = battery_from_line(raw)
        if battery is not None:
            if battery.pack_valid and battery.pack_region is PackVoltageRegion.SOFT_CUT:
                self.soft_streak += 1
                self.evidence.event(
                    "soft_sample",
                    "captured consecutive SoftCut frame",
                    pack_volts=battery.pack_volts,
                    state=int(battery.power_safety_state),
                    soft_streak=self.soft_streak,
                )
            else:
                self.soft_streak = 0
            signature = (
                int(battery.pack_region),
                battery.pack_valid,
                int(battery.power_safety_state),
            )
            if signature != self.last_battery_signature:
                self.evidence.event(
                    "batt_transition",
                    battery.format_compact(),
                    pack_volts=battery.pack_volts,
                    pack_valid=battery.pack_valid,
                    region=int(battery.pack_region),
                    state=int(battery.power_safety_state),
                    soft_streak=self.soft_streak,
                )
                self.last_battery_signature = signature
        if power is not None:
            self.power_messages.append(raw)
            self.evidence.event("pwr", raw, message=power[0], reason=power[1])
        return SerialEvent(raw, at, battery, power, self.soft_streak)

    def until(
        self,
        predicate: Callable[[SerialEvent], bool],
        timeout: float,
        description: str,
    ) -> SerialEvent:
        started = time.monotonic()
        while time.monotonic() - started < timeout:
            event = self.next()
            if event is not None and predicate(event):
                return event
        raise TimeoutError(f"timed out waiting for {description}")

    def quiet_after(
        self, origin: float, timeout: float, quiet_seconds: float = 1.5
    ) -> float:
        """Return seconds from origin once leader telemetry has stopped."""
        last_telemetry = time.monotonic()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            event = self.next()
            if event is not None and event.battery is not None:
                last_telemetry = event.at
            if time.monotonic() - last_telemetry >= quiet_seconds:
                elapsed = last_telemetry - origin
                self.evidence.event(
                    "telemetry_stopped",
                    "normal telemetry became quiet",
                    elapsed_s_from_action=round(elapsed, 3),
                )
                return elapsed
        raise TimeoutError("normal telemetry did not stop")

    def assert_no_resume(self, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            event = self.next()
            if event is None:
                continue
            if event.power and event.power[0] == "RESUMING":
                return False
            if event.battery and event.battery.power_safety_state is PowerSafetyState.OPERATING:
                return False
        return True


def _prompt_voltage(evidence: Evidence, volts: float, note: str = "") -> float:
    suffix = f" ({note})" if note else ""
    input(
        f"  Set and verify Pack VBUS at {volts:.1f} V{suffix}, then press Enter > "
    )
    now = time.monotonic()
    evidence.event(
        "stimulus_confirmed",
        f"operator confirmed Pack VBUS at {volts:.1f} V",
        volts=volts,
    )
    return now


def _wait_battery(
    trace: SerialTrace,
    region: PackVoltageRegion,
    state: PowerSafetyState,
    timeout: float = 8.0,
) -> SerialEvent:
    return trace.until(
        lambda event: event.battery is not None
        and event.battery.pack_valid
        and event.battery.pack_region is region
        and event.battery.power_safety_state is state,
        timeout,
        f"{region.name}/{state.name}",
    )


def _recover(trace: SerialTrace, evidence: Evidence) -> float:
    started = _prompt_voltage(evidence, SAFE_VOLTS, "strictly above 26.4 V")
    event = trace.until(
        lambda item: (
            item.power is not None and item.power[0] == "RESUMING"
        )
        or (
            item.battery is not None
            and item.battery.power_safety_state is PowerSafetyState.OPERATING
            and item.battery.pack_region is PackVoltageRegion.ABOVE_RECOVERY
        ),
        32.0,
        "recovery",
    )
    elapsed = event.at - started
    _wait_battery(
        trace,
        PackVoltageRegion.ABOVE_RECOVERY,
        PowerSafetyState.OPERATING,
        timeout=4.0,
    )
    evidence.event(
        "measurement",
        "recovery detected after operator confirmed 26.5 V",
        seconds_after_confirmation=round(elapsed, 3),
    )
    return elapsed


def _soft_cut(trace: SerialTrace, evidence: Evidence, acknowledge: bool) -> dict:
    starting_message_count = len(trace.power_messages)
    _prompt_voltage(evidence, SOFT_CUT_VOLTS)
    event = _wait_battery(
        trace,
        PackVoltageRegion.SOFT_CUT,
        PowerSafetyState.POWERING_DOWN,
    )
    powering_down_count = trace.power_messages[starting_message_count:].count(
        POWERING_DOWN
    )
    action_at = event.at
    actuators_held = _yes("did the actuators hold/de-energize at POWERING_DOWN?")
    ack_at = None
    if acknowledge:
        ack_at = time.monotonic()
        send_command(trace.serial, SHUTDOWN_ACK)
        evidence.event("tx", SHUTDOWN_ACK)
    origin = ack_at if ack_at is not None else action_at
    cutoff = trace.quiet_after(origin, 20.0 if acknowledge else 65.0)
    observed = _yes("did the relay open when telemetry stopped?")
    evidence.event("observation", "relay open", confirmed=observed)
    recovery = _recover(trace, evidence)
    return {
        "soft_streak": event.soft_streak,
        "powering_down_count": powering_down_count,
        "cutoff_seconds": round(cutoff, 3),
        "actuators_held": actuators_held,
        "relay_confirmed": observed,
        "recovery_seconds": round(recovery, 3),
    }


def _run_instrumented(port: str, evidence: Evidence) -> tuple[bool, str, dict]:
    print("\nInstrumented phase: the Orin must be cleanly off and its barrel unplugged.")
    if not _yes(f"is Pack VBUS {SAFE_VOLTS:.1f} V and the J401 barrel disconnected?"):
        return False, "safe serial-reset precondition not confirmed", {}

    evidence.clear_checkpoints("instrumented.")
    for criterion in ("4d", "4e", "4k"):
        evidence.criterion(criterion, "NOT RUN", "instrumented scenario not completed")
    checks = {}
    with open_bench_port(port) as serial_port:
        evidence.event("serial", "opened one long-lived Task 4 session", port=port)
        trace = SerialTrace(serial_port, evidence)

        baseline = _wait_battery(
            trace,
            PackVoltageRegion.ABOVE_RECOVERY,
            PowerSafetyState.OPERATING,
        )
        checks["baseline"] = baseline.battery.pack_valid
        dmm_volts = _number("DMM Pack VBUS reading at this baseline (V)")
        firmware_volts = baseline.battery.pack_volts
        send_command(serial_port, "V")
        version = trace.until(
            lambda item: item.raw.startswith("VER "),
            4.0,
            "firmware version",
        ).raw
        role = baseline.raw.split(";", 1)[0]
        metadata = evidence.data.setdefault("metadata", {})
        metadata.update(
            firmware_version=version,
            role=role,
            serial_port=port,
            serial_identity=port_identity(port),
        )
        metadata["fingerprint"] = hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode()
        ).hexdigest()[:16]
        evidence.checkpoint(
            "instrumented.baseline",
            checks["baseline"],
            dmm_volts=dmm_volts,
            firmware_volts=firmware_volts,
            delta_volts=round(firmware_volts - dmm_volts, 4),
            version=version,
            port=port,
            role=role,
            serial_identity=metadata["serial_identity"],
            fingerprint=metadata["fingerprint"],
        )

        warning_message_count = len(trace.power_messages)
        _prompt_voltage(evidence, WARN_VOLTS)
        _wait_battery(trace, PackVoltageRegion.WARN, PowerSafetyState.OPERATING)
        checks["warn"] = len(trace.power_messages) == warning_message_count
        evidence.checkpoint("instrumented.warn", checks["warn"])

        acked = _soft_cut(trace, evidence, acknowledge=True)
        checks["soft_ack"] = soft_cut_pass(acked, acknowledged=True)
        evidence.checkpoint("instrumented.soft_ack", checks["soft_ack"], **acked)

        unacked = _soft_cut(trace, evidence, acknowledge=False)
        checks["soft_no_ack"] = soft_cut_pass(unacked, acknowledged=False)
        evidence.checkpoint(
            "instrumented.soft_no_ack", checks["soft_no_ack"], **unacked
        )

        hard_message_count = len(trace.power_messages)
        hard_started = _prompt_voltage(evidence, HARD_CUT_VOLTS)
        hard = trace.until(
            lambda item: item.battery is not None
            and item.battery.pack_valid
            and item.battery.pack_region is PackVoltageRegion.HARD_CUT,
            8.0,
            "first HardCut frame",
        )
        hard_observed_after_confirmation = hard.at - hard_started
        oled_blank = _yes("did the OLED blank immediately and remain blank?")
        motors_held = _yes("did motor EN/PWM drop immediately without a park or handshake?")
        led_times = []
        for number in range(3):
            input(f"  press Enter at LED transition {number + 1}/3 > ")
            led_times.append(time.monotonic())
            evidence.event("observation", "hard-cut LED transition", number=number + 1)
        led_intervals = [b - a for a, b in zip(led_times, led_times[1:])]
        input("  press Enter when the relay opens at the hard-cut deadline > ")
        relay_elapsed = time.monotonic() - hard.at
        evidence.event(
            "observation",
            "hard-cut relay open",
            elapsed_s_from_action=round(relay_elapsed, 3),
        )
        no_hard_message = len(trace.power_messages) == hard_message_count
        hard_recovery = _recover(trace, evidence)
        checks["hard_cut"] = (
            hard.battery.power_safety_state is PowerSafetyState.SLEEP
            and no_hard_message
            and oled_blank
            and motors_held
            and len(led_intervals) == 2
            and all(8.0 <= value <= 12.0 for value in led_intervals)
            and 55.0 <= relay_elapsed <= 66.0
            and hard_recovery <= 31.0
        )
        evidence.checkpoint(
            "instrumented.hard_cut",
            checks["hard_cut"],
            first_frame_seconds_after_confirmation=round(
                hard_observed_after_confirmation, 3
            ),
            led_intervals=[round(value, 3) for value in led_intervals],
            relay_seconds=round(relay_elapsed, 3),
            recovery_seconds=round(hard_recovery, 3),
        )

        over_message_count = len(trace.power_messages)
        _prompt_voltage(evidence, OVER_VOLT_VOLTS, "approach slowly; do not overshoot")
        over_message = trace.until(
            lambda item: item.power is not None
            and item.power[0] == "EMERGENCY_SHUTDOWN",
            8.0,
            "over-voltage message",
        )
        over = _wait_battery(
            trace,
            PackVoltageRegion.OVER_VOLT,
            PowerSafetyState.OVER_VOLTAGE_LOCKOUT,
        )
        relay_immediate = _yes("did the relay open immediately on over-voltage?")
        over_motors_held = _yes("did motor EN/PWM drop immediately on over-voltage?")
        _prompt_voltage(evidence, SAFE_VOLTS, "do not reset yet")
        stayed_locked = trace.assert_no_resume(32.0)
        relay_stayed_open = _yes("after the 32 second hold, was the relay still open?")
        over_panel = _yes("did the OLED remain on the OVER VOLTAGE panel?")
        input("  At 26.5 V, press the Mega RESET button without closing serial, then Enter > ")
        evidence.event("operator", "Mega reset at safe voltage")
        reset_ok = _wait_battery(
            trace,
            PackVoltageRegion.ABOVE_RECOVERY,
            PowerSafetyState.OPERATING,
            timeout=12.0,
        ).battery.pack_valid
        checks["over_voltage"] = bool(
            over_message.power
            and trace.power_messages[over_message_count:].count(OVER_VOLTAGE) == 1
            and over.battery
            and relay_immediate
            and over_motors_held
            and stayed_locked
            and relay_stayed_open
            and over_panel
            and reset_ok
        )
        evidence.checkpoint(
            "instrumented.over_voltage",
            checks["over_voltage"],
            relay_immediate=relay_immediate,
            relay_stayed_open=relay_stayed_open,
            over_voltage_panel=over_panel,
            motors_held=over_motors_held,
            stayed_locked=stayed_locked,
            reset_ok=bool(reset_ok),
        )

    detail = "; ".join(f"{name}={'PASS' if ok else 'FAIL'}" for name, ok in checks.items())
    evidence.criterion(
        "4d",
        "PASS" if checks.get("hard_cut") else "FAIL",
        "HardCut first-frame Sleep, immediate hold/blank, LED cadence, delayed rail cut",
    )
    evidence.criterion(
        "4e",
        "PASS" if checks.get("over_voltage") else "FAIL",
        "OverVolt message, hold, immediate rail cut, and latched output",
    )
    evidence.criterion(
        "4k",
        "PASS" if all(checks.values()) else "FAIL",
        "bench-PSU warning, soft, hard, recovery, and over-voltage scenarios",
    )
    return all(checks.values()), detail, checks


def run(port: str, remote: str, results_path: str) -> int:
    evidence = Evidence(results_path)
    print("Task 4 is checkpointed between phases.")
    print(f"  summary: {evidence.summary_path}")
    print(f"  events : {evidence.events_path}")
    print(f"  report : {evidence.report_path}")

    if evidence.has_previous and any(
        result.get("status") == "PASS"
        for result in evidence.data.get("phases", {}).values()
    ):
        _, current_commit = _local(["git", "rev-parse", "HEAD"])
        _, current_status = _local(["git", "status", "--short"])
        saved = evidence.data.get("metadata", {})
        tree_matches = (
            saved.get("git_commit") == current_commit
            and saved.get("git_status") == current_status
            and saved.get("remote") == remote
        )
        reuse = tree_matches and _yes(
            "reuse prior PASS phases because boards, wiring, and instruments are unchanged?"
        )
        if not reuse:
            print("  prior phase verdicts invalidated for this attempt")
            evidence.fresh_attempt()
    evidence.event("attempt", "Task 4 bench attempt started")

    if evidence.status("static") != "PASS":
        print("\nSTATIC — identify this exact tree and run the native power-policy tests.")
        commit_ok, commit = _local(["git", "rev-parse", "HEAD"])
        status_ok, worktree = _local(["git", "status", "--short"])
        tests_ok, tests_output = _local(
            ["make", "-C", "firmware", "test-native"], timeout=180.0
        )
        metadata = {
            "operator": input("  operator name > ").strip(),
            "psu": input("  adjustable PSU make/model/asset ID > ").strip(),
            "dmm": input("  DMM make/model/asset ID > ").strip(),
            "git_commit": commit,
            "git_status": worktree,
            "remote": remote,
        }
        fingerprint = hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode()
        ).hexdigest()[:16]
        evidence.data["metadata"] = {**metadata, "fingerprint": fingerprint}
        evidence.event(
            "verification",
            "native firmware tests",
            ok=tests_ok,
            output=tests_output,
            fingerprint=fingerprint,
        )
        passed = commit_ok and status_ok and tests_ok
        evidence.phase(
            "static",
            "PASS" if passed else "FAIL",
            "native power policy and timing tests tied to the recorded tree",
            fingerprint=fingerprint,
            git_commit=commit,
            git_status=worktree,
        )
        evidence.criterion(
            "4a",
            "DEVIATION" if passed else "FAIL",
            "native boundary, debounce, and policy tests pass; constants are not "
            "in the grant-specified file",
        )
        if not passed:
            return 1

    # Phase 1: wiring and production-host readiness.
    if evidence.status("preflight") != "PASS":
        print("\nPRE-FLIGHT — use the S02 wiring sheet and start at 26.5 V.")
        wiring = _yes(
            "are Mega 7.5 V, relay H/D38/5 V, switched positive 12 V, direct "
            "negative, Pack INA 0x40, and J401 auto-power-on confirmed?"
        )
        oled_baseline = _yes("does the production OLED show a plausible 26.5 V baseline?")
        commands = {
            "path_unit": "systemctl is-active krabby-poweroff.path",
            "ready_file": "test -f /run/krabby/poweroff.ready && echo ready",
            "run_mount": (
                "docker inspect krabby --format "
                "'{{range .Mounts}}{{println .Source \"->\" .Destination}}{{end}}'"
            ),
            "daemon_armed": (
                "started=$(docker inspect -f '{{.State.StartedAt}}' krabby) && "
                "docker logs --since \"$started\" krabby 2>&1"
            ),
            "boot_id": "cat /proc/sys/kernel/random/boot_id",
        }
        remote_ok = bool(remote)
        outputs = {}
        for name, command in commands.items():
            if not remote:
                break
            ok, output = _ssh(remote, command)
            if name == "path_unit":
                ok = ok and output.strip() == "active"
            elif name == "ready_file":
                ok = ok and output.strip() == "ready"
            elif name == "run_mount":
                ok = ok and "/run/krabby -> /run/krabby" in output
            elif name == "daemon_armed":
                ok = ok and "Orin power daemon armed" in output
            elif name == "boot_id":
                ok = ok and len(output.strip()) == 36
            outputs[name] = output
            evidence.event("ssh", name, ok=ok, output=output)
            remote_ok = remote_ok and ok
        qualified = _yes(
            "has the one-time 30 minute relay/Mega 5 V and temperature qualification passed?"
        )
        passed = wiring and oled_baseline and remote_ok and qualified
        evidence.phase(
            "preflight",
            "PASS" if passed else "FAIL",
            "wiring, voltage comparison, host bridge, and relay qualification",
            wiring=wiring,
            oled_baseline=oled_baseline,
            host_bridge=remote_ok,
            relay_qualified=qualified,
            remote_outputs=outputs,
        )
        if not passed:
            print("Preflight failed; correct it before applying shutdown stimuli.")
            return 1

    # Phase 2: the Orin owns USB and performs the real graceful shutdown.
    if evidence.status("production") != "PASS":
        print("\nPRODUCTION E2E — Mega USB-B remains connected to the Orin.")
        if not _yes("is the Orin running at 26.5 V with Mega USB-B connected to it?"):
            evidence.phase("production", "FAIL", "production topology not confirmed")
            return 1
        boot_id_ok, tested_boot_id = _ssh(
            remote, "cat /proc/sys/kernel/random/boot_id"
        )
        if not boot_id_ok or len(tested_boot_id.strip()) != 36:
            evidence.phase("production", "FAIL", "could not bind evidence to boot ID")
            return 1
        evidence.event("ssh", "tested boot ID", boot_id=tested_boot_id.strip())
        time_ok, tested_at = _ssh(remote, "date --iso-8601=seconds")
        if not time_ok or not tested_at:
            evidence.phase("production", "FAIL", "could not timestamp tested boot")
            return 1
        soft_started = _prompt_voltage(evidence, SOFT_CUT_VOLTS)
        low_battery_oled = _yes("did the OLED enter LOW BATTERY before/at rail cut?")
        evidence.event(
            "observation", "production LOW BATTERY OLED", confirmed=low_battery_oled
        )
        input("  Press Enter when the relay opens and Orin green light turns off > ")
        relay_off_seconds = time.monotonic() - soft_started
        relay_off = _yes("were both the relay and Orin green light off?")
        evidence.event(
            "observation",
            "production relay/green off",
            confirmed=relay_off,
            seconds_from_stimulus=round(relay_off_seconds, 3),
        )
        ssh_unreachable = (
            _wait_for_ssh(_remote_host(remote), False, 15.0) is not None
            if remote
            else False
        )
        recovery_started = _prompt_voltage(evidence, SAFE_VOLTS)
        input("  Press Enter when the relay closes > ")
        relay_on_seconds = time.monotonic() - recovery_started
        relay_on = _yes("did relay closure occur on this recovery attempt?")
        ssh_up_after_relay = (
            _wait_for_ssh(_remote_host(remote), True, 180.0) if remote else None
        )
        journal_ok = True
        journals = {}
        boot_changed = False
        production_ready = False
        if remote and ssh_up_after_relay is not None:
            for name, command in {
                "poweroff_service": "journalctl -b -1 -u krabby-poweroff.service --no-pager",
                "previous_boot_tail": "journalctl -b -1 -e -n 80 --no-pager",
                "previous_locomotion": (
                    "journalctl -b -1 -u krabby-locomotion.service --no-pager"
                ),
                "current_daemon_log": (
                    "started=$(docker inspect -f '{{.State.StartedAt}}' krabby) && "
                    "docker logs --since \"$started\" krabby 2>&1"
                ),
                "container_running": "docker inspect krabby --format '{{.State.Running}}'",
                "boot_id": "cat /proc/sys/kernel/random/boot_id",
                "boot_list": "journalctl --list-boots --no-pager",
            }.items():
                ok, output = _ssh(remote, command)
                if name == "previous_locomotion":
                    ok = ok and previous_shutdown_log_pass(output)
                elif name == "current_daemon_log":
                    ok = ok and current_daemon_log_pass(output)
                    production_ready = ok
                elif name == "container_running":
                    ok = ok and output.strip() == "true"
                    production_ready = production_ready and ok
                elif name == "boot_id":
                    boot_changed = (
                        ok
                        and output.strip() != tested_boot_id.strip()
                    )
                    ok = ok and boot_changed
                elif name == "boot_list":
                    ok = ok and _boot_id_is_listed(tested_boot_id, output)
                elif name in (
                    "poweroff_service",
                    "previous_boot_tail",
                ):
                    ok = ok and journal_has_entries(output)
                journals[name] = output
                evidence.event("ssh", name, ok=ok, output=output)
                journal_ok = journal_ok and ok and bool(output)
        clean = _yes("do the captured prior-boot logs show a clean requested shutdown?")
        ack_path_timing = relay_off_seconds <= 30.0
        production_values = {
            "low_battery_oled": low_battery_oled,
            "ssh_unreachable": ssh_unreachable,
            "relay_off": relay_off,
            "relay_off_seconds": relay_off_seconds,
            "relay_on": relay_on,
            "relay_on_seconds": relay_on_seconds,
            "ssh_returned": ssh_up_after_relay is not None,
            "journal_ok": journal_ok,
            "boot_changed": boot_changed,
            "production_ready": production_ready,
            "clean_shutdown": clean,
        }
        passed = production_pass(production_values)
        evidence.phase(
            "production",
            "PASS" if passed else "FAIL",
            "production daemon shutdown, relay cut, auto-boot, and prior-boot evidence",
            ssh_unreachable=ssh_unreachable,
            ssh_port_return_seconds_after_relay=ssh_up_after_relay,
            low_battery_oled=low_battery_oled,
            relay_off=relay_off,
            relay_off_seconds=round(relay_off_seconds, 3),
            relay_on=relay_on,
            relay_on_seconds=round(relay_on_seconds, 3),
            clean_shutdown=clean,
            ack_path_timing=ack_path_timing,
            boot_changed=boot_changed,
            production_ready=production_ready,
            journals=journals,
            tested_boot_id=tested_boot_id.strip(),
            tested_at=tested_at.strip(),
        )
        evidence.criterion(
            "4j",
            "PASS" if passed else "FAIL",
            "production daemon ACK, host shutdown, reboot, and MCU reconnect evidence",
        )
        if not passed:
            return 1

    # Phase 3: one Mac-owned serial connection, with the Orin load disconnected.
    if evidence.status("instrumented") != "PASS":
        print("\nPREPARE: sudo poweroff the Orin, wait for green-off, unplug its barrel,")
        print("then move Mega USB-B to this Mac. Do not open another serial monitor.")
        if not _yes("is the Orin off, J401 barrel unplugged, USB on Mac, and VBUS 26.5 V?"):
            evidence.phase("instrumented", "FAIL", "instrumented topology not confirmed")
            return 1
        try:
            passed, detail, values = _run_instrumented(port, evidence)
        except (OSError, TimeoutError, ValueError, KeyboardInterrupt) as exc:
            passed, detail, values = False, f"{type(exc).__name__}: {exc}", {}
            print("\nInstrumented phase interrupted. Do not reset at an unknown voltage.")
            measured = _confirm_safe_voltage()
            evidence.event(
                "safety",
                "operator restored 26.5 V after interruption",
                measured_volts=measured,
            )
        evidence.phase(
            "instrumented",
            "PASS" if passed else "FAIL",
            detail,
            checks=values,
        )
        if not passed:
            return 1

    # Phase 4: return the single USB and switched supply to production.
    if evidence.status("restore") != "PASS":
        print("\nRESTORE — voltage must remain at 26.5 V.")
        print("Move Mega USB-B to the Orin, then reconnect the J401 barrel output.")
        restored = _yes("did the Orin auto-boot and reconnect to the production serial bridge?")
        evidence.phase(
            "restore",
            "PASS" if restored else "FAIL",
            "production topology restored; deferred/deviation ledger retained",
        )
        if not restored:
            return 1

    print(f"\nBENCH PROCEDURE PASS — report: {evidence.report_path}")
    print("This is not full M16 acceptance; partial, deferred, and deviating ACs remain.")
    return 0


def send_command(port, command):
    port.write((command + "\n").encode())
    port.flush()
    time.sleep(0.15)


def open_bench_port(path):
    port = serial.Serial(path, DEFAULT_BAUD, timeout=0.4)
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        line = port.readline().decode("utf-8", errors="replace")
        if TelemetryFrame.role_from_line(line) is not None:
            port.reset_input_buffer()
            break
    return port
