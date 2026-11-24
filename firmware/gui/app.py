from __future__ import annotations

import math
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import threading
import time
from typing import Dict, List, NamedTuple, Optional

from firmware.krabby_mcu import (
    CAL_DEFAULT_PWM,
    CAL_HIP_NUDGE_EDGES,
    CAL_HIP_SWEEP_EDGES,
    CAL_KNEE_NUDGE_EDGES,
    CAL_PWM_MAX,
    CAL_PWM_MIN,
    DEFAULT_BAUD,
    FULL_CALIBRATION_STATES,
    JOINT_GROUP_NAMES,
    KrabbyMCUSDK,
    cal_line_state,
    cal_robot_finished,
    cal_status_line,
    curl_button,
    is_hip_or_knee,
    parse_calibration_pwm,
)
from firmware.mcu_port import McuSerialTarget
from firmware.interfaces.joint_telemetry import JointTelemetry

JOG_PWM = 200  # jog magnitude sent while a Retract/Extend button is held
# Firmware convention (driveActuator): +PWM extends the rod, -PWM retracts it. The
# Retract button must therefore send -JOG_PWM and the Extend button +JOG_PWM.
RETRACT_DIRECTION = -1
EXTEND_DIRECTION = +1
TELEMETRY_REFRESH_MS = (
    100  # GUI poll period; decoupled from the firmware's telemetry tick
)
ROLE_STALE_S = 1.0  # a board counts as present if its telemetry arrived within this window

# Placeholder for a joint cell before its first telemetry arrives.
NO_VALUE_TEXT = "---"

# Fonts: (family, size[, style]).
FONT_STATUS = ("Segoe UI", 10)  # connection status line
FONT_TABLE_HEADER = ("Segoe UI", 9, "bold")
FONT_GROUP_LABEL = ("Segoe UI", 9, "italic")  # FRONT/LEFT/RIGHT dividers
FONT_JOINT_NAME = ("Consolas", 11, "bold")  # monospace so names align
GROUP_LABEL_COLOR = "#666"  # muted gray for the dividers

FONT_SENSOR_LABEL = FONT_JOINT_NAME  # Consolas 11 bold, col-0 entity label
FONT_SENSOR_HEADER = FONT_TABLE_HEADER  # Segoe 9 bold caption row
FONT_SENSOR_VALUE = ("Consolas", 10)  # monospace tabular numbers, anchor e
STATE_COLOR_OK = "#2e7d32"
STATE_COLOR_STALE = "#c0392b"
STATE_COLOR_PENDING = "#b26a00"  # amber: an encoder joint that needs homing or a stroke
STATE_COLOR_MUTED = "#808080"  # grey: no value to show
DASH = "—"  # em dash: no position or reading for this joint
NEUTRAL_NOTE_S = 10.0  # how long the Neutral result stays on the status line
CALIBRATION_RUNNING_NOTE = (
    "Calibrate: a run is still in progress - wait for it to end, or press Hold All"
)

# Calibrate dialog (Mega path). The curl directions quoted here come from
# firmware.krabby_mcu.curl_button(), which mirrors calibration_constants.h.
# The counts quoted here are CAL_HIP_SWEEP_EDGES, CAL_HIP_NUDGE_EDGES and
# CAL_KNEE_NUDGE_EDGES; the knee nudge is the shipped default (CAL_DEFAULT_OPTIONS).
# A curled knee springs out 5-9 counts after its nudge stops (measured on FLKL).
CALIBRATE_EXPLANATION = (
    "Calibration finds each hip's and knee's travel limits, moving ONE joint at a "
    "time at a slow, ramped speed. Yaw joints are never moved. Each joint's sensor "
    "(pot or encoder) is detected automatically.\n"
    "\n"
    "1. Curl: every leg on the whole robot is curled first (knee, then hip; FRONT "
    "board, then LEFT, then RIGHT). The hip curls on Retract, the knee on Extend.\n"
    "2. Sweep: only after the whole robot is curled, each hip is tested while its "
    "knee is curled and each knee while its hip is curled. Each hip goes out about "
    f"70% ({CAL_HIP_SWEEP_EDGES} hall counts) and back, not to full extension, because "
    "a curled tibia meets the body; position control then stays inside that 70%. "
    "Each knee goes fully out and back. A board's limits are saved only when its "
    "sweep finishes with no problem.\n"
    "\n"
    "Flash all three boards together with the same firmware. The run will not "
    "start while a board has not reported its sensors yet: wait a few seconds "
    "after connecting.\n"
    "\n"
    "Tick \"Curl only\" to run just the curl pass on every board: nothing is swept or "
    "saved. Encoder joints must be homed with Curl only after every power-up before "
    "Neutral works.\n"
    "\n"
    "After a power-up most hips and knees rest fully curled. A hip or knee resting "
    "fully curled is nudged out a little and straight back so its sensor can be "
    f"checked (at most {CAL_HIP_NUDGE_EDGES} hall counts for a hip, "
    f"{CAL_KNEE_NUDGE_EDGES} for a knee). A curled knee springs a little further "
    "after its nudge stops, so it ends about 20 counts out before it curls back; "
    "that is expected. A joint that still shows no movement stops the run, and "
    "the run names it. If a knee stops with \"curled back only\", run Curl only "
    "again.\n"
    "\n"
    f"Use the default speed (PWM {CAL_DEFAULT_PWM}): ends are found from motor current "
    "and motion; below about PWM 110 a joint may stall short of its end and the run "
    "stops it as stuck mid-stroke. A joint slower than about 20 counts per second "
    "(PWM 40-50) never shows a running phase: its ends are found by stall alone, as "
    "before, and a long pause can then pass for its end.\n"
    "\n"
    "Run the full calibration with the robot on its stand. Keep the legs clear of "
    "the ground and anything else. Hold All stops everything at once and saves "
    "nothing for the board in progress."
)
CAL_STATUS_COLORS = {"fail": STATE_COLOR_STALE, "done": STATE_COLOR_OK}


def joint_button_labels(joint_name: str) -> tuple[str, str]:
    """(retract_text, extend_text) for a joint row; the curl button gets a "(curl)" tag."""
    curl = curl_button(joint_name)
    retract = "\u25c0 Retract" + (" (curl)" if curl == "retract" else "")
    extend = "Extend" + (" (curl)" if curl == "extend" else "") + " \u25b6"
    return retract, extend


# Joint grid columns, left to right; JointRow grids its widgets in this order.
JOINT_GRID_HEADERS = (
    "Joint", "Retract", "Extend", "Sensor", "Pos", "Pot", "Cur", "PWM", "Hall"
)
# Sensor column text and colour per SENS state code (see krabby_mcu.SENSOR_STATE_CODES).
SENSOR_CELLS = {
    "P": ("POT", ""),
    "E": ("ENC", ""),
    "e": ("ENC unhomed", STATE_COLOR_PENDING),
    "s": ("ENC no stroke", STATE_COLOR_PENDING),
    "x": ("ENC home lost", STATE_COLOR_STALE),
    "?": ("?", STATE_COLOR_MUTED),
}
# States whose joint has an encoder: the pot pin carries hall B, so the Pot cell is blank.
ENCODER_STATES = ("E", "e", "s", "x")
# States that show a position when the telemetry has one (pot, or encoder ready for T).
POSITION_STATES = (None, "P", "E")
# States in which a hip or knee has no position yet, so Neutral and T skip it: an encoder
# not ready (e, s, x), or a sensor not yet detected (?: after a fresh flash, or with an
# ignored, corrupt or invalid calibration record).
NOT_READY_STATES = ("e", "s", "x", "?")


class JointCells(NamedTuple):
    """Text and colour of the Sensor, Pos and Pot cells of one joint row."""

    sensor: str
    sensor_colour: str
    pos: str
    pot: str
    pot_colour: str


def sensor_cell(state: Optional[str]) -> tuple[str, str]:
    """(text, colour) of the Sensor cell; blank when no SENS line has been seen."""
    if state is None:
        return "", ""
    return SENSOR_CELLS.get(state, ("?", STATE_COLOR_MUTED))


def joint_cells(state: Optional[str], jt: JointTelemetry) -> JointCells:
    """Sensor, Pos and Pot cells for a joint's SENS ``state`` (None: no SENS seen)."""
    sensor, sensor_colour = sensor_cell(state)
    shows_pos = state in POSITION_STATES and math.isfinite(jt.pos)
    pos = f"{jt.pos:.3f}" if shows_pos else DASH
    if state in ENCODER_STATES:
        return JointCells(sensor, sensor_colour, pos, DASH, STATE_COLOR_MUTED)
    return JointCells(sensor, sensor_colour, pos, str(jt.pot), "")


def calibration_advice(states: Dict[str, str], names: List[str]) -> str:
    """Which Calibrate run gives ``names`` a position: "full calibration" or "Curl only".

    Curl only is enough only when no named joint lacks a saved stroke or a sensor kind.
    """
    if any(states.get(n) in FULL_CALIBRATION_STATES for n in names):
        return "full calibration"
    return "Curl only"


def encoder_banner(states: Dict[str, str]) -> tuple[str, str]:
    """(text, colour) warning about hips and knees with no position yet; empty if none.

    Yaws are ignored: they are never calibrated or moved by Neutral.
    """
    names = [
        name
        for _, joint_names in JOINT_GROUP_NAMES
        for name in joint_names
        if is_hip_or_knee(name) and states.get(name) in NOT_READY_STATES
    ]
    if not names:
        return "", ""
    unknown = [n for n in names if states[n] == "?"]
    encoders = [n for n in names if states[n] != "?"]
    parts = []
    if unknown:
        parts.append(f"Sensor not yet detected: {', '.join(unknown)}")
    if encoders:
        parts.append(f"Encoder joints not ready: {', '.join(encoders)}")
    colour = (
        STATE_COLOR_STALE
        if any(states[n] == "x" for n in names)
        else STATE_COLOR_PENDING
    )
    return (
        f"{'. '.join(parts)}. Run Calibrate > {calibration_advice(states, names)}",
        colour,
    )


def neutral_note(
    sent: int, skipped: List[str], states: Optional[Dict[str, str]] = None
) -> str:
    """Status-line text after Neutral; empty when every hip and knee was sent.

    ``states`` are the SENS codes: when nothing was sent, the advice names the run
    that gives the skipped joints a position (as the banner does).
    """
    if sent == 0:
        action = calibration_advice(states or {}, skipped)
        return f"Neutral: no hip or knee has a position yet - run Calibrate ({action})"
    if skipped:
        return f"Neutral sent to {sent} joints; no position, skipped: {', '.join(skipped)}"
    return ""


class JointRow:
    """One row in the telemetry grid: name, jog buttons, live values."""

    def __init__(self, parent: tk.Widget, name: str, row: int, jog_cb):
        self.name = name
        self._jog_cb = jog_cb
        self._active_dir = 0

        self.lbl_name = ttk.Label(parent, text=name, font=FONT_JOINT_NAME, width=6)
        self.lbl_name.grid(row=row, column=0, padx=4, pady=2, sticky="w")

        retract_text, extend_text = joint_button_labels(name)
        self.btn_retract = ttk.Button(parent, text=retract_text, width=16)
        self.btn_retract.grid(row=row, column=1, padx=2, pady=2)
        self.btn_retract.bind(
            "<ButtonPress-1>", lambda e: self._start_jog(RETRACT_DIRECTION)
        )
        self.btn_retract.bind("<ButtonRelease-1>", lambda e: self._stop_jog())

        self.btn_extend = ttk.Button(parent, text=extend_text, width=16)
        self.btn_extend.grid(row=row, column=2, padx=2, pady=2)
        self.btn_extend.bind(
            "<ButtonPress-1>", lambda e: self._start_jog(EXTEND_DIRECTION)
        )
        self.btn_extend.bind("<ButtonRelease-1>", lambda e: self._stop_jog())

        self.var_sensor = tk.StringVar(value="")
        self.var_pos = tk.StringVar(value=NO_VALUE_TEXT)
        self.var_pot = tk.StringVar(value=NO_VALUE_TEXT)
        self.var_cur = tk.StringVar(value=NO_VALUE_TEXT)
        self.var_pwm = tk.StringVar(value=NO_VALUE_TEXT)
        self.var_hall = tk.StringVar(value=NO_VALUE_TEXT)

        # Columns: 3 Sensor, 4 Pos, 5 Pot, 6 Cur, 7 PWM, 8 Hall. The Sensor and Pot
        # labels are kept so their colour can follow the sensor state.
        self.lbl_sensor = ttk.Label(parent, textvariable=self.var_sensor, width=14)
        self.lbl_sensor.grid(row=row, column=3, padx=4, sticky="w")
        ttk.Label(parent, textvariable=self.var_pos, width=6, anchor="e").grid(
            row=row, column=4, padx=4
        )
        self.lbl_pot = ttk.Label(parent, textvariable=self.var_pot, width=6, anchor="e")
        self.lbl_pot.grid(row=row, column=5, padx=4)
        ttk.Label(parent, textvariable=self.var_cur, width=6, anchor="e").grid(
            row=row, column=6, padx=4
        )
        ttk.Label(parent, textvariable=self.var_pwm, width=10, anchor="e").grid(
            row=row, column=7, padx=4
        )
        ttk.Label(parent, textvariable=self.var_hall, width=7, anchor="e").grid(
            row=row, column=8, padx=4
        )

    def _start_jog(self, direction: int):
        self._active_dir = direction
        self._jog_cb(self.name, direction * JOG_PWM)

    def _stop_jog(self):
        self._active_dir = 0
        self._jog_cb(self.name, 0)

    def update_from_telemetry(
        self, jt: Optional[JointTelemetry], sensor_state: Optional[str] = None
    ):
        """Refresh the cells. ``sensor_state`` is the joint's SENS code, None when no
        SENS line has been seen (old firmware, Rev 2): Sensor is then blank."""
        if jt is None:
            if sensor_state is not None:
                text, colour = sensor_cell(sensor_state)
                self.var_sensor.set(text)
                self.lbl_sensor.configure(foreground=colour)
            return
        cells = joint_cells(sensor_state, jt)
        self.var_sensor.set(cells.sensor)
        self.lbl_sensor.configure(foreground=cells.sensor_colour)
        self.var_pos.set(cells.pos)
        self.var_pot.set(cells.pot)
        self.lbl_pot.configure(foreground=cells.pot_colour)
        self.var_cur.set(str(jt.current))
        self.var_pwm.set(f"L{jt.pwm[0]} R{jt.pwm[1]}")
        self.var_hall.set(str(jt.saf))


class ImuRow:
    """Global IMU readout in the joint-grid idiom (one per leader board, not per
    joint -> its own block, not a joint column). Caption row above, monospace
    anchor-east value cells below, matching the joint table's fonts/alignment."""

    COLS = [
        "",
        "roll°",
        "pitch°",
        "aX g",
        "aY",
        "aZ",
        "gX °/s",
        "gY",
        "gZ",
        "die°C",
        "state",
    ]

    @staticmethod
    def resolve_state(imu) -> tuple[str, str]:
        if imu is None:
            return "—", ""
        if not imu.valid:
            return "STALE", STATE_COLOR_STALE
        return "fresh", STATE_COLOR_OK

    def __init__(self, parent: tk.Widget):
        ttk.Label(parent, text="IMU", font=FONT_SENSOR_LABEL, width=6, anchor="w").grid(
            row=1, column=0, padx=4, pady=2, sticky="w"
        )
        for c, h in enumerate(self.COLS):
            if not h:
                continue
            ttk.Label(parent, text=h, font=FONT_SENSOR_HEADER, anchor="e").grid(
                row=0, column=c, padx=4, sticky="e"
            )
        self._vars = [tk.StringVar(value="—") for _ in self.COLS]
        for c in range(1, len(self.COLS)):
            ttk.Label(
                parent,
                textvariable=self._vars[c],
                font=FONT_SENSOR_VALUE,
                width=7,
                anchor="e",
            ).grid(row=1, column=c, padx=4)
        self._state_lbl = parent.grid_slaves(row=1, column=len(self.COLS) - 1)[0]

    def update(self, imu):
        if imu is None:
            for v in self._vars[1:]:
                v.set("—")
            self._state_lbl.configure(foreground="")
            return
        ag, gd = imu.accel_g, imu.gyro_dps
        fmt = [
            None,
            f"{imu.roll_from_accel_deg:+.1f}",
            f"{imu.pitch_from_accel_deg:+.1f}",
            f"{ag[0]:+.2f}",
            f"{ag[1]:+.2f}",
            f"{ag[2]:+.2f}",
            f"{gd[0]:+.1f}",
            f"{gd[1]:+.1f}",
            f"{gd[2]:+.1f}",
            f"{imu.temp_c:.1f}",
        ]
        for c in range(1, 10):
            self._vars[c].set(fmt[c])
        s, col = self.resolve_state(imu)
        self._vars[10].set(s)
        self._state_lbl.configure(foreground=col)



def _state_label(value) -> str:
    """Defined values show their name; an unknown byte shows the raw number so a
    newer firmware is visible rather than silently mapped onto something known."""
    return value.name if hasattr(value, "name") else str(value)


SENSOR_STALE_S = 1.0


class BattRow:
    """Display battery readings, divergence, measurement validity, and frame age.

    Retain samples between updates. Divergence includes assumed divergence when
    a battery cannot be read; successful reads may still need numeric checks.
    """

    COLS = ["", "pack V", "pack A", "pack W", "charge C", "battA", "battB",
            "region", "diverge", "pack", "mid", "freshness"]
    # Leave room for DIVERGED and the longer region labels.
    COL_WIDTHS = {"region": 8, "diverge": 9, "pack": 6, "mid": 6, "freshness": 9}
    # Status columns have independent colours.
    COLOURED = ("diverge", "pack", "mid", "freshness")

    @staticmethod
    def resolve_state(battery, age_seconds: Optional[float]) -> tuple[str, str]:
        """How old the latched sample is. Says nothing about its contents."""
        if battery is None:
            return "—", ""
        if age_seconds is not None and age_seconds > SENSOR_STALE_S:
            return "stale", STATE_COLOR_STALE
        return "fresh", STATE_COLOR_OK

    @staticmethod
    def resolve_monitor(valid: Optional[bool]) -> tuple[str, str]:
        """Display whether all reads in the monitor measurement succeeded."""
        if valid is None:
            return "—", ""
        return ("up", STATE_COLOR_OK) if valid else ("DOWN", STATE_COLOR_STALE)

    @staticmethod
    def resolve_divergence(battery) -> tuple[str, str]:
        """Keep assumed divergence visible even when a monitor is unavailable."""
        if battery is None:
            return "—", ""
        if battery.divergence:
            return "DIVERGED", STATE_COLOR_STALE
        if not battery.split_available:
            return "—", ""
        return "ok", STATE_COLOR_OK

    @staticmethod
    def latch_sample(previous_sample, previous_timestamp, sample, now):
        if sample is not None and sample is not previous_sample:
            return sample, now
        return previous_sample, previous_timestamp

    def __init__(self, parent: tk.Widget):
        self._sample = None
        self._sample_timestamp: Optional[float] = None
        ttk.Label(parent, text="BATT", font=FONT_SENSOR_LABEL, width=6, anchor="w").grid(
            row=3, column=0, padx=4, pady=2, sticky="w"
        )
        for c, h in enumerate(self.COLS):
            if not h:
                continue
            ttk.Label(parent, text=h, font=FONT_SENSOR_HEADER, anchor="e").grid(
                row=2, column=c, padx=4, sticky="e"
            )
        self._vars = [tk.StringVar(value="—") for _ in self.COLS]
        for c in range(1, len(self.COLS)):
            ttk.Label(
                parent,
                textvariable=self._vars[c],
                font=FONT_SENSOR_VALUE,
                width=self.COL_WIDTHS.get(self.COLS[c], 7),
                anchor="e",
            ).grid(row=3, column=c, padx=4)
        self._lbl = {name: parent.grid_slaves(row=3, column=self.COLS.index(name))[0]
                     for name in self.COLOURED}

    def update(self, battery, now: float):
        self._sample, self._sample_timestamp = BattRow.latch_sample(
            self._sample, self._sample_timestamp, battery, now
        )
        battery = self._sample
        age_seconds = (
            None if self._sample_timestamp is None else now - self._sample_timestamp
        )
        if battery is None:
            for v in self._vars[1:]:
                v.set("—")
            for lbl in self._lbl.values():
                lbl.configure(foreground="")
            return
        fmt = [
            None,
            f"{battery.pack_volts:.2f}",
            f"{battery.pack_current_amperes:+.2f}",
            f"{battery.pack_power_watts:.1f}",
            f"{battery.pack_charge_coulombs:.0f}",
            battery.format_battery_voltage(battery.battery_a_volts),
            battery.format_battery_voltage(battery.battery_b_volts),
            _state_label(battery.pack_region),
        ]
        for c, text in enumerate(fmt):
            if text is not None:
                self._vars[c].set(text)
        for name, (text, colour) in (
            ("diverge", self.resolve_divergence(battery)),
            ("pack", self.resolve_monitor(battery.pack_valid)),
            ("mid", self.resolve_monitor(battery.midpoint_valid)),
            ("freshness", self.resolve_state(battery, age_seconds)),
        ):
            self._vars[self.COLS.index(name)].set(text)
            self._lbl[name].configure(foreground=colour)


CAL_LABEL_PADDING_PX = 16  # the label's horizontal padding (8 each side)
CAL_LABEL_MIN_WRAP_PX = 200


def cal_label_wraplength(width: int) -> int:
    """Wrap width in pixels for the calibration status label at ``width`` pixels wide."""
    return max(CAL_LABEL_MIN_WRAP_PX, int(width) - CAL_LABEL_PADDING_PX)


def cal_status_text(line: Optional[str]) -> tuple[str, str]:
    """(label text, colour) for the latest CAL line; empty text when there is none."""
    if not line:
        return "", ""
    return f"Calibration: {line}", CAL_STATUS_COLORS.get(cal_line_state(line), "")


class CalibrationChoice(NamedTuple):
    """What the Calibrate dialog returns: the speed and which pass(es) to run."""

    pwm: int
    passes: Optional[str]  # "SEEK" for curl only; None for the full SEEK + SWEEP run


class CalibrateDialog(tk.Toplevel):
    """Modal Calibrate dialog: the sequence explained, a speed field and a curl-only box.

    ``self.result`` is a CalibrationChoice after Start (passes "SEEK" when Curl only is
    ticked, else None), otherwise None. Curl only is unticked each time the dialog opens.
    """

    def __init__(self, parent: tk.Widget):
        super().__init__(parent)
        self.title("Calibrate")
        self.resizable(False, False)
        self.transient(parent)
        self.result: Optional[CalibrationChoice] = None

        body = ttk.Frame(self, padding=12)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=CALIBRATE_EXPLANATION, wraplength=460, justify="left").pack(
            anchor="w"
        )

        speed_row = ttk.Frame(body)
        speed_row.pack(anchor="w", pady=(12, 0))
        ttk.Label(
            speed_row, text=f"Speed (PWM {CAL_PWM_MIN}-{CAL_PWM_MAX}, lower is slower):"
        ).pack(side="left")
        self._speed_var = tk.StringVar(value=str(CAL_DEFAULT_PWM))
        self._entry = ttk.Entry(speed_row, textvariable=self._speed_var, width=6)
        self._entry.pack(side="left", padx=(6, 0))

        self._curl_only_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            body,
            text="Curl only: curl every leg, then stop (no sweep, saves nothing)",
            variable=self._curl_only_var,
        ).pack(anchor="w", pady=(8, 0))

        self._error_var = tk.StringVar(value="")
        ttk.Label(body, textvariable=self._error_var, foreground=STATE_COLOR_STALE).pack(
            anchor="w", pady=(4, 0)
        )

        buttons = ttk.Frame(body)
        buttons.pack(anchor="e", pady=(8, 0))
        ttk.Button(buttons, text="Start calibration", command=self._on_start).pack(
            side="left", padx=4
        )
        ttk.Button(buttons, text="Cancel", command=self._on_cancel).pack(
            side="left", padx=4
        )

        self.bind("<Return>", lambda e: self._on_start())
        self.bind("<Escape>", lambda e: self._on_cancel())
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self._entry.focus_set()
        self._entry.selection_range(0, "end")
        self.grab_set()

    def _on_start(self):
        try:
            pwm = parse_calibration_pwm(self._speed_var.get())
        except ValueError as exc:
            self._error_var.set(str(exc))
            return
        self.result = CalibrationChoice(
            pwm, "SEEK" if self._curl_only_var.get() else None
        )
        self.destroy()

    def _on_cancel(self):
        self.result = None
        self.destroy()


def ask_calibration_settings(parent: tk.Widget) -> Optional[CalibrationChoice]:
    """Show the Calibrate dialog; return the speed and passes, or None if cancelled."""
    dialog = CalibrateDialog(parent)
    parent.wait_window(dialog)
    return dialog.result


def create_mcu_sdk(
    *,
    stm: bool = False,
    virtual: bool = False,
    target: McuSerialTarget | None = None,
    baud: int = DEFAULT_BAUD,
):
    if stm:
        from firmware.gui.stm_adapter import KrabbyStmGuiAdapter

        return KrabbyStmGuiAdapter(virtual=virtual)
    return KrabbyMCUSDK(target=target, baud=baud)


class KrabbyTestGUI(tk.Tk):
    # True from a sent C until its robot end line or a Hold All press. A class default,
    # because tk.Tk.__getattr__ would recurse on an unset attribute.
    _cal_awaiting_end = False

    def __init__(
        self,
        target: McuSerialTarget | None = None,
        baud: int = DEFAULT_BAUD,
        *,
        stm: bool = False,
        virtual: bool = False,
    ):
        super().__init__()
        self.title("Krabby MCU Test")
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._stm = stm
        self._mcu = create_mcu_sdk(
            stm=stm, virtual=virtual, target=target, baud=baud
        )
        self._joint_rows: Dict[str, JointRow] = {}
        self._connected = False
        self._neutral_text = ""  # result of the last Neutral press, shown for a while
        self._neutral_until = 0.0
        self._cal_awaiting_end = False

        self._build_ui()
        self._connect()

    def _build_ui(self):
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")

        self._status_var = tk.StringVar(value="Connecting...")
        ttk.Label(top, textvariable=self._status_var, font=FONT_STATUS).pack(
            side="left"
        )
        self._role_var = tk.StringVar(value="Role: ---")
        ttk.Label(top, textvariable=self._role_var, font=FONT_STATUS).pack(
            side="left", padx=(16, 0)
        )

        btn_frame = ttk.Frame(top)
        btn_frame.pack(side="right")
        ttk.Button(btn_frame, text="Hold All", command=self._hold_all).pack(
            side="left", padx=4
        )
        if not self._stm:
            ttk.Button(btn_frame, text="Neutral (0.5)", command=self._neutral).pack(
                side="left", padx=4
            )
        ttk.Button(btn_frame, text="Calibrate", command=self._calibrate).pack(
            side="left", padx=4
        )

        # Latest calibration progress line (Mega path only; the STM adapter has none).
        self._cal_var = tk.StringVar(value="")
        if not self._stm:
            self._cal_label = ttk.Label(
                self, textvariable=self._cal_var, font=FONT_STATUS, padding=(8, 0)
            )
            self._cal_label.pack(fill="x")
            # Wrap to the window width so a long line (the 117-character "no SENS"
            # refusal, an echoed follower reason) is shown whole, never clipped.
            self._cal_label.bind("<Configure>", self._wrap_cal_label)
            # Encoder joints that still need homing (from the SENS lines).
            self._banner_var = tk.StringVar(value="")
            self._banner_label = ttk.Label(
                self, textvariable=self._banner_var, font=FONT_STATUS, padding=(8, 0)
            )
            self._banner_label.pack(fill="x")

        imu_frame = ttk.Frame(self, padding=(8, 0))
        imu_frame.pack(fill="x")
        self._imu_row = ImuRow(imu_frame)
        self._batt_row = BattRow(imu_frame)

        sep = ttk.Separator(self, orient="horizontal")
        sep.pack(fill="x", pady=4)

        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self._grid_frame = ttk.Frame(canvas, padding=8)

        self._grid_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self._grid_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for c, h in enumerate(JOINT_GRID_HEADERS):
            ttk.Label(
                self._grid_frame, text=h, font=FONT_TABLE_HEADER, anchor="center"
            ).grid(row=0, column=c, padx=4, pady=(0, 4), sticky="ew")

        row = 1
        for group_name, joint_names in JOINT_GROUP_NAMES:
            ttk.Label(
                self._grid_frame,
                text=f"── {group_name} ──",
                font=FONT_GROUP_LABEL,
                foreground=GROUP_LABEL_COLOR,
            ).grid(
                row=row,
                column=0,
                columnspan=len(JOINT_GRID_HEADERS),
                sticky="w",
                pady=(6, 2),
            )
            row += 1
            for jname in joint_names:
                jr = JointRow(self._grid_frame, jname, row, self._jog_joint)
                self._joint_rows[jname] = jr
                row += 1

    def _connect(self):
        def _do():
            ok = self._mcu.connect()
            self.after(0, self._on_connected, ok)

        threading.Thread(target=_do, daemon=True).start()

    def _on_connected(self, ok: bool):
        if ok:
            self._connected = True
            self._status_var.set(f"Connected: {self._mcu.target}")
            self._poll_telemetry()
        else:
            self._status_var.set("Connection failed")
            messagebox.showerror(
                "Connection Error", f"Could not connect to {self._mcu.target}"
            )

    def _poll_telemetry(self):
        if not self._connected:
            return
        states = self._sensor_states()
        for name, jr in self._joint_rows.items():
            jt = self._mcu.joints.get(name)
            jr.update_from_telemetry(jt, states.get(name))

        now = time.time()
        self._imu_row.update(self._mcu.imu)
        self._batt_row.update(self._mcu.battery, now)

        if self._stm:
            self._role_var.set("Role: STM CAN (6 legs)")
        else:
            now = time.time()
            fresh = {
                role
                for role, ts in self._mcu.role_last_seen.items()
                if now - ts < ROLE_STALE_S
            }
            leader = next((r for r in ("FRONT", "UNKWN") if r in fresh), "---")
            self._role_var.set(
                f"Role: {leader}  left={'LEFT' in fresh}  right={'RIGHT' in fresh}"
            )

        if not self._stm:
            self._update_cal_status()
            self._update_banner(states)

        note = self._neutral_text if time.time() < self._neutral_until else ""
        suffix = f" | {note}" if note else ""
        if self._mcu.last_error:
            self._status_var.set(f"Error: {self._mcu.last_error}")
        elif self._mcu.last_feedback_ts:
            age = time.time() - self._mcu.last_feedback_ts
            if age < 1.0:
                self._status_var.set(f"Connected: {self._mcu.target}{suffix}")
            else:
                self._status_var.set(
                    f"Connected: {self._mcu.target} (stale {age:.0f}s){suffix}"
                )

        self.after(TELEMETRY_REFRESH_MS, self._poll_telemetry)

    def _sensor_states(self) -> Dict[str, str]:
        """SENS codes by joint name; empty for firmware or adapters without them."""
        getter = getattr(self._mcu, "sensor_states", None)
        states = getter() if callable(getter) else None
        return states if isinstance(states, dict) else {}

    def _update_banner(self, states: Dict[str, str]):
        text, colour = encoder_banner(states)
        self._banner_var.set(text)
        self._banner_label.configure(foreground=colour)

    def _wrap_cal_label(self, event):
        self._cal_label.configure(wraplength=cal_label_wraplength(event.width))

    def _update_cal_status(self):
        # The run's robot end line stays shown even if a follower line follows it.
        recent = getattr(self._mcu, "recent_cal_lines", None)
        if callable(recent):
            line = cal_status_line(recent())
        else:
            line = getattr(self._mcu, "latest_cal_line", None)
        text, colour = cal_status_text(line)
        self._cal_var.set(text)
        self._cal_label.configure(foreground=colour)

    def _jog_joint(self, name: str, pwm: int):
        if not self._connected:
            return
        self._mcu.send_command_jog(name, pwm)

    def _hold_all(self):
        if self._connected:
            self._mcu.send_command_joints_hold()
            self._cal_awaiting_end = False  # H aborts any calibration run

    def _calibration_in_progress(self) -> bool:
        """True while a C sent from this window has no robot end line and no Hold All."""
        if not self._cal_awaiting_end:
            return False
        recent = getattr(self._mcu, "recent_cal_lines", None)
        lines = recent() if callable(recent) else []
        if isinstance(lines, (list, tuple)) and any(cal_robot_finished(l) for l in lines):
            self._cal_awaiting_end = False
            return False
        return True

    def _show_note(self, text: str):
        """Put ``text`` on the status line for NEUTRAL_NOTE_S seconds."""
        self._neutral_text = text
        self._neutral_until = time.time() + NEUTRAL_NOTE_S
        if text:
            self._status_var.set(text)

    def _neutral(self):
        """Send 0.5 to every hip and knee that has a position; yaws are never sent."""
        if not self._connected:
            return
        cmds, skipped = self._mcu.neutral_targets(0.5)
        if cmds:
            self._mcu.send_command_joints(cmds)
        self._show_note(neutral_note(len(cmds), skipped, self._sensor_states()))

    def _calibrate(self):
        if not self._connected:
            return
        if self._stm:
            joint_name = simpledialog.askstring(
                "Calibrate",
                "Joint name (e.g. FLHL for hip pitch, FLKL for knee):",
                parent=self,
            )
            if not joint_name:
                return
            joint_name = joint_name.strip().upper()
            if joint_name not in self._joint_rows:
                messagebox.showerror(
                    "Calibrate",
                    f"Unknown joint {joint_name!r}. Use names like FLHY, FLHL, FLKL.",
                    parent=self,
                )
                return
            try:
                self._mcu.send_calibration_for_joint(joint_name)
            except ValueError as exc:
                messagebox.showerror("Calibrate", str(exc), parent=self)
                return
            messagebox.showinfo(
                "Calibrate",
                f"Sent calibrate for {joint_name} (stall on joints 2-3 only).",
                parent=self,
            )
            return
        if self._calibration_in_progress():
            # The dialog is modal and would block Hold All; a second C only gets BUSY.
            self._show_note(CALIBRATION_RUNNING_NOTE)
            return
        choice = ask_calibration_settings(self)
        if choice is None:
            return
        try:
            sent = self._mcu.send_command_calibrate(choice.pwm, choice.passes)
        except ValueError as exc:
            messagebox.showerror("Calibrate", str(exc), parent=self)
            return
        self._cal_awaiting_end = sent is not None

    def _on_close(self):
        self._connected = False
        try:
            self._mcu.send_command_joints_hold()
        except Exception:
            pass
        self._mcu.close()
        self.destroy()
