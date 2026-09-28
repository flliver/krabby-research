from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from typing import Dict, Optional

from firmware.krabby_mcu import DEFAULT_BAUD, KrabbyMCUSDK, JOINT_GROUP_NAMES
from firmware.interfaces.joint_telemetry import JointTelemetry

JOG_PWM_DEFAULT = 30
TELEMETRY_REFRESH_MS = 100  # GUI poll period; decoupled from the firmware's telemetry tick
JOG_HEARTBEAT_MS = 100  # re-send a held jog faster than the firmware's ~300ms jog watchdog
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


def _jog_sign(name: str) -> int:
    """Wire-PWM sign for this joint's "extend" leg motion. The knee (KL)
    linkages run opposite to the hips: positive PWM extends an HL but tucks a
    KL, so KLs flip. Bench-observed 2026-07-13; the wire protocol itself (jog,
    C retract/extend) stays actuator-relative — this mapping is GUI-only."""
    return -1 if name.endswith("KL") else 1


class JointRow:
    """One row in the telemetry grid: name, jog buttons, live values."""

    def __init__(self, parent: tk.Widget, name: str, row: int, jog_cb, get_jog_pwm):
        self.name = name
        self._jog_cb = jog_cb
        self._get_jog_pwm = get_jog_pwm
        self._active_dir = 0
        self._jog_after_id = None

        self.lbl_name = ttk.Label(parent, text=name, font=FONT_JOINT_NAME, width=6)
        self.lbl_name.grid(row=row, column=0, padx=4, pady=2, sticky="w")

        self.btn_retract = ttk.Button(parent, text="\u25c0 Retract", width=10)
        self.btn_retract.grid(row=row, column=1, padx=2, pady=2)
        self.btn_retract.bind("<ButtonPress-1>", lambda e: self._start_jog(-1))
        self.btn_retract.bind("<ButtonRelease-1>", lambda e: self._stop_jog())

        self.btn_extend = ttk.Button(parent, text="Extend \u25b6", width=10)
        self.btn_extend.grid(row=row, column=2, padx=2, pady=2)
        self.btn_extend.bind("<ButtonPress-1>", lambda e: self._start_jog(1))
        self.btn_extend.bind("<ButtonRelease-1>", lambda e: self._stop_jog())

        self.var_pos = tk.StringVar(value=NO_VALUE_TEXT)
        self.var_cal = tk.StringVar(value=NO_VALUE_TEXT)
        self.var_pot = tk.StringVar(value=NO_VALUE_TEXT)
        self.var_cur = tk.StringVar(value=NO_VALUE_TEXT)
        self.var_pwm = tk.StringVar(value=NO_VALUE_TEXT)
        self.var_hall = tk.StringVar(value=NO_VALUE_TEXT)

        # Normalized [0,1] position is the canonical operator value; it's colored by
        # calibration state so an unusable (PARTIAL) or uncalibrated (UNCAL) reading —
        # where pos still maps through full-range defaults — is visibly distinct from a
        # FULL, end-stop-anchored one. Raw pot ADC and the Hall edge count stay as debug
        # fields for spotting wiring issues.
        self.lbl_pos = tk.Label(parent, textvariable=self.var_pos, width=7, anchor="e",
                                font=FONT_JOINT_NAME)
        self.lbl_pos.grid(row=row, column=3, padx=4)
        self.lbl_cal = tk.Label(parent, textvariable=self.var_cal, width=8, anchor="center")
        self.lbl_cal.grid(row=row, column=4, padx=4)
        ttk.Label(parent, textvariable=self.var_pot, width=6, anchor="e").grid(row=row, column=5, padx=4)
        ttk.Label(parent, textvariable=self.var_cur, width=6, anchor="e").grid(row=row, column=6, padx=4)
        ttk.Label(parent, textvariable=self.var_pwm, width=10, anchor="e").grid(row=row, column=7, padx=4)
        ttk.Label(parent, textvariable=self.var_hall, width=6, anchor="e").grid(row=row, column=8, padx=4)

    def _start_jog(self, direction: int):
        self._active_dir = direction
        self._send_jog_heartbeat()

    def _send_jog_heartbeat(self):
        # While the button is held, keep re-sending the jog so it outlives the firmware's
        # jog watchdog; reschedule until the button is released (_active_dir back to 0).
        if self._active_dir == 0:
            return
        self._jog_cb(self.name, self._active_dir * _jog_sign(self.name) * self._get_jog_pwm())
        self._jog_after_id = self.lbl_name.after(JOG_HEARTBEAT_MS, self._send_jog_heartbeat)

    def _stop_jog(self):
        self._active_dir = 0
        if self._jog_after_id is not None:
            self.lbl_name.after_cancel(self._jog_after_id)
            self._jog_after_id = None
        # Send the stop redundantly: a single J 0 line can be lost or delayed when the
        # board is busy digesting a jog backlog (motor EMI slows its loop), and a lost
        # stop means the motor runs until the ~300ms jog watchdog notices. Re-sends are
        # cheap and skipped if a new jog started in the meantime.
        self._jog_cb(self.name, 0)
        for delay_ms in (120, 260):
            self.lbl_name.after(delay_ms, self._resend_stop)

    def _resend_stop(self):
        if self._active_dir == 0:
            self._jog_cb(self.name, 0)

    # Pos/CAL text color by calibration state: green = FULL (both end-stops recorded,
    # trustworthy), orange = PARTIAL (one stop recorded, pos not yet anchored), gray = UNCAL.
    _CAL_COLORS = {"FULL": "#1a7f1a", "PARTIAL": "#c8780a", "UNCAL": "#999999"}

    def update_from_telemetry(self, jt: Optional[JointTelemetry]):
        if jt is None:
            return
        self.var_pos.set(f"{jt.pos:.3f}" if jt.connected else "DISC")
        self.var_cal.set(jt.cal_state_name)
        color = self._CAL_COLORS.get(jt.cal_state_name, "#000000")
        self.lbl_pos.config(fg=color)
        self.lbl_cal.config(fg=color)
        self.var_pot.set(str(jt.pot))
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


class KrabbyTestGUI(tk.Tk):
    def __init__(self, port: Optional[str] = None, baud: int = DEFAULT_BAUD):
        super().__init__()
        self.title("Krabby MCU Test")
        self.geometry("960x820")
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._mcu = KrabbyMCUSDK(port=port, baud=baud)
        self._joint_rows: Dict[str, JointRow] = {}
        self._connected = False
        self._jog_pwm_var = tk.IntVar(value=JOG_PWM_DEFAULT)

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

        pwm_frame = ttk.LabelFrame(top, text="Jog PWM", padding=(6, 2))
        pwm_frame.pack(side="right", padx=(8, 0))
        self._jog_pwm_label = ttk.Label(pwm_frame, text=str(JOG_PWM_DEFAULT), width=4, anchor="e")
        self._jog_pwm_label.pack(side="right", padx=(4, 0))
        ttk.Scale(
            pwm_frame,
            from_=0,
            to=120,
            orient="horizontal",
            length=120,
            variable=self._jog_pwm_var,
            command=self._on_jog_pwm_changed,
        ).pack(side="left")

        btn_frame = ttk.Frame(top)
        btn_frame.pack(side="right", padx=(8, 0))
        ttk.Button(btn_frame, text="Hold All", command=self._hold_all).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Neutral (0.5)", command=self._neutral).pack(side="left", padx=4)

        imu_frame = ttk.Frame(self, padding=(8, 0))
        imu_frame.pack(fill="x")
        self._imu_row = ImuRow(imu_frame)

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

        headers = ["Joint", "Retract", "Extend", "Pos", "CAL", "Pot", "Cur", "PWM", "Hall"]
        for c, h in enumerate(headers):
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
            ).grid(row=row, column=0, columnspan=9, sticky="w", pady=(6, 2))
            row += 1
            for jname in joint_names:
                jr = JointRow(self._grid_frame, jname, row, self._jog_joint, self._get_jog_pwm)
                self._joint_rows[jname] = jr
                row += 1

    def _get_jog_pwm(self) -> int:
        return max(0, min(255, self._jog_pwm_var.get()))

    def _on_jog_pwm_changed(self, _value: str) -> None:
        self._jog_pwm_label.config(text=str(self._get_jog_pwm()))

    def _connect(self):
        def _do():
            ok = self._mcu.connect()
            self.after(0, self._on_connected, ok)

        threading.Thread(target=_do, daemon=True).start()

    def _on_connected(self, ok: bool):
        if ok:
            self._connected = True
            self._status_var.set(f"Connected: {self._mcu.port}")
            self._poll_telemetry()
        else:
            self._status_var.set("Connection failed")
            messagebox.showerror(
                "Connection Error", f"Could not connect to {self._mcu.port}"
            )

    def _poll_telemetry(self):
        if not self._connected:
            return
        for name, jr in self._joint_rows.items():
            jt = self._mcu.joints.get(name)
            jr.update_from_telemetry(jt)

        self._imu_row.update(self._mcu.imu)

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

        if self._mcu.last_error:
            self._status_var.set(f"Error: {self._mcu.last_error}")
        elif self._mcu.last_feedback_ts:
            age = time.time() - self._mcu.last_feedback_ts
            if age < 1.0:
                self._status_var.set(f"Connected: {self._mcu.port}")
            else:
                self._status_var.set(f"Connected: {self._mcu.port} (stale {age:.0f}s)")

        self.after(TELEMETRY_REFRESH_MS, self._poll_telemetry)

    def _jog_joint(self, name: str, pwm: int):
        if not self._connected:
            return
        self._mcu.send_command_jog(name, pwm)

    def _hold_all(self):
        if self._connected:
            self._mcu.send_command_joints_hold()

    def _neutral(self):
        if not self._connected:
            return
        cmds = {}
        for _, names in JOINT_GROUP_NAMES:
            for n in names:
                cmds[n] = 0.5
        self._mcu.send_command_joints(cmds)

    def _on_close(self):
        self._connected = False
        try:
            self._mcu.send_command_joints_hold()
        except Exception:
            pass
        self._mcu.close()
        self.destroy()
