"""
Krabby MCU test GUI — tkinter app for jogging joints and viewing live telemetry.
Run: python -m firmware.gui [--port COM5]
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from typing import Dict, Optional

from firmware.joints import spec
from firmware.krabby_mcu import KrabbyMCUSDK, JOINT_GROUP_NAMES
from firmware.interfaces.joint_telemetry import JointTelemetry

# Per-joint jog PWM ceilings live in the joint registry (firmware/joints.py) —
# hip-yaw jogs slower so its fast-shaft encoder can't saturate the MCU (hall_storm).
TELEMETRY_REFRESH_MS = 100
JOG_HEARTBEAT_MS = 100  # re-send a held jog faster than the firmware's ~300ms jog watchdog


class JointRow:
    """One row in the telemetry grid: name, jog buttons, live values."""

    def __init__(self, parent: tk.Widget, name: str, row: int, jog_cb):
        self.name = name
        self._jog_cb = jog_cb
        self._active_dir = 0
        self._jog_after_id = None

        self.lbl_name = ttk.Label(parent, text=name, font=("Consolas", 11, "bold"), width=6)
        self.lbl_name.grid(row=row, column=0, padx=4, pady=2, sticky="w")

        # Sign convention matches the firmware: -PWM retracts (pos \u2192 0.0), +PWM extends
        # (pos \u2192 1.0). Retract drives negative, Extend positive.
        self.btn_retract = ttk.Button(parent, text="\u25C0 Retract", width=10)
        self.btn_retract.grid(row=row, column=1, padx=2, pady=2)
        self.btn_retract.bind("<ButtonPress-1>", lambda e: self._start_jog(-1))
        self.btn_retract.bind("<ButtonRelease-1>", lambda e: self._stop_jog())

        self.btn_extend = ttk.Button(parent, text="Extend \u25B6", width=10)
        self.btn_extend.grid(row=row, column=2, padx=2, pady=2)
        self.btn_extend.bind("<ButtonPress-1>", lambda e: self._start_jog(1))
        self.btn_extend.bind("<ButtonRelease-1>", lambda e: self._stop_jog())

        self.var_pos = tk.StringVar(value="---")
        self.var_cal = tk.StringVar(value="---")
        self.var_pot = tk.StringVar(value="---")
        self.var_enc = tk.StringVar(value="---")
        self.var_cur = tk.StringVar(value="---")
        self.var_pwm = tk.StringVar(value="---")

        # Normalized [0,1] position is the canonical operator value (2e §6); it's colored by
        # calibration state so an unanchored (PARTIAL) or uncalibrated (UNCAL) reading — where
        # pos isn't trustworthy — is visibly distinct from a FULL, absolute one. Raw pot ADC
        # and the signed Hall quadrature count (Enc) are debug fields for validating wiring
        # before assembly: Enc must rise driving one way and fall the other — if it climbs
        # both ways or jitters around 0, HallB (direction) is miswired or dead.
        self.lbl_pos = tk.Label(parent, textvariable=self.var_pos, width=7, anchor="e",
                                font=("Consolas", 11, "bold"))
        self.lbl_pos.grid(row=row, column=3, padx=4)
        self.lbl_cal = tk.Label(parent, textvariable=self.var_cal, width=8, anchor="center")
        self.lbl_cal.grid(row=row, column=4, padx=4)
        ttk.Label(parent, textvariable=self.var_pot, width=6, anchor="e").grid(row=row, column=5, padx=4)
        ttk.Label(parent, textvariable=self.var_enc, width=10, anchor="e").grid(row=row, column=6, padx=4)
        ttk.Label(parent, textvariable=self.var_cur, width=6, anchor="e").grid(row=row, column=7, padx=4)
        ttk.Label(parent, textvariable=self.var_pwm, width=10, anchor="e").grid(row=row, column=8, padx=4)

    def _start_jog(self, direction: int):
        self._active_dir = direction
        self._send_jog_heartbeat()

    def _send_jog_heartbeat(self):
        # While the button is held, keep re-sending the jog so it outlives the firmware's
        # jog watchdog; reschedule until the button is released (_active_dir back to 0).
        if self._active_dir == 0:
            return
        self._jog_cb(self.name, self._active_dir * spec(self.name).jog_pwm_max)
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

    # Pos/CAL text color by calibration state: green = FULL (absolute, trustworthy),
    # orange = PARTIAL (Hall, relative until it self-heals at an end-stop), gray = UNCAL.
    _CAL_COLORS = {"FULL": "#1a7f1a", "PARTIAL": "#c8780a", "UNCAL": "#999999"}

    def update_from_telemetry(self, jt: Optional[JointTelemetry]):
        if jt is None:
            return
        self.var_pos.set(f"{jt.pos:.3f}")
        self.var_cal.set(jt.cal_state_name)
        color = self._CAL_COLORS.get(jt.cal_state_name, "#000000")
        self.lbl_pos.config(fg=color)
        self.lbl_cal.config(fg=color)
        self.var_pot.set(str(jt.pot))
        self.var_enc.set(str(jt.saf))
        self.var_cur.set(str(jt.current))
        self.var_pwm.set(f"L{jt.pwm[0]} R{jt.pwm[1]}")


class KrabbyTestGUI(tk.Tk):
    def __init__(self, port: Optional[str] = None, baud: int = 115200):
        super().__init__()
        self.title("Krabby MCU Test")
        self.geometry("960x770")
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._mcu = KrabbyMCUSDK(port=port, baud=baud)
        self._joint_rows: Dict[str, JointRow] = {}
        self._connected = False

        self._build_ui()
        self._connect()

    def _build_ui(self):
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")

        self._status_var = tk.StringVar(value="Connecting...")
        ttk.Label(top, textvariable=self._status_var, font=("Segoe UI", 10)).pack(side="left")

        btn_frame = ttk.Frame(top)
        btn_frame.pack(side="right")
        ttk.Button(btn_frame, text="Hold All", command=self._hold_all).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Neutral (0.5)", command=self._neutral).pack(side="left", padx=4)

        sep = ttk.Separator(self, orient="horizontal")
        sep.pack(fill="x", pady=4)

        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self._grid_frame = ttk.Frame(canvas, padding=8)

        self._grid_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._grid_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        headers = ["Joint", "Retract", "Extend", "Pos", "CAL", "Pot", "Enc", "Cur", "PWM"]
        for c, h in enumerate(headers):
            ttk.Label(self._grid_frame, text=h, font=("Segoe UI", 9, "bold"), anchor="center").grid(
                row=0, column=c, padx=4, pady=(0, 4), sticky="ew"
            )

        row = 1
        for group_name, joint_names in JOINT_GROUP_NAMES:
            ttk.Label(
                self._grid_frame, text=f"── {group_name} ──",
                font=("Segoe UI", 9, "italic"), foreground="#666"
            ).grid(row=row, column=0, columnspan=9, sticky="w", pady=(6, 2))
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
            self._status_var.set(f"Connected: {self._mcu.port}")
            self._poll_telemetry()
        else:
            self._status_var.set("Connection failed")
            messagebox.showerror("Connection Error", f"Could not connect to {self._mcu.port}")

    def _poll_telemetry(self):
        if not self._connected:
            return
        for name, jr in self._joint_rows.items():
            jt = self._mcu.joints.get(name)
            jr.update_from_telemetry(jt)

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
