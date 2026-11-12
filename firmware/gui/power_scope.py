"""Live pack-voltage, voltage-region, and power-lifecycle timeline."""

from __future__ import annotations

import csv
import time
import tkinter as tk
from collections import deque
from tkinter import filedialog, messagebox, ttk
from typing import Deque, NamedTuple, Optional

from firmware.interfaces.battery_telemetry import (
    PackVoltageRegion,
    PowerSafetyState,
)

SAMPLE_LIMIT = 12000
PLOT_REFRESH_MS = 250

THRESHOLDS = (
    (22.4, "hard cut", PackVoltageRegion.HARD_CUT),
    (24.0, "soft cut", PackVoltageRegion.SOFT_CUT),
    (24.8, "warn", PackVoltageRegion.WARN),
    (26.4, "recovery", PackVoltageRegion.ABOVE_RECOVERY),
    (29.6, "over-voltage", PackVoltageRegion.OVER_VOLT),
)

REGION_COLORS = {
    PackVoltageRegion.NORMAL: "#2e7d32",
    PackVoltageRegion.WARN: "#d4a017",
    PackVoltageRegion.SOFT_CUT: "#ef6c00",
    PackVoltageRegion.HARD_CUT: "#c62828",
    PackVoltageRegion.OVER_VOLT: "#8e24aa",
    PackVoltageRegion.ABOVE_RECOVERY: "#1976d2",
}

LIFECYCLE_COLORS = {
    PowerSafetyState.OPERATING: "#2e7d32",
    PowerSafetyState.POWERING_DOWN: "#ef6c00",
    PowerSafetyState.SLEEP: "#616161",
    PowerSafetyState.RESUMING: "#1976d2",
    PowerSafetyState.OVER_VOLTAGE_LOCKOUT: "#8e24aa",
}

# Bottom-to-top display order. These are presentation ranks, independent of the
# stable enum values carried on the wire.
REGION_ORDER = (
    PackVoltageRegion.HARD_CUT,
    PackVoltageRegion.SOFT_CUT,
    PackVoltageRegion.WARN,
    PackVoltageRegion.NORMAL,
    PackVoltageRegion.ABOVE_RECOVERY,
    PackVoltageRegion.OVER_VOLT,
)

LIFECYCLE_ORDER = (
    PowerSafetyState.OVER_VOLTAGE_LOCKOUT,
    PowerSafetyState.SLEEP,
    PowerSafetyState.POWERING_DOWN,
    PowerSafetyState.RESUMING,
    PowerSafetyState.OPERATING,
)

UNKNOWN_COLOR = "#757575"


def _label(value) -> str:
    return value.name if hasattr(value, "name") else str(value)


def _color(value, colors) -> str:
    return colors.get(value, UNKNOWN_COLOR)


def ordered_positions(values, order):
    """Map categorical values to their explicit display ranks."""
    positions = {value: index for index, value in enumerate(order)}
    return [positions.get(value, len(order)) for value in values]


def state_spans(times, values, end_time):
    """Return contiguous time spans for a sampled categorical value."""
    if not times:
        return []
    spans = []
    start = times[0]
    value = values[0]
    for timestamp, next_value in zip(times[1:], values[1:]):
        if next_value != value:
            spans.append((start, timestamp, value))
            start = timestamp
            value = next_value
    spans.append((start, end_time, value))
    return spans


class PowerSample(NamedTuple):
    timestamp: float
    pack_volts: float
    region: object
    state: object
    pack_valid: bool


class PowerEvent(NamedTuple):
    timestamp: float
    text: str


class PowerTimelineRecorder:
    """Bounded telemetry history, deduplicated from the GUI polling cadence."""

    def __init__(self, limit: int = SAMPLE_LIMIT):
        self._samples: Deque[PowerSample] = deque(maxlen=limit)
        self._events: Deque[PowerEvent] = deque(maxlen=limit)
        self._last_battery = None
        self._last_message = None

    def observe(self, battery, power_message, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        if battery is not None and battery is not self._last_battery:
            previous = self._samples[-1] if self._samples else None
            sample = PowerSample(
                now,
                battery.pack_volts,
                battery.pack_region,
                battery.power_safety_state,
                battery.pack_valid,
            )
            self._samples.append(sample)
            self._last_battery = battery
            if previous is None:
                self._events.append(
                    PowerEvent(
                        now,
                        f"start {_label(sample.region)}/{_label(sample.state)}",
                    )
                )
            else:
                if sample.region != previous.region:
                    self._events.append(
                        PowerEvent(
                            now,
                            "region "
                            f"{_label(previous.region)} → {_label(sample.region)}",
                        )
                    )
                if sample.state != previous.state:
                    self._events.append(
                        PowerEvent(
                            now,
                            "lifecycle "
                            f"{_label(previous.state)} → {_label(sample.state)}",
                        )
                    )
        if power_message is not None and power_message is not self._last_message:
            self._events.append(PowerEvent(now, power_message.format_line()))
            self._last_message = power_message

    def samples(self):
        return list(self._samples)

    def events(self):
        return list(self._events)

    def clear(self) -> None:
        self._samples.clear()
        self._events.clear()
        self._last_battery = None
        self._last_message = None


class PowerScopeWindow(tk.Toplevel):
    def __init__(self, parent, recorder: PowerTimelineRecorder):
        super().__init__(parent)
        self.title("Power timeline")
        self.geometry("980x720")
        self._recorder = recorder
        self._closed = False

        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
        except ImportError:
            messagebox.showerror(
                "matplotlib missing",
                "The power timeline needs matplotlib:\n\n    pip install matplotlib",
            )
            self.destroy()
            return

        controls = ttk.Frame(self, padding=6)
        controls.pack(fill="x")
        ttk.Button(controls, text="Clear", command=self._recorder.clear).pack(
            side="left", padx=4
        )
        ttk.Button(controls, text="Export CSV", command=self._export).pack(
            side="left", padx=4
        )
        self._summary = tk.StringVar(value="waiting for BATT telemetry...")
        ttk.Label(self, textvariable=self._summary, font=("Consolas", 9)).pack(
            fill="x", padx=8
        )

        figure = Figure(figsize=(9.8, 6.2), dpi=100)
        self._voltage_axes = figure.add_subplot(311)
        self._region_axes = figure.add_subplot(312, sharex=self._voltage_axes)
        self._state_axes = figure.add_subplot(313, sharex=self._voltage_axes)
        figure.subplots_adjust(hspace=0.35, left=0.17, right=0.97, top=0.95)
        self._canvas = FigureCanvasTkAgg(figure, master=self)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._refresh()

    def _on_close(self):
        self._closed = True
        self.destroy()

    def _export(self):
        samples = self._recorder.samples()
        if not samples:
            messagebox.showinfo("Export", "No power samples recorded yet.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile="power_timeline.csv"
        )
        if not path:
            return
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["timestamp", "pack_volts", "region", "lifecycle", "pack_valid"]
            )
            for sample in samples:
                writer.writerow(
                    [
                        f"{sample.timestamp:.3f}",
                        f"{sample.pack_volts:.3f}",
                        _label(sample.region),
                        _label(sample.state),
                        int(sample.pack_valid),
                    ]
                )

    def _refresh(self):
        if self._closed:
            return
        self._draw(self._recorder.samples(), self._recorder.events())
        self.after(PLOT_REFRESH_MS, self._refresh)

    @staticmethod
    def _shade_states(axes, times, values, end_time, colors):
        for start, end, value in state_spans(times, values, end_time):
            axes.axvspan(
                start,
                end,
                facecolor=_color(value, colors),
                alpha=0.12,
                linewidth=0,
            )

    @staticmethod
    def _draw_colored_line(axes, times, values, states, colors):
        if len(times) == 1:
            axes.scatter(
                times,
                values,
                color=_color(states[0], colors),
                s=18,
                zorder=3,
            )
            return
        for index in range(1, len(times)):
            axes.plot(
                times[index - 1:index + 1],
                values[index - 1:index + 1],
                color=_color(states[index], colors),
                lw=1.8,
            )

    @classmethod
    def _draw_enum_axis(
        cls, axes, times, values, order, ylabel, colors, end_time
    ):
        numeric_values = ordered_positions(values, order)
        cls._shade_states(axes, times, values, end_time, colors)
        cls._draw_colored_line(axes, times, numeric_values, values, colors)
        axes.set_yticks(range(len(order)))
        axes.set_yticklabels([member.name for member in order], fontsize=8)
        for tick, member in zip(axes.get_yticklabels(), order):
            tick.set_color(_color(member, colors))
        axes.set_ylabel(ylabel)
        axes.grid(alpha=0.25)

    def _draw(self, samples, events):
        for axes in (self._voltage_axes, self._region_axes, self._state_axes):
            axes.clear()
        if not samples:
            self._summary.set("waiting for BATT telemetry...")
            self._canvas.draw_idle()
            return

        origin = samples[0].timestamp
        times = [sample.timestamp - origin for sample in samples]
        regions = [sample.region for sample in samples]
        states = [sample.state for sample in samples]
        end_time = max(times[-1], time.time() - origin)
        self._shade_states(
            self._voltage_axes, times, regions, end_time, REGION_COLORS
        )
        self._draw_colored_line(
            self._voltage_axes,
            times,
            [sample.pack_volts for sample in samples],
            regions,
            REGION_COLORS,
        )
        for threshold, label, region in THRESHOLDS:
            self._voltage_axes.axhline(
                threshold,
                color=_color(region, REGION_COLORS),
                ls="--",
                lw=0.9,
                alpha=0.8,
                label=f"{label} {threshold:g}V",
            )
        self._voltage_axes.set_ylabel("pack voltage (V)")
        self._voltage_axes.grid(alpha=0.25)
        self._voltage_axes.legend(loc="upper right", fontsize=7, ncol=2)

        self._draw_enum_axis(
            self._region_axes,
            times,
            regions,
            REGION_ORDER,
            "voltage region",
            REGION_COLORS,
            end_time,
        )
        self._draw_enum_axis(
            self._state_axes,
            times,
            states,
            LIFECYCLE_ORDER,
            "lifecycle",
            LIFECYCLE_COLORS,
            end_time,
        )
        self._state_axes.set_xlabel("seconds")

        for event in events:
            if event.timestamp < origin:
                continue
            x = event.timestamp - origin
            for axes in (self._voltage_axes, self._region_axes, self._state_axes):
                axes.axvline(x, color="#212121", ls=":", lw=0.8, alpha=0.55)

        latest = samples[-1]
        age = max(0.0, time.time() - latest.timestamp)
        recent = " | ".join(event.text for event in events[-3:])
        self._summary.set(
            f"{latest.pack_volts:.2f} V  {_label(latest.region)}/"
            f"{_label(latest.state)}  sample age {age:.1f}s"
            + (f"\n{recent}" if recent else "")
        )
        self._canvas.draw_idle()
