"""Replay firmware renderer traces on the simulated OLED."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ssd1306 import OLED

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_ROOT / "build" / "native-firmware-tests"
BINARY = BUILD_DIR / "oled_trace"
CMAKE_SOURCE = REPO_ROOT / "tests" / "native" / "firmware"

# Sources compiled into oled_trace.
SOURCE_DIRS = (
    REPO_ROOT / "firmware" / "arduino" / "src" / "display",
    REPO_ROOT / "firmware" / "oled_sim" / "native",
)

GLYPHS = ("hold", "extend", "retract", "disc", "unverified")
ROLES = ("FRONT", "LEFT", "RIGHT", "UNKWN")


@dataclass
class KrabState:
    """Hardware state consumed by the native production model builder."""

    role: str = "FRONT"
    roll: int = 0
    pitch: int = 0
    imu_valid: bool = True
    battery_volts: tuple = (13.3, 13.3)
    front: bool = True
    left: bool = True
    right: bool = True
    # FL, FR, ML, MR, RL, RR; each tuple is yaw, hip, knee.
    legs: list = field(default_factory=lambda: [("hold", "hold", "hold")] * 6)

    @classmethod
    def from_payload(cls, payload: dict) -> "KrabState":
        if not isinstance(payload, dict):
            raise ValueError("state must be an object")
        return cls(
            role=str(payload.get("role", "FRONT")),
            roll=int(payload.get("roll", 0)),
            pitch=int(payload.get("pitch", 0)),
            imu_valid=bool(payload.get("imu_valid", True)),
            battery_volts=tuple(payload.get("battery_volts", (13.3, 13.3))),
            front=bool(payload.get("front", True)),
            left=bool(payload.get("left", True)),
            right=bool(payload.get("right", True)),
            legs=payload.get("legs", [("hold", "hold", "hold")] * 6),
        )

    def to_fields(self) -> str:
        if self.role not in ROLES:
            raise ValueError(f"role {self.role!r} not one of {ROLES}")
        if len(self.battery_volts) != 2:
            raise ValueError("expected two battery voltages")
        if len(self.legs) != 6:
            raise ValueError(f"expected 6 legs, got {len(self.legs)}")
        for leg in self.legs:
            if len(leg) != 3:
                raise ValueError(f"expected 3 joints per leg, got {leg!r}")
            for joint in leg:
                if joint not in GLYPHS:
                    raise ValueError(f"glyph {joint!r} not one of {GLYPHS}")
        legs = ";".join(",".join(leg) for leg in self.legs)
        return "\n".join((
            f"role={self.role}",
            f"roll={int(self.roll)}",
            f"pitch={int(self.pitch)}",
            f"imu={int(self.imu_valid)}",
            f"battery={float(self.battery_volts[0])},{float(self.battery_volts[1])}",
            f"front={int(self.front)}",
            f"left={int(self.left)}",
            f"right={int(self.right)}",
            f"legs={legs}",
        ))


class TraceBuildError(RuntimeError):
    """oled_trace build failure."""


def _newest_source_mtime() -> float:
    newest = 0.0
    for directory in SOURCE_DIRS:
        for path in directory.rglob("*"):
            if path.is_file():
                newest = max(newest, path.stat().st_mtime)
    return newest


def _ensure_binary() -> Path:
    """Build oled_trace when its sources are newer."""
    if BINARY.exists() and BINARY.stat().st_mtime >= _newest_source_mtime():
        return BINARY

    cmake = shutil.which("cmake")
    if cmake is None:
        raise TraceBuildError("cmake not on PATH; oled_trace cannot be built")

    for command in (
        [cmake, "-S", str(CMAKE_SOURCE), "-B", str(BUILD_DIR)],
        [cmake, "--build", str(BUILD_DIR), "--target", "oled_trace"],
    ):
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode != 0:
            raise TraceBuildError(
                f"$ {' '.join(command)}\n\n{done.stdout}\n{done.stderr}")
    if not BINARY.exists():
        raise TraceBuildError(f"build reported success but {BINARY} is missing")
    return BINARY


def trace(states) -> list:
    """Draw calls the firmware renderer makes for each state, in order."""
    binary = _ensure_binary()
    stdin = "\n\n".join(state.to_fields() for state in states) + "\n"
    done = subprocess.run([str(binary)], input=stdin, capture_output=True,
                          text=True)
    if done.returncode != 0:
        raise RuntimeError(f"oled_trace failed: {done.stderr}")

    frames = []
    for line in done.stdout.splitlines():
        if line == "frame":
            frames.append([])
        elif line:
            if not frames:
                raise RuntimeError(f"draw call before any frame: {line!r}")
            frames[-1].append(line)
    return frames


def _replay(calls, panel: OLED) -> None:
    for call in calls:
        op, _, rest = call.partition(" ")
        if op == "erase":
            panel.erase()
        elif op == "font":
            panel.setFont(rest)
        elif op == "pixel":
            x, y = (int(v) for v in rest.split())
            panel.pixel(x, y)
        elif op == "line":
            panel.line(*(int(v) for v in rest.split()))
        elif op == "rect":
            panel.rectangle(*(int(v) for v in rest.split()))
        elif op == "fill":
            x, y, width, height, color = (int(v) for v in rest.split())
            panel.rectangleFill(x, y, width, height, color)
        elif op == "text":
            x, y, value = rest.split(" ", 2)
            panel.text(int(x), int(y), value)
        else:
            raise RuntimeError(f"unknown draw call {call!r}")


def render_sequence(states) -> list:
    """Replay incremental states and return each resulting frame."""
    panel = OLED()
    frames = []
    for calls in trace(states):
        _replay(calls, panel)
        snapshot = OLED()
        snapshot.buf = bytearray(panel.buf)
        frames.append(snapshot)
    return frames


def render(state: KrabState) -> OLED:
    """One state as a full frame."""
    return render_sequence([state])[0]
