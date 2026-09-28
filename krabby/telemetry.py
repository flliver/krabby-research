"""Best-effort fleet telemetry for Classic Shadow `state.reported`.

The payload shape is intentionally open-ended: whatever this module can
collect on the device is what `krabby get telemetry` prints and what
`krabby agent` publishes once a minute. Operators and the fleet portal treat
the shadow as opaque JSON keyed by the top-level fields below when present.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from krabby._state import installed_image

LOCOMOTION_CONTAINER = "krabby"
MCU_DEVICE_DEFAULT = "/dev/ttyACM0"

# Runs inside the locomotion container when it is up; polls HAL once.
_HAL_OBSERVATION_PROBE = r"""
import json
import sys
import time

from hal.client.client import HalClient
from hal.client.config import HalClientConfig

client = HalClient(HalClientConfig(observation_endpoint="tcp://127.0.0.1:6001"))
client.initialize()
obs = None
deadline = time.time() + 2.0
while time.time() < deadline:
    obs = client.poll(timeout_ms=50)
    if obs is not None:
        break
client.close()
if obs is None:
    sys.exit(2)
print(
    json.dumps(
        {
            "imu": {
                "base_quat_w": obs.base_quat_w.tolist(),
                "base_ang_vel_b": obs.base_ang_vel_b.tolist(),
                "base_lin_vel_b": obs.base_lin_vel_b.tolist(),
            },
            "pose": {
                "joint_positions": obs.joint_positions.tolist(),
                "joint_velocities": obs.joint_velocities.tolist(),
            },
        }
    )
)
"""


def _run(cmd: list[str], *, timeout: float = 5.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _systemd_unit_state(unit: str) -> str:
    if not shutil.which("systemctl"):
        return "unknown"
    result = _run(["systemctl", "is-active", unit], timeout=3.0)
    state = (result.stdout or "").strip()
    return state if state else "unknown"


def _locomotion_container_state() -> str:
    if not shutil.which("docker"):
        return "unknown"
    result = _run(
        ["docker", "ps", "-a", "--filter", f"name=^{LOCOMOTION_CONTAINER}$", "--format", "{{.State}}"],
        timeout=5.0,
    )
    state = (result.stdout or "").strip().lower()
    if not state:
        return "absent"
    if state == "running":
        return "running"
    return "stopped"


def collect_health() -> dict[str, Any]:
    """Host + service health signals."""
    health: dict[str, Any] = {
        "locomotion_container": _locomotion_container_state(),
        "krabby_agent": _systemd_unit_state("krabby-agent.service"),
        "krabby_locomotion": _systemd_unit_state("krabby-locomotion.service"),
    }
    mcu_path = Path(MCU_DEVICE_DEFAULT)
    health["mcu_present"] = mcu_path.exists()
    return health


def _read_power_supply_value(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def collect_power() -> dict[str, Any] | None:
    """Battery / supply readings when the host exposes sysfs power_supply nodes."""
    supplies: list[dict[str, Any]] = []
    for supply_dir in sorted(Path("/sys/class/power_supply").glob("*")):
        entry: dict[str, Any] = {"name": supply_dir.name}
        capacity = _read_power_supply_value(supply_dir / "capacity")
        if capacity is not None:
            entry["capacity_percent"] = capacity
        voltage_uv = _read_power_supply_value(supply_dir / "voltage_now")
        if voltage_uv is not None:
            entry["voltage_v"] = round(voltage_uv / 1_000_000, 3)
        current_ua = _read_power_supply_value(supply_dir / "current_now")
        if current_ua is not None:
            entry["current_a"] = round(current_ua / 1_000_000, 3)
        if len(entry) > 1:
            supplies.append(entry)
    if not supplies:
        return None
    return {"supplies": supplies}


def _probe_hal_imu_pose() -> dict[str, Any] | None:
    if _locomotion_container_state() != "running":
        return None
    if not shutil.which("docker"):
        return None
    result = _run(
        ["docker", "exec", LOCOMOTION_CONTAINER, "python3", "-c", _HAL_OBSERVATION_PROBE],
        timeout=8.0,
    )
    if result.returncode != 0:
        return None
    try:
        data = json.loads((result.stdout or "").strip())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def collect_red_flags(health: dict[str, Any], hal: dict[str, Any] | None) -> list[str]:
    """Actionable issues derived from the rest of the snapshot."""
    flags: list[str] = []
    if health.get("krabby_agent") not in ("active", "unknown"):
        flags.append("agent_not_running")
    if health.get("krabby_locomotion") == "active" and health.get("locomotion_container") != "running":
        flags.append("locomotion_container_down")
    if health.get("locomotion_container") == "running" and hal is None:
        flags.append("hal_no_observation")
    if health.get("mcu_present") is False:
        flags.append("mcu_missing")
    return flags


def collect_telemetry(*, teleop_edge_connected: bool | None = None) -> dict[str, Any]:
    """Return the full `state.reported` document for fleet shadow updates."""
    health = collect_health()
    hal = _probe_hal_imu_pose()

    reported: dict[str, Any] = {
        "timestamp": int(time.time()),
        "reported_image": installed_image(),
        "health": health,
    }

    if teleop_edge_connected is not None:
        reported["teleop_edge_connected"] = teleop_edge_connected

    if hal:
        if "imu" in hal:
            reported["imu"] = hal["imu"]
        if "pose" in hal:
            reported["pose"] = hal["pose"]

    power = collect_power()
    if power:
        reported["power"] = power

    red_flags = collect_red_flags(health, hal)
    if red_flags:
        reported["red_flags"] = red_flags

    return reported


def cmd_get_telemetry() -> None:
    """CLI entry: print the current telemetry snapshot as JSON."""
    print(json.dumps(collect_telemetry(), indent=2, sort_keys=True))
    sys.stdout.flush()
