# SPDX-License-Identifier: BSD-3-Clause
"""Fixed command schedules for the crab-hex eval harness (Milestone 18, Task 0).

Compiles a scenario manifest into the per-step command arrays the harness writes straight into
``UniformParkourCommand.vel_command_b`` each step, replacing the env's own random resampling. Pure
stdlib + numpy so it is unit-testable without Isaac Sim.

Model: **one episode == the whole schedule**, with each command hold a segment *within* the
episode. Episodes are run as parallel envs (``num_envs == scenario.episodes``), not sequential
launches -- ten episodes is one Isaac process, not ten.

``steady_mask`` marks the steps that count toward metrics: everything except the spawn/settle
prologue and the first ``settle_s`` after each command change. Scoring the transient right after a
command step would measure the controller's step response, not its gait.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Commands below ``lin_vel_clip`` are treated as "stop" by UniformParkourCommand._resample_command
# (``small_commands_to_zero``), so a hold in that dead zone silently means something different from
# what the manifest says.
DEFAULT_LIN_VEL_CLIP = 0.2

DEFAULTS: dict[str, Any] = {
    "prologue_s": 1.5,
    "settle_s": 1.0,
    "min_window_s": 1.0,
    "fall_exclusion_s": 0.5,
    "contact_force_threshold": 1.0,
    "slip_speed_threshold": 0.02,
    "min_cycles": 2,
    "lin_vel_clip": DEFAULT_LIN_VEL_CLIP,
}


class ManifestError(ValueError):
    """Raised for a manifest that is structurally valid but semantically wrong."""


@dataclass
class Hold:
    """One command hold segment."""

    hold_s: float
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    delta_yaw: float | None = None  # yaw probes only; see Scenario.yaw_mode
    label: str = ""

    def as_cmd(self) -> tuple[float, float, float]:
        return (self.vx, self.vy, self.wz)


@dataclass
class Scenario:
    id: str
    task: str
    checkpoint: str
    role: str = "teacher"  # "teacher" | "student"; informational, the harness detects from agent cfg
    schedule: list[Hold] = field(default_factory=list)
    episodes: int = 10
    episode_length_s: float | None = None
    env_seed: int = 1
    terrain_generator_seed: int | None = None
    env_vars: dict[str, str] = field(default_factory=dict)
    rotate_holds_by_env: bool = False
    freeze_terrain_levels: bool = True
    freeze_friction: bool = False
    probe: str | None = None  # None | "vy" | "yaw"
    yaw_mode: str | None = None  # "delta_yaw_inject" for yaw probes
    checkpoint_sha256: str | None = None
    terrain_label: str = ""
    notes: str = ""
    defaults: dict[str, Any] = field(default_factory=lambda: dict(DEFAULTS))

    @property
    def schedule_duration_s(self) -> float:
        return float(sum(h.hold_s for h in self.schedule))

    @property
    def total_duration_s(self) -> float:
        return float(self.defaults["prologue_s"]) + self.schedule_duration_s

    def get_default(self, key: str) -> Any:
        return self.defaults.get(key, DEFAULTS.get(key))


@dataclass
class CompiledSchedule:
    """Per-step arrays, first dim time, second dim env."""

    cmd: np.ndarray  # (T, E, 3) float32 -- written to vel_command_b
    segment_id: np.ndarray  # (T, E) int16 -- -1 during the prologue
    steady_mask: np.ndarray  # (T, E) bool
    delta_yaw: np.ndarray | None  # (T, E) float32 or None
    n_steps: int
    dt: float
    hold_labels: list[str]


def _coerce_hold(raw: dict[str, Any], index: int) -> Hold:
    if "hold_s" not in raw:
        raise ManifestError(f"hold[{index}] is missing 'hold_s'")
    hold = Hold(
        hold_s=float(raw["hold_s"]),
        vx=float(raw.get("vx", 0.0)),
        vy=float(raw.get("vy", 0.0)),
        wz=float(raw.get("wz", 0.0)),
        delta_yaw=(float(raw["delta_yaw"]) if raw.get("delta_yaw") is not None else None),
        label=str(raw.get("label", "") or f"hold{index}"),
    )
    if hold.hold_s <= 0:
        raise ManifestError(f"hold[{index}] '{hold.label}': hold_s must be > 0")
    return hold


def load_manifest(path: str | Path) -> tuple[list[Scenario], dict[str, Any]]:
    """Load a scenario manifest (YAML if PyYAML is importable, else JSON).

    YAML is the committed format for readability; the JSON fallback keeps this module usable in a
    bare interpreter with no third-party deps, which is what makes the unit tests cheap to run.
    """
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ManifestError(
                f"{path} is YAML but PyYAML is not importable; convert to .json or install pyyaml"
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    if not isinstance(data, dict):
        raise ManifestError(f"{path}: top level must be a mapping")
    version = data.get("manifest_version")
    # v2 (gait-formation campaign, 2026-08-20) uses the same schema; it exists as a
    # separate file only because v1's holds are pinned to committed baselines.
    if version not in (1, 2):
        raise ManifestError(f"{path}: unsupported manifest_version {version!r} (expected 1 or 2)")

    file_defaults = dict(DEFAULTS)
    file_defaults.update(data.get("defaults", {}) or {})

    scenarios: list[Scenario] = []
    for raw in data.get("scenarios", []) or []:
        merged = dict(file_defaults)
        merged.update(raw.get("defaults", {}) or {})
        for key in ("id", "task", "checkpoint"):
            if not raw.get(key):
                raise ManifestError(f"scenario {raw.get('id', '<no id>')!r} is missing '{key}'")
        scenario = Scenario(
            id=str(raw["id"]),
            task=str(raw["task"]),
            checkpoint=str(raw["checkpoint"]),
            role=str(raw.get("role", "teacher")),
            schedule=[_coerce_hold(h, i) for i, h in enumerate(raw.get("schedule", []) or [])],
            episodes=int(raw.get("episodes", 10)),
            episode_length_s=(float(raw["episode_length_s"]) if raw.get("episode_length_s") else None),
            env_seed=int(raw.get("env_seed", 1)),
            terrain_generator_seed=(
                int(raw["terrain_generator_seed"]) if raw.get("terrain_generator_seed") is not None else None
            ),
            env_vars={str(k): str(v) for k, v in (raw.get("env") or {}).items()},
            rotate_holds_by_env=bool(raw.get("rotate_holds_by_env", False)),
            freeze_terrain_levels=bool(raw.get("freeze_terrain_levels", True)),
            freeze_friction=bool(raw.get("freeze_friction", False)),
            probe=(str(raw["probe"]) if raw.get("probe") else None),
            yaw_mode=(str(raw["yaw_mode"]) if raw.get("yaw_mode") else None),
            checkpoint_sha256=(str(raw["checkpoint_sha256"]) if raw.get("checkpoint_sha256") else None),
            terrain_label=str(raw.get("terrain_label", "")),
            notes=str(raw.get("notes", "")),
            defaults=merged,
        )
        validate_scenario(scenario)
        scenarios.append(scenario)

    ids = [s.id for s in scenarios]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ManifestError(f"{path}: duplicate scenario ids {sorted(dupes)}")
    return scenarios, file_defaults


def validate_scenario(scenario: Scenario) -> None:
    """Reject manifests that would silently measure something other than what they say."""
    if not scenario.schedule:
        raise ManifestError(f"scenario {scenario.id!r}: empty schedule")
    if scenario.episodes < 1:
        raise ManifestError(f"scenario {scenario.id!r}: episodes must be >= 1")

    lin_clip = float(scenario.get_default("lin_vel_clip"))
    for hold in scenario.schedule:
        if hold.label.lower() == "stop":
            continue
        speed = abs(hold.vx)
        if 0.0 < speed < lin_clip:
            raise ManifestError(
                f"scenario {scenario.id!r} hold {hold.label!r}: |vx|={speed} is inside the "
                f"lin_vel_clip dead zone ({lin_clip}) and would be zeroed as a stop command. "
                "Use a larger speed, or label the hold 'stop' to say you meant it."
            )

    if scenario.probe == "yaw" and scenario.yaw_mode != "delta_yaw_inject":
        # Writing vel_command_b[:, 2] alone is a no-op: wz never enters the observation vector
        # (observations.py concatenates 0*commands[:,0:2] then commands[:,0:1]).
        raise ManifestError(
            f"scenario {scenario.id!r}: probe 'yaw' requires yaw_mode 'delta_yaw_inject'; the "
            "policy cannot observe a wz command."
        )
    if scenario.yaw_mode == "delta_yaw_inject":
        if not any(h.delta_yaw is not None for h in scenario.schedule):
            raise ManifestError(f"scenario {scenario.id!r}: yaw_mode is set but no hold sets delta_yaw")

    if scenario.episode_length_s is not None and scenario.episode_length_s < scenario.total_duration_s:
        raise ManifestError(
            f"scenario {scenario.id!r}: episode_length_s={scenario.episode_length_s} is shorter "
            f"than prologue + schedule ({scenario.total_duration_s:.2f}s); the last holds would "
            "never run."
        )


def compile_schedule(scenario: Scenario, *, dt: float, num_envs: int | None = None) -> CompiledSchedule:
    """Expand a scenario into per-step ``(T, E, …)`` command arrays.

    ``rotate_holds_by_env`` gives env *i* the hold order rotated by *i*, so commanded speed is not
    confounded with position along the course -- on parkour terrain the last hold otherwise always
    lands on the hardest section, and the speed effect and the terrain effect become inseparable.
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")
    n_envs = int(num_envs if num_envs is not None else scenario.episodes)
    prologue_steps = int(round(float(scenario.get_default("prologue_s")) / dt))
    settle_steps = int(round(float(scenario.get_default("settle_s")) / dt))
    hold_steps = [max(1, int(round(h.hold_s / dt))) for h in scenario.schedule]
    total_steps = prologue_steps + sum(hold_steps)
    n_holds = len(scenario.schedule)

    cmd = np.zeros((total_steps, n_envs, 3), dtype=np.float32)
    segment_id = np.full((total_steps, n_envs), -1, dtype=np.int16)
    steady = np.zeros((total_steps, n_envs), dtype=bool)
    uses_delta_yaw = any(h.delta_yaw is not None for h in scenario.schedule)
    delta_yaw = np.zeros((total_steps, n_envs), dtype=np.float32) if uses_delta_yaw else None

    for env in range(n_envs):
        shift = (env % n_holds) if scenario.rotate_holds_by_env and n_holds else 0
        order = [(i + shift) % n_holds for i in range(n_holds)]
        # Prologue holds the first command this env will actually be scored on, so the robot is
        # already settled at that speed when its first steady window opens.
        first = scenario.schedule[order[0]] if order else None
        if first is not None and prologue_steps > 0:
            cmd[:prologue_steps, env, :] = first.as_cmd()
            if delta_yaw is not None and first.delta_yaw is not None:
                delta_yaw[:prologue_steps, env] = first.delta_yaw

        cursor = prologue_steps
        for slot, hold_idx in enumerate(order):
            hold = scenario.schedule[hold_idx]
            n = hold_steps[hold_idx]
            end = cursor + n
            cmd[cursor:end, env, :] = hold.as_cmd()
            segment_id[cursor:end, env] = hold_idx
            steady[cursor + min(settle_steps, n) : end, env] = True
            if delta_yaw is not None and hold.delta_yaw is not None:
                delta_yaw[cursor:end, env] = hold.delta_yaw
            cursor = end

    return CompiledSchedule(
        cmd=cmd,
        segment_id=segment_id,
        steady_mask=steady,
        delta_yaw=delta_yaw,
        n_steps=total_steps,
        dt=float(dt),
        hold_labels=[h.label for h in scenario.schedule],
    )
