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

**Two command frames (omni task, 2026-09-16).** A hold is either a *body* row -- ``vx/vy/wz``
written straight into ``vel_command_b`` (the forward and sideways convention; ``vx`` is speed along
the travel axis) -- or a *world* row: ``speed`` (m/s), ``dir`` (world travel direction, rad) and
``face`` (absolute world heading target, rad). World rows are not written into ``vel_command_b`` at
all: the harness hands each step's row to the command term's ``schedule_hook`` (see
:class:`ScheduleRow`), which does the world -> body conversion and the facing P-control itself, so
the eval sees the same closed loop training does. A scenario is all body rows or all world rows;
mixing them is a manifest error, and every body-row code path below is untouched.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Commands below ``lin_vel_clip`` are treated as "stop" by UniformParkourCommand._resample_command
# (``small_commands_to_zero``), so a hold in that dead zone silently means something different from
# what the manifest says.
DEFAULT_LIN_VEL_CLIP = 0.2

# World-row defaults. ``speed_ellipse`` is the direction-scaled speed band the omni command sampler
# caps commands with (plan D1: fore-aft a, lateral b, backward c); ``ang_vel_max`` is the cap on the
# yaw-rate command the facing P-control may ask for, which sets how long a commanded turn takes.
DEFAULT_SPEED_ELLIPSE: dict[str, float] = {"fwd": 0.35, "lat": 0.25, "back": 0.25}
DEFAULT_ANG_VEL_MAX = 0.3
# Float slack for the "speed inside the ellipse" check: a manifest written to 2 decimals must not
# trip on the cap's last ulp.
_ELLIPSE_TOL = 1e-9

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

#: Hold keys that only a body row may carry.
BODY_HOLD_KEYS: tuple[str, ...] = ("vx", "vy", "wz", "delta_yaw")
#: Hold keys that only a world row may carry.
WORLD_HOLD_KEYS: tuple[str, ...] = ("speed", "dir", "face")


class ManifestError(ValueError):
    """Raised for a manifest that is structurally valid but semantically wrong."""


def wrap_to_pi(angle: float) -> float:
    """Wrap a scalar angle into ``(-pi, pi]``."""
    return float(math.remainder(float(angle), 2.0 * math.pi))


def ellipse_cap(phi: float, fwd: float, lat: float, back: float) -> float:
    """Direction-scaled speed cap: the ellipse radius at body-frame direction ``phi``.

    ``fwd`` is the semi-axis on body +x, ``back`` on body -x, ``lat`` on body +/-y; the two halves
    are separate ellipses joined at +/-pi/2, so ``ellipse_cap(0) == fwd``, ``ellipse_cap(+-pi/2) ==
    lat`` and ``ellipse_cap(pi) == back``, with elliptic interpolation between them:
    ``r(phi) = 1 / hypot(cos(phi)/a, sin(phi)/lat)`` where ``a = fwd`` on the front half and
    ``back`` on the rear half.

    This is the float twin of the omni command term's sampler cap (plan D1 ``ellipse_cap``); the
    manifest validator and the command term must agree or a manifest can ask for a speed the
    sampler would never produce.
    """
    if fwd <= 0.0 or lat <= 0.0 or back <= 0.0:
        raise ManifestError(f"speed_ellipse semi-axes must all be > 0 (got fwd={fwd}, lat={lat}, back={back})")
    c = math.cos(phi)
    s = math.sin(phi)
    a = fwd if c >= 0.0 else back
    return 1.0 / math.hypot(c / a, s / lat)


@dataclass
class ScheduleRow:
    """One world-frame command row handed to the command term's ``schedule_hook`` each step.

    **Contract with the omni command term** (``cmd_term.schedule_hook(row)``):

    * ``row.speed`` -- commanded travel speed, m/s;
    * ``row.dir``   -- commanded travel direction in the WORLD frame, radians;
    * ``row.face``  -- commanded absolute WORLD heading target, radians.

    Each of the three is a python float -- the same value for every env, which is every
    scenario that does not rotate holds per env -- **or** a ``(num_envs,)`` tensor/array on the
    env's device, which is what a ``rotate_holds_by_env`` scenario needs (each env is then on a
    different hold at the same step). A term that only handles scalars must fail loudly on a
    tensor, never silently take element 0. The hook writes the three into the term's own state
    (``travel_dir_w``, ``speed_cmd``, ``heading_target``) and then runs its normal per-step
    ``_update_command()`` so ``vel_command_b`` holds the body-frame command and the yaw-rate slot
    comes from the real facing P-control.

    ``label``/``step`` are informational only (the hold's label and the schedule step index); a
    command term must not depend on them. The object is also dict-like (``row["speed"]``,
    ``row.get("face")``, ``dict(row)``) so a term may accept a plain mapping instead.
    """

    speed: Any
    dir: Any
    face: Any
    label: str = ""
    step: int = -1

    _KEYS = ("speed", "dir", "face", "label", "step")

    def keys(self):  # noqa: D102 - mapping protocol
        return iter(self._KEYS)

    def __getitem__(self, key: str) -> Any:
        if key not in self._KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        return key in self._KEYS

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key) if key in self._KEYS else default

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self._KEYS}


@dataclass
class Hold:
    """One command hold segment.

    Body rows carry ``vx/vy/wz`` (+ ``delta_yaw`` for yaw probes); world rows carry
    ``speed/dir/face`` (all three or none) and nothing else. ``None`` on the world fields is what
    marks a body row -- see the module docstring.
    """

    hold_s: float
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    delta_yaw: float | None = None  # yaw probes only; see Scenario.yaw_mode
    label: str = ""
    speed: float | None = None  # world rows: m/s
    dir: float | None = None  # world rows: world travel direction, rad
    face: float | None = None  # world rows: absolute world heading target, rad

    def as_cmd(self) -> tuple[float, float, float]:
        return (self.vx, self.vy, self.wz)

    @property
    def is_world(self) -> bool:
        """True when this hold carries any world-frame field."""
        return self.speed is not None or self.dir is not None or self.face is not None

    def as_world(self) -> tuple[float, float, float]:
        """``(speed, dir, face)`` as floats; only meaningful on a world row."""
        return (float(self.speed or 0.0), float(self.dir or 0.0), float(self.face or 0.0))


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
    # World-row fields (inert on body rows -- see the module docstring).
    spawn_yaw: str | float | None = None  # "first_face" | a world yaw in radians | None
    speed_ellipse: dict[str, float] | None = None  # {"fwd", "lat", "back"}; world rows only
    ang_vel_max: float = DEFAULT_ANG_VEL_MAX  # WZ_MAX for the turn-feasibility rule

    def __post_init__(self) -> None:
        # A world scenario pins the spawn yaw by default: without it the robot starts at a random
        # heading and the first hold's facing error is a different number in every episode.
        if self.spawn_yaw is None and self.command_frame == "world":
            self.spawn_yaw = "first_face"

    @property
    def schedule_duration_s(self) -> float:
        return float(sum(h.hold_s for h in self.schedule))

    @property
    def total_duration_s(self) -> float:
        return float(self.defaults["prologue_s"]) + self.schedule_duration_s

    @property
    def command_frame(self) -> str:
        """``"world"`` when any hold carries world fields, else ``"body"`` (today's rows).

        Derived, never stored: a scenario with a mix of the two is a validation error, not a
        third frame.
        """
        return "world" if any(h.is_world for h in self.schedule) else "body"

    def spawn_yaw_rad(self) -> float | None:
        """The world yaw the spawn should be pinned to, or ``None`` for "do not pin"."""
        if self.spawn_yaw is None:
            return None
        if isinstance(self.spawn_yaw, str):
            if self.spawn_yaw != "first_face":
                raise ManifestError(
                    f"scenario {self.id!r}: spawn_yaw must be 'first_face' or a number, got {self.spawn_yaw!r}"
                )
            if not self.schedule:
                return None
            return float(self.schedule[0].face or 0.0)
        return float(self.spawn_yaw)

    def get_default(self, key: str) -> Any:
        return self.defaults.get(key, DEFAULTS.get(key))

    def get_speed_ellipse(self) -> tuple[float, float, float]:
        """``(fwd, lat, back)`` semi-axes for :func:`ellipse_cap`."""
        block = self.speed_ellipse or DEFAULT_SPEED_ELLIPSE
        missing = [k for k in ("fwd", "lat", "back") if k not in block]
        if missing:
            raise ManifestError(f"scenario {self.id!r}: speed_ellipse is missing {missing}")
        return (float(block["fwd"]), float(block["lat"]), float(block["back"]))


@dataclass
class CompiledSchedule:
    """Per-step arrays, first dim time, second dim env."""

    cmd: np.ndarray  # (T, E, 3) float32 -- written to vel_command_b (body rows; zeros on world rows)
    segment_id: np.ndarray  # (T, E) int16 -- -1 during the prologue
    steady_mask: np.ndarray  # (T, E) bool
    delta_yaw: np.ndarray | None  # (T, E) float32 or None
    n_steps: int
    dt: float
    hold_labels: list[str]
    # World rows only (None on body rows): the (T, E) command handed to ``schedule_hook`` per step.
    world_dir: np.ndarray | None = None  # (T, E) float64 -- world travel direction, rad
    world_speed: np.ndarray | None = None  # (T, E) float64 -- m/s
    world_face: np.ndarray | None = None  # (T, E) float64 -- absolute world heading target, rad

    @property
    def command_frame(self) -> str:
        return "world" if self.world_dir is not None else "body"

    @property
    def world_uniform(self) -> bool:
        """True when every env is on the same world row at every step (i.e. no hold rotation).

        Decides the :class:`ScheduleRow` field type: floats when uniform (what every committed
        omni scenario is), per-env arrays otherwise.
        """
        if self.world_dir is None:
            return True
        return all(
            bool((arr == arr[:, :1]).all())
            for arr in (self.world_speed, self.world_dir, self.world_face)
        )

    def row(self, step: int, *, label: str | None = None) -> ScheduleRow:
        """The :class:`ScheduleRow` for one step; world rows only.

        Fields are floats when :attr:`world_uniform`, else ``(num_envs,)`` numpy arrays.
        ``label`` defaults to env 0's hold label at that step (``"prologue"`` before the first
        hold) -- informational only, see :class:`ScheduleRow`.
        """
        if self.world_dir is None:
            raise ValueError("row() is only defined for a world-frame schedule")
        idx = min(max(int(step), 0), self.n_steps - 1)
        if label is None:
            seg = int(self.segment_id[idx, 0])
            label = self.hold_labels[seg] if 0 <= seg < len(self.hold_labels) else "prologue"
        if self.world_uniform:
            return ScheduleRow(
                speed=float(self.world_speed[idx, 0]),
                dir=float(self.world_dir[idx, 0]),
                face=float(self.world_face[idx, 0]),
                label=label,
                step=idx,
            )
        return ScheduleRow(
            speed=self.world_speed[idx],
            dir=self.world_dir[idx],
            face=self.world_face[idx],
            label=label,
            step=idx,
        )


def _coerce_hold(raw: dict[str, Any], index: int) -> Hold:
    if "hold_s" not in raw:
        raise ManifestError(f"hold[{index}] is missing 'hold_s'")
    label = str(raw.get("label", "") or f"hold{index}")
    world_keys = [k for k in WORLD_HOLD_KEYS if raw.get(k) is not None]
    if world_keys:
        # World row: all three world fields, and none of the body ones -- a hold that carried both
        # would be read in one frame and written in the other.
        missing = [k for k in WORLD_HOLD_KEYS if raw.get(k) is None]
        if missing:
            raise ManifestError(
                f"hold[{index}] {label!r}: a world-frame hold needs all of {list(WORLD_HOLD_KEYS)}; missing {missing}"
            )
        body_keys = [k for k in BODY_HOLD_KEYS if k in raw]
        if body_keys:
            raise ManifestError(
                f"hold[{index}] {label!r}: world-frame holds carry only {list(WORLD_HOLD_KEYS)}; "
                f"remove {body_keys} (speed/dir/face are converted to a body command by the "
                "command term's schedule_hook, so a vx/vy/wz here would be silently ignored)"
            )
        hold = Hold(
            hold_s=float(raw["hold_s"]),
            label=label,
            speed=float(raw["speed"]),
            dir=float(raw["dir"]),
            face=float(raw["face"]),
        )
    else:
        hold = Hold(
            hold_s=float(raw["hold_s"]),
            vx=float(raw.get("vx", 0.0)),
            vy=float(raw.get("vy", 0.0)),
            wz=float(raw.get("wz", 0.0)),
            delta_yaw=(float(raw["delta_yaw"]) if raw.get("delta_yaw") is not None else None),
            label=label,
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
    # World-row manifest blocks live at the top level (not in ``defaults``) so body-row manifests
    # keep byte-identical ``scenario.defaults`` dicts.
    file_ellipse = data.get("speed_ellipse") or None
    file_ang_vel_max = data.get("ang_vel_max")

    scenarios: list[Scenario] = []
    for raw in data.get("scenarios", []) or []:
        merged = dict(file_defaults)
        merged.update(raw.get("defaults", {}) or {})
        ellipse = raw.get("speed_ellipse") or file_ellipse
        ang_vel_max = raw.get("ang_vel_max", file_ang_vel_max)
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
            spawn_yaw=(
                raw["spawn_yaw"]
                if isinstance(raw.get("spawn_yaw"), str)
                else (float(raw["spawn_yaw"]) if raw.get("spawn_yaw") is not None else None)
            ),
            speed_ellipse=({str(k): float(v) for k, v in ellipse.items()} if ellipse else None),
            ang_vel_max=(float(ang_vel_max) if ang_vel_max is not None else DEFAULT_ANG_VEL_MAX),
        )
        validate_scenario(scenario)
        scenarios.append(scenario)

    ids = [s.id for s in scenarios]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ManifestError(f"{path}: duplicate scenario ids {sorted(dupes)}")
    return scenarios, file_defaults


def _validate_world_scenario(scenario: Scenario) -> None:
    """The world-frame rules (plan D10 C1). Reached only from :func:`validate_scenario`."""
    if any(not h.is_world for h in scenario.schedule):
        mixed = [h.label for h in scenario.schedule if not h.is_world]
        raise ManifestError(
            f"scenario {scenario.id!r}: mixed command frames -- holds {mixed} carry no "
            "speed/dir/face while others do. A scenario is all body rows or all world rows."
        )
    if scenario.probe == "yaw":
        # The world frame steers by an absolute facing target through the command term's own
        # P-control; injecting a delta_yaw on top would fight it and measure neither.
        raise ManifestError(
            f"scenario {scenario.id!r}: probe 'yaw' is a body-row probe (delta_yaw injection); "
            "world rows steer through 'face' and the command term's facing control."
        )

    fwd, lat, back = scenario.get_speed_ellipse()
    wz_max = float(scenario.ang_vel_max)
    if wz_max <= 0.0:
        raise ManifestError(f"scenario {scenario.id!r}: ang_vel_max must be > 0 (got {wz_max})")
    settle_s = float(scenario.get_default("settle_s"))

    for hold in scenario.schedule:
        body = [k for k in BODY_HOLD_KEYS if (getattr(hold, k) or 0.0)]
        if body:
            raise ManifestError(
                f"scenario {scenario.id!r} hold {hold.label!r}: world-frame holds carry only "
                f"{list(WORLD_HOLD_KEYS)}; {body} is set."
            )
        speed, direction, face = hold.as_world()
        if speed < 0.0:
            raise ManifestError(f"scenario {scenario.id!r} hold {hold.label!r}: speed must be >= 0")
        cap = ellipse_cap(wrap_to_pi(direction - face), fwd, lat, back)
        if speed > cap + _ELLIPSE_TOL:
            raise ManifestError(
                f"scenario {scenario.id!r} hold {hold.label!r}: speed={speed} exceeds the "
                f"direction-scaled cap {cap:.4f} m/s at body direction "
                f"{wrap_to_pi(direction - face):+.4f} rad (speed_ellipse fwd={fwd}, lat={lat}, "
                f"back={back}); the sampler would never produce this command."
            )

    # Turn feasibility: the facing P-control is capped at ``ang_vel_max``, so a hold that changes
    # the facing target by more than it can physically turn in the hold measures a turn that never
    # finished -- the "settle then score" contract silently becomes "score the transient".
    faces = [h.as_world()[2] for h in scenario.schedule]
    pairs = list(zip(range(1, len(faces)), faces[:-1], faces[1:]))
    if scenario.rotate_holds_by_env and len(faces) > 1:
        # Every env starts at a different hold, so the wrap-around pair is a real transition too.
        pairs.append((0, faces[-1], faces[0]))
    elif scenario.spawn_yaw is not None:
        # The spawn pins the body heading, so the first hold is a turn too whenever the pin and the
        # first face disagree (they agree by construction under spawn_yaw "first_face").
        spawn = scenario.spawn_yaw_rad()
        if spawn is not None:
            pairs.insert(0, (0, spawn, faces[0]))
    for idx, prev_face, face in pairs:
        d_face = abs(wrap_to_pi(face - prev_face))
        need = settle_s + d_face / wz_max
        hold = scenario.schedule[idx]
        if hold.hold_s < need:
            raise ManifestError(
                f"scenario {scenario.id!r} hold {hold.label!r}: hold_s={hold.hold_s} is shorter "
                f"than settle_s + |wrap(dface)|/ang_vel_max = {settle_s} + {d_face:.4f}/{wz_max} "
                f"= {need:.2f}s; the commanded {d_face:.4f} rad turn cannot finish inside the hold."
            )

    if scenario.spawn_yaw is not None:
        scenario.spawn_yaw_rad()  # raises on a bad spelling


def validate_scenario(scenario: Scenario) -> None:
    """Reject manifests that would silently measure something other than what they say."""
    if not scenario.schedule:
        raise ManifestError(f"scenario {scenario.id!r}: empty schedule")
    if scenario.episodes < 1:
        raise ManifestError(f"scenario {scenario.id!r}: episodes must be >= 1")

    if scenario.command_frame == "world":
        _validate_world_scenario(scenario)
        if scenario.episode_length_s is not None and scenario.episode_length_s < scenario.total_duration_s:
            raise ManifestError(
                f"scenario {scenario.id!r}: episode_length_s={scenario.episode_length_s} is shorter "
                f"than prologue + schedule ({scenario.total_duration_s:.2f}s); the last holds would "
                "never run."
            )
        return

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
    # World rows: ``cmd`` stays all zeros (nothing writes vel_command_b directly) and these three
    # carry the row handed to ``schedule_hook`` instead.
    # float64, unlike ``cmd``: these are handed to the command term as python floats, and a
    # float32 round-trip would turn a manifest's 0.24 into 0.2399999946.
    is_world = scenario.command_frame == "world"
    world_speed = np.zeros((total_steps, n_envs), dtype=np.float64) if is_world else None
    world_dir = np.zeros((total_steps, n_envs), dtype=np.float64) if is_world else None
    world_face = np.zeros((total_steps, n_envs), dtype=np.float64) if is_world else None

    for env in range(n_envs):
        shift = (env % n_holds) if scenario.rotate_holds_by_env and n_holds else 0
        order = [(i + shift) % n_holds for i in range(n_holds)]
        # Prologue holds the first command this env will actually be scored on, so the robot is
        # already settled at that speed when its first steady window opens.
        first = scenario.schedule[order[0]] if order else None
        if first is not None and prologue_steps > 0:
            if is_world:
                s0, d0, f0 = first.as_world()
                world_speed[:prologue_steps, env] = s0
                world_dir[:prologue_steps, env] = d0
                world_face[:prologue_steps, env] = f0
            else:
                cmd[:prologue_steps, env, :] = first.as_cmd()
            if delta_yaw is not None and first.delta_yaw is not None:
                delta_yaw[:prologue_steps, env] = first.delta_yaw

        cursor = prologue_steps
        for slot, hold_idx in enumerate(order):
            hold = scenario.schedule[hold_idx]
            n = hold_steps[hold_idx]
            end = cursor + n
            if is_world:
                s, d, f = hold.as_world()
                world_speed[cursor:end, env] = s
                world_dir[cursor:end, env] = d
                world_face[cursor:end, env] = f
            else:
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
        world_dir=world_dir,
        world_speed=world_speed,
        world_face=world_face,
    )
