# SPDX-License-Identifier: BSD-3-Clause
"""Training phases for the sideways-walk task -- the forward schedule of record, re-pointed.

The sideways task travels along the robot's body **+y** axis ("Shape B": the commanded speed
stays in ``vel_command_b[:, 0]`` and is read in the travel frame, the body frame rotated by
``TRAVEL_YAW = +pi/2``). Everything about the *schedule* is the forward one: this module takes
each forward ``PhaseSpec`` (``crab_hex_forward_task/config/crab_hex/crab_hex_phases.py``) and
changes exactly three things:

1. the gym task id            ``Isaac-Crab-Hex-{Flat-Walk,Teacher,Student}-v0`` -> ``…-Side-…-v0``
2. the rsl-rl experiment name ``crab_hex_{flat_walk,teacher,student}``          -> ``crab_hex_side_…``
3. **RSI off** -- ``KRABBY_RSI_FRAC`` / ``KRABBY_RSI_BANK`` are dropped from every window. The
   bank of record is a *forward* gait (identity yaw, +0.25 m/s body-x); an RSI reset would
   teleport a sideways walker back to yaw 0. (The forward ``_window_env`` re-adds
   ``KRABBY_RSI_BANK`` unconditionally, so the filter is applied to the BUILT env, not to the
   knob blocks.) A sideways bank can be harvested later -- see the campaign charter's fallback.

Phase-3 (distillation) specs additionally export ``KRABBY_STUDENT_MDP=1``: the forward student
config gates on the forward ``is_student_phase()``, which only ever reads ``KRABBY_PHASE`` /
``KRABBY_STUDENT_MDP`` -- without the flag the sideways student would silently build the legacy
2b2 MDP instead of the 2c one.

**One task per process.** The sideways package activates from its own variable
``KRABBY_SIDE_PHASE`` and NEVER writes ``KRABBY_PHASE``; the forward package ignores
``KRABBY_SIDE_PHASE``. The expanded ``KRABBY_*`` knob names are shared, so exactly one of the two
preset variables may be set in a process: ``activate_phase()`` raises when both are.
No ``legacy_golden_*`` phases exist here (forward-only reproduction schedule).

Pure Python (no Isaac, no ``parkour_tasks`` import): the forward presets are loaded BY FILE PATH
under the distinct module name ``crab_hex_phases__forward``, so this module is importable by path
from a test or a driver script without booting anything. Unit-tested in
``tests/unit/test_crab_hex_side_phases.py``.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Mapping, MutableMapping

REPO_ROOT = Path(__file__).resolve().parents[6]
ASSETS = REPO_ROOT / "assets"
EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"  # the task's campaign records

# ---------------------------------------------------------------------------- forward presets
_FWD_MODULE_NAME = "crab_hex_phases__forward"
_FWD_PHASES_PY = (
    Path(__file__).resolve().parents[3]
    / "crab_hex_forward_task" / "config" / "crab_hex" / "crab_hex_phases.py"
)


def _load_by_path(path: Path, name: str):
    """Load a module by file path under ``name`` (never through ``parkour_tasks.*``)."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - unreachable for a real file
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclasses resolve string annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


_fwd = sys.modules.get(_FWD_MODULE_NAME)
if _fwd is None:
    _fwd = _load_by_path(_FWD_PHASES_PY, _FWD_MODULE_NAME)

# ---------------------------------------------------------------------------- re-exports
# The plant table, the knob blocks and the schedule helpers are shared verbatim: the sideways
# campaign runs the same plant (A15+B) and the same windows as the forward lineage.
PLANTS = _fwd.PLANTS
MAIN_PLANT = _fwd.MAIN_PLANT
MAIN_ASSET = _fwd.MAIN_ASSET
MAIN_ASSET_NAMES = _fwd.MAIN_ASSET_NAMES
plant_usd_path = _fwd.plant_usd_path
plant_name_for_path = _fwd.plant_name_for_path

PhaseSpec = _fwd.PhaseSpec
WINDOW_ITERS = _fwd.WINDOW_ITERS
RAMP_ITERS = _fwd.RAMP_ITERS
STEPS_PER_ITER = _fwd.STEPS_PER_ITER
EPS = _fwd.EPS

FORMATION = _fwd.FORMATION
ELEMENTS_5K = _fwd.ELEMENTS_5K
ELEMENTS_10K = _fwd.ELEMENTS_10K
WALKING_SLOTS = _fwd.WALKING_SLOTS
LONG_EPISODES = _fwd.LONG_EPISODES
WIDENED_CORRIDORS = _fwd.WIDENED_CORRIDORS
GAIT_WEIGHTS = _fwd.GAIT_WEIGHTS
GAIT_RAMPS = _fwd.GAIT_RAMPS
phaseout_spec = _fwd.phaseout_spec
promote_fracs_for_horizon = _fwd.promote_fracs_for_horizon

_MDP_KEYS = _fwd._MDP_KEYS
_student_env = _fwd._student_env

# ---------------------------------------------------------------------------- sideways deltas
PHASE_VAR = "KRABBY_SIDE_PHASE"
#: Dropped from every sideways window (the bank of record is a forward gait; see the docstring).
RSI_KEYS = ("KRABBY_RSI_FRAC", "KRABBY_RSI_BANK")

TASK_MAP = {
    "Isaac-Crab-Hex-Flat-Walk-v0": "Isaac-Crab-Hex-Side-Walk-v0",
    "Isaac-Crab-Hex-Teacher-v0": "Isaac-Crab-Hex-Side-Teacher-v0",
    "Isaac-Crab-Hex-Student-v0": "Isaac-Crab-Hex-Side-Student-v0",
}
EXPERIMENT_MAP = {
    "crab_hex_flat_walk": "crab_hex_side_flat_walk",
    "crab_hex_teacher": "crab_hex_side_teacher",
    "crab_hex_student": "crab_hex_side_student",
}

#: The paradigm chain; no ``legacy_golden_*`` (forward-only reproduction schedule).
PHASE_NAMES = ("1a", "2a", "2b", "2c", "3a", "3b")


def build_phases() -> dict[str, PhaseSpec]:
    """The forward schedule of record with the task ids, experiment names and RSI keys re-pointed."""
    p: dict[str, PhaseSpec] = {}
    for name in PHASE_NAMES:
        f = _fwd.PHASES[name]
        env = {k: v for k, v in f.env.items() if k not in RSI_KEYS}
        if f.kind == "distill":
            # the forward student cfg gates on the forward is_student_phase(), which reads
            # KRABBY_PHASE / KRABBY_STUDENT_MDP only -- without this it builds the legacy 2b2 MDP
            env["KRABBY_STUDENT_MDP"] = "1"
        p[name] = replace(
            f,
            task=TASK_MAP[f.task],
            experiment=EXPERIMENT_MAP[f.experiment],
            env=env,
            notes=f"sideways: {f.notes}",
        )
    return p


PHASES: dict[str, PhaseSpec] = build_phases()

PHASE_MODES = {name: spec.teacher_mode for name, spec in PHASES.items() if spec.teacher_mode}
STUDENT_PHASES = tuple(name for name, spec in PHASES.items() if spec.kind == "distill")


def phase_env(name: str, plant: str | None = None) -> dict[str, str]:
    """Full environment a sideways phase expands to, without touching ``os.environ``.

    Writes ``KRABBY_SIDE_PHASE`` and never ``KRABBY_PHASE``. Unknown names -- including the
    forward-only ``legacy_golden_*`` ones -- raise ``KeyError``.
    """
    spec = PHASES[name]
    ev = dict(spec.env)
    ev[PHASE_VAR] = name
    if spec.teacher_mode:
        ev["KRABBY_HEX_TEACHER_MODE"] = spec.teacher_mode
    if plant:
        ev["KRABBY_PLANT"] = plant
        usd = plant_usd_path(plant)
        if usd:
            ev["KRABBY_HEX_USD_PATH"] = usd
    return ev


def activate_phase(environ: MutableMapping[str, str] | None = None) -> dict[str, str]:
    """Expand ``KRABBY_SIDE_PHASE`` / ``KRABBY_PLANT`` into ``environ`` with setdefault.

    Returns the keys that were set (empty when no sideways phase is requested). Explicit
    variables always win, exactly as the forward activation does.

    * both preset variables set -> ``RuntimeError`` (one task per process: the expanded knob
      names are shared, and the forward activation would setdefault the RSI keys first);
    * ``KRABBY_SIDE_PHASE`` unset, ``KRABBY_PHASE`` set -> ``{}`` (the forward package owns
      this process);
    * neither set -> delegate the plant-only case to the forward activation (idempotent).
    """
    env = os.environ if environ is None else environ
    name = env.get(PHASE_VAR, "").strip()
    fwd_name = env.get("KRABBY_PHASE", "").strip()
    if name and fwd_name:
        raise RuntimeError(
            "one task per process: KRABBY_PHASE and KRABBY_SIDE_PHASE are both set"
        )
    if not name:
        if fwd_name:
            return {}
        return _fwd.activate_phase(env)
    plant = env.get("KRABBY_PLANT", "").strip() or None
    applied: dict[str, str] = {}
    for k, v in phase_env(name, plant).items():
        if k in (PHASE_VAR, "KRABBY_PLANT"):
            continue
        if k not in env:
            env[k] = v
            applied[k] = v
    return applied


def is_student_phase(environ: Mapping[str, str] | None = None) -> bool:
    """True when a sideways phase-3 preset is active, or ``KRABBY_STUDENT_MDP`` asks for the
    phase-3 MDP structure without the preset's terrain / DR knobs (student evaluation)."""
    env = os.environ if environ is None else environ
    if env.get(PHASE_VAR, "").strip() in STUDENT_PHASES:
        return True
    return env.get("KRABBY_STUDENT_MDP", "").strip().lower() in ("1", "true", "yes", "phase")
