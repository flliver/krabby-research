# SPDX-License-Identifier: BSD-3-Clause
"""Training phases for the omnidirectional task -- the forward windows, re-pointed (plan D11).

The omni task travels in any WORLD direction while facing any WORLD heading. Nothing about
the *schedule* is new: every window below is a forward ``PhaseSpec``
(``crab_hex_forward_task/config/crab_hex/crab_hex_phases.py``) with a fixed list of key
deltas, so the gait weights, the ``KRABBY_PHASEOUT`` ramps, the tracking knobs, the episode
length, the DR block, the terrain mode/fraction/difficulty and the element doses are the
forward numbers, in one place, unchanged.

Phase 1 is THREE stages, all built from the forward **window 0** env (``FORMATION`` +
``WALKING_SLOTS`` + ``LONG_EPISODES``, no ramps) and differing only in the command sampler's
direction band, its slot mix and the RSI frame source:

=====  =========  ==========  ======================  ====================  =============
stage  iterations resume      ``KRABBY_OMNI_DIR_BAND``  ``KRABBY_OMNI_SLOTS``  RSI ``source``
=====  =========  ==========  ======================  ====================  =============
1a     1000       (scratch)   ``0:0`` (body +x)       ``0.2:0.8:0:0``       ``0`` (forward)
1b     1000       1a          ``1.5708:1.5708`` (+y)  ``0.2:0.8:0:0``       ``1`` (sideways)
1c     5000       1b          unset (world-uniform)   ``0.15:0.55:0.1:0.2`` ``all``
=====  =========  ==========  ======================  ====================  =============

1c additionally arms the facing half: ``KRABBY_OMNI_FACING=0.2:0.6`` and ``KRABBY_YAW_W=0.2``
(the facing reward is inert in 1a/1b -- no turning slots, facing held -- so it stays at the
forward 0.0 there). ``2a`` resumes 1c and widens the facing band (0.2:1.2), ``2b``/``2c``
widen it again (0.2:2.4) and carry the forward elements of their own windows.
``2c_clockheld`` is the 2c env minus ``KRABBY_PHASEOUT`` with ``KRABBY_CLOCK_W=1.0`` -- the
control twin for "did the clock phase-out cost anything?", run from the 2b head.

**Per-window deltas against the forward env, in full** (asserted key-by-key in
``tests/unit/test_crab_hex_omni_phases.py``):

* dropped: ``KRABBY_HEADING`` (the facing sampler replaces it -- and ``_check_omni`` refuses a
  process that still has it set), ``KRABBY_RSI_SPAWN_FIX`` and the corridor knobs
  (:data:`DROPPED_KEYS`; only ``KRABBY_HEADING`` is actually present in a forward window
  today, the rest are defensive);
* re-pointed: ``KRABBY_RSI_BANK`` -> the omni bank, ``KRABBY_FLAT_TERRAIN_GEOM=ring`` (every
  window -- ``apply_omni_ring_knobs`` REQUIRES it and fails loudly on a leaked corridor
  preset), ``KRABBY_TERRAIN_PROMOTE=0.45:0.25`` (the omni parkour term promotes on a
  dimensionless TRACKING RATIO, so the forward horizon scaling does not apply), and -- Revision 4,
  2026-09-18 -- ``KRABBY_TRACK_SIGMA2`` 0.1 -> 0.04 and ``KRABBY_TRACK_L1_W`` -1.0 -> -2.0, the
  two tracking knobs every forward window also sets (so they read as DIFFERING VALUES in the
  delta, not additions);
* added: the ``KRABBY_OMNI_*`` block (:data:`OMNI_COMMON` in every window, plus the per-stage
  slots / direction band / RSI source / facing band), ``KRABBY_YAW_PROGRESS_W=1.0`` in every
  window (the added rotational-progress term; 1a/1b command no turn, so the only thing it can
  pay there is correcting the robot's own yaw drift back to the facing it held),
  ``KRABBY_OMNI_ANG_VEL_STD=0.5`` and ``KRABBY_OMNI_CLOCK_COMBINE=product`` (Revision 4) and
  ``KRABBY_YAW_W`` from 1c.

**Body-frame twins** (2026-10-15, robot-frame velocity commands campaign): ``1a_b`` / ``1b_b`` /
``2a_b`` are the base preset's env plus :data:`BODY_DELTA` (one dict, pinned key by key in the unit
tests); they are in :data:`PHASES` but not in :data:`PHASE_NAMES`, and refuse to build until the
campaign overlay supplies the measured envelope.

**One task per process.** This package activates from its own variable ``KRABBY_OMNI_PHASE``
and NEVER writes ``KRABBY_PHASE`` or ``KRABBY_SIDE_PHASE``. The expanded ``KRABBY_*`` knob
names are shared by all three tasks, so exactly one preset variable may be set:
``activate_phase()`` raises when ``KRABBY_OMNI_PHASE`` is set together with EITHER of the
other two, and is a strict no-op (returns ``{}``) when it is unset -- ``import_packages``
imports forward, then omni, then sideways, and a forward or sideways process must pass
through this module untouched. There is no phase 3 (teacher only; the depth-camera event bug
and the camera mount decision are open).

Pure Python (no Isaac, no ``parkour_tasks`` import): the forward presets are loaded BY FILE
PATH under the distinct module name ``crab_hex_phases__forward``, so this module is
importable by path from a test or a driver script without booting anything. Unit-tested in
``tests/unit/test_crab_hex_omni_phases.py``.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import MutableMapping

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
# The plant table, the knob blocks and the schedule helpers are shared verbatim: the omni
# campaign runs the same plant (A15+B) and the same window stack as the forward lineage.
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

# ---------------------------------------------------------------------------- omni deltas
PHASE_VAR = "KRABBY_OMNI_PHASE"

#: The omni RSI bank: forward + sideways frames, provenance-tagged (plan D6).
OMNI_BANK = EXPERIMENTS / "2026-09-16_1805_omni_formation/rsi_bank_omni.npz"

#: Short stage length (1a / 1b): seeded re-formation of one travel axis, not a full window.
SHORT_ITERS = 1000

#: Dropped from every omni window. ``KRABBY_HEADING`` is the forward/sideways turning-school
#: knob -- inert under the omni command term and refused by ``_check_omni``. The rest never
#: appear in a forward window today and are dropped defensively: they configure the corridor
#: courses (widths, stone width, the platform spawn offset and the along-course spread), none
#: of which exists on a ring.
DROPPED_KEYS = (
    "KRABBY_HEADING",
    "KRABBY_RSI_SPAWN_FIX",
    "KRABBY_CORRIDOR_HALF_WIDTH",
    "KRABBY_STONE_WIDTH",
    "KRABBY_SPAWN_OFFSET",
    "KRABBY_SPAWN_SPREAD",
)

#: Applied to EVERY omni window (re-pointed forward keys + the phase-independent omni block).
OMNI_COMMON: dict[str, str] = {
    "KRABBY_RSI_BANK": str(OMNI_BANK),
    # apply_omni_ring_knobs REQUIRES this exact value in every omni process.
    "KRABBY_FLAT_TERRAIN_GEOM": "ring",
    # ratio semantics (OmniParkourEvent promotes on ep_track_dist / ep_cmd_dist), so the
    # forward horizon scaling (promote_fracs_for_horizon) does NOT apply here.
    "KRABBY_TERRAIN_PROMOTE": "0.45:0.25",
    "KRABBY_OMNI_SPEED_ELLIPSE": "0.35:0.25:0.25",
    "KRABBY_OMNI_ANG_VEL_MAX": "0.3",
    "KRABBY_OMNI_CLOCK_TURN_R": "1.2",
    "KRABBY_OMNI_YAW_RANGE": "-3.14159:3.14159",
    # The added rotational progress term (the 36th reward name), armed in EVERY window --
    # including 1a/1b, which draw no turning slot (slots 2/3 at probability 0) and hold their
    # facing, so there is no COMMANDED turn for it to pay: its `cmd_wz != 0` gate opens there
    # only if the robot's own yaw drifts past the P-control dead band (0.133 rad), and then it
    # pays for turning back -- a heading hold, aligned with everything else those stages want.
    # Arming it everywhere keeps the dose in ONE line and means a stage that later gains a
    # turning slot cannot start it un-rewarded.
    # (2026-09-17: added after the 500-iteration yaw-capability smoke stood and stepped in
    # place but rotated the WRONG WAY -- median -0.025 rad/s against +0.30 commanded.)
    "KRABBY_YAW_PROGRESS_W": "1.0",
    # 0.6 = the 1c facing band's edge (KRABBY_OMNI_FACING=0.2:0.6): the yaw-rate command
    # saturates beyond the dead band (ang_vel_clip 0.2), so an out-of-band facing target needs
    # only the sign, not its raw magnitude, once the P-control is already pegged. Kept at 0.6
    # in 2a+ DELIBERATELY even though the facing band there ramps to 1.2 then 2.4: that ramp
    # only changes how LONG a turn lasts (how far past the dead band the target sits), not what
    # the policy must be able to see to drive it -- the observation only ever needs to resolve
    # the dead-band edge. (2026-09-17: the third yaw-capability smoke's 90-degree canary fell
    # 16/16 with the raw error at 1.57 rad -- 2.6x the 1c training band -- while the in-band
    # 0.5-rad diagnostic settled 16/16; with this clip the same head turned at +0.063 / +0.094 /
    # +0.071 rad/s on turn_l90 / turn_r90 / move_turn, 3/16 falls -- gate PASS. User decision,
    # Revision 3.)
    "KRABBY_OMNI_FACING_OBS_CLIP": "0.6",
    # ---------------------------------------------------------------- Revision 4 (2026-09-18)
    # THE REWARD ECONOMY. User decision D after the verified 1c diagnosis
    # (ANALYSIS_1c_failure_2026-09-18.md): under 1c's terms the cheapest policy is to stand
    # still and keep time. Measured on the seed-3 chain: speed tracking fell 0.747 (1a) ->
    # 0.433 (1b) -> 0.245 (1c) while mean reward ROSE 61 -> 93, and a Monte-Carlo of the 1c
    # sampler predicts a ZERO-VELOCITY policy earns track_lin_vel_xy_exp 0.835 / L1 -0.182
    # against the logged 0.826 / -0.177 -- i.e. the logged head was, to three digits, the
    # standing solution. Motion-REQUIRING terms were 1.2 % of its 2.73/s positive income.
    #
    # All four values are UNIFORM along the chain, deliberately: the 1c failure was a drift
    # under a fixed economy, and a window whose economy differs from its neighbour's is exactly
    # the handoff that lost forward travel at 1a->1b. They are PROVISIONAL until the offline
    # replay on the saved 1a/1c raws confirms the walker out-earns the creeper.
    #
    # KRABBY_TRACK_SIGMA2 0.1 -> 0.04 (every forward window sets 0.1, so this is a DIFFERING
    #   VALUE in the delta, not an addition): track_lin_vel_xy_exp = exp(-|cmd-v|^2 / sigma2).
    #   At v = 0 against the 1c sampler's mean move-row command it pays 0.835 at 0.1 and ~0.21
    #   at 0.04 -- standing stops being 99 % of walking's tracking income.
    # KRABBY_TRACK_L1_W -1.0 -> -2.0 (also set by every forward window): doubles the linear
    #   ||cmd - v|| penalty, the one term whose gradient does not vanish at v = 0 (break-even
    #   for a stander at sigma2 0.1 alone would need ~-2.9; with sigma2 0.04 the pair clears it).
    # KRABBY_OMNI_ANG_VEL_STD 0.15 -> 0.5 (the FORWARD value, sqrt(0.25); the omni rewards class
    #   body declares 0.15 per plan D4, so this knob returns the kernel to the forward width):
    #   at 0.15, ordinary gait yaw wobble on the 88 % of steps that command NO turn costs
    #   0.36/s at 0.10 rad/s and 0.63/s at 0.15 rad/s -- a tax on walking larger than the whole
    #   translation income. At 0.5 the kernel is flat there; the turning signal is carried by
    #   reward_yaw_progress (weight 1.0), which did not exist when the plan chose 0.15.
    # KRABBY_OMNI_CLOCK_COMBINE sum -> product: reward_clock_schedule's income stops being
    #   (swing half + stance half), where a creeper collects the whole stance half for free
    #   (~0.5 ceiling; clock+apex were 34 % of 1c's positive income), and becomes
    #   swing_quality x stance_quality, which a foot that never leaves the ground multiplies
    #   toward 0. The MECHANISM is the forward task's own (clock_schedule_income(..., combine=),
    #   2026-08-22, with its own unit tests); only the omni SPELLING of the knob is new, so no
    #   forward preset moves.
    "KRABBY_TRACK_SIGMA2": "0.04",
    "KRABBY_TRACK_L1_W": "-2.0",
    "KRABBY_OMNI_ANG_VEL_STD": "0.5",
    "KRABBY_OMNI_CLOCK_COMBINE": "product",
}

#: Slot mixes (stand : move : turn : move+turn).
SLOTS_AXIS = "0.2:0.8:0:0"          # 1a / 1b -- one travel axis, facing held
SLOTS_OMNI = "0.15:0.55:0.10:0.20"  # 1c
SLOTS_TEACHER = "0.15:0.45:0.15:0.25"  # 2a+

TASK_MAP = {
    "Isaac-Crab-Hex-Flat-Walk-v0": "Isaac-Crab-Hex-Omni-Walk-v0",
    "Isaac-Crab-Hex-Teacher-v0": "Isaac-Crab-Hex-Omni-Teacher-v0",
}
EXPERIMENT_MAP = {
    "crab_hex_flat_walk": "crab_hex_omni_flat_walk",
    "crab_hex_teacher": "crab_hex_omni_teacher",
}

#: The chain of record (1a -> 1b -> 1c -> 2a -> 2b, then 2c and its clock-held twin from 2b).
PHASE_NAMES = ("1a", "1b", "1c", "2a", "2b", "2c", "2c_clockheld")

# ---------------------------------------------------------------------- the body-frame presets
#: NOTE(robot-frame velocity commands campaign, ``experiments/2026-10-15_body_cmd``, STEP1_SPEC §6):
#: the ``_b`` presets are a base world preset's env with ONE delta, pinned key by key here (a value
#: sets the key, ``None`` DROPS it). They switch the command to a held robot-frame ``(vx, vy, wz)``
#: (``omni_knobs._apply_command_frame``), blank the facing-error observation, dose the windowed
#: yaw-progress tent at 2.0 (the CHARTER's 2026-10-15 amendment) and turn RSI off (an RSI frame
#: stages a WORLD command); they drop the two world-facing knobs the body mode refuses. The measured
#: envelope (``KRABBY_OMNI_BODY_ENVELOPE`` / ``KRABBY_OMNI_BODY_CAP``) is deliberately NOT here --
#: the campaign overlay sets it after step 4 -- so a ``_b`` preset alone refuses to build.
BODY_DELTA: dict[str, str | None] = {
    "KRABBY_OMNI_COMMAND_FRAME": "body",
    "KRABBY_OMNI_FACING_OBS_ENCODING": "none",
    "KRABBY_YAW_PROGRESS_W": "2.0",
    # The 2-s windowed yaw term (reward_tracking_yaw's body func) is the HEADING HOLD on straight rows: weighted
    # in every body preset, 1a_b / 1b_b included (the world 1a / 1b presets leave KRABBY_YAW_W unset = 0 because
    # their facing relay held heading; body mode has no relay). CHARTER 2026-10-15.
    "KRABBY_YAW_W": "0.2",
    "KRABBY_RSI_FRAC": "0",
    "KRABBY_OMNI_FACING": None,
    "KRABBY_OMNI_FACING_OBS_CLIP": None,
}
#: The world presets that get a body twin, and the twins' names (``<base>_b``). NOT in
#: :data:`PHASE_NAMES` (the world chain of record is unchanged); reachable through :func:`phase_env`
#: / :func:`activate_phase` like every preset.
BODY_BASES = ("1a", "1b", "2a")
BODY_PHASE_SUFFIX = "_b"
BODY_PHASE_NAMES = tuple(f"{b}{BODY_PHASE_SUFFIX}" for b in BODY_BASES)


def _body_env(env: dict[str, str]) -> dict[str, str]:
    """A world preset env with :data:`BODY_DELTA` applied (set the str values, drop the ``None`` ones)."""
    ev = dict(env)
    for key, value in BODY_DELTA.items():
        if value is None:
            ev.pop(key, None)
        else:
            ev[key] = value
    return ev

#: No distillation in this plan (teacher only).
STUDENT_PHASES: tuple[str, ...] = ()

#: Eval rows the driver runs after each stage (manifest ``scenarios_omni.yaml``). The short
#: stages get the flat canary only -- ~2 min instead of ~40 -- and are gated on their own hold.
EVAL_SHORT = ("slow",)
EVAL_FULL = ("slow", "light", "ring")


def _omni_env(forward_env: dict[str, str], **extra: str) -> dict[str, str]:
    """A forward window env with the omni deltas applied (drop / re-point / add)."""
    ev = {k: v for k, v in forward_env.items() if k not in DROPPED_KEYS}
    ev.update(OMNI_COMMON)
    ev.update(extra)
    return ev


def build_phases() -> dict[str, PhaseSpec]:
    """The forward windows with the task ids, experiment names and omni knobs re-pointed."""
    f1a = _fwd.PHASES["1a"]
    p: dict[str, PhaseSpec] = {}

    # --- phase 1: three stages off the SAME forward window-0 env ---
    stage_1 = (
        ("1a", None, SHORT_ITERS, {"KRABBY_OMNI_DIR_BAND": "0:0", "KRABBY_OMNI_SLOTS": SLOTS_AXIS,
                                   "KRABBY_OMNI_RSI_SOURCE": "0"},
         "omni stage 1a: forward-axis walk from scratch, forward-bank RSI"),
        ("1b", "1a", SHORT_ITERS, {"KRABBY_OMNI_DIR_BAND": "1.5708:1.5708",
                                   "KRABBY_OMNI_SLOTS": SLOTS_AXIS, "KRABBY_OMNI_RSI_SOURCE": "1"},
         "omni stage 1b: sideways-axis walk (body +y), sideways-bank RSI"),
        ("1c", "1b", WINDOW_ITERS, {"KRABBY_OMNI_SLOTS": SLOTS_OMNI, "KRABBY_OMNI_RSI_SOURCE": "all",
                                    "KRABBY_OMNI_FACING": "0.2:0.6", "KRABBY_YAW_W": "0.2"},
         "omni stage 1c: omnidirectional formation, four slots, merged bank"),
    )
    for name, prev, iters, extra, note in stage_1:
        p[name] = replace(
            f1a,
            name=name,
            task=TASK_MAP[f1a.task],
            experiment=EXPERIMENT_MAP[f1a.experiment],
            iterations=iters,
            env=_omni_env(f1a.env, **extra),
            resume_from=prev,
            notes=note,
            eval_scenarios=EVAL_SHORT if iters == SHORT_ITERS else EVAL_FULL,
        )

    # --- phase 2: the forward teacher windows, facing band widening ---
    teacher = (
        ("2a", "1c", "0.2:1.2", "elements @5k, apex/airtime/stride -> half"),
        ("2b", "2a", "0.2:2.4", "elements @10k, satellites -> eps"),
        ("2c", "2b", "0.2:2.4", "clock 1.0 -> 0.5"),
    )
    for name, prev, facing, note in teacher:
        f = _fwd.PHASES[name]
        p[name] = replace(
            f,
            task=TASK_MAP[f.task],
            experiment=EXPERIMENT_MAP[f.experiment],
            env=_omni_env(f.env, **{"KRABBY_OMNI_SLOTS": SLOTS_TEACHER,
                                    "KRABBY_OMNI_RSI_SOURCE": "all",
                                    "KRABBY_OMNI_FACING": facing}),
            resume_from=prev,
            notes=f"omni phase 2 teacher: {note}",
            eval_scenarios=EVAL_FULL,
        )

    # --- the clock-held control: 2c without the clock phase-out, also from the 2b head ---
    env_ch = {k: v for k, v in p["2c"].env.items() if k != "KRABBY_PHASEOUT"}
    env_ch["KRABBY_CLOCK_W"] = "1.0"
    p["2c_clockheld"] = replace(
        p["2c"],
        name="2c_clockheld",
        env=env_ch,
        resume_from="2b",
        notes="omni phase 2 control: 2c with the gait clock held at 1.0 (no phase-out)",
    )

    # --- the body-frame twins (2026-10-15): base env + BODY_DELTA, same task / experiment /
    # iterations. ``resume_from`` points at the base's own body twin when one exists and is
    # otherwise None (a body head never resumes a WORLD head: its obs cols 6-7 and command differ).
    for base in BODY_BASES:
        spec = p[base]
        twin = f"{spec.resume_from}{BODY_PHASE_SUFFIX}" if spec.resume_from else None
        p[f"{base}{BODY_PHASE_SUFFIX}"] = replace(
            spec,
            name=f"{base}{BODY_PHASE_SUFFIX}",
            env=_body_env(spec.env),
            resume_from=twin if twin in BODY_PHASE_NAMES else None,
            notes=f"{spec.notes} [body-frame command: KRABBY_OMNI_COMMAND_FRAME=body]",
        )
    return p


PHASES: dict[str, PhaseSpec] = build_phases()

PHASE_MODES = {name: spec.teacher_mode for name, spec in PHASES.items() if spec.teacher_mode}


def phase_env(name: str, plant: str | None = None) -> dict[str, str]:
    """Full environment an omni phase expands to, without touching ``os.environ``.

    Writes ``KRABBY_OMNI_PHASE`` and never ``KRABBY_PHASE`` / ``KRABBY_SIDE_PHASE``. Unknown
    names -- including every forward-only and sideways-only one -- raise ``KeyError``.
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
    """Expand ``KRABBY_OMNI_PHASE`` / ``KRABBY_PLANT`` into ``environ`` with setdefault.

    Returns the keys that were set (empty when no omni phase is requested). Explicit
    variables always win, exactly as the forward and sideways activations do.

    * ``KRABBY_OMNI_PHASE`` set together with ``KRABBY_PHASE`` **or** ``KRABBY_SIDE_PHASE``
      -> ``RuntimeError`` (one task per process: the expanded knob names are shared, and
      whichever activation ran first would have setdefault-ed them already). Both checks
      live here because the sideways guard predates this package and knows only about
      ``KRABBY_PHASE``, and the import order is forward -> omni -> sideways: this module is
      the one that sees an omni+sideways collision first.
    * ``KRABBY_OMNI_PHASE`` unset -> ``{}`` and ``environ`` untouched, whatever else is set.
      A strict no-op: every forward and sideways process imports this module through
      ``import_packages``, and their config dumps must not move by one byte.
    """
    env = os.environ if environ is None else environ
    name = env.get(PHASE_VAR, "").strip()
    if name:
        clash = [v for v in ("KRABBY_PHASE", "KRABBY_SIDE_PHASE") if env.get(v, "").strip()]
        if clash:
            raise RuntimeError(
                "one task per process: "
                f"{PHASE_VAR} and {' and '.join(clash)} are both set"
            )
    if not name:
        return {}
    plant = env.get("KRABBY_PLANT", "").strip() or None
    applied: dict[str, str] = {}
    for k, v in phase_env(name, plant).items():
        if k in (PHASE_VAR, "KRABBY_PLANT"):
            continue
        if k not in env:
            env[k] = v
            applied[k] = v
    return applied
