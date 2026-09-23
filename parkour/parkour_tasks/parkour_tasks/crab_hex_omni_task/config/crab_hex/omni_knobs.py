# SPDX-License-Identifier: BSD-3-Clause
"""``KRABBY_OMNI_*`` knobs -> the omni env cfg (plan D11/D12), plus the RSI re-point.

The omni env cfgs inherit the forward task's whole knob block (``apply_flat_walk_knobs``),
which builds the command band, the terrain mix, the DR events, the episode length, the
reward anneal and -- presence-based on ``KRABBY_RSI_FRAC`` -- the ``events.rsi_reference_reset``
event term. :func:`apply_omni_knobs` runs immediately AFTER it, in ``__post_init__``, and does
five things the forward block cannot know about:

1. **command knobs** -- the five per-phase sampler fields (:data:`SLOTS_VAR`,
   :data:`DIR_BAND_VAR`, :data:`SPEED_ELLIPSE_VAR`, :data:`FACING_VAR`,
   :data:`ANG_VEL_MAX_VAR`) onto ``commands.base_velocity``. They exist only on
   ``CrabHexOmniCommandCfg`` -- but ``setattr`` on a ``configclass`` instance does NOT fail by
   ``AttributeError`` for an unknown name (it just creates one), so a leaked forward command cfg
   would otherwise silently grow five orphan fields here rather than training the wrong regime.
   ``_apply_command_knobs`` therefore checks the type explicitly and raises by name. The other
   loud guards against a leaked cfg live elsewhere in this package: ``require_ring_geom`` (the
   terrain-geometry preset), ``_apply_spawn_yaw`` (the ``yaw_range`` param probe, below), and
   ``OmniParkourEvent.note_spawn`` / ``_cmd_term`` (the omni parkour event's own command-type
   checks);
2. **the gait-clock turn radius** (:data:`CLOCK_TURN_R_VAR`) onto ``omni_clock_turn_radius``,
   the env-cfg attribute the shared clock seams read through ``getattr``;
3. **the spawn-yaw band** (:data:`YAW_RANGE_VAR`) onto BOTH reset events -- the ring spawn and
   the RSI re-yaw. One knob, one band: see :func:`_apply_spawn_yaw`;
4. **the RSI re-point** -- ``events.rsi_reference_reset``'s ``func`` and ``params`` are
   replaced with the omni reset (re-yawed frames from the provenance-tagged bank, with the
   first command staged). The ATTRIBUTE NAME is kept, so the event keeps its position in the
   manager's ordering; ``fix_spawn`` (a corridor knob) is dropped and ``yaw_range`` /
   ``walk_floor`` / ``source`` are added. ``walk_floor`` is DERIVED from the command cfg's
   own clip rather than written as a literal: the two must agree or a "seeded" frame is told
   to stand still;
5. **the added reward's weight** (:data:`YAW_PROGRESS_W_VAR`) onto
   ``rewards.reward_yaw_progress``. Every OTHER reward weight knob is applied by the forward
   block's ``_overrides`` table, which keys on names that exist on the forward cfg;
   ``reward_yaw_progress`` is the omni task's one ADDED term (the 36th name), so its knob has
   to be applied here -- after the forward block, in the same one-place-per-dose discipline
   (:func:`_apply_reward_weight_knobs`);
6. **two reward-SHAPE knobs the forward task has no omni-safe spelling for** (Revision 4,
   2026-09-18 -- the reward-economy fix; see :data:`ANG_VEL_STD_VAR` /
   :data:`CLOCK_COMBINE_VAR` and ``README.md``'s "Revision 4"):
   ``KRABBY_OMNI_ANG_VEL_STD`` -> ``rewards.track_ang_vel_z_exp.params["std"]`` and
   ``KRABBY_OMNI_CLOCK_COMBINE`` -> ``params["combine"]`` on every clock reward term that
   HAS that key. Both are applied here rather than by the forward block because the omni
   task needs to move them per-window without touching a forward preset;
7. **one STAGED LEVER, DEFAULT OFF** (2026-09-18 user decision -- see :data:`ANG_TURN_ONLY_VAR`):
   ``KRABBY_OMNI_ANG_TURN_ONLY=1`` swaps ``rewards.track_ang_vel_z_exp``'s FUNC (not a param)
   for the turn-gated twin. NO PRESET SETS IT and the rewards class body keeps the stock func,
   so every shipped omni run is bit-identical unless a launcher arms it by hand;
8. ``apply_omni_ring_knobs`` -- the ring-generator guard and the ``KRABBY_RING_*`` knobs
   (``mdp/omni_terrain_knobs.py``);
9. **six CAPABILITY-CURRICULUM knobs, all DEFAULT-OFF and set by NO preset** (2026-09-18, the
   facing-and-tracking plan, Part A1). Unset or blank, every one of them is a bitwise no-op --
   that is what keeps the shipped 1a/1c windows reproducible while the campaign drives them as
   env overlays:

   ==================================  =======================================================
   variable                            what it writes
   ==================================  =======================================================
   ``KRABBY_OMNI_CMD_POOL``            ``commands.base_velocity.cmd_pool`` (the pool STRING,
                                       verbatim; blank => ``None``, like ``DIR_BAND``)
   ``KRABBY_OMNI_ANG_VEL_CLIP``        ``commands.base_velocity.clips.ang_vel_clip``
   ``KRABBY_OMNI_FACING_TURN_W``       ``rewards.reward_facing_turn.weight``
   ``KRABBY_OMNI_TRACK_FACING_SIGMA``  ``rewards.track_lin_vel_xy_exp``'s FUNC (the
                                       facing-coupled twin) + ``params["facing_sigma"]``
   ``KRABBY_OMNI_PROGRESS_W``          ``rewards.reward_forward_progress_along_command.weight``
   ``KRABBY_OMNI_FACING_OBS_ENCODING`` the omni obs term's ``params["facing_obs_encoding"]``
   ==================================  =======================================================

10. **the clock-radius rule** (:func:`_check_clock_radius_rule`) -- a cross-knob consistency
    check, not a dose: ``omni_clock_turn_radius x ang_vel_max`` is the tangential speed a
    pure turn feeds the gait clock, and below :data:`CLOCK_TURN_MIN_SPEED` the clock STOPS, so
    a turn stage would train a statue while both knobs looked individually reasonable.

11. **the MOTION-QUALITY knobs** (gait-stabilisation plan step 2, 2026-09-22; all DEFAULT OFF,
    set by no preset) -- ``motion_knobs.py``, path-loaded below and re-exported here (their
    ``*_VAR`` names and parsers) so the campaign ladder's whitelist / parser table, which reads
    this module, sees them. :func:`apply_omni_knobs` applies the env-side ones LAST
    (``motion_knobs.apply_motion_knobs``); the runner-side three (std clamp / std reset /
    entropy) are applied by the omni runner cfgs (``agents/rsl_rl_ppo_cfg.py``).

Every parser is pure and lives at module scope with no Isaac import (the Isaac-touching bits
are lazy imports inside :func:`apply_omni_knobs`), so this module can be path-loaded and
unit-tested without booting Kit -- see ``tests/unit/test_crab_hex_omni_phases.py``.

Format conventions follow the rest of the Krabby knobs: ``lo:hi`` for a band, ``:``-separated
for a fixed-length tuple, a bare number for a scalar. Every parser rejects the wrong arity
rather than padding or truncating: a typo in a launcher must not silently retune the sampler.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

_MOTION_KNOBS_PY = Path(__file__).resolve().parent / "motion_knobs.py"
_MOTION_KNOBS_NAME = "crab_hex_omni_motion_knobs"


def _motion_knobs():
    """``motion_knobs.py`` (pure), by path -- the same object for every loader of this module."""
    mod = sys.modules.get(_MOTION_KNOBS_NAME)
    if mod is None:
        spec = importlib.util.spec_from_file_location(_MOTION_KNOBS_NAME, _MOTION_KNOBS_PY)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[_MOTION_KNOBS_NAME] = mod
        spec.loader.exec_module(mod)
    return mod


_mk = _motion_knobs()
# Re-exported so ``dir(omni_knobs)`` -- the ladder's knob whitelist -- names every motion knob.
ACTION_STD_MAX_VAR = _mk.ACTION_STD_MAX_VAR
ACTION_STD_RESET_VAR = _mk.ACTION_STD_RESET_VAR
ENTROPY_COEF_VAR = _mk.ENTROPY_COEF_VAR
ACTION_RATE_SQ_VAR = _mk.ACTION_RATE_SQ_VAR
ACTION_SMOOTH2_W_VAR = _mk.ACTION_SMOOTH2_W_VAR
JOINT_VEL_W_VAR = _mk.JOINT_VEL_W_VAR
FOOT_SLIP_W_VAR = _mk.FOOT_SLIP_W_VAR
STAND_POSE_W_VAR = _mk.STAND_POSE_W_VAR
ACTION_LPF_HZ_VAR = _mk.ACTION_LPF_HZ_VAR
ACTION_LPF_OBS_VAR = _mk.ACTION_LPF_OBS_VAR
RELAY_HYST_VAR = _mk.RELAY_HYST_VAR
STAND_HOLD_VAR = _mk.STAND_HOLD_VAR
STAND_PARK_VAR = _mk.STAND_PARK_VAR
STAND_OBS_VAR = _mk.STAND_OBS_VAR
POWER_W_VAR = _mk.POWER_W_VAR
ACTION_RATE_W_VAR = _mk.ACTION_RATE_W_VAR
AIRTIME_W_VAR = _mk.AIRTIME_W_VAR
MOTION_KNOB_PARSERS = _mk.MOTION_KNOB_PARSERS

__all__ = [
    "SLOTS_VAR",
    "DIR_BAND_VAR",
    "SPEED_ELLIPSE_VAR",
    "FACING_VAR",
    "ANG_VEL_MAX_VAR",
    "ANG_VEL_CLIP_VAR",
    "ANG_VEL_STD_VAR",
    "ANG_TURN_ONLY_VAR",
    "CMD_POOL_VAR",
    "CLOCK_TURN_R_VAR",
    "CLOCK_COMBINE_VAR",
    "CLOCK_TURN_MIN_SPEED",
    "YAW_RANGE_VAR",
    "RSI_SOURCE_VAR",
    "RSI_BANK_VAR",
    "RSI_FRAC_VAR",
    "YAW_PROGRESS_W_VAR",
    "FACING_TURN_W_VAR",
    "TRACK_FACING_SIGMA_VAR",
    "PROGRESS_W_VAR",
    "FACING_OBS_ENCODING_VAR",
    "FACING_OBS_ENCODINGS",
    "parse_slots",
    "parse_dir_band",
    "parse_speed_ellipse",
    "parse_facing",
    "parse_ang_vel_max",
    "parse_ang_vel_clip",
    "parse_ang_vel_std",
    "parse_ang_turn_only",
    "parse_cmd_pool",
    "parse_clock_turn_radius",
    "parse_clock_combine",
    "parse_yaw_range",
    "parse_rsi_source",
    "parse_facing_turn_w",
    "parse_track_facing_sigma",
    "parse_progress_w",
    "parse_facing_obs_encoding",
    "apply_omni_knobs",
    # motion-quality knobs (motion_knobs.py)
    "ACTION_STD_MAX_VAR", "ACTION_STD_RESET_VAR", "ENTROPY_COEF_VAR", "ACTION_RATE_SQ_VAR",
    "ACTION_SMOOTH2_W_VAR", "JOINT_VEL_W_VAR", "FOOT_SLIP_W_VAR", "STAND_POSE_W_VAR",
    "ACTION_LPF_HZ_VAR", "ACTION_LPF_OBS_VAR", "RELAY_HYST_VAR", "STAND_HOLD_VAR",
    "STAND_PARK_VAR", "STAND_OBS_VAR", "POWER_W_VAR", "ACTION_RATE_W_VAR", "AIRTIME_W_VAR",
    "MOTION_KNOB_PARSERS",
]

SLOTS_VAR = "KRABBY_OMNI_SLOTS"
DIR_BAND_VAR = "KRABBY_OMNI_DIR_BAND"
SPEED_ELLIPSE_VAR = "KRABBY_OMNI_SPEED_ELLIPSE"
FACING_VAR = "KRABBY_OMNI_FACING"
ANG_VEL_MAX_VAR = "KRABBY_OMNI_ANG_VEL_MAX"
#: ``track_ang_vel_z_exp``'s kernel width (Revision 4, 2026-09-18). The omni rewards CLASS BODY
#: declares ``std = OMNI_ANG_VEL_STD`` (0.15, plan D4: gradient at the plant's 0.15-0.3 rad/s turn
#: rate); this knob moves it per window. The 1c diagnosis measured the cost of the tight kernel on
#: the 88 % of steps that command NO turn: ordinary gait yaw wobble of 0.10-0.15 rad/s is taxed
#: 0.36-0.63/s at std 0.15 -- more than the entire translation income a walking policy can earn --
#: so the cheapest policy is to hold still. There is no forward-named knob for this param, hence
#: the omni spelling.
ANG_VEL_STD_VAR = "KRABBY_OMNI_ANG_VEL_STD"
#: **A STAGED LEVER, DEFAULT OFF** (2026-09-18 user decision). ``"1"`` swaps
#: ``rewards.track_ang_vel_z_exp``'s FUNC to
#: ``crab_hex_omni_rewards.track_ang_vel_z_exp_turn_gated`` -- the stock term multiplied by
#: ``(cmd[:, 2] != 0)`` -- keeping the weight and every param (``std`` included). ``"0"`` is the
#: explicit off; unset/empty is off; anything else raises at config build.
#:
#: WHY IT IS STAGED. ``track_ang_vel_z_exp`` is an even function of the yaw-rate ERROR, so a
#: motionless robot with zero yaw error earns ~0.98/s on the ~88 % of 1c rows that command no
#: turn -- ~45 % of a stander's whole positive income under the Revision-4 economy. Gating it
#: moves the offline walker-vs-creeper COMMAND margin on the decidable ``fwd`` hold from
#: +16.6 % to +37.5 % (the campaign's >= 25 % gate: FAIL -> PASS). WHY IT IS OFF: the gate also
#: removes the term's yaw-rate DAMPING on non-turn rows, which is a behavioural change rather
#: than a bookkeeping fix, so it is armed by hand for one run, never by a preset. Evidence:
#: ``experiments/2026-09-16_1805_omni_formation/replay_reward_economy.md`` (the PROBE table;
#: restate it with ``replay_reward_economy.py --new-ang-turn-only``).
ANG_TURN_ONLY_VAR = "KRABBY_OMNI_ANG_TURN_ONLY"
#: The yaw-rate DEAD BAND, ``commands.base_velocity.clips.ang_vel_clip`` (2026-09-18, Part A1).
#: The class body's 0.2 rad/s is already the tightened omni value (the forward task's is 0.4), but
#: at stiffness 1.5 it still swallows every facing error below 0.133 rad: a policy that turns most
#: of the way and stops is paid as if it had arrived. A curriculum stage that wants a finer facing
#: demand moves it here rather than by retuning the stiffness, which would also move the P-control
#: gain. Must stay BELOW ``ang_vel_max`` -- the command cfg's own validator refuses the inversion,
#: which is why this knob is applied inside :func:`_apply_command_knobs` alongside it.
ANG_VEL_CLIP_VAR = "KRABBY_OMNI_ANG_VEL_CLIP"
#: The per-type COMMAND POOL (2026-09-18, Part A1): the canonical string
#: ``name:weight[:ang_scale[:speed_scale]],...`` over the type names in
#: ``crab_hex_omni_command_math.POOL_TYPES``. Parsed HERE only to fail a typo at config build;
#: the string itself is what reaches ``cmd_pool``, verbatim, because the config dump and the
#: campaign's ``params/env.yaml`` audit have to show exactly what the launcher asked for.
#:
#: Blank (``""``) means "no pool" -- the shipped four-slot sampler -- exactly like
#: :data:`DIR_BAND_VAR`, and for the same reason: the campaign's overlays must be able to SWITCH
#: a preset's lever off, which an absent key cannot do (``activate_phase`` is ``setdefault``).
#: A pool and a ``dir_band`` are mutually exclusive and the command cfg refuses both at once.
CMD_POOL_VAR = "KRABBY_OMNI_CMD_POOL"
CLOCK_TURN_R_VAR = "KRABBY_OMNI_CLOCK_TURN_R"
#: The gait-clock schedule term's ``combine`` mode (Revision 4, 2026-09-18), omni-only.
#:
#: The MECHANISM already exists in the forward task and is NOT re-implemented here:
#: ``crab_hex_clock_reward.clock_schedule_income(..., combine=...)`` has taken ``"sum"``
#: (the literal default) / ``"product"`` since 2026-08-22, the forward rewards cfg declares
#: ``params["combine"] = "sum"`` on ``reward_clock_schedule``, and ``KRABBY_CLOCK_COMBINE``
#: applies it in the forward ``__post_init__``. This knob exists because the omni presets must
#: set the mode in every omni window WITHOUT writing a forward-owned variable whose meaning is
#: pinned by the forward lineage's own records -- and because it is applied AFTER the forward
#: block, so ``KRABBY_OMNI_CLOCK_COMBINE`` wins over ``KRABBY_CLOCK_COMBINE`` if both are set
#: (documented precedence, one place per dose).
CLOCK_COMBINE_VAR = "KRABBY_OMNI_CLOCK_COMBINE"
YAW_RANGE_VAR = "KRABBY_OMNI_YAW_RANGE"
RSI_SOURCE_VAR = "KRABBY_OMNI_RSI_SOURCE"
#: Shared with the forward task (the omni presets re-point the bank and keep the fraction).
RSI_BANK_VAR = "KRABBY_RSI_BANK"
RSI_FRAC_VAR = "KRABBY_RSI_FRAC"
#: The weight of the omni-only ``reward_yaw_progress`` term. Named like the other weight
#: knobs (``KRABBY_YAW_W``, ``KRABBY_GOAL_VEL_W``) and not ``KRABBY_OMNI_*``, because it is a
#: reward dose, not a sampler field -- but it reaches its term HERE, not through the forward
#: ``_overrides`` table, since no forward cfg has that term. Inert in a forward/sideways
#: process (nothing reads it there).
YAW_PROGRESS_W_VAR = "KRABBY_YAW_PROGRESS_W"
#: Weight of ``rewards.reward_facing_turn`` -- the facing income paid ONLY on rows that command a
#: turn (the term ships at weight 0.0, so this knob is the whole dose). Named like the other
#: weight knobs and applied here, next to ``KRABBY_YAW_PROGRESS_W``, for the same reason: no
#: forward cfg has the term, so the forward ``_overrides`` table cannot reach it.
FACING_TURN_W_VAR = "KRABBY_OMNI_FACING_TURN_W"
#: Couples the TRANSLATION tracking kernel to the facing error (2026-09-18, Part A1):
#: ``rewards.track_lin_vel_xy_exp``'s func becomes
#: ``crab_hex_omni_rewards.track_lin_vel_xy_exp_facing_coupled`` -- the stock kernel times a
#: Gaussian of the facing error at this width -- and ``params["facing_sigma"]`` is this value.
#:
#: WHY A FUNC SWAP AND NOT A WEIGHT. The diagnosis is that the head walks the commanded line
#: while ignoring the commanded facing: the two demands are priced by two INDEPENDENT terms, so
#: the cheapest policy is to buy the larger one and drop the other. Multiplying them makes
#: translation income conditional on facing, which no re-weighting of two separate terms can do.
#: Default OFF (the stock, uncoupled func), like every lever whose effect is behavioural.
TRACK_FACING_SIGMA_VAR = "KRABBY_OMNI_TRACK_FACING_SIGMA"
#: Weight of ``rewards.reward_forward_progress_along_command`` -- the term that pays for actually
#: TRANSLATING along the command (0.6 in the forward class body, and reachable from the forward
#: block only by a name the omni presets must not write). The campaign moves it to trade
#: translation income against the facing income the two knobs above introduce.
PROGRESS_W_VAR = "KRABBY_OMNI_PROGRESS_W"
#: How the facing error reaches the policy's observation (2026-09-18, Part A1): ``"raw"`` (the
#: shipped default -- one clipped signed angle) or ``"sincos"`` (its sine and cosine, which is
#: continuous across ``+-pi`` and cannot be read as "nearly zero" when it is nearly a half turn).
#: The obs term must ALREADY carry the key: writing a param the observation function does not
#: accept would raise inside the observation manager on the first step of training.
FACING_OBS_ENCODING_VAR = "KRABBY_OMNI_FACING_OBS_ENCODING"
FACING_OBS_ENCODINGS = ("raw", "sincos")
"""The only two spellings :func:`parse_facing_obs_encoding` accepts."""

CLOCK_TURN_MIN_SPEED = 0.25
"""Smallest ``omni_clock_turn_radius x ang_vel_max`` a config may ship, m/s.

The gait clock is driven by a commanded SPEED and freezes below ``CLOCK_CMD_STOP_M_S`` (the
0.2 m/s lin_vel_clip). A pure turn has no translation, so the clock is fed the tangential speed
of a point at ``omni_clock_turn_radius`` -- ``radius x ang_vel_max``. Under this floor the
clock stops on every turn row, the gait rewards pay a statue, and NEITHER knob looks wrong on
its own: 1.2 m and 0.3 rad/s are both ordinary numbers, and so are 1.2 and 0.1. Today's presets
give 1.2 x 0.3 = 0.36, comfortably clear. The check is a cross-knob one, hence a rule in
:func:`apply_omni_knobs` rather than a bound inside either parser."""


# ------------------------------------------------------------------------------ parsers
def _floats(var: str, text: str, n: int) -> tuple[float, ...]:
    """``a:b[:c...]`` -> exactly ``n`` finite floats, or ``ValueError`` naming the knob."""
    parts = str(text).strip().split(":")
    if len(parts) != n:
        raise ValueError(f"{var} must be {n} ':'-separated numbers, got {text!r}")
    out = []
    for part in parts:
        try:
            value = float(part.strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{var} entries must be numbers, got {text!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"{var} entries must be finite, got {text!r}")
        out.append(value)
    return tuple(out)


def parse_slots(text: str) -> tuple[float, float, float, float]:
    """``KRABBY_OMNI_SLOTS`` -> ``(stand, move, turn, move_turn)`` categorical weights.

    Not normalised here -- the sampler normalises, and keeping the raw numbers makes the
    config dump say what the launcher asked for. Non-negative and not all zero.
    """
    probs = _floats(SLOTS_VAR, text, 4)
    if any(p < 0.0 for p in probs):
        raise ValueError(f"{SLOTS_VAR} weights must be non-negative, got {text!r}")
    if sum(probs) <= 0.0:
        raise ValueError(f"{SLOTS_VAR} weights must not all be zero, got {text!r}")
    return probs  # type: ignore[return-value]


def parse_dir_band(text: str) -> tuple[float, float] | None:
    """``KRABBY_OMNI_DIR_BAND`` -> ``(lo, hi)`` radians relative to the body, or ``None``.

    Empty / ``"none"`` / ``"off"`` means "no band": the travel direction is drawn
    world-uniform over ``travel_dir_range`` (stage 1c onward). ``"0:0"`` is body ``+x``
    (stage 1a) and ``"1.5708:1.5708"`` is body ``+y`` (stage 1b) -- both are bands of zero
    width, not "unset", which is why the empty spelling has to be distinguishable.
    """
    token = str(text).strip().lower()
    if token in ("", "none", "off", "null"):
        return None
    lo, hi = _floats(DIR_BAND_VAR, token, 2)
    if not lo <= hi:
        raise ValueError(f"{DIR_BAND_VAR} must satisfy lo <= hi, got {text!r}")
    return (lo, hi)


def parse_speed_ellipse(text: str) -> tuple[float, float, float]:
    """``KRABBY_OMNI_SPEED_ELLIPSE`` -> ``(a, b, c)`` m/s: fore-aft, lateral, backward.

    Strictly positive. The command cfg's own validator additionally refuses an ellipse that
    does not clear the walk floor -- that check needs the clip and so lives there.
    """
    axes = _floats(SPEED_ELLIPSE_VAR, text, 3)
    if any(v <= 0.0 for v in axes):
        raise ValueError(f"{SPEED_ELLIPSE_VAR} semi-axes must be positive, got {text!r}")
    return axes  # type: ignore[return-value]


def parse_facing(text: str) -> tuple[float, float]:
    """``KRABBY_OMNI_FACING`` -> ``(lo, hi)`` band on |facing change|, radians. ``0 <= lo <= hi``."""
    lo, hi = _floats(FACING_VAR, text, 2)
    if not (0.0 <= lo <= hi):
        raise ValueError(f"{FACING_VAR} must satisfy 0 <= lo <= hi, got {text!r}")
    return (lo, hi)


def parse_ang_vel_max(text: str) -> float:
    """``KRABBY_OMNI_ANG_VEL_MAX`` -> the yaw-rate command saturation, rad/s. Positive."""
    (value,) = _floats(ANG_VEL_MAX_VAR, text, 1)
    if value <= 0.0:
        raise ValueError(f"{ANG_VEL_MAX_VAR} must be positive, got {text!r}")
    return value


def parse_ang_vel_clip(text: str) -> float:
    """``KRABBY_OMNI_ANG_VEL_CLIP`` -> the yaw-rate dead band, rad/s. Strictly positive.

    Zero is refused rather than read as "no dead band": the stock expression is a strict
    ``|wz| > clip``, so 0 would still zero an exactly-zero command and otherwise pass
    everything -- an intent better spelled by a tiny positive number, and a value a launcher
    reaches for by accident when it means "off". The upper bound is a CROSS-knob one
    (``ang_vel_max`` must exceed it) and lives in the command cfg's own validator, which sees
    both.
    """
    (value,) = _floats(ANG_VEL_CLIP_VAR, text, 1)
    if value <= 0.0:
        raise ValueError(f"{ANG_VEL_CLIP_VAR} must be positive, got {text!r}")
    return value


_CMD_MATH_PY = Path(__file__).resolve().parents[2] / "mdp" / "crab_hex_omni_command_math.py"
_CMD_MATH_NAME = "_crab_hex_omni_command_math__knobs"
_CMD_MATH = None


def _command_math():
    """The pure command-math module (``POOL_TYPES``, ``parse_cmd_pool``), Isaac-free.

    Loaded LAZILY and, in a bare interpreter, BY PATH: this module is pinned pure
    (``test_knobs_module_is_pure``) and ``import parkour_tasks...`` reaches Isaac through the
    package ``__init__``s, so the pool grammar could not otherwise be validated -- or unit
    tested -- without booting Kit. Inside a real Isaac process the command cfg has already
    imported the real module, so the ``sys.modules`` hit below returns exactly the object the
    command term will use. The path copy is only ever a VALIDATOR: what reaches ``cmd_pool``
    is the launcher's string, re-parsed by the term against its own module, so two class
    identities can never disagree about anything that trains.
    """
    global _CMD_MATH
    if _CMD_MATH is None:
        real = sys.modules.get("parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_command_math")
        if real is not None:
            _CMD_MATH = real
        else:
            spec = importlib.util.spec_from_file_location(_CMD_MATH_NAME, _CMD_MATH_PY)
            module = importlib.util.module_from_spec(spec)
            sys.modules[_CMD_MATH_NAME] = module
            spec.loader.exec_module(module)
            _CMD_MATH = module
    return _CMD_MATH


def parse_cmd_pool(text: str):
    """``KRABBY_OMNI_CMD_POOL`` -> a ``CmdPool``, or ``None`` for "no pool".

    Empty / ``"none"`` / ``"off"`` / ``"null"`` mean the shipped four-slot sampler, exactly as
    they do for :func:`parse_dir_band` and for exactly the same reason: an env OVERLAY has to be
    able to switch a preset's lever OFF, and an absent key cannot (``activate_phase`` is
    ``setdefault``). Every other string is handed to
    ``crab_hex_omni_command_math.parse_cmd_pool``, which owns the grammar, the type names and
    every rejection -- this module validates, it does not re-spell.

    The RETURN VALUE is only used to fail loudly here; what is written onto the cfg is the
    launcher's own string (see :func:`_apply_command_knobs`).
    """
    raw = str(text).strip()
    if raw.lower() in ("", "none", "off", "null"):
        return None
    try:
        return _command_math().parse_cmd_pool(raw)
    except ValueError as exc:
        raise ValueError(f"{CMD_POOL_VAR}: {exc}") from exc


def parse_facing_turn_w(text: str) -> float:
    """``KRABBY_OMNI_FACING_TURN_W`` -> ``reward_facing_turn``'s weight. Non-negative.

    A REWARD, not a penalty: a negative weight would pay the policy for facing the wrong way,
    which is not an ablation anybody wants and is invisible in the log except as one curve with
    the wrong sign. ``0.0`` is the legitimate ablation and is accepted.
    """
    (value,) = _floats(FACING_TURN_W_VAR, text, 1)
    if value < 0.0:
        raise ValueError(f"{FACING_TURN_W_VAR} must be non-negative, got {text!r}")
    return value


def parse_track_facing_sigma(text: str) -> float:
    """``KRABBY_OMNI_TRACK_FACING_SIGMA`` -> the facing-coupling kernel width, rad. Positive.

    Strictly positive for the same reason as :func:`parse_ang_vel_std`: the coupling is
    ``exp(-err^2 / sigma^2)``, so 0 is a division by zero and a negative width is the same
    kernel as its absolute value. "Do not couple" is spelled by leaving the knob UNSET, which
    keeps the stock func -- not by a degenerate sigma.
    """
    (value,) = _floats(TRACK_FACING_SIGMA_VAR, text, 1)
    if value <= 0.0:
        raise ValueError(f"{TRACK_FACING_SIGMA_VAR} must be positive, got {text!r}")
    return value


def parse_progress_w(text: str) -> float:
    """``KRABBY_OMNI_PROGRESS_W`` -> ``reward_forward_progress_along_command``'s weight.

    Non-negative: the term pays for clamped progress ALONG the command, so a negative weight
    would pay for standing still. ``0.0`` is the legitimate ablation.
    """
    (value,) = _floats(PROGRESS_W_VAR, text, 1)
    if value < 0.0:
        raise ValueError(f"{PROGRESS_W_VAR} must be non-negative, got {text!r}")
    return value


def parse_facing_obs_encoding(text: str) -> str:
    """``KRABBY_OMNI_FACING_OBS_ENCODING`` -> ``"raw"`` or ``"sincos"``.

    Validated here, at config build, rather than inside the observation function on the first
    step: an unknown encoding that fell through to a default would change the OBSERVATION the
    policy is trained on while the launcher believed the other one was armed, and the two are
    indistinguishable in a training log.
    """
    token = str(text).strip().lower()
    if token not in FACING_OBS_ENCODINGS:
        raise ValueError(
            f"{FACING_OBS_ENCODING_VAR} must be one of {FACING_OBS_ENCODINGS}, got {text!r}"
        )
    return token


def parse_ang_vel_std(text: str) -> float:
    """``KRABBY_OMNI_ANG_VEL_STD`` -> ``track_ang_vel_z_exp``'s kernel std, rad/s. Positive.

    Strictly positive: Isaac Lab's term is ``exp(-(wz_cmd - wz)^2 / std^2)``, so a zero width is a
    division by zero and a negative one is the same kernel as its absolute value -- "disable" is
    what the term's WEIGHT knob is for, not a degenerate std.
    """
    (value,) = _floats(ANG_VEL_STD_VAR, text, 1)
    if value <= 0.0:
        raise ValueError(f"{ANG_VEL_STD_VAR} must be positive, got {text!r}")
    return value


def parse_ang_turn_only(text: str) -> bool:
    """``KRABBY_OMNI_ANG_TURN_ONLY`` -> ``True`` (``"1"``) or ``False`` (``"0"``).

    Deliberately NOT a permissive truthiness parser. Every other spelling a launcher might
    reach for -- ``"true"``, ``"yes"``, ``"on"``, ``"2"``, ``"off"`` -- raises by name, because
    this knob changes which FUNCTION prices the yaw rate: a value silently read as false would
    train the shipped economy while the launcher claimed the lever, and the only difference in
    the log is the height of one ``Episode_Reward`` curve that is supposed to be lower.
    """
    token = str(text).strip()
    if token == "1":
        return True
    if token == "0":
        return False
    raise ValueError(f"{ANG_TURN_ONLY_VAR} must be '0' or '1', got {text!r}")


def parse_clock_combine(text: str) -> str:
    """``KRABBY_OMNI_CLOCK_COMBINE`` -> ``"sum"`` or ``"product"``.

    Validated here rather than left to the reward function, which raises ``ValueError`` from
    inside the reward manager on the FIRST STEP of training: a typo must fail at config build,
    with the knob's name in the message, not thousands of GPU-seconds later.
    """
    token = str(text).strip().lower()
    if token not in ("sum", "product"):
        raise ValueError(f"{CLOCK_COMBINE_VAR} must be 'sum' or 'product', got {text!r}")
    return token


def parse_clock_turn_radius(text: str) -> float:
    """``KRABBY_OMNI_CLOCK_TURN_R`` -> the turn radius the gait clock prices, metres.

    Non-negative; 0 means "turning does not drive the clock" (the forward behaviour). NOTE that
    :func:`_check_clock_radius_rule` then refuses the built cfg -- the PARSER's job is the value's
    own shape, the cross-knob product ``radius x ang_vel_max`` is checked once, where both
    knobs are final.
    """
    (value,) = _floats(CLOCK_TURN_R_VAR, text, 1)
    if value < 0.0:
        raise ValueError(f"{CLOCK_TURN_R_VAR} must be non-negative, got {text!r}")
    return value


def parse_yaw_range(text: str) -> tuple[float, float]:
    """``KRABBY_OMNI_YAW_RANGE`` -> the ``(lo, hi)`` spawn-yaw band, radians. ``lo <= hi``.

    A degenerate band (``a:a``) pins the spawn yaw -- what the eval harness does to make a
    world-frame hold reproducible -- so equal bounds are legal.
    """
    lo, hi = _floats(YAW_RANGE_VAR, text, 2)
    if not lo <= hi:
        raise ValueError(f"{YAW_RANGE_VAR} must satisfy lo <= hi, got {text!r}")
    return (lo, hi)


def parse_rsi_source(text: str):
    """``KRABBY_OMNI_RSI_SOURCE`` -> ``0`` (forward frames), ``1`` (sideways) or ``"all"``.

    Validated here, at config build, rather than at the first reset: a typo would otherwise
    surface thousands of steps into a run, as an exception inside an event.
    """
    token = str(text).strip().lower()
    if token in ("all", "any", "*"):
        return "all"
    if token in ("0", "1"):
        return int(token)
    raise ValueError(f"{RSI_SOURCE_VAR} must be 0, 1 or 'all', got {text!r}")


# ------------------------------------------------------------------------------ application
def _apply_command_knobs(cmd_cfg, env, *, cmd_cls=None) -> None:
    """The SEVEN sampler knobs onto the omni command cfg (unset knob = cfg default kept).

    The four-row table below (``KRABBY_OMNI_SLOTS`` / ``_SPEED_ELLIPSE`` / ``_FACING`` /
    ``_ANG_VEL_MAX``) plus the three that cannot join it: ``KRABBY_OMNI_ANG_VEL_CLIP`` (it
    lives one level down, on ``clips``) and ``KRABBY_OMNI_DIR_BAND`` / ``KRABBY_OMNI_CMD_POOL``
    (an EMPTY value is a meaningful "off" for both, so they are applied when PRESENT rather
    than when non-empty).

    ``isinstance``-checked rather than left to ``setattr`` to fail on its own: a
    ``configclass`` instance happily accepts an unknown attribute (it does not raise
    ``AttributeError``), so a leaked forward/sideways command cfg would otherwise silently grow
    seven orphan fields here instead of failing loudly.

    Args:
        cmd_cls: the class to check against; defaults to the real ``CrabHexOmniCommandCfg``,
            imported lazily (by path, like :func:`apply_omni_knobs`'s own ``omni_terrain_knobs``
            import below) so this module stays Isaac-free at module scope -- see
            ``tests/unit/test_crab_hex_omni_phases.py::test_knobs_module_is_pure``, which only
            calls :func:`parse_slots` and would fail if loading this file alone pulled in
            ``isaaclab``. A test may pass a stand-in class instead (the
            ``omni_terrain_knobs.apply_omni_ring_knobs(..., classes=...)` precedent).
    """
    if cmd_cls is None:
        from parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_command import (  # noqa: PLC0415
            CrabHexOmniCommandCfg,
        )

        cmd_cls = CrabHexOmniCommandCfg

    if not isinstance(cmd_cfg, cmd_cls):
        raise RuntimeError(
            "the omni command knobs require commands.base_velocity to be a "
            f"CrabHexOmniCommandCfg; found {type(cmd_cfg).__name__} (a forward/sideways command "
            "cfg leaked into an omni env)."
        )
    for var, parser, attr in (
        (SLOTS_VAR, parse_slots, "slot_probs"),
        (SPEED_ELLIPSE_VAR, parse_speed_ellipse, "speed_ellipse"),
        (FACING_VAR, parse_facing, "facing_delta"),
        (ANG_VEL_MAX_VAR, parse_ang_vel_max, "ang_vel_max"),
    ):
        raw = env.get(var)
        if raw is not None and str(raw).strip():
            setattr(cmd_cfg, attr, parser(raw))
    # The dead band lives one level down, on ``clips`` (shared with the forward cfg's shape),
    # so it cannot join the table above. Written BEFORE the cfg is ever validated, because the
    # command term refuses ``ang_vel_max <= ang_vel_clip`` and has to see both final values.
    raw_clip = env.get(ANG_VEL_CLIP_VAR)
    if raw_clip is not None and str(raw_clip).strip():
        cmd_cfg.clips.ang_vel_clip = parse_ang_vel_clip(raw_clip)
    # dir_band is special: an EMPTY value is a meaningful "no band" (stage 1c+), so the knob
    # is applied whenever it is present, not only when it is non-empty.
    raw_band = env.get(DIR_BAND_VAR)
    if raw_band is not None:
        cmd_cfg.dir_band = parse_dir_band(raw_band)
    # ... and the pool is special the same way, for the same reason (a curriculum overlay must
    # be able to turn a pool OFF). The parse is a VALIDATOR: what is stored is the launcher's
    # own string, so the config dump and the campaign's env.yaml audit show it verbatim.
    raw_pool = env.get(CMD_POOL_VAR)
    if raw_pool is not None:
        cmd_cfg.cmd_pool = None if parse_cmd_pool(raw_pool) is None else str(raw_pool).strip()


def _apply_spawn_yaw(cfg, yaw_range) -> None:
    """``KRABBY_OMNI_YAW_RANGE`` onto the ring spawn event (no-op when the knob is unset).

    ONE knob drives BOTH places a reset can set the robot's yaw -- the ring spawn here and the
    RSI re-yaw below. They have to agree: pinning the spawn yaw for a world-frame eval hold
    (``yaw_range = (face0, face0)``) while the RSI event still drew ``U(-pi, pi)`` would leave a
    fraction of the episodes -- exactly ``KRABBY_RSI_FRAC`` of them -- starting at a random
    facing, which reads as a policy that cannot hold a heading.
    """
    if yaw_range is None:
        return
    term = getattr(getattr(cfg, "events", None), "reset_root_state", None)
    params = getattr(term, "params", None) if term is not None else None
    if not isinstance(params, dict) or "yaw_range" not in params:
        raise RuntimeError(
            "the omni env's events.reset_root_state does not take a 'yaw_range' param; it is "
            "not reset_root_state_ring (a forward reset event leaked into an omni cfg), so "
            f"{YAW_RANGE_VAR} would be silently ignored."
        )
    params["yaw_range"] = yaw_range


FACING_OBS_CLIP_VAR = "KRABBY_OMNI_FACING_OBS_CLIP"


def _omni_obs_terms(cfg):
    """Every observation term whose func is the omni observation class (by attribute, not import)."""
    found = []
    groups = getattr(cfg, "observations", None)
    for gname in ("policy", "critic"):
        group = getattr(groups, gname, None)
        if group is None:
            continue
        for tname, term in vars(group).items():
            func = getattr(term, "func", None)
            if hasattr(func, "OBSERVED_COMMAND_SLOTS"):
                found.append((gname, tname, term))
    return found


def _apply_facing_obs_clip_knob(cfg, env) -> None:
    """``KRABBY_OMNI_FACING_OBS_CLIP=<rad>`` -> ``params["facing_obs_clip"]`` on the omni obs term.

    Absent -> the raw facing error (the class default). Present -> a symmetric clip so an
    out-of-band facing target reads as the band's edge (see the observation class docstring).
    Raises if no omni observation term is installed (a leaked forward obs cfg would silently
    ignore the knob).
    """
    raw = env.get(FACING_OBS_CLIP_VAR)
    if raw is None or str(raw).strip() == "":
        return
    clip = float(raw)
    if not clip > 0.0:
        raise RuntimeError(f"{FACING_OBS_CLIP_VAR} must be > 0 rad, got {raw!r}")
    terms = _omni_obs_terms(cfg)
    if not terms:
        raise RuntimeError(f"{FACING_OBS_CLIP_VAR} is set but no omni observation term is installed")
    for _g, _t, term in terms:
        if term.params is None:
            term.params = {}
        term.params["facing_obs_clip"] = clip


#: ``(variable, rewards attribute, parser)`` for every reward WEIGHT this module doses.
#:
#: All three name terms the forward ``_overrides`` table cannot reach: ``reward_yaw_progress``
#: and ``reward_facing_turn`` do not exist on a forward cfg at all, and while
#: ``reward_forward_progress_along_command`` does, the omni presets must not write a
#: forward-owned variable whose meaning is pinned by the forward lineage's own records (the
#: ``KRABBY_OMNI_CLOCK_COMBINE`` precedent).
_REWARD_WEIGHT_KNOBS = (
    (YAW_PROGRESS_W_VAR, "reward_yaw_progress", float),
    (FACING_TURN_W_VAR, "reward_facing_turn", parse_facing_turn_w),
    (PROGRESS_W_VAR, "reward_forward_progress_along_command", parse_progress_w),
)


def _apply_reward_weight_knobs(cfg, env) -> None:
    """The :data:`_REWARD_WEIGHT_KNOBS` doses onto their terms (no-op for every unset knob).

    PRESENCE-based, exactly like the forward block's ``_overrides``: an explicitly-set knob
    wins even at ``0.0``, which is how an ablation switches a term off. Only an empty
    value is ignored (``float("")`` would raise).

    The ``hasattr`` check is the same silent-failure insurance as
    :func:`_apply_command_knobs`'s type check: a ``configclass`` instance accepts an unknown
    attribute without complaint, so a forward/sideways rewards cfg leaking into an omni env
    (or a future refactor that stops wiring ``CrabHexOmniFlatWalkRewardsCfg``) would otherwise
    grow an orphan ``<term>.weight`` nobody reads, and the run would train with no rotation
    income at all -- or with no facing income -- while the launcher looked correct.
    """
    for var, attr, parser in _REWARD_WEIGHT_KNOBS:
        raw = env.get(var)
        if raw is None or not str(raw).strip():
            continue
        weight = float(parser(raw))
        term = getattr(getattr(cfg, "rewards", None), attr, None)
        if term is None or not hasattr(term, "weight"):
            raise RuntimeError(
                f"{var} is set but this env's rewards cfg has no {attr} term: it is not a "
                "CrabHexOmniFlatWalkRewardsCfg (a forward/sideways rewards cfg leaked into an "
                "omni env), so the knob would be silently ignored and the omni task would "
                "train without the income it doses."
            )
        term.weight = weight


def _apply_ang_vel_std_knob(cfg, env) -> None:
    """``KRABBY_OMNI_ANG_VEL_STD`` -> ``rewards.track_ang_vel_z_exp.params["std"]`` (no-op unset).

    PRESENCE-based like every other dose knob. The loud guard is the same insurance as
    :func:`_apply_reward_weight_knobs`': a ``configclass`` instance accepts an unknown attribute
    and a ``RewTerm``'s ``params`` is a plain dict, so a rewards cfg without the term -- or with
    the param renamed upstream -- would otherwise grow an orphan ``std`` nobody reads and the run
    would train under the class-body kernel while the launcher looked retuned.
    """
    raw = env.get(ANG_VEL_STD_VAR)
    if raw is None or not str(raw).strip():
        return
    std = parse_ang_vel_std(raw)
    term = getattr(getattr(cfg, "rewards", None), "track_ang_vel_z_exp", None)
    params = getattr(term, "params", None) if term is not None else None
    if not isinstance(params, dict) or "std" not in params:
        raise RuntimeError(
            f"{ANG_VEL_STD_VAR} is set but this env's rewards cfg has no "
            "track_ang_vel_z_exp term with a 'std' param, so the knob would be silently "
            "ignored and the run would train under the class-body kernel width."
        )
    params["std"] = std


def _apply_ang_turn_only_knob(cfg, env, *, gated_func=None, stock_func=None) -> None:
    """``KRABBY_OMNI_ANG_TURN_ONLY=1`` -> swap ``rewards.track_ang_vel_z_exp.func`` for the
    turn-gated twin. Unset / empty / ``"0"`` leaves the shipped stock func in place.

    THE DEFAULT IS OFF AND STAYS OFF: no preset sets this variable (:data:`ANG_TURN_ONLY_VAR`
    explains why), so an omni run is unchanged unless a launcher arms it by hand.

    Runs AFTER :func:`_apply_ang_vel_std_knob`, which is the whole point of the ordering: the
    twin takes ``std`` straight through to the stock function, so the kernel the gated term
    prices is whatever ``KRABBY_OMNI_ANG_VEL_STD`` just wrote -- one dose, one place, whether
    or not the lever is armed.

    Three loud guards, because every failure mode of this knob is silent (the term keeps its
    NAME, so ``Episode_Reward/track_ang_vel_z_exp`` appears either way and merely reads lower):

    * no ``track_ang_vel_z_exp`` term (a forward/sideways rewards cfg leaked into an omni env)
      -> the swap would create an orphan attribute nobody reads;
    * its ``params`` has lost ``std`` -> the twin requires it, and the reward manager would
      raise on the first step of training instead of at config build;
    * its ``func`` is NOT the stock ``track_ang_vel_z_exp`` -> something else already owns the
      term and this knob would silently clobber it.

    Args:
        gated_func / stock_func: the two functions, injected for the pure unit tests (the
            :func:`_apply_command_knobs` ``cmd_cls`` precedent). ``None`` -> the real ones,
            imported lazily so this module stays Isaac-free at module scope.
    """
    raw = env.get(ANG_TURN_ONLY_VAR)
    if raw is None or not str(raw).strip():
        return
    if not parse_ang_turn_only(raw):
        return  # the explicit "0": validated, then the shipped default

    term = getattr(getattr(cfg, "rewards", None), "track_ang_vel_z_exp", None)
    if term is None or not hasattr(term, "func"):
        raise RuntimeError(
            f"{ANG_TURN_ONLY_VAR}=1 is set but this env's rewards cfg has no "
            "track_ang_vel_z_exp term, so the turn gate would be silently ignored and the run "
            "would train under the ungated kernel the lever exists to remove."
        )
    params = getattr(term, "params", None)
    if not isinstance(params, dict) or "std" not in params:
        raise RuntimeError(
            f"{ANG_TURN_ONLY_VAR}=1 is set but rewards.track_ang_vel_z_exp has no 'std' param; "
            "the gated twin passes std straight through to the stock function, so the reward "
            "manager would raise on the first training step rather than here."
        )

    if stock_func is None:
        from isaaclab.envs.mdp.rewards import track_ang_vel_z_exp as stock_func  # noqa: PLC0415
    if gated_func is None:
        from parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_rewards import (  # noqa: PLC0415
            track_ang_vel_z_exp_turn_gated as gated_func,
        )

    if term.func is not stock_func:
        raise RuntimeError(
            f"{ANG_TURN_ONLY_VAR}=1 expects rewards.track_ang_vel_z_exp to carry the stock "
            f"isaaclab track_ang_vel_z_exp, but it carries {getattr(term.func, '__name__', term.func)!r}; "
            "refusing to clobber a func this knob does not know about."
        )
    term.func = gated_func


def _apply_track_facing_sigma_knob(cfg, env, *, coupled_func=None, stock_func=None) -> None:
    """``KRABBY_OMNI_TRACK_FACING_SIGMA=<rad>`` -> couple translation tracking to the facing.

    Swaps ``rewards.track_lin_vel_xy_exp``'s FUNC for
    ``crab_hex_omni_rewards.track_lin_vel_xy_exp_facing_coupled`` (the stock kernel times a
    Gaussian of ``wrap(heading_target - yaw)``) and writes ``params["facing_sigma"]``. Unset or
    empty leaves the shipped stock func and adds no param -- the default is OFF and no preset
    sets it.

    ``track_lin_vel_xy_exp`` is INHERITED from the forward ``CrabHexFlatWalkRewardsCfg`` (the
    omni rewards cfg re-declares four other terms, not this one), so the func this knob expects
    to find is the stock ``isaaclab.envs.mdp.rewards.track_lin_vel_xy_exp`` at weight 1.25, and
    its ``std`` is whatever ``KRABBY_TRACK_SIGMA2`` wrote in the forward ``__post_init__`` --
    which is why this runs AFTER the forward block and passes ``std`` straight through.

    Three loud guards, the :func:`_apply_ang_turn_only_knob` set, for the same reason (the term
    keeps its NAME under the swap, so ``Episode_Reward/track_lin_vel_xy_exp`` appears either way
    and merely reads lower):

    * no ``track_lin_vel_xy_exp`` term -> a leaked forward/sideways rewards cfg, and the swap
      would create an orphan attribute nobody reads;
    * its ``params`` are not the stock term's (``command_name`` / ``std``) -> the coupled twin
      is the stock signature plus one keyword, so the reward manager would raise on the first
      training step rather than here;
    * its ``func`` is not the stock one -> something else already owns the term, and this knob
      must not clobber a func it does not know about.

    Args:
        coupled_func / stock_func: injected for the pure unit tests (the ``cmd_cls`` /
            ``gated_func`` precedent); ``None`` -> the real ones, imported lazily so this
            module stays Isaac-free at module scope.
    """
    raw = env.get(TRACK_FACING_SIGMA_VAR)
    if raw is None or not str(raw).strip():
        return
    sigma = parse_track_facing_sigma(raw)

    term = getattr(getattr(cfg, "rewards", None), "track_lin_vel_xy_exp", None)
    if term is None or not hasattr(term, "func"):
        raise RuntimeError(
            f"{TRACK_FACING_SIGMA_VAR} is set but this env's rewards cfg has no "
            "track_lin_vel_xy_exp term, so the facing coupling would be silently ignored and "
            "the run would train under the uncoupled kernel the knob exists to replace."
        )
    params = getattr(term, "params", None)
    if not isinstance(params, dict) or "command_name" not in params or "std" not in params:
        raise RuntimeError(
            f"{TRACK_FACING_SIGMA_VAR} is set but rewards.track_lin_vel_xy_exp does not carry "
            "the stock params ('command_name', 'std'); the coupled twin is the stock signature "
            "plus a 'facing_sigma' keyword, so the reward manager would raise on the first "
            "training step rather than here."
        )

    if stock_func is None:
        from isaaclab.envs.mdp.rewards import track_lin_vel_xy_exp as stock_func  # noqa: PLC0415
    if coupled_func is None:
        from parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_rewards import (  # noqa: PLC0415
            track_lin_vel_xy_exp_facing_coupled as coupled_func,
        )

    if term.func is not stock_func:
        raise RuntimeError(
            f"{TRACK_FACING_SIGMA_VAR} expects rewards.track_lin_vel_xy_exp to carry the stock "
            f"isaaclab track_lin_vel_xy_exp, but it carries "
            f"{getattr(term.func, '__name__', term.func)!r}: something else owns the func and "
            "this knob refuses to clobber it."
        )
    term.func = coupled_func
    term.params["facing_sigma"] = sigma


def _apply_facing_obs_encoding_knob(cfg, env) -> None:
    """``KRABBY_OMNI_FACING_OBS_ENCODING=raw|sincos`` -> the omni obs term's param.

    Unset or empty is a no-op (the class body's ``"raw"`` survives). The term must exist AND
    must ALREADY carry the key: the observation function takes ``facing_obs_encoding`` as a
    declared keyword, so writing the key onto a term that does not have it would mean either a
    leaked forward observation cfg (the knob silently inert) or a param the function does not
    accept (a crash inside the observation manager on the first step of training).
    """
    raw = env.get(FACING_OBS_ENCODING_VAR)
    if raw is None or not str(raw).strip():
        return
    encoding = parse_facing_obs_encoding(raw)
    terms = _omni_obs_terms(cfg)
    if not terms:
        raise RuntimeError(
            f"{FACING_OBS_ENCODING_VAR} is set but no omni observation term is installed"
        )
    missing = [f"{g}.{t}" for g, t, term in terms
               if not isinstance(getattr(term, "params", None), dict)
               or "facing_obs_encoding" not in term.params]
    if missing:
        raise RuntimeError(
            f"{FACING_OBS_ENCODING_VAR} is set but {missing} do not declare a "
            "'facing_obs_encoding' param; the observation function would be called with a "
            "keyword it does not accept, so the run would die inside the observation manager "
            "on its first step."
        )
    for _g, _t, term in terms:
        term.params["facing_obs_encoding"] = encoding


def _check_clock_radius_rule(cfg) -> None:
    """Refuse a config whose gait clock would STOP on every commanded turn.

    ``omni_clock_turn_radius x ang_vel_max`` is the tangential speed a pure turn hands the gait
    clock (see :data:`CLOCK_TURN_MIN_SPEED`). A cross-knob rule, so it cannot live in either
    parser: both 1.2 m and 0.1 rad/s are perfectly ordinary values, and their product is not.
    Checked after BOTH knobs have been applied, so it sees the effective pair, and it names both
    variables and both values -- the failure it prevents (gait rewards paying a statue on every
    turn row) looks like a training problem, not a config one.
    """
    radius = getattr(cfg, "omni_clock_turn_radius", None)
    if radius is None:
        raise RuntimeError(
            f"{CLOCK_TURN_R_VAR}: this env cfg has no 'omni_clock_turn_radius' attribute, so "
            "the gait clock's turn radius cannot be checked (a forward/sideways env cfg leaked "
            "into the omni knob block)."
        )
    cap = getattr(getattr(cfg.commands, "base_velocity", None), "ang_vel_max", None)
    if cap is None:
        # Same leak, spotted one level down (a forward/sideways command cfg has no
        # ``ang_vel_max``): a bare AttributeError here would not name the knob block.
        raise RuntimeError(
            f"{ANG_VEL_MAX_VAR}: commands.base_velocity has no 'ang_vel_max' attribute, so the "
            "gait clock's turn speed cannot be checked (a forward/sideways command cfg leaked "
            "into the omni knob block)."
        )
    ang_vel_max = float(cap)
    speed = float(radius) * ang_vel_max
    if not speed >= CLOCK_TURN_MIN_SPEED:
        raise RuntimeError(
            f"{CLOCK_TURN_R_VAR}={radius} x {ANG_VEL_MAX_VAR}={ang_vel_max} = {speed} m/s is "
            f"below {CLOCK_TURN_MIN_SPEED} m/s: the gait clock stops at the lin_vel_clip, so "
            "every commanded turn would be a frozen clock and the gait rewards would pay a "
            "motionless robot. Raise the radius or the yaw-rate cap. NOTE this also refuses "
            f"{CLOCK_TURN_R_VAR}=0 (the forward 'turning does not drive the clock' behaviour): "
            "an omni env that wants it has to say so by changing this rule, not by a value "
            "that is indistinguishable from a mis-set radius."
        )


def _clock_combine_terms(cfg):
    """Every reward term whose ``params`` ALREADY carries a ``combine`` key.

    Keyed on the existing param rather than on a term-name list on purpose: the clock SCHEDULE
    term takes ``combine`` (``clock_schedule_income``), the clock APEX term does not
    (``clock_swing_apex_income`` has no such argument), and writing the key onto a term whose
    function does not accept it would raise inside the reward manager on the first step. Presence
    of the key is exactly "this function understands combine".
    """
    found = []
    rewards = getattr(cfg, "rewards", None)
    if rewards is None:
        return found
    for name, term in vars(rewards).items():
        params = getattr(term, "params", None)
        if isinstance(params, dict) and "combine" in params:
            found.append((name, term))
    return found


def _apply_clock_combine_knob(cfg, env) -> None:
    """``KRABBY_OMNI_CLOCK_COMBINE=sum|product`` -> ``params["combine"]`` on the clock terms.

    Runs AFTER the forward rewards ``__post_init__`` (which applies ``KRABBY_CLOCK_COMBINE``), so
    when both are set the omni knob wins -- the documented precedence, and the reason the omni
    presets need a variable of their own at all.

    Raises when no term carries the key: a forward/sideways rewards cfg leaking into an omni env,
    or a future refactor that drops the param, must not leave the knob silently inert.
    """
    raw = env.get(CLOCK_COMBINE_VAR)
    if raw is None or not str(raw).strip():
        return
    mode = parse_clock_combine(raw)
    terms = _clock_combine_terms(cfg)
    if not terms:
        raise RuntimeError(
            f"{CLOCK_COMBINE_VAR} is set but no reward term on this cfg takes a 'combine' "
            "param (expected rewards.reward_clock_schedule); the knob would be silently ignored."
        )
    for _name, term in terms:
        term.params["combine"] = mode


def _repoint_rsi(cfg, env, yaw_range) -> None:
    """Swap ``events.rsi_reference_reset`` onto the omni RSI reset, same attribute name.

    A no-op when the forward knob block did not build the term (``KRABBY_RSI_FRAC`` unset or
    0): RSI-off is a legitimate omni configuration (the eval and the yaw-capability smoke run
    that way), and re-creating the term here would put it back into runs that asked for none.
    """
    events = getattr(cfg, "events", None)
    term = getattr(events, "rsi_reference_reset", None) if events is not None else None
    if term is None:
        return

    from parkour_tasks.crab_hex_omni_task.mdp import crab_hex_omni_rsi as _rsi  # noqa: PLC0415
    from parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_command import (  # noqa: PLC0415
        omni_walk_floor,
    )

    raw_source = env.get(RSI_SOURCE_VAR)
    bank = env.get(RSI_BANK_VAR) or term.params.get("bank_path")
    fraction = float(env.get(RSI_FRAC_VAR, term.params.get("fraction", 0.0)))
    term.func = _rsi.reset_from_reference_states_omni
    term.params = {
        "bank_path": bank,
        "fraction": fraction,
        "yaw_range": (-math.pi, math.pi) if yaw_range is None else yaw_range,
        # derived, never a literal: the staged command must sit above the clock's stop speed
        "walk_floor": omni_walk_floor(cfg.commands.base_velocity),
        "source": parse_rsi_source(raw_source) if raw_source else "all",
    }


def apply_omni_knobs(cfg, environ=None) -> None:
    """Apply every omni knob to a built omni env cfg. Call LAST in ``__post_init__``.

    Args:
        cfg: the omni env cfg (after ``super().__post_init__()``, i.e. after
            ``apply_flat_walk_knobs`` has run).
        environ: mapping to read the knobs from (defaults to ``os.environ``).
    """
    env = os.environ if environ is None else environ

    _apply_command_knobs(cfg.commands.base_velocity, env)

    raw_turn_r = env.get(CLOCK_TURN_R_VAR)
    if raw_turn_r is not None and str(raw_turn_r).strip():
        cfg.omni_clock_turn_radius = parse_clock_turn_radius(raw_turn_r)
    # Both halves of the product are now final (the command knobs wrote ang_vel_max above).
    _check_clock_radius_rule(cfg)

    raw_yaw = env.get(YAW_RANGE_VAR)
    yaw_range = parse_yaw_range(raw_yaw) if raw_yaw and str(raw_yaw).strip() else None
    _apply_spawn_yaw(cfg, yaw_range)
    _repoint_rsi(cfg, env, yaw_range)
    _apply_reward_weight_knobs(cfg, env)
    # Revision 4 (2026-09-18): the reward-economy shape knobs. Both run after the forward
    # rewards __post_init__, so the omni spelling wins over any forward-named twin.
    _apply_ang_vel_std_knob(cfg, env)
    # The staged lever (2026-09-18), default OFF and set by no preset. AFTER the std knob on
    # purpose: the gated twin passes std through, so it prices the kernel that knob just wrote.
    _apply_ang_turn_only_knob(cfg, env)
    # The capability-curriculum levers (2026-09-18, Part A1), all default-off. The facing
    # coupling runs here for the same reason the turn gate does: it swaps a FUNC on a term
    # whose params the forward block has already written, and it passes them through.
    _apply_track_facing_sigma_knob(cfg, env)
    _apply_clock_combine_knob(cfg, env)
    _apply_facing_obs_clip_knob(cfg, env)
    _apply_facing_obs_encoding_knob(cfg, env)

    from parkour_tasks.crab_hex_omni_task.mdp.omni_terrain_knobs import (  # noqa: PLC0415
        apply_omni_ring_knobs,
    )

    apply_omni_ring_knobs(cfg, env)
    # The motion-quality knobs (all default off; each one unset writes nothing). LAST, so a term
    # they add, and the obs params they arm, see the otherwise-final cfg.
    _mk.apply_motion_knobs(cfg, env)
