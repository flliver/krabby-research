# SPDX-License-Identifier: BSD-3-Clause
"""Motion-quality knobs (gait-stabilisation plan, step 2, 2026-09-22) -> the omni env / runner cfgs.

EVERY KNOB HERE IS DEFAULT OFF AND SET BY NO PRESET. Unset, blank or at its off value
(``"0"`` for flags, weights and the filter corner; ``"0"``/``"off"`` for the relay hysteresis)
each one writes NOTHING into the cfg -- no field, no term, no param -- so a knob-off config
dump is byte-identical to one built before this module existed (the ``dumps_ladder_base``
contract), and an eval that carries a knob at its off value builds the same cfg as one that
does not carry it at all (``run_phases.EVAL_CARRY_DEFAULTS``).

==============================  ===========================================================
variable                        what it writes when armed
==============================  ===========================================================
**exploration (runner cfg; ``apply_exploration_knobs``)**
``KRABBY_ACTION_STD_MAX``       ``policy.action_std_max`` -- the upper clamp on the Gaussian
                                std (``CrabHexActorCriticRMA``; stock 2.0)
``KRABBY_ACTION_STD_RESET``     ``policy.action_std_reset`` -- on EVERY checkpoint load
                                (resume / fork) the std parameter is set to this value
``KRABBY_ENTROPY_COEF``         ``algorithm.entropy_coef`` (stock 0.01)
**regularisers (env cfg rewards; weights are the RewTerm weight, sign included, <= 0)**
``KRABBY_ACTION_RATE_SQ=1``     ``rewards.reward_action_rate.params["squared"] = True``
                                (``||da||^2`` instead of ``||da||``; weight stays
                                ``KRABBY_ACTION_RATE_W``'s, stock -0.3)
``KRABBY_ACTION_SMOOTH2_W``     ADDS ``rewards.penalty_action_smooth2`` -- squared second
                                difference of the policy's action history
``KRABBY_JOINT_VEL_W``          ADDS ``rewards.penalty_joint_vel`` -- squared joint speed of
                                the 12 position-actuated (rod) joints
``KRABBY_FOOT_SLIP_W``          ADDS ``rewards.penalty_foot_slip`` -- UNSQUARED planar speed
                                of every foot loaded above 50 N, phase-independent
``KRABBY_STAND_POSE_W``         ADDS ``rewards.penalty_stand_pose`` -- STAND-slot-gated L1
                                deviation of the rod joints from the default pose
**action path**
``KRABBY_ACTION_LPF_HZ``        ``actions.joint_pos.lpf_hz`` -- first-order low-pass corner
                                (Hz) on the rod-length command, before the rod slew limiter
``KRABBY_ACTION_LPF_OBS=1``     omni obs term ``params["lpf_obs"] = True`` -- the filtered
                                rod command as 12 extra head channels (needs LPF_HZ)
**command / clock**
``KRABBY_OMNI_RELAY_HYST``      ``commands.base_velocity.relay_hyst = (engage, release)``
                                rad; ``1`` = :data:`RELAY_HYST_DEFAULT` (0.25, 0.05)
``KRABBY_OMNI_STAND_HOLD=1``    ``commands.base_velocity.stand_hold = True`` -- applied wz 0
                                (hence clock speed 0) on the STAND slot
**stand regime**
``KRABBY_OMNI_STAND_PARK=1``    ``omni_stand_park = True`` (env cfg) -- STAND slot: clock to
                                the next touchdown, all six legs stance, no swing/apex income
``KRABBY_OMNI_STAND_OBS=1``     omni obs term ``params["stand_obs"] = True`` -- the STAND bit
                                as an extra head channel, clock sin/cos zeroed while it is set
==============================  ===========================================================

Three knobs the plan names ALREADY EXIST in the forward rewards cfg's ``_overrides`` table and
are reused, not duplicated (they reach the omni flat-walk / teacher rewards because
``CrabHexOmniFlatWalkRewardsCfg`` inherits that ``__post_init__``): ``KRABBY_POWER_W``
(``penalty_mechanical_power``, sum |tau * qdot| over all joints), ``KRABBY_ACTION_RATE_W``
(``reward_action_rate``) and ``KRABBY_AIRTIME_W`` (``reward_feet_air_time_positive`` -- the
plan's "AIR_TIME_W"). :data:`REUSED_WEIGHT_KNOBS` names them for the ladder's dump audit.

Both obs-changing knobs (``KRABBY_ACTION_LPF_OBS``, ``KRABBY_OMNI_STAND_OBS``) change the
observation WIDTH: a checkpoint trained without them cannot load with them (from-scratch runs
only), and ``crab_hex_omni_mirror.build_mirrors`` grows its head tables from the same params.

Pure at module scope (no Isaac, no torch): the appliers import ``RewTerm`` / ``SceneEntityCfg`` /
the reward functions lazily, and every parser runs in a bare interpreter -- see
``tests/unit/test_crab_hex_omni_motion_knobs.py``.
"""

from __future__ import annotations

import math
import os

# ------------------------------------------------------------------------------ variables
ACTION_STD_MAX_VAR = "KRABBY_ACTION_STD_MAX"
ACTION_STD_RESET_VAR = "KRABBY_ACTION_STD_RESET"
ENTROPY_COEF_VAR = "KRABBY_ENTROPY_COEF"
ACTION_RATE_SQ_VAR = "KRABBY_ACTION_RATE_SQ"
ACTION_SMOOTH2_W_VAR = "KRABBY_ACTION_SMOOTH2_W"
JOINT_VEL_W_VAR = "KRABBY_JOINT_VEL_W"
FOOT_SLIP_W_VAR = "KRABBY_FOOT_SLIP_W"
STAND_POSE_W_VAR = "KRABBY_STAND_POSE_W"
ACTION_LPF_HZ_VAR = "KRABBY_ACTION_LPF_HZ"
ACTION_LPF_OBS_VAR = "KRABBY_ACTION_LPF_OBS"
RELAY_HYST_VAR = "KRABBY_OMNI_RELAY_HYST"
STAND_HOLD_VAR = "KRABBY_OMNI_STAND_HOLD"
STAND_PARK_VAR = "KRABBY_OMNI_STAND_PARK"
STAND_OBS_VAR = "KRABBY_OMNI_STAND_OBS"
#: The three EXISTING forward weight knobs the plan reuses (applied by the forward
#: ``CrabHexFlatWalkRewardsCfg.__post_init__`` ``_overrides`` table) -> the term they dose.
POWER_W_VAR = "KRABBY_POWER_W"
ACTION_RATE_W_VAR = "KRABBY_ACTION_RATE_W"
AIRTIME_W_VAR = "KRABBY_AIRTIME_W"
REUSED_WEIGHT_KNOBS = {
    POWER_W_VAR: "penalty_mechanical_power",
    ACTION_RATE_W_VAR: "reward_action_rate",
    AIRTIME_W_VAR: "reward_feet_air_time_positive",
}

#: Runner-side knobs (``params/agent.yaml``), everything else is env-side (``params/env.yaml``).
AGENT_KNOBS = (ACTION_STD_MAX_VAR, ACTION_STD_RESET_VAR, ENTROPY_COEF_VAR)

#: The env-side knobs that change command, clock, action-path or observation SEMANTICS: an eval
#: of a policy trained with one of them must be built with it too (``run_phases.EVAL_CARRY``).
EVAL_SEMANTIC_KNOBS = (RELAY_HYST_VAR, STAND_HOLD_VAR, STAND_PARK_VAR, ACTION_LPF_HZ_VAR,
                       ACTION_LPF_OBS_VAR, STAND_OBS_VAR)
#: ... and the value each one takes when OFF (what ``EVAL_CARRY_DEFAULTS`` restates).
KNOB_OFF_VALUES = {RELAY_HYST_VAR: "0", STAND_HOLD_VAR: "0", STAND_PARK_VAR: "0",
                   ACTION_LPF_HZ_VAR: "0", ACTION_LPF_OBS_VAR: "0", STAND_OBS_VAR: "0"}

#: ``(engage, release)`` rad of facing error for ``KRABBY_OMNI_RELAY_HYST=1``.
RELAY_HYST_DEFAULT = (0.25, 0.05)
#: ``CrabHexActorCriticRMA``'s stock clamp floor: a std cap / reset below it is meaningless.
ACTION_STD_MIN = 0.05
#: ... and its stock ceiling (``ACTION_STD_MAX`` there), in force while the knob is unset.
STOCK_ACTION_STD_MAX = 2.0
#: Upper bound on the filter corner: the rod command is filtered at the PHYSICS rate (200 Hz on
#: the shipped sim.dt 0.005), so anything at or past its Nyquist frequency is not a low-pass.
LPF_HZ_MAX = 100.0
#: The "loaded foot" threshold of ``KRABBY_FOOT_SLIP_W`` -- the clock reward's ``TOUCH_REF_N``.
FOOT_SLIP_FORCE_N = 50.0

#: Names of the reward terms the weight knobs ADD (``rewards.<name>``).
SMOOTH2_TERM = "penalty_action_smooth2"
JOINT_VEL_TERM = "penalty_joint_vel"
FOOT_SLIP_TERM = "penalty_foot_slip"
STAND_POSE_TERM = "penalty_stand_pose"

_REWARDS_MOD = "parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_motion_rewards"

__all__ = [
    "ACTION_STD_MAX_VAR", "ACTION_STD_RESET_VAR", "ENTROPY_COEF_VAR", "ACTION_RATE_SQ_VAR",
    "ACTION_SMOOTH2_W_VAR", "JOINT_VEL_W_VAR", "FOOT_SLIP_W_VAR", "STAND_POSE_W_VAR",
    "ACTION_LPF_HZ_VAR", "ACTION_LPF_OBS_VAR", "RELAY_HYST_VAR", "STAND_HOLD_VAR",
    "STAND_PARK_VAR", "STAND_OBS_VAR", "POWER_W_VAR", "ACTION_RATE_W_VAR", "AIRTIME_W_VAR",
    "REUSED_WEIGHT_KNOBS", "AGENT_KNOBS", "EVAL_SEMANTIC_KNOBS", "KNOB_OFF_VALUES",
    "RELAY_HYST_DEFAULT", "parse_flag", "parse_std_max", "parse_std_reset",
    "parse_entropy_coef", "parse_penalty_w", "parse_lpf_hz", "parse_relay_hyst",
    "MOTION_KNOB_PARSERS", "apply_motion_knobs", "apply_exploration_knobs",
]


# ------------------------------------------------------------------------------ parsers
def _blank(text) -> bool:
    return text is None or not str(text).strip()


def _float(var: str, text) -> float:
    try:
        value = float(str(text).strip())
    except ValueError as exc:
        raise ValueError(f"{var} must be a number, got {text!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{var} must be finite, got {text!r}")
    return value


def parse_flag(text, var: str = "flag") -> bool:
    """``"1"`` -> True, ``"0"`` / blank -> False; anything else raises (a typo must not pick)."""
    if _blank(text):
        return False
    token = str(text).strip()
    if token == "1":
        return True
    if token == "0":
        return False
    raise ValueError(f"{var} must be '0' or '1', got {text!r}")


def parse_std_max(text) -> float | None:
    """``KRABBY_ACTION_STD_MAX`` -> the std clamp ceiling, or None (blank = stock 2.0)."""
    if _blank(text):
        return None
    value = _float(ACTION_STD_MAX_VAR, text)
    if not value > ACTION_STD_MIN:
        raise ValueError(f"{ACTION_STD_MAX_VAR} must exceed the clamp floor {ACTION_STD_MIN}, got {text!r}")
    return value


def parse_std_reset(text) -> float | None:
    """``KRABBY_ACTION_STD_RESET`` -> the std written on checkpoint load, or None (blank = keep)."""
    if _blank(text):
        return None
    value = _float(ACTION_STD_RESET_VAR, text)
    if not value >= ACTION_STD_MIN:
        raise ValueError(f"{ACTION_STD_RESET_VAR} must be >= {ACTION_STD_MIN}, got {text!r}")
    return value


def parse_entropy_coef(text) -> float | None:
    """``KRABBY_ENTROPY_COEF`` -> PPO's entropy bonus coefficient (>= 0), or None (blank)."""
    if _blank(text):
        return None
    value = _float(ENTROPY_COEF_VAR, text)
    if value < 0.0:
        raise ValueError(f"{ENTROPY_COEF_VAR} must be >= 0, got {text!r}")
    return value


def parse_penalty_w(text, var: str = "penalty weight") -> float:
    """A PENALTY weight: the RewTerm weight, sign included, so it must be <= 0 (blank -> 0).

    A positive value would PAY for the very motion the term exists to suppress; refused rather
    than sign-flipped so the recipe says exactly what trains. ``0`` means OFF (no term added).
    """
    if _blank(text):
        return 0.0
    value = _float(var, text)
    if value > 0.0:
        raise ValueError(f"{var} is a penalty weight and must be <= 0, got {text!r}")
    return value


def parse_lpf_hz(text) -> float | None:
    """``KRABBY_ACTION_LPF_HZ`` -> the corner frequency (Hz), or None for OFF (blank / ``0``)."""
    if _blank(text):
        return None
    value = _float(ACTION_LPF_HZ_VAR, text)
    if value == 0.0:
        return None
    if not 0.0 < value < LPF_HZ_MAX:
        raise ValueError(f"{ACTION_LPF_HZ_VAR} must be in (0, {LPF_HZ_MAX}) Hz or 0 (off), got {text!r}")
    return value


def parse_relay_hyst(text) -> tuple[float, float] | None:
    """``KRABBY_OMNI_RELAY_HYST`` -> ``(engage, release)`` rad, or None for OFF.

    ``""`` / ``"0"`` / ``"off"`` -> None; ``"1"`` -> :data:`RELAY_HYST_DEFAULT`; ``"e:r"`` -> an
    explicit pair with ``0 <= r < e`` (release >= engage would chatter exactly like the dead band
    it replaces).
    """
    if _blank(text):
        return None
    token = str(text).strip().lower()
    if token in ("0", "off", "none"):
        return None
    if token == "1":
        return RELAY_HYST_DEFAULT
    parts = token.split(":")
    if len(parts) != 2:
        raise ValueError(f"{RELAY_HYST_VAR} must be 0, 1 or 'engage:release' (rad), got {text!r}")
    engage, release = (_float(RELAY_HYST_VAR, p) for p in parts)
    if not 0.0 <= release < engage:
        raise ValueError(f"{RELAY_HYST_VAR} needs 0 <= release < engage, got {text!r}")
    return (engage, release)


#: ``{variable: parser}`` for every knob this module owns (the ladder's round-trip check).
MOTION_KNOB_PARSERS = {
    ACTION_STD_MAX_VAR: parse_std_max,
    ACTION_STD_RESET_VAR: parse_std_reset,
    ENTROPY_COEF_VAR: parse_entropy_coef,
    ACTION_RATE_SQ_VAR: lambda t: parse_flag(t, ACTION_RATE_SQ_VAR),
    ACTION_SMOOTH2_W_VAR: lambda t: parse_penalty_w(t, ACTION_SMOOTH2_W_VAR),
    JOINT_VEL_W_VAR: lambda t: parse_penalty_w(t, JOINT_VEL_W_VAR),
    FOOT_SLIP_W_VAR: lambda t: parse_penalty_w(t, FOOT_SLIP_W_VAR),
    STAND_POSE_W_VAR: lambda t: parse_penalty_w(t, STAND_POSE_W_VAR),
    ACTION_LPF_HZ_VAR: parse_lpf_hz,
    ACTION_LPF_OBS_VAR: lambda t: parse_flag(t, ACTION_LPF_OBS_VAR),
    RELAY_HYST_VAR: parse_relay_hyst,
    STAND_HOLD_VAR: lambda t: parse_flag(t, STAND_HOLD_VAR),
    STAND_PARK_VAR: lambda t: parse_flag(t, STAND_PARK_VAR),
    STAND_OBS_VAR: lambda t: parse_flag(t, STAND_OBS_VAR),
}


# ------------------------------------------------------------------------------ application
def _omni_obs_terms(cfg):
    """Every observation term whose func is the omni observation class (by attribute)."""
    found = []
    groups = getattr(cfg, "observations", None)
    for gname in ("policy", "critic"):
        group = getattr(groups, gname, None)
        if group is None:
            continue
        for tname, term in vars(group).items():
            if hasattr(getattr(term, "func", None), "OBSERVED_COMMAND_SLOTS"):
                found.append((gname, tname, term))
    return found


def _set_obs_param(cfg, var: str, key: str) -> None:
    terms = _omni_obs_terms(cfg)
    if not terms:
        raise RuntimeError(f"{var} is set but no omni observation term is installed")
    for _g, _t, term in terms:
        if term.params is None:
            term.params = {}
        term.params[key] = True


def _rewards(cfg, var: str):
    rewards = getattr(cfg, "rewards", None)
    if rewards is None:
        raise RuntimeError(f"{var} is set but this env cfg has no rewards block")
    return rewards


def _default_term_builders():
    """``{name: (func, params)}`` for the four ADDED penalty terms (lazy Isaac imports)."""
    import importlib  # noqa: PLC0415

    from isaaclab.managers import SceneEntityCfg  # noqa: PLC0415

    from parkour_tasks.crab_hex_forward_task.config.crab_hex.agents.parkour_mdp_cfg import (  # noqa: PLC0415
        _CRAB_FOOT_BODY_NAMES,
        _CRAB_POSITION_ACTUATED_JOINT_NAMES,
    )

    rew = importlib.import_module(_REWARDS_MOD)
    rods = SceneEntityCfg("robot", joint_names=_CRAB_POSITION_ACTUATED_JOINT_NAMES, preserve_order=True)
    return {
        SMOOTH2_TERM: (rew.penalty_action_second_diff, {"action_name": "joint_pos"}),
        JOINT_VEL_TERM: (rew.penalty_joint_vel_l2, {"asset_cfg": rods}),
        FOOT_SLIP_TERM: (rew.penalty_loaded_foot_slip, {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_CRAB_FOOT_BODY_NAMES,
                                         preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=_CRAB_FOOT_BODY_NAMES, preserve_order=True),
            "force_threshold": FOOT_SLIP_FORCE_N,
        }),
        STAND_POSE_TERM: (rew.penalty_stand_pose_l1, {
            "asset_cfg": SceneEntityCfg("robot", joint_names=_CRAB_POSITION_ACTUATED_JOINT_NAMES,
                                        preserve_order=True),
            "command_name": "base_velocity",
        }),
    }


def apply_motion_knobs(cfg, environ=None, *, term_cls=None, builders=None) -> None:
    """Apply every ENV-side motion knob to a built omni env cfg (no-op for each one that is off).

    Called by ``omni_knobs.apply_omni_knobs`` after everything else, so a term it adds sees the
    final cfg. ``term_cls`` / ``builders`` let a unit test pass stand-ins for ``RewTerm`` and the
    reward functions (the real ones need Isaac).
    """
    env = os.environ if environ is None else environ
    get = env.get

    # -- regularisers
    if parse_flag(get(ACTION_RATE_SQ_VAR), ACTION_RATE_SQ_VAR):
        term = getattr(_rewards(cfg, ACTION_RATE_SQ_VAR), "reward_action_rate", None)
        if term is None or not isinstance(getattr(term, "params", None), dict):
            raise RuntimeError(f"{ACTION_RATE_SQ_VAR}=1 but the rewards cfg has no reward_action_rate term")
        term.params["squared"] = True
    added = [(var, name, parse_penalty_w(get(var), var)) for var, name in (
        (ACTION_SMOOTH2_W_VAR, SMOOTH2_TERM), (JOINT_VEL_W_VAR, JOINT_VEL_TERM),
        (FOOT_SLIP_W_VAR, FOOT_SLIP_TERM), (STAND_POSE_W_VAR, STAND_POSE_TERM))]
    added = [a for a in added if a[2] != 0.0]
    if added:
        if term_cls is None:
            from isaaclab.managers import RewardTermCfg as term_cls  # noqa: PLC0415,N813
        table = builders if builders is not None else _default_term_builders()
        for var, name, weight in added:
            rewards = _rewards(cfg, var)
            if getattr(rewards, name, None) is not None:
                raise RuntimeError(f"{var}: the rewards cfg already carries a {name!r} term")
            func, params = table[name]
            setattr(rewards, name, term_cls(func=func, weight=weight, params=dict(params)))

    # -- action path
    hz = parse_lpf_hz(get(ACTION_LPF_HZ_VAR))
    if hz is not None:
        action = getattr(getattr(cfg, "actions", None), "joint_pos", None)
        if action is None:
            raise RuntimeError(f"{ACTION_LPF_HZ_VAR} is set but the env cfg has no actions.joint_pos")
        action.lpf_hz = hz
    if parse_flag(get(ACTION_LPF_OBS_VAR), ACTION_LPF_OBS_VAR):
        if hz is None:
            raise RuntimeError(f"{ACTION_LPF_OBS_VAR}=1 observes the filtered rod command, which "
                               f"only exists with {ACTION_LPF_HZ_VAR} > 0")
        _set_obs_param(cfg, ACTION_LPF_OBS_VAR, "lpf_obs")

    # -- command / clock
    hyst = parse_relay_hyst(get(RELAY_HYST_VAR))
    if hyst is not None:
        cfg.commands.base_velocity.relay_hyst = hyst
    stand_hold = parse_flag(get(STAND_HOLD_VAR), STAND_HOLD_VAR)
    if stand_hold:
        cfg.commands.base_velocity.stand_hold = True

    # -- stand regime
    stand_park = parse_flag(get(STAND_PARK_VAR), STAND_PARK_VAR)
    if stand_park:
        cfg.omni_stand_park = True
    stand_obs = parse_flag(get(STAND_OBS_VAR), STAND_OBS_VAR)
    if stand_obs:
        _set_obs_param(cfg, STAND_OBS_VAR, "stand_obs")
    # Every STAND-keyed knob the eval carries also latches the eval hook's slot per hold
    # (``CrabHexOmniCommand.schedule_hook``): otherwise an eval stand hold that drifts past the
    # facing dead band re-derives as "turn" and escapes the very knob it is scoring.
    if stand_hold or stand_park or stand_obs:
        cfg.commands.base_velocity.eval_slot_latch = True


def apply_exploration_knobs(agent_cfg, environ=None) -> None:
    """The three RUNNER-side knobs onto an omni PPO runner cfg (no-op for each one unset).

    The values land on the cfg that ``train.py`` dumps to ``params/agent.yaml`` (the ladder's
    agent-side audit reads them back from there): ``policy.action_std_max`` and
    ``policy.action_std_reset`` are consumed by ``CrabHexActorCriticRMA`` (its constructor pops
    them), ``algorithm.entropy_coef`` by PPO.
    """
    env = os.environ if environ is None else environ
    std_max = parse_std_max(env.get(ACTION_STD_MAX_VAR))
    std_reset = parse_std_reset(env.get(ACTION_STD_RESET_VAR))
    entropy = parse_entropy_coef(env.get(ENTROPY_COEF_VAR))
    ceiling = STOCK_ACTION_STD_MAX if std_max is None else std_max
    if std_reset is not None and std_reset > ceiling:
        raise ValueError(f"{ACTION_STD_RESET_VAR}={std_reset} exceeds the std clamp ceiling {ceiling} "
                         f"({ACTION_STD_MAX_VAR}): the reset value would be clamped away on the "
                         "first rollout")
    if std_max is not None:
        agent_cfg.policy.action_std_max = std_max
    if std_reset is not None:
        agent_cfg.policy.action_std_reset = std_reset
    if entropy is not None:
        agent_cfg.algorithm.entropy_coef = entropy
