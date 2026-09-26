# SPDX-License-Identifier: BSD-3-Clause
"""Stroke/posture knobs (stride/posture plan, step 3, 2026-09-25) -> the omni env cfg.

EVERY KNOB HERE IS DEFAULT OFF AND SET BY NO PRESET. Unset or blank, each one writes NOTHING into
the cfg -- no field, no func swap, no added term -- so a knob-off config dump is byte-identical to
one built before this module existed.

==============================  ===============================================================
variable                        what it writes when armed
==============================  ===============================================================
``KRABBY_OMNI_STROKE_W=<w>``    FUNC SWAP under the EXISTING name ``rewards.reward_stride_length``:
                                ``func`` becomes :class:`RewardFootStroke`, ``params`` become the
                                six-foot asset/sensor cfg, and ``weight`` becomes ``w`` -- all
                                three written together, so the term's telemetry name never
                                changes and a dose without the swap (or vice versa) cannot happen.
``KRABBY_OMNI_CENTRING_W=<w>``  ADDS ``rewards.reward_body_centring`` (:class:`RewardBodyCentring`)
                                at weight ``w``.
``KRABBY_OMNI_REWARD_SPLIT=1``  ``reward_category_split = True`` on the env cfg (the
                                ``motion_knobs``/``omni_stand_park`` precedent: the attribute is
                                written ONLY when the knob is on).
==============================  ===============================================================

WHY A FUNC SWAP UNDER THE OLD NAME, NOT A NEW ONE. ``reward_stride_length`` is dead bit for bit
(``crab_hex_omni_stroke_math``'s module docstring; the float32 ``compute_first_*`` timestamp
drift) but its NAME already carries the whole telemetry/driver machinery this campaign reuses:
``Episode_Reward/reward_stride_length``, the campaign ladder's dump-audit table, and every
existing ``KRABBY_STRIDE_W`` / ``KRABBY_PHASEOUT`` reference to it. Swapping the func under that
name (the ``ANG_TURN_ONLY`` / ``TRACK_FACING_SIGMA`` precedent) means a run armed with this knob
shows up in every one of those places as "the stride term", now alive.

REFUSALS (``KRABBY_OMNI_STROKE_W``), each raising ``ValueError``/``RuntimeError`` by the knob's
own name:

* ``w < 0`` or ``0 < w < 0.001`` (the epsilon floor -- a term at exactly ``0.0`` is skipped by the
  reward manager and its telemetry flatlines, so a value strictly between the two would look armed
  and pay nothing; ``0.0`` itself is a legitimate ablation, like the other weight knobs);
* ``KRABBY_STRIDE_W`` also set (non-blank) -- both dose ``reward_stride_length``, and the func
  swap runs after the forward block, so it would silently discard whatever that knob just set;
* a ``KRABBY_PHASEOUT`` that names ``reward_stride_length`` -- the curriculum term writes the
  term's weight every step, on top of a func the phaseout schedule was never written for;
* the term's current ``func`` is not the stock ``RewardStrideLength`` -- something else (a future
  knob, a leaked cfg) already owns the term and this knob refuses to clobber it. The three guards
  mirror ``omni_knobs._apply_track_facing_sigma_knob``'s (a missing term, a func already swapped),
  read before writing this module.

``KRABBY_OMNI_CENTRING_W`` refuses ``w < 0`` and ``0 < w < 0.001`` the same way, and refuses to
ADD the term twice (a rewards cfg that already carries ``reward_body_centring``).

Pure at module scope (no Isaac, no torch): the appliers import ``RewTerm`` / ``SceneEntityCfg`` /
the reward classes lazily, and every parser runs in a bare interpreter -- see
``tests/unit/test_crab_hex_omni_stroke_knobs.py``. Path-loaded and re-exported by ``omni_knobs.py``
the way ``motion_knobs.py`` / ``clock_knobs.py`` are; forward and sideways cfg modules must never
import this module (the omni rewards it names, ``RewardFootStroke`` / ``RewardBodyCentring``, do
not exist on their rewards cfgs).
"""

from __future__ import annotations

import math
import os

# ------------------------------------------------------------------------------ variables
STROKE_W_VAR = "KRABBY_OMNI_STROKE_W"
CENTRING_W_VAR = "KRABBY_OMNI_CENTRING_W"
REWARD_SPLIT_VAR = "KRABBY_OMNI_REWARD_SPLIT"

#: The names the two knobs write onto ``cfg.rewards`` -- the func swap keeps the OLD name (so the
#: driver's dump audit and every ``KRABBY_STRIDE_W``/``KRABBY_PHASEOUT`` reference still resolve);
#: the centring term is genuinely new.
STROKE_TERM = "reward_stride_length"
CENTRING_TERM = "reward_body_centring"

#: The removal weight both weight knobs share with the clock-deprecation campaign's ``CLOCK_EPS_W``
#: (``run_candidates.py``): a term at exactly ``0.0`` is skipped by the reward manager and its
#: ``Episode_Reward`` telemetry flatlines, so the epsilon floor keeps a "barely armed" dose visible.
EPS_W = 0.001

#: The env-side knob that changes observation/telemetry SEMANTICS the way ``EVAL_SEMANTIC_KNOBS``
#: elsewhere does; none of the three do (the split only adds LOGGING and the two weights are
#: ordinary reward doses), so this module carries no such tuple -- kept as a comment, not a name,
#: so a future reader does not go looking for one that does not exist.

_REWARDS_MOD = "parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_stroke_rewards"

__all__ = [
    "STROKE_W_VAR", "CENTRING_W_VAR", "REWARD_SPLIT_VAR", "STROKE_TERM", "CENTRING_TERM", "EPS_W",
    "parse_stroke_w", "parse_centring_w", "parse_reward_split", "STROKE_KNOB_PARSERS",
    "apply_stroke_knobs",
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


def _parse_added_weight(var: str, text) -> float | None:
    """A REWARD weight that ADDS or SWAPS a term: ``None`` for unset/blank (no-op), else >= 0,
    never strictly between 0 and :data:`EPS_W` (see the module docstring's refusal list)."""
    if _blank(text):
        return None
    value = _float(var, text)
    if value < 0.0:
        raise ValueError(f"{var} must be non-negative, got {text!r}")
    if 0.0 < value < EPS_W:
        raise ValueError(
            f"{var} must be 0 or >= {EPS_W} (the epsilon floor): a term at exactly 0.0 is "
            "skipped by the reward manager and its telemetry flatlines, so a value strictly "
            f"between the two would look armed and pay nothing, got {text!r}"
        )
    return value


def parse_stroke_w(text) -> float | None:
    """``KRABBY_OMNI_STROKE_W`` -> the func-swap weight, or ``None`` for unset/blank (no-op)."""
    return _parse_added_weight(STROKE_W_VAR, text)


def parse_centring_w(text) -> float | None:
    """``KRABBY_OMNI_CENTRING_W`` -> the added term's weight, or ``None`` for unset/blank (no-op)."""
    return _parse_added_weight(CENTRING_W_VAR, text)


def parse_reward_split(text) -> bool:
    """``KRABBY_OMNI_REWARD_SPLIT`` -> ``True`` (``"1"``), ``False`` (``"0"``/blank); else raises."""
    if _blank(text):
        return False
    token = str(text).strip()
    if token == "1":
        return True
    if token == "0":
        return False
    raise ValueError(f"{REWARD_SPLIT_VAR} must be '0' or '1', got {text!r}")


#: ``{variable: parser}`` for every knob this module owns (the ladder's round-trip check).
STROKE_KNOB_PARSERS = {
    STROKE_W_VAR: parse_stroke_w,
    CENTRING_W_VAR: parse_centring_w,
    REWARD_SPLIT_VAR: parse_reward_split,
}


def _phaseout_names(spec, term: str) -> bool:
    """Does a raw ``KRABBY_PHASEOUT`` spec (``term:w0:w1:t0:t1[,term:...]``) name ``term``?

    A light parse of the spec's own grammar, not ``curriculums.parse_phaseout_spec``'s: this
    module stays pure (no Isaac, no path-load of the forward mdp package), and all the stroke
    knob needs is whether ``term`` appears as an entry's own name.
    """
    if _blank(spec):
        return False
    return any(entry.split(":", 1)[0].strip() == term for entry in str(spec).split(","))


# ------------------------------------------------------------------------------ application
def _rewards(cfg, var: str):
    rewards = getattr(cfg, "rewards", None)
    if rewards is None:
        raise RuntimeError(f"{var} is set but this env cfg has no rewards block")
    return rewards


def _default_term_builders():
    """``{name: (func, params)}`` for the two stroke/posture terms (lazy Isaac imports)."""
    import importlib  # noqa: PLC0415

    from isaaclab.managers import SceneEntityCfg  # noqa: PLC0415

    from parkour_tasks.crab_hex_forward_task.config.crab_hex.agents.parkour_mdp_cfg import (  # noqa: PLC0415
        _CRAB_FOOT_BODY_NAMES,
    )

    rew = importlib.import_module(_REWARDS_MOD)

    def feet():
        return SceneEntityCfg("robot", body_names=_CRAB_FOOT_BODY_NAMES, preserve_order=True)

    return {
        STROKE_TERM: (rew.RewardFootStroke, {
            "asset_cfg": feet(),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_CRAB_FOOT_BODY_NAMES,
                                         preserve_order=True),
            "command_name": "base_velocity",
        }),
        CENTRING_TERM: (rew.RewardBodyCentring, {
            "asset_cfg": feet(),
            "command_name": "base_velocity",
        }),
    }


def _apply_stroke_w_knob(cfg, env, weight: float, *, term_cls=None, builders=None,
                         stock_func=None) -> None:
    """``KRABBY_OMNI_STROKE_W``'s func swap: see the module docstring's refusal list."""
    if not _blank(env.get("KRABBY_STRIDE_W")):
        raise RuntimeError(
            f"{STROKE_W_VAR} is set but so is KRABBY_STRIDE_W: both dose {STROKE_TERM}, and this "
            "func swap runs after the forward block, so it would silently discard whatever "
            f"KRABBY_STRIDE_W just set. Unset KRABBY_STRIDE_W to arm {STROKE_W_VAR}."
        )
    if _phaseout_names(env.get("KRABBY_PHASEOUT"), STROKE_TERM):
        raise RuntimeError(
            f"{STROKE_W_VAR} is set but KRABBY_PHASEOUT names {STROKE_TERM}: the curriculum "
            "writes the term's weight every step, on top of a func the phaseout schedule was "
            f"never written for. Drop {STROKE_TERM} from KRABBY_PHASEOUT to arm {STROKE_W_VAR}."
        )
    rewards = _rewards(cfg, STROKE_W_VAR)
    term = getattr(rewards, STROKE_TERM, None)
    if term is None or not hasattr(term, "func"):
        raise RuntimeError(
            f"{STROKE_W_VAR} is set but this env's rewards cfg has no {STROKE_TERM} term to "
            "swap, so the knob would be silently ignored."
        )
    if stock_func is None:
        from parkour_isaaclab.envs.mdp.rewards import RewardStrideLength as stock_func  # noqa: PLC0415
    if term.func is not stock_func:
        raise RuntimeError(
            f"{STROKE_W_VAR} expects rewards.{STROKE_TERM} to carry the stock RewardStrideLength, "
            f"but it carries {getattr(term.func, '__name__', term.func)!r}: something else "
            "already owns the term and this knob refuses to clobber it."
        )
    if term_cls is None:
        from isaaclab.managers import RewardTermCfg as term_cls  # noqa: PLC0415,N813
    table = builders if builders is not None else _default_term_builders()
    func, params = table[STROKE_TERM]
    setattr(rewards, STROKE_TERM, term_cls(func=func, weight=weight, params=dict(params)))


def _apply_centring_w_knob(cfg, weight: float, *, term_cls=None, builders=None) -> None:
    """``KRABBY_OMNI_CENTRING_W``'s ADD: refuses a rewards cfg that already carries the term."""
    rewards = _rewards(cfg, CENTRING_W_VAR)
    if getattr(rewards, CENTRING_TERM, None) is not None:
        raise RuntimeError(f"{CENTRING_W_VAR}: the rewards cfg already carries a "
                           f"{CENTRING_TERM!r} term")
    if term_cls is None:
        from isaaclab.managers import RewardTermCfg as term_cls  # noqa: PLC0415,N813
    table = builders if builders is not None else _default_term_builders()
    func, params = table[CENTRING_TERM]
    setattr(rewards, CENTRING_TERM, term_cls(func=func, weight=weight, params=dict(params)))


def apply_stroke_knobs(cfg, env=os.environ, *, term_cls=None, builders=None) -> None:
    """Apply the three stroke/posture knobs to a built omni env cfg (no-op for each one that is
    off). Called by ``omni_knobs.apply_omni_knobs`` LAST, so the func swap and the added term see
    the otherwise-final cfg. ``term_cls`` / ``builders`` let a unit test pass stand-ins for
    ``RewTerm`` and the reward classes (the ``motion_knobs`` pattern; the real ones need Isaac).
    """
    get = env.get

    stroke_w = parse_stroke_w(get(STROKE_W_VAR))
    if stroke_w is not None:
        _apply_stroke_w_knob(cfg, env, stroke_w, term_cls=term_cls, builders=builders)

    centring_w = parse_centring_w(get(CENTRING_W_VAR))
    if centring_w is not None:
        _apply_centring_w_knob(cfg, centring_w, term_cls=term_cls, builders=builders)

    if parse_reward_split(get(REWARD_SPLIT_VAR)):
        cfg.reward_category_split = True
