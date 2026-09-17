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
6. ``apply_omni_ring_knobs`` -- the ring-generator guard and the ``KRABBY_RING_*`` knobs
   (``mdp/omni_terrain_knobs.py``).

Every parser is pure and lives at module scope with no Isaac import (the Isaac-touching bits
are lazy imports inside :func:`apply_omni_knobs`), so this module can be path-loaded and
unit-tested without booting Kit -- see ``tests/unit/test_crab_hex_omni_phases.py``.

Format conventions follow the rest of the Krabby knobs: ``lo:hi`` for a band, ``:``-separated
for a fixed-length tuple, a bare number for a scalar. Every parser rejects the wrong arity
rather than padding or truncating: a typo in a launcher must not silently retune the sampler.
"""
from __future__ import annotations

import math
import os

__all__ = [
    "SLOTS_VAR",
    "DIR_BAND_VAR",
    "SPEED_ELLIPSE_VAR",
    "FACING_VAR",
    "ANG_VEL_MAX_VAR",
    "CLOCK_TURN_R_VAR",
    "YAW_RANGE_VAR",
    "RSI_SOURCE_VAR",
    "RSI_BANK_VAR",
    "RSI_FRAC_VAR",
    "YAW_PROGRESS_W_VAR",
    "parse_slots",
    "parse_dir_band",
    "parse_speed_ellipse",
    "parse_facing",
    "parse_ang_vel_max",
    "parse_clock_turn_radius",
    "parse_yaw_range",
    "parse_rsi_source",
    "apply_omni_knobs",
]

SLOTS_VAR = "KRABBY_OMNI_SLOTS"
DIR_BAND_VAR = "KRABBY_OMNI_DIR_BAND"
SPEED_ELLIPSE_VAR = "KRABBY_OMNI_SPEED_ELLIPSE"
FACING_VAR = "KRABBY_OMNI_FACING"
ANG_VEL_MAX_VAR = "KRABBY_OMNI_ANG_VEL_MAX"
CLOCK_TURN_R_VAR = "KRABBY_OMNI_CLOCK_TURN_R"
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


def parse_clock_turn_radius(text: str) -> float:
    """``KRABBY_OMNI_CLOCK_TURN_R`` -> the turn radius the gait clock prices, metres.

    Non-negative; 0 means "turning does not drive the clock" (the forward behaviour).
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
    """The five sampler knobs onto the omni command cfg (unset knob = cfg default kept).

    ``isinstance``-checked rather than left to ``setattr`` to fail on its own: a
    ``configclass`` instance happily accepts an unknown attribute (it does not raise
    ``AttributeError``), so a leaked forward/sideways command cfg would otherwise silently grow
    five orphan fields here instead of failing loudly.

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
    # dir_band is special: an EMPTY value is a meaningful "no band" (stage 1c+), so the knob
    # is applied whenever it is present, not only when it is non-empty.
    raw_band = env.get(DIR_BAND_VAR)
    if raw_band is not None:
        cmd_cfg.dir_band = parse_dir_band(raw_band)


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


def _apply_reward_weight_knobs(cfg, env) -> None:
    """``KRABBY_YAW_PROGRESS_W`` -> ``rewards.reward_yaw_progress.weight`` (no-op when unset).

    PRESENCE-based, exactly like the forward block's ``_overrides``: an explicitly-set knob
    wins even at ``0.0``, which is how an ablation switches the added term off. Only an empty
    value is ignored (``float("")`` would raise).

    The ``hasattr`` check is the same silent-failure insurance as
    :func:`_apply_command_knobs`'s type check: a ``configclass`` instance accepts an unknown
    attribute without complaint, so a forward/sideways rewards cfg leaking into an omni env
    (or a future refactor that stops wiring ``CrabHexOmniFlatWalkRewardsCfg``) would otherwise
    grow an orphan ``reward_yaw_progress.weight`` nobody reads, and the run would train with
    no rotation income at all while the launcher looked correct.
    """
    raw = env.get(YAW_PROGRESS_W_VAR)
    if raw is None or not str(raw).strip():
        return
    term = getattr(getattr(cfg, "rewards", None), "reward_yaw_progress", None)
    if term is None or not hasattr(term, "weight"):
        raise RuntimeError(
            f"{YAW_PROGRESS_W_VAR} is set but this env's rewards cfg has no "
            "reward_yaw_progress term: it is not a CrabHexOmniFlatWalkRewardsCfg (a "
            "forward/sideways rewards cfg leaked into an omni env), so the knob would be "
            "silently ignored and the omni task would train with no yaw-progress income."
        )
    term.weight = float(raw)


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

    raw_yaw = env.get(YAW_RANGE_VAR)
    yaw_range = parse_yaw_range(raw_yaw) if raw_yaw and str(raw_yaw).strip() else None
    _apply_spawn_yaw(cfg, yaw_range)
    _repoint_rsi(cfg, env, yaw_range)
    _apply_reward_weight_knobs(cfg, env)
    _apply_facing_obs_clip_knob(cfg, env)

    from parkour_tasks.crab_hex_omni_task.mdp.omni_terrain_knobs import (  # noqa: PLC0415
        apply_omni_ring_knobs,
    )

    apply_omni_ring_knobs(cfg, env)
