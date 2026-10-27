# SPDX-License-Identifier: BSD-3-Clause
"""Payload knobs (M18 Task 4 pipeline, step 3 "M1a", 2026-10-16) -> the ``events.payload_reset`` term.

ALL FOUR KNOBS ARE DEFAULT OFF AND SET BY NO PRESET. With ``KRABBY_PAYLOAD_KG`` unset or blank they write NOTHING
into the cfg -- no event term, no param -- so a knob-off config dump is byte-identical to one built before this
module existed. ``KRABBY_PAYLOAD_KG`` is the arming knob; once it is set the other three are REQUIRED (a recipe
always states its whole payload geometry), and any of the other three set without it is refused.

================================  ==================================================================================
variable                          meaning (D5, approved at STOP-0 2026-10-16)
================================  ==================================================================================
``KRABBY_PAYLOAD_KG=lo:hi``       added mass per env, drawn U[lo, hi] kg at EVERY reset (``0 <= lo <= hi <=``
                                  :data:`PAYLOAD_KG_MAX`). ``L:L`` pins every env at exactly L kg; ``0:0`` arms the
                                  term but adds nothing (a true no-op: no PhysX write at all, see ``omni_events``)
``KRABBY_PAYLOAD_ZERO_FRAC=p``    per-reset probability (``0 <= p <= 1``) that the env carries NO payload this
                                  episode, independently of the U draw (D5 training: 0.25; level rows: 0)
``KRABBY_PAYLOAD_COM_XY=r``       the payload CoM's body-frame x and y offsets, each U[-r, r] m per reset
                                  (``0 <= r <=`` :data:`PAYLOAD_COM_XY_MAX`, the deck's half width; D5: 0.10;
                                  level rows: 0)
``KRABBY_PAYLOAD_COM_Z=z``        the payload CoM's body-frame z, fixed (``0 <= z <=`` :data:`PAYLOAD_COM_Z_MAX`).
                                  D5: +0.309 m = half the 0.3175 m body height + half the 0.30 m load box: a
                                  0.30 m load resting on the deck, measured from the body link's origin (the
                                  body box centre)
================================  ==================================================================================

The payload is a solid box of :data:`PAYLOAD_BOX_SIZE_M` (the deck footprint 0.7112 x 1.2192 m from
``crab_hex_forward_task/mdp/crab_hex_dimensions.py`` ``BODY_SIZE_M``, by 0.30 m) rigidly added to the ``body``
link: total mass, mass-weighted CoM and parallel-axis inertia, written ABSOLUTELY from each env's captured
post-startup-DR base at every reset (``omni_events.payload_reset``). The box size is a code constant, written
into the term's params so the cfg dump records it.

Gate level rows (PLAN Gates -> Gate 0) always set ``KG=L:L, ZERO_FRAC=0, COM_XY=0, COM_Z=0.309`` (0 kg as
``0:0``): every env then carries exactly L kg at body-frame CoM (0, 0, 0.309), deterministically.

Pure at module scope (no Isaac, no torch), path-loaded and re-exported by ``omni_knobs.py`` the way
``clock_knobs.py`` is (the campaign driver's knob whitelist reads ``dir(omni_knobs)``); the Isaac imports are
lazy, inside :func:`payload_event_cfg`. See ``tests/unit/test_task4_payload.py``.
"""

from __future__ import annotations

import math
import os

# ------------------------------------------------------------------------------ variables
PAYLOAD_KG_VAR = "KRABBY_PAYLOAD_KG"
PAYLOAD_ZERO_FRAC_VAR = "KRABBY_PAYLOAD_ZERO_FRAC"
PAYLOAD_COM_XY_VAR = "KRABBY_PAYLOAD_COM_XY"
PAYLOAD_COM_Z_VAR = "KRABBY_PAYLOAD_COM_Z"
#: Every payload knob, arming knob first.
PAYLOAD_VARS = (PAYLOAD_KG_VAR, PAYLOAD_ZERO_FRAC_VAR, PAYLOAD_COM_XY_VAR, PAYLOAD_COM_Z_VAR)

#: The ``cfg.events`` attribute the armed knobs create (absent while they are off).
PAYLOAD_EVENT = "payload_reset"
#: The articulation and the link the payload is added to.
PAYLOAD_ASSET = "robot"
PAYLOAD_BODY = "body"

#: Payload box (x, y, z) in metres: the deck footprint (``crab_hex_dimensions.BODY_SIZE_M[:2]``: 28 in x 48 in)
#: by a 0.30 m load height. Code constant (D5); the unit tests pin it to the dimensions module.
PAYLOAD_BOX_SIZE_M = (0.7112, 1.2192, 0.30)

#: Typo guards, not physics: 1.85x the plan's top level (270 kg); the deck's half width (a CoM off the deck is not
#: a deck load); a CoM more than 1 m above the body centre is not a deck load either. Raise them here, with a
#: reason, if a campaign needs more.
PAYLOAD_KG_MAX = 500.0
PAYLOAD_COM_XY_MAX = 0.5 * min(PAYLOAD_BOX_SIZE_M[0], PAYLOAD_BOX_SIZE_M[1])
PAYLOAD_COM_Z_MAX = 1.0

__all__ = [
    "PAYLOAD_KG_VAR", "PAYLOAD_ZERO_FRAC_VAR", "PAYLOAD_COM_XY_VAR", "PAYLOAD_COM_Z_VAR", "PAYLOAD_VARS",
    "PAYLOAD_EVENT", "PAYLOAD_ASSET", "PAYLOAD_BODY", "PAYLOAD_BOX_SIZE_M", "PAYLOAD_KG_MAX",
    "PAYLOAD_COM_XY_MAX", "PAYLOAD_COM_Z_MAX",
    "parse_payload_kg", "parse_payload_zero_frac", "parse_payload_com_xy", "parse_payload_com_z",
    "PAYLOAD_KNOB_PARSERS", "parse_payload_knobs", "payload_event_params", "payload_event_cfg",
    "apply_payload_knobs",
]


# ------------------------------------------------------------------------------ parsers
def _blank(text) -> bool:
    return text is None or not str(text).strip()


def _number(text, var: str) -> float:
    try:
        value = float(str(text).strip())
    except ValueError as exc:
        raise ValueError(f"{var} must be a number, got {text!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{var} must be finite, got {text!r}")
    return value


def parse_payload_kg(text) -> tuple[float, float] | None:
    """``KRABBY_PAYLOAD_KG`` -> ``(lo, hi)`` kg with ``0 <= lo <= hi <= PAYLOAD_KG_MAX``; blank -> None (off).

    Exactly ``lo:hi`` (0 kg is ``0:0``, a level is ``L:L``): a bare number is refused rather than read as either a
    level or an upper bound.
    """
    if _blank(text):
        return None
    parts = str(text).strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"{PAYLOAD_KG_VAR} must be 'lo:hi' kg (0 kg = '0:0', a level = 'L:L'), got {text!r}")
    lo, hi = (_number(p, PAYLOAD_KG_VAR) for p in parts)
    if not 0.0 <= lo <= hi <= PAYLOAD_KG_MAX:
        raise ValueError(f"{PAYLOAD_KG_VAR} needs 0 <= lo <= hi <= {PAYLOAD_KG_MAX:g} kg, got {text!r}")
    return (lo + 0.0, hi + 0.0)     # -0.0 -> 0.0


def parse_payload_zero_frac(text) -> float:
    """``KRABBY_PAYLOAD_ZERO_FRAC`` -> the per-reset no-payload probability, ``0 <= p <= 1``. Blank raises."""
    if _blank(text):
        raise ValueError(f"{PAYLOAD_ZERO_FRAC_VAR} is blank")
    p = _number(text, PAYLOAD_ZERO_FRAC_VAR)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{PAYLOAD_ZERO_FRAC_VAR} must be in [0, 1], got {text!r}")
    return p + 0.0


def parse_payload_com_xy(text) -> float:
    """``KRABBY_PAYLOAD_COM_XY`` -> r m, the per-axis U[-r, r] CoM shift, ``0 <= r <= PAYLOAD_COM_XY_MAX``."""
    if _blank(text):
        raise ValueError(f"{PAYLOAD_COM_XY_VAR} is blank")
    r = _number(text, PAYLOAD_COM_XY_VAR)
    if not 0.0 <= r <= PAYLOAD_COM_XY_MAX:
        raise ValueError(f"{PAYLOAD_COM_XY_VAR} must be in [0, {PAYLOAD_COM_XY_MAX:g}] m (the deck half width), "
                         f"got {text!r}")
    return r + 0.0


def parse_payload_com_z(text) -> float:
    """``KRABBY_PAYLOAD_COM_Z`` -> z m above the body link origin, ``0 <= z <= PAYLOAD_COM_Z_MAX``."""
    if _blank(text):
        raise ValueError(f"{PAYLOAD_COM_Z_VAR} is blank")
    z = _number(text, PAYLOAD_COM_Z_VAR)
    if not 0.0 <= z <= PAYLOAD_COM_Z_MAX:
        raise ValueError(f"{PAYLOAD_COM_Z_VAR} must be in [0, {PAYLOAD_COM_Z_MAX:g}] m, got {text!r}")
    return z + 0.0


#: ``{variable: parser}`` for the four knobs (the driver's overlay round-trip table, ``run_candidates.knob_parsers``).
PAYLOAD_KNOB_PARSERS = {
    PAYLOAD_KG_VAR: parse_payload_kg,
    PAYLOAD_ZERO_FRAC_VAR: parse_payload_zero_frac,
    PAYLOAD_COM_XY_VAR: parse_payload_com_xy,
    PAYLOAD_COM_Z_VAR: parse_payload_com_z,
}


def parse_payload_knobs(environ=None) -> dict | None:
    """All four knobs -> ``{"kg_range", "zero_frac", "com_xy", "com_z"}``, or None while the payload is off.

    Raises ``ValueError`` for a malformed value, for a companion knob set without ``KRABBY_PAYLOAD_KG``, and for
    ``KRABBY_PAYLOAD_KG`` set without all three companions.
    """
    env = os.environ if environ is None else environ
    kg = parse_payload_kg(env.get(PAYLOAD_KG_VAR))
    others = (PAYLOAD_ZERO_FRAC_VAR, PAYLOAD_COM_XY_VAR, PAYLOAD_COM_Z_VAR)
    if kg is None:
        stray = [v for v in others if not _blank(env.get(v))]
        if stray:
            raise ValueError(f"{', '.join(stray)} set but {PAYLOAD_KG_VAR} is not: the payload is armed by "
                             f"{PAYLOAD_KG_VAR} alone, so these would silently do nothing")
        return None
    missing = [v for v in others if _blank(env.get(v))]
    if missing:
        raise ValueError(f"{PAYLOAD_KG_VAR}={env.get(PAYLOAD_KG_VAR)!r} needs {', '.join(missing)} too: an armed "
                         "payload states its whole geometry (D5 training: ZERO_FRAC 0.25, COM_XY 0.10, COM_Z 0.309; "
                         "a level row: 0, 0, 0.309)")
    return {
        "kg_range": kg,
        "zero_frac": parse_payload_zero_frac(env.get(PAYLOAD_ZERO_FRAC_VAR)),
        "com_xy": parse_payload_com_xy(env.get(PAYLOAD_COM_XY_VAR)),
        "com_z": parse_payload_com_z(env.get(PAYLOAD_COM_Z_VAR)),
    }


# ------------------------------------------------------------------------------ application
def payload_event_params(spec: dict, asset_cfg) -> dict:
    """The ``payload_reset`` term params for a parsed ``spec`` (every keyword of the term's ``__call__``)."""
    return {
        "asset_cfg": asset_cfg,
        "kg_range": tuple(spec["kg_range"]),
        "zero_frac": float(spec["zero_frac"]),
        "com_xy": float(spec["com_xy"]),
        "com_z": float(spec["com_z"]),
        "box_size": tuple(PAYLOAD_BOX_SIZE_M),
    }


def payload_event_cfg(spec: dict):
    """The ``EventTermCfg`` of the armed payload (Isaac imports are lazy: this module stays pure)."""
    from isaaclab.managers import EventTermCfg, SceneEntityCfg  # noqa: PLC0415

    from parkour_tasks.crab_hex_omni_task.mdp.omni_events import payload_reset  # noqa: PLC0415

    return EventTermCfg(
        func=payload_reset,
        mode="reset",
        params=payload_event_params(spec, SceneEntityCfg(PAYLOAD_ASSET, body_names=PAYLOAD_BODY)),
    )


def apply_payload_knobs(cfg, environ=None) -> None:
    """Arm ``cfg.events.payload_reset`` from the four knobs; writes nothing while ``KRABBY_PAYLOAD_KG`` is off.

    Called by ``omni_knobs.apply_omni_knobs`` (before the command-frame swap). Idempotent: a second call replaces the
    term. Raises ``ValueError`` for a malformed or incomplete knob set and ``RuntimeError`` for a cfg with no
    ``events`` block.
    """
    spec = parse_payload_knobs(environ)
    if spec is None:
        return
    events = getattr(cfg, "events", None)
    if events is None:
        raise RuntimeError(f"{PAYLOAD_KG_VAR} is set but the env cfg has no events block to arm")
    setattr(events, PAYLOAD_EVENT, payload_event_cfg(spec))
    lo, hi = spec["kg_range"]
    print(f"[KNOB] {PAYLOAD_KG_VAR}={lo:g}:{hi:g} kg zero_frac={spec['zero_frac']:g} com_xy=+-{spec['com_xy']:g} m "
          f"com_z={spec['com_z']:g} m -> events.{PAYLOAD_EVENT} (box {PAYLOAD_BOX_SIZE_M} m, absolute composite "
          "on the post-DR base at every reset)", flush=True)
