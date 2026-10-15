# SPDX-License-Identifier: BSD-3-Clause
"""Body-frame command knobs (robot-frame velocity commands campaign, step 1, 2026-10-15).

``experiments/2026-10-15_body_cmd`` (PLAN D1 / D3, ``STEP1_SPEC.md`` §1). ALL THREE ARE DEFAULT
OFF. Unset or blank they write NOTHING into the cfg -- no field, no param, no attribute -- so a
knob-off config dump is byte-identical to one built before this module existed (the inertness
fixture compares 13 Kit dumps before / after).

=================================  ==============================================================
variable                           meaning
=================================  ==============================================================
``KRABBY_OMNI_COMMAND_FRAME``      ``world`` (= unset / blank: the shipped world travel direction +
                                   world facing, bitwise) or ``body`` (a held robot-frame
                                   ``(vx, vy, wz)`` per slot); anything else is refused
``KRABBY_OMNI_BODY_ENVELOPE``      ``w_min:ceiling:w_max`` rad/s, ``0 < w_min < ceiling <= w_max``
                                   -- REQUIRED in body mode, REFUSED in world mode
``KRABBY_OMNI_BODY_CAP``           ``v0@wz0,v1@wz1,...`` -- the coupled move+turn cap (m/s @ rad/s):
                                   ``v0 == 0``, ``v`` strictly increasing, ``wz`` non-increasing,
                                   every ``wz`` in ``(0, w_max]`` -- REQUIRED in body mode, REFUSED
                                   in world mode
=================================  ==============================================================

The envelope is MEASURED (PLAN D3: step 4 freezes it in the CHARTER), so it is a knob / manifest
parameter and never a code constant; the ``_b`` presets deliberately do not carry it, so a ``_b``
preset alone refuses to build until the campaign overlay supplies it.

THE GRAMMAR LIVES IN ``mdp/crab_hex_omni_command_math.py`` (``parse_body_envelope`` and its two
halves), loaded lazily BY PATH here -- the ``omni_knobs._command_math`` pattern, sharing its module
object -- so this module stays pure (no Isaac, and torch only when a body knob is actually parsed).
What is applied, where and with which refusals is ``omni_knobs._apply_command_frame`` (the one
documented point, after every other knob), which reads this module's names.

Path-loaded and re-exported by ``omni_knobs.py`` exactly like ``motion_knobs.py`` /
``clock_knobs.py``, so the candidates driver's whitelist (``dir(omni_knobs)``'s ``*_VAR`` names) and
its parser table see the three knobs. Unit-tested in ``tests/unit/test_crab_hex_omni_body_cmd.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# ------------------------------------------------------------------------------ variables
COMMAND_FRAME_VAR = "KRABBY_OMNI_COMMAND_FRAME"
BODY_ENVELOPE_VAR = "KRABBY_OMNI_BODY_ENVELOPE"
BODY_CAP_VAR = "KRABBY_OMNI_BODY_CAP"

FRAME_WORLD = "world"
FRAME_BODY = "body"
#: The only two frames :func:`parse_command_frame` accepts (blank = ``world``).
COMMAND_FRAMES = (FRAME_WORLD, FRAME_BODY)

#: The env-side knob that changes the COMMAND and OBSERVATION semantics (a head trained in one frame
#: must be evaluated in it, or through a named adapter -- PLAN D2 / D6) and its off value. The eval
#: harness's carry is step 2's; this is the vocabulary it will read.
EVAL_SEMANTIC_KNOBS = (COMMAND_FRAME_VAR,)
KNOB_OFF_VALUES = {COMMAND_FRAME_VAR: FRAME_WORLD}

__all__ = [
    "COMMAND_FRAME_VAR", "BODY_ENVELOPE_VAR", "BODY_CAP_VAR", "FRAME_WORLD", "FRAME_BODY",
    "COMMAND_FRAMES", "EVAL_SEMANTIC_KNOBS", "KNOB_OFF_VALUES", "parse_command_frame",
    "parse_body_envelope_knob", "parse_body_cap_knob", "parse_body_knobs", "BODY_KNOB_PARSERS",
]

# ------------------------------------------------------------------------------ the pure math
_CMD_MATH_PY = Path(__file__).resolve().parents[2] / "mdp" / "crab_hex_omni_command_math.py"
#: The SAME ``sys.modules`` key ``omni_knobs._command_math`` uses, so the two path-loaded validators
#: share one module object.
_CMD_MATH_NAME = "_crab_hex_omni_command_math__knobs"
_REAL_MATH_NAME = "parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_command_math"


def _command_math():
    """The pure command math (``parse_body_envelope``), Isaac-free; the real module inside Isaac."""
    mod = sys.modules.get(_REAL_MATH_NAME) or sys.modules.get(_CMD_MATH_NAME)
    if mod is None:
        spec = importlib.util.spec_from_file_location(_CMD_MATH_NAME, _CMD_MATH_PY)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[_CMD_MATH_NAME] = mod
        spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------ parsers
def _blank(text) -> bool:
    return text is None or not str(text).strip()


def parse_command_frame(text) -> str:
    """``KRABBY_OMNI_COMMAND_FRAME`` -> ``"world"`` (blank / unset / ``world``) or ``"body"``.

    Case- and space-tolerant, otherwise strict: a ``"robot"`` or ``"1"`` silently read as world would
    train the facing lineage while the recipe claimed the body mode.
    """
    if _blank(text):
        return FRAME_WORLD
    token = str(text).strip().lower()
    if token not in COMMAND_FRAMES:
        raise ValueError(f"{COMMAND_FRAME_VAR} must be one of {COMMAND_FRAMES} (blank = world), got {text!r}")
    return token


def parse_body_envelope_knob(text) -> tuple[float, float, float] | None:
    """``KRABBY_OMNI_BODY_ENVELOPE`` alone -> ``(w_min, ceiling, w_max)``, or None when blank.

    The single-knob half of the grammar (the driver's round-trip table); the cross-knob rule
    against the cap is :func:`parse_body_knobs`'.
    """
    if _blank(text):
        return None
    return _command_math().parse_body_envelope_triple(str(text))


def parse_body_cap_knob(text) -> tuple[tuple[float, float], ...] | None:
    """``KRABBY_OMNI_BODY_CAP`` alone -> ``((v, wz), ...)``, or None when blank (shape rules only)."""
    if _blank(text):
        return None
    return _command_math().parse_body_cap_points(str(text))


def parse_body_knobs(environ) -> tuple[str, object]:
    """``(frame, envelope)`` for an env mapping -- ``envelope`` a ``BodyEnvelope`` in body mode, else None.

    Body mode REQUIRES both envelope knobs (and validates them together, ``wz <= w_max``); world mode
    REFUSES either. ``ValueError`` names the knob and the value.
    """
    frame = parse_command_frame(environ.get(COMMAND_FRAME_VAR))
    env_raw, cap_raw = environ.get(BODY_ENVELOPE_VAR), environ.get(BODY_CAP_VAR)
    if frame == FRAME_WORLD:
        for var, raw in ((BODY_ENVELOPE_VAR, env_raw), (BODY_CAP_VAR, cap_raw)):
            if not _blank(raw):
                raise ValueError(f"{var}={raw!r} is set but {COMMAND_FRAME_VAR} is world (or unset): the "
                                 "body envelope has no meaning for the world command")
        return frame, None
    missing = [var for var, raw in ((BODY_ENVELOPE_VAR, env_raw), (BODY_CAP_VAR, cap_raw)) if _blank(raw)]
    if missing:
        raise ValueError(f"{COMMAND_FRAME_VAR}=body REQUIRES {missing}: the command range is the robot's "
                         "MEASURED envelope (PLAN D3), never a default")
    return frame, _command_math().parse_body_envelope(str(env_raw), str(cap_raw))


#: ``{variable: parser}`` for the three knobs (the candidates driver's round-trip table).
BODY_KNOB_PARSERS = {
    COMMAND_FRAME_VAR: parse_command_frame,
    BODY_ENVELOPE_VAR: parse_body_envelope_knob,
    BODY_CAP_VAR: parse_body_cap_knob,
}
