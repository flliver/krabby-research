# SPDX-License-Identifier: BSD-3-Clause
"""Training-time domain-randomisation knobs (rod-speed DR campaign, 2026-11-07) -> the omni env cfg.

EVERY KNOB IS DEFAULT OFF AND SET BY NO PRESET. Unset or blank, a knob writes NOTHING into the cfg and prints nothing,
so a knob-off config dump is byte-identical to one built before this module existed.

=========================================  ==========================================================================
variable                                   effect
=========================================  ==========================================================================
``KRABBY_DR_ROD_SPEED=lo:hi[:ramp_steps]``  ``actions.joint_pos.rod_speed_dr = {lo, hi, ramp_steps}``: the action term
                                           draws a per-env rod slew factor ``s ~ U(lo_eff, hi)`` at each env reset
                                           (``crab_hex_forward_task/mdp/rod_speed_dr.py``; hip and knee share it);
                                           ``lo_eff`` ramps 1.0 -> ``lo`` over ``ramp_steps`` env steps (default 0 =
                                           no ramp). :data:`ROD_SPEED_LO_MIN` <= lo <= hi <= :data:`ROD_SPEED_HI_MAX`;
                                           with a ramp, lo <= 1 <= hi. The current proxy keeps the rated speed
=========================================  ==========================================================================

Refused together with the stress sweep's fixed rod-speed scale -- the caller (``omni_knobs.apply_omni_knobs``) passes
that variable's name in ``exclusive_with`` (config modules may not spell a stress variable), and the action term
refuses the pair again. Any OTHER ``KRABBY_DR_ROD*`` name in the environment is refused (a typo would otherwise train
at nominal); the forward task's ``KRABBY_DR_PUSH`` / ``_MASS`` / ``_COM`` are not this module's and are left alone.
Pure at module scope (no Isaac, no torch), path-loaded and re-exported by ``omni_knobs.py`` the way
``stress_knobs.py`` is.
"""

from __future__ import annotations

import math
import os

# ------------------------------------------------------------------------------ variables
ROD_SPEED_VAR = "KRABBY_DR_ROD_SPEED"
#: Every DR knob.
DR_VARS = (ROD_SPEED_VAR,)
#: The prefix whose look-alikes are refused (a misspelled rod knob).
_ROD_PREFIX = "KRABBY_DR_ROD"

#: The action-cfg attribute the rod-speed knob writes (``crab_hex_forward_task/mdp/rod_speed_dr.ROD_SPEED_DR_ATTR``).
ROD_SPEED_DR_ATTR = "rod_speed_dr"
#: The band's floor: below ~0.3 the current-proxy drive gate saturates (CHARTER 3.1), and its typo-guard ceiling.
ROD_SPEED_LO_MIN = 0.3
ROD_SPEED_HI_MAX = 2.0

__all__ = [
    "ROD_SPEED_VAR", "DR_VARS", "ROD_SPEED_DR_ATTR", "ROD_SPEED_LO_MIN", "ROD_SPEED_HI_MAX",
    "parse_rod_speed_dr", "DR_KNOB_PARSERS", "parse_dr_knobs", "apply_dr_knobs",
]


# ------------------------------------------------------------------------------ parsers
def _blank(text) -> bool:
    return text is None or not str(text).strip()


def _number(text, var: str, what: str) -> float:
    try:
        value = float(str(text).strip())
    except ValueError as exc:
        raise ValueError(f"{var}: {what} must be a number, got {text!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{var}: {what} must be finite, got {text!r}")
    return value


def parse_rod_speed_dr(text) -> dict | None:
    """``lo:hi`` or ``lo:hi:ramp_steps`` -> ``{"lo", "hi", "ramp_steps"}`` (plain floats / int); blank = off (None)."""
    var = ROD_SPEED_VAR
    if _blank(text):
        return None
    parts = str(text).strip().split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"{var} must be lo:hi or lo:hi:ramp_steps, got {text!r}")
    lo = _number(parts[0], var, "lo") + 0.0
    hi = _number(parts[1], var, "hi") + 0.0
    ramp = 0
    if len(parts) == 3:
        r = _number(parts[2], var, "ramp_steps")
        if r != int(r) or int(r) < 0:
            raise ValueError(f"{var}: ramp_steps must be an integer >= 0, got {parts[2]!r}")
        ramp = int(r)
    if not ROD_SPEED_LO_MIN <= lo <= hi <= ROD_SPEED_HI_MAX:
        raise ValueError(f"{var} needs {ROD_SPEED_LO_MIN:g} <= lo <= hi <= {ROD_SPEED_HI_MAX:g}, got {text!r}")
    if ramp > 0 and not lo <= 1.0 <= hi:
        raise ValueError(f"{var}: a ramp starts at the nominal 1.0, so it needs lo <= 1 <= hi, got {text!r}")
    return {"lo": lo, "hi": hi, "ramp_steps": ramp}


#: ``{variable: parser}`` (the driver's overlay round-trip table, ``run_candidates.knob_parsers``).
DR_KNOB_PARSERS = {
    ROD_SPEED_VAR: parse_rod_speed_dr,
}


def parse_dr_knobs(environ=None) -> dict:
    """``{variable: value}`` of the ARMED knobs only (empty while all are off). Raises ``ValueError`` for a
    malformed value and for an unknown ``KRABBY_DR_ROD*`` name."""
    env = os.environ if environ is None else environ
    unknown = sorted(k for k in env if str(k).startswith(_ROD_PREFIX) and k not in DR_KNOB_PARSERS)
    if unknown:
        raise ValueError(f"unknown DR knob(s) {unknown}: a misspelled knob would train at nominal "
                         f"(known: {list(DR_VARS)})")
    armed = {}
    for var, parser in DR_KNOB_PARSERS.items():
        value = parser(env.get(var))
        if value is not None:
            armed[var] = value
    return armed


# ------------------------------------------------------------------------------ application
def apply_dr_knobs(cfg, environ=None, *, exclusive_with=()) -> dict:
    """Apply every armed DR knob to a built omni env cfg; writes nothing and prints nothing while all are off.

    Called by ``omni_knobs.apply_omni_knobs`` right after the stress knobs. ``exclusive_with`` names environment
    variables that may not be set together with an armed DR knob (the stress rod-speed scale). Returns the armed
    ``{variable: value}``.
    """
    env = os.environ if environ is None else environ
    armed = parse_dr_knobs(env)
    if not armed:
        return armed
    spec = armed.get(ROD_SPEED_VAR)
    if spec is not None:
        clash = [v for v in exclusive_with if not _blank(env.get(v))]
        if clash:
            raise ValueError(f"{ROD_SPEED_VAR} and {clash} are exclusive: a per-env DR draw on top of a fixed "
                             "rod-speed scale is not one dose")
        action = getattr(getattr(cfg, "actions", None), "joint_pos", None)
        if action is None:
            raise RuntimeError(f"{ROD_SPEED_VAR} is set but the env cfg has no actions.joint_pos")
        if getattr(action, "stress_rod_speed_scale", None) is not None:
            raise ValueError(f"{ROD_SPEED_VAR} is set but actions.joint_pos already carries stress_rod_speed_scale "
                             "(the stress rod-speed knob): exclusive")
        value = {"lo": float(spec["lo"]), "hi": float(spec["hi"]), "ramp_steps": int(spec["ramp_steps"])}
        setattr(action, ROD_SPEED_DR_ATTR, value)
        ramp = (f", lo ramps 1 -> {value['lo']:g} over {value['ramp_steps']} env steps" if value["ramp_steps"]
                else ", no ramp")
        print(f"[KNOB] {ROD_SPEED_VAR}={str(env.get(ROD_SPEED_VAR)).strip()} -> actions.joint_pos.{ROD_SPEED_DR_ATTR}"
              f" = {value} (per-env rod slew x U({value['lo']:g}, {value['hi']:g}) redrawn at each env reset{ramp})",
              flush=True)
    return armed
