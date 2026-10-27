# SPDX-License-Identifier: BSD-3-Clause
"""Clock-hide knobs (clock-deprecation plan, step 2, 2026-09-23) -> the omni observation term.

BOTH KNOBS ARE DEFAULT OFF AND SET BY NO PRESET. Unset, blank or at ``"0"`` they write NOTHING
into the cfg -- no param, no field -- so a knob-off config dump is byte-identical to one built
before this module existed, and an eval that carries ``KRABBY_OMNI_CLOCK_HIDE=0`` builds the same
cfg as one that omits it.

================================  ==============================================================
variable                          what it writes when armed
================================  ==============================================================
``KRABBY_OMNI_CLOCK_HIDE=1``      omni obs term ``params["clock_hide"] = True`` -- the gait
                                  clock's sin/cos (obs dims 15-16, hence every history copy)
                                  read (0, 0). WIDTH-PRESERVING: a head trained with the clock
                                  visible loads unchanged (1149 wide), which is the point
``KRABBY_OMNI_CLOCK_HIDE_RAMP``   ``params["clock_hide_ramp"] = (t0, t1)``, a TUPLE of env steps
``=t0:t1``                        (24 per iteration, the counter restarts in every process): a
                                  per-EPISODE mask, latched at reset with probability
                                  ``p(t)`` = cosine 0 -> 1 over ``[t0, t1]``, every row hidden
                                  from ``t1`` on. REFUSED unless ``KRABBY_OMNI_CLOCK_HIDE=1``
================================  ==============================================================

WHY "HIDE" AND NOT "DROP". Deleting dims 15-16 changes the observation width to 1127, which only
a from-scratch run can train; the campaign forks formed heads at the C1d stage boundaries, so the
channels stay and carry a constant. Zero is the deployment value (a robot with no runtime clock)
and it is mirror-invariant (both planes map 0 to 0 under any sign), so
``crab_hex_omni_mirror.build_mirrors`` needs no change. The history copies and the critic follow
automatically: ``obs_buf`` feeds the history buffer and the critic has no separate group.

WHY THE RAMP IS REFUSED WITHOUT HIDE. The ramp only says WHEN rows are hidden; without the hide
flag the observation term would never read it, so a launcher that set the ramp alone would train
the full clock while its recipe claimed a removal schedule.

The ramp is never carried into an eval (``run_phases.EVAL_CARRY``): the eval process's counter
starts at 0, so it would evaluate the head with the clock visible on every episode.

Pure at module scope (no Isaac, no torch), path-loaded and re-exported by ``omni_knobs.py`` the
way ``motion_knobs.py`` is -- see ``tests/unit/test_crab_hex_omni_clock_hide.py``.
"""

from __future__ import annotations

import os

# ------------------------------------------------------------------------------ variables
CLOCK_HIDE_VAR = "KRABBY_OMNI_CLOCK_HIDE"
CLOCK_HIDE_RAMP_VAR = "KRABBY_OMNI_CLOCK_HIDE_RAMP"

#: The omni obs term params the two knobs write (read by ``CrabHexOmniParkourObservations``).
CLOCK_HIDE_PARAM = "clock_hide"
CLOCK_HIDE_RAMP_PARAM = "clock_hide_ramp"

#: The env-side knob that changes observation SEMANTICS (an eval of a head trained with it must
#: be built with it too) -- the ramp is deliberately NOT listed (see the module docstring).
EVAL_SEMANTIC_KNOBS = (CLOCK_HIDE_VAR,)
#: ... and the value it takes when OFF.
KNOB_OFF_VALUES = {CLOCK_HIDE_VAR: "0"}

__all__ = [
    "CLOCK_HIDE_VAR", "CLOCK_HIDE_RAMP_VAR", "CLOCK_HIDE_PARAM", "CLOCK_HIDE_RAMP_PARAM",
    "EVAL_SEMANTIC_KNOBS", "KNOB_OFF_VALUES", "parse_clock_hide", "parse_clock_hide_ramp",
    "CLOCK_KNOB_PARSERS", "apply_clock_knobs",
]


# ------------------------------------------------------------------------------ parsers
def _blank(text) -> bool:
    return text is None or not str(text).strip()


def parse_clock_hide(text) -> bool:
    """``KRABBY_OMNI_CLOCK_HIDE`` -> True (``"1"``) / False (``"0"`` or blank); anything else raises.

    Not a permissive truthiness parser: a ``"true"`` silently read as false would train the clock
    the recipe claims to have removed.
    """
    if _blank(text):
        return False
    token = str(text).strip()
    if token == "1":
        return True
    if token == "0":
        return False
    raise ValueError(f"{CLOCK_HIDE_VAR} must be '0' or '1', got {text!r}")


def parse_clock_hide_ramp(text) -> tuple[int, int] | None:
    """``KRABBY_OMNI_CLOCK_HIDE_RAMP`` -> ``(t0, t1)`` env steps, ``0 <= t0 < t1``; blank -> None.

    Integers only (env steps are counted, not measured); a ``t1 <= t0`` window is refused here
    rather than by ``curriculums.ramp_value`` on the first reset of training. The ladder's own
    rules (multiples of 24, inside the stage, after the reward PHASEOUT) live in its validator.
    """
    if _blank(text):
        return None
    parts = str(text).strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"{CLOCK_HIDE_RAMP_VAR} must be 't0:t1' (env steps), got {text!r}")
    try:
        t0, t1 = (int(p.strip()) for p in parts)
    except ValueError as exc:
        raise ValueError(f"{CLOCK_HIDE_RAMP_VAR} must be two integers 't0:t1' (env steps), "
                         f"got {text!r}") from exc
    if not 0 <= t0 < t1:
        raise ValueError(f"{CLOCK_HIDE_RAMP_VAR} needs 0 <= t0 < t1, got {text!r}")
    return (t0, t1)


#: ``{variable: parser}`` for both knobs (the ladder's round-trip check).
CLOCK_KNOB_PARSERS = {
    CLOCK_HIDE_VAR: parse_clock_hide,
    CLOCK_HIDE_RAMP_VAR: parse_clock_hide_ramp,
}


# ------------------------------------------------------------------------------ application
def _omni_obs_terms(cfg):
    """Every observation term whose func is the omni observation class (by attribute).

    The same probe as ``motion_knobs._omni_obs_terms`` (kept local so this module stays a pure,
    self-contained path-load).
    """
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


def apply_clock_knobs(cfg, environ=None) -> None:
    """Apply both clock-hide knobs to a built omni env cfg (writes nothing while they are off).

    Called by ``omni_knobs.apply_omni_knobs``. Raises ``ValueError`` for a ramp without the hide
    flag and ``RuntimeError`` if the cfg carries no omni observation term to arm.
    """
    env = os.environ if environ is None else environ
    hide = parse_clock_hide(env.get(CLOCK_HIDE_VAR))
    ramp = parse_clock_hide_ramp(env.get(CLOCK_HIDE_RAMP_VAR))
    if ramp is not None and not hide:
        raise ValueError(f"{CLOCK_HIDE_RAMP_VAR} is set but {CLOCK_HIDE_VAR} is not '1': the ramp "
                         "only schedules WHEN the hidden clock applies, so alone it would train the "
                         "full clock while the recipe claimed a removal")
    if not hide:
        return
    terms = _omni_obs_terms(cfg)
    if not terms:
        raise RuntimeError(f"{CLOCK_HIDE_VAR} is set but no omni observation term is installed")
    for _g, _t, term in terms:
        if term.params is None:
            term.params = {}
        term.params[CLOCK_HIDE_PARAM] = True
        if ramp is not None:
            term.params[CLOCK_HIDE_RAMP_PARAM] = ramp
