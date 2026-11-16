# SPDX-License-Identifier: BSD-3-Clause
"""Fault-injection knobs (fault-injection sweep, 2026-11-15) -> ONE mid-episode fault of a frozen policy. DEFAULT OFF.

EVERY KNOB IS DEFAULT OFF AND SET BY NO PRESET. With none of the three variables set, nothing is written into the cfg
-- no attribute, no print -- so a knob-off config dump is byte-identical to one built before this module existed
(the stress-knob contract, ``stress_knobs.py``). The sweep (``experiments/2026-11-15_fault_injection``, CHARTER §3)
arms one cell per launch:

==================================  ===================================================================================
variable                            meaning
==================================  ===================================================================================
``KRABBY_FAULT=<id>:<level>``       the cell, e.g. ``S1:0.5``, ``S1:perm``, ``A1:mid``, ``B2:0.2`` (:data:`FAULTS`)
``KRABBY_FAULT_ONSET=<lo>:<hi>``    the onset window, seconds from each episode's step 0; per (wave, env) the onset is
                                    drawn once from U(lo, hi) on a private generator (``mdp/fault_inject.py``)
``KRABBY_FAULT_FROM_START=1``       the S12 / D1 anchors only (required there, refused elsewhere): active from step 0;
                                    the onset table is still drawn and recorded
==================================  ===================================================================================

Armed, :func:`apply_fault_knobs` writes ONE plain dict, ``cfg.fault_inject`` (:data:`FAULT_ATTR`), which the action
term (``crab_hex_forward_task/mdp/parkour_actions.py``) turns into the per-env injector shared with the observation
and depth terms. Refused (``ValueError``): an unknown id or level; a malformed value; any other ``KRABBY_FAULT*``
name or a value naming two cells (one fault per cell); an armed fault with no onset window; an onset / from-start
knob without a fault; ``KRABBY_FAULT_FROM_START`` on a fault that is not an anchor (or missing on one); and any
variable of ``exclusive_with`` (the caller passes the stress sweep's ``STRESS_VARS``: config modules may not spell a
stress variable) set together with a fault. Refused (``RuntimeError``): a cfg that cannot carry the fault (a depth
fault on an env without the four depth groups, a delay fault without ``use_delay`` / a 4-deep action history, A5
without the 90 kg payload event).

Pure at module scope (no Isaac, no torch), path-loaded and re-exported by ``omni_knobs.py`` the way
``stress_knobs.py`` is.
"""

from __future__ import annotations

import math
import os

# ------------------------------------------------------------------------------ variables
FAULT_VAR = "KRABBY_FAULT"
ONSET_VAR = "KRABBY_FAULT_ONSET"
FROM_START_VAR = "KRABBY_FAULT_FROM_START"
#: Every fault knob.
FAULT_VARS = (FAULT_VAR, ONSET_VAR, FROM_START_VAR)
#: The env-cfg attribute the knobs write (``crab_hex_forward_task/mdp/fault_inject.FAULT_ATTR``).
FAULT_ATTR = "fault_inject"
#: The depth groups a depth fault needs (``omni_student_spec.DEPTH_GROUPS``) and their term.
DEPTH_GROUPS = ("depth_front", "depth_left", "depth_right", "depth_rear")
DEPTH_TERM = "depth_image"
#: Typo guard on the onset window (s); an "armed but never" row puts the window beyond the episode.
ONSET_MAX_S = 3600.0
#: A5's payload (CHARTER §3.5: A1 mid at L090 = 90 kg, the F19 ``L090.env`` level file).
A5_PAYLOAD_KG = 90.0
#: S4's per-axis random-walk step (CHARTER §3.3: 0.2 / sqrt(1000) m/s per step, ~0.2 m/s RMS at 10 s).
S4_SIGMA = 0.2 / math.sqrt(1000.0)

LEGS = {"front": "FL", "mid": "ML", "rear": "RL"}   # LEFT side only (CHARTER §3.5)

#: The catalog (CHARTER §3.3-§3.6). ``levels``: token -> (duration, params). Duration: ``("s", x)`` a transient of x
#: seconds, ``("steps", n)`` a transient of n policy steps, ``None`` permanent.
FAULTS = {
    "S1": {"family": "sensor", "levels": {"0.1": (("s", 0.1), {}), "0.5": (("s", 0.5), {}), "2": (("s", 2.0), {}),
                                          "perm": (None, {})}},
    "S2": {"family": "sensor", "levels": {"0.5": (("s", 0.5), {}), "2": (("s", 2.0), {}), "perm": (None, {})}},
    "S3": {"family": "sensor", "levels": {"0.1": (None, {"bias": 0.1}), "0.2": (None, {"bias": 0.2})}},
    "S4": {"family": "sensor", "levels": {"perm": (None, {"sigma": S4_SIGMA})}},
    "S5": {"family": "sensor", "levels": {"0.1": (("s", 0.1), {}), "0.5": (("s", 0.5), {}), "2": (("s", 2.0), {})}},
    "S6": {"family": "sensor", "levels": {"0.05": (None, {"bias": 0.05}), "0.1": (None, {"bias": 0.1})}},
    "S8": {"family": "sensor", "levels": {"zero": (None, {"mode": "zero"}), "held": (None, {"mode": "held"})}},
    "S10": {"family": "depth", "levels": {"0.5": (("s", 0.5), {}), "2": (("s", 2.0), {}), "perm": (None, {})}},
    "S11": {"family": "depth", "levels": {"2": (("s", 2.0), {}), "perm": (None, {})}},
    "Z1": {"family": "compound", "levels": {"0.5": (("s", 0.5), {}), "2": (("s", 2.0), {}), "perm": (None, {})}},
    "B1": {"family": "action", "levels": {"2": (("steps", 2), {}), "5": (("steps", 5), {}), "10": (("steps", 10), {}),
                                          "25": (("steps", 25), {})}},
    "B2": {"family": "action", "levels": {"0.05": (None, {"p": 0.05}), "0.2": (None, {"p": 0.2})}},
    "B3": {"family": "action", "levels": {"perm": (None, {"delays": [1, 2, 3]})}},
    "B4": {"family": "action", "levels": {"0.1": (("s", 0.1), {}), "0.5": (("s", 0.5), {})}},
    "B5": {"family": "action", "levels": {"perm": (None, {"delay": 2})}},
    "B6": {"family": "action", "levels": {"0.1": (("s", 0.1), {}), "0.5": (("s", 0.5), {})}},
    "A1": {"family": "actuator", "levels": {k: (None, {"leg": v}) for k, v in LEGS.items()}},
    "A2": {"family": "actuator", "levels": {k: (None, {"leg": v}) for k, v in LEGS.items()}},
    "A3": {"family": "actuator", "levels": {"mid": (None, {"leg": "ML", "slew_scale": 0.3})}},
    "A4": {"family": "actuator", "levels": {"mid": (None, {"leg": "ML"})}},
    "A5": {"family": "actuator", "levels": {"mid": (None, {"leg": "ML", "as": "A1", "payload_kg": A5_PAYLOAD_KG})}},
    "S12": {"family": "anchor", "levels": {"30": (None, {"hz": 30})}, "from_start": True},
    "D1": {"family": "anchor", "levels": {"1": (None, {})}, "from_start": True},
}
#: The faults that need the four depth groups (a student env).
DEPTH_FAULTS = ("S10", "S11", "Z1")
#: The faults that read the action history at another delay.
DELAY_FAULTS = ("B3", "B5")
MIN_HISTORY_FOR_DELAY = 4

__all__ = [
    "FAULT_VAR", "ONSET_VAR", "FROM_START_VAR", "FAULT_VARS", "FAULT_ATTR", "FAULTS", "LEGS", "S4_SIGMA",
    "ONSET_MAX_S", "A5_PAYLOAD_KG", "DEPTH_FAULTS", "DELAY_FAULTS", "FAULT_KNOB_PARSERS", "parse_fault_cell",
    "parse_fault_onset", "parse_fault_knobs", "apply_fault_knobs",
]


# ------------------------------------------------------------------------------ parsers
def _blank(text) -> bool:
    return text is None or not str(text).strip()


def _number(text, what: str) -> float:
    try:
        value = float(str(text).strip())
    except ValueError as exc:
        raise ValueError(f"{what} must be a number, got {text!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{what} must be finite, got {text!r}")
    return value


def _level_token(fid: str, raw: str) -> str:
    """The catalog token a level spells (numeric levels compare as numbers: ``2`` == ``2.0``)."""
    levels = FAULTS[fid]["levels"]
    t = raw.strip().lower()
    if t in levels:
        return t
    try:
        v = float(t)
    except ValueError:
        v = None
    if v is not None and math.isfinite(v):
        for tok in levels:
            try:
                if abs(float(tok) - v) <= 1e-12:
                    return tok
            except ValueError:
                continue
    raise ValueError(f"{FAULT_VAR}: unknown level {raw!r} for {fid} (known: {list(levels)})")


def parse_fault_cell(text) -> dict | None:
    """``"<id>:<level>"`` -> ``{"id", "level"}`` (blank = off). One cell only: ``S1:0.5,S2:2`` is refused."""
    if _blank(text):
        return None
    s = str(text).strip()
    if any(c in s for c in ",; \t+&|") or s.count(":") != 1:
        raise ValueError(f"{FAULT_VAR} must be ONE '<id>:<level>' cell, got {text!r}")
    fid, level = s.split(":")
    fid = fid.strip().upper()
    if fid not in FAULTS:
        raise ValueError(f"{FAULT_VAR}: unknown fault id {fid!r} (known: {list(FAULTS)})")
    return {"id": fid, "level": _level_token(fid, level)}


def parse_fault_onset(text) -> tuple[float, float] | None:
    """``"<lo>:<hi>"`` seconds, 0 <= lo < hi <= :data:`ONSET_MAX_S` (blank = unset)."""
    if _blank(text):
        return None
    parts = str(text).strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"{ONSET_VAR} must be '<lo>:<hi>' seconds, got {text!r}")
    lo, hi = (_number(p, ONSET_VAR) for p in parts)
    if not (0.0 <= lo < hi <= ONSET_MAX_S):
        raise ValueError(f"{ONSET_VAR} must satisfy 0 <= lo < hi <= {ONSET_MAX_S:g}, got {text!r}")
    return (lo + 0.0, hi + 0.0)


def parse_flag(text, var: str) -> bool:
    """``1`` / ``0`` (or blank = off); anything else is refused rather than guessed."""
    if _blank(text):
        return False
    t = str(text).strip().lower()
    if t in ("1", "true", "yes", "on"):
        return True
    if t in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{var} must be 1 or 0, got {text!r}")


#: ``{variable: parser}`` (the driver's overlay round-trip table).
FAULT_KNOB_PARSERS = {
    FAULT_VAR: parse_fault_cell,
    ONSET_VAR: parse_fault_onset,
    FROM_START_VAR: lambda text: parse_flag(text, FROM_START_VAR) or None,
}


def parse_fault_knobs(environ=None, *, exclusive_with=()) -> dict | None:
    """The armed fault spec (the dict :func:`apply_fault_knobs` writes), or ``None`` while all three are off.

    Raises ``ValueError`` for every refusal listed in the module docstring that needs only the environment.
    """
    env = os.environ if environ is None else environ
    unknown = sorted(k for k in env if str(k).startswith("KRABBY_FAULT") and k not in FAULT_VARS)
    if unknown:
        raise ValueError(f"unknown fault knob(s) {unknown}: one fault per cell, and a misspelled knob would run the "
                         f"cell at nominal (known: {list(FAULT_VARS)})")
    cell = parse_fault_cell(env.get(FAULT_VAR))
    onset = parse_fault_onset(env.get(ONSET_VAR))
    from_start = parse_flag(env.get(FROM_START_VAR), FROM_START_VAR)
    if cell is None:
        if onset is not None or from_start:
            raise ValueError(f"{ONSET_VAR} / {FROM_START_VAR} are set but {FAULT_VAR} is not (half a cell)")
        return None
    if onset is None:
        raise ValueError(f"{FAULT_VAR}={env.get(FAULT_VAR)!r} is armed with no onset window ({ONSET_VAR})")
    entry = FAULTS[cell["id"]]
    anchor = bool(entry.get("from_start", False))
    if from_start and not anchor:
        raise ValueError(f"{FROM_START_VAR}=1 is for the S12 / D1 anchors only, not {cell['id']}")
    if anchor and not from_start:
        raise ValueError(f"{cell['id']} is an anchor: it needs {FROM_START_VAR}=1 (CHARTER §3.6)")
    clash = [v for v in exclusive_with if not _blank(env.get(v))]
    if clash:
        raise ValueError(f"{FAULT_VAR} and {clash} are exclusive: one fault per cell, never on top of a stress knob")
    duration, params = entry["levels"][cell["level"]]
    spec = {
        "id": cell["id"],
        "level": cell["level"],
        "knob": f"{cell['id']}:{cell['level']}",
        "family": entry["family"],
        "duration_s": float(duration[1]) if duration is not None and duration[0] == "s" else None,
        "duration_steps": int(duration[1]) if duration is not None and duration[0] == "steps" else None,
        "permanent": duration is None,
        "params": dict(params),
        "onset_s": [onset[0], onset[1]],
        "from_start": bool(from_start),
    }
    return spec


# ------------------------------------------------------------------------------ application
def _check_cfg(cfg, spec: dict) -> None:
    fid = spec["id"]
    action = getattr(getattr(cfg, "actions", None), "joint_pos", None)
    if action is None:
        raise RuntimeError(f"{FAULT_VAR}={spec['knob']} is set but the env cfg has no actions.joint_pos")
    if fid in DEPTH_FAULTS:
        obs = getattr(cfg, "observations", None)
        missing = [g for g in DEPTH_GROUPS if getattr(getattr(obs, g, None), DEPTH_TERM, None) is None]
        if missing:
            raise RuntimeError(f"{FAULT_VAR}={spec['knob']} needs the depth groups {missing} (a student env only)")
    if fid in DELAY_FAULTS:
        hist = int(getattr(action, "history_length", 0) or 0)
        if not getattr(action, "use_delay", False) or hist < MIN_HISTORY_FOR_DELAY:
            raise RuntimeError(f"{FAULT_VAR}={spec['knob']} needs actions.joint_pos.use_delay and history_length >= "
                               f"{MIN_HISTORY_FOR_DELAY} (got use_delay={getattr(action, 'use_delay', None)}, "
                               f"history_length={hist})")
    if fid == "A5":
        ev = getattr(getattr(cfg, "events", None), "payload_reset", None)
        kg = tuple((getattr(ev, "params", None) or {}).get("kg_range", ())) if ev is not None else ()
        if tuple(float(x) for x in kg) != (A5_PAYLOAD_KG, A5_PAYLOAD_KG):
            raise RuntimeError(f"{FAULT_VAR}=A5:mid is A1 mid at {A5_PAYLOAD_KG:g} kg: it needs the payload event at "
                               f"kg_range ({A5_PAYLOAD_KG:g}, {A5_PAYLOAD_KG:g}) (the L090 level file), got {kg!r}")


def apply_fault_knobs(cfg, environ=None, *, exclusive_with=()) -> dict | None:
    """Write ``cfg.fault_inject`` for an armed fault; writes nothing and prints nothing while all three are off.

    Called by ``omni_knobs.apply_omni_knobs`` right after the stress / DR knobs (and so after the payload knobs, which
    A5 checks). ``exclusive_with``: variables refused together with a fault (the stress knobs). Returns the spec.
    """
    spec = parse_fault_knobs(environ, exclusive_with=exclusive_with)
    if spec is None:
        return None
    _check_cfg(cfg, spec)
    setattr(cfg, FAULT_ATTR, spec)
    dur = ("permanent" if spec["permanent"] else
           f"{spec['duration_s']:g} s" if spec["duration_s"] is not None else f"{spec['duration_steps']} steps")
    print(f"[KNOB] {FAULT_VAR}={spec['knob']} ({spec['family']}, {dur}, params {spec['params']}) onset U"
          f"({spec['onset_s'][0]:g}, {spec['onset_s'][1]:g}) s per (wave, env)"
          + (", active from step 0 (anchor)" if spec["from_start"] else "") + f" -> cfg.{FAULT_ATTR}", flush=True)
    return spec
