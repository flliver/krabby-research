#!/usr/bin/env python3
"""Stand-height campaign -- generator of the step-8 RAISE / LOWER VIDEO schedule (VIDEO ONLY; read by no gate).

Adapted copy of the closed predecessor's ``2026-10-11_stand_network/scripts/sn_video_schedule.py`` (never edited).
It writes a world-frame ``kind: schedule`` row PLUS the campaign's ``stand_height`` block (the per-hold height map the
stand_height launcher reads on its ``--sh-command manifest`` path), which ``sh_record_video.py`` records (S1 s18 walks
every walk hold, the stand-height network stands at every stand hold's height, the H6 restore hands back):

    sh_video_schedule.py [--plan raise|smoke|random|tour|tour_smoke|mix] [--seed N] [--total-s S]
                         [--table-start TOP[:POSE]] [--out-dir DIR] [--dry-run]

-> ``videos/scenarios_sh_video_<plan>.yaml`` (``manifest_version: 2``, ONE row ``sh_<plan>__omni`` + the
``stand_height`` block -- the tour plans: the ``stand_height_video`` block, see below) and
``videos/sh_video_schedule_<plan>.json`` (holds, heights, kinds, times, summary).

THE TOUR PLANS (2026-10-15, the user's "stand up off a table, walk forward 20 s, stand at min then max, then each
command type 10 s, stopping every 2-3 commands to stand at min or max"; world frame, ``spawn_yaw: first_face``; the
suite's speeds -- fwd 0.25, lat 0.22, back 0.22, diag 0.25 at 45 deg, move_turn 0.22 -- checked against
``scenarios_p2.yaml``'s ``field_suite__omni`` at generation):

* ``tour`` (~221 s): (a) stand 1.064 for 12 s (the table get-up: >= the rods' rise + the ramp from the table spawn
  height + 3 s settle + 3 s); (b) fwd 20 s; (c) stand h_min then stand h_max (one facing); (d) fwd, lat_l, back 10 s
  each -> stand h_min; (e) lat_r, diag -> stand h_max; (f) turn_l90 (face 0 -> +90), turn_r90 (+90 -> 0) -> stand
  h_min; (g) move_turn_l (dir 0, face 0 -> +90), move_turn_r (dir 0, face +90 -> 0) -> stand h_max; (h) fwd 8 s;
* ``tour_smoke`` (~65 s, the GPU smoke): table stand 12 s, fwd 6, stand h_min, turn_l90 6, stand h_max (at face +90:
  a stand after a facing change), fwd 8 (dir = face = +90).

Tour lengths: a stand (other than the first) = ``|dh| / 0.08 + 3 s settle + 3 s hold`` from the level it is entered
at; a command hold that follows a stand at ``h`` = its base + ``|h - 1.064| / 0.08 + 1 s`` (the restore), so every
command gets its base of actual motion; whole deciseconds, rounded up. A tour row has turns and a stand first, which
the launcher's one-facing / walk-first rules refuse, so it is written under ``stand_height_video`` (rules ``tour``) and
validated by ``sh_video_core.tour_row_reasons`` (those rules, adapted for turns and the table first stand) + the
harness validator + the tour lengths above; the first stand is checked against the table it is designed for
(:data:`TOUR_TABLE`, 0.635 m knee_folded: ``sh_table_core.env_need_s`` + 3 s + 3 s) and, with ``--table-start``,
against that table too (the K3 margin: first height >= top + 0.31).

THE MIX PLAN (2026-10-24, the user's "a longer video (3 min) running through various commands with the occasional
stand at various heights"; ``--plan mix --seed N --total-s S``, default 180 s at the launcher): the tour's commands
(:data:`TOUR_CMDS`, the suite's speeds, checked as the tour checks them) in a SEEDED order, a single stand every 2-3
commands at a seeded height spread over [h_min, h_max], no table. A DECK of command units (:data:`MIX_UNITS`: fwd,
lat_l, lat_r, back, diag, the pair turn_l90 -> turn_r90, the pair move_turn_l -> move_turn_r; a pair is atomic, so the
facing is 0 between units and at every stand) is dealt without replacement and re-shuffled when empty (the first deal
opens with fwd), so every command type appears before any repeats; a BLOCK draws units until it holds >= U{2, 3}
commands. The row is block, then (stand, block) while one more fits in ``--total-s`` (the final command stretched by
<= 4 s toward it; never a stand last). A command's base is U[9, 11] s (+ the tour's restore allowance after a stand:
``|h - 1.064| / 0.08 + 1 s``); a stand is entered at 1.064 and lasts the tour's ``|dh| / 0.08 + 3 + 3 s`` + U[0, 2] s;
its height is drawn uniformly inside a bin dealt from a deck of :data:`MIX_HEIGHT_BINS` equal bins of [h_min, h_max]
(re-shuffled when empty), to the mm, so 4-6 stands cover the range instead of one end. Written under
``stand_height_video`` (rules ``tour``; turns cannot pass the launcher's one-facing rule) and validated by
``sh_video_core.tour_row_reasons`` (first hold a walk: entered at 1.064) + the tour lengths + the harness validator;
a ``--table-start`` is refused (the first hold is a command). Lengths are whole deciseconds; ``random.Random(seed)``
(``random()`` only) drives every draw.

THE PLANS (one facing, face 0, for the whole row: every speed-0 hold is a stand slot, never a turn):

* ``raise`` (default, ~155 s): walk, then a stand whose commanded height changes INSIDE the stand (consecutive speed-0
  holds keep the latched stand slot) 1.064 -> 1.300 -> 0.750 -> 1.150 -> 0.850 -> 1.064, a walk, 1.346 (h_max) ->
  0.719 (h_min) -> 1.250, a walk (restore from 1.25), 0.800 -> 1.200, a walk (restore from 1.20), 0.900 -> 1.064, a
  final walk (no restore: the stand ends at walking height). Walks cycle fwd 0.25 / lat_l 0.20 / back 0.20 / lat_r
  0.20 (the campaign rows' speeds);
* ``smoke`` (~20 s): walk 3 s -> stand 1.20 (a raise) -> 1.00 (an in-stand lower) -> walk (restore from 1.00);
* ``random`` (``--seed``; ``--total-s`` default 150): random walks (a uniform WORLD direction at face 0, speed
  U[0.12, ellipse cap] m/s, U[5, 9] s and >= the rule's need + 1 s), each followed by a block of 1-3 consecutive
  stand holds at random heights (1.064 with probability 0.2, else U[h_min, h_max] to the mm) and random lengths (rule
  + U[1, 4] s), and a final walk.
  The video length is <= ``--total-s`` (blocks are added while one more block + walk fits; the final walk is
  stretched by at most 4 s toward the target).

HOLD LENGTHS = the manifest rules of ``scenarios_sh.yaml`` (``sh_eval_common.height_row_reasons``, re-checked on every
schedule): a stand entered at level ``l`` (the previous stand's height, or 1.064 after a walk) lasts >= ``|h - l| / 0.08
+ 1 s + 2 s``; a walk after a stand at ``h`` lasts >= ``|h - 1.064| / 0.08 + 1.5 s + 2 s``. The raise plan adds 1.5 s
to every stand (rounded up to 0.5 s) and walks 8 s or the rule's need + 2 s, whichever is longer (the smoke: stands
+ 0.5 s, the first walk 3 s, the last the rule's need, rounded up). Every stand hold is labelled ``"stand"`` (the
launcher's rule: heights are positional), walks carry their direction name. Lengths are whole deciseconds.

THE CONSTRAINTS (every schedule): ``sh_eval_common.height_row_reasons`` (above + heights in the frozen [h_min, h_max],
walk first and last, one facing, labels) and the harness's own validator ``gait_eval.schedule.validate_scenario``
(speed <= the ellipse cap at ``wrap(dir - face)``); the written YAML is re-loaded through
``gait_eval.schedule.load_manifest`` AND ``sh_eval_common.height_row`` (the launcher's manifest loader) and compared
with the generated holds / heights bitwise. The header (``manifest_version``, ``speed_ellipse``, ``ang_vel_max``,
``defaults``) and the row's ``task`` / ``terrain_label`` / ``env_seed`` / ``spawn_yaw`` / ``role`` are copied from
``scenarios_sh.yaml``'s ``hraise__omni`` row (the SH_T18 hraise stamp's scenario, the recorder's reference). Python's
``random.Random(seed)`` drives every draw (``random()`` only), so a seed reproduces the schedule bit for bit.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
VIDEOS = CAMPAIGN / "videos"
PKG = HERE.parents[2]                                      # crab_hex_omni_task
HARNESS_DIR = PKG.parent / "crab_hex_forward_task" / "scripts"
REF_MANIFEST = CAMPAIGN / "scenarios_sh.yaml"
REF_ROW = "hraise__omni"
CORE_PATH = HERE / "sh_video_core.py"

PLANS = ("raise", "smoke", "random", "tour", "tour_smoke", "mix")
TOUR_PLANS = ("tour", "tour_smoke")
MIX_PLAN = "mix"
#: the plans whose row lives in the stand_height_video block (turns: not a launcher row)
VIDEO_BLOCK_PLANS = TOUR_PLANS + (MIX_PLAN,)
#: the plans steered by --seed / --total-s
SEEDED_PLANS = ("random", MIX_PLAN)
DEFAULT_PLAN = "raise"
DEFAULT_SEED = 20261014
DEFAULT_TOTAL_S = 150.0
CHECKPOINT_PLACEHOLDER = "SUPPLIED-PER-RUN--pass---checkpoint"

DS = 10                                   # lengths are whole deciseconds
ROUND_S = 0.5                             # fixed plans: stand / walk lengths rounded up to this
# the fixed plans: ("walk", label, speed, dir) / ("stand", height)
_FWD, _LATL, _BACK, _LATR = ("fwd", 0.25, 0.0), ("lat_l", 0.20, 1.5707963), ("back", 0.20, 3.1415927), \
    ("lat_r", 0.20, -1.5707963)
RAISE_PLAN = (
    ("walk",) + _FWD,
    ("stand", 1.064), ("stand", 1.300), ("stand", 0.750), ("stand", 1.150), ("stand", 0.850), ("stand", 1.064),
    ("walk",) + _LATL,
    ("stand", "h_max"), ("stand", "h_min"), ("stand", 1.250),
    ("walk",) + _BACK,
    ("stand", 0.800), ("stand", 1.200),
    ("walk",) + _LATR,
    ("stand", 0.900), ("stand", 1.064),
    ("walk",) + _FWD,
)
SMOKE_PLAN = (("walk",) + _FWD, ("stand", 1.200), ("stand", 1.000), ("walk",) + _LATL)
#: per fixed plan: (steps, stand extra s beyond the rule's need, walk s, walk margin s beyond the rule's need)
FIXED = {"raise": (RAISE_PLAN, 1.5, 8.0, 2.0), "smoke": (SMOKE_PLAN, 0.5, 3.0, 0.0)}
# the random plan
WALK_DS = (50, 90)
WALK_MARGIN_S = 1.0                       # a walk after a stand: >= the rule's need + this
STAND_EXTRA_DS = (10, 40)
BLOCK_STANDS = (1, 3)
P_SETTLED = 0.2
SPEED_MIN = 0.12
FINAL_STRETCH_DS = 40
MAX_ATTEMPTS = 10000

# the tour plans (module docstring)
_HALF_PI, _PI, _QUARTER_PI = 1.5707963, 3.1415927, 0.7853982
#: (speed, world dir, world face) per tour command; the speeds / translation dirs are the suite's (checked against
#: scenarios_p2.yaml field_suite__omni), the turn faces the tour's (0 -> +90 -> 0)
TOUR_CMDS = {
    "fwd": (0.25, 0.0, 0.0), "lat_l": (0.22, _HALF_PI, 0.0), "back": (0.22, _PI, 0.0),
    "lat_r": (0.22, -_HALF_PI, 0.0), "diag": (0.25, _QUARTER_PI, 0.0),
    "turn_l90": (0.0, 0.0, _HALF_PI), "turn_r90": (0.0, 0.0, 0.0),
    "move_turn_l": (0.22, 0.0, _HALF_PI), "move_turn_r": (0.22, 0.0, 0.0),
}
#: the suite label each tour translation command takes its speed / dir from (move_turn_* take move_turn's speed)
SUITE_ROW = "field_suite__omni"
SUITE_OF = {"fwd": "fwd", "lat_l": "lat_l", "back": "back", "lat_r": "lat_r", "diag": "diag",
            "move_turn_l": "move_turn", "move_turn_r": "move_turn_r"}
TOUR_TABLE_STAND_S = 12.0                 # (a): >= rod rise + ramp from the table spawn height + 3 s + 3 s
TOUR_STAND_SETTLE_S = 3.0
TOUR_STAND_HOLD_S = 3.0
TOUR_RESTORE_EXTRA_S = 1.0                # a command after a stand at h: base + |h - 1.064| / 0.08 + 1 s
#: the table the tour's first stand is designed for (the recorder's --table-start for the user's video)
TOUR_TABLE = "0.635:knee_folded"
# ("table", h, s) | ("stand", h) | ("cmd", label, base_s[, (speed, dir, face)])
TOUR_PLAN = (
    ("table", "h_settled", TOUR_TABLE_STAND_S),                                           # (a)
    ("cmd", "fwd", 20.0),                                                                 # (b)
    ("stand", "h_min"), ("stand", "h_max"),                                               # (c)
    ("cmd", "fwd", 10.0), ("cmd", "lat_l", 10.0), ("cmd", "back", 10.0), ("stand", "h_min"),        # (d)
    ("cmd", "lat_r", 10.0), ("cmd", "diag", 10.0), ("stand", "h_max"),                    # (e)
    ("cmd", "turn_l90", 10.0), ("cmd", "turn_r90", 10.0), ("stand", "h_min"),             # (f)
    ("cmd", "move_turn_l", 10.0), ("cmd", "move_turn_r", 10.0), ("stand", "h_max"),       # (g)
    ("cmd", "fwd", 8.0),                                                                  # (h)
)
TOUR_SMOKE_PLAN = (
    ("table", "h_settled", TOUR_TABLE_STAND_S),
    ("cmd", "fwd", 6.0),
    ("stand", "h_min"),
    ("cmd", "turn_l90", 6.0),
    ("stand", "h_max"),                                   # face +90: a stand after a facing change
    ("cmd", "fwd", 8.0, (0.25, _HALF_PI, _HALF_PI)),      # body-forward at face +90
)
TOUR = {"tour": TOUR_PLAN, "tour_smoke": TOUR_SMOKE_PLAN}

# the mix plan (module docstring "THE MIX PLAN")
#: the command deck: units of TOUR_CMDS labels; a pair is dealt as one unit (the facing returns to 0 inside it)
MIX_UNITS = (("fwd",), ("lat_l",), ("lat_r",), ("back",), ("diag",), ("turn_l90", "turn_r90"),
             ("move_turn_l", "move_turn_r"))
MIX_OPENER = ("fwd",)                     # the first deal opens with this unit
MIX_GAP = (2, 3)                          # commands per block (between two stands): >= U{2, 3}
MIX_CMD_DS = (90, 110)                    # a command's base: U[9, 11] s
MIX_STAND_EXTRA_DS = (0, 20)              # a stand: the tour rule + U[0, 2] s
MIX_HEIGHT_BINS = 5                       # stand heights: a deck of equal bins over [h_min, h_max]
MIX_DEFAULT_TOTAL_S = 180.0               # the launcher's default for --plan mix


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


VC = _load(CORE_PATH, "_shv_schedule_core")
EC = VC.EC
CORE = VC.CORE


def schedule_module():
    """``gait_eval.schedule`` (stdlib + numpy), imported the way the harness / record_categories import it."""
    if str(HARNESS_DIR) not in sys.path:
        sys.path.insert(0, str(HARNESS_DIR))
    from gait_eval import schedule as S  # noqa: PLC0415

    return S


def row_id_for(plan: str) -> str:
    if plan not in PLANS:
        raise ValueError(f"plan {plan!r} not in {PLANS}")
    return f"sh_{plan}__omni"


TOUR_ROW_IDS = tuple(row_id_for(p) for p in TOUR_PLANS)


# ===================================================================================== reference header
def reference_header(ref_manifest: Path = REF_MANIFEST, ref_row: str = REF_ROW) -> dict:
    """The manifest-level fields the harness reads and the reference row's task / terrain_label / env_seed /
    spawn_yaw / role, from the raw YAML (the row checked to load through ``gait_eval.schedule.load_manifest``)."""
    import yaml  # noqa: PLC0415

    S = schedule_module()
    data = yaml.safe_load(Path(ref_manifest).read_text())
    if data.get("manifest_version") != 2:
        raise SystemExit(f"{ref_manifest}: manifest_version {data.get('manifest_version')!r}, expected 2")
    rows = {r.get("id"): r for r in data.get("scenarios") or []}
    if ref_row not in rows:
        raise SystemExit(f"{ref_manifest}: no row {ref_row!r}")
    raw = rows[ref_row]
    scen = {s.id: s for s in S.load_manifest(ref_manifest)[0]}[ref_row]
    hdr = {
        "manifest_version": 2,
        "speed_ellipse": {k: float(v) for k, v in (raw.get("speed_ellipse") or data["speed_ellipse"]).items()},
        "ang_vel_max": float(raw.get("ang_vel_max", data["ang_vel_max"])),
        "defaults": dict(data.get("defaults") or {}),
        "row": {"task": str(raw["task"]), "terrain_label": str(raw["terrain_label"]),
                "env_seed": int(raw.get("env_seed", 1)), "spawn_yaw": raw.get("spawn_yaw", "first_face"),
                "role": str(raw.get("role", "teacher"))},
        "source": {"manifest": str(ref_manifest), "row": ref_row},
    }
    if scen.get_speed_ellipse() != (hdr["speed_ellipse"]["fwd"], hdr["speed_ellipse"]["lat"],
                                    hdr["speed_ellipse"]["back"]):
        raise SystemExit("reference header: speed_ellipse does not round-trip through load_manifest")
    if raw.get("env"):
        raise SystemExit(f"reference row {ref_row!r} carries an env: block; the video row must not")
    return hdr


# ===================================================================================== the rules
def stand_need_s(h: float, level: float) -> float:
    """A stand hold entered at ``level``: ``|h - level| / 0.08 + 1 + 2`` s (``sh_eval_common.height_row_reasons``)."""
    return abs(float(h) - float(level)) / CORE.HEIGHT_RATE_M_S + EC.ARRIVAL_SETTLE_S + EC.K1_WINDOW_S


def walk_need_s(prev_stand_h: float | None) -> float:
    """A walk after a stand at ``prev_stand_h`` (None: no stand before it): ``|h - 1.064| / 0.08 + 1.5 + 2`` s."""
    if prev_stand_h is None:
        return 0.0
    return abs(float(prev_stand_h) - CORE.H_SETTLED) / CORE.HEIGHT_RATE_M_S + EC.K4_ONSET_SLACK_S \
        + EC.WALK_AFTER_RESTORE_S


def _ceil_ds(seconds: float) -> int:
    return int(math.ceil(float(seconds) * DS - 1e-9))


def _ceil_to(seconds: float, step: float) -> float:
    k = math.ceil(float(seconds) / step - 1e-9)
    return round(k * step, 6)


def _hold(S, label, speed, direction, hold_ds, face=0.0):
    return S.Hold(hold_s=hold_ds / DS, label=label, speed=float(speed), dir=float(direction), face=float(face))


def _scenario(S, hdr: dict, holds: list, row_id: str):
    defaults = dict(S.DEFAULTS)
    defaults.update(hdr["defaults"])
    return S.Scenario(id=row_id, task=hdr["row"]["task"], checkpoint=CHECKPOINT_PLACEHOLDER, role=hdr["row"]["role"],
                      schedule=list(holds), episodes=1, episode_length_s=None, env_seed=hdr["row"]["env_seed"],
                      terrain_label=hdr["row"]["terrain_label"], defaults=defaults, spawn_yaw=hdr["row"]["spawn_yaw"],
                      speed_ellipse=dict(hdr["speed_ellipse"]), ang_vel_max=hdr["ang_vel_max"])


def hold_dicts(holds) -> list[dict]:
    """The ``height_row_reasons`` view of the holds (+ ``dir``, which the tour rules' kinds read)."""
    return [{"label": h.label, "speed": float(h.speed), "dir": float(h.dir), "face": float(h.face),
             "hold_s": float(h.hold_s)} for h in holds]


def tour_table(text: str | None = None) -> dict:
    """The table a tour's first stand is checked against (``--table-start`` text; default :data:`TOUR_TABLE`)."""
    return VC.parse_table_start(text or TOUR_TABLE)


def tour_first_need(table: dict, h0: float) -> dict:
    """The first stand's needs on ``table``: ``rule_s`` (the scenarios_sh stand rule adapted: rod rise + ramp + 1 + 2
    s, ``sh_video_core.table_need_s``) and ``tour_s`` (the tour's: rod rise + ramp + 3 s settle + 3 s hold)."""
    n = VC.table_need_s(table, float(h0))
    return {**n, "tour_s": n["reach_s"] + TOUR_STAND_SETTLE_S + TOUR_STAND_HOLD_S}


def tour_length_reasons(holds, heights) -> list[str]:
    """The tour's own (stricter) lengths: every stand after the first >= ``|dh| / 0.08 + 3 + 3`` s from its entry
    level; every command hold after a stand at ``h`` >= ``|h - 1.064| / 0.08 + 1 s`` + 1 s of motion at least."""
    out = []
    kinds = VC.hold_kinds(holds)
    level, prev = CORE.H_SETTLED, None
    for i, (h, hh, k) in enumerate(zip(holds, heights, kinds)):
        if k == VC.KIND_STAND:
            if i > 0:
                need = abs(hh - level) / CORE.HEIGHT_RATE_M_S + TOUR_STAND_SETTLE_S + TOUR_STAND_HOLD_S
                if h.hold_s + 1e-9 < need:
                    out.append(f"hold {i} stand {hh}: {h.hold_s} s < the tour's {need:.3f} s")
            level, prev = hh, hh
        else:
            if prev is not None:
                need = abs(prev - CORE.H_SETTLED) / CORE.HEIGHT_RATE_M_S + TOUR_RESTORE_EXTRA_S + 1.0
                if h.hold_s + 1e-9 < need:
                    out.append(f"hold {i} {h.label}: {h.hold_s} s < the restore allowance + 1 s = {need:.3f} s")
            level, prev = CORE.H_SETTLED, None
    return out


def check_rules(S, hdr, holds, heights, row_id, *, table: dict | None = None) -> None:
    """Both validators (module docstring); raises with every reason. A tour row: ``sh_video_core.tour_row_reasons``
    (first stand entered at ``table``'s spawn height with its need; default :data:`TOUR_TABLE`) + the tour lengths +
    the first stand's tour need on that table + the harness validator."""
    if row_id in TOUR_ROW_IDS:
        tb = table or tour_table()
        why = VC.table_start_reasons(tb, VC.hold_kinds(holds), list(heights))
        if not why:
            need = tour_first_need(tb, heights[0])
            why = VC.tour_row_reasons(hold_dicts(holds), list(heights), first_level=tb["h_init_plan"],
                                      first_need_s=need["rule_s"])
            if holds[0].hold_s + 1e-9 < need["tour_s"]:
                why.append(f"hold 0 (the table stand): {holds[0].hold_s} s < the rods' rise + ramp + 3 + 3 = "
                           f"{need['tour_s']:.3f} s on the {tb['top']} m table ({tb['pose']})")
            why += tour_length_reasons(holds, heights)
        if why:
            raise ValueError(f"{row_id}: the tour rules refuse the schedule:\n  " + "\n  ".join(why))
    elif row_id == row_id_for(MIX_PLAN):
        # the mix: the tour rules with a walk first (entered at 1.064, no table need) + the tour lengths
        why = VC.tour_row_reasons(hold_dicts(holds), list(heights))
        if not why:
            why = tour_length_reasons(holds, heights)
        if why:
            raise ValueError(f"{row_id}: the tour rules refuse the schedule:\n  " + "\n  ".join(why))
    else:
        why = EC.height_row_reasons(hold_dicts(holds), list(heights))
        if why:
            raise ValueError(f"{row_id}: the stand_height manifest rules refuse the schedule:\n  " + "\n  ".join(why))
    S.validate_scenario(_scenario(S, hdr, holds, row_id))


def _resolve_h(h) -> float:
    if h == "h_max":
        return float(CORE.H_MAX)
    if h == "h_min":
        return float(CORE.H_MIN)
    if h == "h_settled":
        return float(CORE.H_SETTLED)
    return float(h)


def suite_speed_reasons(manifest=None, row: str = SUITE_ROW) -> list[str]:
    """:data:`TOUR_CMDS`' speeds (and translation dirs at face 0) vs the phase-2 suite's holds (read from the file)."""
    import yaml  # noqa: PLC0415

    path = Path(manifest) if manifest else VC.FIELD_MANIFEST
    rows = {r.get("id"): r for r in (yaml.safe_load(path.read_text()).get("scenarios") or [])}
    if row not in rows:
        return [f"{path}: no row {row!r}"]
    suite = {}
    for cat in rows[row].get("categories") or []:
        for h in cat.get("holds") or []:
            suite[str(h["label"])] = (float(h["speed"]), float(h["dir"]), float(h["face"]))
    out = []
    for name, slabel in SUITE_OF.items():
        if slabel not in suite:
            out.append(f"{row}: no hold {slabel!r}")
            continue
        sp, di, fa = suite[slabel]
        if TOUR_CMDS[name][0] != sp:
            out.append(f"tour {name} speed {TOUR_CMDS[name][0]} != {row} {slabel}'s {sp}")
        if fa == 0.0 and TOUR_CMDS[name][1] != di:
            out.append(f"tour {name} dir {TOUR_CMDS[name][1]} != {row} {slabel}'s {di}")
    return out


def _tour(S, plan: str) -> tuple[list, list]:
    """The tour holds (module docstring): stands at ``|dh| / 0.08 + 3 + 3`` s (the first: the plan's table length),
    commands at base + the previous stand's restore allowance; every stand at the previous hold's face."""
    why = suite_speed_reasons()
    if why:
        raise ValueError("the tour's speeds are not the suite's:\n  " + "\n  ".join(why))
    rate = CORE.HEIGHT_RATE_M_S
    holds, heights = [], []
    level, prev_stand, face = CORE.H_SETTLED, None, 0.0
    for step in TOUR[plan]:
        if step[0] in ("table", "stand"):
            h = _resolve_h(step[1])
            if step[0] == "table":
                if holds:
                    raise ValueError(f"{plan}: the table stand must be the first hold")
                ds = _ceil_ds(float(step[2]))
            else:
                ds = _ceil_ds(abs(h - level) / rate + TOUR_STAND_SETTLE_S + TOUR_STAND_HOLD_S)
            holds.append(_hold(S, EC.STAND_LABEL, 0.0, 0.0, ds, face=face))
            heights.append(h)
            level, prev_stand = h, h
        else:
            label, base = step[1], float(step[2])
            sp, di, fa = step[3] if len(step) > 3 else TOUR_CMDS[label]
            extra = 0.0 if prev_stand is None else abs(prev_stand - CORE.H_SETTLED) / rate + TOUR_RESTORE_EXTRA_S
            holds.append(_hold(S, label, sp, di, _ceil_ds(base + extra), face=fa))
            heights.append(None)
            level, prev_stand, face = CORE.H_SETTLED, None, float(fa)
    return holds, heights


# ===================================================================================== generation
def _fixed(S, plan: str) -> tuple[list, list]:
    steps, stand_extra, walk_s, walk_margin = FIXED[plan]
    holds, heights = [], []
    level, prev_stand = CORE.H_SETTLED, None
    for step in steps:
        if step[0] == "walk":
            _, label, speed, direction = step
            hold_s = walk_s if prev_stand is None else \
                max(walk_s, _ceil_to(walk_need_s(prev_stand) + walk_margin, ROUND_S))
            holds.append(_hold(S, label, speed, direction, _ceil_ds(hold_s)))
            heights.append(None)
            level, prev_stand = CORE.H_SETTLED, None
        else:
            h = _resolve_h(step[1])
            hold_s = _ceil_to(stand_need_s(h, level) + stand_extra, ROUND_S)
            holds.append(_hold(S, EC.STAND_LABEL, 0.0, 0.0, _ceil_ds(hold_s)))
            heights.append(h)
            level, prev_stand = h, h
    return holds, heights


def _u(rng: random.Random, lo: float, hi: float) -> float:
    return lo + (hi - lo) * rng.random()


def _int(rng: random.Random, lo: int, hi: int) -> int:
    """Uniform integer in [lo, hi] from ``rng.random()`` only (stable across Python versions)."""
    if hi < lo:
        raise ValueError(f"empty integer range [{lo}, {hi}]")
    return lo + min(int(rng.random() * (hi - lo + 1)), hi - lo)


def _draw_walk(rng, S, hdr, idx: int, prev_stand):
    fwd, lat, back = (hdr["speed_ellipse"][k] for k in ("fwd", "lat", "back"))
    for _ in range(MAX_ATTEMPTS):
        direction = round(_u(rng, -math.pi, math.pi), 7)
        cap = S.ellipse_cap(S.wrap_to_pi(direction - 0.0), fwd, lat, back)
        speed = math.floor(_u(rng, SPEED_MIN, cap) * 100.0 + 1e-9) / 100.0
        if speed < SPEED_MIN or speed > cap + 1e-12:
            continue
        hold_ds = max(_int(rng, *WALK_DS), _ceil_ds(walk_need_s(prev_stand) + (WALK_MARGIN_S if prev_stand is not None
                                                                                    else 0.0)))
        return _hold(S, f"walk{idx:02d}", speed, direction, hold_ds)
    raise RuntimeError("no valid walk command")  # pragma: no cover


def _draw_block(rng, level: float) -> list[tuple[float, int]]:
    n = _int(rng, *BLOCK_STANDS)
    out = []
    for _ in range(n):
        if rng.random() < P_SETTLED:
            h = CORE.H_SETTLED
        else:
            h = min(max(round(_u(rng, CORE.H_MIN, CORE.H_MAX), 3), CORE.H_MIN), CORE.H_MAX)
        hold_ds = _ceil_ds(stand_need_s(h, level)) + _int(rng, *STAND_EXTRA_DS)
        out.append((float(h), hold_ds))
        level = h
    return out


def _random(S, hdr, seed: int, total_s: float, prologue_s: float) -> tuple[list, list]:
    rng = random.Random(int(seed))
    total_ds = int(round(float(total_s) * DS))
    used = int(round(prologue_s * DS))
    holds = [_draw_walk(rng, S, hdr, 0, None)]
    heights = [None]
    used += int(round(holds[0].hold_s * DS))
    while True:
        block = _draw_block(rng, CORE.H_SETTLED)
        walk = _draw_walk(rng, S, hdr, len([h for h in heights if h is None]), block[-1][0])
        need = sum(ds for _h, ds in block) + int(round(walk.hold_s * DS))
        if used + need > total_ds:
            break
        for h, ds in block:
            holds.append(_hold(S, EC.STAND_LABEL, 0.0, 0.0, ds))
            heights.append(h)
        holds.append(walk)
        heights.append(None)
        used += need
    if len(holds) < 3:
        raise ValueError(f"--total-s {total_s} leaves no room for one stand block between two walks (seed {seed})")
    stretch = min(FINAL_STRETCH_DS, total_ds - used)
    if stretch > 0:
        last = holds[-1]
        holds[-1] = _hold(S, last.label, last.speed, last.dir, int(round(last.hold_s * DS)) + stretch)
    return holds, heights


def _shuffled(rng: random.Random, items) -> list:
    """Fisher-Yates from ``rng.random()`` only (via :func:`_int`; stable across Python versions)."""
    out = list(items)
    for i in range(len(out) - 1, 0, -1):
        j = _int(rng, 0, i)
        out[i], out[j] = out[j], out[i]
    return out


class _Deck:
    """Deal ``items`` without replacement and re-shuffle all of them when the pile is empty (the mix plan's command
    units and height bins). The first pile may open with ``first``; a re-shuffled pile never opens with the item just
    dealt (no back-to-back repeat across a re-shuffle)."""

    def __init__(self, rng: random.Random, items, first=None):
        self.rng, self.items, self.last = rng, tuple(items), None
        if first is not None and first not in self.items:
            raise ValueError(f"deck opener {first!r} not in {self.items}")
        rest = [x for x in self.items if first is None or x != first]
        self.pile = ([first] if first is not None else []) + _shuffled(rng, rest)

    def deal(self):
        if not self.pile:
            self.pile = _shuffled(self.rng, self.items)
            if len(self.pile) > 1 and self.pile[0] == self.last:
                self.pile[0], self.pile[1] = self.pile[1], self.pile[0]
        self.last = self.pile.pop(0)
        return self.last


def _mix_command(S, label: str, base_ds: int, prev_stand):
    """One :data:`TOUR_CMDS` command: its base + the tour's restore allowance after a stand at ``prev_stand``."""
    sp, di, fa = TOUR_CMDS[label]
    extra = 0.0 if prev_stand is None else \
        abs(float(prev_stand) - CORE.H_SETTLED) / CORE.HEIGHT_RATE_M_S + TOUR_RESTORE_EXTRA_S
    return _hold(S, label, sp, di, _ceil_ds(base_ds / DS + extra), face=fa)


def _mix_block(rng, S, deck: _Deck, prev_stand) -> list:
    """Units dealt from ``deck`` until the block holds >= U{MIX_GAP} commands; the first gets the restore allowance."""
    gap = _int(rng, *MIX_GAP)
    holds = []
    while len(holds) < gap:
        for label in deck.deal():
            holds.append(_mix_command(S, label, _int(rng, *MIX_CMD_DS), None if holds else prev_stand))
    return holds


def _mix_stand(rng, S, bins: _Deck, face: float):
    """A stand at the previous hold's ``face``: height uniform in a dealt bin of [h_min, h_max] (to the mm), length the
    tour's ``|h - 1.064| / 0.08 + 3 + 3 s`` (entered after a command) + U[MIX_STAND_EXTRA_DS]."""
    k = bins.deal()
    w = (CORE.H_MAX - CORE.H_MIN) / MIX_HEIGHT_BINS
    h = min(max(round(CORE.H_MIN + w * (k + rng.random()), 3), CORE.H_MIN), CORE.H_MAX)
    ds = _ceil_ds(abs(h - CORE.H_SETTLED) / CORE.HEIGHT_RATE_M_S + TOUR_STAND_SETTLE_S + TOUR_STAND_HOLD_S) \
        + _int(rng, *MIX_STAND_EXTRA_DS)
    return _hold(S, EC.STAND_LABEL, 0.0, 0.0, ds, face=face), float(h)


def _mix(S, seed: int, total_s: float, prologue_s: float) -> tuple[list, list]:
    """The mix plan (module docstring): block, then (stand, block) while one more fits in ``total_s``; the final
    command stretched by <= :data:`FINAL_STRETCH_DS` toward it."""
    why = suite_speed_reasons()
    if why:
        raise ValueError("the mix's speeds are not the suite's:\n  " + "\n  ".join(why))
    rng = random.Random(int(seed))
    deck = _Deck(rng, MIX_UNITS, first=MIX_OPENER)
    bins = _Deck(rng, range(MIX_HEIGHT_BINS))
    total_ds = int(round(float(total_s) * DS))
    used = int(round(prologue_s * DS))
    holds = _mix_block(rng, S, deck, None)
    heights = [None] * len(holds)
    used += sum(int(round(h.hold_s * DS)) for h in holds)
    while True:
        stand, h = _mix_stand(rng, S, bins, float(holds[-1].face))
        block = _mix_block(rng, S, deck, h)
        need = int(round(stand.hold_s * DS)) + sum(int(round(b.hold_s * DS)) for b in block)
        if used + need > total_ds:
            break
        holds += [stand] + block
        heights += [h] + [None] * len(block)
        used += need
    if all(h is None for h in heights):
        raise ValueError(f"--total-s {total_s} leaves no room for one stand between two command blocks (seed {seed})")
    stretch = min(FINAL_STRETCH_DS, total_ds - used)
    if stretch > 0:
        last = holds[-1]
        holds[-1] = _hold(S, last.label, last.speed, last.dir, int(round(last.hold_s * DS)) + stretch, face=last.face)
    return holds, heights


def generate(plan: str = DEFAULT_PLAN, seed: int = DEFAULT_SEED, total_s: float = DEFAULT_TOTAL_S, *,
             hdr: dict | None = None, table_start: str | None = None) -> dict:
    """The schedule (``holds`` as ``gait_eval.schedule.Hold``, ``heights`` one per hold) + records + summary.
    Deterministic in ``(plan, seed, total_s)``; ``seed`` / ``total_s`` only steer the random plan. ``table_start``
    (``TOP[:POSE]``): also check the schedule against that table start (``sh_video_core.table_start_reasons``; a tour:
    its first stand's needs on that table) -- the recorder's ``--table-start`` refusals, run on the CPU first."""
    S = schedule_module()
    hdr = hdr or reference_header()
    row_id = row_id_for(plan)
    prologue_s = float(hdr["defaults"].get("prologue_s", S.DEFAULTS["prologue_s"]))
    table = VC.parse_table_start(table_start) if table_start else None
    if plan in FIXED:
        holds, heights = _fixed(S, plan)
    elif plan in TOUR_PLANS:
        holds, heights = _tour(S, plan)
    else:
        if abs(round(float(total_s) * DS) / DS - float(total_s)) > 1e-9:
            raise ValueError(f"--total-s {total_s} is not a whole number of deciseconds")
        if plan == MIX_PLAN:
            holds, heights = _mix(S, seed, total_s, prologue_s)
        else:
            holds, heights = _random(S, hdr, seed, total_s, prologue_s)
    check_rules(S, hdr, holds, heights, row_id, table=table)
    if table is not None:
        why = VC.table_start_reasons(table, VC.hold_kinds(holds), list(heights))
        if why:
            raise ValueError(f"{row_id}: " + "; ".join(why))
    first = None
    if plan in TOUR_PLANS:
        tb = table or tour_table()
        need = tour_first_need(tb, heights[0])
        first = {"table": tb, "level": tb["h_init_plan"], "need_s": need["rule_s"], "tour_need_s": need["tour_s"],
                 "sizing": {k: need[k] for k in ("rise_to_init_s", "ramp_s", "rise_to_cmd_s", "reach_s")}}
    out = _describe(hdr, holds, heights, plan, seed, total_s, row_id, prologue_s, first=first)
    out["table_start"] = None if table is None else dict(table)
    return out


def _mix_summary(summary: dict, recs: list, rules: dict) -> None:
    """The mix plan's additions to ``_describe``'s summary / rules (in place): the commands, turns, stand spacing."""
    summary["n_turns"] = sum(1 for r in recs if r["kind"] == VC.KIND_TURN)
    summary["commands"] = [r["label"] for r in recs if r["kind"] != VC.KIND_STAND]
    summary["command_types"] = sorted(set(summary["commands"]))
    summary["missing_command_types"] = sorted(set(TOUR_CMDS) - set(summary["commands"]))
    summary["height_bins"] = [min(int((r["height"] - CORE.H_MIN) / ((CORE.H_MAX - CORE.H_MIN) / MIX_HEIGHT_BINS)),
                                  MIX_HEIGHT_BINS - 1) for r in recs if r["kind"] == VC.KIND_STAND]
    summary["stand_t0_s"] = [r["t0_s"] for r in recs if r["kind"] == VC.KIND_STAND]
    rules.update({"source": "sh_video_core.tour_row_reasons (sh_eval_common.height_row_reasons adapted: turns; a walk "
                            "first, entered at 1.064)",
                  "tour_stand": f"hold_s >= |dh| / 0.08 + {TOUR_STAND_SETTLE_S} + {TOUR_STAND_HOLD_S} s",
                  "tour_command_after_stand": f"base + |h - 1.064| / 0.08 + {TOUR_RESTORE_EXTRA_S} s",
                  "mix": {"units": [list(u) for u in MIX_UNITS], "opener": list(MIX_OPENER), "gap": list(MIX_GAP),
                          "command_base_s": [d / DS for d in MIX_CMD_DS],
                          "stand_extra_s": [d / DS for d in MIX_STAND_EXTRA_DS], "height_bins": MIX_HEIGHT_BINS}})


def _describe(hdr, holds, heights, plan, seed, total_s, row_id, prologue_s, *, first: dict | None = None) -> dict:
    kinds = VC.hold_kinds(holds)
    recs, t = [], prologue_s
    level = CORE.H_SETTLED if first is None else float(first["level"])
    prev_stand = None
    for i, (h, hh, k) in enumerate(zip(holds, heights, kinds)):
        if k == VC.KIND_STAND:
            need = stand_need_s(hh, level)
            if i == 0 and first is not None:
                need = max(need, float(first["need_s"]))
            level, prev_stand = hh, hh
        else:
            need = walk_need_s(prev_stand)
            level, prev_stand = CORE.H_SETTLED, None
        recs.append({"label": h.label, "kind": k, "speed": h.speed, "dir": h.dir, "face": h.face, "hold_s": h.hold_s,
                     "height": hh, "need_s": round(need, 6), "slack_s": round(h.hold_s - need, 6),
                     "t0_s": round(t, 6), "t1_s": round(t + h.hold_s, 6)})
        t += h.hold_s
    stands = [r for r in recs if r["kind"] == VC.KIND_STAND]
    changes = [(a["height"], b["height"]) for a, b in zip(recs, recs[1:])
               if a["kind"] == b["kind"] == VC.KIND_STAND]
    summary = {
        "n_holds": len(recs), "n_walks": sum(1 for r in recs if r["kind"] == VC.KIND_WALK), "n_stands": len(stands),
        "heights": [r["height"] for r in stands],
        "height_min": min((r["height"] for r in stands), default=None),
        "height_max": max((r["height"] for r in stands), default=None),
        "in_stand_raises": sum(1 for a, b in changes if b > a), "in_stand_lowers": sum(1 for a, b in changes if b < a),
        "restores": sum(1 for a, b in zip(recs, recs[1:]) if a["kind"] == VC.KIND_STAND
                        and b["kind"] != VC.KIND_STAND and abs(a["height"] - CORE.H_SETTLED) > CORE.RESTORE_ENGAGE_M),
        "min_slack_s": min(r["slack_s"] for r in recs),
        "prologue_s": prologue_s, "holds_s": round(sum(r["hold_s"] for r in recs), 6),
        "video_s": round(prologue_s + sum(r["hold_s"] for r in recs), 6),
    }
    rules = {"stand": "hold_s >= |h - level| / 0.08 + 1 + 2 s", "walk_after_stand":
             "hold_s >= |h - 1.064| / 0.08 + 1.5 + 2 s", "source": "sh_eval_common.height_row_reasons",
             "h_min": CORE.H_MIN, "h_max": CORE.H_MAX, "validator": "gait_eval.schedule.validate_scenario"}
    if first is not None:
        summary["n_turns"] = sum(1 for r in recs if r["kind"] == VC.KIND_TURN)
        summary["commands"] = [r["label"] for r in recs if r["kind"] != VC.KIND_STAND]
        summary["first_stand"] = {"height": heights[0], "hold_s": holds[0].hold_s, **first}
        rules.update({"source": "sh_video_core.tour_row_reasons (sh_eval_common.height_row_reasons adapted: turns, "
                                "a table first stand)",
                      "tour_stand": f"hold_s >= |dh| / 0.08 + {TOUR_STAND_SETTLE_S} + {TOUR_STAND_HOLD_S} s",
                      "tour_command_after_stand": f"base + |h - 1.064| / 0.08 + {TOUR_RESTORE_EXTRA_S} s",
                      "tour_first_stand": "rod rise + ramp from the table spawn height + 3 + 3 s "
                                          "(sh_table_core.env_need_s)"})
    elif plan == MIX_PLAN:
        _mix_summary(summary, recs, rules)
    return {"plan": plan, "seed": int(seed), "total_s": float(total_s), "row_id": row_id, "header": hdr,
            "rules": rules, "holds": holds, "heights": list(heights), "records": recs, "summary": summary}


# ===================================================================================== output
def manifest_dict(gen: dict) -> dict:
    hdr = gen["header"]
    sm = gen["summary"]
    row = {
        "id": gen["row_id"], "role": hdr["row"]["role"], "task": hdr["row"]["task"],
        "terrain_label": hdr["row"]["terrain_label"], "checkpoint": CHECKPOINT_PLACEHOLDER, "episodes": 1,
        "env_seed": hdr["row"]["env_seed"], "episode_length_s": round(sm["video_s"] + 2.0, 6),
        "spawn_yaw": hdr["row"]["spawn_yaw"],
        "schedule": [{"speed": h.speed, "dir": h.dir, "face": h.face, "hold_s": h.hold_s, "label": h.label}
                     for h in gen["holds"]],
        "notes": (f"VIDEO ONLY (step-8 raise / lower video; read by no gate): plan {gen['plan']}"
                  + (f" seed {gen['seed']}" if gen["plan"] == "random" else "")
                  + f", {sm['n_walks']} walks, {sm['n_stands']} stand holds at {sm['height_min']}-{sm['height_max']} m "
                  f"({sm['in_stand_raises']} in-stand raises, {sm['in_stand_lowers']} lowers, {sm['restores']} "
                  f"restored hand-backs); {sm['video_s']:.1f} s incl. the {sm['prologue_s']} s prologue. Generated by "
                  "sh_video_schedule.py; heights from the stand_height block below."),
    }
    if gen["plan"] == MIX_PLAN:
        row["notes"] = (f"VIDEO ONLY (the stand-height command mix; read by no gate): plan mix seed {gen['seed']}, "
                        f"{len(sm['commands'])} commands ({', '.join(sm['commands'])}; {sm['n_turns']} turns) with "
                        f"{sm['n_stands']} stands at {sm['height_min']}-{sm['height_max']} m between them; "
                        f"{sm['video_s']:.1f} s incl. the {sm['prologue_s']} s prologue. Generated by "
                        f"sh_video_schedule.py; heights from the {VC.VIDEO_BLOCK} block below (turns: not a launcher "
                        "row).")
    elif gen["plan"] in TOUR_PLANS:
        row["notes"] = (f"VIDEO ONLY (the stand-height tour; read by no gate): plan {gen['plan']}, a table get-up "
                        f"stand then {len(sm['commands'])} commands ({', '.join(sm['commands'])}; {sm['n_turns']} "
                        f"turns) with {sm['n_stands'] - 1} stands at {sm['height_min']}-{sm['height_max']} m between "
                        f"them; {sm['video_s']:.1f} s incl. the {sm['prologue_s']} s prologue. Generated by "
                        f"sh_video_schedule.py; heights from the {VC.VIDEO_BLOCK} block below (turns + a stand first: "
                        "not a launcher row).")
    if gen["plan"] in VIDEO_BLOCK_PLANS:
        blk = {"version": VC.VIDEO_BLOCK_VERSION,
               "rows": {gen["row_id"]: {"hold_heights": list(gen["heights"]), "rules": VC.VIDEO_RULES,
                                        "notes": "VIDEO ONLY: no gate reads this row"}}}
        return {"manifest_version": 2, "speed_ellipse": dict(hdr["speed_ellipse"]), "ang_vel_max": hdr["ang_vel_max"],
                "defaults": dict(hdr["defaults"]), "scenarios": [row], VC.VIDEO_BLOCK: blk}
    blk = {"version": EC.MANIFEST_BLOCK_VERSION,
           "rows": {gen["row_id"]: {"gates": [], "hold_heights": list(gen["heights"]),
                                    "notes": "VIDEO ONLY: no gate reads this row"}}}
    return {"manifest_version": 2, "speed_ellipse": dict(hdr["speed_ellipse"]), "ang_vel_max": hdr["ang_vel_max"],
            "defaults": dict(hdr["defaults"]), "scenarios": [row], EC.MANIFEST_BLOCK: blk}


def manifest_text(gen: dict) -> str:
    import yaml  # noqa: PLC0415

    args = f"--plan {gen['plan']}" + (f" --seed {gen['seed']} --total-s {gen['total_s']:g}"
                                      if gen["plan"] in SEEDED_PLANS else "")
    if gen.get("table_start"):
        args += f" --table-start {gen['table_start']['top']:g}:{gen['table_start']['pose']}"
    what = ("the stand-height COMMAND MIX" if gen["plan"] == MIX_PLAN
            else "the stand-height TOUR" if gen["plan"] in TOUR_PLANS else "step-8 raise / lower")
    head = (f"# Stand-height campaign -- {what} VIDEO schedule (VIDEO ONLY; read by no gate, ledger or "
            "reader).\n"
            f"# GENERATED by scripts/sh_video_schedule.py {args} -- do not hand-edit; re-run the generator.\n"
            "# Header + row task / terrain_label / env_seed / spawn_yaw copied from "
            f"{gen['header']['source']['manifest']}\n"
            f"# (row {gen['header']['source']['row']}). Recorded by scripts/sh_record_video.py (S1 s18 on walk holds, "
            "the stand-height\n# network on stand holds at the "
            f"{VC.VIDEO_BLOCK if gen['plan'] in VIDEO_BLOCK_PLANS else EC.MANIFEST_BLOCK} block's heights, the H6 "
            "restore armed).\n")
    return head + yaml.safe_dump(manifest_dict(gen), sort_keys=False, default_flow_style=None, width=120)


def schedule_json(gen: dict) -> dict:
    out = {k: v for k, v in gen.items() if k not in ("holds",)}
    out["generator"] = str(Path(__file__).resolve())
    return out


def check_written(path: Path, gen: dict) -> dict:
    """Re-load the written YAML through ``gait_eval.schedule.load_manifest`` (validates the row as written) and
    ``sh_eval_common.height_row`` (the launcher's loader); compare holds and heights bitwise. Returns the height row."""
    S = schedule_module()
    scen = {s.id: s for s in S.load_manifest(path)[0]}
    if list(scen) != [gen["row_id"]]:
        raise SystemExit(f"{path}: rows {list(scen)}, expected [{gen['row_id']!r}]")
    sc = scen[gen["row_id"]]
    got = [(h.label, h.hold_s, h.as_world()) for h in sc.schedule]
    want = [(h.label, h.hold_s, h.as_world()) for h in gen["holds"]]
    if got != want:
        raise SystemExit(f"{path}: the re-loaded holds differ from the generated ones")
    if sc.command_frame != "world" or sc.kind != S.SCHEDULE_KIND:
        raise SystemExit(f"{path}: not a world-frame schedule row")
    if abs(sc.total_duration_s - gen["summary"]["video_s"]) > 1e-9:
        raise SystemExit(f"{path}: total {sc.total_duration_s} != {gen['summary']['video_s']}")
    fs = (gen["summary"].get("first_stand") or {})
    kw = {"first_level": fs["level"], "first_need_s": fs["need_s"]} if fs else {}
    row = VC.load_height_row(path, gen["row_id"], **kw)
    if row["hold_heights"] != list(gen["heights"]) or row["labels"] != [h.label for h in gen["holds"]]:
        raise SystemExit(f"{path}: sh_eval_common.height_row reads other heights / labels than generated")
    return row


def print_summary(gen: dict, file=sys.stdout) -> None:
    sm = gen["summary"]
    print(f"[SHV] plan {gen['plan']}" + (f" seed {gen['seed']}" if gen["plan"] in SEEDED_PLANS else "")
          + f" row {gen['row_id']}: {sm['n_holds']} holds, video {sm['video_s']:.1f} s (prologue {sm['prologue_s']} s "
          f"+ holds {sm['holds_s']:.1f} s); {sm['n_walks']} walks, {sm['n_stands']} stands at {sm['heights']}; "
          f"{sm['in_stand_raises']} in-stand raises, {sm['in_stand_lowers']} lowers, {sm['restores']} restores; "
          f"min slack {sm['min_slack_s']:.2f} s", file=file)
    for r in gen["records"]:
        if r["kind"] == VC.KIND_STAND:
            what = f"STAND {r['height']:.3f} m face {math.degrees(r['face']):+5.0f}"
        elif r["kind"] == VC.KIND_TURN:
            what = f"TURN  to face {math.degrees(r['face']):+5.0f}"
        else:
            what = (f"WALK  {r['speed']:.2f} m/s dir {math.degrees(r['dir']):+6.1f}"
                    + (f" face {math.degrees(r['face']):+4.0f}" if r["face"] else ""))
        print(f"[SHV]   {r['t0_s']:7.2f}-{r['t1_s']:7.2f} s  {r['label']:11s} {what:36s} {r['hold_s']:5.1f} s "
              f"(rule {r['need_s']:5.2f} s, slack {r['slack_s']:+5.2f})", file=file)
    fs = sm.get("first_stand")
    if fs:
        tb = fs["table"]
        print(f"[SHV]   first stand on the {tb['top']} m table ({tb['pose']}): {fs['hold_s']} s >= the tour's rod rise + "
              f"ramp + 3 + 3 = {fs['tour_need_s']:.2f} s (rule {fs['need_s']:.2f} s); height {fs['height']} >= top + "
              f"{tb['cmd_margin_m']} = {tb['cmd_min']:.4f}", file=file)
    if gen["plan"] == MIX_PLAN:
        print(f"[SHV]   mix: {len(sm['commands'])} commands, {sm['n_turns']} turns; types {sm['command_types']} "
              f"(missing {sm['missing_command_types'] or 'none'}); stands at t0 {sm['stand_t0_s']} s in height bins "
              f"{sm['height_bins']} of {MIX_HEIGHT_BINS} over [{CORE.H_MIN}, {CORE.H_MAX}]", file=file)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", choices=PLANS, default=DEFAULT_PLAN)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random / mix plans only")
    ap.add_argument("--total-s", type=float, default=None,
                    help=f"random / mix plans only: target video length (default {DEFAULT_TOTAL_S:g} s; mix "
                         f"{MIX_DEFAULT_TOTAL_S:g} s)")
    ap.add_argument("--table-start", type=str, default=None, metavar="TOP[:POSE]",
                    help="check the schedule against this table start (the recorder's --table-start refusals)")
    ap.add_argument("--out-dir", type=str, default=str(VIDEOS))
    ap.add_argument("--dry-run", action="store_true", help="print the schedule; write nothing")
    args = ap.parse_args(argv)
    try:
        total_s = args.total_s if args.total_s is not None else (MIX_DEFAULT_TOTAL_S if args.plan == MIX_PLAN
                                                                 else DEFAULT_TOTAL_S)
        gen = generate(args.plan, args.seed, total_s, table_start=args.table_start)
    except ValueError as exc:
        print(f"[SHV] REFUSED: {exc}", file=sys.stderr)
        return 2
    print_summary(gen)
    out = Path(args.out_dir)
    yml = out / f"scenarios_sh_video_{args.plan}.yaml"
    js = out / f"sh_video_schedule_{args.plan}.json"
    if args.dry_run:
        print(f"[SHV] --dry-run: would write {yml} and {js}")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    yml.write_text(manifest_text(gen))
    check_written(yml, gen)
    js.write_text(json.dumps(schedule_json(copy.deepcopy(gen)), indent=1) + "\n")
    loader = ("sh_video_core.video_height_row (the tour rules)" if args.plan in VIDEO_BLOCK_PLANS
              else "sh_eval_common.height_row")
    print(f"[SHV] wrote {yml} (re-loaded by gait_eval.schedule.load_manifest and {loader}) and {js}")
    print(f"[SHV] MANIFEST={yml}")
    print(f"[SHV] ROW={gen['row_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
