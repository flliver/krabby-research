#!/usr/bin/env python3
"""Stand-height campaign -- generator of the step-8 RAISE / LOWER VIDEO schedule (VIDEO ONLY; read by no gate).

Adapted copy of the closed predecessor's ``2026-10-11_stand_network/scripts/sn_video_schedule.py`` (never edited).
It writes a world-frame ``kind: schedule`` row PLUS the campaign's ``stand_height`` block (the per-hold height map the
stand_height launcher reads on its ``--sh-command manifest`` path), which ``sh_record_video.py`` records (S1 s18 walks
every walk hold, the stand-height network stands at every stand hold's height, the H6 restore hands back):

    sh_video_schedule.py [--plan raise|smoke|random] [--seed N] [--total-s S] [--out-dir DIR] [--dry-run]

-> ``videos/scenarios_sh_video_<plan>.yaml`` (``manifest_version: 2``, ONE row ``sh_<plan>__omni`` + the
``stand_height`` block) and ``videos/sh_video_schedule_<plan>.json`` (holds, heights, kinds, times, summary).

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

PLANS = ("raise", "smoke", "random")
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


def _hold(S, label, speed, direction, hold_ds):
    return S.Hold(hold_s=hold_ds / DS, label=label, speed=float(speed), dir=float(direction), face=0.0)


def _scenario(S, hdr: dict, holds: list, row_id: str):
    defaults = dict(S.DEFAULTS)
    defaults.update(hdr["defaults"])
    return S.Scenario(id=row_id, task=hdr["row"]["task"], checkpoint=CHECKPOINT_PLACEHOLDER, role=hdr["row"]["role"],
                      schedule=list(holds), episodes=1, episode_length_s=None, env_seed=hdr["row"]["env_seed"],
                      terrain_label=hdr["row"]["terrain_label"], defaults=defaults, spawn_yaw=hdr["row"]["spawn_yaw"],
                      speed_ellipse=dict(hdr["speed_ellipse"]), ang_vel_max=hdr["ang_vel_max"])


def hold_dicts(holds) -> list[dict]:
    """The ``height_row_reasons`` view of the holds."""
    return [{"label": h.label, "speed": float(h.speed), "face": float(h.face), "hold_s": float(h.hold_s)}
            for h in holds]


def check_rules(S, hdr, holds, heights, row_id) -> None:
    """Both validators (module docstring); raises with every reason."""
    why = EC.height_row_reasons(hold_dicts(holds), list(heights))
    if why:
        raise ValueError(f"{row_id}: the stand_height manifest rules refuse the schedule:\n  " + "\n  ".join(why))
    S.validate_scenario(_scenario(S, hdr, holds, row_id))


def _resolve_h(h) -> float:
    if h == "h_max":
        return float(CORE.H_MAX)
    if h == "h_min":
        return float(CORE.H_MIN)
    return float(h)


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


def generate(plan: str = DEFAULT_PLAN, seed: int = DEFAULT_SEED, total_s: float = DEFAULT_TOTAL_S, *,
             hdr: dict | None = None) -> dict:
    """The schedule (``holds`` as ``gait_eval.schedule.Hold``, ``heights`` one per hold) + records + summary.
    Deterministic in ``(plan, seed, total_s)``; ``seed`` / ``total_s`` only steer the random plan."""
    S = schedule_module()
    hdr = hdr or reference_header()
    row_id = row_id_for(plan)
    prologue_s = float(hdr["defaults"].get("prologue_s", S.DEFAULTS["prologue_s"]))
    if plan in FIXED:
        holds, heights = _fixed(S, plan)
    else:
        if abs(round(float(total_s) * DS) / DS - float(total_s)) > 1e-9:
            raise ValueError(f"--total-s {total_s} is not a whole number of deciseconds")
        holds, heights = _random(S, hdr, seed, total_s, prologue_s)
    check_rules(S, hdr, holds, heights, row_id)
    return _describe(hdr, holds, heights, plan, seed, total_s, row_id, prologue_s)


def _describe(hdr, holds, heights, plan, seed, total_s, row_id, prologue_s) -> dict:
    kinds = VC.hold_kinds(holds)
    recs, t = [], prologue_s
    level = CORE.H_SETTLED
    prev_stand = None
    for h, hh, k in zip(holds, heights, kinds):
        if k == VC.KIND_STAND:
            need = stand_need_s(hh, level)
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
        "n_holds": len(recs), "n_walks": len(recs) - len(stands), "n_stands": len(stands),
        "heights": [r["height"] for r in stands],
        "height_min": min((r["height"] for r in stands), default=None),
        "height_max": max((r["height"] for r in stands), default=None),
        "in_stand_raises": sum(1 for a, b in changes if b > a), "in_stand_lowers": sum(1 for a, b in changes if b < a),
        "restores": sum(1 for a, b in zip(recs, recs[1:]) if a["kind"] == VC.KIND_STAND
                        and b["kind"] == VC.KIND_WALK and abs(a["height"] - CORE.H_SETTLED) > CORE.RESTORE_ENGAGE_M),
        "min_slack_s": min(r["slack_s"] for r in recs),
        "prologue_s": prologue_s, "holds_s": round(sum(r["hold_s"] for r in recs), 6),
        "video_s": round(prologue_s + sum(r["hold_s"] for r in recs), 6),
    }
    return {"plan": plan, "seed": int(seed), "total_s": float(total_s), "row_id": row_id, "header": hdr,
            "rules": {"stand": "hold_s >= |h - level| / 0.08 + 1 + 2 s", "walk_after_stand":
                      "hold_s >= |h - 1.064| / 0.08 + 1.5 + 2 s", "source": "sh_eval_common.height_row_reasons",
                      "h_min": CORE.H_MIN, "h_max": CORE.H_MAX, "validator": "gait_eval.schedule.validate_scenario"},
            "holds": holds, "heights": list(heights), "records": recs, "summary": summary}


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
    blk = {"version": EC.MANIFEST_BLOCK_VERSION,
           "rows": {gen["row_id"]: {"gates": [], "hold_heights": list(gen["heights"]),
                                    "notes": "VIDEO ONLY: no gate reads this row"}}}
    return {"manifest_version": 2, "speed_ellipse": dict(hdr["speed_ellipse"]), "ang_vel_max": hdr["ang_vel_max"],
            "defaults": dict(hdr["defaults"]), "scenarios": [row], EC.MANIFEST_BLOCK: blk}


def manifest_text(gen: dict) -> str:
    import yaml  # noqa: PLC0415

    args = f"--plan {gen['plan']}" + (f" --seed {gen['seed']} --total-s {gen['total_s']:g}"
                                      if gen["plan"] == "random" else "")
    head = ("# Stand-height campaign -- step-8 raise / lower VIDEO schedule (VIDEO ONLY; read by no gate, ledger or "
            "reader).\n"
            f"# GENERATED by scripts/sh_video_schedule.py {args} -- do not hand-edit; re-run the generator.\n"
            "# Header + row task / terrain_label / env_seed / spawn_yaw copied from "
            f"{gen['header']['source']['manifest']}\n"
            f"# (row {gen['header']['source']['row']}). Recorded by scripts/sh_record_video.py (S1 s18 on walk holds, "
            "the stand-height\n# network on stand holds at the stand_height block's heights, the H6 restore armed).\n")
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
    row = EC.height_row(path, gen["row_id"])
    if row["hold_heights"] != list(gen["heights"]) or row["labels"] != [h.label for h in gen["holds"]]:
        raise SystemExit(f"{path}: sh_eval_common.height_row reads other heights / labels than generated")
    return row


def print_summary(gen: dict, file=sys.stdout) -> None:
    sm = gen["summary"]
    print(f"[SHV] plan {gen['plan']}" + (f" seed {gen['seed']}" if gen["plan"] == "random" else "")
          + f" row {gen['row_id']}: {sm['n_holds']} holds, video {sm['video_s']:.1f} s (prologue {sm['prologue_s']} s "
          f"+ holds {sm['holds_s']:.1f} s); {sm['n_walks']} walks, {sm['n_stands']} stands at {sm['heights']}; "
          f"{sm['in_stand_raises']} in-stand raises, {sm['in_stand_lowers']} lowers, {sm['restores']} restores; "
          f"min slack {sm['min_slack_s']:.2f} s", file=file)
    for r in gen["records"]:
        what = f"STAND {r['height']:.3f} m" if r["kind"] == VC.KIND_STAND else \
            f"WALK  {r['speed']:.2f} m/s dir {math.degrees(r['dir']):+7.1f}"
        print(f"[SHV]   {r['t0_s']:7.2f}-{r['t1_s']:7.2f} s  {r['label']:8s} {what:28s} {r['hold_s']:5.1f} s "
              f"(rule {r['need_s']:5.2f} s, slack {r['slack_s']:+5.2f})", file=file)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", choices=PLANS, default=DEFAULT_PLAN)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random plan only")
    ap.add_argument("--total-s", type=float, default=DEFAULT_TOTAL_S, help="random plan only: target video length")
    ap.add_argument("--out-dir", type=str, default=str(VIDEOS))
    ap.add_argument("--dry-run", action="store_true", help="print the schedule; write nothing")
    args = ap.parse_args(argv)
    gen = generate(args.plan, args.seed, args.total_s)
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
    print(f"[SHV] wrote {yml} (re-loaded by gait_eval.schedule.load_manifest and sh_eval_common.height_row) and {js}")
    print(f"[SHV] MANIFEST={yml}")
    print(f"[SHV] ROW={gen['row_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
