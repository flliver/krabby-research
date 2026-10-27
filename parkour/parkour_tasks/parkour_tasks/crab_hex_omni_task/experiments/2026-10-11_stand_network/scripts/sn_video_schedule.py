#!/usr/bin/env python3
"""Stand-network campaign -- seeded generator of the random-command VIDEO schedule (VIDEO ONLY; read by no gate).

User request 2026-10-13: "Record a 3 minute video with random commands and a random length of stand between each".
This writes the world-frame ``kind: schedule`` row that ``sn_record_video.py`` records (S1 s18 walks every command
hold, the stand network acts on every stand hold):

    sn_video_schedule.py [--seed 20261013] [--total-s 180] [--kinds move,turn,move_turn] [--tag smoke] [--dry-run]

-> ``videos/scenarios_sn_video[_<tag>].yaml`` (``manifest_version: 2``, ONE row ``sn_random[_<tag>]__omni``) and
``videos/sn_video_schedule[_<tag>].json`` (the holds with their kind / stand flag / start-end times + the summary).

THE SCHEDULE. ``[settle stand] + n x [random COMMAND hold, random-length STAND hold]``:

* the settle stand: U[2, 3] s at face 0 (the spawn pin, ``spawn_yaw: first_face``); the 1.5 s prologue in front of
  it holds the same stand command, so the robot stands ~4 s before the first command;
* a command: ``move`` (half the draws) -- a uniform WORLD direction at the current face, speed U[0.12, cap] m/s;
  ``turn`` (a quarter) -- in place, facing change U[30, 110] deg, random sign; ``move_turn`` (a quarter) -- a move
  plus a U[20, 90] deg facing change. Length U[6, 14] s;
* a stand: speed 0 at the face the command left (so it is a STAND under ``sn_video_core``'s rule, also after a
  turn). Length U[3, 12] s;
* the last pair is fitted so that ``prologue + every hold == --total-s`` EXACTLY (lengths are whole deciseconds);
  every intermediate pair leaves room for at least one more pair.

THE CONSTRAINTS (a candidate command is RESAMPLED until it passes all of them):

* the harness's own validator, ``gait_eval.schedule.validate_scenario`` (the one ``record_categories.py`` and the
  eval use) on the schedule so far plus the candidate: speed <= the manifest ellipse cap at ``wrap(dir - face)``
  (``speed_ellipse`` fwd 0.35 / lat 0.25 / back 0.25, the reference manifest's) and ``hold_s >= settle_s +
  |dface| / ang_vel_max`` (0.3 rad/s);
* stricter, for the picture: a move_turn's speed is <= the MINIMUM cap along the whole turn (the body direction
  sweeps from ``dir - previous face`` to ``dir - face``), so the command term never re-caps it mid-turn; and a facing
  change gets ``hold_s >= settle_s + |dface| / 0.15`` (the plant's measured ~0.15-0.2 rad/s, not the 0.3 command
  cap -- the omni manifest's step-16 note), so the turn finishes and the following stand starts at its face.

The final YAML is re-loaded with ``gait_eval.schedule.load_manifest`` (which validates the row as written) and its
holds compared to the generated ones bitwise; the header fields (``manifest_version``, ``speed_ellipse``,
``ang_vel_max``, ``defaults``) and the row's ``task`` / ``terrain_label`` / ``env_seed`` / ``spawn_yaw`` are copied
from the reference manifest's ``suite__omni`` row (the SN_Amask_s17 suite es1 stamp's scenario). Python's
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
REF_MANIFEST = PKG / "experiments" / "eval" / "scenarios_omni.yaml"
REF_ROW = "suite__omni"
CORE_PATH = HERE / "sn_video_core.py"

DEFAULT_SEED = 20261013
DEFAULT_TOTAL_S = 180.0
ROW_ID = "sn_random__omni"
CHECKPOINT_PLACEHOLDER = "SUPPLIED-PER-RUN--pass---checkpoint"

DS = 10                                   # lengths are whole deciseconds
SETTLE_DS = (20, 30)                      # the first (settle) stand
COMMAND_DS = (60, 140)                    # every command hold
STAND_DS = (30, 120)                      # every stand between commands
KIND_WEIGHTS = {"move": 0.5, "turn": 0.25, "move_turn": 0.25}
TURN_DEG = (30.0, 110.0)
MOVE_TURN_DEG = (20.0, 90.0)
SPEED_MIN = 0.12
PLANT_TURN_RATE = 0.15                    # rad/s: the planning rate (the validator uses ang_vel_max 0.3)
CAP_PATH_SAMPLES = 65
MAX_ATTEMPTS = 10000


def _load_core():
    spec = importlib.util.spec_from_file_location("_sn_video_core", str(CORE_PATH))
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


CORE = _load_core()


def schedule_module():
    """``gait_eval.schedule`` (stdlib + numpy), imported the way the harness / record_categories import it."""
    if str(HARNESS_DIR) not in sys.path:
        sys.path.insert(0, str(HARNESS_DIR))
    from gait_eval import schedule as S  # noqa: PLC0415

    return S


# ===================================================================================== reference header
def reference_header(ref_manifest: Path = REF_MANIFEST, ref_row: str = REF_ROW) -> dict:
    """The manifest-level fields the harness reads (``manifest_version`` / ``speed_ellipse`` / ``ang_vel_max`` /
    ``defaults``) and the reference row's ``task`` / ``terrain_label`` / ``env_seed`` / ``spawn_yaw``, from the raw
    YAML (and the row checked to load through ``gait_eval.schedule.load_manifest``)."""
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
        raise SystemExit(f"reference row {ref_row!r} carries an env: block; the video row must not (record_categories "
                         "refuses a row env the reference stamp did not run)")
    return hdr


# ===================================================================================== draws
def _u(rng: random.Random, lo: float, hi: float) -> float:
    return lo + (hi - lo) * rng.random()


def _int(rng: random.Random, lo: int, hi: int) -> int:
    """Uniform integer in [lo, hi] from ``rng.random()`` only (stable across Python versions)."""
    if hi < lo:
        raise ValueError(f"empty integer range [{lo}, {hi}]")
    return lo + min(int(rng.random() * (hi - lo + 1)), hi - lo)


def _choice(rng: random.Random, weights: dict) -> str:
    keys = list(weights)
    tot = sum(weights[k] for k in keys)
    x = rng.random() * tot
    acc = 0.0
    for k in keys:
        acc += weights[k]
        if x < acc:
            return k
    return keys[-1]


def min_cap_along_turn(direction: float, face0: float, face1: float, ellipse: tuple[float, float, float],
                       S=None) -> float:
    """The minimum ellipse cap over the body direction ``wrap(dir - face)`` as the face sweeps the SHORTEST way from
    ``face0`` to ``face1`` (endpoints included; ``face0 == face1`` -> the one cap)."""
    S = S or schedule_module()
    fwd, lat, back = ellipse
    d = CORE.wrap_to_pi(face1 - face0)
    n = 1 if abs(d) <= CORE.FACE_TOL else CAP_PATH_SAMPLES
    caps = [S.ellipse_cap(S.wrap_to_pi(direction - (face0 + d * k / n)), fwd, lat, back) for k in range(n + 1)]
    caps.append(S.ellipse_cap(S.wrap_to_pi(direction - face1), fwd, lat, back))    # the exact end (the validator's)
    return float(min(caps))


def _r7(x: float) -> float:
    return float(round(float(x), 7))


def _label(i: int, kind: str, speed: float, dface: float, hold_ds: int) -> str:
    deg = int(round(math.degrees(dface)))
    if kind == "stand":
        return f"{i:02d}_stand{hold_ds / DS:.1f}s"
    if kind == "move":
        return f"{i:02d}_move{speed:.2f}"
    if kind == "turn":
        return f"{i:02d}_turn{deg:+d}"
    return f"{i:02d}_moveturn{speed:.2f}{deg:+d}"


def _hold(S, label, speed, direction, face, hold_ds):
    return S.Hold(hold_s=hold_ds / DS, label=label, speed=float(speed), dir=float(direction), face=float(face))


def _scenario(S, hdr: dict, holds: list, row_id: str):
    defaults = dict(S.DEFAULTS)
    defaults.update(hdr["defaults"])
    sc = S.Scenario(id=row_id, task=hdr["row"]["task"], checkpoint=CHECKPOINT_PLACEHOLDER, role=hdr["row"]["role"],
                    schedule=list(holds), episodes=1, episode_length_s=None, env_seed=hdr["row"]["env_seed"],
                    terrain_label=hdr["row"]["terrain_label"], defaults=defaults, spawn_yaw=hdr["row"]["spawn_yaw"],
                    speed_ellipse=dict(hdr["speed_ellipse"]), ang_vel_max=hdr["ang_vel_max"])
    return sc


def _draw_command(rng, S, hdr, holds, face, idx, ds_range, kinds, row_id, stats):
    """One command hold at the current ``face``, resampled until it passes every constraint (module docstring)."""
    ellipse = (hdr["speed_ellipse"]["fwd"], hdr["speed_ellipse"]["lat"], hdr["speed_ellipse"]["back"])
    settle_s = float(hdr["defaults"].get("settle_s", S.DEFAULTS["settle_s"]))
    wz_max = float(hdr["ang_vel_max"])
    lo, hi = ds_range
    for _attempt in range(MAX_ATTEMPTS):
        stats["attempts"] += 1
        kind = _choice(rng, kinds)
        if kind == "move":
            dface = 0.0
        else:
            span = TURN_DEG if kind == "turn" else MOVE_TURN_DEG
            dface = math.radians(_u(rng, *span)) * (1.0 if rng.random() < 0.5 else -1.0)
        new_face = face if kind == "move" else _r7(CORE.wrap_to_pi(face + dface))
        dface = CORE.wrap_to_pi(new_face - face)
        need_ds = lo
        if kind != "move":
            need_s = settle_s + abs(dface) / min(PLANT_TURN_RATE, wz_max)
            need_ds = max(lo, int(math.ceil(need_s * DS - 1e-9)))
        if need_ds > hi:
            stats["rejected_length"] += 1
            continue
        hold_ds = _int(rng, need_ds, hi)
        if kind == "turn":
            speed, direction = 0.0, 0.0
        else:
            direction = _r7(_u(rng, -math.pi, math.pi))
            cap = min_cap_along_turn(direction, face, new_face, ellipse, S)
            speed = math.floor(_u(rng, SPEED_MIN, cap) * 100.0 + 1e-9) / 100.0
            if speed < SPEED_MIN or speed > cap + 1e-12:
                stats["rejected_speed"] += 1
                continue
        cand = _hold(S, _label(idx, kind, speed, dface, hold_ds), speed, direction, new_face, hold_ds)
        try:
            S.validate_scenario(_scenario(S, hdr, holds + [cand], row_id))
        except S.ManifestError:
            stats["rejected_validator"] += 1
            continue
        return cand, kind, dface
    raise RuntimeError(f"no valid command after {MAX_ATTEMPTS} draws (hold {idx}, range {ds_range} ds)")


def generate(seed: int = DEFAULT_SEED, total_s: float = DEFAULT_TOTAL_S, *, kinds=None, hdr: dict | None = None,
             row_id: str = ROW_ID) -> dict:
    """The schedule (``holds`` as ``gait_eval.schedule.Hold``) + per-hold records + the summary. Deterministic in
    ``seed``. ``kinds``: the allowed command kinds (default all three at their weights)."""
    S = schedule_module()
    hdr = hdr or reference_header()
    weights = dict(KIND_WEIGHTS)
    if kinds is not None:
        bad = [k for k in kinds if k not in KIND_WEIGHTS]
        if bad or not kinds:
            raise ValueError(f"kinds must be a non-empty subset of {list(KIND_WEIGHTS)}, got {kinds}")
        weights = {k: KIND_WEIGHTS[k] for k in KIND_WEIGHTS if k in kinds}
    rng = random.Random(int(seed))
    prologue_ds = int(round(float(hdr["defaults"].get("prologue_s", S.DEFAULTS["prologue_s"])) * DS))
    total_ds = int(round(float(total_s) * DS))
    if abs(total_ds / DS - float(total_s)) > 1e-9:
        raise ValueError(f"--total-s {total_s} is not a whole number of deciseconds")
    min_pair = COMMAND_DS[0] + STAND_DS[0]
    max_pair = COMMAND_DS[1] + STAND_DS[1]
    if total_ds < prologue_ds + SETTLE_DS[1] + min_pair:
        raise ValueError(f"--total-s {total_s} leaves no room for settle + one command + one stand "
                         f"(need >= {(prologue_ds + SETTLE_DS[1] + min_pair) / DS} s)")
    stats = {"attempts": 0, "rejected_length": 0, "rejected_speed": 0, "rejected_validator": 0}
    face = 0.0
    settle_ds = _int(rng, *SETTLE_DS)
    holds = [_hold(S, _label(0, "stand", 0.0, 0.0, settle_ds), 0.0, 0.0, face, settle_ds)]
    rem = total_ds - prologue_ds - settle_ds
    while rem > 0:
        idx = len(holds)
        if rem <= max_pair:          # the last pair: fit the remainder exactly
            cmd_range = (max(COMMAND_DS[0], rem - STAND_DS[1]), min(COMMAND_DS[1], rem - STAND_DS[0]))
        else:                        # leave >= one more pair after this one
            cmd_range = (COMMAND_DS[0], min(COMMAND_DS[1], rem - min_pair - STAND_DS[0]))
        cmd, _kind, _dface = _draw_command(rng, S, hdr, holds, face, idx, cmd_range, weights, row_id, stats)
        holds.append(cmd)
        face = float(cmd.face)
        cmd_ds = int(round(cmd.hold_s * DS))
        if rem <= max_pair:
            st_ds = rem - cmd_ds
        else:
            st_ds = _int(rng, STAND_DS[0], min(STAND_DS[1], rem - cmd_ds - min_pair))
        holds.append(_hold(S, _label(idx + 1, "stand", 0.0, 0.0, st_ds), 0.0, 0.0, face, st_ds))
        rem -= cmd_ds + st_ds
    if rem != 0:
        raise AssertionError(f"schedule total off by {rem} ds")
    scen = _scenario(S, hdr, holds, row_id)
    S.validate_scenario(scen)
    return _describe(S, hdr, holds, seed, total_s, row_id, weights, stats, prologue_ds)


def _describe(S, hdr, holds, seed, total_s, row_id, weights, stats, prologue_ds) -> dict:
    kinds = CORE.hold_kinds(holds, spawn_face=0.0)
    flags = CORE.hold_stand_flags(holds, spawn_face=0.0)
    recs, t = [], prologue_ds / DS
    prev = 0.0
    for h, k, fl in zip(holds, kinds, flags):
        recs.append({"label": h.label, "kind": k, "stand": bool(fl), "speed": h.speed, "dir": h.dir, "face": h.face,
                     "hold_s": h.hold_s, "dface_deg": math.degrees(CORE.wrap_to_pi(h.face - prev)),
                     "t0_s": round(t, 6), "t1_s": round(t + h.hold_s, 6)})
        t += h.hold_s
        prev = h.face
    stands = [r["hold_s"] for r in recs if r["stand"]]
    cmds = [r for r in recs if not r["stand"]]
    summary = {
        "n_holds": len(recs), "n_commands": len(cmds), "n_stands": len(stands),
        "n_stands_between_commands": len(stands) - 1,
        "commands_by_kind": {k: sum(1 for r in cmds if r["kind"] == k) for k in ("move", "turn", "move_turn")},
        "stand_s": stands, "settle_s": stands[0] if stands else None,
        "stand_between_s_min": min(stands[1:]) if len(stands) > 1 else None,
        "stand_between_s_max": max(stands[1:]) if len(stands) > 1 else None,
        "command_s": [r["hold_s"] for r in cmds],
        "stands_after_a_facing_change": sum(1 for a, b in zip(recs, recs[1:]) if b["stand"] and a["kind"] in
                                            ("turn", "move_turn")),
        # the eval's latched slot (no reset in one continuous episode) keeps TURN on EVERY stand after the first
        # facing change, not only the one right after it: sn_video_core.eval_slot_stand_flags (fall-free prediction)
        "stands_eval_slot_reads_turn": [r["label"] for r, ev in
                                        zip(recs, CORE.eval_slot_stand_flags(holds, spawn_face=0.0))
                                        if r["stand"] and not ev],
        "prologue_s": prologue_ds / DS, "holds_s": round(sum(r["hold_s"] for r in recs), 6),
        "video_s": round(prologue_ds / DS + sum(r["hold_s"] for r in recs), 6),
        "draws": stats,
    }
    return {"seed": int(seed), "total_s": float(total_s), "row_id": row_id, "header": hdr, "kind_weights": weights,
            "constraints": {"command_s": [COMMAND_DS[0] / DS, COMMAND_DS[1] / DS],
                            "stand_s": [STAND_DS[0] / DS, STAND_DS[1] / DS],
                            "settle_s": [SETTLE_DS[0] / DS, SETTLE_DS[1] / DS], "turn_deg": list(TURN_DEG),
                            "move_turn_deg": list(MOVE_TURN_DEG), "speed_min": SPEED_MIN,
                            "plant_turn_rate": PLANT_TURN_RATE, "validator": "gait_eval.schedule.validate_scenario"},
            "stand_rule": CORE.STAND_RULE_SCHEDULE, "holds": holds, "records": recs, "summary": summary}


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
        "notes": (f"VIDEO ONLY (2026-10-13 user request; read by no gate): flat, {sm['n_commands']} random commands "
                  f"({', '.join(f'{n} {k}' for k, n in sm['commands_by_kind'].items())}) each followed by a "
                  f"random-length stand (U[{STAND_DS[0] / DS:g}, {STAND_DS[1] / DS:g}] s), after a "
                  f"{sm['settle_s']:.1f} s settle stand; seed {gen['seed']}; {sm['video_s']:.1f} s incl. the "
                  f"{sm['prologue_s']} s prologue. Generated by sn_video_schedule.py; S1 walks the command holds, "
                  "the stand network the stand holds (sn_record_video.py)."),
    }
    return {"manifest_version": 2, "speed_ellipse": dict(hdr["speed_ellipse"]), "ang_vel_max": hdr["ang_vel_max"],
            "defaults": dict(hdr["defaults"]), "scenarios": [row]}


def manifest_text(gen: dict) -> str:
    import yaml  # noqa: PLC0415

    head = (f"# Stand-network campaign -- random-command VIDEO schedule (VIDEO ONLY; read by no gate, ledger or reader).\n"
            f"# GENERATED by scripts/sn_video_schedule.py --seed {gen['seed']} --total-s {gen['total_s']:g}"
            + (f" --kinds {','.join(gen['kind_weights'])}" if list(gen['kind_weights']) != list(KIND_WEIGHTS) else "")
            + (f" --tag {gen['row_id'][len('sn_random_'):-len('__omni')]}" if gen["row_id"] != ROW_ID else "")
            + " -- do not hand-edit; re-run the generator.\n"
            f"# Header + row task / terrain_label / env_seed / spawn_yaw copied from {gen['header']['source']['manifest']}\n"
            f"# (row {gen['header']['source']['row']}, the SN_Amask_s17 suite es1 stamp's scenario). Recorded by\n"
            "# scripts/sn_record_video.py (S1 s18 on command holds, the stand network on stand holds).\n")
    return head + yaml.safe_dump(manifest_dict(gen), sort_keys=False, default_flow_style=None, width=120)


def schedule_json(gen: dict) -> dict:
    out = {k: v for k, v in gen.items() if k not in ("holds",)}
    out["generator"] = str(Path(__file__).resolve())
    return out


def check_written(path: Path, gen: dict) -> None:
    """Re-load the written YAML through ``gait_eval.schedule.load_manifest`` (validates the row as written) and compare
    every hold bitwise with the generated one."""
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


def print_summary(gen: dict, file=sys.stdout) -> None:
    sm = gen["summary"]
    print(f"[SNV] seed {gen['seed']} row {gen['row_id']}: {sm['n_holds']} holds, video {sm['video_s']:.1f} s "
          f"(prologue {sm['prologue_s']} s + holds {sm['holds_s']:.1f} s); {sm['n_commands']} commands "
          f"{sm['commands_by_kind']}, {sm['n_stands']} stands (settle {sm['settle_s']:.1f} s + "
          f"{sm['n_stands_between_commands']} between commands, {sm['stand_between_s_min']}-"
          f"{sm['stand_between_s_max']} s); {sm['stands_after_a_facing_change']} stands follow a turn / move_turn "
          f"directly; the eval's latched slot would read TURN on {len(sm['stands_eval_slot_reads_turn'])} stands "
          f"{sm['stands_eval_slot_reads_turn']}", file=file)
    for r in gen["records"]:
        print(f"[SNV]   {r['t0_s']:7.2f}-{r['t1_s']:7.2f} s  {r['label']:22s} {r['kind']:9s} "
              f"{'STAND NET' if r['stand'] else 'S1':9s} speed {r['speed']:.2f} dir {math.degrees(r['dir']):+7.1f} "
              f"face {math.degrees(r['face']):+7.1f} (dface {r['dface_deg']:+6.1f}) {r['hold_s']:.1f} s", file=file)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--total-s", type=float, default=DEFAULT_TOTAL_S,
                    help="video length = prologue + every hold, seconds (default 180)")
    ap.add_argument("--kinds", type=str, default=None, help="CSV subset of move,turn,move_turn (default all three)")
    ap.add_argument("--tag", type=str, default=None,
                    help="file / row suffix: videos/scenarios_sn_video_<tag>.yaml, row sn_random_<tag>__omni")
    ap.add_argument("--out-dir", type=str, default=str(VIDEOS))
    ap.add_argument("--dry-run", action="store_true", help="print the schedule; write nothing")
    args = ap.parse_args(argv)
    tag = args.tag
    if tag is not None and (not tag or not all(c.isalnum() or c == "_" for c in tag)):
        raise SystemExit(f"--tag must be [A-Za-z0-9_]+, got {tag!r}")
    row_id = ROW_ID if tag is None else f"sn_random_{tag}__omni"
    kinds = None if args.kinds is None else [k.strip() for k in args.kinds.split(",") if k.strip()]
    gen = generate(args.seed, args.total_s, kinds=kinds, row_id=row_id)
    print_summary(gen)
    suffix = "" if tag is None else f"_{tag}"
    out = Path(args.out_dir)
    yml = out / f"scenarios_sn_video{suffix}.yaml"
    js = out / f"sn_video_schedule{suffix}.json"
    if args.dry_run:
        print(f"[SNV] --dry-run: would write {yml} and {js}")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    yml.write_text(manifest_text(gen))
    check_written(yml, gen)
    js.write_text(json.dumps(schedule_json(copy.deepcopy(gen)), indent=1) + "\n")
    print(f"[SNV] wrote {yml} (re-loaded and validated by gait_eval.schedule.load_manifest) and {js}")
    print(f"[SNV] MANIFEST={yml}")
    print(f"[SNV] ROW={row_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
