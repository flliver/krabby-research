#!/usr/bin/env python3
"""Omni phase 2 -- the CHARTER 5 smokes S1-S4 in ONE headless Isaac process (pre-launch audit B1, 2026-10-06).

It never trains. It builds the env a p2a arm trains on, runs four checks against it, and writes
``<campaign>/smokes/<UTC>/smokes.json`` with one PASS/FAIL line per check (also in ``smokes.txt``). It exits 0 only if
S1-S4 all PASS. S5 is a separate row eval (``run_arm.sh --row-eval SMOKE_S5_s18 ... field_light__omni 1 --episodes
7``), read by ``p2_read.py --smoke-s5``. After all five pass, ``p2_read.py --write-smokes-ok`` writes the queue's
``$OUT/SMOKES_OK`` sentinel.

THE ENV IS THE ARM'S. The parent (CPU, no Isaac) composes the env with run_candidates' own functions, the way the
driver builds a p2a stage (run_experiment -> ``rp.train``):
``reference_recipe(cands, <ref>)`` -> ``stage_env(recipe, "p2a")`` -> ``{KRABBY_OMNI_PHASE: <base>, KRABBY_PLANT:
<plant>, PYTHONUNBUFFERED: 1, **overlay}`` -> ``rp._subenv`` (every parent KRABBY_* stripped, EULA/TERM set), and
task ``ph.PHASES[<base>].task`` (Isaac-Crab-Hex-Omni-Teacher-v0; ``activate_phase`` expands the 2a preset, including
KRABBY_HEX_TEACHER_MODE=2a). A placeholder overlay value (the driver's "TBD_" prefix or "PENDING") cannot be parsed
at config import. It takes the value the CAL1 chain launches with (``--fill-from``, default P2__CAL1_s18). Today the
only one is the A1 KRABBY_TERRAIN_PROMOTE, which becomes 0.45:0.25. That knob affects only promotions in S4's
second episode, and smokes.json records every such substitution. The child is started with that env (cwd
``parkour/``, ``--num-envs``, ``env_cfg.seed`` = the reference's seed).

THE CHECKS (CHARTER 5; thresholds are the plan's, pinned below):
  S1 cfg dump   the built terrain generator: sub-terrain keys and classes (parkour_flat OmniFlatTerrainCfg,
                omni_field_sparse / _dense OmniFieldTerrainCfg), normalised proportions 0.5 / 0.25 / 0.25,
                noise_range (0.02, 0.02) on every sub-terrain, difficulty_range (0.20, 0.70), curriculum true,
                clear_radius 1.7 / ring_outer_radius 4.80 on EVERY sub-terrain and parkours.base_parkour
                clear_radius_m / ring_outer_radius_m, vertical_edges false, bar:slab:pit 0.4:0.4:0.2 (bar_fraction
                0.4, pit_fraction 0.2), n_blocks 14 / 28.
  S2 terrain    on every FIELD tile of the built generator (height_fields x vertical_scale, block_masks, the edge
                masks), thresholds RECALIBRATED 2026-10-06 from the CPU replica (``--calibrate-s2``; derivations at
                the constants): the pad (r <= pad from the tile centre, pixel i at (i - (W-1)/2) hs, the parkour
                term's own pixel frame) has NO block occupancy (exact, from block_masks) and |h| <= 0.045 m; coverage
                (occupied share of the tile) inside +-50 % of 8.5 % per 14 blocks (sparse ~8.5 %, dense ~17 %); the
                per-corner population rate beyond r 4.5 m, pooled over a key's field tiles, >= 0.70 sparse / >= 0.90
                dense; max height <= max(hurdle_hi, step_height)(d) + 0.050 m, with d the tile's difficulty
                (curriculum row d, an upper bound under random_difficulty); pits <= 0: every field tile with pits has
                sunken occupied pixels and none deeper than pit_depth(d) + 0.050 m; no edge-mask pixel (ramp mode).
                Info alongside: all-4-corner shares, each tile's off-block roughness amplitude, edge pixels by row.
                ``--print-plan`` prints each threshold next to the CPU-predicted value.
  S3 resets     ceil(--s3-resets / num_envs) full resets (>= 200 resets) with terrain levels FROZEN (the launch
                levels; ``freeze_terrain_levels`` set for the check, then restored). At t = 0 of each round: no foot
                over occupancy (``feet_block_flags_tiled`` on the term's exact mask, neighbour tiles included), foot z
                >= terrain z - 0.01 m (the foot's own height-field pixel). Then 1 s of zero actions: crab_failure
                events counted (pass = 0). Max toe radius = the max horizontal distance of any footpad from its root,
                over t = 0 and the 1-s window (pass <= 1.6 m). Feet and roots are read from the PhysX view, which is
                fresh after reset's sim.forward().
  S4 walking    the s18 head (anchors.yaml; sha256 verified), built as the eval harness builds a teacher: the
                runner from the task's agent cfg in THIS env (KRABBY_STD_PARAM soft, KRABBY_INIT_STD ... from the
                composed overlay), runner.load, estimator into the privileged slice, deterministic actions. A full
                reset first (levels frozen, so it starts at the launch levels), then a FRESH exposure ledger and
                levels unfrozen, then --episodes x max_episode_length steps with the training command sampler. Reads
                the omni exposure ledger: reach_edge_frac >= 0.9, field_frac_mean >= 0.51, block_steps_mean (block
                steps per obstacle-tile episode) >= 230, off_tile_block_steps_mean > 0. Precondition: the policy
                carry keys STD_PARAM / INIT_STD and the CHARTER 6.1 validity keys (CLOCK_HIDE, SPEED_ELLIPSE,
                FACING_OBS_CLIP) in this process equal the anchor stamp's run_meta env_vars. Any other EVAL_CARRY
                difference is reported (``carry_diff_info``), not failed.

    p2_smoke.py [--reference P2__A1_s18] [--num-envs 256] [--episodes 2] [--s3-resets 200] [--checks S1,S2,S3,S4]
                [--fill-from P2__CAL1_s18] [--out-root DIR] [--device cuda:0] [--allow-dirty] [--no-idle-check]
    p2_smoke.py --print-plan [...]           compose and print the plan + the S2 threshold table (CPU only; nothing
                                             launched, nothing written)
    p2_smoke.py --calibrate-s2 N             S2 on N seeded CPU replicas of the generator (the real terrain functions;
                                             CPU only, nothing written); exit 0 iff S2 passes on them

Refusals before Kit starts: the code + eval pathspecs are dirty (run_candidates.dirty_paths; --allow-dirty is for a
dry look only, and SMOKES_OK refuses it), the head's sha256 differs from anchors.yaml, or another Isaac driver /
eval / train process is running (the run_arm.sh anchored match on /proc cmdlines, never ``pgrep -f``). Exit codes:
0 all requested checks PASS, 1 a check FAILED or the child crashed (smokes.json says which), 2 usage, 6 busy, 7 refused.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
R = Path("/home/nickmagus/krabby/krabby-research")
PARKOUR = R / "parkour"
RC_PATH = R / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/run_candidates.py"
PY = "/home/nickmagus/krabby/isaac_venv/bin/python"
SMOKES = CAMPAIGN / "smokes"
DEFAULT_REF, DEFAULT_FILL = "P2__A1_s18", "P2__CAL1_s18"
CHECKS = ("S1", "S2", "S3", "S4")
EPS = 1e-9

# ------------------------------------------------------------------ the CHARTER 3 / 5 values every check pins
FLAT_KEY = "parkour_flat"
FIELD_KEYS = ("omni_field_sparse", "omni_field_dense")
S1_CLASSES = {FLAT_KEY: "OmniFlatTerrainCfg", "omni_field_sparse": "OmniFieldTerrainCfg",
              "omni_field_dense": "OmniFieldTerrainCfg"}
S1_PROPORTIONS = {FLAT_KEY: 0.5, "omni_field_sparse": 0.25, "omni_field_dense": 0.25}
S1_NOISE = (0.02, 0.02)
S1_DIFFICULTY = (0.20, 0.70)
PAD_R, OUTER_R = 1.7, 4.80
S1_BLOCKS = {"omni_field_sparse": 14, "omni_field_dense": 28}
S1_BAR, S1_PIT = 0.4, 0.2                         # bar:slab:pit 0.4:0.4:0.2
COVER_PER_14 = 0.085                              # sparse ~8.5 %, dense ~17 % of the tile
COVER_BAND = 0.5                                  # +-50 %
CORNER_R = 4.5
# ---- S2 recalibration (2026-10-06, after the pit/raised 2-px gap; from the CPU replica of the generator,
# cpu_s2_calibration / --calibrate-s2, the run recorded in S2_CPU and CHARTER 9). Each bound is
# <the height's source> + <the roughness on top of it> + <rounding steps> + S2_MARGIN:
VS_S2 = 0.005                                     # vertical_scale: one height step
NOISE_NOMINAL = 0.02                              # noise_range (0.02, 0.02): the sampled grid spans +-4 steps
#: the REALISED roughness amplitude. random_uniform_terrain fits a cubic spline through the +-0.02 m grid and rints
#: it: the spline overshoots to 7 steps (a field tile's off-block max is 0.035 m on 20 % of tiles, 0.040 m on 1/4000),
#: so the nominal 0.02 m understates what the pad and the block tops carry. With 0.02 the bounds below would be
#: 0.030 (pad) and 0.035 (heights/pits), which the CPU replica FAILS: pad |h| 0.035 on 80/4000 tiles (a smoke of 100
#: field tiles fails ~87 % of the time) and a pit 0.0363 below pit_depth(d) on 6/4000 (~14 % of smokes).
NOISE_REALISED = 0.035
S2_MARGIN = 0.005
#: pad pixels carry roughness only (no block may be stamped there: checked exactly from occupancy): A + 1 step
#: (the rint) + margin = 0.035 + 0.005 + 0.005 = 0.045 m.
PAD_ABS_MAX = round(NOISE_REALISED + VS_S2 + S2_MARGIN, 6)
#: block pixels: a bar/slab value is round(h / vs) <= formula + 1/2 step, a pit -round(depth / vs) >= -(formula + 1/2
#: step), and the roughness lands on top: A + 2 steps (block rounding, rint) + margin = 0.035 + 0.01 + 0.005 = 0.050 m,
#: max height <= formula(d) + 0.050 and min height >= -(pit_depth(d) + 0.050).
HEIGHT_TOL = round(NOISE_REALISED + 2 * VS_S2 + S2_MARGIN, 6)
#: per-corner population rate (beyond r 4.5 m) pooled over a key's field tiles (was "all 4 corners on >= 80 % of
#: tiles", which sparse tiles cannot meet: all-4 on 0.415 of sparse tiles, 0.936 of dense)
CORNER_RATE_MIN = {"omni_field_sparse": 0.70, "omni_field_dense": 0.90}
FOOT_Z_TOL = 0.01
S3_RESETS = 200
S3_ZERO_S = 1.0
TOE_MAX = 1.6
S4_MIN = {"reach_edge_frac": 0.9, "field_frac_mean": 0.51, "block_steps_mean": 230.0}
S4_POSITIVE = ("off_tile_block_steps_mean",)
S4_KEYS = tuple(S4_MIN) + S4_POSITIVE
#: carry keys that must equal the anchor stamp's (the policy's std structure + the CHARTER 6.1 validity four)
S4_CARRY_STRICT = ("KRABBY_STD_PARAM", "KRABBY_INIT_STD", "KRABBY_OMNI_CLOCK_HIDE", "KRABBY_OMNI_SPEED_ELLIPSE",
                   "KRABBY_OMNI_FACING_OBS_CLIP")
#: the anchored Isaac match of run_arm.sh drivers_idle / run_phases._wait_isaac_clear
ISAAC_RE = re.compile(r"^\S*isaac_venv/bin/python \S*(run_phases|run_candidates|eval_crab_hex_gait|train|"
                      r"record_categories|play_stream_video|p2_smoke)\.py")


# ================================================================================================ small helpers
def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def is_placeholder(v) -> bool:
    """run_arm.sh's placeholder rule: a value starting "TBD" (case-insensitive) or containing "PENDING"."""
    return bool(re.match(r"(?i)^\s*TBD", str(v))) or "PENDING" in str(v).upper()


def utc_stamp(now: datetime.datetime | None = None) -> str:
    now = now or datetime.datetime.now(datetime.timezone.utc)
    return now.strftime("%Y-%m-%dT%H-%M-%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def isaac_busy(own_pid: int | None = None) -> list[str]:
    """Command lines of running Isaac driver / eval / train processes (anchored, from /proc; never pgrep -f)."""
    own = {os.getpid(), own_pid or -1}
    hits = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) in own:
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            continue
        if ISAAC_RE.match(cmd) and "--print-plan" not in cmd and "--calibrate-s2" not in cmd:
            hits.append(f"{p.name}: {cmd[:160]}")
    return hits


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _check(name, ok, detail=""):
    return [str(name), bool(ok), str(detail)]


def _result(checks: list, **extra) -> dict:
    ok = bool(checks) and all(c[1] for c in checks)
    return {"result": "PASS" if ok else "FAIL", "checks": checks, **extra}


# ================================================================================================ the plan (CPU)
def compose_plan(rc, cands: dict, ref: str, *, fill_from: str | None = DEFAULT_FILL, num_envs: int = 256,
                 anchors: dict | None = None, repo: Path = R) -> dict:
    """The env + task a p2a arm of ``ref`` trains on, composed with run_candidates' own functions (pure: no
    os.environ read, nothing written). Raises on an unknown reference or a placeholder that cannot be filled."""
    recipe = rc.reference_recipe(cands, ref)
    stages = recipe["stages"]
    if len(stages) != 1:
        raise ValueError(f"{ref}: expected one stage, got {[s['stage'] for s in stages]}")
    stage = stages[0]
    sname, base = str(stage["stage"]), str(stage["base"])
    overlay = rc.stage_env(recipe, sname)
    fill = {}
    if fill_from:
        frec = rc.reference_recipe(cands, fill_from)
        fill = rc.stage_env(frec, str(frec["stages"][0]["stage"]))
    substituted, dropped = {}, []
    for k, v in list(overlay.items()):
        if is_placeholder(v):
            if k in fill and not is_placeholder(fill[k]):
                overlay[k] = fill[k]
                substituted[k] = {"was": v, "now": fill[k], "from": fill_from}
            else:
                del overlay[k]
                dropped.append(k)
    spec = rc.ph.PHASES[base]
    plant = str(cands.get("plant") or "")
    child_env_vars = {"KRABBY_OMNI_PHASE": spec.name, "KRABBY_PLANT": plant, "PYTHONUNBUFFERED": "1"}
    child_env_vars.update(overlay)
    child_env_vars["PYTHONUNBUFFERED"] = "1"
    reserved = [k for k in ("KRABBY_PHASE", "KRABBY_SIDE_PHASE") if k in overlay]
    if reserved:
        raise ValueError(f"{ref}: the overlay sets {reserved} (one task per process)")
    effective = dict(rc.ph.phase_env(spec.name, plant))
    effective.update(overlay)
    seed = rc.reference_spec(cands, ref).get("seed")
    head = None
    if anchors is not None:
        a = (anchors.get("anchors") or {}).get(int(seed)) if seed is not None else None
        if a:
            head = {"name": a["name"], "checkpoint": str(repo / a["checkpoint"]), "sha256": a["sha256"],
                    "env_stamp": str(repo / a["env_stamp"])}
    return {"reference": ref, "chain": recipe["chain"], "stage": sname, "base": base, "task": spec.task,
            "teacher_mode": spec.teacher_mode, "plant": plant, "seed": None if seed is None else int(seed),
            "num_envs": int(num_envs), "overlay": overlay, "substituted": substituted, "dropped": dropped,
            "child_env_vars": child_env_vars, "effective_env": effective, "head": head,
            "recipe_id": recipe.get("recipe_id")}


def child_environ(rc, plan: dict, parent: dict | None = None) -> dict:
    """The child's process environment: run_phases._subenv of the plan's vars (parent KRABBY_* stripped)."""
    saved = os.environ.copy()
    try:
        if parent is not None:
            os.environ.clear()
            os.environ.update(parent)
        return rc.rp._subenv(plan["child_env_vars"])
    finally:
        os.environ.clear()
        os.environ.update(saved)


# ================================================================================================ pure judges
def judge_s1(summary: dict) -> dict:
    """``summary`` = {curriculum, difficulty_range, sub_terrains: {key: {class, proportion, noise_range,
    clear_radius, ring_outer_radius, n_blocks, bar_fraction, pit_fraction, vertical_edges}}, parkour: {clear_radius_m,
    ring_outer_radius_m}} read off the BUILT generator cfg."""
    c = []
    subs = summary.get("sub_terrains") or {}
    c.append(_check("sub-terrain keys", set(subs) == set(S1_CLASSES), sorted(subs)))
    for k, cls in S1_CLASSES.items():
        c.append(_check(f"{k} class", (subs.get(k) or {}).get("class") == cls, (subs.get(k) or {}).get("class")))
    tot = sum(_num((v or {}).get("proportion")) or 0.0 for v in subs.values())
    for k, p in S1_PROPORTIONS.items():
        got = (_num((subs.get(k) or {}).get("proportion")) or 0.0) / tot if tot > 0 else None
        c.append(_check(f"{k} proportion {p}", got is not None and abs(got - p) < 1e-6, got))
    for k, v in subs.items():
        nr = tuple(_num(x) for x in (v.get("noise_range") or ()))
        c.append(_check(f"{k} noise_range {S1_NOISE}", len(nr) == 2 and all(
            a is not None and abs(a - b) < 1e-9 for a, b in zip(nr, S1_NOISE)), nr))
        c.append(_check(f"{k} clear_radius {PAD_R}", abs((_num(v.get("clear_radius")) or -1) - PAD_R) < 1e-9,
                        v.get("clear_radius")))
        c.append(_check(f"{k} ring_outer_radius {OUTER_R}",
                        abs((_num(v.get("ring_outer_radius")) or -1) - OUTER_R) < 1e-9, v.get("ring_outer_radius")))
    for k in FIELD_KEYS:
        v = subs.get(k) or {}
        c.append(_check(f"{k} vertical_edges false", v.get("vertical_edges") is False, v.get("vertical_edges")))
        c.append(_check(f"{k} kinds 0.4:0.4:0.2", abs((_num(v.get("bar_fraction")) or -1) - S1_BAR) < 1e-9
                        and abs((_num(v.get("pit_fraction")) or -1) - S1_PIT) < 1e-9,
                        (v.get("bar_fraction"), v.get("pit_fraction"))))
        c.append(_check(f"{k} n_blocks {S1_BLOCKS[k]}", v.get("n_blocks") == S1_BLOCKS[k], v.get("n_blocks")))
    dr = tuple(_num(x) for x in (summary.get("difficulty_range") or ()))
    c.append(_check(f"difficulty_range {S1_DIFFICULTY}", len(dr) == 2 and all(
        a is not None and abs(a - b) < 1e-9 for a, b in zip(dr, S1_DIFFICULTY)), dr))
    c.append(_check("curriculum true", summary.get("curriculum") is True, summary.get("curriculum")))
    pk = summary.get("parkour") or {}
    c.append(_check(f"parkour clear_radius_m {PAD_R}", abs((_num(pk.get("clear_radius_m")) or -1) - PAD_R) < 1e-9,
                    pk.get("clear_radius_m")))
    c.append(_check(f"parkour ring_outer_radius_m {OUTER_R}",
                    abs((_num(pk.get("ring_outer_radius_m")) or -1) - OUTER_R) < 1e-9, pk.get("ring_outer_radius_m")))
    return _result(c)


def row_difficulty(row: int, rows: int, lo: float, hi: float, curriculum: bool, random_difficulty: bool) -> float:
    """The tile difficulty ParkourTerrainGenerator used for a row (an UPPER bound when it was random)."""
    if not curriculum:
        return float(hi)
    if random_difficulty:
        return float(lo + (hi - lo) * (row + 1) / rows)
    return float(lo + (hi - lo) * (row / (rows - 1) if rows > 1 else 0.0))


def tile_stats(h, occ, edge, hs: float, d: float, cfg: dict) -> dict:
    """One field tile (``h`` metres (W, L), ``occ`` bool, ``edge`` bool) against its cfg strings at difficulty d."""
    import numpy as np  # noqa: PLC0415

    h = np.asarray(h, dtype=float)
    occ = np.zeros(h.shape, dtype=bool) if occ is None else np.asarray(occ, dtype=bool)
    edge = np.zeros(h.shape, dtype=bool) if edge is None else np.asarray(edge).astype(bool)
    w, ell = h.shape
    dx, dy = np.meshgrid((np.arange(w) - (w - 1) / 2.0) * hs, (np.arange(ell) - (ell - 1) / 2.0) * hs, indexing="ij")
    r = np.hypot(dx, dy)
    pad = r <= float(cfg.get("clear_radius", PAD_R)) + EPS
    half = float(cfg.get("ring_outer_radius", OUTER_R))
    corners = []
    for sx in (1, -1):
        for sy in (1, -1):
            reg = (r > CORNER_R) & (np.sign(dx) == sx) & (np.sign(dy) == sy) & (np.abs(dx) <= half + EPS) & \
                  (np.abs(dy) <= half + EPS)
            corners.append(bool(occ[reg].any()))
    _hl, hh = eval(str(cfg["hurdle_height_range"]), {"difficulty": d})  # noqa: S307 -- the cfg's own formula strings
    step = eval(str(cfg["step_height"]), {"difficulty": d})  # noqa: S307
    pit = eval(str(cfg["pit_depth"]), {"difficulty": d})  # noqa: S307
    top = max(float(hh), float(step))
    sunk = occ & (h < 0)
    raised, pitp = occ & (h > 0), occ & (h < 0)
    near_both = 0
    if edge.any():
        from scipy.ndimage import binary_dilation  # noqa: PLC0415

        st = np.ones((5, 5), dtype=bool)
        near_both = int((edge & binary_dilation(raised, st) & binary_dilation(pitp, st)).sum())
    return {"pad_abs_max": float(np.abs(h[pad]).max()) if pad.any() else 0.0,
            "pad_occ_px": int(occ[pad].sum()),
            "coverage": float(occ.mean()), "corners": corners,
            "max_h": float(h.max()), "formula_top": top, "excess": float(h.max()) - top,
            "min_h": float(h.min()), "pit_depth": float(pit), "pit_over": -float(h.min()) - float(pit),
            "sunken_px": int(sunk.sum()),
            "rough_amp": float(np.abs(h[~occ]).max()) if (~occ).any() else 0.0,
            "edge_px": int(edge.sum()), "edge_px_block_pit": near_both}


def judge_s2(tiles: list[dict], cfgs: dict) -> dict:
    """``tiles`` = [{key, row, col, d, stats (tile_stats)}] over every FIELD tile; ``cfgs`` = {key: {n_blocks, n_pit,
    ...}}."""
    c = []
    if not tiles:
        return _result([_check("field tiles present", False, 0)])
    n = len(tiles)
    st = [t["stats"] for t in tiles]
    pad_occ = [(t["row"], t["col"], s.get("pad_occ_px")) for t, s in zip(tiles, st) if s.get("pad_occ_px") != 0]
    c.append(_check("pad: no block occupancy within r <= pad radius (exact, block_masks)", not pad_occ,
                    f"tiles with pad occupancy {len(pad_occ)}/{n} {pad_occ[:8]}"))
    pad_bad = [(t["row"], t["col"], round(s["pad_abs_max"], 4)) for t, s in zip(tiles, st) if s["pad_abs_max"] > PAD_ABS_MAX + EPS]
    c.append(_check(f"pad |h| <= {PAD_ABS_MAX:.3f} m on every field tile", not pad_bad,
                    f"max {max(s['pad_abs_max'] for s in st):.4f} (CPU {S2_CPU['pad_abs_max']:.4f}); bad {pad_bad[:8]}"))
    cov_bad, cov_by = [], {}
    for t, s in zip(tiles, st):
        want = COVER_PER_14 * int(cfgs[t["key"]]["n_blocks"]) / 14.0
        lo, hi = (1 - COVER_BAND) * want, (1 + COVER_BAND) * want
        cov_by.setdefault(t["key"], []).append(s["coverage"])
        if not lo - EPS <= s["coverage"] <= hi + EPS:
            cov_bad.append((t["key"], t["row"], t["col"], round(s["coverage"], 4)))
    c.append(_check("coverage per field tile within +-50 % of 8.5 % per 14 blocks", not cov_bad,
                    "; ".join(f"{k} mean {sum(v) / len(v):.4f} [{min(v):.4f}, {max(v):.4f}]" for k, v in cov_by.items())
                    + (f"; bad {cov_bad[:8]}" if cov_bad else "")))
    corner_rate, all4_by = {}, {}
    for k in sorted({t["key"] for t in tiles}):
        sk = [s for t, s in zip(tiles, st) if t["key"] == k]
        corner_rate[k] = sum(sum(s["corners"]) for s in sk) / (4.0 * len(sk))
        all4_by[k] = sum(all(s["corners"]) for s in sk) / len(sk)
    corner_bad = {k: round(v, 3) for k, v in corner_rate.items() if v < CORNER_RATE_MIN.get(k, 1.0) - EPS}
    c.append(_check("per-corner population rate beyond r "
                    f"{CORNER_R} m, pooled over field tiles: "
                    + ", ".join(f"{k.rsplit('_', 1)[-1]} >= {v:.2f}" for k, v in CORNER_RATE_MIN.items()),
                    not corner_bad and set(corner_rate) <= set(CORNER_RATE_MIN),
                    "; ".join(f"{k} {v:.3f} (CPU {S2_CPU['corner_rate'].get(k, float('nan')):.3f}; all 4 on "
                              f"{all4_by[k]:.3f}, info)" for k, v in corner_rate.items())
                    + (f"; below {corner_bad}" if corner_bad else "")))
    h_bad = [(t["row"], t["col"], round(s["excess"], 4)) for t, s in zip(tiles, st) if s["excess"] > HEIGHT_TOL + EPS]
    c.append(_check(f"max height <= formula(d) + {HEIGHT_TOL:.3f} m", not h_bad,
                    f"max excess {max(s['excess'] for s in st):.4f} (CPU {S2_CPU['max_excess']:.4f}); bad {len(h_bad)}/{n} "
                    f"{h_bad[:6]}; excess within the tile's own off-block roughness amplitude on "
                    f"{sum(s['excess'] <= s['rough_amp'] + EPS for s in st)}/{n} (info)"))
    p_bad = [(t["row"], t["col"], round(s["min_h"], 4)) for t, s in zip(tiles, st)
             if s["min_h"] < -(s["pit_depth"] + HEIGHT_TOL) - EPS]
    no_pit = [(t["row"], t["col"]) for t, s in zip(tiles, st) if int(cfgs[t["key"]].get("n_pit", 0)) > 0 and s["sunken_px"] == 0]
    pit_over = max(-s["min_h"] - s["pit_depth"] for s in st)
    c.append(_check(f"pits <= 0: sunken occupied pixels on every tile with pits, none deeper than pit_depth(d) + "
                    f"{HEIGHT_TOL:.3f} m", not p_bad and not no_pit,
                    f"max over-depth {pit_over:.4f} (CPU {S2_CPU['max_pit_over']:.4f}); too deep {len(p_bad)}/{n} "
                    f"{p_bad[:6]}; no sunken pixel on {no_pit[:6]}"))
    e_tiles = [(t["row"], s["edge_px"]) for t, s in zip(tiles, st) if s["edge_px"]]
    by_row = {}
    for r_, e in e_tiles:
        by_row[r_] = by_row.get(r_, 0) + 1
    # Hard: no raised-to-pit wall (what the 2-px gap exists for). Info + cap: other edge pixels come from a single block's
    # step plus roughness (~0.2 % of d-0.70 dense tiles, not gap-related -- review 2026-10-06); a hard "== 0" would
    # spuriously FAIL ~1.3 % of smokes, so they FAIL only above EDGE_TILE_CAP of field tiles.
    bp = sum(s["edge_px_block_pit"] for s in st)
    c.append(_check("no raised-to-pit edge pixel on field tiles (ramp mode)", bp == 0,
                    f"raised-to-pit edge px {bp}; all edge px {sum(e for _, e in e_tiles)} on {len(e_tiles)}/{n} tiles "
                    f"(by row {by_row})"))
    c.append(_check(f"edge-tile share <= {EDGE_TILE_CAP:.3f} of field tiles (ramp mode; single-block risers, info)",
                    len(e_tiles) <= EDGE_TILE_CAP * n, f"{len(e_tiles)}/{n} tiles with any edge px"))
    cov = {k: [min(v), sum(v) / len(v), max(v)] for k, v in cov_by.items()}
    return _result(c, n_field_tiles=n, corner_rate=corner_rate, all4_share=all4_by,
                   per_corner_share=sum(sum(s["corners"]) for s in st) / (4.0 * n),
                   pad_abs_max=max(s["pad_abs_max"] for s in st),
                   pad_occ_px=sum(int(s.get("pad_occ_px") or 0) for s in st),
                   max_excess=max(s["excess"] for s in st), max_pit_over=pit_over,
                   rough_amp_max=max(s["rough_amp"] for s in st), coverage=cov,
                   edge_px=sum(s["edge_px"] for s in st), edge_px_block_pit=bp)


#: Single-block risers (block step + roughness) reach the 0.12 m wall threshold on ~0.2 % of d-0.70 dense tiles and
#: ~0.03 % of all field tiles (review 2026-10-06, 9000 top-row tiles); 2 % is ~60x that and still catches a generator
#: whose blocks wall broadly (pre-gap: 59-98 % of tiles at d 0.50).
EDGE_TILE_CAP = 0.02

# ================================================================================================ S2 CPU replica
#: the REAL terrain functions, path-loaded with inert package stubs (no Isaac): the golden fixture's loader
TERRAIN_LOADER = R / "tests/unit/fixtures/gen_omni_ring_golden.py"
#: OMNI_FIELD_TERRAINS_CFG's grid (rows x cols; columns by proportion: 10 flat, 5 sparse, 5 dense)
GEN_ROWS, GEN_COLS = 10, 20
#: S2 as the CPU replica predicts it (``--calibrate-s2 40``, 2026-10-06, after the pit/raised 2-px gap: 40 seeded
#: generators x 100 field tiles = 4000 tiles, rows d 0.20..0.70, roughness on; recorded in CHARTER 9).
#: ``--print-plan`` prints these next to each threshold; ``--calibrate-s2 N`` recomputes them. Per 100-tile replica
#: (one smoke's worth) the corner rate ranged sparse 0.770-0.885, dense 0.965-1.000.
S2_CPU = {"n_generators": 40, "n_field_tiles": 4000, "pad_abs_max": 0.035, "pad_occ_px": 0,
          "max_excess": 0.033, "max_pit_over": 0.0363, "rough_amp_max": 0.040,
          "corner_rate": {"omni_field_sparse": 0.8179, "omni_field_dense": 0.9839},
          "all4_share": {"omni_field_sparse": 0.415, "omni_field_dense": 0.936},
          "coverage": {"omni_field_sparse": [0.0595, 0.0840, 0.1156], "omni_field_dense": [0.1310, 0.1636, 0.2068]},
          "edge_px": 0}


def _field_stand_in(key: str):
    """The sub-terrain cfg as ``TerrainGenerator.__init__`` hands it to the function: OMNI_FIELD_TERRAINS_CFG's
    scales / slope threshold, the S1-pinned values and the OmniFieldTerrainCfg / ExtremeParkourRoughTerrainCfg
    defaults (the same stand-in the field-terrain unit tests use)."""
    import types  # noqa: PLC0415

    cfg = types.SimpleNamespace(
        border_width=0.0, horizontal_scale=0.08, vertical_scale=VS_S2, platform_len=2.5, platform_height=0.0,
        slope_threshold=1.5, edge_width_thresh=0.05, use_simplified=False, size=(10.0, 10.0), apply_roughness=True,
        apply_flat=False, downsampled_scale=0.075, noise_range=S1_NOISE, noise_step=0.005, pad_width=0.1,
        pad_height=0.0, clear_radius=PAD_R, ring_outer_radius=OUTER_R)
    if key != FLAT_KEY:
        cfg.__dict__.update(
            field_margin=0.16, n_blocks=S1_BLOCKS[key], bar_fraction=S1_BAR, pit_fraction=S1_PIT, min_gap=0.0,
            max_tries=64, bar_length_range=(1.0, 2.0), bar_thickness_range=(0.16, 0.32), slab_side_range=(0.6, 1.2),
            hurdle_height_range="0.02 + 0.05 * difficulty, 0.03 + 0.06 * difficulty",
            step_height="0.02 + 0.06 * difficulty", pit_depth="0.02 + 0.06 * difficulty", vertical_edges=False)
    return cfg


def gen_column_keys(cols: int = GEN_COLS) -> list[str]:
    """ParkourTerrainGenerator._generate_curriculum_terrains' column -> sub-terrain assignment (proportions 0.5 /
    0.25 / 0.25, in the sub_terrains order)."""
    import numpy as np  # noqa: PLC0415

    keys = list(S1_PROPORTIONS)
    cum = np.cumsum([S1_PROPORTIONS[k] for k in keys])
    return [keys[int(np.min(np.where(i / cols + 0.001 < cum)[0]))] for i in range(cols)]


def cpu_s2_calibration(n_generators: int = 4, seed0: int = 0, rows: int = GEN_ROWS, cols: int = GEN_COLS) -> dict:
    """S2 on a CPU replica of the generator: ``n_generators`` seeded grids, generated in the generator's order
    (column-major, flat columns included, so the module-global ``numpy.random`` stream advances as in the build),
    rows at the curriculum difficulties of S1_DIFFICULTY, every field tile through the REAL decorated
    ``omni_field_terrain`` (roughness on), judged by :func:`judge_s2`. Returns the judge's doc (its measured values
    are what ``S2_CPU`` records)."""
    import numpy as np  # noqa: PLC0415

    golden = _load(TERRAIN_LOADER, "_p2_smoke_terrain_loader")
    _u, _f, omni = golden.load_terrain_modules()
    keys = gen_column_keys(cols)
    cfgs = {k: _field_stand_in(k) for k in set(keys)}
    judge_cfgs = {}
    for k in FIELD_KEYS:
        n_bar, _n_slab, n_pit = omni.field_kind_counts(S1_BLOCKS[k], S1_BAR, S1_PIT)
        judge_cfgs[k] = dict(vars(cfgs[k]), n_pit=n_pit) if k in cfgs else {}
    lo, hi = S1_DIFFICULTY
    tiles = []
    for g in range(int(n_generators)):
        np.random.seed(int(seed0) + g)
        for col in range(cols):
            for row in range(rows):
                key, d = keys[col], row_difficulty(row, rows, lo, hi, True, False)
                if key == FLAT_KEY:
                    omni.omni_flat_terrain(d, cfgs[key], 8)
                    continue
                out = omni.omni_field_terrain(d, cfgs[key], 8)
                stt = tile_stats(np.asarray(out[5], dtype=float) * VS_S2, out[6], out[4], 0.08, d, judge_cfgs[key])
                tiles.append({"key": key, "row": row, "col": g * cols + col, "d": d, "stats": stt})
    res = judge_s2(tiles, judge_cfgs)
    res["n_generators"] = int(n_generators)
    return res


def s2_threshold_table(measured: dict | None = None) -> list[dict]:
    """Each S2 threshold beside the CPU-predicted value (``S2_CPU``) and, when given, a fresh measurement."""
    m = measured or {}
    rows = [
        {"check": "pad block occupancy (exact)", "threshold": "== 0 px", "cpu": S2_CPU["pad_occ_px"],
         "now": m.get("pad_occ_px")},
        {"check": "pad |h| max (m)", "threshold": f"<= {PAD_ABS_MAX:.3f}", "cpu": S2_CPU["pad_abs_max"],
         "now": m.get("pad_abs_max")},
        {"check": "max height - formula(d) (m)", "threshold": f"<= {HEIGHT_TOL:.3f}", "cpu": S2_CPU["max_excess"],
         "now": m.get("max_excess")},
        {"check": "pit depth beyond pit_depth(d) (m)", "threshold": f"<= {HEIGHT_TOL:.3f}",
         "cpu": S2_CPU["max_pit_over"], "now": m.get("max_pit_over")},
    ]
    for k, lo in CORNER_RATE_MIN.items():
        rows.append({"check": f"per-corner rate {k}", "threshold": f">= {lo:.2f}",
                     "cpu": (S2_CPU["corner_rate"] or {}).get(k), "now": (m.get("corner_rate") or {}).get(k)})
    for k in FIELD_KEYS:
        want = COVER_PER_14 * S1_BLOCKS[k] / 14.0
        rows.append({"check": f"coverage {k} [min, mean, max]",
                     "threshold": f"[{(1 - COVER_BAND) * want:.4f}, {(1 + COVER_BAND) * want:.4f}]",
                     "cpu": (S2_CPU["coverage"] or {}).get(k), "now": (m.get("coverage") or {}).get(k)})
    rows.append({"check": "raised-to-pit edge px on field tiles (ramp mode)", "threshold": "== 0", "cpu": 0,
                 "now": m.get("edge_px_block_pit")})
    rows.append({"check": "field tiles with any edge px (ramp mode)", "threshold": f"<= {EDGE_TILE_CAP:.3f} share",
                 "cpu": S2_CPU["edge_px"], "now": m.get("edge_px")})
    return rows


def judge_s3(stats: dict) -> dict:
    c = [_check(f"resets >= {S3_RESETS}", int(stats.get("resets", 0)) >= S3_RESETS, stats.get("resets")),
         _check("terrain levels as at launch (frozen through the resets)", stats.get("levels_unchanged") is True,
                stats.get("levels_unchanged")),
         _check("t = 0: no foot over occupancy", stats.get("feet_over_block_t0") == 0, stats.get("feet_over_block_t0")),
         _check(f"t = 0: foot z >= terrain z - {FOOT_Z_TOL} m", stats.get("feet_below_terrain_t0") == 0,
                f"{stats.get('feet_below_terrain_t0')} (min clearance {stats.get('min_foot_clearance_t0')})"),
         _check(f"crab_failure in the first {S3_ZERO_S:g} s with zero actions = 0", stats.get("crab_failure_1s") == 0,
                stats.get("crab_failure_1s"))]
    toe = _num(stats.get("toe_radius_max_m"))
    c.append(_check(f"max toe radius from the root <= {TOE_MAX} m", toe is not None and toe <= TOE_MAX + EPS, toe))
    return _result(c, toe_radius_max_m=toe, resets=stats.get("resets"))


def judge_s4(means: dict, counts: dict, carry_strict_diff: dict) -> dict:
    c = [_check("policy carry == the anchor stamp's (STD_PARAM / INIT_STD / CLOCK_HIDE / SPEED_ELLIPSE / "
                "FACING_OBS_CLIP)", not carry_strict_diff, carry_strict_diff or "equal"),
         _check("obstacle-tile episodes finished > 0", int(counts.get("obst_episodes") or 0) > 0, counts)]
    for k, lo in S4_MIN.items():
        v = _num(means.get(k))
        c.append(_check(f"{k} >= {lo:g}", v is not None and v >= lo - EPS, v))
    for k in S4_POSITIVE:
        v = _num(means.get(k))
        c.append(_check(f"{k} > 0", v is not None and v > 0, v))
    return _result(c, **{k: _num(means.get(k)) for k in S4_KEYS}, **counts)


def carry_diff(anchor_env: dict, live_env: dict, keys, defaults: dict) -> dict:
    """{key: (anchor, live)} for every ``keys`` entry whose values differ (a missing key reads as its default)."""
    out = {}
    for k in keys:
        a = anchor_env.get(k, defaults.get(k))
        b = live_env.get(k, defaults.get(k))
        same = a == b
        if not same:
            fa, fb = _num(a), _num(b)
            same = fa is not None and fb is not None and abs(fa - fb) < 1e-9
        if not same:
            out[k] = [a, b]
    return out


def format_s2_table(measured: dict | None = None) -> str:
    out = [f"S2 thresholds vs the CPU replica (recorded: {S2_CPU['n_field_tiles']} field tiles over "
           f"{S2_CPU['n_generators']} seeded generators, rows d {S1_DIFFICULTY[0]:.2f}..{S1_DIFFICULTY[1]:.2f})"]
    for r in s2_threshold_table(measured):
        line = f"  {r['check']:<44} {r['threshold']:<20} CPU {_fmt(r['cpu'])}"
        if measured is not None:
            line += f"   now {_fmt(r['now'])}"
        out.append(line)
    return "\n".join(out)


def _fmt(v) -> str:
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


# ================================================================================================ the child (Isaac)
def _write(stamp: Path, doc: dict) -> None:
    tmp = stamp / "smokes.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1, default=str))
    tmp.replace(stamp / "smokes.json")


def child_main(stamp: Path, argv: list[str]) -> int:  # pragma: no cover -- needs Isaac
    plan = json.loads((stamp / "plan.json").read_text())
    doc = {"stamp": str(stamp), "plan": plan, "results": {}, "finished": False, "errors": []}
    _write(stamp, doc)

    from isaaclab.app import AppLauncher  # noqa: PLC0415

    cli = _load(PARKOUR / "scripts/rsl_rl/cli_args.py", "parkour_rsl_rl_cli_args")
    ap = argparse.ArgumentParser()
    ap.add_argument("--_child", type=Path)
    cli.add_rsl_rl_args(ap)
    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args(argv)
    app = AppLauncher(args).app
    try:
        _child_checks(stamp, plan, doc, args, cli)
    except BaseException as e:  # noqa: BLE001 -- recorded, then the app is closed
        import traceback  # noqa: PLC0415

        doc["errors"].append(f"{e.__class__.__name__}: {e}")
        doc["traceback"] = traceback.format_exc()
        _write(stamp, doc)
        traceback.print_exc()
    finally:
        doc["finished"] = True
        _write(stamp, doc)
        sys.stdout.flush()
        app.close()
    return 0


def _child_checks(stamp, plan, doc, args, cli):  # pragma: no cover -- needs Isaac
    # the eval harness's path surgery: parkour/scripts would shadow pip rsl_rl
    for p in (str(PARKOUR / "scripts"), str(PARKOUR / "scripts/rsl_rl"), str(PARKOUR / "parkour_tasks")):
        while p in sys.path:
            sys.path.remove(p)
    for p in (str(R), str(PARKOUR), str(PARKOUR / "parkour_tasks")):
        sys.path.insert(0, p)
    import gymnasium as gym  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    import parkour_tasks  # noqa: F401,PLC0415
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: PLC0415
    from parkour_tasks.crab_hex_omni_task.mdp.omni_exposure import (  # noqa: PLC0415
        feet_block_flags_tiled,
        tiled_pixel_index,
    )
    from scripts.rsl_rl.runner_factory import agent_cfg_to_train_dict, make_on_policy_runner  # noqa: PLC0415
    from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper  # noqa: PLC0415

    want = set(plan["checks"])
    doc["effective_krabby_env"] = {k: v for k, v in sorted(os.environ.items()) if k.startswith("KRABBY_")}
    env_cfg = parse_env_cfg(plan["task"], device=args.device, num_envs=int(plan["num_envs"]))
    if plan.get("seed") is not None:
        env_cfg.seed = int(plan["seed"])
    agent_cfg = cli.parse_rsl_rl_cfg(plan["task"], args)
    env = ParkourRslRlVecEnvWrapper(gym.make(plan["task"], cfg=env_cfg, render_mode=None),
                                    clip_actions=agent_cfg.clip_actions)
    uw = env.unwrapped
    term = uw.parkour_manager.get_term("base_parkour")
    terrain = uw.scene.terrain
    gen = terrain.terrain_generator_class
    tg = uw.cfg.scene.terrain.terrain_generator
    robot = uw.scene["robot"]
    dev = uw.device

    # ---------------------------------------------------------------- S1
    def sub_summary(v):
        return {"class": type(v).__name__, "proportion": getattr(v, "proportion", None),
                "noise_range": list(getattr(v, "noise_range", ()) or ()),
                "clear_radius": getattr(v, "clear_radius", None), "ring_outer_radius": getattr(v, "ring_outer_radius", None),
                "n_blocks": getattr(v, "n_blocks", None), "bar_fraction": getattr(v, "bar_fraction", None),
                "pit_fraction": getattr(v, "pit_fraction", None), "vertical_edges": getattr(v, "vertical_edges", None),
                "hurdle_height_range": getattr(v, "hurdle_height_range", None),
                "step_height": getattr(v, "step_height", None), "pit_depth": getattr(v, "pit_depth", None)}
    summary = {"curriculum": bool(tg.curriculum), "difficulty_range": list(tg.difficulty_range),
               "random_difficulty": bool(getattr(tg, "random_difficulty", False)),
               "num_rows": int(tg.num_rows), "num_cols": int(tg.num_cols),
               "sub_terrains": {k: sub_summary(v) for k, v in tg.sub_terrains.items()},
               "parkour": {"clear_radius_m": term.cfg.clear_radius_m, "ring_outer_radius_m": term.cfg.ring_outer_radius_m}}
    doc["s1_summary"] = summary
    if "S1" in want:
        doc["results"]["S1"] = judge_s1(summary)
        _write(stamp, doc)

    # ---------------------------------------------------------------- S2
    if "S2" in want:
        vs, hs = float(tg.vertical_scale), float(tg.horizontal_scale)
        names = np.asarray(gen.terrain_names)[..., 0]
        hfs = np.asarray(gen.height_fields, dtype=float) * vs
        occs = None if gen.block_masks is None else np.asarray(gen.block_masks, dtype=bool)
        edges = np.asarray(gen.x_edge_maskes)
        lo, hi = (float(x) for x in tg.difficulty_range)
        cfgs = {}
        for k in FIELD_KEYS:
            if k in tg.sub_terrains:
                sc = tg.sub_terrains[k]
                n_pit = min(int(round(int(sc.n_blocks) * float(sc.pit_fraction))),
                            int(sc.n_blocks) - int(round(int(sc.n_blocks) * float(sc.bar_fraction))))
                cfgs[k] = dict(sub_summary(sc), n_pit=n_pit)
        tiles = []
        for r_ in range(names.shape[0]):
            for c_ in range(names.shape[1]):
                key = str(names[r_, c_])
                if key not in cfgs:
                    continue
                d = row_difficulty(r_, names.shape[0], lo, hi, bool(tg.curriculum),
                                   bool(getattr(tg, "random_difficulty", False)))
                stt = tile_stats(hfs[r_, c_], None if occs is None else occs[r_, c_], edges[r_, c_], hs, d, cfgs[key])
                tiles.append({"key": key, "row": r_, "col": c_, "d": d, "stats": stt})
        res = judge_s2(tiles, cfgs)
        if occs is None:
            res["checks"].append(_check("generator block_masks present", False, None))
            res["result"] = "FAIL"
        doc["results"]["S2"] = res
        doc["s2_tiles"] = [{"key": t["key"], "row": t["row"], "col": t["col"], "d": round(t["d"], 4),
                            **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in t["stats"].items()}}
                           for t in tiles]
        _write(stamp, doc)

    # ---------------------------------------------------------------- the S4 head, built BEFORE any inference-mode
    # stepping (as the eval harness builds it: right after the env, from the agent cfg of this env)
    if "S4" in want:
        head = plan["head"]
        meta = json.loads((Path(head["env_stamp"]) / "run_meta.json").read_text())
        anchor_env = meta.get("env_vars") or {}
        live_env = {k: v for k, v in os.environ.items() if k.startswith("KRABBY_")}
        strict = carry_diff(anchor_env, live_env, S4_CARRY_STRICT, plan.get("carry_defaults") or {})
        info = carry_diff(anchor_env, live_env, [k for k in plan.get("eval_carry") or [] if k not in S4_CARRY_STRICT],
                          plan.get("carry_defaults") or {})
        runner = make_on_policy_runner(env, agent_cfg_to_train_dict(agent_cfg), log_dir=None, device=dev)
        runner.load(head["checkpoint"])
        est = agent_cfg.to_dict()["estimator"]
        n_prop, n_scan, n_priv = int(est["num_prop"]), int(est["num_scan"]), int(est["num_priv_explicit"])
        policy = runner.get_inference_policy(device=dev)
        estimator = runner.get_estimator_inference_policy(device=dev)
        std_type = getattr(runner.alg.policy, "noise_std_type", None)
        if str(live_env.get("KRABBY_STD_PARAM", "")).strip() == "soft" and std_type != "soft":
            strict["noise_std_type"] = ["soft", std_type]

    # ---------------------------------------------------------------- shared readers
    foot_ids = list(term._foot_ids)
    hf_global = (torch.as_tensor(np.asarray(gen.height_fields, dtype=np.float32)) * float(tg.vertical_scale)) \
        .permute(0, 2, 1, 3).reshape(gen.height_fields.shape[0] * gen.height_fields.shape[2], -1).to(dev)
    w_px, l_px = term._tile_px

    def feet_root():
        lt = robot.root_physx_view.get_link_transforms()
        rt = robot.root_physx_view.get_root_transforms()
        return lt[:, foot_ids, :3].to(dev), rt[:, :3].to(dev)

    def toe_radius(feet, root):
        return float(torch.linalg.norm(feet[..., :2] - root[:, None, :2], dim=-1).max())

    def terrain_z(feet):
        nr, nc, ix, iy, on, _off = tiled_pixel_index(
            feet[..., 0], feet[..., 1], term.env_origins[:, :2], terrain.terrain_levels, terrain.terrain_types,
            term._tile_size, term._hscale, term._tile_px, term._grid_rc)
        z = hf_global[nr * w_px + ix, nc * l_px + iy]
        return torch.where(on, z, torch.zeros_like(z))

    def zero_actions():
        return torch.zeros(uw.num_envs, uw.action_manager.total_action_dim, device=dev)

    # ---------------------------------------------------------------- S3
    if "S3" in want or "S4" in want:
        freeze0 = bool(term.cfg.freeze_terrain_levels)
        levels0 = terrain.terrain_levels.clone()
        types0 = terrain.terrain_types.clone()
    if "S3" in want:
        rounds = max(1, math.ceil(int(plan["s3_resets"]) / uw.num_envs))
        n_steps = max(1, int(round(S3_ZERO_S / float(uw.step_dt))))
        over = below = fails = 0
        min_clear, toe = float("inf"), 0.0
        cf = uw.termination_manager.get_term("crab_failure")
        term.cfg.freeze_terrain_levels = True
        try:
            with torch.inference_mode():
                for _ in range(rounds):
                    env.reset()
                    feet, root = feet_root()
                    over += int(feet_block_flags_tiled(
                        feet[..., 0], feet[..., 1], term.env_origins[:, :2], terrain.terrain_levels,
                        terrain.terrain_types, term._block_mask, term._tile_size, term._hscale, term._tile_px,
                        term._grid_rc)[0].sum())
                    clear = feet[..., 2] - terrain_z(feet)
                    below += int((clear < -FOOT_Z_TOL).any(dim=1).sum())
                    min_clear = min(min_clear, float(clear.min()))
                    toe = max(toe, toe_radius(feet, root))
                    for _s in range(n_steps):
                        env.step(zero_actions())
                        fails += int(uw.termination_manager.get_term("crab_failure").sum()) if cf is not None else 10 ** 6
                        feet, root = feet_root()
                        toe = max(toe, toe_radius(feet, root))
        finally:
            term.cfg.freeze_terrain_levels = freeze0
        stats = {"resets": rounds * uw.num_envs, "rounds": rounds, "zero_steps_per_round": n_steps,
                 "feet_over_block_t0": over, "feet_below_terrain_t0": below,
                 "min_foot_clearance_t0": round(min_clear, 4), "crab_failure_1s": fails if cf is not None else None,
                 "toe_radius_max_m": round(toe, 4),
                 "levels_unchanged": bool(torch.equal(terrain.terrain_levels, levels0)
                                          and torch.equal(terrain.terrain_types, types0))}
        doc["results"]["S3"] = judge_s3(stats)
        doc["results"]["S3"]["stats"] = stats
        _write(stamp, doc)

    # ---------------------------------------------------------------- S4
    if "S4" in want:
        term.cfg.freeze_terrain_levels = True
        try:
            with torch.inference_mode():
                env.reset()                                   # S4 starts from the launch levels
        finally:
            term.cfg.freeze_terrain_levels = freeze0
        levels_at_start = bool(torch.equal(terrain.terrain_levels, levels0))
        term.exposure = type(term.exposure)(term.n_sectors, capacity=int(getattr(term.cfg, "exposure_window", 1024)),
                                            device="cpu", field_mode=term.field_mode)
        steps = int(plan["episodes"]) * int(uw.max_episode_length)
        toe4, n_done = 0.0, 0
        with torch.inference_mode():
            obs, _ = env.get_observations()
            for _s in range(steps):
                obs[:, n_prop + n_scan: n_prop + n_scan + n_priv] = estimator.inference(obs[:, :n_prop])
                obs, _rew, dones, _ex = env.step(policy(obs, hist_encoding=True))
                n_done += int(dones.sum())
                feet, root = feet_root()
                toe4 = max(toe4, toe_radius(feet, root))
        means = term.exposure.means()
        counts = {"episodes_ended": n_done, "steps": steps, "levels_at_start_launch": levels_at_start,
                  "obst_episodes": int(term.exposure._extra["block_steps_mean"].n),
                  "toe_radius_max_walking_m": round(toe4, 4)}
        doc["results"]["S4"] = judge_s4(means, counts, strict)
        doc["results"]["S4"]["carry_diff_info"] = info
        doc["results"]["S4"]["exposure_means"] = {k: _num(v) for k, v in means.items()}
        _write(stamp, doc)
    env.close()


# ================================================================================================ the parent (CPU)
def build_plan(a, rc=None) -> dict:
    import yaml  # noqa: PLC0415

    rc = rc or _load(RC_PATH, "_p2_smoke_rc")
    cands = rc.load_candidates(Path(a.candidates)) if hasattr(rc, "load_candidates") else yaml.safe_load(open(a.candidates))
    anchors = yaml.safe_load(open(CAMPAIGN / "anchors.yaml"))
    plan = compose_plan(rc, cands, a.reference, fill_from=a.fill_from or None, num_envs=a.num_envs, anchors=anchors)
    plan.update(checks=[c for c in CHECKS if c in a.checks], episodes=int(a.episodes), s3_resets=int(a.s3_resets),
                eval_carry=list(rc.rp.EVAL_CARRY), carry_defaults=dict(rc.rp.EVAL_CARRY_DEFAULTS),
                candidates=str(a.candidates), device=a.device, s2_thresholds=s2_threshold_table())
    dirty = rc.dirty_paths(R)
    plan["code_state"] = {"code_sha": rc.code_sha(R), "eval_sha": rc.eval_sha(R), "dirty": bool(dirty),
                          "dirty_paths": dirty}
    return plan


def summary_lines(doc: dict) -> list[str]:
    out = []
    for c in doc.get("plan", {}).get("checks") or CHECKS:
        r = (doc.get("results") or {}).get(c)
        if not r:
            out.append(f"{c} FAIL (not run{': ' + doc['errors'][0] if doc.get('errors') else ''})")
            continue
        bad = [x[0] + " = " + x[2] for x in r.get("checks") or [] if not x[1]]
        extra = ""
        if c == "S3":
            extra = f" toe_radius_max {r.get('toe_radius_max_m')} m, resets {r.get('resets')}"
        if c == "S4":
            extra = " " + ", ".join(f"{k} {r.get(k)}" for k in S4_KEYS)
        out.append(f"{c} {r.get('result')}{extra}" + (f" :: {' | '.join(bad)[:900]}" if bad else ""))
    return out


def all_pass(doc: dict, need=CHECKS) -> bool:
    res = doc.get("results") or {}
    return bool(doc.get("finished")) and not doc.get("errors") and all((res.get(c) or {}).get("result") == "PASS"
                                                                        for c in need)


def parent_main(a) -> int:
    rc = _load(RC_PATH, "_p2_smoke_rc")
    plan = build_plan(a, rc)
    if a.print_plan:
        print(json.dumps(plan, indent=1, default=str))
        print(format_s2_table())
        return 0
    if plan["code_state"]["dirty"] and not a.allow_dirty:
        print(f"REFUSED: code/eval pathspecs dirty {plan['code_state']['dirty_paths']} (smokes run on landed code)")
        return 7
    head = plan.get("head")
    if "S4" in plan["checks"]:
        if not head or not Path(head["checkpoint"]).is_file():
            print(f"REFUSED: no anchor head for {a.reference} ({head})")
            return 7
        got = sha256_file(Path(head["checkpoint"]))
        if got != head["sha256"]:
            print(f"REFUSED: sha256 of {head['checkpoint']} is {got}, anchors.yaml records {head['sha256']}")
            return 7
    if not a.no_idle_check:
        busy = isaac_busy()
        if busy:
            print("BUSY: another Isaac process is running: " + " || ".join(busy))
            return 6
    stamp = Path(a.out_root) / utc_stamp()
    stamp.mkdir(parents=True, exist_ok=False)
    (stamp / "plan.json").write_text(json.dumps(plan, indent=1, default=str))
    env = child_environ(rc, plan)
    cmd = [a.python, str(Path(__file__).resolve()), "--_child", str(stamp), "--headless", "--device", a.device]
    print(f"[p2_smoke] {stamp} :: {plan['task']} x {plan['num_envs']} envs, {plan['reference']} ({plan['chain']}, "
          f"stage {plan['stage']}), substituted {plan['substituted']}", flush=True)
    with open(stamp / "child.log", "w") as log:
        proc = subprocess.Popen(cmd, cwd=PARKOUR, env=env, stdout=log, stderr=subprocess.STDOUT)

        def _fwd(sig, _frm):
            proc.send_signal(sig)
        prev = {sg: signal.signal(sg, _fwd) for sg in (signal.SIGTERM, signal.SIGINT)}
        try:
            rc_child = proc.wait()
        finally:
            for sg, h in prev.items():
                signal.signal(sg, h)
    try:
        doc = json.loads((stamp / "smokes.json").read_text())
    except (OSError, ValueError):
        doc = {"plan": plan, "results": {}, "finished": False, "errors": [f"no smokes.json (child rc {rc_child})"]}
    doc["child_rc"] = rc_child
    doc["all_pass"] = all_pass(doc, plan["checks"])
    doc["complete_s1_s4"] = all_pass(doc, CHECKS)
    _write(stamp, doc)
    lines = summary_lines(doc) + [f"SMOKES {'PASS' if doc['all_pass'] else 'FAIL'} {stamp}/smokes.json"]
    (stamp / "smokes.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if doc["all_pass"] else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--_child"]:
        return child_main(Path(argv[1]), argv)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reference", default=DEFAULT_REF)
    ap.add_argument("--fill-from", default=DEFAULT_FILL)
    ap.add_argument("--candidates", type=Path, default=CAMPAIGN / "candidates.yaml")
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--s3-resets", type=int, default=S3_RESETS)
    ap.add_argument("--checks", default=",".join(CHECKS))
    ap.add_argument("--out-root", type=Path, default=SMOKES)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--python", default=PY)
    ap.add_argument("--print-plan", action="store_true")
    ap.add_argument("--calibrate-s2", type=int, default=0, metavar="N",
                    help="CPU only: run S2 on N seeded replicas of the generator (real terrain functions) and print "
                         "each threshold beside the recorded and the fresh values; nothing launched, nothing written")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--no-idle-check", action="store_true")
    a = ap.parse_args(argv)
    a.checks = [c.strip().upper() for c in a.checks.split(",") if c.strip()]
    if not a.checks or any(c not in CHECKS for c in a.checks):
        ap.error(f"--checks must be a subset of {','.join(CHECKS)}")
    if a.num_envs < 1 or a.episodes < 1 or a.s3_resets < 1:
        ap.error("--num-envs / --episodes / --s3-resets must be positive")
    if a.calibrate_s2:
        res = cpu_s2_calibration(a.calibrate_s2)
        print(json.dumps({k: v for k, v in res.items() if k != "checks"}, indent=1, default=str))
        for name, ok, detail in res["checks"]:
            print(f"{'PASS' if ok else 'FAIL'} {name} :: {detail}")
        print(format_s2_table(res))
        return 0 if res["result"] == "PASS" else 1
    return parent_main(a)


if __name__ == "__main__":
    sys.exit(main())
