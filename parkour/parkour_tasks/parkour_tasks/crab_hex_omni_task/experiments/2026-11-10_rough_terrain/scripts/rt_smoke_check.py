#!/usr/bin/env python3
"""PLAN step 5 CPU checks (no Kit): the neutrality compare, the training-smoke verdict and the eval-smoke rows.

    rt_smoke_check.py neutrality --new STAMP --ref STAMP [--ref STAMP ...] [--out JSON]
    rt_smoke_check.py train --rt JSON --ref JSON [--out JSON]
    rt_smoke_check.py eval --rows-root LR --probe-dir DIR [--out JSON]

NEUTRALITY (B1). TDR relaunched on its saved nominal L090 field_light es1 row at the new build, every new knob off,
against the saved stamps of that row (keyed by sha256, rt_machinery.neutral_refs: the G1 cell and its closing
relaunch). Verdict, in order: ``BITWISE`` (every raw array of every episode and the scenario_metrics episode records,
volatile time fields aside, equal byte for byte to ONE saved stamp); else the repo's class rule, run through the
rod-speed campaign's own ``classflip_check.py`` against each saved stamp (eval process-state note: a first divergence
at step 0 allowed, no 1e-5 bound, the class drawn per launch): ``CLASS`` when any pair reads CLASS-FLIP; else
``FAIL``. The new stamp must be recorded at the pinned build (its code_state.json: code_sha / eval_sha ==
results/pin.json's, dirty "False"), else ``FAIL`` before any compare: a stamp left from a pre-re-pin smoke never
reads as the new build's neutrality. Across code pins classflip_check needs a bitwise match to another stamp to call a class flip (its
``same_code`` clause), so a non-bitwise teacher row here reads FAIL unless the class rule passes: the teacher has
been insensitive in every recorded launch, and a FAIL is a code-path finding for the parent.

TRAIN (B2): the rt smoke's own checks (rt_kit_train_smoke.py) plus the 2-s termination comparison against the
knobs-off reference run (PLAN item 20: spawn faults only -- a kind is flagged when its share of episodes terminating
within 2 s exceeds the reference's by more than max(0.05, 3 SE) with >= 2 such episodes; gravel is reported, never
flagged, since mu 0.2 envs may fall legitimately).

EVAL (B3): per kind x band row, the harness stamp (gate_row.sh, 4 episodes, es1) and the probe dump
(rt_kit_eval_probe.py, the same env file + manifest block + seed) -- spawn root height above the local surface at
the first recorded step within +-0.05 m of 1.085 m, no termination inside 2 s, the fall share and termination
reasons, the probe matching the harness (spawn xy, pad level), every env's tile of the row's kind with its d inside
the band, the realised dose within tolerance of the design value at that d, the three slope assertions on the
realised eval tiles (rim <= 2.0 m on every heading, median rim -> 3.0 m gradient >= 0.9 x table, max gradient along
the commanded eval track >= 0.9 x table), and mu_eff on gravel rows (U(0.2, 0.5), = the contract's keyed draw,
every shape equal, the friction input 2 mu - 1). Both halves must carry the pinned build (the stamp's
code_state.json, the probe JSON's ``build``), else the row FAILs; ``eval`` picks each slot's newest complete stamp
AT the pinned build (an older-build stamp in the slot is ignored).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rt_machinery as M  # noqa: E402

G = 9.81
VOLATILE_EP = {"started_utc"}
PAD_R_M = 1.7
SEAM_R_M = 4.4
EVAL_TRACK = ((0.0, 0.0), (2.0, 0.0), (2.0, 1.76), (3.41, 3.17))
DOSE_TOL = 0.10
HILL_TOL = (0.85, 1.15)
PROBE_XY_TOL_M = 0.05
PROBE_PAD_TOL_M = 0.03
# harness spawn terrain_z vs the probe's surface under the root at reset. 0.04 m, not 0.02: the spawn pad carries
# main's 2 cm roughness (spline overshoot to ~0.035 m), so the root-point surface and the harness pad level differ
# by up to that much on a correct build (smoke run 2: 0.0203-0.0242 m on single envs, dpad <= 0.009 m, dxy 0). The
# fallback this guards (terrain_z = root_z - 1.05) is caught by the terrain_mesh check independently.
PROBE_SURFACE_TOL_M = 0.04
MU_TOL = 1e-5
PIN_FILE = object()                         # default ``pin``: read results/pin.json (M.read_pin)


def _pin(pin):
    return M.read_pin() if pin is PIN_FILE else pin
#: slide test: the low-mu env must read near mu_eff and clearly below the 'average' combine's (mu_eff + 1) / 2
SLIDE_LOW_ABS, SLIDE_LOW_REL = 0.10, 0.25
SLIDE_HIGH_MIN_REL = 0.70


# ============================================================================== pure helpers (also used by Kit)
def fit_decel(t, v, v_min: float = 0.05) -> dict:
    """Least-squares slope of the horizontal speed over the sliding phase (samples from the first until the speed
    first drops below ``v_min``): ``mu_meas = -slope / g``."""
    t = np.asarray(t, np.float64)
    v = np.asarray(v, np.float64)
    stop = int(np.argmax(v < v_min)) if (v < v_min).any() else len(v)
    rise = np.flatnonzero(np.diff(v) > 0)      # the sliding phase also ends at the first rebound (a pivoting body)
    if rise.size:
        stop = min(stop, int(rise[0]) + 1)
    tt, vv = t[:stop], v[:stop]
    if len(tt) < 3:
        return {"n": int(len(tt)), "mu_meas": None, "why": "fewer than 3 sliding samples"}
    a = np.polyfit(tt - tt[0], vv, 1)[0]
    return {"n": int(len(tt)), "decel": float(-a), "mu_meas": float(-a / G), "v0": float(vv[0]),
            "t_stop": float(tt[-1] - tt[0])}


def slide_verdict(mu_meas, mu_eff: float, low: bool) -> dict:
    """Contact-level combine check. Low-mu env: |mu_meas - mu_eff| <= 0.10 + 0.25 mu_eff AND below the midpoint to
    the 'average' combine's (mu_eff + 1) / 2 (the PhysX priority under test: the ground's multiply must win over
    the robot's average). High-mu env: mu_meas >= 0.7 mu_eff (tipping and rotation make the high read noisy)."""
    if mu_meas is None:
        return {"pass": False, "why": "no fit"}
    if low:
        avg = 0.5 * (mu_eff + 1.0)
        ok = abs(mu_meas - mu_eff) <= SLIDE_LOW_ABS + SLIDE_LOW_REL * mu_eff and mu_meas < 0.5 * (mu_eff + avg)
        return {"pass": bool(ok), "mu_eff": mu_eff, "mu_meas": mu_meas, "average_combine_would_read": avg}
    # High-mu env: REPORT ONLY (CHARTER 2026-11-10 step-5 entry). The first smoke showed the high-friction body
    # pivots on stuck feet (speed falls, then rebounds), so a root-velocity slide does not measure sliding friction
    # there; the low-mu gravel read is the decisive combine-priority measurement.
    return {"pass": True, "report_only": True, "mu_eff": mu_eff, "mu_meas": mu_meas,
            "would_pass_old_line": bool(mu_meas >= SLIDE_HIGH_MIN_REL * mu_eff)}


def within_spawn(values, target: float = M.SPAWN_Z_M, tol: float = M.SPAWN_TOL_M) -> dict:
    v = np.asarray(list(values), np.float64)
    if not v.size:
        return {"n": 0, "pass": False, "why": "no samples"}
    bad = np.abs(v - target) > tol
    return {"n": int(v.size), "pass": not bool(bad.any()), "n_out": int(bad.sum()), "min": float(v.min()),
            "max": float(v.max()), "median": float(np.median(v)), "target": target, "tol": tol}


def early_flags(rt_kinds: dict, ref_kinds: dict, exempt=("gravel",)) -> dict:
    """PLAN item 20: per kind ``{"episodes", "early"}`` against the reference's pooled share."""
    re_n = sum(int(v.get("episodes", 0)) for v in ref_kinds.values())
    re_k = sum(int(v.get("early", 0)) for v in ref_kinds.values())
    p = re_k / re_n if re_n else 0.0
    out = {"ref": {"episodes": re_n, "early": re_k, "share": p}, "kinds": {}, "flagged": []}
    for kind, v in sorted(rt_kinds.items()):
        n, k = int(v.get("episodes", 0)), int(v.get("early", 0))
        s = k / n if n else None
        se = math.sqrt(max(p * (1 - p), 1e-12) / n) if n else None
        lim = None if n == 0 else p + max(0.05, 3 * se)
        flag = bool(n and s > lim and k >= 2)
        rec = {"episodes": n, "early": k, "share": s, "limit": lim, "flag": flag and kind not in exempt}
        if flag and kind in exempt:
            rec["note"] = "above the reference but exempt: low-mu gravel may fall legitimately (report only)"
        out["kinds"][kind] = rec
        if rec["flag"]:
            out["flagged"].append(kind)
    return out


# ============================================================================== neutrality
def _episodes(st: Path) -> list:
    return json.loads((Path(st) / "scenario_metrics.json").read_text())["episodes"]


def bitwise_compare(a: Path, b: Path) -> dict:
    """Every raw array of every episode byte for byte, and the scenario_metrics episode records (minus
    :data:`VOLATILE_EP`)."""
    ea, eb = _episodes(a), _episodes(b)
    ia = sorted(int(e["env_index"]) for e in ea)
    if ia != sorted(int(e["env_index"]) for e in eb):
        return {"equal": False, "why": "episode sets differ"}
    first = None
    for i in ia:
        with np.load(Path(a) / "raw" / f"episode_{i:02d}.npz") as za, np.load(Path(b) / "raw" / f"episode_{i:02d}.npz") as zb:
            if set(za.files) != set(zb.files):
                return {"equal": False, "why": f"episode {i}: npz keys differ {sorted(set(za.files) ^ set(zb.files))}"}
            for k in za.files:
                u, w = za[k], zb[k]
                if u.dtype != w.dtype or u.shape != w.shape or u.tobytes() != w.tobytes():
                    step = None
                    if u.shape == w.shape and u.ndim >= 1 and u.dtype == w.dtype:
                        rows = np.any(np.ascontiguousarray(u).view(np.uint8).reshape(len(u), -1)
                                      != np.ascontiguousarray(w).view(np.uint8).reshape(len(w), -1), axis=1)
                        step = int(np.argmax(rows)) if rows.any() else None
                    if first is None or (step is not None and (first["step"] is None or step < first["step"])):
                        first = {"episode": i, "key": k, "step": step}
    if first:
        return {"equal": False, "first_diff": first}

    def strip(eps):
        return {int(e["env_index"]): {k: v for k, v in e.items() if k not in VOLATILE_EP} for e in eps}

    if strip(ea) != strip(eb):
        diff = [i for i in ia if strip(ea)[i] != strip(eb)[i]]
        return {"equal": False, "why": f"scenario_metrics episode records differ at {diff[:10]}"}
    return {"equal": True, "episodes": len(ia)}


def classflip(new: Path, ref: Path) -> dict:
    r = subprocess.run([M.PY, "-I", "-B", str(M.CLASSFLIP), str(new), str(ref)], capture_output=True, text=True,
                       timeout=3600)
    last = (r.stdout.strip().splitlines() or ["{}"])[-1]
    try:
        out = json.loads(last)
    except ValueError:
        out = {"verdict": None, "error": (r.stderr or r.stdout)[-800:]}
    out["rc"] = r.returncode
    return out


def neutrality(new: Path, refs: list, pin=PIN_FILE) -> dict:
    pin = _pin(pin)
    out = {"new": str(new), "refs": [str(r) for r in refs], "checkpoint_sha256": None, "pairs": [],
           "new_build": M.stamp_build(new), "pin": {k: (pin or {}).get(k) for k in ("code_sha", "eval_sha")}}
    off = M.build_mismatch(out["new_build"], pin)
    if off:
        out.update(verdict="FAIL", why=f"the new stamp is not at the pinned build: {off}")
        return out
    if not refs:
        out.update(verdict="FAIL", why="no saved TDR nominal L090 field_light es1 stamp found (rt_machinery.neutral_refs)")
        return out
    meta = json.loads((Path(new) / "run_meta.json").read_text())
    out["checkpoint_sha256"] = meta.get("checkpoint_sha256")
    if meta.get("checkpoint_sha256") != M.TDR_SHA:
        out.update(verdict="FAIL", why=f"the new stamp's checkpoint sha {str(meta.get('checkpoint_sha256'))[:8]} is "
                                       "not TDR's")
        return out
    newknobs = sorted(k for k in (meta.get("env_vars") or {}) if k.startswith(("KRABBY_SITE_", "KRABBY_RT_")))
    if newknobs:
        out.update(verdict="FAIL", why=f"new knobs in the row's env: {newknobs}")
        return out
    for r in refs:
        rm = json.loads((Path(r) / "run_meta.json").read_text())
        pair = {"ref": str(r), "env_equal": (rm.get("env_vars") or {}) == (meta.get("env_vars") or {}),
                "bitwise": bitwise_compare(Path(new), Path(r))}
        out["pairs"].append(pair)
    if any(p["bitwise"]["equal"] and p["env_equal"] for p in out["pairs"]):
        out["verdict"] = "BITWISE"
        return out
    for p in out["pairs"]:
        p["classflip"] = classflip(Path(new), Path(p["ref"]))
    if any((p.get("classflip") or {}).get("verdict") == "CLASS-FLIP" for p in out["pairs"]):
        out["verdict"] = "CLASS"
        out["rule"] = ("classflip_check.py (rod-speed CHARTER section 4 as amended 2026-11-08): cfg identical, dose "
                       "identical, step-0 state identical, first divergence an action at a depth frame")
    else:
        out["verdict"] = "FAIL"
        out["why"] = "not bitwise equal to any saved stamp and the class rule does not hold"
    return out


# ============================================================================== train
def train_verdict(rt: dict, ref: dict) -> dict:
    out = {"rt": {k: v.get("pass") for k, v in (rt.get("checks") or {}).items()},
           "ref_checks": {k: v.get("pass") for k, v in (ref.get("checks") or {}).items()},
           "complete": bool(rt.get("complete")) and bool(ref.get("complete"))}
    out["early"] = early_flags(rt.get("early_by_kind") or {}, ref.get("early_by_kind") or {})
    fails = [k for k, v in out["rt"].items() if not v]
    ref_fails = [k for k, v in out["ref_checks"].items() if not v]
    out["fails"] = fails
    out["ref_fails"] = ref_fails
    out["pass"] = out["complete"] and not fails and not ref_fails and not out["early"]["flagged"]
    return out


# ============================================================================== eval: realised tiles from a probe dump
def _sr():
    return M.load_by_path(M.OMNI / "mdp" / "site_raster.py", "rt_smoke_site_raster")


def _rtf():
    return M.load_by_path(M.OMNI / "mdp" / "rt_friction.py", "rt_smoke_rt_friction")


class Tile:
    """One tile of a :func:`site_raster.dump_site_layers` dump in metres, sampled in WORLD xy (bilinear inside the
    tile, the pixel frame of ``site_raster.world_pixel``)."""

    def __init__(self, dump: dict, row: int, col: int):
        self.d, self.r, self.c = dump, int(row), int(col)
        fr = dump["frame"]
        self.w, self.ell = fr.tile_px
        self.sx, self.sy = fr.size_xy
        self.hs, self.vs = fr.hs, fr.vs
        o = np.asarray(dump["terrain_origins"], np.float64)
        self.origin = o[self.r, self.c, :2]
        self.o00 = o[0, 0, :2]
        sl = (slice(self.r * self.w, (self.r + 1) * self.w), slice(self.c * self.ell, (self.c + 1) * self.ell))
        self.level = np.asarray(dump["level"][sl], np.float64) * self.vs
        self.surface0 = np.asarray(dump["surface0"][sl], np.float64) * self.vs
        self.surface = np.asarray(dump["height_field"][sl], np.float64) * self.vs
        self.kind = np.asarray(dump["kind"][sl])
        ii, jj = np.meshgrid(np.arange(self.w), np.arange(self.ell), indexing="ij")
        x0, y0 = self.origin[0] - 0.5 * self.sx, self.origin[1] - 0.5 * self.sy
        self.px_x, self.px_y = x0 + ii * self.hs - self.origin[0], y0 + jj * self.hs - self.origin[1]
        self.px_r = np.hypot(self.px_x, self.px_y)

    def sample(self, field: np.ndarray, x_w, y_w) -> np.ndarray:
        fi = (np.asarray(x_w, np.float64) - (self.origin[0] - 0.5 * self.sx)) / self.hs
        fj = (np.asarray(y_w, np.float64) - (self.origin[1] - 0.5 * self.sy)) / self.hs
        i0 = np.clip(np.floor(fi).astype(int), 0, self.w - 2)
        j0 = np.clip(np.floor(fj).astype(int), 0, self.ell - 2)
        a, b = np.clip(fi - i0, 0, 1), np.clip(fj - j0, 0, 1)
        return ((1 - a) * (1 - b) * field[i0, j0] + a * (1 - b) * field[i0 + 1, j0]
                + (1 - a) * b * field[i0, j0 + 1] + a * b * field[i0 + 1, j0 + 1])


def slope_checks(tile: Tile, grade_deg: float, spawn_xy, pad_level: float) -> dict:
    """PLAN step 2's three slope assertions on a realised eval tile (surface0, metres)."""
    g = math.tan(math.radians(grade_deg))
    cx, cy = tile.origin
    ang = np.arange(720) * (2 * np.pi / 720)
    rr = np.arange(0.0, 3.0 + 1e-9, 0.02)
    xs, ys = cx + rr[None, :] * np.cos(ang)[:, None], cy + rr[None, :] * np.sin(ang)[:, None]
    prof = tile.sample(tile.surface0, xs, ys)
    left = np.abs(prof - pad_level) > tile.vs + 1e-9
    rim = np.where(left.any(axis=1), rr[np.argmax(left, axis=1)], np.inf)
    h17 = tile.sample(tile.surface0, cx + PAD_R_M * np.cos(ang), cy + PAD_R_M * np.sin(ang))
    h30 = tile.sample(tile.surface0, cx + 3.0 * np.cos(ang), cy + 3.0 * np.sin(ang))
    med = float(np.median(np.abs(h17 - h30) / (3.0 - PAD_R_M)))
    pts = []
    for (x0, y0), (x1, y1) in zip(EVAL_TRACK[:-1], EVAL_TRACK[1:]):
        n = int(math.ceil(math.hypot(x1 - x0, y1 - y0) / 0.02))
        t = np.arange(n) / n
        pts.append(np.stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t], 1))
    pts = np.concatenate(pts + [np.array([EVAL_TRACK[-1]])]) + np.asarray(spawn_xy, np.float64)[None, :2]
    h = tile.sample(tile.surface0, pts[:, 0], pts[:, 1])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    lag = 12
    track = np.abs(h[lag:] - h[:-lag]) / (s[lag:] - s[:-lag])
    out = {"rim_max_m": float(rim.max()), "median_grad_deg": math.degrees(math.atan(med)),
           "track_max_grad_deg": math.degrees(math.atan(float(track.max()))), "table_deg": grade_deg,
           "variant": "up_sunk_pad" if pad_level < -0.01 else ("down_raised_pad" if pad_level > 0.01 else "flat")}
    out["pass"] = bool(rim.max() <= 2.0 and med >= 0.9 * g and track.max() >= 0.9 * g)
    return out


def plane_grad_max(z: np.ndarray, hs: float, mask: np.ndarray, half: int = 2) -> float:
    from numpy.lib.stride_tricks import sliding_window_view

    k = 2 * half + 1
    off = (np.arange(k) - half) * hs
    ox, oy = np.meshgrid(off, off, indexing="ij")
    sxx = float((ox * ox).sum())
    win = sliding_window_view(z, (k, k))
    g = np.hypot((win * ox).sum(axis=(-1, -2)) / sxx, (win * oy).sum(axis=(-1, -2)) / sxx)
    m = mask[half:-half, half:-half]
    return float(g[m].max()) if m.any() else float("nan")


def realised_dose(kind: str, tile: Tile, d_site: float, spawn_xy, pad_level: float) -> dict:
    """The realised primary quantity of one eval tile against the design value at its own d."""
    spec = M.rt_spec()
    sr = _sr()
    design = spec.ladder_value(kind, d_site)
    if kind == "slopes":
        return {"design": design, **slope_checks(tile, design, spawn_xy, pad_level)}
    if kind == "hills":
        mask = (tile.px_r > PAD_R_M) & (tile.px_r < SEAM_R_M)
        got = math.degrees(math.atan(plane_grad_max(tile.surface0, tile.hs, mask)))
        ok = HILL_TOL[0] * design <= got <= HILL_TOL[1] * design
        return {"design": design, "realised": got, "estimator": "5x5 plane fit (test_rt_terrain's)", "pass": bool(ok)}
    if kind in ("ledge_up", "ledge_down"):
        got = abs(float(pad_level))
        sign_ok = (pad_level < 0) if kind == "ledge_up" else (pad_level > 0)
    elif kind == "pits":
        m = tile.kind == sr.KIND_PIT
        got = float((tile.level - tile.surface0)[m].max()) if m.any() else 0.0
        sign_ok = bool(m.any())
    elif kind == "rubble":
        m = tile.kind == sr.KIND_RUBBLE
        got = float((tile.surface0 - tile.level)[m].max()) if m.any() else 0.0
        sign_ok = bool(m.any())
    elif kind == "gravel":
        m = (tile.px_r >= PAD_R_M + 0.3) & (tile.px_r <= SEAM_R_M - 0.3)
        got = float(np.abs(tile.surface - tile.level)[m].max())
        sign_ok = True
    else:
        raise ValueError(kind)
    tol = DOSE_TOL * abs(design) + 1.5 * tile.vs
    return {"design": design, "realised": got, "tol": tol, "pass": bool(sign_ok and abs(got - design) <= tol)}


def eval_row(rid: str, row: dict, stamp: Path | None, probe_json: Path, pin=PIN_FILE) -> dict:
    pin = _pin(pin)
    spec = M.rt_spec()
    kind, band = row["kind"], row["band"]
    out = {"row": rid, "kind": kind, "band": band, "stamp": str(stamp) if stamp else None, "checks": {}}
    chk = out["checks"]
    if stamp is None:
        chk["stamp"] = {"pass": False, "why": "no complete eval-smoke stamp at the pinned build"
                                              + ("" if pin is not None else f" (no pin {M.PIN_PATH})")}
        out["pass"] = False
        return out
    off = M.build_mismatch(M.stamp_build(stamp), pin)
    chk["stamp_build"] = {"pass": not off, "why": off}
    if off:
        out["pass"] = False
        return out
    eps = _episodes(stamp)
    spawn_z, early, reasons, harness = [], [], collections.Counter(), {}
    for e in eps:
        i = int(e["env_index"])
        with np.load(Path(stamp) / "raw" / f"episode_{i:02d}.npz") as z:
            r = np.asarray(z["root_pos_w"], np.float64)
            tz = np.asarray(z["terrain_z"], np.float64)
            dt = float(z["dt"]) if "dt" in z.files else None
        if dt is None:
            dt = float(json.loads((Path(stamp) / "run_meta.json").read_text()).get("dt") or 0.02)
        spawn_z.append(float(r[0, 2] - tz[0].mean()))
        reason = str(e["termination_reason"])
        reasons[reason] += 1
        if reason != "schedule_complete" and int(e["n_steps"]) * dt < M.EARLY_S:
            early.append(i)
        harness[i] = {"spawn_xy": r[0, :2].tolist(), "pad": float(tz[0].mean())}
    chk["spawn_height"] = within_spawn(spawn_z)
    # the harness falls back to terrain_z = root_z - 1.05 when it has no raycast mesh (eval_crab_hex_gait.py): that is
    # inside 1.085 +- 0.05 by construction, so the spawn check means nothing then -- the stamp must say it used the mesh
    try:
        src = json.loads((Path(stamp) / "run_meta.json").read_text()).get("terrain_source")
    except (OSError, ValueError):
        src = None
    chk["terrain_mesh"] = {"pass": src == "raycast_mesh", "terrain_source": src,
                           "why": None if src == "raycast_mesh" else "the stamp's run_meta.terrain_source is not "
                           "'raycast_mesh' (root-offset fallback or unreadable): terrain_z is root_z - 1.05, so the "
                           "spawn height is not measured"}
    if src != "raycast_mesh":
        chk["spawn_height"] = {**chk["spawn_height"], "pass": False, "why": "terrain_z is the harness's root-offset "
                               "fallback (no raycast mesh), not a measured surface"}
    chk["no_early_termination"] = {"pass": not early, "early_envs": early, "first_s": M.EARLY_S}
    out["fall_share"] = 1.0 - reasons.get("schedule_complete", 0) / max(len(eps), 1)
    out["termination_reasons"] = dict(sorted(reasons.items()))
    if not Path(probe_json).is_file():
        chk["probe"] = {"pass": False, "why": f"no probe dump {probe_json}"}
        out["pass"] = False
        return out
    pj = json.loads(Path(probe_json).read_text())
    if not pj.get("complete"):
        chk["probe"] = {"pass": False, "why": "probe dump incomplete"}
        out["pass"] = False
        return out
    off = M.build_mismatch(pj.get("build"), pin)
    chk["probe_build"] = {"pass": not off, "why": off, "build": pj.get("build")}
    if off:
        out["pass"] = False
        return out
    mods = pj.get("modules") or {}
    chk["probe_process_paths"] = {"pass": pj.get("pythonpath") == M.PYTHONPATH_W and bool(mods) and all(
        str(Path(v).resolve()).startswith(str(M.W) + "/") for v in mods.values()), "pythonpath": pj.get("pythonpath"),
        "modules": mods}
    dump = _sr().load_layers_dump(pj["layers_npz"])
    lo, hi = spec.BANDS[band]
    blo, bhi = spec.band_range(kind, band)
    match, kinds_ok, band_ok, dose, mu = [], [], [], [], []
    for env in pj["envs"]:
        i = int(env["index"])
        h = harness.get(i)
        if h is not None:
            dxy = float(np.hypot(*(np.asarray(h["spawn_xy"]) - np.asarray(env["root_reset"][:2]))))
            dpad = abs(h["pad"] - float(env["centre_level_m"]))
            surf = env.get("surface_under_root_reset")
            dsurf = abs(h["pad"] - float(surf)) if surf is not None else None
            match.append({"env": i, "dxy": dxy, "dpad": dpad, "dsurf": dsurf,
                          "ok": dxy <= PROBE_XY_TOL_M and dpad <= PROBE_PAD_TOL_M + 0.02
                          and dsurf is not None and dsurf <= PROBE_SURFACE_TOL_M})
        kinds_ok.append(env["tile_kind"] == kind)
        d = float(env["tile_d"])
        design = spec.ladder_value(kind, d)
        band_ok.append(lo - 1e-6 <= d <= hi + 1e-6 and min(blo, bhi) - 1e-6 <= design <= max(blo, bhi) + 1e-6)
        tile = Tile(dump, env["row"], env["col"])
        rec = realised_dose(kind, tile, d, (h or {}).get("spawn_xy", env["root_reset"][:2]),
                            float(env["centre_level_m"]))
        rec.update(env=i, d=d)
        dose.append(rec)
        if row.get("rt_friction"):
            mu.append(env)
    chk["probe_matches_harness"] = {"pass": bool(match) and all(m["ok"] for m in match), "envs": match}
    chk["tile_kind"] = {"pass": all(kinds_ok), "n": len(kinds_ok)}
    chk["tile_d_in_band"] = {"pass": all(band_ok), "band_d": [lo, hi], "band_value": [blo, bhi]}
    chk["realised_dose"] = {"pass": all(r["pass"] for r in dose), "tiles": dose}
    if kind == "slopes":
        out["slopes_variant"] = dict(collections.Counter(r.get("variant") for r in dose))
    if row.get("rt_friction"):
        _tag, glo, ghi = row["rt_friction"].split(":")
        glo, ghi = float(glo), float(ghi)
        rtf = _rtf()
        p = pj.get("rt_friction_params") or {}
        idx = [int(e["index"]) for e in mu]
        want = rtf.contract_mu(pj.get("env_seed"), idx, [True] * len(idx), tuple(p.get("friction_range", (0.6, 2.0))),
                               int(p.get("num_buckets", 64)), tuple(p.get("gravel_range", (glo, ghi))))
        bad = []
        for e, w in zip(mu, want):
            m_attr = e.get("mu_eff_attr")
            ok = (m_attr is not None and glo - MU_TOL <= m_attr <= ghi + MU_TOL and abs(m_attr - w) <= MU_TOL
                  and e.get("shapes_equal") and e.get("static_eq_dynamic")
                  and abs(float(e["mu_shape_min"]) - m_attr) <= MU_TOL
                  and abs(float(e["friction_obs"]) - (2 * m_attr - 1)) <= MU_TOL)
            if not ok:
                bad.append({"env": e["index"], "mu_eff_attr": m_attr, "want": float(w), "shape_min": e["mu_shape_min"],
                            "obs": e["friction_obs"]})
        chk["gravel_mu_eff"] = {"pass": bool(mu) and not bad, "range": [glo, ghi], "bad": bad,
                                "mu": [e.get("mu_eff_attr") for e in mu]}
    else:
        chk["friction_contract_off"] = {"pass": all(e.get("mu_eff_attr") is None for e in pj["envs"]),
                                        "mu_shape": [[e["mu_shape_min"], e["mu_shape_max"]] for e in pj["envs"]]}
    out["pass"] = all(c["pass"] for c in chk.values())
    return out


def eval_all(rows_root: Path, probe_dir: Path, pin=PIN_FILE) -> dict:
    pin = _pin(pin)
    rows = M.rt_spec().load_rows()["kind_rows"]
    out = {"rows": {}, "by_kind_band": {}, "pin": {k: (pin or {}).get(k) for k in ("code_sha", "eval_sha")}}
    for r in rows:
        name = M.row_name("RT5", M.TDR_SHA, r["id"])
        # the newest complete stamp AT the pinned build (none without a pin: the row then FAILs "no stamp")
        stamps = [] if pin is None else M.complete_stamps(Path(rows_root) / name / r["scenario"] / "seed001", pin=pin)
        try:
            rec = eval_row(r["id"], r, stamps[-1] if stamps else None, Path(probe_dir) / f"{r['id']}.json",
                           pin=pin)
        except Exception as exc:  # noqa: BLE001 -- one row's error is that row's FAIL
            rec = {"row": r["id"], "pass": False, "error": f"{type(exc).__name__}: {exc}"}
        out["rows"][r["id"]] = rec
        out["by_kind_band"].setdefault(r["kind"], {})[r["band"]] = {
            "pass": rec.get("pass"), "fall_share": rec.get("fall_share"),
            "termination_reasons": rec.get("termination_reasons")}
    out["fails"] = [k for k, v in out["rows"].items() if not v.get("pass")]
    out["pass"] = not out["fails"]
    return out


# ============================================================================== CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("neutrality")
    p.add_argument("--new", required=True, type=Path)
    p.add_argument("--ref", action="append", default=[], type=Path)
    p.add_argument("--out", type=Path)
    p = sub.add_parser("train")
    p.add_argument("--rt", required=True, type=Path)
    p.add_argument("--ref", required=True, type=Path)
    p.add_argument("--out", type=Path)
    p = sub.add_parser("eval")
    p.add_argument("--rows-root", required=True, type=Path)
    p.add_argument("--probe-dir", required=True, type=Path)
    p.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "neutrality":
        res = neutrality(a.new, a.ref)
        print(f"RTSMOKE NEUTRALITY verdict={res['verdict']}" + (f" why={res.get('why')}" if res.get("why") else ""))
        ok = res["verdict"] in ("BITWISE", "CLASS")
    elif a.cmd == "train":
        rt, ref = (json.loads(p.read_text()) if p.is_file() else {} for p in (a.rt, a.ref))
        res = train_verdict(rt, ref)
        print(f"RTSMOKE TRAIN pass={res['pass']} fails={res['fails']} ref_fails={res['ref_fails']} "
              f"early_flagged={res['early']['flagged']}")
        ok = res["pass"]
    else:
        res = eval_all(a.rows_root, a.probe_dir)
        for rid, r in res["rows"].items():
            bad = [k for k, c in (r.get("checks") or {}).items() if not c.get("pass")]
            print(f"  {rid} pass={r.get('pass')} fall_share={r.get('fall_share')} bad={bad or r.get('error') or ''}")
        print(f"RTSMOKE EVAL pass={res['pass']} fails={len(res['fails'])}/{len(res['rows'])}")
        ok = res["pass"]
    if a.out:
        M.write_json(a.out, res)
        print(f"RTSMOKE wrote {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
