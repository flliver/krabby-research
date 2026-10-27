#!/usr/bin/env python3
"""Stand-network campaign, PLAN.md step 9 -- the closed-loop OBSERVE PASS of a ``stand_net`` head (CPU, read-only).

PLAN.md D6 "Reported, not gated: ... the observe-pass per-branch |mu - R|"; step 9 "Evaluate both arms (..., observe
pass)"; lit review section 12 item 3 ("grade both arms on mu_hip on LIFT rows (Z2d: -0.054) beside G1/G2"). No extra
run: everything is read from the head's own eval stamps, which the ``stand_net`` launcher writes with the raw series

    sn_stand_action (T, 18)  the stand net's action (PRE-clip policy output) on stand rows, 0 elsewhere
    sn_r_target     (T, 18)  R's target (D3 constants, clamped to +-1) on the SAME row's obs, 0 elsewhere
    sn_r_branch     (T, 6)   R's per-leg branch code (crab_hex_omni_restance BRANCH_*), -1 elsewhere
    p2r_stand_mode  (T,)     the latched stand slot (the stand net acts on these rows)
    joint_pos       (T, 24)  recorded on the same row as the obs the action was computed from (harness, before
                             env.step); cams = the joints named ``*_Body_CamShaft_RevoluteJoint``, wrap_to_pi'd = R's c

ROWS: latched stand rows (``p2r_stand_mode``) of walkstop es1 / es2 and suite es1 / es2 (``--rows``), the newest
COMPLETE stamp of each (``complete.json`` or ``p2_complete.json``: p2_read.stamp_complete's rule) under
``lr/<HEAD>/<row>__omni/seed00<k>/``. A row without one is PENDING (a newer stamp still being written is named).

"APPLIED" = ``clip(sn_stand_action, -1, 1)``: the env wrapper clamps the routed action to +-clip_actions = 1.0 before
the action term (IMPLEMENTATION_MAP 0.5); R's target is clamped to +-1 by ``restance_target``.

STATISTICS (per row and pooled per family and over every READ row; legs FL FR ML MR RL RR; a leg-row = one leg on
one stand row; channel groups cam i / hip 6+i / knee 12+i):

1. ``by_branch``: per R branch (HOLD / WAIT / LIFT / HOME / SEAT of that leg) and channel group, mean and p90 of
   |applied - R target| (and the signed mean) over the leg-rows in that branch.
2. ``lift_hip``: mu_hip = the mean applied hip action on LIFT leg-rows (R's lift is ``lift`` = -0.5; Z2d read -0.054),
   the share of LIFT leg-rows with hip < -0.25 (Z2d 1.9 %), the pre-clip mean, median |c| on LIFT rows; HOME rows'
   hip (R also -0.5 there) as context.
3. ``launch``: per leg, at R's LIFT ONSETS -- the leg's branch is LIFT on a stand row and was not LIFT on the
   previous row of the SAME stand (``sn_dagger_core.lift_onset``'s rule: the first row of a stand is no onset) --
   does the applied hip go below -0.25 within 0.3 s (rows t .. t + round(0.3 / dt), inclusive)? An onset whose
   window is cut by the end of the stand (exit, fall, episode end) without a launch is CENSORED and left out of the
   share's denominator. ``launch_share = launched / (onsets - censored)``; latency median over launched onsets. The
   same read on stand ENTRY rows whose branch is LIFT is reported separately (``entry``).
4. ``hold_slope``: per leg, OLS of the applied cam action on the wrapped cam position c over HOLD leg-rows with |c|
   in [0.03, 0.3] (R's law there: -clip(k_h c, +-trim) = slope -1 clipped at +-0.1; Z2d fitted -0.04 .. -0.12), with
   intercept, R^2, the fixed point -intercept / slope, R's own OLS slope on the same rows (the saturation-aware
   reference), and the fit on the unsaturated sub-band |c| in [0.03, trim / k_h] (R exactly -1 there). Fewer than
   30 rows -> None.
5. ``counts``: episodes, episodes with stand rows, stands (maximal stand runs), stand rows, leg-rows per branch,
   onsets, entry LIFTs, the share of stand rows whose pre-clip action leaves +-1 per channel group.

VALIDITY (per stamp, fail closed -> INVALID, exit 3): ``run_meta.p2r_mode.variant == stand_net``; ``stand_mask`` an
int 0 / 1; ``specialist.sha256`` 64 hex; the ``stand_net`` block's ``r_knob`` == ``sn_common.RESTANCE_D3`` (the R the
raws were labelled with); ``raw_series`` carries every sn raw + ``p2r_stand_mode``; ``sn_common.env_reasons`` on
``run_meta.env_vars`` (D5 obs parity + ENV4); six cam joints; ``dt`` > 0. Across the head's stamps: one stand
checkpoint sha, one mask, one S1 sha (sn_calib's consistency rule). Read self-checks per stamp: the routed action
(``actions``) equals ``sn_stand_action`` on every stand row (the stand net drove them); ``sn_r_branch`` is set on
exactly the stand rows; on HOLD leg-rows R's cam target equals ``-clip(k_h c, +-trim)`` of the c read here
(<= 1e-4: c is R's own input on that row); on LIFT leg-rows R's hip target is ``lift``. The GATE validity rule
(REFS's S1 sha, the head's D5 contract for G1-G5) is ``sn_calib.py --head``'s; this read gates nothing.

CLI (CPU only; writes ONLY the two given paths):

    CUDA_VISIBLE_DEVICES="" python sn_observe_read.py --head NAME [--rows walkstop@1 walkstop@2 suite@1 suite@2]
                                                      --json OUT.json --md OUT.md [--lr DIR]

    rows: <walkstop|suite>[__omni]@<env seed>.  Exit: 0 every row READ; 2 a row PENDING (none INVALID); 3 a row
    INVALID (its reasons in the JSON; nothing from it is pooled); 1 usage.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True      # sn_common / R are loaded by path: never write __pycache__ beside them

import argparse  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[6]
LR_DEFAULT = REPO / "parkour/outputs/omni_plan/phase2/lr"
COMMON_PATH = HERE / "sn_common.py"


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


#: the shared run_meta / raw-key contract and R's D3 constants (torch-free at import; R loads on first use)
SN = _load(COMMON_PATH, "_sn_common_observe")

ROW_DIRS = {"walkstop": "walkstop__omni", "suite": "suite__omni"}
ROWS_DEFAULT = ("walkstop@1", "walkstop@2", "suite@1", "suite@2")
LEGS = ("FL", "FR", "ML", "MR", "RL", "RR")
GROUPS = ("cam", "hip", "knee")
CAM_JOINT_SUFFIX = "_Body_CamShaft_RevoluteJoint"
RAW_STAND = "p2r_stand_mode"
CLIP = 1.0                      #: the env wrapper's clip_actions (S1 agent cfg; IMPLEMENTATION_MAP 0.5)
LIFT_HIP_THR = -0.25            #: half R's lift: the Z2d / lit-review launch threshold on the hip action
LAUNCH_WINDOW_S = 0.3
SLOPE_BAND = (0.03, 0.3)        #: |c| band of the HOLD cam-trim fit (rad)
MIN_FIT_ROWS = 30
ALIGN_TOL = 1e-4                #: R's HOLD cam target vs -clip(k_h c, +-trim) of the c read here (float32 rounding)
READ, PENDING, INVALID = "READ", "PENDING", "INVALID"
RC_OK, RC_USAGE, RC_PENDING, RC_INVALID = 0, 1, 2, 3
#: reference numbers quoted in the markdown (p2r Z2d closed-loop observe, phase-2 CHARTER 2026-10-11 ~13:15)
Z2D_REF = {"mu_hip_lift": -0.054, "lift_share_below": 0.019, "hold_slope": "−0.04…−0.12"}


# ===================================================================================== R (shared module, lazy)
def rst():
    """R's module (``crab_hex_omni_restance``; imports torch) -- BRANCH_* codes, action slices, SIDE."""
    return SN.load_rst()


def r_params():
    """R's D3 params (``sn_common.r_params``: the exact validated string, never the class defaults)."""
    return SN.r_params()


def group_slices() -> dict:
    m = rst()
    return {"cam": m.CAM_ACT, "hip": m.HIP_ACT, "knee": m.KNEE_ACT}


# ===================================================================================== small pure helpers
def wrap_to_pi(x):
    """``(x + pi) mod 2 pi - pi`` (the cam observation's own wrapping; p2s_stand.wrap_to_pi)."""
    return (np.asarray(x, dtype=np.float64) + math.pi) % (2.0 * math.pi) - math.pi


def stand_runs(stand) -> list[tuple[int, int]]:
    """Maximal runs ``[start, stop)`` of True in a (T,) bool vector (one stand = one run)."""
    s = np.asarray(stand, dtype=bool).astype(np.int8)
    d = np.diff(np.r_[0, s, 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _f(x, nd=5):
    if x is None:
        return None
    x = float(x)
    return None if not math.isfinite(x) else round(x, nd)


def _stats(v) -> dict | None:
    v = np.asarray(v, dtype=np.float64)
    if not v.size:
        return None
    a = np.abs(v)
    return {"mean_abs": _f(a.mean()), "p90_abs": _f(np.quantile(a, 0.9)), "mean_signed": _f(v.mean())}


def ols(x, y, min_rows: int = MIN_FIT_ROWS) -> dict:
    """OLS ``y = slope x + intercept`` -> {n, slope, intercept, r2, fixed_point}; slope None under ``min_rows`` rows
    or a constant x. ``fixed_point`` = -intercept / slope (where the fitted action is zero)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = int(x.size)
    out = {"n": n, "slope": None, "intercept": None, "r2": None, "fixed_point": None}
    if n < min_rows:
        return out
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    if sxx <= 0.0:
        return out
    slope = float(((x - xm) * (y - ym)).sum()) / sxx
    icpt = float(ym - slope * xm)
    syy = float(((y - ym) ** 2).sum())
    resid = float(((y - (slope * x + icpt)) ** 2).sum())
    out.update(slope=_f(slope), intercept=_f(icpt), r2=_f(1.0 - resid / syy) if syy > 0 else None,
               fixed_point=_f(-icpt / slope) if abs(slope) > 1e-9 else None)
    return out


def launch_events(lift, low, window: int) -> dict:
    """LIFT onsets and stand-entry LIFTs of ONE stand run and their launch read (vectorised).

    ``lift`` (n, 6) bool: R's branch is LIFT; ``low`` (n, 6) bool: the applied hip < the threshold; ``window`` rows
    after the event row (inclusive window ``[t, t + window]``). Returns {kind: {leg, t, latency (-1 = none in the
    window), censored}} for kind ``onset`` (t >= 1, LIFT now and not on the previous row) and ``entry`` (t = 0)."""
    lift = np.asarray(lift, dtype=bool)
    low = np.asarray(low, dtype=bool)
    n = lift.shape[0]
    big = np.iinfo(np.int64).max // 4
    idx = np.where(low, np.arange(n, dtype=np.int64)[:, None], big)
    nxt = np.minimum.accumulate(idx[::-1], axis=0)[::-1] if n else idx          # first low row at or after t
    onset = np.zeros_like(lift)
    if n > 1:
        onset[1:] = lift[1:] & ~lift[:-1]
    entry = np.zeros_like(lift)
    if n:
        entry[0] = lift[0]
    out = {}
    for kind, ev in (("onset", onset), ("entry", entry)):
        t, leg = np.nonzero(ev)
        first = nxt[t, leg] if t.size else np.zeros(0, dtype=np.int64)
        lat = first - t
        launched = (first < big) & (lat <= window)
        censored = ~launched & (t + window > n - 1)
        # the leg's FIRST event of this run (np.nonzero is row-major: t ascending, so the first hit per leg)
        is_first = np.zeros(t.shape, dtype=bool)
        if t.size:
            _, i0 = np.unique(leg, return_index=True)
            is_first[i0] = True
        out[kind] = {"leg": leg.astype(np.int64), "t": t.astype(np.int64),
                     "latency": np.where(launched, lat, -1).astype(np.int64), "censored": censored,
                     "first": is_first}
    return out


# ===================================================================================== per-episode extraction
_ARRAYS = ("a", "a_raw", "tgt", "br", "c")
_EVENT = ("leg", "latency", "censored", "first")


class Acc:
    """Stand-row arrays + LIFT events of one stamp (or a pool of stamps); counts and self-checks alongside."""

    def __init__(self):
        self.parts = {k: [] for k in _ARRAYS}
        self.events = {kind: {k: [] for k in _EVENT} for kind in ("onset", "entry")}
        self.n_episodes = 0
        self.n_episodes_stand = 0
        self.n_stands = 0
        self.checks = {"routing_max_abs": 0.0, "branch_rows_ok": True, "c_align_max_abs": 0.0, "c_align_rows": 0,
                       "lift_target_max_abs": 0.0}

    def add_episode(self, d: dict, cam_cols: list[int], window: int, params, lift_code: int, hold_code: int) -> None:
        """One raw episode (a mapping of the npz keys)."""
        self.n_episodes += 1
        stand = np.asarray(d[RAW_STAND], dtype=bool)
        br_all = np.asarray(d[SN.RAW_R_BRANCH]).astype(np.int64)
        off_ok = bool((br_all[~stand] == -1).all())
        if not stand.any():
            self.checks["branch_rows_ok"] &= off_ok
            return
        self.n_episodes_stand += 1
        a_raw = np.asarray(d[SN.RAW_STAND_ACTION], dtype=np.float32)
        tgt = np.asarray(d[SN.RAW_R_TARGET], dtype=np.float32)
        act = np.asarray(d["actions"], dtype=np.float32)
        c_all = wrap_to_pi(np.asarray(d["joint_pos"])[:, cam_cols])
        self.checks["routing_max_abs"] = max(self.checks["routing_max_abs"],
                                             float(np.abs(act[stand] - a_raw[stand]).max()))
        self.checks["branch_rows_ok"] &= off_ok and bool((br_all[stand] >= 0).all())
        for s, e in stand_runs(stand):
            a_r = np.clip(a_raw[s:e], -CLIP, CLIP)
            br = br_all[s:e]
            tg = tgt[s:e]
            c = c_all[s:e]
            # self-check: c is R's own input (HOLD cam law) and R's LIFT hip target is lift
            hold = br == hold_code
            if hold.any():
                want = -np.clip(params.k_h * c, -params.trim, params.trim)
                self.checks["c_align_max_abs"] = max(self.checks["c_align_max_abs"],
                                                     float(np.abs(want - tg[:, 0:6])[hold].max()))
                self.checks["c_align_rows"] += int(hold.sum())
            lift = br == lift_code
            if lift.any():
                self.checks["lift_target_max_abs"] = max(self.checks["lift_target_max_abs"],
                                                         float(np.abs(tg[:, 6:12][lift] - params.lift).max()))
            self.parts["a"].append(a_r)
            self.parts["a_raw"].append(a_raw[s:e])
            self.parts["tgt"].append(tg)
            self.parts["br"].append(br.astype(np.int8))
            self.parts["c"].append(c.astype(np.float32))
            ev = launch_events(lift, a_r[:, 6:12] < LIFT_HIP_THR, window)
            for kind, blk in ev.items():
                for k in _EVENT:
                    self.events[kind][k].append(blk[k])
            self.n_stands += 1

    def arrays(self) -> dict:
        out = {}
        for k, parts in self.parts.items():
            width = 6 if k in ("br", "c") else 18
            dtype = np.int8 if k == "br" else np.float32
            out[k] = np.concatenate(parts) if parts else np.zeros((0, width), dtype=dtype)
        ev = {}
        for kind, blk in self.events.items():
            ev[kind] = {k: (np.concatenate(v) if v else np.zeros(0, dtype=bool if k in ("censored", "first")
                                                                    else np.int64))
                        for k, v in blk.items()}
        out["events"] = ev
        return out

    @classmethod
    def pool(cls, accs: list["Acc"]) -> "Acc":
        p = cls()
        for a in accs:
            for k in _ARRAYS:
                p.parts[k] += a.parts[k]
            for kind in p.events:
                for k in _EVENT:
                    p.events[kind][k] += a.events[kind][k]
            p.n_episodes += a.n_episodes
            p.n_episodes_stand += a.n_episodes_stand
            p.n_stands += a.n_stands
            p.checks["routing_max_abs"] = max(p.checks["routing_max_abs"], a.checks["routing_max_abs"])
            p.checks["branch_rows_ok"] &= a.checks["branch_rows_ok"]
            p.checks["c_align_max_abs"] = max(p.checks["c_align_max_abs"], a.checks["c_align_max_abs"])
            p.checks["c_align_rows"] += a.checks["c_align_rows"]
            p.checks["lift_target_max_abs"] = max(p.checks["lift_target_max_abs"], a.checks["lift_target_max_abs"])
        return p


def check_reasons(acc: Acc) -> list[str]:
    """The read's self-checks (module docstring, VALIDITY) -> reasons; [] = the raws mean what this read assumes."""
    ch = acc.checks
    out = []
    if ch["routing_max_abs"] != 0.0:
        out.append(f"the routed action differs from sn_stand_action on stand rows (max |diff| "
                   f"{ch['routing_max_abs']:.3g}): the stand net did not drive every stand row")
    if not ch["branch_rows_ok"]:
        out.append("sn_r_branch is not set on exactly the stand rows (p2r_stand_mode)")
    if ch["c_align_max_abs"] > ALIGN_TOL:
        out.append(f"R's HOLD cam target != -clip(k_h c, +-trim) of the wrapped joint_pos cam (max |diff| "
                   f"{ch['c_align_max_abs']:.3g} > {ALIGN_TOL}): c is not R's input on that row")
    if ch["lift_target_max_abs"] > 1e-6:
        out.append(f"R's hip target on LIFT leg-rows is not lift (max |diff| {ch['lift_target_max_abs']:.3g})")
    return out


# ===================================================================================== statistics (pure)
def by_branch(ar: dict) -> dict:
    names = rst().BRANCH_NAMES
    sl = group_slices()
    br = ar["br"].astype(np.int64)
    d = {g: ar["a"][:, s].astype(np.float64) - ar["tgt"][:, s].astype(np.float64) for g, s in sl.items()}
    out = {}
    for k, nm in enumerate(names):
        m = br == k
        blk = {"leg_rows": int(m.sum())}
        for g in GROUPS:
            blk[g] = _stats(d[g][m])
        out[nm] = blk
    return out


def lift_hip(ar: dict, params) -> dict:
    m_ = rst()
    br = ar["br"].astype(np.int64)
    hip = ar["a"][:, 6:12].astype(np.float64)
    hip_raw = ar["a_raw"][:, 6:12].astype(np.float64)
    cabs = np.abs(ar["c"].astype(np.float64))

    def blk(code):
        m = br == code
        n = int(m.sum())
        if not n:
            return {"leg_rows": 0, "mu_hip": None, "mu_hip_preclip": None, "median_hip": None,
                    "share_below_thr": None, "cam_abs_median": None}
        h = hip[m]
        return {"leg_rows": n, "mu_hip": _f(h.mean()), "mu_hip_preclip": _f(hip_raw[m].mean()),
                "median_hip": _f(np.median(h)), "share_below_thr": _f((h < LIFT_HIP_THR).mean()),
                "cam_abs_median": _f(np.median(cabs[m]))}

    return {"r_hip_target": params.lift, "thr": LIFT_HIP_THR, "lift": blk(m_.BRANCH_LIFT),
            "home": blk(m_.BRANCH_HOME)}


def launch_summary(ev: dict, dt: float) -> dict:
    def one(leg_mask, blk):
        lat = blk["latency"][leg_mask]
        cen = blk["censored"][leg_mask]
        n = int(lat.size)
        launched = int((lat >= 0).sum())
        censored = int(cen.sum())
        den = n - censored
        return {"events": n, "launched": launched, "censored": censored,
                "launch_share": _f(launched / den) if den > 0 else None,
                "already_low_at_event": int((lat == 0).sum()),
                "latency_median_s": _f(np.median(lat[lat >= 0]) * dt) if launched else None}

    out = {"thr": LIFT_HIP_THR, "window_s": LAUNCH_WINDOW_S}
    for kind in ("onset", "entry"):
        blk = ev[kind]
        legs = blk["leg"]
        out[kind] = {"per_leg": {LEGS[i]: one(legs == i, blk) for i in range(6)},
                     "pooled": one(np.ones(legs.shape, dtype=bool), blk),
                     # flicker-robust: only each (stand, leg)'s first event (R's tripod choice can flip LIFT <->
                     # WAIT while a leg stays down, so one un-launched lift can count as many onsets)
                     "pooled_first_per_stand_leg": one(blk["first"], blk)}
    return out


def ols_within(xs: list, ys: list, min_rows: int = MIN_FIT_ROWS) -> dict:
    """The WITHIN-group (fixed-effects) OLS slope: each group (leg) demeaned on its own, one common slope -> {n,
    slope, r2}. Pooling legs with a plain OLS would mix their intercepts (a leg biased +0.17 and one -0.17 can flatten
    two -0.7 slopes to ~0); this is the common per-leg slope instead."""
    xd = [np.asarray(x, np.float64) - np.mean(x) for x in xs if np.size(x) > 1]
    yd = [np.asarray(y, np.float64) - np.mean(y) for x, y in zip(xs, ys) if np.size(x) > 1]
    n = int(sum(np.size(x) for x in xs))
    out = {"n": n, "slope": None, "r2": None}
    if n < min_rows or not xd:
        return out
    xd, yd = np.concatenate(xd), np.concatenate(yd)
    sxx = float((xd ** 2).sum())
    if sxx <= 0.0:
        return out
    slope = float((xd * yd).sum()) / sxx
    syy = float((yd ** 2).sum())
    out.update(slope=_f(slope), r2=_f(1.0 - float(((yd - slope * xd) ** 2).sum()) / syy) if syy > 0 else None)
    return out


def hold_slope(ar: dict, params) -> dict:
    """Per leg: OLS of the applied cam on c over HOLD leg-rows with |c| in :data:`SLOPE_BAND` (+ R's own slope on the
    same rows, the unsaturated sub-band fit, the fixed point). Pooled: the within-leg slope (:func:`ols_within`) and
    the per-leg slope range."""
    hold_code = rst().BRANCH_HOLD
    br = ar["br"].astype(np.int64)
    c = ar["c"].astype(np.float64)
    a = ar["a"][:, 0:6].astype(np.float64)
    t = ar["tgt"][:, 0:6].astype(np.float64)
    lo, hi = SLOPE_BAND
    sub_hi = min(hi, params.trim / params.k_h)
    hold = br == hold_code
    cabs = np.abs(c)
    band = hold & (cabs >= lo) & (cabs <= hi)
    sub = hold & (cabs >= lo) & (cabs <= sub_hi)

    def share(i_legs):
        nh = int(hold[:, i_legs].sum())
        return nh, (_f(int(band[:, i_legs].sum()) / nh) if nh else None)

    per_leg = {}
    for i in range(6):
        m, ms = band[:, i], sub[:, i]
        f = ols(c[m, i], a[m, i])
        f["r_slope_same_rows"] = ols(c[m, i], t[m, i])["slope"]
        f["n_unsat"] = int(ms.sum())
        f["slope_unsat"] = ols(c[ms, i], a[ms, i])["slope"]
        f["hold_leg_rows"], f["band_share_of_hold"] = share([i])
        per_leg[LEGS[i]] = f

    def cols(mask, arr):
        return [arr[mask[:, i], i] for i in range(6)]

    pooled = ols_within(cols(band, c), cols(band, a))
    pooled["r_slope_same_rows"] = ols_within(cols(band, c), cols(band, t))["slope"]
    pooled["n_unsat"] = int(sub.sum())
    pooled["slope_unsat"] = ols_within(cols(sub, c), cols(sub, a))["slope"]
    pooled["hold_leg_rows"], pooled["band_share_of_hold"] = share(list(range(6)))
    sl = [f["slope"] for f in per_leg.values() if f["slope"] is not None]
    pooled["per_leg_slope_min"] = min(sl) if sl else None
    pooled["per_leg_slope_max"] = max(sl) if sl else None
    pooled["intercept"] = pooled["fixed_point"] = None          # per leg only (the within fit has none)
    return {"band": [lo, hi], "unsat_band": [lo, sub_hi], "r_slope": -params.k_h, "r_clip": params.trim,
            "pooled_rule": "within-leg (fixed-effects) OLS: each leg demeaned, one common slope",
            "per_leg": per_leg, "pooled": pooled}


def counts(acc: Acc, ar: dict) -> dict:
    names = rst().BRANCH_NAMES
    sl = group_slices()
    br = ar["br"].astype(np.int64)
    ev = ar["events"]
    raw = ar["a_raw"]
    return {"episodes": acc.n_episodes, "episodes_with_stand": acc.n_episodes_stand, "stands": acc.n_stands,
            "stand_rows": int(br.shape[0]),
            "leg_rows_by_branch": {nm: int((br == k).sum()) for k, nm in enumerate(names)},
            "lift_onsets": int(ev["onset"]["leg"].size), "entry_lifts": int(ev["entry"]["leg"].size),
            "preclip_out_of_range_share": {g: (_f((np.abs(raw[:, s]) > CLIP).mean()) if raw.shape[0] else None)
                                           for g, s in sl.items()}}


def summarize(acc: Acc, dt: float, params) -> dict:
    ar = acc.arrays()
    return {"counts": counts(acc, ar), "by_branch": by_branch(ar), "lift_hip": lift_hip(ar, params),
            "launch": launch_summary(ar["events"], dt), "hold_slope": hold_slope(ar, params),
            "checks": {k: (_f(v, 8) if isinstance(v, float) else v) for k, v in acc.checks.items()}}


# ===================================================================================== stamps: resolve + validity
def parse_row(spec: str) -> tuple[str, int]:
    """``walkstop@1`` / ``suite__omni@2`` -> (``walkstop``, 1). ValueError otherwise."""
    m = re.fullmatch(r"(walkstop|suite)(?:__omni)?@([1-9][0-9]*)", str(spec).strip())
    if not m:
        raise ValueError(f"bad row {spec!r}: want <walkstop|suite>[__omni]@<env seed>")
    return m.group(1), int(m.group(2))


def stamp_complete(p: Path) -> bool:
    """p2_read.stamp_complete's rule: the harness's complete.json (suite) or run_arm.sh's p2_complete.json."""
    return (p / "complete.json").is_file() or (p / "p2_complete.json").is_file()


def resolve(lr: Path, head: str, row: str, seed: int) -> tuple[Path | None, list[str]]:
    """``(newest complete stamp or None, newer incomplete stamps)`` under ``lr/<head>/<row>__omni/seed00<k>/``."""
    d = lr / head / ROW_DIRS[row] / f"seed{seed:03d}"
    dirs = sorted(p for p in d.glob("*/") if p.is_dir()) if d.is_dir() else []
    hits = [p for p in dirs if stamp_complete(p)]
    best = hits[-1] if hits else None
    newer = [str(p) for p in dirs if not stamp_complete(p) and (best is None or p.name > best.name)]
    return best, newer


def _json(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def cam_columns(meta: dict) -> list[int] | None:
    names = meta.get("joint_names")
    if not isinstance(names, (list, tuple)):
        return None
    cols = [i for i, n in enumerate(names) if str(n).endswith(CAM_JOINT_SUFFIX)]
    return cols if len(cols) == 6 else None


def meta_reasons(meta: dict | None) -> list[str]:
    """Per-stamp validity from run_meta (module docstring, VALIDITY). [] = valid."""
    if not isinstance(meta, dict):
        return ["run_meta.json missing or unreadable"]
    out = []
    pm = meta.get("p2r_mode")
    if not isinstance(pm, dict):
        return ["no run_meta.p2r_mode (not a stand_net stamp)"]
    if pm.get("variant") != SN.VARIANT:
        out.append(f"p2r_mode.variant {pm.get('variant')!r} != {SN.VARIANT!r}")
    mk = pm.get(SN.MASK_KEY)
    if type(mk) is not int or mk not in (0, 1):
        out.append(f"p2r_mode.{SN.MASK_KEY} {mk!r} is not an int 0 / 1")
    sp = pm.get(SN.CKPT_KEY[0])
    sha = sp.get(SN.CKPT_KEY[1]) if isinstance(sp, dict) else None
    if not (isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha)):
        out.append(f"p2r_mode.{'.'.join(SN.CKPT_KEY)} missing or malformed")
    blk = pm.get(SN.BLOCK_KEY)
    if not isinstance(blk, dict):
        out.append(f"no p2r_mode.{SN.BLOCK_KEY} block")
    elif blk.get("r_knob") != SN.RESTANCE_D3:
        out.append(f"p2r_mode.{SN.BLOCK_KEY}.r_knob {blk.get('r_knob')!r} != the D3 constants (the raws' R is not D3)")
    rs = pm.get("raw_series") or []
    miss = [k for k in (RAW_STAND,) + tuple(SN.RAW_KEYS) if k not in rs]
    if miss:
        out.append(f"p2r_mode.raw_series lacks {miss}")
    env = meta.get("env_vars")
    if not isinstance(env, dict):
        out.append("run_meta.env_vars missing")
    else:
        out += [f"env: {r}" for r in SN.env_reasons(env)]
    if cam_columns(meta) is None:
        out.append(f"run_meta.joint_names does not name six *{CAM_JOINT_SUFFIX} joints")
    dt = meta.get("dt")
    if not isinstance(dt, (int, float)) or not dt > 0:
        out.append(f"run_meta.dt {dt!r} is not > 0")
    return out


def arm_of(meta: dict | None) -> dict:
    pm = (meta or {}).get("p2r_mode") or {}
    blk = pm.get(SN.BLOCK_KEY) if isinstance(pm.get(SN.BLOCK_KEY), dict) else {}
    sp = pm.get(SN.CKPT_KEY[0]) if isinstance(pm.get(SN.CKPT_KEY[0]), dict) else {}
    return {"specialist_sha256": sp.get(SN.CKPT_KEY[1]), "specialist_path": sp.get("path"),
            "stand_mask": pm.get(SN.MASK_KEY), "s1_checkpoint_sha256": pm.get("s1_checkpoint_sha256"),
            "stand_ckpt_kind": blk.get("stand_ckpt_kind"), "trained_stand_mask": blk.get("trained_stand_mask")}


def consistency_reasons(arms: dict) -> list[str]:
    """One stand checkpoint, one mask and one S1 per head (sn_calib.stand_net_consistency's rule, + S1's sha)."""
    out = []
    for key in ("specialist_sha256", "stand_mask", "s1_checkpoint_sha256"):
        vals = sorted({str(a.get(key)) for a in arms.values()})
        if len(vals) > 1:
            out.append(f"the head's stamps mix {key}: {vals} (one arm, one checkpoint, one S1 per head)")
    return out


RAW_NEED = (RAW_STAND, "actions", "joint_pos") + tuple(SN.RAW_KEYS)


def read_stamp(stamp: Path, meta: dict, params) -> tuple[Acc | None, list[str]]:
    """Every raw episode of ``stamp`` into an :class:`Acc`; ``(acc, reasons)`` -- a missing raw key is a reason."""
    m_ = rst()
    cams = cam_columns(meta)
    dt = float(meta["dt"])
    window = int(round(LAUNCH_WINDOW_S / dt))
    files = sorted((stamp / "raw").glob("episode_*.npz"))
    if not files:
        return None, ["no raw/episode_*.npz"]
    acc = Acc()
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            miss = [k for k in RAW_NEED if k not in z.files]
            if miss:
                return None, [f"{f.name} lacks raw key(s) {miss}"]
            d = {k: z[k] for k in RAW_NEED}
        acc.add_episode(d, cams, window, params, m_.BRANCH_LIFT, m_.BRANCH_HOLD)
    return acc, check_reasons(acc)


# ===================================================================================== the read
def observe(head: str, rows: list[str], lr: Path) -> dict:
    params = r_params()
    keys = []
    for spec in rows:
        row, seed = parse_row(spec)
        k = f"{row}@{seed}"
        if k not in keys:
            keys.append(k)
    out = {"head": head, "lr": str(lr), "rows_requested": keys, "r_knob": SN.RESTANCE_D3,
           "r_params": params.to_dict(), "rows": {}, "pending": [], "invalid": {}, "pooled": {}}
    stamps, metas = {}, {}
    for k in keys:
        row, seed = parse_row(k)
        st, newer = resolve(lr, head, row, seed)
        if st is None:
            msg = f"{k}: no complete stamp under {lr / head / ROW_DIRS[row] / f'seed{seed:03d}'}"
            if newer:
                msg += f" (in progress: {', '.join(Path(n).name for n in newer)})"
            out["rows"][k] = {"status": PENDING, "stamp": None, "in_progress": newer}
            out["pending"].append(msg)
            continue
        stamps[k] = st
        metas[k] = _json(st / "run_meta.json")
        out["rows"][k] = {"status": None, "stamp": str(st), "in_progress": newer}
    arms = {k: arm_of(metas[k]) for k in stamps}
    cons = consistency_reasons(arms)
    out["arm"] = next(iter(arms.values())) if arms and not cons else (arms or None)
    accs, dts = {}, {}
    for k, st in stamps.items():
        why = meta_reasons(metas[k]) + cons
        acc = None
        if not why:
            acc, why2 = read_stamp(st, metas[k], params)
            why += why2
        if why:
            out["rows"][k]["status"] = INVALID
            out["invalid"][k] = why
            continue
        dts[k] = float(metas[k]["dt"])
        accs[k] = acc
        out["rows"][k]["status"] = READ
        out["rows"][k].update(summarize(acc, dts[k], params))
    fams = {"walkstop": [k for k in accs if k.startswith("walkstop@")],
            "suite": [k for k in accs if k.startswith("suite@")], "all": list(accs)}
    for fam, ks in fams.items():
        if not ks:
            continue
        dt_set = {dts[k] for k in ks}
        if len(dt_set) != 1:
            out["pooled"][fam] = {"rows_included": ks, "error": f"rows differ in dt {sorted(dt_set)}"}
            continue
        blk = summarize(Acc.pool([accs[k] for k in ks]), dt_set.pop(), params)
        blk["rows_included"] = ks
        out["pooled"][fam] = blk
    out["status"] = ("refused" if out["invalid"] else "partial" if out["pending"] else "complete")
    return out


# ===================================================================================== markdown
def _m(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _pct(v):
    return "—" if v is None else f"{100 * v:.1f} %"


def markdown(out: dict) -> str:
    L = [f"# Observe pass — `{out['head']}`", ""]
    L.append(f"Generated {out.get('generated', '')} by `{Path(out.get('tool', 'sn_observe_read.py')).name}` "
             f"(sha {str(out.get('tool_sha256', ''))[:12]}); status **{out['status']}**. Reported, not gated "
             "(PLAN D6). Applied = clip(stand-net action, ±1); R = D3 constants on the same row.")
    arm = out.get("arm")
    if isinstance(arm, dict) and "specialist_sha256" in arm:
        L.append(f"Arm: stand ckpt `{str(arm.get('specialist_sha256'))[:12]}` ({arm.get('stand_ckpt_kind')}), "
                 f"stand_mask {arm.get('stand_mask')}, S1 `{str(arm.get('s1_checkpoint_sha256'))[:12]}`.")
    L.append("")
    L += ["| row | status | stamp | episodes | stands | stand rows | LIFT onsets |", "|---|---|---|---|---|---|---|"]
    for k, r in out["rows"].items():
        c = r.get("counts") or {}
        L.append(f"| {k} | {r['status']} | {Path(r['stamp']).name if r.get('stamp') else '—'} | "
                 f"{_m(c.get('episodes'))} | {_m(c.get('stands'))} | {_m(c.get('stand_rows'))} | "
                 f"{_m(c.get('lift_onsets'))} |")
    for p in out["pending"]:
        L.append(f"- PENDING: {p}")
    for k, why in out["invalid"].items():
        L.append(f"- INVALID {k}: " + "; ".join(why))
    L.append("")
    # headline per row + pooled
    L += ["## Headline (per row; pooled = every READ row)", "",
          f"| block | μ_hip LIFT (R {out['r_params']['lift']:g}; Z2d {Z2D_REF['mu_hip_lift']:g}) | LIFT hip < "
          f"{LIFT_HIP_THR:g} (Z2d {100 * Z2D_REF['lift_share_below']:.1f} %) | onset launch ≤ {LAUNCH_WINDOW_S:g} s "
          "| "
          f"first onset per stand-leg | HOLD cam slope per leg, min … max (R {-out['r_params']['k_h']:g}; Z2d "
          f"{Z2D_REF['hold_slope']}) | within-leg slope | R's within-leg slope, same rows | within-leg slope, unsat "
          "band |",
          "|---|---|---|---|---|---|---|---|---|"]
    blocks = [(k, r) for k, r in out["rows"].items() if r.get("status") == READ]
    blocks += [(f"pooled {f}", b) for f, b in out["pooled"].items() if "counts" in b]
    for name, b in blocks:
        lh = b["lift_hip"]["lift"]
        la = b["launch"]["onset"]["pooled"]
        lf = b["launch"]["onset"]["pooled_first_per_stand_leg"]
        hs = b["hold_slope"]["pooled"]
        L.append(f"| {name} | {_m(lh['mu_hip'])} (n {lh['leg_rows']}) | {_pct(lh['share_below_thr'])} | "
                 f"{_pct(la['launch_share'])} ({la['launched']}/{la['events'] - la['censored']}) | "
                 f"{_pct(lf['launch_share'])} ({lf['launched']}/{lf['events'] - lf['censored']}) | "
                 f"{_m(hs['per_leg_slope_min'])} … {_m(hs['per_leg_slope_max'])} | {_m(hs['slope'])} (n {hs['n']}) | "
                 f"{_m(hs['r_slope_same_rows'])} | {_m(hs['slope_unsat'])} |")
    allb = out["pooled"].get("all")
    if allb and "counts" in allb:
        L += ["", f"## |applied − R| by R branch (pooled: {', '.join(allb['rows_included'])})", "",
              "| branch | leg-rows | cam mean / p90 | hip mean / p90 | knee mean / p90 | hip signed |",
              "|---|---|---|---|---|---|"]
        for nm, blk in allb["by_branch"].items():
            cells = []
            for g in GROUPS:
                s = blk[g]
                cells.append("—" if s is None else f"{s['mean_abs']:.3f} / {s['p90_abs']:.3f}")
            hs_ = blk["hip"]
            L.append(f"| {nm.upper()} | {blk['leg_rows']} | " + " | ".join(cells)
                     + f" | {_m(None if hs_ is None else hs_['mean_signed'])} |")
        home = allb["lift_hip"]["home"]
        lift = allb["lift_hip"]["lift"]
        L += ["", f"LIFT rows: μ_hip {_m(lift['mu_hip'])} (pre-clip {_m(lift['mu_hip_preclip'])}, median "
                  f"{_m(lift['median_hip'])}), median |c| {_m(lift['cam_abs_median'])} rad. HOME rows (R −0.5 too): "
                  f"μ_hip {_m(home['mu_hip'])}, < −0.25 {_pct(home['share_below_thr'])} (n {home['leg_rows']}).",
              "", "### LIFT-onset launch per leg (hip < −0.25 within 0.3 s; censored = stand ended first)", "",
              "| leg | onsets | launched | censored | share | already low at onset | latency median (s) | "
              "entry LIFTs | entry launched |", "|---|---|---|---|---|---|---|---|---|"]
        lo_, le_ = allb["launch"]["onset"], allb["launch"]["entry"]
        rows_ = [(leg, lo_["per_leg"][leg], le_["per_leg"][leg]) for leg in LEGS]
        rows_ += [("pooled", lo_["pooled"], le_["pooled"]),
                  ("pooled, first per stand-leg", lo_["pooled_first_per_stand_leg"], None)]
        for leg, o, e in rows_:
            L.append(f"| {leg} | {o['events']} | {o['launched']} | {o['censored']} | {_pct(o['launch_share'])} | "
                     f"{o['already_low_at_event']} | {_m(o['latency_median_s'], 2)} | "
                     f"{'—' if e is None else e['events']} | {'—' if e is None else _pct(e['launch_share'])} |")
        hsb = allb["hold_slope"]
        L += ["", f"### HOLD cam-trim slope per leg (|c| in {hsb['band']} rad; applied cam on c)", "",
              "| leg | n | slope | intercept | fixed point (rad) | R² | R's slope same rows | slope on "
              f"{hsb['unsat_band']} | band share of HOLD |", "|---|---|---|---|---|---|---|---|---|"]
        for leg, f in [(leg, hsb["per_leg"][leg]) for leg in LEGS] + [("pooled (within-leg)", hsb["pooled"])]:
            L.append(f"| {leg} | {f['n']} | {_m(f['slope'])} | {_m(f['intercept'])} | {_m(f['fixed_point'])} | "
                     f"{_m(f['r2'])} | {_m(f['r_slope_same_rows'])} | {_m(f['slope_unsat'])} | "
                     f"{_pct(f['band_share_of_hold'])} |")
        ck = allb["checks"]
        L += ["", f"Self-checks (pooled): routed action = sn_stand_action (max |Δ| {ck['routing_max_abs']}); "
                  f"R's HOLD cam target vs −clip(k_h c, ±trim) max |Δ| {ck['c_align_max_abs']} over "
                  f"{ck['c_align_rows']} leg-rows; branch rows ok {ck['branch_rows_ok']}. Pre-clip |a| > 1 share: "
                  + ", ".join(f"{g} {_pct(v)}" for g, v in allb["counts"]["preclip_out_of_range_share"].items())
                  + "."]
    return "\n".join(L) + "\n"


# ===================================================================================== CLI
def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--head", required=True, help="the stand_net head name (lr/<NAME>)")
    ap.add_argument("--rows", nargs="+", default=list(ROWS_DEFAULT),
                    help="rows to read, <walkstop|suite>[__omni]@<es> (default: walkstop + suite, es1 + es2)")
    ap.add_argument("--json", type=Path, required=True, help="write the JSON here (no default: never the campaign "
                                                              "dir by accident)")
    ap.add_argument("--md", type=Path, required=True, help="write the markdown here")
    ap.add_argument("--lr", type=Path, default=LR_DEFAULT, help="the phase-2 lr dir (default %(default)s)")
    a = ap.parse_args(argv)
    try:
        for r in a.rows:
            parse_row(r)
    except ValueError as exc:
        print(f"sn_observe_read: {exc}", file=sys.stderr)
        return RC_USAGE
    for p in (a.json, a.md):
        if not p.parent.is_dir():
            print(f"sn_observe_read: output directory {p.parent} does not exist", file=sys.stderr)
            return RC_USAGE
    out = observe(a.head, a.rows, a.lr)
    out["generated"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    out["tool"] = str(Path(__file__).resolve())
    out["tool_sha256"] = _sha(Path(__file__).resolve())
    out["common_sha256"] = _sha(COMMON_PATH)
    out["restance_module_sha256"] = _sha(SN.RESTANCE_MODULE)
    out = _jsonable(out)
    a.json.write_text(json.dumps(out, indent=1, default=str))
    a.md.write_text(markdown(out))
    rc = RC_INVALID if out["invalid"] else RC_PENDING if out["pending"] else RC_OK
    print(f"sn_observe_read: {a.head} status {out['status']} -- "
          + ", ".join(f"{k} {r['status']}" for k, r in out["rows"].items()) + f"; wrote {a.json} and {a.md}")
    for p in out["pending"]:
        print(f"PENDING: {p}")
    for k, why in out["invalid"].items():
        print(f"INVALID {k}: {'; '.join(why)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
