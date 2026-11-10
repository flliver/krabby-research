"""Rod-speed DR campaign, step 4 / K2 (GPU, Kit): the DOSE SMOKE for ``KRABBY_DR_ROD_SPEED`` -- what the sim REALISES.

Builds ``Isaac-Crab-Hex-Omni-Teacher-v0`` (256 envs, seed 7) headless under the process env ``rsd_smoke.sh`` passes (the
F19 ``p3b.distill.env`` keys plus the group's knob), drives it ~600 env steps with the sweep's +-1 rod-swing action
pattern (it saturates the rod slew limiter: pitch +s, knee -s, cams +1, s flipping every 50 steps), forces 64-env
subsets to time out at steps 100 / 200 / 400 (``episode_length_buf[ids] = max_episode_length - 1``), and checks the
per-env slew ceilings the physics step actually produced against an independent CPU replica of the draw.

Per step it records ``term._rod_speed_scale[:, 0]`` (None when off) taken BEFORE the step, the per-env peak of
``|hip_rod_speed|`` and ``|knee_rod_speed|`` after it, and the dones; the DR object's ``reset`` is tapped (ids, step
argument, scale afterwards, returned log dict). Attribution rule: a step's rod speeds belong to the scale in force
BEFORE that step (a reset at the end of step t redraws for step t+1).

    group D0  KRABBY_DR_ROD_SPEED unset   the knob-off path: ``term._rod_speed_dr`` and ``_rod_speed_scale`` None, no cfg
                                          attribute, every env's realised ceiling = the rated 0.028 / 0.033 m/s (rel
                                          1e-4), no ``DR/`` log key, no ``[DR]`` line
    group D1  0.5:1.4                     the construction-time (first reset) draw equals a CPU replica of rod_speed_dr's
                                          generator bit for bit and passes bounds / mean / KS; for every (env, episode
                                          segment) max |hip / knee rod speed| = s * rated (rel 1e-4) and never above it;
                                          s changes only where an env is done; exactly the reset envs are redrawn, equal to
                                          the replica in ascending-id order; the logged ``DR/`` values equal recomputed
                                          stats; one ``[DR]`` line
    group D2  0.5:1.0:300                 construction draws exactly 1.0; the forced resets at 100 / 200 / 400 fall in
                                          [lo_eff(t), 1.0], match the replica, and ``DR/rod_speed_lo`` = lo_eff(t)

    <isaac python> rsd_smoke.py --headless --group D0|D1|D2 --out JSON [--num-envs 256] [--steps 600] [--seed 7]
    <python>       rsd_smoke.py --selftest            (CPU, no Kit: the analysis against a fake rollout of the real
                                                       rod_speed_dr / dr_knobs modules, plus mutation tests)

The JSON ends with ``"all_ok"`` and ``"complete": true`` (written last). Every check is ``{"ok", "detail"}``.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
R = HERE.parents[6]                                   # this checkout's root (scripts/ -> campaign -> experiments -> omni -> pkg -> parkour_tasks -> parkour)
TASK = "Isaac-Crab-Hex-Omni-Teacher-v0"
DR_VAR = "KRABBY_DR_ROD_SPEED"
LOG_KEYS = ("DR/rod_speed_lo", "DR/rod_speed_mean", "DR/rod_speed_min", "DR/rod_speed_max")
SALT = 0x726F64737064                                 # b"rodspd": restated here so the replica stays independent
HIP_RATED, KNEE_RATED = 0.028, 0.033                  # loaded rod speeds, m/s (the rated ceilings)
REL = 1e-4
MIN_EQ_LEN = 10                                       # shortest segment whose peak must REACH the ceiling: step 0 of a
                                                      # segment applies the zeroed action history, so it can read ~0
KS_CRIT = 1.949                                       # one-sample KS, alpha = 0.001: D_crit = KS_CRIT / sqrt(n)
FORCED = ((100, 0, 64), (200, 64, 128), (400, 128, 192))        # (loop step, first id, one past the last id)
GROUPS = {
    "D0": None,
    "D1": {"lo": 0.5, "hi": 1.4, "ramp": 0, "text": "0.5:1.4"},
    "D2": {"lo": 0.5, "hi": 1.0, "ramp": 300, "text": "0.5:1.0:300"},
}
DR_LINE = re.compile(
    r"^\[DR\] rod_speed n=(\d+) step=(\d+) bounds=\[([^,]+), ([^\]]+)\] \(lo=(\S+) hi=(\S+) ramp_steps=(\d+)\) "
    r"seed=(\d+) (.*?) sha16=([0-9a-f]{16}) module=(\S+)$")


# ===================================================================================== the independent replica
def replica_seed(env_seed: int) -> int:
    return (int(env_seed) + SALT) % (1 << 63)


def replica_bounds(step: int, lo: float, hi: float, ramp: int):
    """``(lo_eff, hi)``: lo_eff runs linearly 1.0 -> lo over ``ramp`` env steps (0 = lo from the start)."""
    if ramp <= 0:
        return lo, hi
    frac = min(max(int(step), 0), ramp) / ramp
    return 1.0 + (lo - 1.0) * frac, hi


def replica_draw(gen: torch.Generator, k: int, lo_eff: float, hi: float) -> torch.Tensor:
    u = torch.rand(int(k), generator=gen, dtype=torch.float32)
    return lo_eff + (hi - lo_eff) * u


def sha16(values: torch.Tensor) -> str:
    return hashlib.sha256(values.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()).hexdigest()[:16]


def ks_uniform01(x: np.ndarray):
    """One-sample KS statistic of ``x`` (already mapped to [0, 1]) against U(0, 1) and its alpha = 0.001 critical value."""
    xs = np.sort(np.asarray(x, dtype=np.float64))
    n = len(xs)
    i = np.arange(1, n + 1)
    d = max(float((i / n - xs).max()), float((xs - (i - 1) / n).max()))
    return d, KS_CRIT / math.sqrt(n)


class Checks:
    def __init__(self):
        self.rows = {}

    def add(self, key: str, ok, detail=""):
        self.rows[key] = {"ok": bool(ok), "detail": str(detail)}

    @property
    def ok(self) -> bool:
        return all(r["ok"] for r in self.rows.values())

    def failed(self):
        return {k: r["detail"] for k, r in self.rows.items() if not r["ok"]}


def rel_err(got, want):
    return abs(float(got) - float(want)) / max(abs(float(want)), 1e-12)


# ===================================================================================== the analysis (pure CPU)
def analyse(group: str, rec: dict, meta: dict) -> Checks:
    """All of the group's checks on a recorded rollout. ``rec``: s_before (T, N) float32 or None, hip / knee (T, N)
    per-env peaks of |rod speed|, done (T, N) bool, calls (the tapped DR resets), logs {t: {key: float}} read after the
    step (t = -1: after ``env.reset()``), d0_dr_keys (D0: any ``DR/`` key seen in a reset log). ``meta``: num_envs,
    seed, spec_env (the process env string), pre (the pre-reset term facts), lines (stdout lines), parkour_file."""
    c = Checks()
    spec = GROUPS[group]
    armed = spec is not None
    N, seed = int(meta["num_envs"]), int(meta["seed"])
    pre, lines = meta["pre"], meta["lines"]
    done = np.asarray(rec["done"], dtype=bool)
    hip, knee = np.asarray(rec["hip"], dtype=np.float64), np.asarray(rec["knee"], dtype=np.float64)
    T = done.shape[0]
    c.add("env.shape", done.shape == (T, N) and hip.shape == (T, N) and knee.shape == (T, N),
          f"T={T} N={N} done={done.shape} hip={hip.shape}")
    c.add("env.imports_this_checkout", str(Path(meta["parkour_file"]).resolve()).startswith(str(R)),
          meta["parkour_file"])
    c.add("env.rated_speeds", rel_err(meta["hip_rated"], HIP_RATED) < 1e-9 and rel_err(meta["knee_rated"], KNEE_RATED) < 1e-9,
          f"linkage hip={meta['hip_rated']} knee={meta['knee_rated']}, want {HIP_RATED} / {KNEE_RATED}")
    c.add("env.knob_env_var", (meta["spec_env"] or "") == (spec["text"] if armed else ""),
          f"{DR_VAR}={meta['spec_env']!r}, group {group} wants {spec['text'] if armed else '(unset)'!r}")
    forced_ok = []
    for ts, lo_id, hi_id in FORCED:
        if ts < T:
            forced_ok.append(bool(done[ts, lo_id:hi_id].all()))
    c.add("forced.resets_happened", len(forced_ok) == len(FORCED) and all(forced_ok),
          f"forced subsets all done at their step: {forced_ok}; total done env-steps {int(done.sum())}")

    dr_lines = [ln for ln in lines if ln.startswith("[DR]")]
    knob_lines = [ln for ln in lines if ln.startswith("[KNOB]") and DR_VAR in ln]

    # ---------------------------------------------------------------------------------------------- D0: off
    if not armed:
        c.add("d0.no_dr_object", not pre["rd_present"], f"term._rod_speed_dr present={pre['rd_present']}")
        c.add("d0.no_scale", not pre["scale_present"], f"term._rod_speed_scale present={pre['scale_present']}")
        c.add("d0.no_cfg_attribute", pre["cfg_attr"] == "ABSENT", f"actions.joint_pos.rod_speed_dr = {pre['cfg_attr']!r}")
        c.add("d0.no_dr_line", not dr_lines, dr_lines[:2])
        c.add("d0.no_dr_knob_line", not knob_lines, knob_lines[:2])
        c.add("d0.no_dr_log_keys", not rec["d0_dr_keys"], sorted(set(rec["d0_dr_keys"])))
        c.add("d0.no_tapped_calls", not rec["calls"], f"{len(rec['calls'])} calls")
        for name, arr, rated in (("hip", hip, HIP_RATED), ("knee", knee, KNEE_RATED)):
            peak = arr.max(axis=0)
            worst = float(np.abs(peak / rated - 1.0).max())
            c.add(f"d0.ceiling_{name}", worst <= REL,
                  f"per-env max |{name} rod speed| / rated {rated}: min {peak.min() / rated:.7f} max {peak.max() / rated:.7f}"
                  f" (worst rel err {worst:.2e}, tol {REL:g}) over {N} envs")
        return c

    # ---------------------------------------------------------------------------------------------- armed
    lo, hi, ramp = spec["lo"], spec["hi"], spec["ramp"]
    c.add("dr.object_built", pre["rd_present"], f"term._rod_speed_dr present={pre['rd_present']}")
    c.add("dr.scale_aliased", pre["alias"], "term._rod_speed_scale is term._rod_speed_dr.scale")
    c.add("dr.scale_layout", pre["scale_shape"] == [N, 1] and pre["scale_dtype"] == "torch.float32",
          f"shape {pre['scale_shape']} dtype {pre['scale_dtype']} device {pre['scale_device']}")
    c.add("dr.generator_cpu", pre["gen_device"] == "cpu", pre["gen_device"])
    c.add("dr.cfg_spec", pre["cfg_attr"] == {"lo": lo, "hi": hi, "ramp_steps": ramp}, repr(pre["cfg_attr"]))
    c.add("dr.object_spec", (pre["rd_lo"], pre["rd_hi"], pre["rd_ramp"]) == (lo, hi, ramp),
          f"object lo={pre['rd_lo']} hi={pre['rd_hi']} ramp={pre['rd_ramp']}")
    c.add("dr.seed", pre["rd_seed"] == replica_seed(seed) and pre["cfg_seed"] == seed,
          f"object seed {pre['rd_seed']}, replica {replica_seed(seed)}, env cfg seed {pre['cfg_seed']}")
    c.add("dr.module_under_checkout", str(Path(pre["module_file"]).resolve()).startswith(str(R)), pre["module_file"])
    c.add("dr.untouched_before_first_reset", pre["resets_before"] == 0 and pre["scale_all_one_before"],
          f"resets before env.reset() = {pre['resets_before']}, scale all 1.0 = {pre['scale_all_one_before']}")

    calls = rec["calls"]
    by_t: dict[int, list] = {}
    for call in calls:
        by_t.setdefault(call["t"], []).append(call)

    # ---- call hygiene
    first = calls[0] if calls else None
    c.add("calls.first_is_construction",
          first is not None and first["t"] == -1 and first["ids"].tolist() == list(range(N)) and first["step"] == 0
          and first["counter"] == 0,
          f"first call t={first and first['t']} k={first and len(first['ids'])} step={first and first['step']}"
          f" counter={first and first['counter']}")
    bad_order = [cl["t"] for cl in calls if len(cl["ids"]) > 1 and not bool((cl["ids"][1:] > cl["ids"][:-1]).all())]
    bad_range = [cl["t"] for cl in calls if len(cl["ids"]) and (int(cl["ids"].min()) < 0 or int(cl["ids"].max()) >= N)]
    c.add("calls.ids_ascending_unique", not bad_order and not bad_range,
          f"{len(calls)} calls; non-ascending/duplicate at {bad_order[:5]}, out of range at {bad_range[:5]}")
    bad_step = [(cl["t"], cl["step"], cl["counter"]) for cl in calls if cl["step"] != cl["counter"]]
    c.add("calls.step_argument_is_counter", not bad_step, f"mismatches (t, step, counter): {bad_step[:5]}")
    wrong_sets = []
    for t in range(T):
        want = set(np.nonzero(done[t])[0].tolist())
        got = set(i for cl in by_t.get(t, []) for i in cl["ids"].tolist())
        if want != got:
            wrong_sets.append((t, len(want), len(got)))
    c.add("calls.reset_ids_equal_dones", not wrong_sets,
          f"steps where the redrawn set != the done set (t, n_done, n_redrawn): {wrong_sets[:6]}")

    # ---- the replica, advanced call by call in the order the calls happened
    gen = torch.Generator(device="cpu")
    gen.manual_seed(replica_seed(seed))
    rep = torch.ones(N, dtype=torch.float32)
    mism_scale, mism_in_force, out_of_bounds, mism_logs, mism_ret, bad_keys = [], [], [], [], [], []
    construction = None
    stats_after = {}                                  # t -> (lo_eff, replica scale after the LAST call of step t)
    s_before = np.asarray(rec["s_before"], dtype=np.float32)

    def apply_call(cl):
        lo_eff, hi_ = replica_bounds(cl["step"], lo, hi, ramp)
        ids = cl["ids"].to(torch.long)
        s = replica_draw(gen, len(ids), lo_eff, hi_)
        rep[ids] = s
        if not torch.equal(cl["scale_after"], rep):    # the device scale after the call == the replica, all N envs
            diff = (cl["scale_after"] != rep).nonzero().reshape(-1).tolist()
            mism_scale.append((cl["t"], len(diff), diff[:4]))
        stats_after[cl["t"]] = (lo_eff, rep.clone())
        return lo_eff, hi_, s

    for cl in by_t.get(-1, []):
        lo_eff, hi_, s = apply_call(cl)
        if construction is None:
            construction = (s.clone(), lo_eff, hi_, sha16(s))
    for t in range(T):
        if not np.array_equal(s_before[t], rep.numpy()):    # the dose the step actually ran under
            mism_in_force.append(t)
        for cl in by_t.get(t, []):
            lo_eff, hi_, s = apply_call(cl)
            lo32 = float(np.float32(lo_eff))
            sv = s.numpy().astype(np.float64)
            if len(sv) and (sv.min() < lo32 - 1e-6 or sv.max() > hi_ + 1e-6):
                out_of_bounds.append((t, float(sv.min()), float(sv.max()), lo32, hi_))
    c.add("replica.draws_bit_for_bit", not mism_scale and bool(calls),
          f"{len(calls)} calls: scale vector after each call == the replica generator's draws, in ascending-id order;"
          f" mismatches (t, n_envs, ids): {mism_scale[:4]}")
    c.add("replica.scale_in_force_each_step", not mism_in_force,
          f"recorded scale before step t == replica for all {T} steps; mismatches at t={mism_in_force[:6]}")
    c.add("replica.draws_in_bounds", not out_of_bounds,
          f"every redraw in [lo_eff(step), hi]; violations (t, min, max, lo_eff, hi): {out_of_bounds[:3]}")

    # ---- construction draw (the DEVICE values, against the spec's band at step 0)
    if construction is not None and by_t.get(-1):
        s0, _, _, rep_sha = construction
        lo_eff0, hi0 = replica_bounds(0, lo, hi, ramp)
        sv = by_t[-1][0]["scale_after"].numpy().astype(np.float64)
        if hi0 > lo_eff0:
            mid = 0.5 * (lo_eff0 + hi0)
            sd_mean = (hi0 - lo_eff0) / math.sqrt(12.0 * len(sv))
            c.add("construction.mean", abs(sv.mean() - mid) <= 4 * sd_mean,
                  f"mean {sv.mean():.5f} vs {mid:.5f} (4 sd of the mean = {4 * sd_mean:.5f}, z = {(sv.mean() - mid) / sd_mean:+.2f})")
            d, crit = ks_uniform01((sv - lo_eff0) / (hi0 - lo_eff0))
            c.add("construction.ks", d <= crit, f"KS D={d:.4f} vs crit {crit:.4f} (alpha 0.001, n={len(sv)})")
            c.add("construction.bounds", sv.min() >= float(np.float32(lo_eff0)) - 1e-6 and sv.max() <= hi0 + 1e-6,
                  f"min {sv.min():.5f} max {sv.max():.5f} in [{lo_eff0}, {hi0}]")
        else:
            c.add("construction.exactly_one", bool((sv == 1.0).all()) and lo_eff0 == 1.0 and hi0 == 1.0,
                  f"all {len(sv)} construction draws == 1.0: {bool((sv == 1.0).all())} (band at step 0: [{lo_eff0}, {hi0}])")
        c.add("construction.equals_replica", torch.equal(by_t[-1][0]["scale_after"], s0),
              "device scale after env.reset() == the replica generator's first draw (bit for bit)")
    else:
        c.add("construction.equals_replica", False, "no construction-time reset call recorded")

    # ---- redraws
    redraw_u = []
    for t, cls in by_t.items():
        if t == -1:
            continue
        for cl in cls:
            lo_eff, hi_ = replica_bounds(cl["step"], lo, hi, ramp)
            if hi_ > lo_eff:
                sv = cl["scale_after"][cl["ids"]].numpy().astype(np.float64)
                redraw_u.extend(((sv - lo_eff) / (hi_ - lo_eff)).tolist())
    if len(redraw_u) >= 64:
        d, crit = ks_uniform01(np.asarray(redraw_u))
        c.add("redraw.ks", d <= crit, f"pooled normalised redraws: KS D={d:.4f} vs crit {crit:.4f} (n={len(redraw_u)})")
    else:
        c.add("redraw.ks", False, f"only {len(redraw_u)} redraws with a non-degenerate band")

    if group == "D2":
        want_lo = {}
        for ts, _, _ in FORCED:
            cl = [x for x in by_t.get(ts, [])]
            step = cl[-1]["step"] if cl else None
            want_lo[ts] = None if step is None else 1.0 + (spec["lo"] - 1.0) * min(step, ramp) / ramp
        ranges = []
        for ts, lo_id, hi_id in FORCED:
            cl = by_t.get(ts, [])
            if not cl:
                ranges.append((ts, "no call"))
                continue
            fid = torch.arange(lo_id, hi_id)
            sv = cl[-1]["scale_after"][fid].numpy().astype(np.float64)
            lo_t = want_lo[ts]
            ranges.append((ts, f"step {cl[-1]['step']} lo_eff {lo_t:.5f} draws [{sv.min():.4f}, {sv.max():.4f}]",
                           bool(sv.min() >= float(np.float32(lo_t)) - 1e-6 and sv.max() <= spec["hi"] + 1e-6)))
        c.add("d2.forced_draws_in_ramped_band",
              all(len(r) == 3 and r[2] for r in ranges) and len(ranges) == len(FORCED), f"{ranges}")
        got_lo = {ts: (rec["logs"].get(ts) or {}).get(LOG_KEYS[0]) for ts, _, _ in FORCED}
        c.add("d2.logged_lo_equals_lo_eff",
              all(got_lo[ts] is not None and want_lo[ts] is not None
                  and got_lo[ts] == float(np.float32(want_lo[ts])) for ts, _, _ in FORCED),
              f"logged DR/rod_speed_lo {got_lo} vs lo_eff(t) {want_lo}")
        c.add("d2.ramp_spans_the_forced_resets",
              all(want_lo[ts] is not None for ts, _, _ in FORCED) and want_lo[100] > want_lo[200] > want_lo[400] == lo,
              f"lo_eff at the forced resets {want_lo} (want strictly falling to {lo})")

    # ---- the logged DR/ values
    for t, cls in sorted(by_t.items()):
        log = rec["logs"].get(t)
        last = cls[-1]
        if log is None:
            bad_keys.append((t, "no log read"))
            continue
        keys = sorted(k for k in log if k.startswith("DR/"))
        if keys != sorted(LOG_KEYS):
            bad_keys.append((t, keys))
            continue
        lo_eff, rep_after = stats_after[t]
        col = rep_after.numpy().astype(np.float64)
        want = {LOG_KEYS[0]: float(np.float32(lo_eff)), LOG_KEYS[1]: float(col.mean()),
                LOG_KEYS[2]: float(col.min()), LOG_KEYS[3]: float(col.max())}
        errs = {k: (log[k], want[k]) for k in LOG_KEYS
                if abs(log[k] - want[k]) > (2e-6 if k == LOG_KEYS[1] else 0.0)}
        if errs:
            mism_logs.append((t, errs))
        if any(log[k] != last["ret"][k] for k in LOG_KEYS):
            mism_ret.append(t)
    c.add("logs.keys_exactly_the_four", not bad_keys and bool(by_t), f"problems (t, keys): {bad_keys[:4]}")
    c.add("logs.values_equal_recomputed", not mism_logs,
          f"lo / min / max exact, mean within 2e-6, at all {len(by_t)} reset steps; mismatches: {mism_logs[:3]}")
    c.add("logs.extras_equal_returned", not mism_ret,
          f"extras['log'] DR/ values == the DR object's returned values at every reset step; mismatches at t={mism_ret[:5]}")

    # ---- the [DR] / [KNOB] lines
    c.add("lines.one_knob_line", len(knob_lines) == 1, knob_lines)
    if len(dr_lines) == 1 and construction is not None:
        m = DR_LINE.match(dr_lines[0])
        if not m:
            c.add("lines.one_dr_line", False, f"unparsable: {dr_lines[0][:200]}")
        else:
            n_, step_, b_lo, b_hi, l_lo, l_hi, l_ramp, l_seed, _stats, l_sha, l_mod = m.groups()
            lo_eff0, hi0 = construction[1], construction[2]
            ok = (int(n_) == N and int(step_) == 0 and float(b_lo) == float(f"{lo_eff0:g}") and float(b_hi) == float(f"{hi0:g}")
                  and float(l_lo) == lo and float(l_hi) == hi and int(l_ramp) == ramp
                  and int(l_seed) == replica_seed(seed) and l_sha == construction[3]
                  and str(Path(l_mod).resolve()).startswith(str(R)))
            c.add("lines.one_dr_line", ok, f"{dr_lines[0][:240]} | replica sha16 {construction[3]}")
    else:
        c.add("lines.one_dr_line", False, f"{len(dr_lines)} [DR] lines (want exactly 1): {dr_lines[:2]}")

    # ---- the dose: per (env, episode segment)
    seg = np.cumsum(done, axis=0) - done.astype(np.int64)          # segment index of step t; a done step ends its segment
    n_seg = n_eq = n_short = 0
    nonconst, over, short_of, first_seg_checked = [], [], [], 0
    worst_hip = worst_knee = 0.0
    for e in range(N):
        sid = seg[:, e]
        for k in np.unique(sid):
            m = sid == k
            sv = s_before[m, e]
            n_seg += 1
            if sv.min() != sv.max():
                nonconst.append((e, int(k)))
                continue
            sc = float(sv[0])
            for name, arr, rated in (("hip", hip, HIP_RATED), ("knee", knee, KNEE_RATED)):
                peak = float(arr[m, e].max())
                ceil = sc * rated
                if peak > ceil * (1.0 + REL):
                    over.append((e, int(k), name, peak, ceil))
                if m.sum() >= MIN_EQ_LEN:
                    err = abs(peak - ceil) / ceil
                    if name == "hip":
                        worst_hip = max(worst_hip, err)
                    else:
                        worst_knee = max(worst_knee, err)
                    if err > REL:
                        short_of.append((e, int(k), name, peak, ceil))
            if m.sum() >= MIN_EQ_LEN:
                n_eq += 1
                if k == 0:
                    first_seg_checked += 1
            else:
                n_short += 1
    changed = s_before[1:] != s_before[:-1]
    bad_change = np.argwhere(changed & ~done[:-1])
    c.add("dose.s_constant_within_segment", not nonconst, f"{n_seg} segments; non-constant: {nonconst[:4]}")
    c.add("dose.s_changes_only_where_done", len(bad_change) == 0,
          f"{int(changed.sum())} changes, all at done steps; offenders (t, env): {bad_change[:4].tolist()}")
    c.add("dose.never_exceeds_ceiling", not over, f"peak <= s*rated*(1+{REL:g}) in all {n_seg} segments; over: {over[:3]}")
    c.add("dose.peak_equals_s_times_rated", not short_of and n_eq >= N + 150,
          f"{n_eq} segments of >= {MIN_EQ_LEN} steps checked ({n_short} shorter skipped; first segments checked "
          f"{first_seg_checked}/{N}); worst rel err hip {worst_hip:.2e} knee {worst_knee:.2e} (tol {REL:g}); "
          f"misses (env, seg, rod, peak, ceiling): {short_of[:3]}")
    return c


# ===================================================================================== recording
class Tee:
    """sys.stdout wrapper: keeps the ``[DR]`` / ``[KNOB] KRABBY_DR_...`` lines, passes everything through."""

    def __init__(self, stream):
        self._stream, self._buf, self.lines = stream, "", []

    def write(self, text):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.startswith("[DR]") or (line.startswith("[KNOB]") and DR_VAR in line):
                self.lines.append(line.rstrip())
        return self._stream.write(text)

    def flush(self):
        return self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class DRTap:
    """Wraps ``rd.reset`` (the instance attribute shadows the method; the action term looks it up at call time)."""

    def __init__(self, rd, counter_fn, num_envs):
        self.rd, self.counter_fn, self.n, self.calls, self.cur_t = rd, counter_fn, num_envs, [], -1
        self._orig = rd.reset
        rd.reset = self._tapped

    def _tapped(self, env_ids=None, step=0):
        ids = (torch.arange(self.n) if env_ids is None
               else torch.as_tensor(env_ids).detach().cpu().to(torch.long).reshape(-1).clone())
        call = {"t": self.cur_t, "ids": ids, "step": int(step), "counter": int(self.counter_fn())}
        out = self._orig(env_ids, step)
        call["scale_after"] = self.rd.scale[:, 0].detach().cpu().clone()
        call["ret"] = {k: float(v) for k, v in out.items()}
        self.calls.append(call)
        return out


def read_log(extras) -> dict:
    log = extras.get("log") if isinstance(extras, dict) else None
    return {} if not isinstance(log, dict) else {k: float(v) for k, v in log.items() if str(k).startswith("DR/")}


def pre_state(term, u, group, env_cfg, module_file_fn) -> dict:
    rd = getattr(term, "_rod_speed_dr", "missing")
    scale = getattr(term, "_rod_speed_scale", "missing")
    cfg_attr = getattr(u.cfg.actions.joint_pos, "rod_speed_dr", "ABSENT")
    pre = {"rd_present": rd is not None and not isinstance(rd, str),
           "scale_present": scale is not None and not isinstance(scale, str),
           "cfg_attr": cfg_attr if cfg_attr == "ABSENT" else dict(cfg_attr), "cfg_seed": getattr(u.cfg, "seed", None)}
    if pre["rd_present"]:
        pre.update(alias=scale is rd.scale, scale_shape=list(rd.scale.shape), scale_dtype=str(rd.scale.dtype),
                   scale_device=str(rd.scale.device), gen_device=str(rd.gen.device), rd_lo=rd.lo, rd_hi=rd.hi,
                   rd_ramp=rd.ramp_steps, rd_seed=rd.seed, resets_before=rd.resets,
                   scale_all_one_before=bool((rd.scale == 1.0).all()), module_file=module_file_fn(rd))
    return pre


def action_pattern(step, n, n_act, term, device):
    """The sweep's pattern: pitch +s, knee -s (s flips every 50 steps, a 2 s period), cams a constant +1."""
    a = torch.zeros(n, n_act, device=device)
    s = 1.0 if (step // 50) % 2 == 0 else -1.0
    a[:, term._pitch_action_cols] = s
    a[:, term._knee_action_cols] = -s
    a[:, term._cam_action_cols] = 1.0
    return a


def pack(rows_s, rows_hip, rows_knee, rows_done, calls, logs, d0_keys) -> dict:
    return {"s_before": None if rows_s[0] is None else np.stack(rows_s), "hip": np.stack(rows_hip),
            "knee": np.stack(rows_knee), "done": np.stack(rows_done), "calls": calls, "logs": logs,
            "d0_dr_keys": d0_keys}


# ===================================================================================== the Kit run
def run_kit(args, app) -> int:
    t0 = time.time()
    sys.stdout = tee = Tee(sys.stdout)
    pt = R / "parkour" / "parkour_tasks"
    if str(pt) not in sys.path:
        sys.path.insert(0, str(pt))
    import gymnasium as gym  # noqa: PLC0415
    import parkour_tasks  # noqa: PLC0415,F401
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: PLC0415
    from parkour_tasks.crab_hex_forward_task.mdp import crab_hex_linkage as linkage  # noqa: PLC0415

    spec_env = os.environ.get(DR_VAR)
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env = gym.make(TASK, cfg=env_cfg)
    u = env.unwrapped
    term = u.action_manager.get_term("joint_pos")
    n, dev, n_act = u.num_envs, u.device, term.action_dim
    if int(u.action_manager.total_action_dim) != int(n_act):
        raise RuntimeError(f"the env has more than the joint_pos action term: {u.action_manager.total_action_dim} != {n_act}")
    pre = pre_state(term, u, args.group, env_cfg,
                    lambda rd: sys.modules[type(rd).__module__].__file__)
    tap = DRTap(term._rod_speed_dr, lambda: u.common_step_counter, n) if pre["rd_present"] else None
    res = {"tool": "rsd_smoke.py", "group": args.group, "task": TASK, "num_envs": n, "seed": args.seed,
           "steps": args.steps, "parkour_tasks_file": parkour_tasks.__file__, "max_episode_length": int(u.max_episode_length),
           "step_dt": float(u.step_dt), "total_action_dim": int(u.action_manager.total_action_dim), "pre": pre,
           "krabby_env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("KRABBY_")}}

    rows_s, rows_hip, rows_knee, rows_done, logs, d0_keys = [], [], [], [], {}, []
    if tap is not None:
        tap.cur_t = -1
    env.reset()
    logs[-1] = read_log(u.extras)
    d0_keys += list(read_log(u.extras))
    c0 = int(u.common_step_counter)
    for t in range(args.steps):
        if tap is not None:
            tap.cur_t = t
        for ts, lo_id, hi_id in FORCED:
            if t == ts:
                u.episode_length_buf[lo_id:hi_id] = u.max_episode_length - 1
        scale = term._rod_speed_scale
        rows_s.append(None if scale is None else scale[:, 0].detach().cpu().clone().numpy())
        ncalls = 0 if tap is None else len(tap.calls)
        _, _, terminated, truncated, _ = env.step(action_pattern(t, n, n_act, term, dev))
        rows_hip.append(term.hip_rod_speed.detach().abs().amax(dim=1).cpu().numpy().copy())
        rows_knee.append(term.knee_rod_speed.detach().abs().amax(dim=1).cpu().numpy().copy())
        d = (terminated | truncated).detach().cpu().numpy().astype(bool)
        rows_done.append(d)
        if tap is not None and len(tap.calls) > ncalls:
            logs[t] = read_log(u.extras)
        elif tap is None and d.any():
            d0_keys += list(read_log(u.extras))
    res["initial_counter"] = c0
    res["final_counter"] = int(u.common_step_counter)
    rec = pack(rows_s, rows_hip, rows_knee, rows_done, [] if tap is None else tap.calls, logs, d0_keys)
    meta = {"num_envs": n, "seed": args.seed, "spec_env": spec_env, "pre": pre, "lines": list(tee.lines),
            "parkour_file": parkour_tasks.__file__, "hip_rated": float(linkage.HIP_SPEED_M_S),
            "knee_rated": float(linkage.KNEE_SPEED_M_S)}
    checks = analyse(args.group, rec, meta)
    res["checks"] = checks.rows
    res["all_ok"] = checks.ok
    res["dr_lines"] = list(tee.lines)
    res["n_calls"] = len(rec["calls"])
    res["calls"] = [{"t": c_["t"], "step": c_["step"], "k": int(len(c_["ids"])),
                     "min": float(c_["scale_after"][c_["ids"]].min()) if len(c_["ids"]) else None,
                     "max": float(c_["scale_after"][c_["ids"]].max()) if len(c_["ids"]) else None,
                     "ret": c_["ret"]} for c_ in rec["calls"]]
    res["done_env_steps"] = int(np.asarray(rows_done).sum())
    res["wall_s"] = time.time() - t0
    res["complete"] = True
    Path(args.out).write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    bad = checks.failed()
    for k, v in bad.items():
        print(f"RSD_SMOKE FAIL {k}: {v}", flush=True)
    print(f"RSD_SMOKE END group={args.group} all_ok={checks.ok} checks={len(checks.rows)} failed={sorted(bad)}", flush=True)
    env.close()
    return 0


# ===================================================================================== CPU selftest
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def fake_rollout(group, steps=600, n=256, seed=7, rsd=None, drk=None, knob_text=None, nat_p=0.004, nat_seed=0,
                 gain=1.0):
    """The recording loop against a fake env that mimics the parkour env's order of events (counter += 1, dones, reset
    of the done envs, extras['log'] rebuilt at each reset) around the REAL RodSpeedDR / dr_knobs modules (CPU)."""
    spec = GROUPS[group]
    text = knob_text if knob_text is not None else (spec["text"] if spec else "")
    cfg = SimpleNamespace(actions=SimpleNamespace(joint_pos=SimpleNamespace()))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        drk.apply_dr_knobs(cfg, {DR_VAR: text} if text else {})
    lines = [ln for ln in out.getvalue().splitlines() if ln.startswith("[KNOB]")]
    action_cfg = cfg.actions.joint_pos
    rd = rsd.build_rod_speed_dr(action_cfg, n, "cpu", seed)
    cfg_attr = getattr(action_cfg, "rod_speed_dr", "ABSENT")
    state = {"counter": 0, "ep": np.zeros(n, dtype=np.int64), "fresh": np.zeros(n, dtype=bool), "extras": {"log": {}}}
    max_ep = 1000
    rng = np.random.RandomState(nat_seed)
    tap = DRTap(rd, lambda: state["counter"], n) if rd is not None else None

    def do_reset(ids):
        ids_t = torch.as_tensor(ids, dtype=torch.long)
        state["extras"]["log"] = {}
        if rd is not None:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                log = rd.reset(ids_t, state["counter"])
            lines.extend(ln for ln in buf.getvalue().splitlines() if ln.startswith("[DR]"))
            state["extras"]["log"].update(log)
        state["ep"][ids] = 0
        state["fresh"][ids] = True

    pre = {"rd_present": rd is not None, "scale_present": rd is not None, "cfg_attr": cfg_attr, "cfg_seed": seed}
    if rd is not None:
        pre.update(alias=True, scale_shape=list(rd.scale.shape), scale_dtype=str(rd.scale.dtype),
                   scale_device=str(rd.scale.device), gen_device=str(rd.gen.device), rd_lo=rd.lo, rd_hi=rd.hi,
                   rd_ramp=rd.ramp_steps, rd_seed=rd.seed, resets_before=rd.resets,
                   scale_all_one_before=bool((rd.scale == 1.0).all()), module_file=rsd.__file__)
    rows_s, rows_hip, rows_knee, rows_done, logs, d0_keys = [], [], [], [], {}, []
    if tap is not None:
        tap.cur_t = -1
    do_reset(np.arange(n))
    logs[-1] = read_log(state["extras"])
    for t in range(steps):
        if tap is not None:
            tap.cur_t = t
        for ts, lo_id, hi_id in FORCED:
            if t == ts:
                state["ep"][lo_id:hi_id] = max_ep - 1
        s = np.ones(n, dtype=np.float32) if rd is None else rd.scale[:, 0].clone().numpy()
        rows_s.append(None if rd is None else s.copy())
        ncalls = 0 if tap is None else len(tap.calls)
        state["counter"] += 1
        state["ep"] += 1
        hip = (s * np.float32(HIP_RATED) * np.float32(gain)).astype(np.float32)
        knee = (s * np.float32(KNEE_RATED) * np.float32(gain)).astype(np.float32)
        hip[state["fresh"]] = 0.0           # the first step after a reset applies the zeroed action history
        knee[state["fresh"]] = 0.0
        state["fresh"][:] = False
        done = (state["ep"] >= max_ep) | ((rng.rand(n) < nat_p) & (t > 20))
        rows_hip.append(hip)
        rows_knee.append(knee)
        rows_done.append(done.copy())
        if done.any():
            do_reset(np.nonzero(done)[0])
            if tap is not None and len(tap.calls) > ncalls:
                logs[t] = read_log(state["extras"])
            elif tap is None:
                d0_keys += list(read_log(state["extras"]))
    rec = pack(rows_s, rows_hip, rows_knee, rows_done, [] if tap is None else tap.calls, logs, d0_keys)
    meta = {"num_envs": n, "seed": seed, "spec_env": text, "pre": pre, "lines": lines, "parkour_file": str(R / "x"),
            "hip_rated": HIP_RATED, "knee_rated": KNEE_RATED}
    return rec, meta


def selftest() -> int:
    mdp = R / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/mdp/rod_speed_dr.py"
    cfgdir = R / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/config/crab_hex/dr_knobs.py"
    rsd = _load("rsd_selftest_rod_speed_dr", mdp)
    drk = _load("rsd_selftest_dr_knobs", cfgdir)
    fails = []

    def expect(label, cond, extra=""):
        print(f"SELFTEST {'ok  ' if cond else 'FAIL'} {label} {extra}")
        if not cond:
            fails.append(label)

    for g in ("D0", "D1", "D2"):
        rec, meta = fake_rollout(g, rsd=rsd, drk=drk)
        ch = analyse(g, rec, meta)
        expect(f"{g} clean rollout passes every check", ch.ok, f"({len(ch.rows)} checks) failed={ch.failed()}")

    def mutate(label, group, want_fail, edit):
        rec, meta = fake_rollout(group, rsd=rsd, drk=drk)
        edit(rec, meta)
        ch = analyse(group, rec, meta)
        bad = set(ch.failed())
        expect(f"mutation: {label} -> {want_fail} fails", want_fail in bad, f"(failed: {sorted(bad)})")

    def flip_draw(rec, meta):
        cl = rec["calls"][1]
        cl["scale_after"][int(cl["ids"][0])] += 1e-7
    mutate("one draw off by 1e-7", "D1", "replica.draws_bit_for_bit", flip_draw)

    def cut_peak(rec, meta):
        rec["hip"][0:101, 5] *= 0.9
    mutate("hip peak 10% low over env 5's whole first segment", "D1", "dose.peak_equals_s_times_rated", cut_peak)

    def over_peak(rec, meta):
        rec["knee"][300, 9] *= 1.01
    mutate("knee peak 1% over its ceiling", "D1", "dose.never_exceeds_ceiling", over_peak)

    def silent_change(rec, meta):
        rec["s_before"][250:, 3] = rec["s_before"][250:, 3] * 0.99
    mutate("s changes with no done", "D1", "dose.s_changes_only_where_done", silent_change)

    def rd_missing_line(rec, meta):
        meta["lines"] = [ln for ln in meta["lines"] if not ln.startswith("[DR]")]
    mutate("no [DR] line", "D1", "lines.one_dr_line", rd_missing_line)

    def stray_line(rec, meta):
        meta["lines"].append("[DR] rod_speed n=1 step=0")
    mutate("D0 with a [DR] line", "D0", "d0.no_dr_line", stray_line)

    def low_ceiling(rec, meta):
        rec["hip"][:, 11] *= 0.8
    mutate("D0 with one env ceiling at 0.8", "D0", "d0.ceiling_hip", low_ceiling)

    def bad_log(rec, meta):
        rec["logs"][100][LOG_KEYS[2]] += 0.01
    mutate("logged min off by 0.01", "D1", "logs.values_equal_recomputed", bad_log)

    def wrong_lo(rec, meta):
        rec["logs"][100][LOG_KEYS[0]] = 0.5
    mutate("logged lo not lo_eff(t)", "D2", "d2.logged_lo_equals_lo_eff", wrong_lo)

    def unsorted(rec, meta):
        cl = rec["calls"][1]
        cl["ids"] = cl["ids"].flip(0)
    mutate("reset ids not ascending", "D1", "calls.ids_ascending_unique", unsorted)

    def missing_redraw(rec, meta):
        cl = rec["calls"][1]
        cl["ids"] = cl["ids"][:-1]
    mutate("one reset env not redrawn", "D1", "calls.reset_ids_equal_dones", missing_redraw)

    # a wrong seed / spec in the system under test (the analysis expects seed 7 and the group's spec)
    rec, meta = fake_rollout("D1", rsd=rsd, drk=drk, seed=8)
    meta["seed"] = 7
    expect("mutation: DR seeded 8, analysis expects 7 -> replica fails", "replica.draws_bit_for_bit" in analyse("D1", rec, meta).failed())
    rec, meta = fake_rollout("D2", rsd=rsd, drk=drk, knob_text="0.5:1.0:0")
    meta["spec_env"] = "0.5:1.0:0"
    ch = analyse("D2", rec, meta)
    expect("mutation: D2 run without the ramp -> cfg / construction / forced-reset checks fail",
           {"dr.cfg_spec", "construction.exactly_one", "d2.logged_lo_equals_lo_eff"} <= set(ch.failed()),
           f"(failed: {sorted(ch.failed())})")
    rec, meta = fake_rollout("D0", rsd=rsd, drk=drk, knob_text="0.5:1.4")
    meta["spec_env"] = "0.5:1.4"
    expect("mutation: knob set in the D0 process -> D0 fails", not analyse("D0", rec, meta).ok)

    # the numbers the real run will see for the construction draw (seed 7): KS / mean margins
    g = torch.Generator(device="cpu")
    g.manual_seed(replica_seed(7))
    s = replica_draw(g, 256, 0.5, 1.4).numpy().astype(np.float64)
    d, crit = ks_uniform01((s - 0.5) / 0.9)
    print(f"SELFTEST info D1 construction draw (seed 7): mean {s.mean():.5f} (want 0.95 +- {4 * 0.9 / math.sqrt(12 * 256):.4f}), "
          f"KS D={d:.4f} crit {crit:.4f}, min {s.min():.4f} max {s.max():.4f}")
    print(f"SELFTEST END failed={fails}")
    return 1 if fails else 0


# ===================================================================================== main
def build_parser(with_launcher: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rod-speed DR dose smoke (K2).")
    parser.add_argument("--group", choices=sorted(GROUPS))
    parser.add_argument("--out")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--selftest", action="store_true", help="CPU analysis self-test (no Kit)")
    if with_launcher:
        from isaaclab.app import AppLauncher  # noqa: PLC0415
        AppLauncher.add_app_launcher_args(parser)
    return parser


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    from isaaclab.app import AppLauncher  # noqa: E402,PLC0415

    cli = build_parser(True).parse_args()
    if not cli.group or not cli.out:
        raise SystemExit("--group and --out are required")
    simulation_app = AppLauncher(cli).app
    rc = 1
    try:
        rc = run_kit(cli, simulation_app)
    except Exception:  # noqa: BLE001 -- print before the Kit teardown can hang
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()
    sys.exit(rc)
