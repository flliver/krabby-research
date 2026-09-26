"""Diagnostics for the step-7 failures (R3 Spearman confound, R6 liftoff clause, R8 hands vs noise).

usage: diag.py <which> [k=v ...]   which in {r3, r6, r8}
"""
import json
import sys
from pathlib import Path
import importlib.util

import numpy as np

TOOL = Path("/home/nickmagus/krabby/krabby-research/.claude/worktrees/stroke-posture/parkour/parkour_tasks/"
            "parkour_tasks/crab_hex_omni_task/experiments/tools/replay_stroke_posture.py")
spec = importlib.util.spec_from_file_location("rsp_diag", TOOL)
RSP = importlib.util.module_from_spec(spec)
sys.modules["rsp_diag"] = RSP
spec.loader.exec_module(RSP)

B = "/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval"
RUNS = {
    "W": f"{B}/omni_phases/2026-09-23_clock_deprecation_w2/seed3_X03_c6x_A15pB/suite__omni/seed001/2026-09-25_20-31-29",
    "X0x_s3": f"{B}/omni_phases/2026-09-23_clock_deprecation_w2/seed3_X01_c6x_A15pB/suite__omni/seed001/2026-09-25_17-01-55",
    "X0x_s2": f"{B}/omni_phases/2026-09-23_clock_deprecation_w2/seed2_X02_c6x_A15pB/suite__omni/seed001/2026-09-25_18-45-27",
    "E1": f"{B}/omni_phases/2026-09-23_clock_deprecation/step0_E1_X08-c6_visible/suite__omni/seed001/2026-09-23_21-23-41",
    "E2": f"{B}/omni_phases/2026-09-23_clock_deprecation/step0_E2_X08-c4_visible/suite__omni/seed001/2026-09-23_21-31-11",
    "side_slow": f"{B}/side_phases/2026-09-11_0414_sideways_formation/seed2_2a_006_A15pB/slow__side/seed001/2026-09-12_09-18-39",
    "side_step": f"{B}/side_phases/2026-09-11_0414_sideways_formation/seed2_2a_006_A15pB/step__side/seed001/2026-09-12_09-20-17",
}


def rank(x):
    from scipy.stats import rankdata
    return rankdata(x)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 8:
        return None
    return float(np.corrcoef(rank(x[m]), rank(y[m]))[0, 1])


def partial_spearman(x, y, z):
    """Spearman of x and y controlling for z (rank residuals)."""
    x, y, z = (np.asarray(a, float) for a in (x, y, z))
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 8:
        return None
    rx, ry, rz = rank(x[m]), rank(y[m]), rank(z[m])
    A = np.stack([rz, np.ones_like(rz)], 1)
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def r3(params):
    out = {}
    for name in ("W", "X0x_s3", "X0x_s2", "E1"):
        st = RSP.load_stamp(RUNS[name])
        rp = RSP.replay(st, params)
        vals = RSP.episode_values(st, rp)
        sd = rp.stances
        gated = st.own & rp.gate
        for c in RSP.MOVING:
            idx = [i for i, e in enumerate(st.episodes) if e["category"] == c]
            inc, sw, trk, spd, nsc, medB, lifts = [], [], [], [], [], [], []
            for i in idx:
                j = st.json_eps[i] or {}
                h = (j.get("holds") or {}).get(st.episodes[i]["label"]) or {}
                q = h.get("quality") or {}
                t = h.get("tracking") or {}
                inc.append(vals["stroke_per_s"][i])
                sw.append(q.get("contact_switch_hz") if q.get("contact_switch_hz") is not None else np.nan)
                key = "wz" if c == "turn" else "vx"
                trk.append((t.get(key) or {}).get("ratio", np.nan) if t else np.nan)
                n = gated[i].sum()
                k = (sd["env"] == i) & sd["payable"] & st.own[sd["env"], sd["step"]] & rp.gate[sd["env"], sd["step"]]
                nsc.append(k.sum() / (n * st.dt) if n else np.nan)
                medB.append(np.median(sd["B"][k]) if k.any() else np.nan)
            row = {"n": len(idx),
                   "rho(income,switch)": spearman(inc, sw),
                   "rho(income,tracking)": spearman(inc, trk),
                   "rho(switch,tracking)": spearman(sw, trk),
                   "partial rho(income,switch | tracking)": partial_spearman(inc, sw, trk),
                   "rho(switch, median stance B)": spearman(sw, medB),
                   "rho(income, scored stances/s)": spearman(inc, nsc),
                   "rho(income, median stance B)": spearman(inc, medB)}
            out[f"{name}.{c}"] = row
            print(name, c, {k: (None if v is None else round(v, 3)) for k, v in row.items()}, flush=True)
    return out


def r6(params):
    out = {}
    for name in ("W", "E1", "side_slow", "side_step"):
        st = RSP.load_stamp(RUNS[name])
        rp = RSP.replay(st, params)
        T = st.arr["segment_id"].shape[1]
        rpd = RSP.replay(st, params, dup=RSP.dup_pattern(T))
        cats = ("side",) if st.kind == "side" else RSP.MOVING
        catv = np.array([e["category"] for e in st.episodes])
        for c in cats:
            def sel(r):
                sd = r.stances
                k = st.own[sd["env"], sd["step"]] & r.gate[sd["env"], sd["step"]] & (catv[sd["env"]] == c)
                return k, sd
            k0, s0 = sel(rp)
            k1, s1 = sel(rpd)
            closed0, closed1 = k0.sum(), k1.sum()
            sc0, sc1 = (k0 & s0["payable"]).sum(), (k1 & s1["payable"]).sum()
            inc0, inc1 = s0["pay"][k0].sum(), s1["pay"][k1].sum()
            short0, short1 = (k0 & s0["short_n"]).sum(), (k1 & s1["short_n"]).sum()
            row = {"closed": int(closed0), "closed_dup": int(closed1), "closed_shift": closed1 / closed0 - 1,
                   "scored": int(sc0), "scored_dup": int(sc1), "scored_shift": sc1 / sc0 - 1 if sc0 else None,
                   "short": int(short0), "short_dup": int(short1),
                   "income_shift": inc1 / inc0 - 1 if inc0 else None,
                   "median_B": float(np.median(s0["B"][k0 & s0["payable"]])) if sc0 else None,
                   "median_B_dup": float(np.median(s1["B"][k1 & s1["payable"]])) if sc1 else None}
            out[f"{name}.{c}"] = row
            print(name, c, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return out


def r8(params):
    out = {}
    for name in ("W", "X0x_s3", "X0x_s2", "E1", "E2"):
        st = RSP.load_stamp(RUNS[name])
        rp = RSP.replay(st, params)
        g = RSP.GateRun(name=name, role="healthy", st=st, rp=rp)
        for c in RSP.PAIRED:
            a, b = RSP.cat_income(g, c, hand="l"), RSP.cat_income(g, c, hand="r")
            ca, cb = RSP.cat_income(g, c, hand="l", term="centring"), RSP.cat_income(g, c, hand="r", term="centring")
            # bootstrap the hand gap over episodes (eval noise at n=100 per hand)
            rng = np.random.default_rng(0)
            m_l = np.array([e["category"] == c and e["hand"] == "l" for e in st.episodes])
            m_r = np.array([e["category"] == c and e["hand"] == "r" for e in st.episodes])
            gated = st.own & rp.gate
            pay = rp.pay.sum(-1)
            num_l = np.array([pay[i][gated[i]].sum() for i in np.flatnonzero(m_l)])
            den_l = np.array([gated[i].sum() for i in np.flatnonzero(m_l)])
            num_r = np.array([pay[i][gated[i]].sum() for i in np.flatnonzero(m_r)])
            den_r = np.array([gated[i].sum() for i in np.flatnonzero(m_r)])
            gaps = []
            for _ in range(2000):
                il = rng.integers(0, len(num_l), len(num_l))
                ir = rng.integers(0, len(num_r), len(num_r))
                x = num_l[il].sum() / den_l[il].sum()
                y = num_r[ir].sum() / den_r[ir].sum()
                gaps.append((x - y) / (0.5 * (x + y)))
            gaps = np.array(gaps)
            row = {"l": a, "r": b, "gap": RSP._hand_gap(a, b), "signed_gap": (a - b) / (0.5 * (a + b)),
                   "gap_ci95": [float(np.quantile(gaps, 0.025)), float(np.quantile(gaps, 0.975))],
                   "centring_gap": RSP._hand_gap(ca, cb) if ca is not None else None}
            out[f"{name}.{c}"] = row
            print(name, c, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return out


if __name__ == "__main__":
    which = sys.argv[1]
    params = RSP.parse_params(sys.argv[2:])
    res = {"r3": r3, "r6": r6, "r8": r8}[which](params)
    (Path.cwd() / f"diag_{which}.json").write_text(
        json.dumps(RSP._jsonable({"params": RSP.dataclasses.asdict(params), "rows": res}), indent=1))
