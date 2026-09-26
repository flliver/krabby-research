"""Restated R1/R3b/R6/R8/P1 (REPLAY_GATE.md 'restated' section), computed from a set of replays.

R1r  kept >= 0.80 of the anchor-capped ('planted') credit: sum(pay of scored stances) / sum(after_anchor of every
     closing stance in the window), omni fwd/back/turn and the sideways head's side (both stamps).
R2r  = the tool's R2 clauses b (slide stances keep <= 0.25) and c (omni side <= 0.2 x sideways head); a reported.
R3r  a (short stances <= 2 %) + b judged on its bootstrap CI: FAIL only if the 95 % CI lower bound of
     Spearman(income/s, contact_switch_hz) is above +0.2 (per category and head); + v4 era clause (fwd_era.py).
R6r  income shift <= 5 % (signed shift reported); liftoff shift reported.
R8r  FAIL only if |gap| > 10 % AND the gap's 95 % episode-bootstrap CI excludes 0.
P1r  the flat sideways stamp (slow__side) >= 0.85 of max; step__side reported.
"""
import numpy as np


def rank(x):
    from scipy.stats import rankdata
    return rankdata(x)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 8:
        return None
    rx, ry = rank(x[m]), rank(y[m])
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _sel(RSP, st, rp, cats):
    sd = rp.stances
    catv = np.array([e["category"] for e in st.episodes])
    if not sd["env"].size:
        return np.zeros(0, bool), sd
    k = st.own[sd["env"], sd["step"]] & rp.gate[sd["env"], sd["step"]] & np.isin(catv[sd["env"]], list(cats))
    return k, sd


def r1r(RSP, runs):
    """runs: {name: (role, st, rp)}"""
    rows = {}
    for n, (role, st, rp) in runs.items():
        cats = ("fwd", "back", "turn") if role == "healthy" else (("side",) if role == "sideways" else ())
        for c in cats:
            k, sd = _sel(RSP, st, rp, (c,))
            den = sd["after_anchor"][k].sum()
            U = sd["ungated"][k].sum()
            pk = k & sd["payable"]
            rows[f"{n}.{c}"] = {"kept_planted": float(sd["pay"][pk].sum() / den) if den > 0 else None,
                                "kept_literal": float(sd["pay"][pk].sum() / U) if U > 0 else None,
                                "anchor_removed": float(1 - den / U) if U > 0 else None,
                                "back_slide_over_B": float(np.maximum(sd["B"] - sd["Ba"], 0)[pk].sum()
                                                           / np.maximum(sd["B"][pk], 0).sum()) if pk.any() else None}
    ok = all(v["kept_planted"] is not None and v["kept_planted"] >= 0.80 for v in rows.values())
    return {"pass": ok, "rows": rows}


def r3r(RSP, runs, B=1000, seed=7):
    rng = np.random.default_rng(seed)
    rows = {}
    ok = True
    for n, (role, st, rp) in runs.items():
        if role != "healthy":
            continue
        vals = RSP.episode_values(st, rp)
        sd = rp.stances
        for c in RSP.MOVING:
            idx = [i for i, e in enumerate(st.episodes) if e["category"] == c]
            inc = np.array([vals["stroke_per_s"][i] for i in idx], float)
            sw = np.array([(((st.json_eps[i] or {}).get("holds") or {}).get(st.episodes[i]["label"], {})
                            .get("quality") or {}).get("contact_switch_hz", np.nan) for i in idx], float)
            pps = []
            for i in idx:
                kk = (sd["env"] == i) & sd["payable"] & st.own[sd["env"], sd["step"]] & rp.gate[sd["env"], sd["step"]]
                pps.append(sd["pay"][kk].mean() if kk.any() else np.nan)
            rho = spearman(inc, sw)
            m = np.flatnonzero(np.isfinite(inc) & np.isfinite(sw))
            bs = [spearman(inc[b], sw[b]) for b in (rng.choice(m, m.size) for _ in range(B))]
            bs = np.array([x for x in bs if x is not None])
            lo, hi = (float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))) if bs.size else (None, None)
            rows[f"{n}.{c}"] = {"rho": rho, "ci95": [lo, hi], "rho_pay_per_stance": spearman(pps, sw)}
            if lo is not None and lo > 0.2:
                ok = False
    return {"pass": ok, "rows": rows}


def r8r(RSP, runs, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    rows = {}
    ok = True
    for n, (role, st, rp) in runs.items():
        if role != "healthy":
            continue
        gated = st.own & rp.gate
        pay = rp.pay.sum(-1)
        for c in RSP.PAIRED:
            per = {}
            for h in ("l", "r"):
                m = np.flatnonzero([e["category"] == c and e["hand"] == h for e in st.episodes])
                per[h] = (np.array([pay[i][gated[i]].sum() for i in m]), np.array([gated[i].sum() for i in m]))
            (nl, dl), (nr, dr) = per["l"], per["r"]
            a, b = nl.sum() / dl.sum(), nr.sum() / dr.sum()
            gap = (a - b) / (0.5 * (a + b))
            gs = []
            for _ in range(B):
                il = rng.integers(0, nl.size, nl.size)
                ir = rng.integers(0, nr.size, nr.size)
                x, y = nl[il].sum() / dl[il].sum(), nr[ir].sum() / dr[ir].sum()
                gs.append((x - y) / (0.5 * (x + y)))
            lo, hi = float(np.quantile(gs, 0.025)), float(np.quantile(gs, 0.975))
            fail = abs(gap) > 0.10 and (lo > 0 or hi < 0)
            ok &= not fail
            rows[f"{n}.{c}"] = {"gap": float(gap), "ci95": [lo, hi], "fail": bool(fail)}
    return {"pass": bool(ok), "rows": rows}


def r6r(rep):
    rows = rep["criteria"]["R6"]["rows"]
    ok = True
    out = {}
    for k, v in rows.items():
        if not v.get("gated", True):
            continue
        a, b = v["income"], v["income_dup"]
        shift = (b / a - 1) if a else None
        ok &= shift is None or abs(shift) <= 0.05
        out[k] = {"income_shift_signed": shift, "liftoff_shift": v.get("liftoff_shift")}
    return {"pass": bool(ok), "rows": out}


def r2r(rep):
    cl = rep["criteria"]["R2"]["clauses"]
    return {"pass": bool(cl["slide_kept<=0.25"] and cl["side<=0.2xsideways"] is not False),
            "side_kept_reported": rep["criteria"]["R2"]["side_kept"]}


def p1r(rep, flat="side_slow"):
    p = rep["criteria"]["P1"]
    sw = p["sideways_side"].get(flat)
    cl = p["clauses"]
    band = [v for k, v in cl.items() if "in [0.15,0.45]" in k][0]
    return {"pass": bool(sw is not None and sw >= 0.85 and band and cl["fwd/back/stand/turn==0"]),
            "flat": sw, "step_reported": p["sideways_side"].get("side_step")}
