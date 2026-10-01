"""Symmetry decomposition of eval stamps -> sym_results.json + printed progress (CPU, read-only).
Run extract.py first (the cache). See symlib.py for the conventions (M_y maps, pi clock shift).

ppo_construction copy (2026-10-01) of the side bisect's scripts/rootcause/sym/sym.py. Changes: the per-hand CLASS
registry is DATA (--class NAME=L[,R] and/or --class-json, the side22_classify.py / along_cmd.py JSON outputs; a
name with no class reads "?"), the cache is symlib.CACHE (--cache), --names picks a subset of the registry, and
--out names the results file (default <cache>/sym_results.json). row() and cross() are verbatim.
  sym.py [--cache DIR] [--out FILE] [--class NAME=L[,R] ...] [--class-json FILE ...] [--names A,B]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import symlib  # noqa: E402
from symlib import *  # noqa: E402,F401,F403

#: class per hand (L = +y = lat_l, R = -y = lat_r), filled by main() from the CLI
CLASS: dict = {}


def load_classes(specs, json_files) -> dict:
    """``--class NAME=L[,R]`` and side22_classify / along_cmd JSONs -> {name: (L, R)}. A JSON's
    ``class``/``class_r`` (feet_ss) are combined with ``class_along``/``class_along_r`` when both files
    name the run: the hand is FAST only if both are (CHARTER 4.1); otherwise the feet_ss class wins."""
    out: dict = {}
    feet: dict = {}
    along: dict = {}
    for f in json_files or []:
        for name, r in json.loads(Path(f).read_text()).items():
            if "class" in r or "class_r" in r:
                feet[name] = (r.get("class"), r.get("class_r"))
            if "class_along" in r or "class_along_r" in r:
                along[name] = (r.get("class_along"), r.get("class_along_r"))
    for name in set(feet) | set(along):
        f = feet.get(name, (None, None)); a = along.get(name, (None, None))
        hands = []
        for h in range(2):
            fc, ac = f[h], a[h]
            if fc is None:
                hands.append(ac)
            elif fc == "FAST" and ac is not None and ac != "FAST":
                hands.append(ac)
            else:
                hands.append(fc)
        out[name] = tuple(hands)
    for spec in specs or []:
        name, v = spec.split("=", 1)
        parts = [x.strip() or None for x in v.split(",")]
        out[name] = (parts[0], parts[1] if len(parts) > 1 else None)
    return out


REPS = ["act", "act_rod", "act_cam", "jp_rod", "jp_yaw", "foot", "foot_y", "foot_z"]


def static_block(M, rep, h1, h2):
    """M (E,C) per-episode means. Returns mean vector, its odd part norm, even part norm, split-half noise of odd part."""
    m = M.mean(0); om = (m - mir_static(m, rep)) / 2
    o1 = (M[h1].mean(0) - mir_static(M[h1].mean(0), rep)) / 2
    o2 = (M[h2].mean(0) - mir_static(M[h2].mean(0), rep)) / 2
    return m, om, dict(odd_norm=float(np.linalg.norm(om)), even_norm=float(np.linalg.norm((m + mir_static(m, rep)) / 2)),
                       odd_noise=float(np.linalg.norm(o1 - o2) / 2))


def propulsion(d, sgn):
    """Lift-stroke coupling per leg, phase-free: J_i = <(z_i - mean z_i) * dy_i/dt> (m^2/s), sign-normalised so that
    J>0 = pushing the body toward the commanded side; and its correlation form."""
    y = d["foot"][:, :, 6:12].astype(float); z = d["foot"][:, :, 12:18].astype(float)
    vy = np.gradient(y, DT, axis=1)
    zc = z - z.mean(1, keepdims=True)
    J = sgn * (zc * vy).mean(1)                              # (E,6)
    rho = sgn * (zc * (vy - vy.mean(1, keepdims=True))).mean(1) / (zc.std(1) * vy.std(1) + 1e-9)
    return J, rho


def row(name, lab, d):
    hand = "L" if "_l_" in lab else "R"
    sgn = 1.0 if hand == "L" else -1.0
    E = len(d["act"]); h1, h2 = split_idx(E)
    cl = CLASS.get(name, ("?", "?"))[0 if hand == "L" else 1] or "?"
    fs = np.asarray(d["feet_ss"])
    out = dict(stamp=name, label=lab, hand=hand, cls=cl, n=E, feet_ss=float(np.median(fs)),
               sweep=float(np.nanmedian(d["sweep"])), lean_abs=float(np.median(d["lean_abs"])),
               stall_frac=float(np.mean(fs < 0.03)))
    th, f0, _ = intrinsic_phase(d)
    has_clock = bool(np.isfinite(d["ph"]).all())
    out["f0_intrinsic_hz"] = f0
    # choose phase: clock if the gait is locked to it (rod joints), else intrinsic
    r2i = pattern(sub(d, "jp_rod"), th)[1]
    r2c = pattern(sub(d, "jp_rod"), d["ph"])[1] if has_clock else 0.0
    if has_clock:
        cf = np.median(np.diff(np.unwrap(d["ph"], axis=1), axis=1).mean(1)) / DT / (2 * np.pi)
        out["clock_hz"] = float(cf)
    use_clock = has_clock and r2c >= 0.8 * r2i
    ph = d["ph"] if use_clock else th
    out["phase"] = "clock" if use_clock else "intrinsic"
    out["jp_rod_lockR2_clock"] = r2c; out["jp_rod_lockR2_intr"] = r2i
    pats = {}
    for rep in REPS:
        X = sub(d, rep) if rep in SUB else d[rep].astype(float)
        o, A = selfmirror(X, ph, rep)
        pats[rep] = A
        out[rep] = o
        # the other phase choice, for robustness
        if has_clock:
            o2, _ = selfmirror(X, th if use_clock else d["ph"], rep)
            out[rep]["odd_pi_otherphase"] = o2["odd_pi"]; out[rep]["lockR2_otherphase"] = o2["lockR2"]
    out["act"]["energy_share"] = {b: pair_odd(pats["act"], "act")[b + "_share_of_A"] for b in ("cam", "hip", "knee")}
    out["act_rod"]["harm"] = harmonic_split(pats["act_rod"], "act_rod")
    out["foot_yz"] = {"harm": harmonic_split(np.concatenate([pats["foot_y"], pats["foot_z"]], 1), "foot_yz")}
    # per-pair odd fraction of the rod-action pattern (front/mid/rear) and per-leg energy (leading vs trailing)
    A = pats["act_rod"]; O = (A - mir(A, "act_rod", NB // 2)) / 2
    out["act_rod"]["pair_odd"] = {pr: float(np.sum(O[:, [j, j + 1, 6 + j, 7 + j]] ** 2) / np.sum(A[:, [j, j + 1, 6 + j, 7 + j]] ** 2))
                                  for pr, j in (("front", 0), ("mid", 2), ("rear", 4))}
    lead = LEFT if hand == "L" else RIGHT; trail = RIGHT if hand == "L" else LEFT
    for rep in ("act_rod", "foot_y", "foot_z"):
        Ap = pats[rep]; nb_ = Ap.shape[1] // 6
        e = np.array([np.sum(Ap[:, [6 * b + j for b in range(nb_)]] ** 2) for j in range(6)]); e = e / e.sum()
        out[rep]["leg_share"] = e.tolist(); out[rep]["lead_share"] = float(e[lead].sum())
    # foot y stroke amplitude per leg (pattern peak-to-peak), leading vs trailing
    Ay = pats["foot_y"]; p2p = Ay.max(0) - Ay.min(0)
    out["foot_y"]["p2p_lead"] = float(p2p[lead].mean()); out["foot_y"]["p2p_trail"] = float(p2p[trail].mean())
    # static parts
    st = {}
    for rep in ("act", "act_rod", "act_cam", "jp_rod", "jp_yaw", "foot_y", "camv", "body"):
        X = (sub(d, rep) if rep in SUB else d[rep].astype(float))
        M = X.mean(1)
        m, om, s = static_block(M, rep, h1, h2)
        s["mean"] = m.tolist(); s["odd"] = om.tolist()
        st[rep] = s
    out["static"] = st
    # static vs pattern share of the M_y-odd energy (rod actions; and full action)
    for rep in ("act", "act_rod", "jp_rod", "foot_y"):
        Ao = out[rep]["amp"] * np.sqrt(out[rep]["odd_pi"])   # rms odd pattern
        so = st[rep]["odd_norm"]
        out[rep]["odd_pattern_rms"] = float(Ao)
        out[rep]["static_odd"] = so
        out[rep]["static_share_of_odd"] = float(so ** 2 / (so ** 2 + Ao ** 2 + 1e-12))
    J, rho = propulsion(d, sgn)
    out["prop"] = dict(J_total=float(np.median(J.sum(1))), J_lead=float(np.median(J[:, lead].sum(1))),
                       J_trail=float(np.median(J[:, trail].sum(1))), rho_mean=float(np.median(rho.mean(1))))
    return out, pats, ph


def cross(name, spd, dl, dr, pl, pr):
    """Two hands of one stamp at one speed, clock phase (all two-hand stamps are clock-locked)."""
    res = {}
    for rep in ("act", "act_rod", "jp_rod", "foot_y"):
        XL = sub(dl, rep) if rep in SUB else dl[rep].astype(float)
        XR = sub(dr, rep) if rep in SUB else dr[rep].astype(float)
        AL, _, _ = pattern(XL, dl["ph"]); AR, _, _ = pattern(XR, dr["ph"])
        l1, l2 = split_idx(len(XL)); r1, r2 = split_idx(len(XR))
        AL1 = pattern(XL, dl["ph"], l1)[0]; AL2 = pattern(XL, dl["ph"], l2)[0]
        AR1 = pattern(XR, dr["ph"], r1)[0]; AR2 = pattern(XR, dr["ph"], r2)[0]
        n = np.sqrt(rms(AL) * rms(AR))
        noise = np.sqrt((rms(AL1 - AL2) ** 2 + rms(AR1 - AR2) ** 2) / 4) / n
        mL = XL.mean(1).mean(0); mR = XR.mean(1).mean(0)
        sL1 = XL[l1].mean(1).mean(0) - XL[l2].mean(1).mean(0); sR1 = XR[r1].mean(1).mean(0) - XR[r2].mean(1).mean(0)
        snoise = float(np.sqrt((np.linalg.norm(sL1) ** 2 + np.linalg.norm(sR1) ** 2) / 4))
        pat_agn = rms(AR - AL); pat_equi = rms(AR - mir(AL, rep, NB // 2))
        st_agn = float(np.linalg.norm(mR - mL)); st_equi = float(np.linalg.norm(mR - mir_static(mL, rep)))
        res[rep] = dict(pat_agn=pat_agn / n, pat_equi=pat_equi / n, pat_noise=float(noise),
                        pat_equi0=rms(AR - mir(AL, rep, 0)) / n,
                        st_agn=st_agn, st_equi=st_equi, st_noise=snoise,
                        static_share_of_LR_diff=float(st_agn ** 2 / (st_agn ** 2 + pat_agn ** 2 + 1e-12)),
                        amp_L=rms(AL), amp_R=rms(AR))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="M_y symmetry decomposition of the cached stamps.")
    ap.add_argument("--cache", default=None, help=f"extract cache (default {symlib.CACHE})")
    ap.add_argument("--out", default=None, help="results JSON (default <cache>/sym_results.json)")
    ap.add_argument("--class", dest="classes", action="append", default=[], metavar="NAME=L[,R]")
    ap.add_argument("--class-json", action="append", default=[], metavar="FILE")
    ap.add_argument("--names", default=None, help="comma list: only these registry names")
    a = ap.parse_args(argv)
    if a.cache:
        symlib.set_cache(a.cache)
    CLASS.update(load_classes(a.classes, a.class_json))
    reg = json.load(open(symlib.CACHE / "registry.json"))
    want = set(a.names.split(",")) if a.names else None
    rows, crosses = [], []
    for name, group, path in reg["stamps"]:
        if want is not None and name not in want:
            continue
        loaded = {}
        for lab in ("lat_l_022", "lat_l_025", "lat_r_022", "lat_r_025"):
            d = load(name, lab)
            if d is None:
                continue
            o, pats, ph = row(name, lab, d)
            o["group"] = group
            rows.append(o); loaded[lab] = d
            print(name, lab, "done", flush=True)
        for spd in ("022", "025"):
            if f"lat_l_{spd}" in loaded and f"lat_r_{spd}" in loaded and np.isfinite(loaded[f"lat_l_{spd}"]["ph"]).all():
                c = cross(name, spd, loaded[f"lat_l_{spd}"], loaded[f"lat_r_{spd}"], None, None)
                c.update(stamp=name, speed=spd, group=group)
                crosses.append(c)
    out = Path(a.out) if a.out else symlib.CACHE / "sym_results.json"
    json.dump(dict(rows=rows, cross=crosses), open(out, "w"), indent=1)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
