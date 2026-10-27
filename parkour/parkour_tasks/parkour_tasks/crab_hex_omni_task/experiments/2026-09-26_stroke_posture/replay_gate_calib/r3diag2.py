import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from diag import RSP, RUNS, spearman, partial_spearman
rng = np.random.default_rng(1)
for name in ("W", "X0x_s3", "X0x_s2", "E1", "E2"):
    st = RSP.load_stamp(RUNS[name]); rp = RSP.replay(st); vals = RSP.episode_values(st, rp)
    sd = rp.stances
    for c in ("side", "diag", "fwd"):
        idx = [i for i, e in enumerate(st.episodes) if e["category"] == c]
        inc = np.array([vals["stroke_per_s"][i] for i in idx])
        sw = np.array([((st.json_eps[i] or {}).get("holds", {}).get(st.episodes[i]["label"], {}).get("quality") or {}).get("contact_switch_hz", np.nan) for i in idx], float)
        # participation: number of feet with >= 3 scored stances in the window
        part = []; pay_per_stance = []; st_per_s = []
        for i in idx:
            k = (sd["env"] == i) & sd["payable"] & st.own[sd["env"], sd["step"]] & rp.gate[sd["env"], sd["step"]]
            feet = np.bincount(sd["foot"][k], minlength=6)
            part.append((feet >= 3).sum())
            pay_per_stance.append(sd["pay"][k].mean() if k.any() else np.nan)
            n = (st.own[i] & rp.gate[i]).sum()
            st_per_s.append(k.sum() / (n * st.dt) if n else np.nan)
        part = np.array(part, float); pps = np.array(pay_per_stance); sps = np.array(st_per_s)
        boots = []
        m = np.isfinite(inc) & np.isfinite(sw)
        ii = np.flatnonzero(m)
        for _ in range(2000):
            b = rng.choice(ii, ii.size)
            boots.append(spearman(inc[b], sw[b]))
        boots = np.array([x for x in boots if x is not None])
        print(f"{name:7s} {c:5s} rho={spearman(inc, sw):+.3f} CI95=[{np.quantile(boots,.025):+.3f},{np.quantile(boots,.975):+.3f}] "
              f"rho(sw,part)={spearman(sw, part):+.3f} rho(inc,part)={spearman(inc, part):+.3f} "
              f"partial(inc,sw|part)={partial_spearman(inc, sw, part):+.3f} rho(sw,pay/stance)={spearman(sw, pps):+.3f} "
              f"rho(sw,stances/s)={spearman(sw, sps):+.3f} part_mean={np.nanmean(part):.2f}")
