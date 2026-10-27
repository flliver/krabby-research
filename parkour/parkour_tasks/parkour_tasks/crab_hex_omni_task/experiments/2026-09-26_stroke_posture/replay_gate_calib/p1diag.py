import sys, numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from diag import RSP, RUNS
SM = RSP.SM
for name in ("side_slow", "side_step", "W", "E1"):
    st = RSP.load_stamp(RUNS[name])
    rp = RSP.replay(st)
    catv = np.array([e["category"] for e in st.episodes])
    g = st.own & rp.gate & (catv == "side")[:, None]
    o = rp.o[g]; b = rp.b[g]
    # recompute phi via a fresh centring pass to get g_phi: b = lam*gphi*exp(-(o/s)^2); lam=1 on side
    gauss = np.exp(-(o / 0.2) ** 2)
    gphi = np.where(gauss > 1e-6, b / np.maximum(gauss, 1e-9), np.nan)
    lat = st.lat_share[g]
    print(f"{name:10s} n={g.sum()} b_mean={b.mean():.3f} o_med={np.median(o):+.3f} o_p10={np.quantile(o,.1):+.3f} o_p90={np.quantile(o,.9):+.3f} "
          f"gauss_mean={gauss.mean():.3f} gphi*lam_mean={np.nanmean(gphi):.3f} frac_gphi<0.99={np.nanmean(gphi<0.99):.3f} lat_mean={lat.mean():.3f}")
    # per-episode tracking ratio for sideways
    if st.kind == "side":
        segs = st.meta["hold_labels"]; print("   holds", segs)
        seg = st.arr["segment_id"]
        for sid, lab in enumerate(segs):
            m = g & (seg == sid)
            if m.any():
                print(f"   hold {lab}: n={m.sum()} b={rp.b[m].mean():.3f} o_med={np.median(rp.o[m]):+.3f} gauss={np.exp(-(rp.o[m]/0.2)**2).mean():.3f} cmd={np.linalg.norm(st.arr['cmd_applied'][m][:, :2], axis=-1).mean():.3f}")
