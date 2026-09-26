import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from diag import RSP, RUNS
for name in sys.argv[1:]:
    st = RSP.load_stamp(RUNS[name]); rp = RSP.replay(st)
    catv = np.array([e["category"] for e in st.episodes])
    seg = st.arr["segment_id"].astype(int)
    N, T = seg.shape
    # time since the current segment began, per step
    tss = np.zeros((N, T))
    for i in range(N):
        start = 0
        for t in range(T):
            if t > 0 and seg[i, t] != seg[i, t - 1]:
                start = t
            tss[i, t] = (t - start) * st.dt
    cmd = st.arr["cmd_applied"].astype(float)
    V = np.zeros_like(st.arr["root_lin_vel_b"], dtype=float); V[:, :-1] = st.arr["root_lin_vel_b"][:, 1:]
    W = np.zeros(st.arr["root_ang_vel_b"].shape[:2]); W[:, :-1] = st.arr["root_ang_vel_b"][:, 1:, 2]
    r = rp.r.astype(float)
    vi = np.stack((cmd[..., None, 0] - cmd[..., None, 2] * r[..., 1], cmd[..., None, 1] + cmd[..., None, 2] * r[..., 0]), -1)
    vpt = np.stack((V[..., None, 0] - W[..., None] * r[..., 1], V[..., None, 1] + W[..., None] * r[..., 0]), -1)
    q = np.clip((vpt * vi).sum(-1) / np.maximum((vi * vi).sum(-1), 1e-6), -1, 1.5)
    for c in ("turn", "fwd"):
        m = (catv == c)[:, None] & rp.gate & st.own
        print(f"{name} {c}: own-gated steps start at tss p5/p50 {np.quantile(tss[m], [.05, .5]).round(2)}; own window length p50 {np.median(m.sum(1)[catv==c])*st.dt:.1f}s")
        for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 20)):
            mb = m & (tss >= lo) & (tss < hi)
            mg = (catv == c)[:, None] & rp.gate & (tss >= lo) & (tss < hi)   # gated, incl. non-own
            if mg.sum() == 0: continue
            wr = (W[mg] / np.where(np.abs(cmd[..., 2][mg]) > 1e-3, cmd[..., 2][mg], np.nan))
            print(f"   t∈[{lo},{hi}) own-gated {mb.sum():6d} all-gated {mg.sum():6d} median q {np.median(q[mg]):+.3f} q<0.30 frac {np.mean(np.median(q[mg],-1)<0.3):.2f} wz ratio med {np.nanmedian(wr):+.2f}")
        sd = rp.stances
        k = st.own[sd["env"], sd["step"]] & rp.gate[sd["env"], sd["step"]] & (catv[sd["env"]] == c) & sd["payable"]
        t_close = tss[sd["env"][k], sd["step"][k]]
        lostp = (sd["after_slip"] - sd["pay"])[k]
        for lo, hi in ((0, 2), (2, 4), (4, 6), (6, 20)):
            mm = (t_close >= lo) & (t_close < hi)
            print(f"   stances closing t∈[{lo},{hi}): n {mm.sum():5d} prog-lost share of their after_slip {lostp[mm].sum()/max(sd['after_slip'][k][mm].sum(),1e-9):.3f} mean g_prog {sd['g_prog'][k][mm].mean() if mm.any() else float('nan'):.3f}")
