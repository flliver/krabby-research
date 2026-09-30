import glob, numpy as np, torch
from recon import *
from mirrors import build
OUT = "/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/side_bisect"
LOGS = "/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-28_side_bisect"
one = lambda g: sorted(glob.glob(g))[-1]
HIST = True
for name, st, scal, K in [("O", one(f"{LOGS}/seed3_X01_s1_A15pB/side22_25__omni/seed001/*/"), False, 4),
                          ("C_mo_s2", one(f"{LOGS}/seed2_X08_s1_A15pB/side22_25__omni/seed001/*/"), True, 2),
                          ("LR4", one(f"{OUT}/lr/LR4/side22_25lr__omni/seed001/*/"), False, 4)]:
    d, meta = load_stamp(st); net = Net(meta["checkpoint"]); lr, fa = build(meta)
    O = []
    for _, z, tix in episodes(d, meta, ["lat_l_022"])[:16]:
        P = prop_series(z, meta, env_flags=(1.0, 0.0), cmd_scalar=scal)
        O.append(obs_at(P, z["root_lin_vel_b"].astype(np.float64), meta["body_masses_kg"][0], tix[::3]))
    S = torch.tensor(np.concatenate(O), dtype=torch.float32)
    with torch.no_grad():
        est = net.estimator(S[:, :75])
    print(name, "estimator output mean (com3, friction, stiff5):", np.round(est.mean(0).numpy(), 3))
    for fill in ["estimator"]:
        X = S.clone()
        if fill != "zeros(com,fric)": X[:, 207:216] = est
        if fill == "est+fric1.3": X[:, 210] = 1.3
        with torch.no_grad():
            m = net.mu(X, hist=HIST, use_est=False)
            r = {}
            for iname, Mo, Ma in [("M_y", lr.mirror_obs, lr.mirror_actions), ("M_x", fa.mirror_obs, fa.mirror_actions),
                                  ("M_xM_y", lambda x: fa.mirror_obs(lr.mirror_obs(x)), lambda a: fa.mirror_actions(lr.mirror_actions(a)))]:
                r[iname] = float(((net.mu(Mo(X), hist=HIST, use_est=False) - Ma(m)) ** 2).mean())
        tb = np.mean([r["M_y"], r["M_x"], r["M_xM_y"]]) if K == 4 else r["M_x"]
        print(f"  {fill:16s} med|mu| {np.median(np.abs(m.numpy())):.2f} | res My {r['M_y']:.3f} Mx {r['M_x']:.3f} MxMy {r['M_xM_y']:.3f} -> TB-equivalent {tb:.3f}")
