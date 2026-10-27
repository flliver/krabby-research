"""H6 probe (robust version): EXECUTED-action (clipped) mirror disagreement per image, two input fillings.
fill A: hist_encoding=False, raw priv slice with com/friction = 0 (training mirror-loss path, crude priv values)
fill B: hist_encoding=True, priv slice = estimator output (deploy-path proxy for the true values)
|clip(mu(M s)) - clip(M mu(s))| averaged over rows and the 18 channels (range 0..2)."""
import glob, json, numpy as np, torch
from recon import *
from mirrors import build
OUT = "/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/side_bisect"
LOGS = "/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-28_side_bisect"
RUN = "/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/"
one = lambda g: sorted(glob.glob(g))[-1]
ARMS = {
 "C (F, fa)": (one(f"{LOGS}/seed3_X02_s1_A15pB/side22_25__omni/seed001/*/"), True, "2026-09-29_01-52-56", 2),
 "C_s2 (F, fa)": (one(f"{LOGS}/seed2_X07_s1_A15pB/side22_25__omni/seed001/*/"), True, "2026-09-29_13-23-08", 2),
 "C_mo_s2 (F, fa)": (one(f"{LOGS}/seed2_X08_s1_A15pB/side22_25__omni/seed001/*/"), True, "2026-09-29_14-36-09", 2),
 "C_mo_s1 (F, fa)": (one(f"{LOGS}/seed1_X09_s1_A15pB/side22_25__omni/seed001/*/"), True, "2026-09-29_18-09-45", 2),
 "C_mo (I, fa)": (one(f"{LOGS}/seed3_X04_s1_A15pB/side22_25__omni/seed001/*/"), True, "2026-09-29_09-39-22", 2),
 "O (S, 4img +y)": (one(f"{LOGS}/seed3_X01_s1_A15pB/side22_25__omni/seed001/*/"), False, "2026-09-29_00-40-18", 4),
 "R (S, 4img +y)": (one(f"{LOGS}/seed3_X03_s1_A15pB/side22_25__omni/seed001/*/"), False, "2026-09-29_08-28-30", 4),
 "C_band (S, 4img +y)": (one(f"{LOGS}/seed3_X05_s1_A15pB/side22_25__omni/seed001/*/"), False, "2026-09-29_10-52-23", 4),
 "C_rsi (S, 4img +y)": (one(f"{LOGS}/seed3_X06_s1_A15pB/side22_25__omni/seed001/*/"), False, "2026-09-29_12-09-37", 4),
 "LR4 (S, 4img ±y)": (one(f"{OUT}/lr/LR4/side22_25lr__omni/seed001/*/"), False, "2026-09-29_20-33-42", 4),
 "LRfa (S, fa ±y)": (one(f"{OUT}/lr/LRfa/side22_25lr__omni/seed001/*/"), False, "2026-09-29_19-18-51", 2),
 "C_lr (S, fa ±y)": (one(f"{OUT}/lr/C_lr/side22_25lr__omni/seed001/*/"), False, "2026-09-30_00-00-38", 2),
}
IMG = lambda lr, fa: [("M_y", lr.mirror_obs, lr.mirror_actions), ("M_x", fa.mirror_obs, fa.mirror_actions),
                      ("MxMy", lambda x: fa.mirror_obs(lr.mirror_obs(x)), lambda a: fa.mirror_actions(lr.mirror_actions(a)))]
res = {}
for name, (st, scal, pre, K) in ARMS.items():
    d, meta = load_stamp(st); lr, fa = build(meta); O = []
    for _, z, tix in episodes(d, meta, ["lat_l_022"])[:16]:
        P = prop_series(z, meta, env_flags=(1.0, 0.0), cmd_scalar=scal)
        O.append(obs_at(P, z["root_lin_vel_b"].astype(np.float64), meta["body_masses_kg"][0], tix[::3]))
    S = torch.tensor(np.concatenate(O), dtype=torch.float32)
    run = glob.glob(RUN + pre + "*")[0]; res[name] = {}
    line = []
    for it in [300, 600, 1500, 2999]:
        net = Net(f"{run}/model_{it}.pt")
        with torch.no_grad():
            est = net.estimator(S[:, :75]); SB = S.clone(); SB[:, 207:216] = est
            out = {}
            for fill, X, hist in (("A", S, False), ("B", SB, True)):
                m = net.mu(X, hist=hist, use_est=False)
                for iname, Mo, Ma in IMG(lr, fa):
                    out[f"{fill}_{iname}"] = float((net.mu(Mo(X), hist=hist, use_est=False).clamp(-1, 1) - Ma(m).clamp(-1, 1)).abs().mean())
        res[name][it] = out
        line.append(f"it{it}: Mx {out['A_M_x']:.3f}/{out['B_M_x']:.3f} My {out['A_M_y']:.3f}/{out['B_M_y']:.3f}")
    print(f"{name:20s} " + " | ".join(line), flush=True)
json.dump(res, open("probe_h6.json", "w"), indent=1)
