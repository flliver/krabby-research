"""Network probes B (saturation timeline) and C (per-image mirror residuals / gradient share). CPU, read-only.

States: each arm's own final side-eval states (steady window), reconstructed. Checkpoints model_<it>.pt of the run.
B: rollout path (hist_encoding=False, estimator-replaced priv slice): median |mu| and the share of rod dims with
   |mu| > 1 + 2*std (Gaussian exploration then leaves the CLIPPED action unchanged >97.7% of the time).
C: the PPO mirror term exactly as ppo_with_extractor.py:441-451 computes it (act_inference on the raw obs,
   hist_encoding=False, no estimator; target = M(mu(s)) detached): residual per image and the gradient norm of
   (coef * MSE) per image, coef = 0.5 / (K-1) per image (MSE is a mean over the (K-1)*B rows).
"""
import glob, json, sys
import numpy as np, torch
from recon import *
from mirrors import build
OUT = "/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/side_bisect"
LOGS = "/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-28_side_bisect"
RUN = "/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/"
one = lambda g: sorted(glob.glob(g))[-1]
ARMS = {  # name: (stamp, labels, scalar, run prefix, images)
    "C": (one(f"{LOGS}/seed3_X02_s1_A15pB/side22_25__omni/seed001/*/"), ["lat_l_022"], True, "2026-09-29_01-52-56", "fa"),
    "C_mo_s2": (one(f"{LOGS}/seed2_X08_s1_A15pB/side22_25__omni/seed001/*/"), ["lat_l_022"], True, "2026-09-29_14-36-09", "fa"),
    "C_mo": (one(f"{LOGS}/seed3_X04_s1_A15pB/side22_25__omni/seed001/*/"), ["lat_l_022"], True, "2026-09-29_09-39-22", "fa"),
    "O": (one(f"{LOGS}/seed3_X01_s1_A15pB/side22_25__omni/seed001/*/"), ["lat_l_022"], False, "2026-09-29_00-40-18", "4"),
    "C_band": (one(f"{LOGS}/seed3_X05_s1_A15pB/side22_25__omni/seed001/*/"), ["lat_l_022"], False, "2026-09-29_10-52-23", "4"),
    "LRfa": (one(f"{OUT}/lr/LRfa/side22_25lr__omni/seed001/*/"), ["lat_l_022", "lat_r_022"], False, "2026-09-29_19-18-51", "fa"),
    "LR4": (one(f"{OUT}/lr/LR4/side22_25lr__omni/seed001/*/"), ["lat_l_022", "lat_r_022"], False, "2026-09-29_20-33-42", "4"),
    "C_lr": (one(f"{OUT}/lr/C_lr/side22_25lr__omni/seed001/*/"), ["lat_l_022", "lat_r_022"], False, "2026-09-30_00-00-38", "fa"),
}
ITS = [100, 200, 300, 400, 600, 900, 1500, 2999]

def states(stamp, labels, scalar, nep=16, stride=3):
    d, meta = load_stamp(stamp); O = []
    for lab in labels:
        for _, z, tix in episodes(d, meta, [lab])[:nep]:
            P = prop_series(z, meta, env_flags=(1.0, 0.0), cmd_scalar=scalar)
            O.append(obs_at(P, z["root_lin_vel_b"].astype(np.float64), meta["body_masses_kg"][0], tix[::stride]))
    return meta, torch.tensor(np.concatenate(O), dtype=torch.float32)

def mu_train(net, o):  # act_inference on raw obs (mirror-loss path)
    return net.mu(o, hist=False, use_est=False)

if __name__ != "__main__":
    raise SystemExit
res = {}
for name, (stamp, labels, scalar, pre, imgs) in ARMS.items():
    meta, S = states(stamp, labels, scalar)
    lr, fa = build(meta)
    run = glob.glob(RUN + pre + "*")[0]
    res[name] = {}
    for it in ITS:
        net = Net(f"{run}/model_{it}.pt"); std = float(net.std.clamp(0.05, 2.0).mean())
        with torch.no_grad():
            m = net.mu(S, hist=False, use_est=True).numpy()
        rod = np.abs(m[:, 6:]); cam = np.abs(m[:, :6])
        row = dict(std=std, med_mu_rod=float(np.median(rod)), med_mu_cam=float(np.median(cam)),
                   frac_rod_locked=float(np.mean(rod > 1 + 2 * std)), frac_rod_sat=float(np.mean(rod > 1)))
        # C: mirror residuals and gradient norms per image (train path)
        for p in net.sd.values(): p.requires_grad_(False)
        params = [v for k, v in net.sd.items() if k.startswith("actor.")]
        for p in params: p.requires_grad_(True)
        base = mu_train(net, S).detach()
        K = 4 if imgs == "4" else 2
        coef = 0.5 / (K - 1)
        for iname, Mo, Ma in [("M_y", lambda x: lr.mirror_obs(x), lambda a: lr.mirror_actions(a)),
                              ("M_x", lambda x: fa.mirror_obs(x), lambda a: fa.mirror_actions(a)),
                              ("M_xM_y", lambda x: fa.mirror_obs(lr.mirror_obs(x)), lambda a: fa.mirror_actions(lr.mirror_actions(a)))]:
            out = mu_train(net, Mo(S)); tgt = Ma(base)
            r = ((out - tgt) ** 2).mean()
            g = torch.autograd.grad(coef * r, params, allow_unused=True)
            gn = float(torch.sqrt(sum((gi ** 2).sum() for gi in g if gi is not None)))
            row[f"res_{iname}"] = float(r.detach()); row[f"grad_{iname}"] = gn
            row[f"res_rod_{iname}"] = float(((out - tgt)[:, 6:] ** 2).mean().detach())
            # executed-action disagreement (clipped) -- what the mirror mismatch means behaviourally
            row[f"clipdis_{iname}"] = float((out.clamp(-1, 1) - tgt.clamp(-1, 1)).abs().mean().detach())
        for p in params: p.requires_grad_(False)
        res[name][it] = row
        print("%-8s it%5d std %.2f | med|mu| rod %.2f cam %.2f | rod locked(>1+2s) %.2f sat %.2f | res My %.3f Mx %.3f MxMy %.3f | "
              "clipdis My %.3f Mx %.3f | grad(coef*res) My %.4f Mx %.4f MxMy %.4f" % (
                  name, it, std, row["med_mu_rod"], row["med_mu_cam"], row["frac_rod_locked"], row["frac_rod_sat"],
                  row["res_M_y"], row["res_M_x"], row["res_M_xM_y"], row["clipdis_M_y"], row["clipdis_M_x"],
                  row["grad_M_y"], row["grad_M_x"], row["grad_M_xM_y"]), flush=True)
json.dump(res, open("probe_sat_mirror.json", "w"), indent=1)
