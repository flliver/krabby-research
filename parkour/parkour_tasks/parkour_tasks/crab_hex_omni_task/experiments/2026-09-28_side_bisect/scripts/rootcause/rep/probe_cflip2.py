"""Network probe A: how does each policy use the COMMAND DIRECTION input?  (CPU, reconstructed obs, read-only)

At real steady-state states s of one hand, compare the executed action clip(mu(s)) with clip(mu(s')), s' = s with ONLY the
travel-command sign flipped (slot 9 for side, slot 8 for fore-aft; current row and all 10 history rows). Everything else
(body velocity, lean, joint state, clock) is left as it is. Delta = (a(s) - a(s'))/2 is binned by clock phase:
  static  = phase-mean of Delta (a direction-dependent posture/bias: lean or cam bias),
  mod     = phase-varying part of Delta (a direction x phase interaction: the stroke flipping with the command).
Reference 'target': canonical C (the fast two-hand policy pi*) -- its command flip flips the canonicalizer flag,
pi*(s') = M_y C(M_y s') -- i.e. what a direction-GATED policy does.
"""
import glob, json, sys
import numpy as np, torch
from recon import *
from mirrors import build
NB = 16
OUT = "/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/side_bisect"
LOGS = "/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases"
one = lambda g: sorted(glob.glob(g))[-1]

def stats(D, ph):
    b = np.minimum((np.mod(ph, 2 * np.pi) / (2 * np.pi) * NB).astype(int), NB - 1)
    Pm = np.stack([D[b == i].mean(0) if (b == i).any() else np.full(D.shape[1], np.nan) for i in range(NB)])
    st = np.nanmean(Pm, 0); mod = Pm - st
    return float(np.sqrt(np.nanmean(st ** 2))), float(np.sqrt(np.nanmean(mod ** 2)))

def run(name, stamp, labels, slot, canon=False, scalar=False, nep=40):
    d, meta = load_stamp(stamp); net = Net(meta["checkpoint"]); lr, fa = build(meta)
    res = {}
    for lab in labels:
        Ds, PHs, flips, mus = [], [], [], []
        for _, z, tix in episodes(d, meta, [lab])[:nep]:
            P = prop_series(z, meta, env_flags=(1.0, 0.0), cmd_scalar=scalar)
            o = torch.tensor(obs_at(P, z["root_lin_vel_b"].astype(np.float64), meta["body_masses_kg"][0], tix), dtype=torch.float32)
            of = o.clone()
            for r in range(11):   # current row + 10 history rows
                base = 0 if r == 0 else 259 + 89 * (r - 1)
                of[:, base + slot] = -of[:, base + slot]
            with torch.no_grad():
                if canon:
                    flag = torch.tensor(z["canon_flag"][tix].astype(bool))[:, None]
                    def pol(x, fl):
                        xc = torch.where(fl, lr.mirror_obs(x), x); a = net.mu(xc, hist=HIST, use_est=True)
                        return torch.where(fl, lr.mirror_actions(a), a)
                    m0 = pol(o, flag); m1 = pol(of, ~flag)
                else:
                    m0 = net.mu(o, hist=HIST, use_est=True); m1 = net.mu(of, hist=HIST, use_est=True)
            a0 = m0.clamp(-1, 1).numpy(); a1 = m1.clamp(-1, 1).numpy()
            Ds.append((a0 - a1) / 2); PHs.append(z["clock_phase"][tix].astype(float))
            flips.append(np.mean(np.sign(a0[:, 6:]) != np.sign(a1[:, 6:])))
            mus.append(np.median(np.abs(m0.numpy()[:, 6:])))
        D = np.concatenate(Ds); PH = np.concatenate(PHs)
        rs, rm = stats(D[:, 6:], PH); cs, cm = stats(D[:, :6], PH)
        res[lab] = dict(rod_static=rs, rod_mod=rm, cam_static=cs, cam_mod=cm, rod_signflip=float(np.mean(flips)),
                        med_abs_mu_rod=float(np.median(mus)), n=len(Ds))
        print("%-26s %-10s | rod static %.3f mod %.3f | cam static %.3f mod %.3f | rod sign-flip frac %.3f | med|mu| rod %.2f n%d" % (
            name, lab, rs, rm, cs, cm, np.mean(flips), np.median(mus), len(Ds)), flush=True)
    return res

HIST = True
if __name__ == "__main__":
    import sys
    HIST = sys.argv[1] == "hist"
    R = {}
    print("== side: flip slot 9 (cmd vy) only")
    for h in ["C"]:
        R[f"canon {h}"] = run(f"canon {h} (pi*, target)", one(f"{OUT}/canon/{h}/wave/side22_25lr__omni/seed001/*/"),
                              ["lat_l_022", "lat_r_022"], 9, canon=True, scalar=True)
    for a in ["LR4", "C_lr", "C_lr_rsi"]:
        R[a] = run(a, one(f"{OUT}/lr/{a}/side22_25lr__omni/seed001/*/"), ["lat_l_022", "lat_r_022"], 9)
    for a, tag in [("O", "seed3_X01")]:
        R[a] = run(a + " (+y only, 4img)", one(f"{LOGS}/2026-09-28_side_bisect/{tag}_s1_A15pB/side22_25__omni/seed001/*/"), ["lat_l_022"], 9)
    SU = {"t3": f"{LOGS}/2026-09-23_clock_deprecation/step0_E3_X08-t3_visible/suite__omni/seed001/2026-09-23_21-38-11",
          "c6": f"{LOGS}/2026-09-23_clock_deprecation/seed3_X01_c6_A15pB/suite__omni/seed001/2026-09-24_00-55-05"}
    for k, st in SU.items():
        R[f"{k} side"] = run(f"{k} full omni side", st, ["lat_l", "lat_r"], 9)
        print("== fore-aft: flip slot 8 (cmd vx) only")
        R[f"{k} fa"] = run(f"{k} full omni fore-aft", st, ["fwd", "back"], 8)
    json.dump(R, open("probe_cflip_train_path.json", "w"), indent=1)
