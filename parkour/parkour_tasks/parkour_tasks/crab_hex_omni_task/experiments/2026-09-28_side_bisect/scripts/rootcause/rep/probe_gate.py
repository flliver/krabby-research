"""Can any first-layer unit act as a command gate?  Per unit u of actor_backbone.0 (512 units), on the arm's own
steady side states (both hands where available): c-drive = |W1[u,9]| * |c| (|c| = 0.22) and the unit's
pre-activation spread (std over states, both hands pooled). A unit can switch its regime on the command only if
2*c-drive (the swing from -c to +c) is comparable to its spread. Also: the same for body lateral velocity
(slot 14, obs = 2*vy) and for the whole input 'hand' difference (mean pre-activation difference between hands)."""
import glob, numpy as np, torch
from recon import *
OUT = "/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/side_bisect"
RUN = "/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/"
one = lambda g: sorted(glob.glob(g))[-1]
for name, pre, stamp in [("LR4", "2026-09-29_20-33-42", one(f"{OUT}/lr/LR4/side22_25lr__omni/seed001/*/")),
                         ("C_lr", "2026-09-30_00-00-38", one(f"{OUT}/lr/C_lr/side22_25lr__omni/seed001/*/")),
                         ("LRfa", "2026-09-29_19-18-51", one(f"{OUT}/lr/LRfa/side22_25lr__omni/seed001/*/")),
                         ("C_lr_rsi", "2026-09-30_09-05-43", one(f"{OUT}/lr/C_lr_rsi/side22_25lr__omni/seed001/*/"))]:
    d, meta = load_stamp(stamp); run = glob.glob(RUN + pre + "*")[0]
    O, Hh = [], []
    for k, lab in enumerate(["lat_l_022", "lat_r_022"]):
        for _, z, tix in episodes(d, meta, [lab])[:20]:
            P = prop_series(z, meta, env_flags=(1.0, 0.0))
            O.append(obs_at(P, z["root_lin_vel_b"].astype(np.float64), meta["body_masses_kg"][0], tix[::3])); Hh.append(np.full(len(tix[::3]), k))
    S = torch.tensor(np.concatenate(O), dtype=torch.float32); Hh = np.concatenate(Hh)
    for it in [300, 900, 2999]:
        net = Net(f"{run}/model_{it}.pt")
        with torch.no_grad():
            x = net.backbone_input(S, hist=True)
            W1 = net.sd["actor.actor_backbone.0.weight"]; b1 = net.sd["actor.actor_backbone.0.bias"]
            z1 = (x @ W1.T + b1).numpy()
        sd = z1.std(0) + 1e-9
        cdrive = 2 * np.abs(W1[:, 9].numpy()) * 0.22
        vdrive = 2 * np.abs(W1[:, 14].numpy()) * np.abs(x[:, 14].numpy()).mean()
        hdiff = np.abs(z1[Hh == 0].mean(0) - z1[Hh == 1].mean(0))
        # within-hand spread (what the phase / state variation contributes)
        wsd = 0.5 * (z1[Hh == 0].std(0) + z1[Hh == 1].std(0)) + 1e-9
        print("%-8s it%5d | c-swing/within-hand spread: median %.2f, frac>1 %.3f | vy-swing/spread median %.2f | "
              "hand mean-diff/within spread: median %.2f frac>1 %.3f | units in ELU-negative regime for one hand only (>80%% vs <20%%): %.3f" % (
                  name, it, np.median(cdrive / wsd), np.mean(cdrive / wsd > 1), np.median(vdrive / wsd),
                  np.median(hdiff / wsd), np.mean(hdiff / wsd > 1),
                  np.mean(((z1[Hh == 0] < 0).mean(0) > 0.8) & ((z1[Hh == 1] < 0).mean(0) < 0.2) | ((z1[Hh == 1] < 0).mean(0) > 0.8) & ((z1[Hh == 0] < 0).mean(0) < 0.2))), flush=True)
