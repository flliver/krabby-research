"""Validate the reconstruction: actor(obs_recon) vs the recorded actions (same checkpoint, hist_encoding=True)."""
import sys, json, numpy as np, torch
from recon import *
OUT = "/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/side_bisect"
cases = [("LR4", f"{OUT}/lr/LR4/side22_25lr__omni/seed001/2026-09-30_01-44-42", False),
         ("C_lr", f"{OUT}/lr/C_lr/side22_25lr__omni/seed001/2026-09-30_05-11-37", False)]
for name, st, scal in cases:
    d, meta = load_stamp(st)
    net = Net(meta["checkpoint"])
    mass = None
    for flags in [(0.0, 1.0), (1.0, 0.0)]:
        agree = []; mae = []
        for lab, z, tix in episodes(d, meta, ["lat_l_022", "lat_r_022"])[:12]:
            P = prop_series(z, meta, env_flags=flags, cmd_scalar=scal)
            base_mass = meta["body_masses_kg"][0] if isinstance(meta["body_masses_kg"], list) else 10.0
            o = torch.tensor(obs_at(P, z["root_lin_vel_b"].astype(np.float64), base_mass, tix), dtype=torch.float32)
            with torch.no_grad():
                m = net.mu(o, hist=True)
            a = torch.clamp(m, -1, 1).numpy(); ar = np.clip(z["actions"][tix], -1, 1)
            agree.append(np.mean(np.sign(a) == np.sign(ar))); mae.append(np.mean(np.abs(a - ar)))
        print(name, "env flags", flags, "sign agree %.3f  MAE(clipped) %.3f" % (np.mean(agree), np.mean(mae)))
    print("body_masses_kg[0:3]", meta["body_masses_kg"][:3] if isinstance(meta["body_masses_kg"], list) else meta["body_masses_kg"])
