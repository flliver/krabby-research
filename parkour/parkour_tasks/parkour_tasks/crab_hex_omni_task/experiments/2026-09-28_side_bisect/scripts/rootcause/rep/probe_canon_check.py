"""Sanity: on canonical-C lat_r (flag on), actor(M_y obs) mapped back must reproduce the recorded actions."""
import numpy as np, torch, glob
from recon import *
from mirrors import build
OUT = "/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/side_bisect"
st = sorted(glob.glob(f"{OUT}/canon/C/wave/side22_25lr__omni/seed001/*/"))[-1]
d, meta = load_stamp(st); net = Net(meta["checkpoint"]); lr, fa = build(meta)
for lab in ["lat_l_022", "lat_r_022"]:
    ag = []
    for _, z, tix in episodes(d, meta, [lab])[:10]:
        P = prop_series(z, meta, env_flags=(1.0, 0.0), cmd_scalar=True)
        o = torch.tensor(obs_at(P, z["root_lin_vel_b"].astype(np.float64), meta["body_masses_kg"][0], tix), dtype=torch.float32)
        flag = z["canon_flag"][tix].astype(bool)
        oc = torch.where(torch.tensor(flag)[:, None], lr.mirror_obs(o), o)
        with torch.no_grad():
            a = net.mu(oc, hist=True)
        a = torch.where(torch.tensor(flag)[:, None], lr.mirror_actions(a), a).clamp(-1, 1).numpy()
        ag.append(np.mean(np.sign(a) == np.sign(np.clip(z["actions"][tix], -1, 1))))
    print(lab, "flag frac", flag.mean(), "sign agree %.3f" % np.mean(ag))
