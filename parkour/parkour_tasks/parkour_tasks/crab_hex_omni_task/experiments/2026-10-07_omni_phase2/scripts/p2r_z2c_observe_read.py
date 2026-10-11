#!/usr/bin/env python3
"""Z2c diagnosis (2026-10-10): read the closed-loop observe banks of P2__Z2c_s18 model_24993.

For each observe bank (suite es1, walkstop es1): R's branch shares on the closed-loop stand states (vs the probe's
own banks), the applied action mu - R by branch and dim group, mu - R on all-HOLD rows, the CPU eval-path mu on the
banked obs vs the recorded applied mu (the scorer reproduces the harness), and the distance of the closed-loop obs
from the training banks on the columns R and the actor read (rod positions, proxy, last action, body height proxy).
"""
import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(16)
REPO = Path("/home/nickmagus/krabby/krabby-research")
C = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-07_omni_phase2"
spec = importlib.util.spec_from_file_location("_fid", str(C / "scripts/p2r_z2_fidelity.py"))
fid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fid)
rst = fid.load_restance()
W = fid.widths(fid.S1_AGENT_YAML)
CK = REPO / "parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-10_19-01-45_seed18_X18_p2r_A15pB/model_24993.pt"
LR = REPO / "parkour/outputs/omni_plan/phase2/lr"
PARAMS = rst.parse_restance("lift=-0.5,rho=1,up_frac=0.5,down_frac=0.4,k_c=2,k_h=1,trim=0.1,eps_c=0.1,eps_air=0.05,"
                            "tilt_max=0.1,eps_step=0.3")
GROUPS = {"cam": rst.CAM_ACT, "hip": rst.HIP_ACT, "knee": rst.KNEE_ACT}
COLS = {"hip_pos(29:35)": slice(29, 35), "cam(18:30:2)": slice(18, 30, 2), "proxy(83:89)": slice(83, 89),
        "last_act_rods(71:83)": slice(71, 83), "roll_pitch(3:5)": slice(3, 5), "joint_vel(41:65)": slice(41, 65)}


def load(p):
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files}


def summarize(name, b, sd, esd, ref_banks):
    obs = torch.as_tensor(b["obs"])
    tgt, br = rst.restance_target(obs, PARAMS)
    br = br.numpy()
    mu_app = torch.as_tensor(b["mu"])
    with torch.no_grad():
        mu_cpu = fid.eval_mu(obs, sd, esd, W)
    out = {"rows": int(obs.shape[0]),
           "cpu_vs_applied_mu_max_abs": float((mu_cpu - mu_app).abs().max()),
           "cpu_vs_applied_mu_mean_abs": float((mu_cpu - mu_app).abs().mean())}
    shares = {nm: float((br == k).mean()) for k, nm in enumerate(rst.BRANCH_NAMES)}
    out["branch_leg_shares"] = shares
    ah = (br == 0).all(1)
    out["allhold_share"] = float(ah.mean())
    d = (mu_app - tgt).numpy()
    out["allhold_mean_mu_minus_R"] = {g: d[ah][:, s].mean(0).round(4).tolist() for g, s in GROUPS.items()} if ah.any() else None
    byb = {}
    for k, nm in enumerate(rst.BRANCH_NAMES):
        m = br == k
        if not m.any():
            continue
        byb[nm] = {"leg_rows": int(m.sum())}
        for g, s in GROUPS.items():
            byb[nm][g + "_abs"] = round(float(np.abs(d[:, s])[m].mean()), 4)
            byb[nm][g + "_signed"] = round(float(d[:, s][m].mean()), 4)
    out["by_branch"] = byb
    # how far the closed-loop states are from the training banks, per column group (mean / p90 of |x|, both sides)
    dist = {}
    for cn, sl in COLS.items():
        a = b["obs"][:, sl]
        r = np.concatenate([rb["obs"][:, sl] for rb in ref_banks])
        dist[cn] = {"observe_mean_abs": round(float(np.abs(a).mean()), 4), "observe_p90_abs": round(float(np.quantile(np.abs(a), 0.9)), 4),
                    "train_mean_abs": round(float(np.abs(r).mean()), 4), "train_p90_abs": round(float(np.quantile(np.abs(r), 0.9)), 4)}
    out["column_scale_vs_train_banks"] = dist
    out["proxy_parked_share"] = {"observe": float((np.abs(b["obs"][:, 83:89] + 0.5) < 1e-4).mean()),
                                 "train": float(np.mean([(np.abs(rb["obs"][:, 83:89] + 0.5) < 1e-4).mean() for rb in ref_banks]))}
    # per step since episode start (suite statue from spawn): when does the hip drift begin
    t = b["t"]
    prof = {}
    for lo, hi in ((0, 3), (3, 6), (6, 10), (10, 20), (20, 50), (50, 100), (100, 300), (300, 10 ** 6)):
        m = (t >= lo) & (t < hi)
        if m.sum():
            prof[f"{lo}-{hi}"] = {"n": int(m.sum()), "mu_hip_mean": round(float(b["mu"][m][:, 6:12].mean()), 4),
                                  "R_hold_leg_share": round(float((br[m] == 0).mean()), 3),
                                  "R_lift_or_seat_share": round(float(((br[m] == 2) | (br[m] == 4)).mean()), 3)}
    out["by_step"] = prof
    return out


def main():
    sd, esd = fid.load_ckpt(CK)
    train = [load(p) for p in sorted(glob.glob(str(LR / "P2R_RST*g_s18/*__omni/seed001/*/p2r_stand_bank.npz")))]
    res = {}
    for row in ("suite", "walkstop"):
        ps = sorted(glob.glob(str(LR / f"P2R_OBSZ2c_s18/{row}__omni/seed001/*/p2r_stand_bank.npz")))
        if not ps:
            print(row, "no observe bank yet")
            continue
        b = load(ps[-1])
        assert bool(b.get("observe", False)), ps[-1]
        res[row] = summarize(row, b, sd, esd, train)
        res[row]["bank"] = ps[-1]
    print(json.dumps(res, indent=1))
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
