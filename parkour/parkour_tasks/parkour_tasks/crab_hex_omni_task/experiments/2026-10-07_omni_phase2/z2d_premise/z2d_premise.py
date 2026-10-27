#!/usr/bin/env python3
"""Z2d premise probe (CPU, read-only): under the stand-row mask (last action cols 65:83 -> 0, proxy 83:89 -> -0.5,
and history frames' 65:75 -> 0), can the GRAFT actor be fit to R -- including the stepping branches -- from state?

Fit: GRAFT actor (scan enc + backbone trainable; priv encoder frozen, as Z2c's detach; history encoder + estimator
frozen) on the masked Z2c training banks (RSTg es1 walkstop/suite/field_turn/field_hard + RSTresg es1), training
path, loss 0.5*sum(((mu-R)/sigma)^2). Score (eval path, masked): held-out RSTg walkstop es2 and suite es2 by branch;
Z2c's closed-loop observe banks (crouch states, never trained on) by branch. Also scores Z2c model_24993 masked
without fine-tuning (the mask-only counterfactual) on the same sets.
"""
import glob
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("NT", "16")))
REPO = Path("/home/nickmagus/krabby/krabby-research")
C = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-07_omni_phase2"
spec = importlib.util.spec_from_file_location("_fid", str(C / "scripts/p2r_z2_fidelity.py"))
fid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fid)
rst = fid.load_restance()
W = fid.widths(fid.S1_AGENT_YAML)
P0, H = W["num_prop"], W["num_hist"]
GRAFT = REPO / "parkour/outputs/omni_plan/phase2/graft/S1_s18_c15/model_19994.pt"
Z2C = REPO / "parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-10_19-01-45_seed18_X18_p2r_A15pB/model_24993.pt"
LR = REPO / "parkour/outputs/omni_plan/phase2/lr"
PR = rst.parse_restance("lift=-0.5,rho=1,up_frac=0.5,down_frac=0.4,k_c=2,k_h=1,trim=0.1,eps_c=0.1,eps_air=0.05,"
                        "tilt_max=0.1,eps_step=0.3")


def mask(o):
    o = o.clone()
    o[:, 65:83] = 0.0
    o[:, 83:89] = -0.5
    hs = o.shape[1] - H * P0
    for k in range(H):
        o[:, hs + k * P0 + 65:hs + k * P0 + 75] = 0.0
    return o


def bank(pattern):
    (p,) = glob.glob(str(LR / pattern))
    d = np.load(p, allow_pickle=True)
    o = mask(torch.as_tensor(d["obs"][:, :]))
    t, br = rst.restance_target(o, PR)
    return {"obs": o, "target": t, "branch": br, "sigma": torch.as_tensor(d["s1_sigma"]).clamp_min(1e-9)}


TRAIN = [bank(f"P2R_RSTg_s18/{r}__omni/seed001/*/p2r_stand_bank.npz") for r in ("walkstop", "suite", "field_turn", "field_hard")]
TRAIN += [bank(f"P2R_RSTresg_s18/{r}__omni/seed001/*/p2r_stand_bank.npz") for r in ("walkstop", "suite")]
if os.environ.get("ADD_OBS_SUITE"):
    TRAIN.append(bank("P2R_OBSZ2c_s18/suite__omni/seed001/*/p2r_stand_bank.npz"))
HELD = {"walkstop_es2": bank("P2R_RSTg_s18/walkstop__omni/seed002/*/p2r_stand_bank.npz"),
        "suite_es2": bank("P2R_RSTg_s18/suite__omni/seed002/*/p2r_stand_bank.npz"),
        "obs_suite_Z2c": bank("P2R_OBSZ2c_s18/suite__omni/seed001/*/p2r_stand_bank.npz"),
        "obs_walkstop_Z2c": bank("P2R_OBSZ2c_s18/walkstop__omni/seed001/*/p2r_stand_bank.npz")}
SIG = TRAIN[0]["sigma"]
Xtr = torch.cat([b["obs"] for b in TRAIN])
Ttr = torch.cat([b["target"] for b in TRAIN])
print(json.dumps({"n_train": int(Xtr.shape[0]), "held": {k: int(v["obs"].shape[0]) for k, v in HELD.items()},
                  "train_branch_leg_shares": {nm: round(float((torch.cat([b['branch'] for b in TRAIN]) == k).float().mean()), 4)
                                              for k, nm in enumerate(rst.BRANCH_NAMES)}}), flush=True)


def score(sd, esd, b):
    with torch.no_grad():
        mu = fid.eval_mu(b["obs"], sd, esd, W)
    d = mu - b["target"]
    br = b["branch"]
    ah = (br == rst.BRANCH_HOLD).all(1)
    out = {"allhold_n": int(ah.sum())}
    if bool(ah.any()):
        out["allhold_hip_max"] = round(float(d[ah][:, rst.HIP_ACT].mean(0).abs().max()), 4)
        out["allhold_cam_max"] = round(float(d[ah][:, rst.CAM_ACT].mean(0).abs().max()), 4)
    for k, nm in enumerate(rst.BRANCH_NAMES):
        m = br == k
        if bool(m.any()) and nm != "hold":
            out[nm] = {"n": int(m.sum()), "hip_abs": round(float(d[:, rst.HIP_ACT].abs()[m].mean()), 3),
                       "cam_abs": round(float(d[:, rst.CAM_ACT].abs()[m].mean()), 3),
                       "hip_signed": round(float(d[:, rst.HIP_ACT][m].mean()), 3)}
    return out


res = {}
OVS = float(os.environ.get("OVERSAMPLE", "1"))
BRtr = torch.cat([b["branch"] for b in TRAIN])
step_row = (BRtr != rst.BRANCH_HOLD).any(1)
WROW = torch.where(step_row, torch.full((BRtr.shape[0],), OVS), torch.ones(BRtr.shape[0]))
print("stepping-row share", round(float(step_row.float().mean()), 4), "oversample", OVS, flush=True)
for name, ck in (() if os.environ.get("SKIP_BASE") else (("GRAFT_masked", GRAFT), ("Z2c_masked_no_finetune", Z2C))):
    sd, esd = fid.load_ckpt(ck)
    res[name] = {k: score(sd, esd, b) for k, b in HELD.items()}
    print(name, json.dumps(res[name]), flush=True)

sd0, esd0 = fid.load_ckpt(GRAFT)
TRAINABLE = ("actor.scan_encoder.", "actor.actor_backbone.")
STEPS = int(os.environ.get("STEPS", "4000"))
for lr in [float(x) for x in os.environ.get("LRS", "3e-4").split(",")]:
    sd = {k: v.clone() for k, v in sd0.items()}
    keys = [k for k in sd if k.startswith(TRAINABLE)]
    for k in keys:
        sd[k].requires_grad_(True)
    opt = torch.optim.Adam([sd[k] for k in keys], lr=lr)
    g = torch.Generator().manual_seed(18)
    t0 = time.time()
    run = {}
    for step in range(STEPS + 1):
        if step in (0, 4000, 8000, STEPS):
            run[step] = {k: score(sd, esd0, b) for k, b in HELD.items()}
            print(f"[fit lr={lr:g}] step {step} t={time.time() - t0:.0f}s", json.dumps(run[step]), flush=True)
        if step == STEPS:
            break
        idx = (torch.multinomial(WROW, 2048, replacement=True, generator=g) if OVS != 1.0
               else torch.randint(0, Xtr.shape[0], (2048,), generator=g))
        mu = fid.eval_mu(Xtr[idx], sd, esd0, W, hist_encoding=False, priv="true")
        L = (0.5 * (((mu - Ttr[idx]) / SIG) ** 2).sum(1)).mean()
        opt.zero_grad()
        L.backward()
        opt.step()
    res[f"fit_lr{lr:g}"] = run
Path(os.environ.get("OUT", "/tmp/claude-1000/-home-nickmagus-krabby-krabby-research/f09fa754-dd17-413f-8553-6b22fdb41cf7/scratchpad/z2d_premise.json")).write_text(json.dumps(res, indent=1))
print("DONE")
