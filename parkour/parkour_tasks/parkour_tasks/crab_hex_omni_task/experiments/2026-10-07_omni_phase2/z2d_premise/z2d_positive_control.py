"""Z2d step-3 positive control: an OFFLINE masked BC fit of the GRAFT actor (priv encoder frozen, as the run detaches
it) on the masked seven Z2d training banks (RSTg es1 x4 + RSTresg es1 x2 + Z2c's suite crouch observe bank), R
recomputed, 12k steps; saves its actor state for p2r_z2c_gate.py --ckpt-state. CPU only."""
import glob, importlib.util, os, sys
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(16)
REPO = Path("/home/nickmagus/krabby/krabby-research")
C = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-07_omni_phase2"
spec = importlib.util.spec_from_file_location("_fid", str(C / "scripts/p2r_z2_fidelity.py"))
fid = importlib.util.module_from_spec(spec); spec.loader.exec_module(fid)
rst = fid.load_restance()
W = fid.widths(fid.S1_AGENT_YAML)
PR = rst.parse_restance("lift=-0.5,rho=1,up_frac=0.5,down_frac=0.4,k_c=2,k_h=1,trim=0.1,eps_c=0.1,eps_air=0.05,tilt_max=0.1,eps_step=0.3")
GRAFT = REPO / "parkour/outputs/omni_plan/phase2/graft/S1_s18_c15/model_19994.pt"
LR = REPO / "parkour/outputs/omni_plan/phase2/lr"
PATS = [f"P2R_RSTg_s18/{t}__omni/seed001/*/p2r_stand_bank.npz" for t in ("walkstop", "suite", "field_turn", "field_hard")]
PATS += [f"P2R_RSTresg_s18/{t}__omni/seed001/*/p2r_stand_bank.npz" for t in ("walkstop", "suite")]
PATS += ["P2R_OBSZ2c_s18/suite__omni/seed001/*/p2r_stand_bank.npz"]
g = torch.Generator().manual_seed(18)
Xs, Ts = [], []
for pat in PATS:
    (p,) = glob.glob(str(LR / pat))
    d = np.load(p); n = d["obs"].shape[0]; k = min(n, 30000)
    idx = np.sort(np.random.default_rng(1).choice(n, k, replace=False))
    o = torch.as_tensor(d["obs"][idx]); t, _ = rst.restance_target(o, PR)
    Xs.append(rst.stand_mask_obs(o)); Ts.append(t); sig = d["s1_sigma"]
X = torch.cat(Xs); T = torch.cat(Ts); SIG = torch.as_tensor(sig).clamp_min(1e-9)
print("rows", X.shape, flush=True)
sd0, esd = fid.load_ckpt(GRAFT)
sd = {k: v.clone() for k, v in sd0.items()}
keys = [k for k in sd if k.startswith(("actor.scan_encoder.", "actor.actor_backbone."))]
for k in keys: sd[k].requires_grad_(True)
opt = torch.optim.Adam([sd[k] for k in keys], lr=3e-4)
for step in range(int(os.environ.get("STEPS", "12000"))):
    idx = torch.randint(0, X.shape[0], (2048,), generator=g)
    mu = fid.eval_mu(X[idx], sd, esd, W, hist_encoding=False, priv="true")
    L = (0.5 * (((mu - T[idx]) / SIG) ** 2).sum(1)).mean()
    opt.zero_grad(); L.backward(); opt.step()
    if step % 2000 == 0: print(step, float(L), flush=True)
out = {k: sd[k].detach().clone() for k in keys}
torch.save({"model_state_dict": out}, sys.argv[1]); print("saved", sys.argv[1], len(out))
