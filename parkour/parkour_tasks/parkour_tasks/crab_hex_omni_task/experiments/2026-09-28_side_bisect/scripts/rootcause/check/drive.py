import sys, glob, json, numpy as np, torch
sys.path.insert(0,"/tmp/claude-1000/-home-nickmagus-krabby-krabby-research/4b0248a2-13ef-464e-81a9-16e93543e283/scratchpad/rep")
from recon import *
OUT="/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/side_bisect"
R="/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/"
for nm,st,run in (("C_lr",f"{OUT}/lr/C_lr/side22_25lr__omni/seed001/*/","2026-09-30_00-00-38"),("LR4",f"{OUT}/lr/LR4/side22_25lr__omni/seed001/*/","2026-09-29_20-33-42")):
    d,meta=load_stamp(sorted(glob.glob(st))[-1]); O=[]
    for lab in ("lat_l_022","lat_r_022"):
        for _,z,tix in episodes(d,meta,[lab])[:12]:
            P=prop_series(z,meta,env_flags=(1.0,0.0),cmd_scalar=False)
            O.append(obs_at(P,z["root_lin_vel_b"].astype(np.float64),meta["body_masses_kg"][0],tix[::3]))
    S=np.concatenate(O)[:,:75]
    rmsin=np.sqrt((S**2).mean(0))
    for it in (100,600,2999):
        sd=torch.load(glob.glob(R+run+"*")[0]+f"/model_{it}.pt",map_location="cpu",weights_only=False)["model_state_dict"]
        W1=sd["actor.actor_backbone.0.weight"].numpy(); cn=np.linalg.norm(W1,axis=0)[:75]
        drive=cn*rmsin; med=np.median(drive[17:75])
        print(f"{nm} it{it}: slot9 in-rms {rmsin[9]:.3f} colnorm {cn[9]:.2f} drive {drive[9]:.3f} ({drive[9]/med:.2f}x median proprio drive {med:.3f}) | slot8 drive {drive[8]:.3f} | speed10 drive {drive[10]:.3f} | clock15/16 drive {drive[15]:.3f}/{drive[16]:.3f} | jpos drive med {np.median(drive[17:41]):.3f} jvel med {np.median(drive[41:65]):.3f} lastact med {np.median(drive[65:75]):.3f} | in-rms jpos {np.median(rmsin[17:41]):.2f} jvel {np.median(rmsin[41:65]):.2f}")
