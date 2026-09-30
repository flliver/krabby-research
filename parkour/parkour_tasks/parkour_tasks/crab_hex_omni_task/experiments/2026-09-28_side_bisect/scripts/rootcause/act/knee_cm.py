"""The lean carrier in raw joint coordinates: common-mode (all-six) knee bias whose sign = travel direction."""
import numpy as np
from lib import *
print("arm            hand        feet   lean   knee_cm(raw mean of 6)  knee per-leg clipped mean            hip_cm  cam spin(|mean|>0.5 legs)")
for key, labs in [("W",["lat_l","lat_r"]),("lr:C_lr_rsi",["lat_l_022","lat_r_022"]),("lr:C_lr",["lat_l_022","lat_r_022"]),("lr:LRfa",["lat_l_022","lat_r_022"]),
                  ("lr:LR4",["lat_l_022","lat_r_022"]),("lr:C_lr_10",["lat_l_022","lat_r_022"]),("O",["lat_l_022"]),("C_rsi",["lat_l_022"]),("R",["lat_l_022"]),("C_band",["lat_l_022"]),
                  ("canon:C",["lat_l_022","lat_r_022"]),("canon:C_mo_s2",["lat_l_022","lat_r_022"]),("C_mo",["lat_l_022"]),("C_mo_s1",["lat_l_022"]),("C_s2",["lat_l_022"])]:
    m, jn, an, E = episodes(key, labs); ji = {n:i for i,n in enumerate(jn)}
    for lab in labs:
        eps = [e for e in E if e["label"] == lab]
        K = np.concatenate([np.clip(e["a"][:, 12:18], -1, 1) for e in eps]); H = np.concatenate([np.clip(e["a"][:, 6:12], -1, 1) for e in eps])
        cam = np.median([[e["qd"][:, ji[l+"_Body_CamShaft_RevoluteJoint"]].mean() for l in LEGS] for e in eps], 0)
        lean = np.median([-(e["fb"].mean(1)[:, 1] * np.sign(e["cmd"][:, 1].mean())).mean() for e in eps])
        cen = [ (e["fw"].mean(1)[-1] - e["fw"].mean(1)[0]) for e in eps]
        feet = np.median([abs(c[1]) / (len(e["fw"]) * 0.02) for c, e in zip(cen, eps)])
        spin = [f"{LEGS[i]}{cam[i]:+.1f}" for i in range(6) if abs(cam[i]) > 0.5]
        print(f"{key:14s} {lab:10s} {feet:.3f} {lean:+.3f}   {K.mean():+.2f}                 {np.round(K.mean(0),2)}   {H.mean():+.2f}   {spin}")
