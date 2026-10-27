"""Per hand: phase-locked radial stroke amplitude (tripod-phase-binned rho, std over bins, mean over legs) vs static carriers."""
import numpy as np
from lib import *
from parity import profiles
rows = []
for key, labs in [("canon:C",["lat_l_022","lat_r_022"]),("canon:C_mo_s2",["lat_l_022","lat_r_022"]),("canon:C_s2",["lat_l_022","lat_r_022"]),("canon:C_mo_s1",["lat_l_022","lat_r_022"]),
                  ("C_mo",["lat_l_022"]),("O",["lat_l_022"]),("R",["lat_l_022"]),("C_band",["lat_l_022"]),("C_rsi",["lat_l_022"]),
                  ("lr:LRfa",["lat_l_022","lat_r_022"]),("lr:LR4",["lat_l_022","lat_r_022"]),("lr:LR4_dwell",["lat_l_022","lat_r_022"]),("lr:C_lr",["lat_l_022","lat_r_022"]),
                  ("lr:C_lr_10",["lat_l_022","lat_r_022"]),("lr:C_lr_rsi",["lat_l_022","lat_r_022"]),("lr:C_lr_10_rsi",["lat_l_022","lat_r_022"]),("W",["lat_l","lat_r"])]:
    for lab in labs:
        try:
            r, h, k, c, n = profiles(key, lab)
        except Exception as ex:
            print(key, lab, "ERR", ex); continue
        amp = (r - r.mean(0)).std(0).mean(); kamp = (k - k.mean(0)).std(0).mean(); hamp = (h - h.mean(0)).std(0).mean()
        m, jn, an, E = episodes(key, [lab])
        feet = np.median([abs(((e["fw"].mean(1)[-1] - e["fw"].mean(1)[0]))[1]) / (len(e["fw"]) * 0.02) for e in E])
        kcm = np.mean([np.clip(e["a"][:, 12:18], -1, 1).mean() for e in E])
        rows.append((key, lab, feet, amp, kamp, hamp, abs(kcm), n))
        print(f"{key:15s} {lab:10s} n{n:3d} feet {feet:.3f} | rho stroke amp {amp:.3f} | knee dyn amp {kamp:.2f} hip dyn amp {hamp:.2f} | |knee_cm| {abs(kcm):.2f}")
import json; json.dump(rows, open("amp.json", "w"))
f = np.array([r[2] for r in rows]); a = np.array([r[3] for r in rows]); kc = np.array([r[6] for r in rows])
print("corr(feet, rho stroke amp) =", np.corrcoef(f, a)[0, 1].round(3), " corr(feet, |knee_cm|) =", np.corrcoef(f, kc)[0, 1].round(3))
