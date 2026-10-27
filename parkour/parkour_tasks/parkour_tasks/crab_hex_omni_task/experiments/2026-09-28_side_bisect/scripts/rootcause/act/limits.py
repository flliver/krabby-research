"""Are the leaned rows parked at their radial stroke ends? rho percentiles, rod-slew saturation, knee/hip action saturation by row."""
import numpy as np
from lib import *
for key, lab in [("canon:C","lat_l_022"),("W","lat_l"),("W","fwd"),("lr:C_lr_rsi","lat_r_022"),("lr:LR4","lat_l_022"),("O","lat_l_022"),("lr:C_lr_10","lat_r_022")]:
    m, jn, an, E = episodes(key, [lab])
    RHO=[]; LEAD=[]; ROD=[]; AH=[]; AK=[]; LEADSIDE=[]
    for e in E:
        c = e["cmd"][:, :2].mean(0)
        rho = np.linalg.norm(e["fb"][..., :2] - e["hb"][..., :2], axis=-1); RHO.append(rho)
        if abs(c[1]) > abs(c[0]): lead = np.sign(e["hb"][:, :, 1].mean(0)) * np.sign(c[1]) > 0
        else: lead = np.sign(e["hb"][:, :, 0].mean(0)) * np.sign(c[0]) > 0
        LEAD.append(np.broadcast_to(lead, rho.shape)); ROD.append(np.abs(e["rod"])); a = e["a"]
        AH.append(a[:, 6:12]); AK.append(a[:, 12:18])
    RHO = np.concatenate(RHO); LEAD = np.concatenate(LEAD); ROD = np.concatenate(ROD); AH = np.concatenate(AH); AK = np.concatenate(AK)
    rmax_h = ROD[:, :6].max(); rmax_k = ROD[:, 6:].max()
    sat_h = (ROD[:, :6] > 0.95 * rmax_h).mean(); sat_k = (ROD[:, 6:] > 0.95 * rmax_k).mean()
    def row(X, mask): return X[mask]
    print(f"== {key} {lab}: rho p5/50/95 lead {np.percentile(RHO[LEAD],[5,50,95]).round(3)} trail {np.percentile(RHO[~LEAD],[5,50,95]).round(3)} | all-leg p1/p99 {np.percentile(RHO,[1,99]).round(3)}")
    print(f"     rod max hip {rmax_h:.4f} knee {rmax_k:.4f} m/s; time at >=95% slew: hip {sat_h:.2f} knee {sat_k:.2f}")
    for nm, X in [("hip", AH), ("knee", AK)]:
        s = np.sign(X); 
        print(f"     a_{nm}: mean lead {np.clip(X,-1,1)[LEAD].mean():+.2f} trail {np.clip(X,-1,1)[~LEAD].mean():+.2f} | per-leg clipped mean {np.round(np.clip(X,-1,1).mean(0),2)} | |mean sign| per leg {np.round(np.abs(s.mean(0)),2)}")
