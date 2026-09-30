"""Q3: onset time course. Body vs feet-centroid lateral speed, knee common mode, leading/trailing radial position, per 0.5-1 s bin from hold start."""
import numpy as np
from lib import *
BINS = [(0,25),(25,50),(50,100),(100,150),(150,200),(200,300),(300,500)]
def run(key, lab):
    m, jn, an, E = episodes(key, [lab], lo=0, hi=500)
    rows = []
    for e in E:
        s = np.sign(e["cmd"][:, 1].mean()); d = np.array([0, s, 0.])
        vb = e["vb"] @ d
        cen = e["fw"].mean(1); dw = e["R"][:, :, 1] * s  # world direction of body +d
        vfe = np.r_[0, np.einsum('ti,ti->t', np.diff(cen, axis=0), dw[1:]) / 0.02]
        rho = np.linalg.norm(e["fb"][..., :2] - e["hb"][..., :2], axis=-1)
        lead = np.sign(e["hb"][:, :, 1].mean(0)) * s > 0
        kcm = np.clip(e["a"][:, 12:18], -1, 1).mean(1) * (-s)   # signed so + = lean toward travel
        lean = -(e["fb"].mean(1) @ d)
        nst = (e["F"] > 50).sum(1)
        r = []
        for a, b in BINS:
            r.append([vb[a:b].mean(), vfe[a:b].mean(), kcm[a:b].mean(), rho[a:b][:, lead].mean(), rho[a:b][:, ~lead].mean(), lean[a:b].mean(), nst[a:b].mean()])
        rows.append(r)
    R = np.median(np.array(rows), 0)
    print(f"== {key} {lab} n{len(rows)}   bins(s): " + " ".join(f"{a*0.02:.1f}-{b*0.02:.1f}" for a, b in BINS))
    for i, nm in enumerate(["v_body", "v_feet", "knee_cm*", "rho_lead", "rho_trail", "lean", "n_loaded"]):
        print(f"   {nm:9s} " + " ".join(f"{v:+.3f}" for v in R[:, i]))
for key, lab in [("canon:C","lat_l_022"),("canon:C_mo_s2","lat_l_022"),("W","lat_l"),("O","lat_l_022"),("lr:C_lr_rsi","lat_r_022"),("lr:LRfa","lat_l_022"),("lr:LR4","lat_l_022"),("lr:C_lr_10","lat_r_022")]:
    run(key, lab)
