"""Per-leg roles: lead/trail row, load share, rod thrust, slip along travel / fore-aft, cam spin, radial-vs-load coherence."""
import numpy as np
from lib import *
DT = 0.02; FTH = 50.0
def run(key, lab):
    m, jn, an, E = episodes(key, [lab]); ji = {n:i for i,n in enumerate(jn)}
    rows = []
    for e in E:
        c = e["cmd"][:, :2].mean(0); d = np.array([0., np.sign(c[1]), 0.])
        fb, hb = e["fb"], e["hb"]
        side = np.sign(hb[:, :, 1].mean(0)) * d[1]   # +1 leading row
        u = fb[..., :2] - hb[..., :2]; rho = np.linalg.norm(u, axis=-1)
        rhod = np.gradient(rho, DT, axis=0)
        st = e["F"] > FTH; load = e["F"] / np.maximum(e["F"].sum(1, keepdims=True), 1)
        fvb = np.einsum('tij,tkj->tki', np.transpose(e["R"], (0,2,1)), e["fv"])
        thrust_k = -side[None] * rhod   # body-direction velocity if foot fixed, from radial motion  (lead: retract=+)
        r = {}
        r["load"] = load.mean(0)
        r["thr"] = (thrust_k * load).mean(0) * 1.0  # load-weighted contribution
        r["slipd"] = np.array([np.mean(fvb[st[:, k], k] @ d) if st[:, k].any() else np.nan for k in range(6)])
        r["slipx"] = np.array([np.mean(np.abs(fvb[st[:, k], k, 0])) if st[:, k].any() else np.nan for k in range(6)])
        r["coh"] = np.array([np.corrcoef(e["F"][:, k], thrust_k[:, k])[0, 1] if e["F"][:, k].std() > 0 else np.nan for k in range(6)])
        r["cam"] = np.array([e["qd"][:, ji[l+"_Body_CamShaft_RevoluteJoint"]].mean() for l in LEGS])
        r["rho"] = rho.mean(0); r["rhoamp"] = rho.std(0) * 2.83
        r["side"] = side
        rows.append(r)
    out = {k: np.nanmedian(np.stack([r[k] for r in rows]), 0) for k in rows[0]}
    print(f"== {key} {lab} n{len(rows)}   legs {LEGS}")
    for k in ["side","load","thr","coh","slipd","slipx","cam","rho","rhoamp"]:
        print(f"   {k:6s} " + " ".join(f"{v:+.3f}" for v in out[k]))
    lead = out["side"] > 0
    print(f"   rows: load lead {out['load'][lead].sum():.2f} trail {out['load'][~lead].sum():.2f} | thrust lead {out['thr'][lead].sum():+.3f} trail {out['thr'][~lead].sum():+.3f} "
          f"| slip_d lead {np.nanmean(out['slipd'][lead]):+.3f} trail {np.nanmean(out['slipd'][~lead]):+.3f} | coh lead {np.nanmean(out['coh'][lead]):+.2f} trail {np.nanmean(out['coh'][~lead]):+.2f}")
    return out
for key, lab in [("W","fwd"),("W","lat_l"),("W","lat_r"),("canon:C","lat_l_022"),("canon:C","lat_r_022"),("C_mo_s1","lat_l_022"),("O","lat_l_022"),
                 ("lr:LR4","lat_l_022"),("lr:LR4","lat_r_022"),("lr:C_lr","lat_l_022"),("lr:C_lr","lat_r_022"),("lr:C_lr_10","lat_l_022"),("lr:C_lr_10","lat_r_022"),("lr:C_lr_rsi","lat_r_022")]:
    if key == "W" and lab == "fwd": continue
    run(key, lab)
