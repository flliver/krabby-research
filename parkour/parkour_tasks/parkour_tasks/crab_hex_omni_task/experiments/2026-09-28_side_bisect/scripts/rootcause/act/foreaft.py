"""Q1: fore-aft is cam-driven. Per-leg cam (yaw) stance thrust in W fwd/back, spinning vs rocking cams, and fwd-vs-back parity of the yaw stroke at fixed gait phase."""
import numpy as np
from lib import *
from parity import analytic_phase, A, B, NB
DT = 0.02
def per_leg(key, lab):
    m, jn, an, E = episodes(key, [lab]); ji = {n:i for i,n in enumerate(jn)}
    T, C, Y, P = [], [], [], []
    for e in E:
        s = np.sign(e["cmd"][:, 0].mean()); d = np.array([s, 0.])
        u = e["fb"][..., :2] - e["hb"][..., :2]; rho = np.linalg.norm(u, axis=-1); er = u / rho[..., None]; et = np.stack([-er[..., 1], er[..., 0]], -1)
        vrel = np.gradient(e["fb"], DT, axis=0)[..., :2]; vt = (vrel * et).sum(-1); vr = (vrel * er).sum(-1)
        ld = e["F"] / np.maximum(e["F"].sum(1, keepdims=True), 1)
        T.append(np.stack([(-(vt * (et @ d)) * ld).mean(0), (-(vr * (er @ d)) * ld).mean(0)]))
        C.append([e["qd"][:, ji[l+"_Body_CamShaft_RevoluteJoint"]].mean() for l in LEGS])
        yaw = np.stack([e["q"][:, ji[l+"_Body_Hip_RevoluteJoint"]] for l in LEGS], 1)
        sgl = ld[:, A].sum(1) - ld[:, B].sum(1); sgl = np.convolve(sgl, np.ones(5)/5, mode="same")
        b = ((analytic_phase(sgl) + np.pi) / (2*np.pi) * NB).astype(int) % NB
        Y.append(np.stack([yaw[b == i].mean(0) if (b == i).sum() > 2 else np.full(6, np.nan) for i in range(NB)]))
        P.append(np.stack([np.clip(e["a"][:, :6], -1, 1)[b == i].mean(0) if (b == i).sum() > 2 else np.full(6, np.nan) for i in range(NB)]))
    T = np.median(np.array(T), 0); C = np.median(np.array(C), 0)
    print(f"== {key} {lab}: per-leg cam-thrust {np.round(T[0],3)} (sum {T[0].sum():+.3f}) rod-thrust sum {T[1].sum():+.3f} | cam spin {np.round(C,2)}")
    return np.nanmean(np.stack(Y), 0), np.nanmean(np.stack(P), 0)
yf, pf = per_leg("W", "fwd"); yb, pb = per_leg("W", "back")
dyf, dyb = yf - yf.mean(0), yb - yb.mean(0)
print("fwd~back yaw-profile corr per leg (fixed tripod phase):", np.round([np.corrcoef(dyf[:, j], dyb[:, j])[0, 1] for j in range(6)], 2))
print("fwd~back cam-action corr per leg:", np.round([np.corrcoef(pf[:, j]-pf[:, j].mean(), pb[:, j]-pb[:, j].mean())[0, 1] for j in range(6)], 2))
print("yaw amp (phase-binned std) fwd", np.round(dyf.std(0), 3), "back", np.round(dyb.std(0), 3))
