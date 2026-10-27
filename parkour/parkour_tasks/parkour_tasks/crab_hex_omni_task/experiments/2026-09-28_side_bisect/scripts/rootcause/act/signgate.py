"""Premise check for a sign-gated odd head: at a FIXED clock phase, is each physical leg's loading the same on both hands
while its radial stroke / knee+hip actions flip sign?  (canonical fast heads, clock visible)"""
import numpy as np
from lib import *
NB = 12
def prof(key, lab):
    m, jn, an, E = episodes(key, [lab])
    L, Rh, AH, AK = [], [], [], []
    for e in E:
        b = ((np.mod(e["clock"], 2*np.pi)) / (2*np.pi) * NB).astype(int) % NB
        F = e["F"]; rho = np.linalg.norm(e["fb"][..., :2] - e["hb"][..., :2], axis=-1); a = np.clip(e["a"], -1, 1)
        pb = lambda x: np.stack([x[b == i].mean(0) if (b == i).sum() > 1 else np.full(x.shape[1], np.nan) for i in range(NB)])
        L.append(pb(F)); Rh.append(pb(rho)); AH.append(pb(a[:, 6:12])); AK.append(pb(a[:, 12:18]))
    f = lambda X: np.nanmean(np.stack(X), 0)
    return f(L), f(Rh), f(AH), f(AK)
def c(a, b): a = a - a.mean(0); b = b - b.mean(0); return np.array([np.corrcoef(a[:, j], b[:, j])[0, 1] for j in range(6)])
for key in ["canon:C", "canon:C_mo_s2", "canon:C_s2", "canon:C_mo_s1", "lr:LR4", "lr:C_lr", "lr:C_lr_rsi"]:
    l = prof(key, "lat_l_022"); r = prof(key, "lat_r_022")
    # clock-locking quality: loading modulation depth by clock phase
    depth = (l[0].max(0) - l[0].min(0)) / (l[0].mean(0) + 1)
    print(f"{key:14s} load-depth {np.round(depth,2)}")
    for n, i in [("load", 0), ("rho", 1), ("a_hip", 2), ("a_knee", 3)]:
        cc = c(l[i], r[i]); print(f"   {n:7s} clock-phase corr L~R per leg {np.round(cc,2)}  mean {np.nanmean(cc):+.2f}")
