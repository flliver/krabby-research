"""Q2/Q3: is the direction carried by the STATIC (mean) part or by the DYNAMIC phase-locked stroke?
Gait phase phi from the tripod loading signal s = load(A) - load(B) (A = FL, MR, RL), analytic-signal phase.
For each physical leg: phase-binned radial rho_k(phi) and hip/knee actions; compare the two hands."""
import numpy as np
from lib import *
A = [0, 3, 4]; B = [1, 2, 5]; NB = 12
def analytic_phase(x):
    x = x - x.mean(); n = len(x); X = np.fft.fft(x); h = np.zeros(n); h[0] = 1
    if n % 2 == 0: h[n//2] = 1; h[1:n//2] = 2
    else: h[1:(n+1)//2] = 2
    return np.angle(np.fft.ifft(X * h))
def profiles(key, lab):
    m, jn, an, E = episodes(key, [lab])
    P_r, P_ah, P_ak, P_cam = [], [], [], []
    for e in E:
        F = e["F"]; ld = F / np.maximum(F.sum(1, keepdims=True), 1)
        s = ld[:, A].sum(1) - ld[:, B].sum(1)
        # smooth s lightly
        k = np.ones(5) / 5; s = np.convolve(s, k, mode="same")
        ph = analytic_phase(s); b = ((ph + np.pi) / (2*np.pi) * NB).astype(int) % NB
        u = e["fb"][..., :2] - e["hb"][..., :2]; rho = np.linalg.norm(u, axis=-1)
        a = np.clip(e["a"], -1, 1)
        def pb(x): return np.stack([x[b == i].mean(0) if (b == i).sum() > 2 else np.full(x.shape[1], np.nan) for i in range(NB)])
        P_r.append(pb(rho)); P_ah.append(pb(a[:, 6:12])); P_ak.append(pb(a[:, 12:18])); P_cam.append(pb(a[:, 0:6]))
    f = lambda L: np.nanmean(np.stack(L), 0)  # (NB, 6)
    return f(P_r), f(P_ah), f(P_ak), f(P_cam), len(E)
def compare(key, labL, labR):
    rL, hL, kL, cL, nL = profiles(key, labL); rR, hR, kR, cR, nR = profiles(key, labR)
    out = []
    print(f"== {key}  {labL}(n{nL}) vs {labR}(n{nR})   [per leg FL FR ML MR RL RR]")
    for name, XL, XR in [("rho", rL, rR), ("a_hip", hL, hR), ("a_knee", kL, kR), ("a_cam", cL, cR)]:
        mL, mR = XL.mean(0), XR.mean(0); dL, dR = XL - mL, XR - mR
        corr = [np.corrcoef(dL[:, j], dR[:, j])[0, 1] for j in range(6)]
        ampL = dL.std(0); ampR = dR.std(0)
        # static part split: direction-even (mL+mR)/2, direction-odd (mL-mR)/2 ; dynamic: even (dL+dR)/2 vs odd (dL-dR)/2
        s_odd = np.abs(mL - mR) / 2; d_even = ((dL + dR) / 2).std(0); d_odd = ((dL - dR) / 2).std(0)
        print(f"  {name:6s} dyn corr L~R {np.round(corr,2)}  | static odd {s_odd.mean():.3f}  dyn even {d_even.mean():.3f}  dyn odd {d_odd.mean():.3f}  (ampL {ampL.mean():.3f} ampR {ampR.mean():.3f})")
        out.append((name, np.nanmean(corr), s_odd.mean(), d_even.mean(), d_odd.mean()))
    return out
res = {}
for key in ["canon:C", "canon:C_mo_s2", "canon:C_mo_s1", "canon:C_s2", "W", "lr:LR4", "lr:LRfa", "lr:C_lr", "lr:C_lr_10", "lr:C_lr_rsi"]:
    if key == "W": res[key] = compare(key, "lat_l", "lat_r")
    else: res[key] = compare(key, "lat_l_022", "lat_r_022")
print("\nSUMMARY (rho and hip/knee actions): mean dyn corr L~R | static-odd | dyn-even | dyn-odd | odd share of dynamic = dyn_odd/(dyn_even+dyn_odd)")
for k, v in res.items():
    s = " || ".join(f"{n}: c{c:+.2f} so{so:.3f} de{de:.3f} do{do:.3f} share{do/(de+do):.2f}" for n, c, so, de, do in v[:3])
    print(f"{k:14s} {s}")
