"""M_y symmetry decomposition of side gaits (CPU, read-only). Library used by sym.py and odd_rms.py.

Copied from the side bisect (scripts/rootcause/sym/symlib.py) for the ppo_construction campaign; the only
changes are the durable, overridable CACHE (set_cache) -- every formula is verbatim.

M_y maps (verified against crab_hex_forward_task/mdp/crab_hex_mirror.py:49-56 and omni mirror):
  legs FL<->FR, ML<->MR, RL<->RR;  CamShaft -1, Body_Hip -1, Hip_Femur +1, Femur_Tibia -1;
  body-frame vectors (x, y, z) -> (x, -y, z);  roll -1, pitch +1.
Clock: HEAD_SIGNS_LR slots 15,16 = -1 (crab_hex_omni_mirror.py:38,149-150) => sin/cos(phi) -> sin/cos(phi+pi), i.e. the
M_y image of a state at clock phase phi is a state at phase phi+pi. For an M_y-equivariant policy the -y gait is
A_-y(phi) = S * A_+y[P](phi+pi). The "pi" mirror of a pattern is therefore (M_pi A)(phi) = S*A[P](phi+pi); the "0"
mirror (M_0 A)(phi) = S*A[P](phi) ignores the clock. Both are involutions, so A = E + O exactly with
E=(A+MA)/2, O=(A-MA)/2 orthogonal, ||A||^2=||E||^2+||O||^2 and odd_frac = ||A-MA||^2/(4||A||^2).
"""
import json
import os
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt, hilbert

#: The extract cache (ppo_construction copy, 2026-10-01: a DURABLE path, not the side bisect's /tmp
#: scratchpad). Override with KRABBY_SYM_CACHE or extract.py / sym.py / odd_rms.py --cache
#: (set_cache). Readers must use ``symlib.CACHE`` (module attribute), not a ``from symlib import *`` copy.
DEFAULT_CACHE = "/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/ppo_construction/sym_cache"
CACHE = Path(os.environ.get("KRABBY_SYM_CACHE", DEFAULT_CACHE))


def set_cache(path) -> Path:
    """Point every loader of this module at ``path`` (created if missing)."""
    global CACHE
    CACHE = Path(path)
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE
LEGS = ["FL", "FR", "ML", "MR", "RL", "RR"]
SWAP6 = np.array([1, 0, 3, 2, 5, 4])
LEFT = np.array([0, 2, 4]); RIGHT = np.array([1, 3, 5])
NB = 24
DT = 0.02


def blockmap(signs):
    """signs per 6-leg block -> (P, S) with mirrored[i] = S[i]*x[P[i]]."""
    P = np.concatenate([SWAP6 + 6 * b for b in range(len(signs))])
    S = np.concatenate([np.full(6, s, float) for s in signs])
    return P, S


MAPS = {
    "act": blockmap([-1, +1, -1]),   # cam speed, hip rod (Hip_Femur), knee rod (Femur_Tibia)
    "jp": blockmap([-1, +1, -1]),    # Body_Hip (passive yaw), Hip_Femur, Femur_Tibia
    "foot": blockmap([+1, -1, +1]),  # foot x_b, y_b, z_b
    "camv": blockmap([-1]),
    "body": (np.arange(4), np.array([1., -1., -1., 1.])),
}
BLOCKS = {"act": ("cam", "hip", "knee"), "jp": ("yaw", "hip", "knee"), "foot": ("x", "y", "z")}


def load(name, label):
    p = CACHE / f"{name}__{label}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {k: z[k] for k in z.files}


def mir(A, rep, shift_bins=0):
    """A: (NB, C) pattern -> M A with a circular phase shift (bins)."""
    P, S = MAPS[rep]
    return S[None, :] * np.roll(A, -shift_bins, axis=0)[:, P]


def mir_static(m, rep):
    P, S = MAPS[rep]
    return S * m[..., P]


def rms(A):
    return float(np.sqrt(np.mean(np.sum(A ** 2, axis=-1)))) if A.ndim == 2 else float(np.linalg.norm(A))


def pattern(X, ph, sel=None, nb=NB):
    """X (E,T,C) signal, ph (E,T) phase. Per-episode mean removed. Returns pattern (nb,C), lock R^2, counts."""
    if sel is not None:
        X, ph = X[sel], ph[sel]
    Xc = X - X.mean(1, keepdims=True)
    b = (np.floor((ph % (2 * np.pi)) / (2 * np.pi) * nb).astype(int)) % nb
    C = X.shape[-1]
    A = np.zeros((nb, C)); n = np.zeros(nb)
    flat = Xc.reshape(-1, C); bf = b.reshape(-1)
    ok = np.isfinite(flat).all(1) & (bf >= 0)
    np.add.at(A, bf[ok], flat[ok]); np.add.at(n, bf[ok], 1)
    A = A / np.maximum(n, 1)[:, None]
    tot = np.mean(np.sum(flat[ok] ** 2, 1))
    expl = np.sum(n[:, None] * A ** 2) / n.sum()
    return A, float(expl / tot), n


def intrinsic_phase(d, fs=1 / DT, fband=(0.25, 0.7)):
    """Pooled-PCA + Hilbert phase from joint positions over [t0+100, t0+550). Returns (E,300) phase for [t0+200,t0+500)
    and the dominant frequency. Phase origin is shared across the stamp-hand's episodes (common PCA basis); the
    mirror's phase shift under this phase is NOT defined, so only min-over-shift statistics use it."""
    X = d["jpx"].astype(float)                       # (E,450,18)
    mu = np.nanmean(X, 1, keepdims=True)
    Xc = np.nan_to_num(X - mu)
    ok = np.isfinite(X).all(2)
    flat = Xc[ok]
    flat = flat / (flat.std(0) + 1e-6)
    sd = Xc.reshape(-1, 18)[ok.reshape(-1)].std(0) + 1e-6
    U, s, Vt = np.linalg.svd(flat - flat.mean(0), full_matrices=False)
    v1 = Vt[0]
    p1 = (Xc / sd) @ v1                              # (E,450)
    # dominant frequency from pooled spectrum of p1 (zero-padded)
    nfft = 4096
    F = np.abs(np.fft.rfft(p1 * np.hanning(450)[None, :], n=nfft, axis=1)) ** 2
    f = np.fft.rfftfreq(nfft, 1 / fs)
    band = (f > fband[0]) & (f < fband[1])
    f0 = float(f[band][np.argmax(F.mean(0)[band])])
    lo, hi = 0.6 * f0, 1.6 * f0
    bb, aa = butter(2, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    y = filtfilt(bb, aa, p1, axis=1)
    th = np.angle(hilbert(y, axis=1))
    return th[:, 100:400], f0, float(s[0] ** 2 / np.sum(s ** 2))


def split_idx(n):
    i = np.arange(n)
    return i[0::2], i[1::2]


def selfmirror(X, ph, rep, nb=NB):
    """Self-mirror statistics of a pattern with phase ph. Returns dict."""
    E = len(X)
    A, r2, n = pattern(X, ph, nb=nb)
    h1, h2 = split_idx(E)
    A1, _, _ = pattern(X, ph, h1, nb); A2, _, _ = pattern(X, ph, h2, nb)
    nA = rms(A)
    out = dict(amp=nA, lockR2=r2)
    half = rms(A1 - A2) / nA
    out["d_half"] = half
    for tag, k in (("0", 0), ("pi", nb // 2)):
        MA = mir(A, rep, k)
        d = rms(A - MA) / nA
        out[f"d_{tag}"] = d
        out[f"odd_{tag}"] = d ** 2 / 4
        # noise-corrected odd fraction (noise energy of the full-sample mean ~ ||A1-A2||^2/4, half of it odd)
        nse = rms(A1 - A2) ** 2 / 4
        out[f"odd_{tag}_corr"] = float(max(rms(A - MA) ** 2 / 4 - nse / 2, 0) / max(nA ** 2 - nse, 1e-12))
        out[f"odd_{tag}_h1"] = (rms(A1 - mir(A1, rep, k)) / rms(A1)) ** 2 / 4
        out[f"odd_{tag}_h2"] = (rms(A2 - mir(A2, rep, k)) / rms(A2)) ** 2 / 4
        out[f"dx_{tag}"] = rms(A1 - mir(A2, rep, k)) / nA     # cross-half self-mirror; floor = d_half if even
    ds = np.array([rms(A - mir(A, rep, k)) / nA for k in range(nb)])
    out["d_min"] = float(ds.min()); out["tau_min_deg"] = float(np.argmin(ds) * 360 / nb)
    out["odd_min"] = float(ds.min() ** 2 / 4)
    return out, A


def pair_odd(A, rep, k=NB // 2):
    """Odd fraction per mirror pair (front, mid, rear) and per block."""
    O = (A - mir(A, rep, k)) / 2
    res = {}
    for pi, pr in enumerate(("front", "mid", "rear")):
        cols = [6 * b + j for b in range(3) for j in (2 * pi, 2 * pi + 1)]
        res[pr] = float(np.sum(O[:, cols] ** 2) / max(np.sum(A[:, cols] ** 2), 1e-12))
    for bi, bn in enumerate(BLOCKS[rep]):
        cols = list(range(6 * bi, 6 * bi + 6))
        res[bn] = float(np.sum(O[:, cols] ** 2) / max(np.sum(A[:, cols] ** 2), 1e-12))
        res[bn + "_share_of_A"] = float(np.sum(A[:, cols] ** 2) / np.sum(A ** 2))
    return res


def leg_amp(A):
    """Per-leg share of pattern energy (sum over the 3 blocks)."""
    e = np.array([np.sum(A[:, [6 * b + j for b in range(3)]] ** 2) for j in range(6)])
    return e / e.sum()


def lagged(X, rep, maxlag=200, clock_half=None):
    """Phase-free per-episode statistics (no pooling across episodes, no phase variable).
    r(tau) = <x(t), x(t+tau)> / <x,x>   (autocorrelation, mean over episodes)
    m(tau) = <x(t), M x(t+tau)> / <x,x>  (self-mirror correlation; m(-tau)=m(tau) since M is a symmetric involution)
    T* = argmax r over [40, maxlag] steps (gait period). Evenness at shift tau: m(tau)/r(T*) (+1 even, -1 odd).
    odd fraction at shift tau (normalized): (1 - m(tau)/r(T*))/2."""
    P, S = MAPS[rep]
    Xc = X - X.mean(1, keepdims=True)
    Mx = S[None, None, :] * Xc[:, :, P]
    E, T, C = Xc.shape
    r = np.zeros(maxlag + 1); m = np.zeros(maxlag + 1)
    den = np.sum(Xc ** 2, axis=(1, 2)) / T
    for tau in range(maxlag + 1):
        a = Xc[:, : T - tau]; b = Xc[:, tau:]; mb = Mx[:, tau:]
        r[tau] = np.mean(np.sum(a * b, axis=(1, 2)) / (T - tau) / den)
        m[tau] = np.mean(np.sum(a * mb, axis=(1, 2)) / (T - tau) / den)
    Ts = int(40 + np.argmax(r[40:maxlag + 1]))
    rp = r[Ts]
    out = dict(T_star_s=Ts * DT, r_peak=float(rp), m0=float(m[0] / rp), mhalf=float(m[Ts // 2] / rp),
               odd0_lag=float((1 - m[0] / rp) / 2), oddhalf_lag=float((1 - m[Ts // 2] / rp) / 2),
               m_max=float(m[: Ts + 1].max() / rp), tau_mmax_frac=float(np.argmax(m[: Ts + 1]) / Ts))
    if clock_half is not None:
        k = int(round(clock_half))
        out["mclock"] = float(m[k] / rp); out["oddclock_lag"] = float((1 - m[k] / rp) / 2)
    return out


# sub-representations: (base rep, block indices) -> own (P, S)
SUB = {
    "act_rod": ("act", (1, 2)), "act_cam": ("act", (0,)),
    "jp_rod": ("jp", (1, 2)), "jp_yaw": ("jp", (0,)),
    "foot_yz": ("foot", (1, 2)), "foot_y": ("foot", (1,)), "foot_z": ("foot", (2,)), "foot_x": ("foot", (0,)),
}
_SIGNS = {"act": (-1, +1, -1), "jp": (-1, +1, -1), "foot": (+1, -1, +1)}
for _k, (_b, _blocks) in SUB.items():
    MAPS[_k] = blockmap([_SIGNS[_b][i] for i in _blocks])


def sub(d, key):
    base, blocks = SUB[key]
    cols = np.concatenate([np.arange(6 * b, 6 * b + 6) for b in blocks])
    return d[base][..., cols].astype(float)


def harmonic_split(A, rep):
    """Energy of pattern A (NB,C) split into {L/R-symmetric (M_0-even), antisymmetric (M_0-odd)} x {odd, even harmonics}.
    M_pi-odd = sym@odd-k + anti@even-k. Returns fractions of ||A||^2."""
    P, S = MAPS[rep]
    Ms = S[None, :] * A[:, P]
    sym = (A + Ms) / 2; anti = (A - Ms) / 2
    out = {}
    for nm, X in (("sym", sym), ("anti", anti)):
        F = np.fft.rfft(X, axis=0)
        k = np.arange(F.shape[0])
        e = np.abs(F) ** 2 * np.where((k == 0) | (k == A.shape[0] // 2), 1, 2)[:, None]
        out[nm + "_oddk"] = float(e[k % 2 == 1].sum()); out[nm + "_evenk"] = float(e[(k % 2 == 0) & (k > 0)].sum())
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items()}
