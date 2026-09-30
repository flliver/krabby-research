"""Direction-parity decomposition of side (and fore-aft) gaits, CPU only, from eval raw npz.

For a pair of opposite holds (X, Y) of ONE policy (lat_l vs lat_r, or fwd vs back), per channel k and
clock-phase bin b (steady window hold+4..10 s):
    E = (P_X + P_Y)/2  (direction-EVEN)      O = (P_X - P_Y)/2  (direction-ODD)
each split into static (mean over phase) and modulated (phase-varying) parts:
    E0, E1 (shuffle: phase-locked, same for both directions), O0 (lean / static bias), O1 (the STROKE:
    phase-locked AND flips with direction; needs a direction x phase interaction in the policy).
Channels: foot position in the body frame along the travel axis (y for side, x for fore-aft), the rod
actions (clipped), the cam speed actions (clipped). Split-half noise: same stats between even/odd episodes
of one hold (should be ~0 for a clean signal).
"""
import json, glob, sys
from pathlib import Path
import numpy as np

NB = 16
def qrot_inv(q, v):  # q (T,4) wxyz, v (T,N,3): rotate world->body (R^T v)
    w, x, y, z = [q[:, i][:, None] for i in range(4)]
    # rotation matrix
    R = np.stack([
        np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], -1),
        np.stack([2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], -1),
        np.stack([2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1)], -2)  # (T,1,3,3)
    return np.einsum('tnji,tnj->tni', np.broadcast_to(R, (v.shape[0], v.shape[1], 3, 3)), v)

def load(stamp, labels):
    d = Path(stamp); meta = json.loads((d/"run_meta.json").read_text()); hl = meta["hold_labels"]
    out = {l: [] for l in labels}
    for e in meta["env_index_map"]:
        lab = e["label"]
        if lab not in out: continue
        z = np.load(d/"raw"/f"episode_{e['env_index']:02d}.npz")
        done = z["done"].astype(bool); S = int(np.argmax(done)) if done.any() else len(done)
        sid = hl.index(lab); seg = z["segment_id"]
        idx = np.where((seg == sid) & (np.arange(len(seg)) < S))[0]
        if len(idx) < 500: continue
        t0 = idx[0]; sl = slice(t0 + 200, t0 + 500)
        fp = z["foot_pos_w"][sl].astype(float) - z["root_pos_w"][sl][:, None, :].astype(float)
        fb = qrot_inv(z["root_quat_w"][sl].astype(float), fp)
        out[lab].append(dict(ph=np.mod(z["clock_phase"][sl].astype(float), 2*np.pi),
                             a=np.clip(z["actions"][sl].astype(float), -1, 1), fb=fb,
                             F=(z["foot_force_norm"][sl] > 1.0).astype(float),
                             run=z["clock_running"][sl].astype(bool) if "clock_running" in z.files else None))
    return meta, out

def prof(ep, key, sub=None):
    X = ep[key] if sub is None else sub(ep[key])
    b = np.minimum((ep["ph"] / (2*np.pi) * NB).astype(int), NB-1)
    P = np.full((NB, X.shape[1]), np.nan)
    for i in range(NB):
        m = b == i
        if m.any(): P[i] = X[m].mean(0)
    return P

def mean_prof(eps, key, sub=None):
    return np.nanmean(np.stack([prof(e, key, sub) for e in eps]), 0)

def parity(PX, PY):
    E = (PX + PY) / 2; O = (PX - PY) / 2
    E0 = E.mean(0, keepdims=True); O0 = O.mean(0, keepdims=True)
    E1 = E - E0; O1 = O - O0
    rms = lambda A: float(np.sqrt(np.nanmean(A**2)))
    e1, o1, o0, e0 = rms(E1), rms(O1), rms(O0), rms(E0)
    return dict(E1=e1, O1=o1, O0=o0, E0=e0, r_int=o1/np.sqrt(o1**2+e1**2+1e-12))

def analyse(name, stamp, X, Y, axis):
    meta, W = load(stamp, [X, Y])
    if not W[X] or not W[Y]:
        print(f"{name}: missing holds"); return None
    ax = {"x": 0, "y": 1}[axis]
    subs = {
        f"foot_{axis}_b": ("fb", lambda f: f[:, :, ax]),
        "rod_act": ("a", lambda a: a[:, 6:]),
        "cam_act": ("a", lambda a: a[:, :6]),
        "contact": ("F", None),
    }
    res = {"n": (len(W[X]), len(W[Y]))}
    for ch, (key, sub) in subs.items():
        PX = mean_prof(W[X], key, sub); PY = mean_prof(W[Y], key, sub)
        r = parity(PX, PY)
        # split-half noise on X alone: parity between even/odd episodes of X (O-part should be ~noise)
        hx = parity(mean_prof(W[X][0::2], key, sub), mean_prof(W[X][1::2], key, sub))
        r["noise_O1"] = hx["O1"]; r["noise_O0"] = hx["O0"]
        # per-hand modulated amplitude (how much each hand oscillates at all)
        r["amp_X"] = float(np.sqrt(np.nanmean((PX - PX.mean(0))**2))); r["amp_Y"] = float(np.sqrt(np.nanmean((PY - PY.mean(0))**2)))
        res[ch] = r
    return res

if __name__ == "__main__":
    OUT = "/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/side_bisect"
    LOGS = "/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases"
    one = lambda g: sorted(glob.glob(g))[-1]
    CASES = []
    for h in ["C", "C_s2", "C_mo_s2", "C_mo_s1"]:
        CASES.append((f"canon {h}", one(f"{OUT}/canon/{h}/wave/side22_25lr__omni/seed001/*/"), "FAST(canon)"))
    for a, c in [("LRfa", "SLOW"), ("LR4", "SLOW"), ("LR4_dwell", "DEAD"), ("C_lr", "SLOW"), ("C_lr_10", "SLOW"),
                 ("C_lr_rsi", "SLOW"), ("C_lr_10_rsi", "SLOW/DEAD")]:
        CASES.append((a, one(f"{OUT}/lr/{a}/side22_25lr__omni/seed001/*/"), c))
    res = {}
    hdr = "%-18s %-11s %-9s | %6s %6s %6s %6s  r_int (noise O1) | amp X/Y"
    for ch in ["foot_y_b", "rod_act", "cam_act", "contact"]:
        print(f"\n=== side pair lat_l vs lat_r (0.22 and 0.25 pooled per speed), channel {ch}")
        print(hdr % ("policy", "class", "speed", "E1", "O1", "O0", "E0"))
        for name, st, cls in CASES:
            for spd in ("022", "025"):
                key = (name, spd)
                if key not in res:
                    res[key] = analyse(name, st, f"lat_l_{spd}", f"lat_r_{spd}", "y")
                r = res[key]
                if r is None: continue
                v = r[ch]
                print("%-18s %-11s %-9s | %6.3f %6.3f %6.3f %6.3f  %5.2f (%5.3f) | %5.3f/%5.3f n%s" % (
                    name, cls, spd, v["E1"], v["O1"], v["O0"], v["E0"], v["r_int"], v["noise_O1"], v["amp_X"], v["amp_Y"], r["n"]))
    # Fore-aft reversal inside full-omni heads (clock visible): fwd vs back; and the side pair of the same heads
    SUITE = {"t3 (X08, 4-img, clock vis)": f"{LOGS}/2026-09-23_clock_deprecation/step0_E3_X08-t3_visible/suite__omni/seed001/2026-09-23_21-38-11",
             "c6 (X01, 4-img, clock vis)": f"{LOGS}/2026-09-23_clock_deprecation/seed3_X01_c6_A15pB/suite__omni/seed001/2026-09-24_00-55-05"}
    for name, st in SUITE.items():
        rf = analyse(name, st, "fwd", "back", "x"); rs = analyse(name, st, "lat_l", "lat_r", "y")
        for lab, r, axis in (("fwd/back", rf, "x"), ("lat_l/lat_r", rs, "y")):
            for ch in [f"foot_{axis}_b", "rod_act", "cam_act", "contact"]:
                v = r[ch]
                print("%-28s %-11s %-9s | E1 %6.3f O1 %6.3f O0 %6.3f E0 %6.3f  r_int %5.2f (noise %5.3f) | amp %5.3f/%5.3f n%s" % (
                    name, lab, ch, v["E1"], v["O1"], v["O0"], v["E0"], v["r_int"], v["noise_O1"], v["amp_X"], v["amp_Y"], r["n"]))
    json.dump({f"{k[0]}|{k[1]}": v for k, v in res.items()}, open("decomp.json", "w"), indent=1, default=float)
