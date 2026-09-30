"""One-hand parity: how HANDED is a +y-only policy's own phase-locked schedule?
P(phi) = per-channel clock-phase profile on lat_l (steady window). Its M_y image is Q P(phi+pi) (leg swap + joint
signs, tripod sets swap => clock advanced by pi). Split P into the M_y-symmetric part S = (P + Q P(.+pi))/2 and the
M_y-antisymmetric part A = (P - Q P(.+pi))/2, each phase-modulated (phase mean removed). A is the handed schedule
(a gait whose mirror image is itself carries A = 0 and cannot travel sideways by stroking). foot_y_b: the same with
M_y acting as leg swap and y -> -y."""
import glob, json, numpy as np
from pathlib import Path
from decomp import load, mean_prof, NB
LOGS = "/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-28_side_bisect"
OUT = "/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/side_bisect"
SWAP = {"FL": "FR", "FR": "FL", "ML": "MR", "MR": "ML", "RL": "RR", "RR": "RL"}
def sgn(n):
    for t, s in (("_Body_CamShaft_", -1.), ("_Body_Hip_", -1.), ("_Hip_Femur_", 1.), ("_Femur_Tibia_", -1.)):
        if t in n: return s
def amap(names):
    idx = {n: i for i, n in enumerate(names)}
    return np.array([idx[SWAP[n[:2]] + n[2:]] for n in names]), np.array([sgn(n) for n in names])
LEGS = ["FL", "FR", "ML", "MR", "RL", "RR"]; LP = np.array([LEGS.index(SWAP[l]) for l in LEGS])
one = lambda g: sorted(glob.glob(g))[-1]
CASES = [("C", "F", f"{LOGS}/seed3_X02_s1_A15pB/side22_25__omni/seed001/*/"), ("C_s2", "F", f"{LOGS}/seed2_X07_s1_A15pB/side22_25__omni/seed001/*/"),
         ("C_mo_s2", "F", f"{LOGS}/seed2_X08_s1_A15pB/side22_25__omni/seed001/*/"), ("C_mo_s1", "F", f"{LOGS}/seed1_X09_s1_A15pB/side22_25__omni/seed001/*/"),
         ("C_mo", "I", f"{LOGS}/seed3_X04_s1_A15pB/side22_25__omni/seed001/*/"),
         ("O", "S", f"{LOGS}/seed3_X01_s1_A15pB/side22_25__omni/seed001/*/"), ("R", "S", f"{LOGS}/seed3_X03_s1_A15pB/side22_25__omni/seed001/*/"),
         ("C_band", "S", f"{LOGS}/seed3_X05_s1_A15pB/side22_25__omni/seed001/*/"), ("C_rsi", "S", f"{LOGS}/seed3_X06_s1_A15pB/side22_25__omni/seed001/*/"),
         ("LR4 (lat_l)", "S", f"{OUT}/lr/LR4/side22_25lr__omni/seed001/*/"), ("C_lr (lat_l)", "S", f"{OUT}/lr/C_lr/side22_25lr__omni/seed001/*/"),
         ("LRfa (lat_l)", "S", f"{OUT}/lr/LRfa/side22_25lr__omni/seed001/*/"), ("C_lr_rsi (lat_l)", "S", f"{OUT}/lr/C_lr_rsi/side22_25lr__omni/seed001/*/")]
rms = lambda X: float(np.sqrt(np.nanmean(X ** 2)))
print("%-18s %-2s %-5s | rods: A_mod  S_mod  handed-frac | foot_y_b: A_mod S_mod handed-frac" % ("policy", "cl", "spd"))
out = {}
for name, cl, g in CASES:
    st = one(g); meta, W = load(st, ["lat_l_022", "lat_l_025"])
    Pa, Sa = amap(meta["action_joint_names"])
    for lab in ("lat_l_022", "lat_l_025"):
        if not W[lab]: continue
        P = mean_prof(W[lab], "a", lambda a: a)                     # NB x 18
        Pm = Sa[None] * P[:, Pa]; Pm = np.roll(Pm, -NB // 2, axis=0)  # Q P(phi + pi)
        A = (P - Pm) / 2; S = (P + Pm) / 2; A -= A.mean(0); S -= S.mean(0)
        F = mean_prof(W[lab], "fb", lambda f: f[:, :, 1])            # NB x 6 (foot y in body frame)
        Fm = -F[:, LP]; Fm = np.roll(Fm, -NB // 2, axis=0)
        FA = (F - Fm) / 2; FS = (F + Fm) / 2; FA -= FA.mean(0); FS -= FS.mean(0)
        ra, rs = rms(A[:, 6:]), rms(S[:, 6:]); fa_, fs = rms(FA), rms(FS)
        out[f"{name}|{lab}"] = dict(rodA=ra, rodS=rs, footA=fa_, footS=fs)
        print("%-18s %-2s %-5s | %6.3f %6.3f %6.2f | %6.3f %6.3f %6.2f" % (name, cl, lab[-3:], ra, rs, ra / np.hypot(ra, rs), fa_, fs, fa_ / np.hypot(fa_, fs)))
json.dump(out, open("selfmirror.json", "w"), indent=1)
