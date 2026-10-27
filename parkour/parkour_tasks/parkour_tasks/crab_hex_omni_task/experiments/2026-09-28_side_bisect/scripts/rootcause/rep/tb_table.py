import pickle, sys
import numpy as np
D = pickle.load(open("tb_cache.pkl", "rb"))
ORDER = ["C", "C_s2", "C_mo_s2", "C_mo_s1", "C_mo", "O", "R", "C_band", "C_rsi", "LRfa", "LR4", "LR4_dwell",
         "C_lr", "C_lr_10", "C_lr_rsi", "C_lr_10_rsi", "C_lr_rsi33"]
WINS = [(0, 25), (25, 50), (50, 100), (100, 200), (200, 300), (300, 450), (450, 600), (600, 900), (900, 1200),
        (1200, 1700), (1700, 2200), (2200, 2600), (2600, 3000)]
def ser(lab, t):
    a = D[lab].get(t)
    if a is None or len(a) == 0: return None
    s = np.full(int(a[:, 0].max()) + 1, np.nan); s[a[:, 0].astype(int)] = a[:, 1]; return s
def win(s, lo, hi):
    if s is None or len(s) <= lo: return np.nan
    return np.nanmean(s[lo:hi])
def table(name, f, fmt="%7.3f"):
    print("==", name)
    print("%-12s %-4s %-8s " % ("arm", "cls", "dirs") + " ".join("%7s" % f"{a}-{b}" for a, b in WINS))
    for lab in ORDER:
        vals = [f(lab, a, b) for a, b in WINS]
        print("%-12s %-4s %-8s " % (lab, D[lab]["cls"], D[lab]["dirs"]) + " ".join(fmt % v for v in vals))
T = lambda t: (lambda lab, a, b: win(ser(lab, t), a, b))
table("symmetry loss (MSE on unclipped means, raw)", T("Loss/symmetry"), "%7.3f")
def share(lab, a, b):
    s = ser(lab, "Loss/symmetry"); su = ser(lab, "Loss/surrogate"); v = ser(lab, "Loss/value_function")
    e = ser(lab, "Loss/entropy"); pr = ser(lab, "Loss/priv_reg"); pc = ser(lab, "Loss/priv_reg_coef")
    tot = su + v - 0.01 * e + pc * pr + 0.5 * s
    return win(0.5 * s / np.abs(tot), a, b)
table("0.5*symmetry / |total loss| (total = surr + value - 0.01*ent + prc*priv_reg + 0.5*sym)", share)
def share_actor(lab, a, b):
    s = ser(lab, "Loss/symmetry"); e = ser(lab, "Loss/entropy"); su = ser(lab, "Loss/surrogate")
    return win(0.5 * s / (0.5 * s + np.abs(su) + 0.01 * np.abs(e)), a, b)
table("0.5*sym / (0.5*sym + |surrogate| + 0.01*|entropy|)  [actor-side terms only]", share_actor)
table("surrogate", T("Loss/surrogate"), "%7.4f")
table("value_function", T("Loss/value_function"), "%7.3f")
table("noise std", T("Policy/mean_noise_std"), "%7.3f")
table("learning rate (x1e4)", lambda l, a, b: 1e4 * win(ser(l, "Loss/learning_rate"), a, b), "%7.2f")
table("track_ratio_mean", T("Metrics/base_parkour/track_ratio_mean"), "%7.3f")
table("mean_reward", T("Train/mean_reward"), "%7.2f")
