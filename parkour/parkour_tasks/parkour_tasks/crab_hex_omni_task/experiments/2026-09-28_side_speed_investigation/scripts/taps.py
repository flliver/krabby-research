import sys, numpy as np, json, math
sys.path.insert(0, "/tmp/claude-1000/sidespeed")
from kin3 import eps, LEFT, jidx, L1, L2
def taps(tag, label):
    out = {"lead": [], "trail": []}
    for z, meta, ids in eps(tag, label):
        seg = z["segment_id"].astype(int); done = z["done"].astype(bool)
        S = int(np.argmax(done)) if done.any() else len(seg)-1
        m = np.isin(seg, ids) & z["steady_mask"].astype(bool) & (np.arange(len(seg)) < S)
        if m.sum() < 50: continue
        cmd = z["cmd_applied"][:, :2].astype(float)
        if label == "side": sgn = 1.0
        else: sgn = np.sign(np.median(cmd[m, 1]))
        lead_leg = ~LEFT if sgn > 0 else LEFT
        F = z["foot_force_norm"].astype(float); ld = F > 50
        fz = z["foot_pos_w"][..., 2].astype(float) - z["terrain_z"].astype(float)
        idx = np.where(m)[0]; lo, hi = idx.min(), idx.max()
        for f in range(6):
            row = "lead" if lead_leg[f] else "trail"
            l = ld[lo:hi+1, f]; d = np.diff(np.concatenate([[0], l.astype(int), [0]]))
            st = np.where(d == 1)[0]; en = np.where(d == -1)[0]
            runs = [(a, b) for a, b in zip(st, en) if a > 0 and b < len(l)]
            for i, (a, b) in enumerate(runs):
                if b - a >= 5: continue
                prev = [r for r in runs[:i] if r[1]-r[0] >= 5]; nxt = [r for r in runs[i+1:] if r[1]-r[0] >= 5]
                gap_prev = (a - prev[-1][1])*0.02 if prev else np.nan
                gap_next = (nxt[0][0] - b)*0.02 if nxt else np.nan
                pk = F[lo+a:lo+b, f].max()
                zmax_before = fz[lo+max(a-10, 0):lo+a, f].max() - fz[lo+a, f]
                out[row].append((b-a, gap_prev, gap_next, pk, zmax_before))
    for row, v in out.items():
        v = np.array(v)
        if len(v) == 0: continue
        print(tag, label, row, "n", len(v), "len_steps p50 %.1f" % np.median(v[:, 0]), "gap_after_stance p50 %.2fs" % np.nanmedian(v[:, 1]), "gap_to_next_stance p50 %.2fs" % np.nanmedian(v[:, 2]),
              "peakF p50 %.0f N" % np.median(v[:, 3]), "rise before tap p50 %.3f m" % np.median(v[:, 4]),
              "frac within 0.2s of stance end %.2f" % np.nanmean(v[:, 1] <= 0.2), "frac within 0.2s before next stance %.2f" % np.nanmean(v[:, 2] <= 0.2))
for tag, label in [("W", "lat_l"), ("t3", "lat_l"), ("SH", "side"), ("W", "fwd")]:
    taps(tag, label)
