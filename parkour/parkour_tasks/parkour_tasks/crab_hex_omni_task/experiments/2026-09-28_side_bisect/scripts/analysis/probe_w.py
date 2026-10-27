import json, numpy as np
from pathlib import Path
d = Path("/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-23_clock_deprecation_w2/seed3_X03_c6x_A15pB/suite__omni/seed001/2026-09-25_20-31-29")
meta = json.loads((d/"run_meta.json").read_text()); labels = meta["hold_labels"]
print(meta["checkpoint"]); print(labels)
def yaw(q):
    w,x,y,z=q[...,0],q[...,1],q[...,2],q[...,3]; return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
acc = {}
for e in meta["env_index_map"]:
    lab = e["label"]
    if lab not in ("fwd","back","lat_l","lat_r") and not lab.startswith(("fwd","back","lat_")): continue
    z = np.load(d/"raw"/f"episode_{e['env_index']:02d}.npz")
    done = z["done"].astype(bool); S = int(np.argmax(done)) if done.any() else len(done)
    sid = labels.index(lab); seg = z["segment_id"]; idx = np.where((seg==sid)&(np.arange(len(seg))<S))[0]
    if len(idx) < 400: continue
    t0 = idx[0]; a4, hi = t0+200, min(t0+500, idx[-1])
    A = np.clip(z["actions"][a4:hi].astype(float), -1, 1)
    rp = z["root_pos_w"][:, :2].astype(float); fc = z["foot_pos_w"][:, :, :2].astype(float).mean(1)
    u = rp[idx[-1]] - rp[t0]; n = np.linalg.norm(u)
    if n < 0.05: continue
    u /= n; v = (fc[hi]-fc[a4]) @ u / ((hi-a4)*0.02)
    q = z["joint_pos"][a4:hi].astype(float)
    acc.setdefault(lab, []).append((A.mean(0), A.std(0), v, q.std(0)))
jn = meta["joint_names"]
yaw_ids = [i for i,n in enumerate(jn) if "_Body_Hip_" in n]; pit = [i for i,n in enumerate(jn) if "Hip_Femur" in n]; kn = [i for i,n in enumerate(jn) if "Femur_Tibia" in n]
for lab, rows in acc.items():
    am = np.mean([r[0] for r in rows],0); asd = np.mean([r[1] for r in rows],0); v = np.median([r[2] for r in rows]); qs = np.mean([r[3] for r in rows],0)
    print(f"{lab:12s} n{len(rows):3d} feet_ss {v:.3f} | cam mean {np.round(am[:6],2)} | yaw-joint std {np.mean(qs[yaw_ids]):.3f} pitch std {np.mean(qs[pit]):.3f} knee std {np.mean(qs[kn]):.3f}")
