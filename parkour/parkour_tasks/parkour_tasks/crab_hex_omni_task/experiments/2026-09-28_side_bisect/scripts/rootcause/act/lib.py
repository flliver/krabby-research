import json, ast, glob
from pathlib import Path
import numpy as np
ROOT = Path("/home/nickmagus/krabby/krabby-research/parkour")
OUT = ROOT/"outputs/omni_plan/side_bisect"
ARMLOG = ROOT/"logs/rsl_rl/gait_eval/omni_phases/2026-09-28_side_bisect"
TAG = {"O":"seed3_X01","C":"seed3_X02","R":"seed3_X03","C_mo":"seed3_X04","C_band":"seed3_X05","C_rsi":"seed3_X06",
       "C_s2":"seed2_X07","C_mo_s2":"seed2_X08","C_mo_s1":"seed1_X09"}
W_SUITE = ROOT/"logs/rsl_rl/gait_eval/omni_phases/2026-09-23_clock_deprecation_w2/seed3_X03_c6x_A15pB/suite__omni/seed001/2026-09-25_20-31-29"
LEGS = ["FL","FR","ML","MR","RL","RR"]
SIDE = np.array([1,-1,1,-1,1,-1])  # +1 left (+y side), -1 right
def stamp(key):
    if key == "W": return W_SUITE
    if key.startswith("canon:"):
        h = key.split(":")[1]
        return Path(sorted(glob.glob(str(OUT/"canon"/h/"wave/side22_25lr__omni/seed001/*/")))[-1])
    if key.startswith("lr:"):
        a = key.split(":")[1]
        return Path(sorted(glob.glob(str(OUT/"lr"/a/"side22_25lr__omni/seed001/*/")))[-1])
    return Path(sorted(glob.glob(str(ARMLOG/(TAG[key]+"_s1_A15pB")/"side22_25__omni/seed001/*/")))[-1])
def lit(v): return ast.literal_eval(v) if isinstance(v,str) else v
def quat_rot(q):  # wxyz -> R (N,3,3) body->world
    w,x,y,z = q[:,0],q[:,1],q[:,2],q[:,3]
    R = np.stack([np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],-1),
                  np.stack([2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],-1),
                  np.stack([2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1)],-2)
    return R
def episodes(key, labels=None, lo=200, hi=500, full=False):
    d = stamp(key); m = json.loads((d/"run_meta.json").read_text())
    hl = m["hold_labels"]; em = lit(m["env_index_map"]); bn = lit(m["body_names"]); jn = lit(m["joint_names"])
    hip_ids = [bn.index(l+"_Hip") for l in LEGS]
    out = []
    for e in em:
        lab = e["label"]
        if labels and lab not in labels: continue
        z = np.load(d/"raw"/f"episode_{e['env_index']:02d}.npz")
        done = z["done"].astype(bool); S = int(np.argmax(done)) if done.any() else len(done)
        seg = z["segment_id"]; sid = hl.index(lab)
        idx = np.where((seg == sid) & (np.arange(len(seg)) < S))[0]
        if len(idx) < hi: continue
        t0 = idx[0]; sl = slice(t0+lo, t0+hi) if not full else slice(t0, idx[-1]+1)
        R = quat_rot(z["root_quat_w"][sl].astype(float)); p = z["root_pos_w"][sl].astype(float)
        Rt = np.transpose(R,(0,2,1))
        fw = z["foot_pos_w"][sl].astype(float); fv = z["foot_lin_vel_w"][sl].astype(float)
        hw = z["body_pos_w"][sl][:,hip_ids].astype(float)
        fb = np.einsum('tij,tkj->tki', Rt, fw - p[:,None])  # foot in body frame
        hb = np.einsum('tij,tkj->tki', Rt, hw - p[:,None])
        vb_w = np.einsum('tij,tj->ti', R, z["root_lin_vel_b"][sl].astype(float))
        out.append(dict(label=lab, t0=t0, R=R, p=p, fw=fw, fv=fv, fb=fb, hb=hb, F=z["foot_force_norm"][sl].astype(float),
                        a=z["actions"][sl].astype(float), q=z["joint_pos"][sl].astype(float), qd=z["joint_vel"][sl].astype(float),
                        vb=z["root_lin_vel_b"][sl].astype(float), vw=vb_w, wb=z["root_ang_vel_b"][sl].astype(float),
                        tz=z["terrain_z"][sl].astype(float), rod=z["rod_speed"][sl].astype(float),
                        clock=z["clock_phase"][sl].astype(float), cmd=z["cmd_applied"][sl].astype(float)))
    return m, jn, lit(m["action_joint_names"]), out
