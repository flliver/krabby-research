"""Extract hold-window [4,10) s signals for every side-bisect stamp into a compact cache (CPU, read-only).

Per (stamp, hold label), per valid episode (classifier hold_stats != None AND alive through t0+500):
  act   (300,18) clipped actions  (action_joint_names order)
  jp    (300,18) joint pos: Body_Hip x6, Hip_Femur x6, Femur_Tibia x6 (leg order FL FR ML MR RL RR)
  camv  (300,6)  cam-shaft joint velocity (signed shaft speed, leg order)
  foot  (300,18) foot pos in yaw-body frame rel. root: x x6, y x6, z x6
  body  (300,4)  lean_x_b, lean_y_b (root - feet centroid, yaw-body frame), roll, pitch
  contact (300,6) foot force > 50 N
  ph    (300,)   clock_phase (nan if not logged)
  jpx   (450,18) joint pos over [t0+100, t0+550) (clipped at death) for the intrinsic phase
  stats: feet_ss, sweep, lean_abs (classifier hold_stats verbatim)
"""
import json, sys, importlib.util
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor

spec = importlib.util.spec_from_file_location("sc", "/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect/scripts/side22_classify.py")
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)

OUT = Path("/tmp/claude-1000/-home-nickmagus-krabby-krabby-research/4b0248a2-13ef-464e-81a9-16e93543e283/scratchpad/sym/cache")
OUT.mkdir(exist_ok=True)
LEGS = ["FL", "FR", "ML", "MR", "RL", "RR"]
B = "/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-28_side_bisect"
S = "/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/side_bisect"
# name, group, path
STAMPS = [
    ("O", "phantom", f"{B}/seed3_X01_s1_A15pB/side22_25__omni/seed001/2026-09-29_05-50-19"),
    ("R", "phantom", f"{B}/seed3_X03_s1_A15pB/side22_25__omni/seed001/2026-09-29_13-36-31"),
    ("C_band", "phantom", f"{B}/seed3_X05_s1_A15pB/side22_25__omni/seed001/2026-09-29_16-04-24"),
    ("C_rsi", "phantom", f"{B}/seed3_X06_s1_A15pB/side22_25__omni/seed001/2026-09-29_17-19-39"),
    ("C", "oneway", f"{B}/seed3_X02_s1_A15pB/side22_25__omni/seed001/2026-09-29_07-00-57"),
    ("C_s2", "oneway", f"{B}/seed2_X07_s1_A15pB/side22_25__omni/seed001/2026-09-29_18-33-09"),
    ("C_mo", "oneway", f"{B}/seed3_X04_s1_A15pB/side22_25__omni/seed001/2026-09-29_14-49-24"),
    ("C_mo_s2", "oneway", f"{B}/seed2_X08_s1_A15pB/side22_25__omni/seed001/2026-09-29_19-46-10"),
    ("C_mo_s1", "oneway", f"{B}/seed1_X09_s1_A15pB/side22_25__omni/seed001/2026-09-29_23-15-45"),
    ("LRfa", "twoway", f"{S}/lr/LRfa/side22_25lr__omni/seed001/2026-09-30_00-29-51"),
    ("LR4", "twoway", f"{S}/lr/LR4/side22_25lr__omni/seed001/2026-09-30_01-44-42"),
    ("LR4_dwell", "twoway", f"{S}/lr/LR4_dwell/side22_25lr__omni/seed001/2026-09-30_03-26-29"),
    ("C_lr", "twoway", f"{S}/lr/C_lr/side22_25lr__omni/seed001/2026-09-30_05-11-37"),
    ("C_lr_10", "twoway", f"{S}/lr/C_lr_10/side22_25lr__omni/seed001/2026-09-30_06-26-32"),
    ("C_lr_rsi", "twoway", f"{S}/lr/C_lr_rsi/side22_25lr__omni/seed001/2026-09-30_14-18-42"),
    ("C_lr_10_rsi", "twoway", f"{S}/lr/C_lr_10_rsi/side22_25lr__omni/seed001/2026-09-30_15-46-33"),
    ("canon_C", "canon", f"{S}/canon/C/wave/side22_25lr__omni/seed001/2026-09-30_12-27-21"),
    ("canon_C_s2", "canon", f"{S}/canon/C_s2/wave/side22_25lr__omni/seed001/2026-09-30_12-37-35"),
    ("canon_C_mo_s2", "canon", f"{S}/canon/C_mo_s2/wave/side22_25lr__omni/seed001/2026-09-30_12-16-56"),
    ("canon_C_mo_s1", "canon", f"{S}/canon/C_mo_s1/wave/side22_25lr__omni/seed001/2026-09-30_12-47-53"),
    ("W", "omni_head", f"{S}/harness_W/side22_25__omni/seed001/2026-09-29_04-23-49"),
    ("side_s2_3000", "sideways", f"{S}/pin_side/s2_model_3000/side22_25__side/seed001/2026-09-29_04-29-19"),
    ("side_s2_4999", "sideways", f"{S}/pin_side/s2_model_4999/side22_25__side/seed001/2026-09-29_04-30-45"),
    ("side_s3_3000", "sideways", f"{S}/pin_side/s3_model_3000/side22_25__side/seed001/2026-09-29_04-26-13"),
    ("side_s3_4999", "sideways", f"{S}/pin_side/s3_model_4999/side22_25__side/seed001/2026-09-29_04-27-52"),
]


def quat_rp(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return roll, pitch


def one(item):
    name, group, d = item
    d = Path(d)
    meta = json.loads((d / "run_meta.json").read_text())
    labels = meta["hold_labels"]
    want = [l for l in labels if l.startswith("lat_")]
    an = meta["action_joint_names"]; jn = meta["joint_names"]
    jidx = {n: i for i, n in enumerate(jn)}
    jsel = [jidx[f"{L}_{t}_RevoluteJoint"] for t in ("Body_Hip", "Hip_Femur", "Femur_Tibia") for L in LEGS]
    csel = [jidx[f"{L}_Body_CamShaft_RevoluteJoint"] for L in LEGS]
    fb = meta["foot_body_names"]; assert [f[:2] for f in fb] == LEGS, fb
    if meta.get("suite"):
        files = [(d / "raw" / f"episode_{e['env_index']:02d}.npz", e["label"]) for e in meta["env_index_map"] if e["label"] in want]
    else:
        files = [(p, None) for p in sorted((d / "raw").glob("episode_*.npz"))]
    rec = {l: {k: [] for k in ("act", "jp", "camv", "foot", "body", "contact", "ph", "jpx", "feet_ss", "sweep", "lean_abs", "canon", "ep")} for l in want}
    nexcl = {l: 0 for l in want}
    for p, lab in files:
        z = np.load(p)
        done = z["done"].astype(bool)
        Sd = int(np.argmax(done)) if done.any() else len(z["segment_id"])
        seg = z["segment_id"]
        for l in ([lab] if lab else want):
            sid = labels.index(l)
            idx = np.where((seg == sid) & (np.arange(len(seg)) < Sd))[0]
            if len(idx) == 0:
                continue
            hs = sc.hold_stats(z, sid, Sd)
            t0 = idx[0]
            if hs is None or idx[-1] < t0 + 500:
                nexcl[l] += 1
                continue
            a4, hi = t0 + 200, t0 + 500
            sl = slice(a4, hi)
            r = rec[l]
            r["act"].append(np.clip(z["actions"][sl].astype(np.float32), -1, 1))
            r["jp"].append(z["joint_pos"][sl][:, jsel].astype(np.float32))
            r["camv"].append(z["joint_vel"][sl][:, csel].astype(np.float32))
            rp = z["root_pos_w"].astype(float); q = z["root_quat_w"].astype(float)
            psi = sc.yaw(q)[sl]; c, s = np.cos(psi), np.sin(psi)
            fp = z["foot_pos_w"][sl].astype(float) - rp[sl][:, None, :]
            fx = c[:, None] * fp[..., 0] + s[:, None] * fp[..., 1]
            fy = -s[:, None] * fp[..., 0] + c[:, None] * fp[..., 1]
            fz = fp[..., 2]
            r["foot"].append(np.concatenate([fx, fy, fz], 1).astype(np.float32))
            lx, ly = -fx.mean(1), -fy.mean(1)   # root minus feet centroid, body frame
            roll, pitch = quat_rp(q[sl])
            r["body"].append(np.stack([lx, ly, roll, pitch], 1).astype(np.float32))
            r["contact"].append((z["foot_force_norm"][sl] > 50).astype(np.int8))
            r["ph"].append(z["clock_phase"][sl].astype(np.float32) if "clock_phase" in z.files else np.full(300, np.nan, np.float32))
            end = min(t0 + 550, idx[-1] + 1, Sd)
            jpx = np.full((450, 18), np.nan, np.float32)
            jpx[: end - (t0 + 100)] = z["joint_pos"][t0 + 100:end][:, jsel]
            r["jpx"].append(jpx)
            r["feet_ss"].append(hs["feet_ss"]); r["sweep"].append(hs["sweep"]); r["lean_abs"].append(hs["lean_abs"])
            r["canon"].append(float(z["canon_flag"][sl].mean()) if "canon_flag" in z.files else np.nan)
            r["ep"].append(str(p.name))
    for l, r in rec.items():
        if not r["act"]:
            continue
        np.savez_compressed(OUT / f"{name}__{l}.npz", **{k: np.array(v) for k, v in r.items()})
    return name, {l: (len(rec[l]["act"]), nexcl[l]) for l in want}, an, jn


if __name__ == "__main__":
    items = STAMPS if len(sys.argv) == 1 else [s for s in STAMPS if s[0] in sys.argv[1:]]
    with ProcessPoolExecutor(8) as ex:
        for name, counts, an, jn in ex.map(one, items):
            print(name, counts, flush=True)
    json.dump({"stamps": STAMPS, "action_joint_names": an, "joint_names": jn}, open(OUT / "registry.json", "w"), indent=1)
