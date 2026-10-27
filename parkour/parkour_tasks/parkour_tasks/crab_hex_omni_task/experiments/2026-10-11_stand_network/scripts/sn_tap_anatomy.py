"""What are the post-settle 'taps' / 'steps' physically? Uses p2s_stand's own settle + step_tap_runs definitions."""
import glob, json, sys
from pathlib import Path
import numpy as np

R = str(Path(__file__).resolve().parents[7])  # this checkout's root
sys.path.insert(0, f"{R}/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-07_omni_phase2/scripts")
sys.dont_write_bytecode = True
import p2s_stand as S  # noqa: E402

M = S.M
LR = Path(f"{R}/parkour/outputs/omni_plan/phase2/lr")


def newest(name, row, es):
    ds = sorted(glob.glob(str(LR / name / row / f"seed{es:03d}" / "*/")))
    return ds[-1] if ds else None


def anatomy(name, rows=(("walkstop__omni", 1), ("walkstop__omni", 2))):
    ev = []
    n_stands = n_settled = 0
    post_s = 0.0
    for row, es in rows:
        st = newest(name, row, es)
        meta = json.load(open(Path(st) / "run_meta.json"))
        dt = float(meta.get("dt", 0.02))
        cam_cols = S.joint_columns(meta, S.CAM_JOINT_SUFFIX)
        for h in S.iter_stand_holds(st):
            d, hi = h["d"], h["hold_idx"]
            full, _ = S.hold_masks(d, hi, dt)
            idx = np.flatnonzero(full)
            if not idx.size:
                continue
            n_stands += 1
            jp = np.asarray(d["joint_pos"])
            cam_abs = np.abs(S.wrap_to_pi(jp[:, cam_cols])).max(axis=-1)
            need = max(1, int(round(S.Z2_SETTLE_HOLD_S / dt)))
            t0 = S._first_run_start(cam_abs[idx] <= S.Z2_HOMED_RAD, need)
            if t0 is None:
                continue
            n_settled += 1
            force = np.asarray(d["foot_force_norm"])
            loaded = force[idx] > S.TOUCH_REF_N
            post = idx[t0:]
            post_s += post.size * dt
            steps, taps, _ = M.step_tap_runs(loaded[t0:], min_step_steps=S.MIN_STEP_STEPS)
            fz = np.asarray(d["foot_pos_w"])[:, :, 2] - np.asarray(d["terrain_z"])
            act = np.asarray(d["sn_stand_action"]) if "sn_stand_action" in d.files else np.asarray(d["actions"])
            br = np.asarray(d["sn_r_branch"]) if "sn_r_branch" in d.files else None
            for kind, runs in (("tap", taps), ("step", steps)):
                for foot, a, b in runs:
                    ra, rb = post[a], post[min(b, len(post)) - 1]
                    pre = slice(max(post[0], ra - 10), ra)
                    z0 = float(np.median(fz[pre, foot])) if ra > post[0] else float(fz[ra, foot])
                    lift_mm = 1000.0 * (float(fz[ra:rb + 1, foot].max()) - z0)
                    hip_cmd = float(act[ra:rb + 1, 6 + foot].min())       # most-lifting hip command in the event
                    ev.append(dict(kind=kind, foot=int(foot), dur_s=(rb - ra + 1) * dt, lift_mm=lift_mm,
                                   fmin=float(force[ra:rb + 1, foot].min()),
                                   fpre=float(force[pre, foot].mean()) if ra > post[0] else None,
                                   hip_cmd=hip_cmd, t_after_settle=(ra - post[0]) * dt,
                                   r_branch=int(br[ra, foot]) if br is not None else None))
    return ev, n_stands, n_settled, post_s


def summarize(name):
    ev, n, ns, ps = anatomy(name)
    out = {"head": name, "stands": n, "settled": ns, "post_settle_s": round(ps, 1)}
    for kind in ("tap", "step"):
        e = [x for x in ev if x["kind"] == kind]
        if not e:
            out[kind] = {"n": 0}
            continue
        lift = np.array([x["lift_mm"] for x in e]); dur = np.array([x["dur_s"] for x in e])
        hip = np.array([x["hip_cmd"] for x in e]); tas = np.array([x["t_after_settle"] for x in e])
        fmin = np.array([x["fmin"] for x in e]); fpre = np.array([x["fpre"] for x in e if x["fpre"] is not None])
        out[kind] = {
            "n": len(e), "per_s": round(len(e) / ps, 3),
            "lift_mm_median": round(float(np.median(lift)), 2), "lift_mm_p90": round(float(np.percentile(lift, 90)), 2),
            "lift_mm_max": round(float(lift.max()), 1),
            "share_lift_lt_1mm": round(float((lift < 1).mean()), 3), "share_lift_lt_5mm": round(float((lift < 5).mean()), 3),
            "share_lift_gt_20mm": round(float((lift > 20).mean()), 3),
            "dur_s_median": round(float(np.median(dur)), 3),
            "force_min_median_N": round(float(np.median(fmin)), 1), "force_before_median_N": round(float(np.median(fpre)), 1) if fpre.size else None,
            "hip_cmd_median": round(float(np.median(hip)), 3), "share_hip_cmd_lt_-0.25": round(float((hip < -0.25).mean()), 3),
            "t_after_settle_median_s": round(float(np.median(tas)), 2),
            "feet": {i: int(sum(1 for x in e if x["foot"] == i)) for i in range(6)},
            "r_branch_counts": {int(k): int(v) for k, v in zip(*np.unique([x["r_branch"] for x in e if x["r_branch"] is not None], return_counts=True))} if e[0]["r_branch"] is not None else None,
        }
    return out


if __name__ == "__main__":
    res = [summarize(n) for n in sys.argv[1:]]
    print(json.dumps(res, indent=1))
