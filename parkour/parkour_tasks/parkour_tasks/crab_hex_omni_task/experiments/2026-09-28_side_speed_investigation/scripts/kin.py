#!/usr/bin/env python3
"""Kinematics / slip / actuation readout for omni side holds vs sideways head. CPU only, read-only."""
import json, sys, math
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/mdp")
import crab_hex_linkage as L  # noqa

LOG = Path("/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval")
STAMPS = {
    "W": LOG / "omni_phases/2026-09-23_clock_deprecation_w2/seed3_X03_c6x_A15pB/suite__omni/seed001/2026-09-25_20-31-29",
    "t3": LOG / "omni_phases/2026-09-23_clock_deprecation/step0_E3_X08-t3_visible/suite__omni/seed001/2026-09-23_21-38-11",
    "SH": LOG / "side_phases/2026-09-11_0414_sideways_formation/seed2_2a_006_A15pB/slow__side/seed001/2026-09-12_09-18-39",
}
LEGS = ["FL", "FR", "ML", "MR", "RL", "RR"]
LEFT = np.array([1, 0, 1, 0, 1, 0], bool)
F_LOAD = 50.0


def heading(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def episodes(tag, want):
    """yield (label, npz dict, own mask) for episodes whose own hold label is in want."""
    d = STAMPS[tag]
    meta = json.loads((d / "run_meta.json").read_text())
    labels = meta["hold_labels"]
    if meta.get("suite"):
        for ent in meta["env_index_map"]:
            if ent["label"] not in want:
                continue
            p = d / "raw" / f"episode_{ent['env_index']:02d}.npz"
            z = dict(np.load(p))
            hid = labels.index(ent["label"])
            yield ent["label"], z, meta
    else:
        for p in sorted((d / "raw").glob("episode_*.npz")):
            z = dict(np.load(p))
            yield "side", z, meta


def own_mask(z, meta, label):
    labels = meta["hold_labels"]
    seg = z["segment_id"].astype(int)
    done = z["done"].astype(bool)
    S = int(np.argmax(done)) if done.any() else len(seg) - 1
    alive = np.arange(len(seg)) < S
    if meta.get("suite"):
        own = seg == labels.index(label)
    else:
        own = np.isin(seg, [labels.index("creep"), labels.index("low")])
    return own & z["steady_mask"].astype(bool) & alive


def jidx(meta, suffix):
    names = meta["joint_names"]
    return [names.index(f"{l}_{suffix}") for l in LEGS]


def analyse(tag, want):
    rows = []
    for label, z, meta in episodes(tag, want):
        m = own_mask(z, meta, label)
        if m.sum() < 50:
            continue
        T = len(m)
        dt = 0.02
        psi = heading(z["root_quat_w"].astype(np.float64))
        c, s = np.cos(psi), np.sin(psi)
        cmd = z["cmd_applied"][:, :2].astype(np.float64)
        if not meta.get("suite"):
            ty = meta["travel_yaw"]
            cmd = np.stack([cmd[:, 0] * math.cos(ty) - cmd[:, 1] * math.sin(ty),
                            cmd[:, 0] * math.sin(ty) + cmd[:, 1] * math.cos(ty)], -1)
        vb = z["root_lin_vel_b"][:, :2].astype(np.float64)
        cn = np.linalg.norm(cmd, axis=-1, keepdims=True)
        u = cmd / np.maximum(cn, 1e-9)
        # speed along command (body frame, both); use next-row state
        vs = np.zeros_like(vb); vs[:-1] = vb[1:]
        along = (vs * u).sum(-1)
        # feet in heading frame relative to root
        fp = z["foot_pos_w"].astype(np.float64)
        rp = z["root_pos_w"].astype(np.float64)
        dx = fp[..., 0] - rp[:, None, 0]; dy = fp[..., 1] - rp[:, None, 1]
        fx = c[:, None] * dx + s[:, None] * dy
        fy = -s[:, None] * dx + c[:, None] * dy
        fz = fp[..., 2] - z["terrain_z"].astype(np.float64)
        F = z["foot_force_norm"].astype(np.float64)
        loaded = F > F_LOAD
        fv = z["foot_lin_vel_w"][..., :2].astype(np.float64)
        fsp = np.linalg.norm(fv, axis=-1)
        # joint states
        jp = z["joint_pos"].astype(np.float64); jv = z["joint_vel"].astype(np.float64)
        tq = z["applied_torque"].astype(np.float64)
        ih = jidx(meta, "Hip_Femur_RevoluteJoint"); ik = jidx(meta, "Femur_Tibia_RevoluteJoint")
        iy = jidx(meta, "Body_Hip_RevoluteJoint"); ic = jidx(meta, "Body_CamShaft_RevoluteJoint")
        ksign = np.where(LEFT, 1.0, -1.0)
        th = jp[:, ih]; tk = jp[:, ik] * ksign
        lh = L.hip_actuator_length(torch.tensor(th)).numpy()
        lk = L.knee_actuator_length(torch.tensor(th), torch.tensor(tk)).numpy()
        rh = np.zeros_like(lh); rk = np.zeros_like(lk)
        rh[1:] = np.diff(lh, axis=0) / dt; rk[1:] = np.diff(lk, axis=0) / dt
        a = z["actions"].astype(np.float64)
        # action cols: cam 0-5, hip 6-11, knee 12-17
        ac, ah, ak = a[:, 0:6], a[:, 6:12], a[:, 12:18]
        # leading row: legs on the side of travel (dot of foot lateral with cmd)
        # command dir in body frame (heading frame == body frame modulo roll/pitch)
        uy = u[:, 1]
        lead = (np.sign(fy) == np.sign(uy)[:, None])  # foot on travel side
        # contact runs per foot within own window
        r = dict(label=label, n=int(m.sum()))
        r["speed"] = float(along[m].mean())
        r["cmd"] = float(cn[m].mean())
        # --- actions
        r["act_cam_sat"] = float((np.abs(ac[m]) >= 0.98).mean())
        r["act_hip_sat"] = float((np.abs(ah[m]) >= 4.7).mean())
        r["act_knee_sat"] = float((np.abs(ak[m]) >= 4.7).mean())
        r["act_hip_abs_p50"] = float(np.median(np.abs(ah[m])))
        r["act_knee_abs_p50"] = float(np.median(np.abs(ak[m])))
        r["act_hip_abs_gt2"] = float((np.abs(ah[m]) > 2.0).mean())
        r["act_knee_abs_gt2"] = float((np.abs(ak[m]) > 2.0).mean())
        r["cam_act_mean"] = float(ac[m].mean())
        r["cam_act_absmean"] = float(np.abs(ac[m]).mean())
        camv = jv[:, ic]
        r["cam_vel_mean"] = float(camv[m].mean())
        r["cam_rev_hz"] = float(np.abs(camv[m]).mean() / (2 * math.pi))
        # --- rod speeds (joint-pos derived), fraction at cap
        r["hip_rod_cap_frac"] = float((np.abs(rh[m]) >= 0.9 * L.HIP_SPEED_M_S).mean())
        r["knee_rod_cap_frac"] = float((np.abs(rk[m]) >= 0.9 * L.KNEE_SPEED_M_S).mean())
        r["hip_rod_absmean_frac"] = float(np.abs(rh[m]).mean() / L.HIP_SPEED_M_S)
        r["knee_rod_absmean_frac"] = float(np.abs(rk[m]).mean() / L.KNEE_SPEED_M_S)
        # rod travel windows
        r["hip_rod_at_win"] = float(((lh[m] < L.HIP_LEN_MIN_M + 0.003) | (lh[m] > L.HIP_LEN_MAX_M - 0.003)).mean())
        r["knee_rod_at_win"] = float(((lk[m] < L.KNEE_LEN_MIN_M + 0.003) | (lk[m] > L.KNEE_LEN_MAX_M - 0.003)).mean())
        r["hip_rod_p5"], r["hip_rod_p95"] = map(float, np.percentile(lh[m], [5, 95]))
        r["knee_rod_p5"], r["knee_rod_p95"] = map(float, np.percentile(lk[m], [5, 95]))
        # per-leg rod stroke used (p95-p5)
        r["hip_rod_range_perleg"] = float(np.mean(np.percentile(lh[m], 95, 0) - np.percentile(lh[m], 5, 0)))
        r["knee_rod_range_perleg"] = float(np.mean(np.percentile(lk[m], 95, 0) - np.percentile(lk[m], 5, 0)))
        # recorded rod speed if present (validation)
        if "rod_speed" in z:
            rs = z["rod_speed"].astype(np.float64)
            r["rec_hip_rod_cap"] = float((np.abs(rs[m, :6]) >= 0.9 * L.HIP_SPEED_M_S).mean())
            r["rec_knee_rod_cap"] = float((np.abs(rs[m, 6:]) >= 0.9 * L.KNEE_SPEED_M_S).mean())
        # joint angles
        r["hip_deg_p5"], r["hip_deg_p95"] = map(float, np.degrees(np.percentile(th[m], [5, 95])))
        r["knee_deg_p5"], r["knee_deg_p95"] = map(float, np.degrees(np.percentile(tk[m], [5, 95])))
        yaw = jp[:, iy]
        r["yaw_deg_absp95"] = float(np.degrees(np.percentile(np.abs(yaw[m]), 95)))
        r["yaw_range_perleg_deg"] = float(np.degrees(np.mean(np.percentile(yaw[m], 95, 0) - np.percentile(yaw[m], 5, 0))))
        # torques
        r["knee_tq_cap"] = float((np.abs(tq[m][:, ik]) >= 395).mean())
        r["hip_tq_cap"] = float((np.abs(tq[m][:, ih]) >= 990).mean())
        r["cam_tq_cap"] = float((np.abs(tq[m][:, ic]) >= 19.8).mean())
        r["knee_tq_p95"] = float(np.percentile(np.abs(tq[m][:, ik]), 95))
        r["hip_tq_p95"] = float(np.percentile(np.abs(tq[m][:, ih]), 95))
        # joint vel vs sim caps (hip 2.0, knee 3.0)
        r["hip_jv_p95"] = float(np.percentile(np.abs(jv[m][:, ih]), 95))
        r["knee_jv_p95"] = float(np.percentile(np.abs(jv[m][:, ik]), 95))
        # foot geometry
        mm = m[:, None] & np.ones((1, 6), bool)
        r["foot_absy_mean"] = float(np.abs(fy[m]).mean())
        r["foot_absy_lead"] = float(np.abs(fy)[mm & lead].mean())
        r["foot_absy_trail"] = float(np.abs(fy)[mm & ~lead].mean())
        r["foot_absx_mean"] = float(np.abs(fx[m]).mean())
        r["body_h"] = float((rp[m, 2] - z["terrain_z"][m].mean(-1)).mean())
        # per foot lateral range (p95-p5) in own window
        r["foot_y_range"] = float(np.mean(np.percentile(fy[m], 95, 0) - np.percentile(fy[m], 5, 0)))
        r["foot_x_range"] = float(np.mean(np.percentile(fx[m], 95, 0) - np.percentile(fx[m], 5, 0)))
        # contacts / stances
        st_along = []; st_len = []; st_slip = []; st_body = []; taps = 0; steps = 0; swings_h = []
        slip_lead = []; slip_trail = []; slip_phase = [[], [], []]
        stance_lead_n = 0; stance_trail_n = 0; tap_lead = 0; tap_trail = 0
        duty = loaded[m].mean()
        uw = np.stack([c * u[:, 0] - s * u[:, 1], s * u[:, 0] + c * u[:, 1]], -1)  # world cmd dir
        idx = np.where(m)[0]
        lo, hi = idx.min(), idx.max()
        for f in range(6):
            ld = loaded[lo:hi + 1, f]
            # runs
            d = np.diff(np.concatenate([[0], ld.astype(int), [0]]))
            starts = np.where(d == 1)[0]; ends = np.where(d == -1)[0]
            for a0, a1 in zip(starts, ends):
                t0, t1 = lo + a0, lo + a1  # [t0, t1)
                if a0 == 0 or a1 == len(ld):
                    continue
                n = t1 - t0
                isl = bool(lead[t0:t1, f].mean() > 0.5)
                if n < 5:
                    taps += 1
                    tap_lead += isl; tap_trail += (not isl)
                    continue
                steps += 1
                stance_lead_n += isl; stance_trail_n += (not isl)
                # foot world disp along cmd during stance vs body disp
                fw = fp[t0:t1, f, :2]
                dfw = fw[-1] - fw[0]
                db = rp[t1 - 1, :2] - rp[t0, :2]
                ud = uw[t0]
                foot_along = float(dfw @ ud); body_along = float(db @ ud)
                st_slip.append(np.linalg.norm(dfw)); st_body.append(body_along)
                # relative (body frame) foot sweep along cmd = body - foot
                st_along.append(body_along - foot_along)
                st_len.append(n)
                sp = fsp[t0:t1, f]
                (slip_lead if isl else slip_trail).append(float(sp.mean()))
                thirds = np.array_split(sp, 3)
                for k in range(3):
                    slip_phase[k].append(float(thirds[k].mean()))
            # swings: unloaded runs >=5
            ul = ~ld
            d = np.diff(np.concatenate([[0], ul.astype(int), [0]]))
            s0 = np.where(d == 1)[0]; s1 = np.where(d == -1)[0]
            for a0, a1 in zip(s0, s1):
                if a0 == 0 or a1 == len(ul) or a1 - a0 < 5:
                    continue
                swings_h.append(float(fz[lo + a0:lo + a1, f].max() - fz[lo + a0, f]))
        dur = (hi - lo + 1) * dt
        r["duty"] = float(duty)
        r["stances_per_s"] = steps / dur
        r["taps_per_s"] = taps / dur
        r["tap_frac_lead"] = tap_lead / max(taps, 1)
        r["stance_frac_lead"] = stance_lead_n / max(steps, 1)
        r["stance_len_s"] = float(np.median(st_len) * dt) if st_len else float("nan")
        r["stance_sweep"] = float(np.median(st_along)) if st_along else float("nan")
        r["stance_foot_slip_m"] = float(np.median(st_slip)) if st_slip else float("nan")
        r["stance_body_adv"] = float(np.median(st_body)) if st_body else float("nan")
        r["slip_speed_lead"] = float(np.median(slip_lead)) if slip_lead else float("nan")
        r["slip_speed_trail"] = float(np.median(slip_trail)) if slip_trail else float("nan")
        for k in range(3):
            r[f"slip_speed_ph{k}"] = float(np.median(slip_phase[k])) if slip_phase[k] else float("nan")
        r["swing_apex"] = float(np.median(swings_h)) if swings_h else float("nan")
        # load share lead vs trail
        Fm = F[m]; lm = lead[m]
        r["load_share_lead"] = float(Fm[lm].sum() / Fm.sum())
        rows.append(r)
    return rows


def summarize(rows, keys=None):
    out = {}
    keys = keys or [k for k in rows[0] if k not in ("label",)]
    for k in keys:
        v = np.array([r[k] for r in rows], float)
        out[k] = float(np.nanmedian(v))
    out["n_eps"] = len(rows)
    return out


if __name__ == "__main__":
    res = {}
    for tag, want in [("W", ["lat_l", "lat_r"]), ("W", ["fwd"]), ("W", ["back"]), ("t3", ["lat_l", "lat_r"]), ("t3", ["fwd"]), ("SH", ["side"])]:
        rows = analyse(tag, want)
        key = f"{tag}:{'/'.join(want)}"
        res[key] = summarize(rows)
        # also lat_l only for omni
        if "lat_l" in want:
            res[f"{tag}:lat_l"] = summarize([r for r in rows if r["label"] == "lat_l"])
        print(key, "done", len(rows), file=sys.stderr)
    json.dump(res, open("/tmp/claude-1000/sidespeed/kin.json", "w"), indent=1)
    keys = list(next(iter(res.values())).keys())
    cols = list(res.keys())
    print(f"{'metric':24s}" + "".join(f"{c[:14]:>15s}" for c in cols))
    for k in keys:
        print(f"{k:24s}" + "".join(f"{res[c].get(k, float('nan')):15.4f}" for c in cols))
