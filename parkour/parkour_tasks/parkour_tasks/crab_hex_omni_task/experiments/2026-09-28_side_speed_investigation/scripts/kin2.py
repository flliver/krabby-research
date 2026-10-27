import sys, math, json
import numpy as np, torch
sys.path.insert(0, "/tmp/claude-1000/sidespeed")
from kin import *  # noqa
L1, L2 = L.FEMUR_LEN_M, L.TIBIA_M if hasattr(L, "TIBIA_M") else 0.8255

def rowstats(tag, want):
    acc = {"lead": {}, "trail": {}}
    def add(row, k, v):
        acc[row].setdefault(k, []).append(v)
    for label, z, meta in episodes(tag, want):
        m = own_mask(z, meta, label)
        if m.sum() < 50: continue
        dt = 0.02
        psi = heading(z["root_quat_w"].astype(np.float64)); c, s = np.cos(psi), np.sin(psi)
        cmd = z["cmd_applied"][:, :2].astype(np.float64)
        if not meta.get("suite"):
            ty = meta["travel_yaw"]
            cmd = np.stack([cmd[:, 0]*math.cos(ty)-cmd[:, 1]*math.sin(ty), cmd[:, 0]*math.sin(ty)+cmd[:, 1]*math.cos(ty)], -1)
        u = cmd / np.maximum(np.linalg.norm(cmd, axis=-1, keepdims=True), 1e-9)
        side_sign = np.sign(np.median(u[m, 1]))  # +1 moving left
        lead_leg = LEFT if side_sign > 0 else ~LEFT
        fp = z["foot_pos_w"].astype(np.float64); rp = z["root_pos_w"].astype(np.float64)
        dx = fp[..., 0]-rp[:, None, 0]; dy = fp[..., 1]-rp[:, None, 1]
        fy = -s[:, None]*dx + c[:, None]*dy
        F = z["foot_force_norm"].astype(np.float64); loaded = F > F_LOAD
        fsp = np.linalg.norm(z["foot_lin_vel_w"][..., :2].astype(np.float64), axis=-1)
        jp = z["joint_pos"].astype(np.float64); tq = z["applied_torque"].astype(np.float64)
        ih = jidx(meta, "Hip_Femur_RevoluteJoint"); ik = jidx(meta, "Femur_Tibia_RevoluteJoint")
        ksign = np.where(LEFT, 1.0, -1.0)
        th = jp[:, ih]; tk = jp[:, ik]*ksign
        phi = th + tk
        utoe = L1*np.cos(th) - L2*np.sin(phi); dtoe = L1*np.sin(th) + L2*np.cos(phi)
        lh = L.hip_actuator_length(torch.tensor(th)).numpy(); lk = L.knee_actuator_length(torch.tensor(th), torch.tensor(tk)).numpy()
        rh = np.zeros_like(lh); rk = np.zeros_like(lk); rh[1:] = np.diff(lh, axis=0)/dt; rk[1:] = np.diff(lk, axis=0)/dt
        a = z["actions"].astype(np.float64); ak = a[:, 12:18]*ksign; ah = a[:, 6:12]
        idx = np.where(m)[0]; lo, hi = idx.min(), idx.max(); dur = (hi-lo+1)*dt
        # lateral velocity of foot relative to body in body frame, "outward" sign
        out_sign = np.where(LEFT, 1.0, -1.0)
        for f in range(6):
            row = "lead" if lead_leg[f] else "trail"
            mf = m
            add(row, "u_toe_p50", float(np.median(utoe[mf, f])))
            add(row, "u_toe_p5", float(np.percentile(utoe[mf, f], 5)))
            add(row, "u_toe_p95", float(np.percentile(utoe[mf, f], 95)))
            add(row, "d_toe_p50", float(np.median(dtoe[mf, f])))
            add(row, "absy_p50", float(np.median(np.abs(fy[mf, f]))))
            add(row, "knee_deg_p50", float(np.degrees(np.median(tk[mf, f]))))
            add(row, "hip_deg_p50", float(np.degrees(np.median(th[mf, f]))))
            ld = loaded[mf, f]
            add(row, "loaded_frac", float(ld.mean()))
            add(row, "F_mean_loaded", float(F[mf, f][ld].mean()) if ld.any() else np.nan)
            add(row, "knee_tq_cap_loaded", float((np.abs(tq[mf, ik[f]])[ld] >= 395).mean()) if ld.any() else np.nan)
            add(row, "knee_rod_spd_stance", float(np.abs(rk[mf, f][ld]).mean()/L.KNEE_SPEED_M_S) if ld.any() else np.nan)
            add(row, "knee_rod_spd_swing", float(np.abs(rk[mf, f][~ld]).mean()/L.KNEE_SPEED_M_S) if (~ld).any() else np.nan)
            add(row, "hip_rod_spd_swing", float(np.abs(rh[mf, f][~ld]).mean()/L.HIP_SPEED_M_S) if (~ld).any() else np.nan)
            add(row, "knee_act_pos_sat", float((ak[mf, f] >= 4.7).mean()))
            add(row, "knee_act_neg_sat", float((ak[mf, f] <= -4.7).mean()))
            add(row, "hip_act_pos_sat", float((ah[mf, f] >= 4.7).mean()))
            # stances & taps
            ldw = loaded[lo:hi+1, f]
            d = np.diff(np.concatenate([[0], ldw.astype(int), [0]])); st = np.where(d == 1)[0]; en = np.where(d == -1)[0]
            ntap = 0; nst = 0; sweeps = []; slips = []; kc = []; durs = []
            for a0, a1 in zip(st, en):
                if a0 == 0 or a1 == len(ldw): continue
                t0, t1 = lo+a0, lo+a1
                if a1-a0 < 5: ntap += 1; continue
                nst += 1; durs.append((a1-a0)*dt)
                # body-frame lateral foot sweep; stance should move foot opposite to travel: -side_sign
                sw = (fy[t1-1, f]-fy[t0, f]) * (-side_sign)
                sweeps.append(sw)
                slips.append(float(fsp[t0:t1, f].mean()))
                # u change: lead row u decreases (pull), trail row u increases (push)
                kc.append(float(utoe[t1-1, f]-utoe[t0, f]))
            add(row, "taps_per_s", ntap/dur); add(row, "stances_per_s", nst/dur)
            if sweeps:
                add(row, "stance_sweep", float(np.median(sweeps))); add(row, "stance_slipspd", float(np.median(slips)))
                add(row, "stance_du", float(np.median(kc))); add(row, "stance_dur", float(np.median(durs)))
    return {row: {k: float(np.nanmedian(v)) for k, v in d.items()} for row, d in acc.items()}

res = {}
for tag, want in [("W", ["lat_l", "lat_r"]), ("t3", ["lat_l", "lat_r"]), ("SH", ["side"])]:
    r = rowstats(tag, want)
    for row in r: res[f"{tag}:{row}"] = r[row]
json.dump(res, open("/tmp/claude-1000/sidespeed/kin2.json", "w"), indent=1)
cols = list(res); keys = list(res[cols[0]])
print(f"{'metric':22s}" + "".join(f"{c:>12s}" for c in cols))
for k in keys: print(f"{k:22s}" + "".join(f"{res[c].get(k, np.nan):12.4f}" for c in cols))
