#!/usr/bin/env python3
"""Stand-height campaign, PLAN step 1 -- facts and premises (CPU only, no Kit, existing raws only).

Writes ``../facts.json`` (every number with its method) and ``../facts.md`` (short human summary):

(a) the posture map q(h) (``sh_kin.q``), h_min by the PLAN H2 rule, the action-unit limits, re-step toe clearance;
(b) measured static joint torques from the walkstop eval stamps (A-mask s17 seeds 1/2, R = P2R_RSTg_s18 seeds 1/2),
    converted to rod forces with the COUPLED linkage (sign conventions verified against finite differences and the
    torch linkage module), compared with the static model at 1.064 m; the scaled model curve F_h(h), F_k(h); h_max;
(c) table-top range for retracted table starts, the K3 eval tops, ground-start geometry (incl. a random-fold family);
(d) the three step-1 STOP flags with their evidence.

Run:  CUDA_VISIBLE_DEVICES="" /home/nickmagus/krabby/isaac_venv/bin/python sh_facts.py
"""
from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
sys.dont_write_bytecode = True   # write nothing but facts.json / facts.md
sys.path.insert(0, str(HERE))
import sh_kin as K  # noqa: E402

REPO = HERE.parents[6]
SN_SCRIPTS = CAMPAIGN.parent / "2026-10-11_stand_network" / "scripts"
LR = REPO / "parkour" / "outputs" / "omni_plan" / "phase2" / "lr"
STAMPS = {
    "SN_Amask_s17": {"es1": "SN_Amask_s17/walkstop__omni/seed001", "es2": "SN_Amask_s17/walkstop__omni/seed002"},
    "R_P2R_RSTg_s18": {"es1": "P2R_RSTg_s18/walkstop__omni/seed001", "es2": "P2R_RSTg_s18/walkstop__omni/seed002"},
}

# ------------------------------------------------------------------ pre-set rules (fixed before reading the raws)
SETTLE_AFTER_S = 2.0          #: a stand row counts once the stand slot has been held this long
FOOT_LOADED_N = 50.0          #: a foot is loaded above this net contact force (the campaign's tap threshold)
MIN_LOADED = 5                #: >= 5 feet loaded
ROD_STILL_M_S = 1e-6          #: all 12 rods not slewing
JOINT_STILL_RAD_S = 0.05      #: all hip-pitch / knee joint speeds below this
FORCE_CAP_N = 2000.0          #: the six-leg mean |F_hip| cap (user, 2026-10-13)
MODEL_TOL = 0.10              #: PLAN step 1: model vs measured joint torques
H_MAX_FLOOR = 1.10            #: PLAN step 1: STOP if h_max < 1.10 m
HIP_ROOM_RAD = 0.125          #: R_h's lift |R lift| x 0.25 = 0.5 x 0.25
FLOOR_MARGIN_M = 0.02         #: h_min term 1 margin above the kinematic floor
TOE_FLOOR_CLEAR_M = 0.02      #: table spawn: every leg part >= this above the floor
BELLY_RESTORE_GAP_M = 0.10    #: the belly at 1.064 must clear the table top by this
TABLE_CMD_MARGIN_TRAIN = 0.264
TABLE_CMD_MARGIN_EVAL = 0.31
SPAWN_CLEARANCE_M = 0.005     #: sn_table_core.SPAWN_CLEARANCE_M (same drop rule on the floor)
RANDOM_FOLD_HIP_MAX = math.radians(-30.0)   #: the random-fold family's least-raised femur


def _load_by_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


STC = _load_by_path(SN_SCRIPTS / "sn_table_core.py", "_sh_sn_table_core")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _r(x, n=4):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [_r(v, n) for v in x]
    if isinstance(x, np.ndarray):
        return [_r(v, n) for v in x.tolist()]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    return round(float(x), n)


def bisect(fn, lo, hi, tol=1e-7):
    """Root of a monotone sign change of ``fn`` on [lo, hi]."""
    flo = fn(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = fn(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ===================================================================================== verification of sh_kin
def verify_kin() -> dict:
    import torch  # noqa: PLC0415

    lk = STC.linkage()
    t = lambda x: torch.tensor(np.asarray(x, dtype=np.float64))  # noqa: E731
    th = np.linspace(math.radians(-45.0), math.radians(60.0), 211)
    tk = np.linspace(math.radians(-50.0), math.radians(85.0), 136)
    TH, TKk = np.meshgrid(th, tk, indexing="ij")
    TH, TKk = TH.ravel(), TKk.ravel()
    lens = np.linspace(0.45, 0.65, 101)
    out = {
        "defaults_vs_linkage": {"hip": [K.TH0, float(lk.hip_default_rad())],
                                "knee_left": [K.TK0, float(lk.knee_default_left_rad())]},
        "max_abs_diff_vs_torch_linkage": {
            "hip_len": float(np.abs(K.hip_len(th) - lk.hip_actuator_length(t(th)).numpy()).max()),
            "hip_angle": float(np.abs(K.hip_angle(lens) - lk.hip_angle_from_length(t(lens)).numpy()).max()),
            "knee_len": float(np.abs(K.knee_len(TH, TKk) - lk.knee_actuator_length(t(TH), t(TKk)).numpy()).max()),
            "abs_dLh_dth_vs_hip_moment_arm": float(np.abs(np.abs(K.dLh_dth(th)) - lk.hip_moment_arm(t(th)).numpy()).max()),
            "abs_dLk_dtk_vs_knee_moment_arm": float(np.abs(np.abs(K.dLk_dtk(TH, TKk))
                                                           - lk.knee_moment_arm(t(TH), t(TKk)).numpy()).max()),
            "dLk_dth_vs_knee_hip_coupling": float(np.abs(K.dLk_dth(TH, TKk)
                                                         - lk.knee_hip_coupling(t(TH), t(TKk)).numpy()).max()),
        },
    }
    # knee_angle round trip at several hips (on the operating branch)
    rt = []
    for h in np.linspace(math.radians(-44.0), math.radians(40.0), 9):
        kk = np.linspace(math.radians(-30.0), math.radians(75.0), 50)
        L = K.knee_len(h, kk)
        rt.append(float(np.abs(K.knee_angle(L, h) - kk).max()))
        rt.append(float(np.abs(K.knee_angle(L, h) - lk.knee_angle_from_length(t(L), t(np.full_like(L, h))).numpy()).max()))
    out["knee_angle_roundtrip_and_vs_torch_max_abs"] = max(rt)
    # finite differences (central, eps 1e-6 rad) on the stance range and the full ROM
    eps = 1e-6
    fd_h = (K.hip_len(th + eps) - K.hip_len(th - eps)) / (2 * eps)
    fd_k = (K.knee_len(TH, TKk + eps) - K.knee_len(TH, TKk - eps)) / (2 * eps)
    fd_c = (K.knee_len(TH + eps, TKk) - K.knee_len(TH - eps, TKk)) / (2 * eps)
    out["finite_difference_max_abs_err"] = {
        "dLh_dth": float(np.abs(fd_h - K.dLh_dth(th)).max()),
        "dLk_dtk": float(np.abs(fd_k - K.dLk_dtk(TH, TKk)).max()),
        "dLk_dth": float(np.abs(fd_c - K.dLk_dth(TH, TKk)).max()),
    }
    out["sign_over_full_rom_grid"] = {
        "dLh_dth_min": float(K.dLh_dth(th).min()), "dLh_dth_max": float(K.dLh_dth(th).max()),
        "dLk_dtk_min": float(K.dLk_dtk(TH, TKk).min()), "dLk_dtk_max": float(K.dLk_dtk(TH, TKk).max()),
        "dLk_dth_min": float(K.dLk_dth(TH, TKk).min()), "dLk_dth_max": float(K.dLk_dth(TH, TKk).max()),
    }
    # virtual-work identity of the coupled conversion: tau . dtheta == F . dL for random states / perturbations
    rng = np.random.default_rng(0)
    n = 2000
    a_th = rng.uniform(math.radians(-40), math.radians(30), n)
    a_tk = rng.uniform(math.radians(-10), math.radians(55), n)
    tau_h, tau_k = rng.normal(0, 150, n), rng.normal(0, 150, n)
    fh, fk = K.rod_forces(a_th, a_tk, tau_h, tau_k)
    dth, dtk = rng.normal(0, 1e-6, n), rng.normal(0, 1e-6, n)
    dLh = K.hip_len(a_th + dth) - K.hip_len(a_th)
    dLk = K.knee_len(a_th + dth, a_tk + dtk) - K.knee_len(a_th, a_tk)
    vw = (tau_h * dth + tau_k * dtk) - (fh * dLh + fk * dLk)
    out["virtual_work_rel_err_max"] = float(np.abs(vw).max() / np.abs(tau_h * dth + tau_k * dtk).max())
    # the action layer: sh_kin.angles_from_action vs the plant arithmetic run through the torch linkage
    a = np.linspace(-4.8, 4.8, 97)
    AH, AK = np.meshgrid(a, a, indexing="ij")
    AH, AK = AH.ravel(), AK.ravel()
    th0, tk0 = float(lk.hip_default_rad()), float(lk.knee_default_left_rad())
    lh = lk.hip_actuator_length(t(th0 + 0.25 * AH)).clamp(lk.HIP_LEN_MIN_M, lk.HIP_LEN_MAX_M)
    th_t = lk.hip_angle_from_length(lh)
    lkk = lk.knee_actuator_length(t(np.full_like(AK, th0)), t(tk0 + 0.25 * AK)).clamp(lk.KNEE_LEN_MIN_M, lk.KNEE_LEN_MAX_M)
    tk_t = lk.knee_angle_from_length(lkk, th_t)
    nth, ntk = K.angles_from_action(AH, AK)
    out["action_layer_max_abs_diff_rad"] = float(max(np.abs(nth - th_t.numpy()).max(), np.abs(ntk - tk_t.numpy()).max()))
    out["method"] = ("sh_kin (numpy) vs crab_hex_linkage (torch, float64) on a 211 x 136 hip x knee grid over the USD "
                     "ROM; central finite differences (eps 1e-6 rad) of the rod-length functions; the coupled conversion "
                     "checked by virtual work (tau.dtheta = F.dL) on 2000 random states; the action layer "
                     "(parkour_actions.py:345-376 arithmetic through the torch linkage) on a 97 x 97 grid of raw actions "
                     "in +-4.8")
    return out


# ===================================================================================== (a) posture map
def posture_facts() -> dict:
    lim = K.action_limits()
    # kinematic floor at constant reach: the stance hip reaches the -45 deg stop
    th_of = lambda h: float(K.stance_angles(h)[0])  # noqa: E731
    h_floor = bisect(lambda h: th_of(h) - K.HIP_STOP_LO, 0.55, 1.0)
    # top of the constant-reach family: first binding of hip stop hi, knee stop lo, rod windows, reach, raw clip
    def top_violation(h):
        th, tk, ok = K.stance_angles(h)
        a_h, a_k = K.action_from_angles(th, tk)
        v = {
            "reach": not bool(ok),
            "hip_stop_hi": th > K.HIP_STOP_HI,
            "knee_stop_lo": tk < K.KNEE_STOP_LO,
            "hip_rod_max": K.hip_len(th) > K.HIP_LEN_MAX_M,
            "knee_rod_window": not (K.KNEE_LEN_MIN_M <= K.knee_len(th, tk) <= K.KNEE_LEN_MAX_M),
            "raw_clip": abs(a_h) > K.RAW_CLIP or abs(a_k) > K.RAW_CLIP,
        }
        return v
    hs = np.arange(1.064, 1.70, 0.0005)
    h_top, top_binding = None, None
    for h in hs:
        v = top_violation(h)
        if any(v.values()):
            h_top, top_binding = round(float(h - 0.0005), 4), [k for k, b in v.items() if b]
            break
    # also below: rod windows at the floor
    th_f, tk_f, _ = K.stance_angles(h_floor)
    floor_rods = {"hip_rod_m": float(K.hip_len(th_f)), "knee_rod_m": float(K.knee_len(th_f, tk_f)),
                  "knee_left_rad": float(tk_f)}
    # h_min
    h_room = bisect(lambda h: float(K.hip_room_rad(h)) - HIP_ROOM_RAD, h_floor, 1.0)
    term1 = h_floor + FLOOR_MARGIN_M
    h_min_exact = max(term1, h_room)
    h_min = math.ceil(h_min_exact * 1000.0) / 1000.0
    # q over the reachable range: monotone + smooth (fine grid 1 mm)
    grid = np.round(np.arange(math.ceil(h_floor * 1000) / 1000, (h_top or 1.5) + 1e-9, 0.001), 6)
    a_h, a_k = K.q(grid)
    dh, dk = np.diff(a_h), np.diff(a_k)
    d2h, d2k = np.diff(a_h, 2), np.diff(a_k, 2)
    # rigid reference values (exploration table, flat pad) for the record
    flat_at_default = float(K.h_geom_flat_pad(K.TH0, K.TK0))
    table_rows = []
    for h in [h_floor, 0.70, h_min, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.03, 1.064, 1.10, 1.15, 1.20, 1.25, 1.30,
              1.35, 1.40]:
        if h_top is not None and h > h_top:
            continue
        th, tk, _ = K.stance_angles(h)
        qh, qk = K.q(h)
        table_rows.append({
            "h": _r(h, 4), "hip_deg": _r(math.degrees(th), 2), "knee_left_deg": _r(math.degrees(tk), 2),
            "q_hip": _r(qh, 4), "q_knee_left": _r(qk, 4),
            "hip_rod_mm": _r(1000 * K.hip_len(th), 1), "knee_rod_mm": _r(1000 * K.knee_len(th, tk), 1),
            "hip_room_to_stop_rad": _r(th - K.HIP_STOP_LO, 4),
            "toe_rise_hip_only_m": _r(K.relift_toe_rise(h), 4),
            "toe_rise_R_lift_m": _r(K.relift_toe_rise(h, knee_lift=-K.R_RHO * K.R_LIFT), 4),
            "belly_m": _r(h - K.BODY_HALF_Z_M, 4),
        })
    # dense q table for the build (5 mm, plus the 1.064 node and h_min)
    nodes = sorted(set(np.round(np.arange(round(h_min, 3), 1.50 + 1e-9, 0.005), 4).tolist()) | {1.064, round(h_min, 3)})
    nodes = [h for h in nodes if h_top is None or h <= h_top]
    qt = K.q_table(np.array(nodes))
    return {
        "anchor": {
            "H_SETTLED_m": K.H_SETTLED,
            "rigid_tilted_pad_h_at_a0_m": float(K.h_geom(K.TH0, K.TK0)),
            "rigid_flat_pad_h_at_a0_m": flat_at_default,
            "H_OFFSET_m": K.H_OFFSET,
            "mapping": "h_model = h - H_OFFSET, H_OFFSET = 1.064 - h_geom(defaults) = %+.5f m (tilted-pad rigid model: "
                       "toe centre + the footpad box's lowest corner, the pad tilting with the tibia; 1.0649 m at a = 0). "
                       "The flat-pad reading (pad bottom a constant 0.02 m below the toe centre) gives %.4f m at a = 0, "
                       "i.e. offset %+.4f m; it drifts with the tibia tilt (pad drop 0.0236 m at 0.69 m to 0.0288 m at "
                       "1.064), the tilted-pad offset is the one held constant." % (
                           K.H_OFFSET, flat_at_default, K.H_SETTLED - flat_at_default),
            "q_at_1p064": [float(v) for v in K.q(1.064)],
            "solver_residual_subtracted": list(K._Q_ANCHOR),
            "U0_toe_reach_m": K.U0,
            "defaults_rad": {"hip": K.TH0, "knee_left": K.TK0},
        },
        "method": ("q(h): fixed-point / closed-form two-link IK of the leg-plane chain (pivot (0.0635, -0.2413), femur "
                   "0.5842, tibia 0.8255) for toe centre (U0, -(h - H_OFFSET - pad_drop)), the knee-outboard branch; "
                   "a_hip = (theta_h - default)/0.25; a_knee_left = (knee angle that gives the solved knee ROD at the "
                   "DEFAULT hip - knee default)/0.25 (parkour_actions.py:364-376: the knee action is a rod length read at "
                   "the default hip and re-solved at the hip target); right legs' policy knee = -a_knee_left. Feet on a "
                   "flat floor, body level, cams 0, rigid joints (the PD sag is inside the constant anchor)."),
        "h_kinematic_floor_m": h_floor,
        "h_kinematic_floor_rule": "stance hip at the -45 deg USD stop at constant reach U0",
        "floor_rods": floor_rods,
        "h_top_constant_reach_m": h_top, "h_top_binding": top_binding,
        "h_min": {
            "term1_floor_plus_0p02_m": term1,
            "term2_hip_room_0p125_rad_m": h_room,
            "binding": "hip_room" if h_room >= term1 else "kinematic_floor",
            "exact_m": h_min_exact, "frozen_m": h_min,
            "rule": "h_min = max(kinematic floor at constant reach + 0.02 m, lowest h whose stance hip keeps >= 0.125 rad "
                    "(|R lift| 0.5 x 0.25) above the -45 deg stop); frozen = ceil to 1 mm",
            "q_at_h_min": [float(v) for v in K.q(h_min)],
        },
        "action_limits": lim,
        "q_range_reachable": {"grid_m": [float(grid[0]), float(grid[-1])], "step_m": 0.001,
                              "q_hip_range": [float(a_h.min()), float(a_h.max())],
                              "q_knee_left_range": [float(a_k.min()), float(a_k.max())],
                              "hip_monotone_increasing": bool((dh > 0).all()),
                              "knee_monotone_decreasing": bool((dk < 0).all()),
                              "knee_turning_point_m": (None if (dk < 0).all() else float(grid[1:][dk >= 0][0])),
                              "dq_hip_dh_per_m": [float(dh.min() / 0.001), float(dh.max() / 0.001)],
                              "dq_knee_dh_per_m": [float(dk.min() / 0.001), float(dk.max() / 0.001)],
                              "max_abs_second_diff_hip": float(np.abs(d2h).max()),
                              "max_abs_second_diff_knee": float(np.abs(d2k).max()),
                              "within_raw_clip": bool((np.abs(a_h) <= K.RAW_CLIP).all() and (np.abs(a_k) <= K.RAW_CLIP).all())},
        "table": table_rows,
        "q_table_5mm": {"columns": ["h_m", "q_hip", "q_knee_left"], "rows": [[_r(v, 6) for v in row] for row in qt.tolist()],
                        "note": "linear interpolation in this table reproduces sh_kin.q to < 1e-4 action units; 1.064 is "
                                "a node (q exactly 0); the build may call sh_kin.q directly"},
        "restep_clearance_method": ("body held, stance leg at q(h) (target angles) commanded q + (-0.5, 0) (hip-only lift, "
                                    "knee rod command held) and q + (-0.5, +0.5) (R's D3 LIFT: knee = -rho*side*hip, rho 1, "
                                    "= +0.5 left-convention); rise of the footpad's lowest corner"),
    }


# ===================================================================================== (b) forces
def _settled_rows(run_dir: Path):
    meta = json.loads((run_dir / "run_meta.json").read_text())
    jn = meta["joint_names"]
    hi = [jn.index(f"{l}_Hip_Femur_RevoluteJoint") for l in K.LEGS]
    ki = [jn.index(f"{l}_Femur_Tibia_RevoluteJoint") for l in K.LEGS]
    dt = float(meta["dt"])
    settle = int(round(SETTLE_AFTER_S / dt))
    acc = {k: [] for k in ("th", "tk", "tau_h", "tau_k", "ff", "h", "nload", "ep")}
    n_stand = 0
    files = sorted(glob.glob(str(run_dir / "raw" / "episode_*.npz")))
    for f in files:
        z = np.load(f)
        sm = z["p2r_stand_mode"].astype(bool)
        n_stand += int(sm.sum())
        run = np.zeros(len(sm), dtype=np.int64)
        c = 0
        for i, s in enumerate(sm):
            c = c + 1 if s else 0
            run[i] = c
        jp, tq, jv = z["joint_pos"], z["applied_torque"], z["joint_vel"]
        ff = z["foot_force_norm"]
        nload = (ff > FOOT_LOADED_N).sum(1)
        sel = (sm & (run > settle) & (np.abs(z["rod_speed"]).max(1) < ROD_STILL_M_S)
               & (np.abs(jv[:, hi + ki]).max(1) < JOINT_STILL_RAD_S) & (nload >= MIN_LOADED) & ~z["done"].astype(bool))
        if not sel.any():
            continue
        acc["th"].append(jp[sel][:, hi])
        acc["tk"].append(jp[sel][:, ki] * K.SIDE)
        acc["tau_h"].append(tq[sel][:, hi])
        acc["tau_k"].append(tq[sel][:, ki] * K.SIDE)
        acc["ff"].append(ff[sel])
        acc["h"].append(K.h_meas(z["root_pos_w"][sel, 2], z["terrain_z"][sel]))
        acc["nload"].append(nload[sel])
        acc["ep"].append(np.full(int(sel.sum()), len(acc["ep"])))
    out = {k: np.concatenate(v) for k, v in acc.items()}
    out["n_episodes"] = len(files)
    out["n_stand_rows"] = n_stand
    out["meta"] = {"checkpoint": meta.get("checkpoint"), "num_envs": meta.get("num_envs"), "dt": dt,
                   "total_mass_kg": float(sum(meta["body_masses_kg"])) if meta.get("body_masses_kg") else None}
    return out


def _stamp_stats(rows: dict) -> dict:
    th, tk, tau_h, tau_k = rows["th"], rows["tk"], rows["tau_h"], rows["tau_k"]
    fh, fk = K.rod_forces(th, tk, tau_h, tau_k)
    fh_u, fk_u = K.rod_forces_uncoupled(th, tk, tau_h, tau_k)
    mean_abs_fh_row = np.abs(fh).mean(1)
    mean_abs_fk_row = np.abs(fk).mean(1)
    per_leg_fh = np.abs(fh).mean(0)
    per_leg_fk = np.abs(fk).mean(0)
    th_m, tk_m = th.mean(0), tk.mean(0)
    f_nom = K.foot_load_n(6)
    # models at the measured mean posture of each leg (even 6-way share of the whole weight)
    mt_h0, mt_k0 = K.stance_joint_torques(th_m, tk_m, f_nom, leg_weight=False)
    mt_h1, mt_k1 = K.stance_joint_torques(th_m, tk_m, f_nom, leg_weight=True)
    mf_h0, mf_k0 = K.rod_forces(th_m, tk_m, mt_h0, mt_k0)
    mf_h1, mf_k1 = K.rod_forces(th_m, tk_m, mt_h1, mt_k1)
    # per-row model with each foot's MEASURED load (taken vertical, at the pad centre), leg weight included
    rt_h, rt_k = K.stance_joint_torques(th, tk, rows["ff"], leg_weight=True)
    rf_h, rf_k = K.rod_forces(th, tk, rt_h, rt_k)
    # per-leg regression of the measured torques on the measured foot load (lever check)
    reg = {}
    for i, leg in enumerate(K.LEGS):
        A = np.stack([rows["ff"][:, i], np.ones(len(th))], 1)
        ch = np.linalg.lstsq(A, tau_h[:, i], rcond=None)[0]
        ck = np.linalg.lstsq(A, tau_k[:, i], rcond=None)[0]
        lev_h = float(np.mean(K.toe_uz(th[:, i], tk[:, i])[0] - K.PIVOT_U_M))
        ku = K.PIVOT_U_M + K.FEMUR_LEN_M * np.cos(th[:, i])
        lev_k = float(np.mean(ku - K.toe_uz(th[:, i], tk[:, i])[0]))
        reg[leg] = {"hip_slope_m": _r(ch[0], 4), "hip_icpt_Nm": _r(ch[1], 2), "knee_slope_m": _r(ck[0], 4),
                    "knee_icpt_Nm": _r(ck[1], 2), "geom_hip_lever_m": _r(lev_h, 4), "geom_knee_lever_m": _r(-lev_k, 4),
                    "slope_sum_hip_plus_abs_knee_m": _r(ch[0] - ck[0], 4),
                    "geom_sum_femur_horizontal_m": _r(lev_h + lev_k, 4)}
    n = len(th)
    six = lambda a: float(np.mean(a))  # noqa: E731
    return {
        "n_rows": int(n), "n_episodes_with_rows": int(len(np.unique(rows["ep"]))),
        "n_episodes": rows["n_episodes"], "n_stand_rows": rows["n_stand_rows"],
        "share_6_loaded": float(np.mean(rows["nload"] == 6)),
        "h_meas_mean_m": float(rows["h"].mean()), "h_meas_p5_p95_m": [float(np.percentile(rows["h"], 5)),
                                                                     float(np.percentile(rows["h"], 95))],
        "mean_hip_deg": float(np.degrees(th.mean())), "mean_knee_left_deg": float(np.degrees(tk.mean())),
        "foot_load_sum_mean_N": float(rows["ff"].sum(1).mean()),
        "per_leg_foot_load_mean_N": [float(v) for v in rows["ff"].mean(0)],
        "tau_hip_per_leg_mean_Nm": [float(v) for v in tau_h.mean(0)],
        "tau_knee_left_per_leg_mean_Nm": [float(v) for v in tau_k.mean(0)],
        "six_leg_mean_tau_hip_Nm": six(tau_h), "six_leg_mean_tau_knee_left_Nm": six(tau_k),
        "sign_share_tau_hip_pos_loaded_legs": float(np.mean(tau_h[rows["ff"] > FOOT_LOADED_N] > 0)),
        "sign_share_tau_knee_neg_loaded_legs": float(np.mean(tau_k[rows["ff"] > FOOT_LOADED_N] < 0)),
        "sign_share_row_six_leg_mean_tau_hip_pos": float(np.mean(tau_h.mean(1) > 0)),
        "sign_share_row_six_leg_mean_tau_knee_neg": float(np.mean(tau_k.mean(1) < 0)),
        "F_hip_signed_mean_N": six(fh), "F_knee_signed_mean_N": six(fk),
        "sign_share_F_hip_pos_loaded_legs": float(np.mean(fh[rows["ff"] > FOOT_LOADED_N] > 0)),
        "sign_share_F_knee_neg_loaded_legs": float(np.mean(fk[rows["ff"] > FOOT_LOADED_N] < 0)),
        "six_leg_mean_abs_F_hip_N": {"mean": float(mean_abs_fh_row.mean()),
                                     "p50": float(np.percentile(mean_abs_fh_row, 50)),
                                     "p90": float(np.percentile(mean_abs_fh_row, 90)),
                                     "max": float(mean_abs_fh_row.max()),
                                     "by_loaded_feet": {str(k): float(mean_abs_fh_row[rows["nload"] == k].mean())
                                                        for k in (5, 6) if (rows["nload"] == k).any()}},
        "six_leg_mean_abs_F_knee_N": {"mean": float(mean_abs_fk_row.mean()),
                                      "p90": float(np.percentile(mean_abs_fk_row, 90))},
        "per_leg_mean_abs_F_hip_N": dict(zip(K.LEGS, [float(v) for v in per_leg_fh])),
        "worst_leg_hip": {"leg": K.LEGS[int(per_leg_fh.argmax())], "mean_abs_F_N": float(per_leg_fh.max()),
                          "row_max_p95_N": float(np.percentile(np.abs(fh).max(1), 95))},
        "per_leg_mean_abs_F_knee_N": dict(zip(K.LEGS, [float(v) for v in per_leg_fk])),
        "worst_leg_knee": {"leg": K.LEGS[int(per_leg_fk.argmax())], "mean_abs_F_N": float(per_leg_fk.max()),
                           "row_max_p95_N": float(np.percentile(np.abs(fk).max(1), 95))},
        "uncoupled_six_leg_mean_abs_F_hip_N": float(np.abs(fh_u).mean()),
        "model_at_measured_posture": {
            "no_leg_weight": {"tau_hip_Nm": six(mt_h0), "tau_knee_left_Nm": six(mt_k0), "F_hip_N": six(mf_h0),
                              "F_knee_N": six(mf_k0)},
            "with_leg_weight": {"tau_hip_Nm": six(mt_h1), "tau_knee_left_Nm": six(mt_k1), "F_hip_N": six(mf_h1),
                                "F_knee_N": six(mf_k1)},
            "measured_foot_loads_vertical_with_leg_weight": {"tau_hip_Nm": six(rt_h), "tau_knee_left_Nm": six(rt_k),
                                                             "six_leg_mean_abs_F_hip_N": float(np.abs(rf_h).mean()),
                                                             "six_leg_mean_abs_F_knee_N": float(np.abs(rf_k).mean())},
        },
        "lever_regression_per_leg": reg,
    }


def force_facts(posture: dict) -> dict:
    stamps = {}
    for ctrl, seeds in STAMPS.items():
        for es, rel in seeds.items():
            runs = sorted((LR / rel).iterdir())
            run = [r for r in runs if (r / "raw").is_dir()][-1]
            rows = _settled_rows(run)
            st = _stamp_stats(rows)
            st["run_dir"] = str(run.relative_to(REPO))
            st["meta"] = rows["meta"]
            stamps[f"{ctrl}/{es}"] = st
    keys = list(stamps)
    pool = lambda f: float(np.mean([f(stamps[k]) for k in keys]))  # noqa: E731
    meas_tau_h = pool(lambda s: s["six_leg_mean_tau_hip_Nm"])
    meas_tau_k = pool(lambda s: s["six_leg_mean_tau_knee_left_Nm"])
    meas_fh = pool(lambda s: s["six_leg_mean_abs_F_hip_N"]["mean"])
    meas_fk = pool(lambda s: s["six_leg_mean_abs_F_knee_N"]["mean"])
    m0 = {k: pool(lambda s, k=k: s["model_at_measured_posture"]["no_leg_weight"][k]) for k in
          ("tau_hip_Nm", "tau_knee_left_Nm", "F_hip_N", "F_knee_N")}
    m1 = {k: pool(lambda s, k=k: s["model_at_measured_posture"]["with_leg_weight"][k]) for k in
          ("tau_hip_Nm", "tau_knee_left_Nm", "F_hip_N", "F_knee_N")}
    mff = {k: pool(lambda s, k=k: s["model_at_measured_posture"]["measured_foot_loads_vertical_with_leg_weight"][k])
           for k in ("tau_hip_Nm", "tau_knee_left_Nm", "six_leg_mean_abs_F_hip_N", "six_leg_mean_abs_F_knee_N")}
    err = lambda model, meas: 100.0 * (model - meas) / abs(meas)  # noqa: E731
    errors = {
        "exploration_model_no_leg_weight": {"tau_hip_pct": err(m0["tau_hip_Nm"], meas_tau_h),
                                            "tau_knee_pct": err(m0["tau_knee_left_Nm"], meas_tau_k),
                                            "F_hip_pct": err(abs(m0["F_hip_N"]), meas_fh),
                                            "F_knee_pct": err(abs(m0["F_knee_N"]), meas_fk)},
        "model_with_leg_weight": {"tau_hip_pct": err(m1["tau_hip_Nm"], meas_tau_h),
                                  "tau_knee_pct": err(m1["tau_knee_left_Nm"], meas_tau_k),
                                  "F_hip_pct": err(abs(m1["F_hip_N"]), meas_fh),
                                  "F_knee_pct": err(abs(m1["F_knee_N"]), meas_fk)},
        "measured_foot_loads_model": {"tau_hip_pct": err(mff["tau_hip_Nm"], meas_tau_h),
                                      "tau_knee_pct": err(mff["tau_knee_left_Nm"], meas_tau_k),
                                      "F_hip_pct": err(mff["six_leg_mean_abs_F_hip_N"], meas_fh)},
    }
    per_stamp_err = {k: {"tau_hip_pct": err(s["model_at_measured_posture"]["no_leg_weight"]["tau_hip_Nm"],
                                            s["six_leg_mean_tau_hip_Nm"]),
                         "tau_knee_pct": err(s["model_at_measured_posture"]["no_leg_weight"]["tau_knee_left_Nm"],
                                             s["six_leg_mean_tau_knee_left_Nm"])} for k, s in stamps.items()}
    # exploration reference at 1.064 (target angles of q(1.064) = defaults)
    ref0 = K.static_rod_forces_at_h(1.064, leg_weight=False)
    ref1 = K.static_rod_forces_at_h(1.064, leg_weight=True)
    tau_ref0 = K.stance_joint_torques(K.TH0, K.TK0, K.foot_load_n(), leg_weight=False)
    # scaled model curves (scale = measured / model at the measured posture, pooled over the four stamps)
    s0_h = meas_fh / abs(m0["F_hip_N"])
    s0_k = meas_fk / abs(m0["F_knee_N"])
    s1_h = meas_fh / abs(m1["F_hip_N"])
    hs = np.round(np.arange(math.ceil(posture["h_kinematic_floor_m"] * 1000) / 1000,
                            min(1.50, posture["h_top_constant_reach_m"] or 1.5) + 1e-9, 0.001), 4)
    fh0, fk0 = K.static_rod_forces_at_h(hs, leg_weight=False)
    fh1, fk1 = K.static_rod_forces_at_h(hs, leg_weight=True)
    curves = {"exploration_model_unscaled": np.abs(fh0), "exploration_model_scaled": np.abs(fh0) * s0_h,
              "leg_weight_model_unscaled": np.abs(fh1), "leg_weight_model_scaled": np.abs(fh1) * s1_h,
              "foot_load_model": np.abs(fh1) * (mff["six_leg_mean_abs_F_hip_N"] / abs(m1["F_hip_N"]))}

    def hmax_of(curve):
        ok = curve <= FORCE_CAP_N
        if not ok.any():
            return None
        if ok.all():
            return float(hs[-1])
        # the curve rises with h: highest h before the first exceedance
        first_bad = int(np.argmax(~ok)) if ok[0] else 0
        if not ok[0]:
            return None
        return float(hs[first_bad - 1])
    hmax = {k: hmax_of(v) for k, v in curves.items()}
    sel_h = [0.70, 0.72, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.03, 1.064, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40]
    curve_rows = []
    for h in sel_h:
        if h < hs[0] or h > hs[-1]:
            continue
        i = int(np.argmin(np.abs(hs - h)))
        curve_rows.append({"h": h, "F_hip_expl_N": _r(curves["exploration_model_unscaled"][i], 0),
                           "F_hip_expl_scaled_N": _r(curves["exploration_model_scaled"][i], 0),
                           "F_hip_lw_N": _r(curves["leg_weight_model_unscaled"][i], 0),
                           "F_hip_lw_scaled_N": _r(curves["leg_weight_model_scaled"][i], 0),
                           "F_hip_foot_load_model_N": _r(curves["foot_load_model"][i], 0),
                           "F_knee_expl_N": _r(abs(fk0[i]), 0), "F_knee_expl_scaled_N": _r(abs(fk0[i]) * s0_k, 0),
                           "F_knee_lw_N": _r(abs(fk1[i]), 0)})
    return {
        "rows_rule": (f"settled stand rows: p2r_stand_mode (latched stand slot) held > {SETTLE_AFTER_S} s, all 12 rods "
                      f"still (|rod_speed| < {ROD_STILL_M_S}), all hip-pitch / knee |joint_vel| < {JOINT_STILL_RAD_S} "
                      f"rad/s, >= {MIN_LOADED} feet with net contact > {FOOT_LOADED_N} N, not done; joints picked by "
                      "run_meta.joint_names (*_Hip_Femur_RevoluteJoint, *_Femur_Tibia_RevoluteJoint); right-leg knee "
                      "angle and torque multiplied by -1 into the left convention; applied_torque = the implicit PD's "
                      "k (q_target - q) + d (qd_target - qd) (Isaac Lab ImplicitActuator estimate; static rows)"),
        "conversion": ("COUPLED: [tau_h, tau_k] = J^T [F_h, F_k], J = [[dLh/dth, 0], [dLk/dth, dLk/dtk]] at the measured "
                       "joint angles, i.e. F_k = tau_k / (dLk/dtk), F_h = (tau_h - (dLk/dth) F_k) / (dLh/dth); + = rod "
                       "pushes (extends), - = rod pulls. Sign conventions: dLh/dth > 0 and dLk/dtk > 0 over the whole "
                       "ROM (the linkage module's moment arms are their absolute values; the action term's velocity "
                       "feedforward omega_k = (dL_k/dt - coupling omega_h)/ma_k uses the same signs), dLk/dth = "
                       "knee_hip_coupling, signed, > 0 over the stance range; all three equal central finite differences "
                       "of the rod-length functions (verification block)."),
        "statistic": "six-leg mean of |F_hip| per row, averaged over rows (the user's cap statistic); worst leg = the leg "
                     "with the largest per-leg mean |F|; knee reported, hold-only (non-backdrivable), not capped",
        "per_stamp": stamps,
        "pooled_four_stamps_equal_weight": {
            "six_leg_mean_tau_hip_Nm": meas_tau_h, "six_leg_mean_tau_knee_left_Nm": meas_tau_k,
            "six_leg_mean_abs_F_hip_N": meas_fh, "six_leg_mean_abs_F_knee_N": meas_fk,
            "h_meas_mean_m": pool(lambda s: s["h_meas_mean_m"]),
            "model_at_measured_posture_no_leg_weight": m0, "model_at_measured_posture_with_leg_weight": m1,
            "model_measured_foot_loads": mff,
        },
        "model_vs_measured_error_pct": errors,
        "model_vs_measured_error_pct_per_stamp_exploration_model": per_stamp_err,
        "exploration_reference_at_1p064": {"F_hip_N": float(ref0[0]), "F_knee_N": float(ref0[1]),
                                           "tau_hip_Nm": float(tau_ref0[0]), "tau_knee_left_Nm": float(tau_ref0[1]),
                                           "with_leg_weight_F_hip_N": float(ref1[0]), "with_leg_weight_F_knee_N": float(ref1[1]),
                                           "method": "even 6-way share of the whole robot weight (230.06 kg x 9.81 / 6 = "
                                                     "376.1 N per toe, vertical, at the toe centre), q(1.064) = default "
                                                     "angles, virtual work + the coupled conversion; reproduces the "
                                                     "exploration's 1784 N / 1424 N"},
        "scale_factors": {"exploration_model_hip": s0_h, "exploration_model_knee": s0_k, "leg_weight_model_hip": s1_h,
                          "rule": "scale = pooled measured six-leg mean |F| / the model's |F| at the measured mean "
                                  "posture (per stamp, pooled), applied as a constant factor to the model curve"},
        "h_max_by_curve_m": hmax,
        "h_max_rule": f"highest h on a 1 mm grid whose modelled six-leg mean |F_hip| <= {FORCE_CAP_N} N",
        "curves": curve_rows,
        "excess_hip_torque_diagnostic": None,  # filled below
    }


def excess_diagnostic(ff: dict) -> dict:
    """Is the measured hip torque explainable by ANY force at the foot? For one force at the contact, the hip and knee
    actuator torques per unit vertical load satisfy slope_hip + |slope_knee| = Lf cos(th) (the femur's horizontal
    projection) whatever the contact point and the in-plane horizontal force: both shift the hip and knee torques by
    (almost) the same amount (moment arms about the hip pivot and the knee differ only by Lf sin(th) ~ 0.01 m)."""
    rows = []
    per_ctrl = {}
    for k, s in ff["per_stamp"].items():
        ctrl = k.split("/")[0]
        per_ctrl.setdefault(ctrl, []).append(s["six_leg_mean_tau_hip_Nm"] - s["six_leg_mean_tau_knee_left_Nm"])
        for leg, r in s["lever_regression_per_leg"].items():
            rows.append((k, leg, r["slope_sum_hip_plus_abs_knee_m"], r["geom_sum_femur_horizontal_m"], r["hip_slope_m"],
                         r["geom_hip_lever_m"], r["knee_slope_m"], r["geom_knee_lever_m"]))
    meas_sum = float(np.mean([r[2] for r in rows]))
    geom_sum = float(np.mean([r[3] for r in rows]))
    P = ff["pooled_four_stamps_equal_weight"]
    m0 = P["model_at_measured_posture_no_leg_weight"]
    m1 = P["model_at_measured_posture_with_leg_weight"]
    c_over_b = float(K.dLk_dth(K.TH0, K.TK0) / K.dLk_dtk(K.TH0, K.TK0))
    return {
        "mean_measured_slope_sum_m": meas_sum, "mean_geometric_sum_m": geom_sum,
        "mean_hip_slope_m": float(np.mean([r[4] for r in rows])), "mean_geom_hip_lever_m": float(np.mean([r[5] for r in rows])),
        "mean_knee_slope_m": float(np.mean([r[6] for r in rows])), "mean_geom_knee_lever_m": float(np.mean([r[7] for r in rows])),
        "tau_hip_minus_tau_knee_by_controller_Nm": {c: [float(v) for v in vs] for c, vs in per_ctrl.items()},
        "tau_hip_minus_tau_knee_model_Nm": {"no_leg_weight": m0["tau_hip_Nm"] - m0["tau_knee_left_Nm"],
                                            "with_leg_weight": m1["tau_hip_Nm"] - m1["tau_knee_left_Nm"]},
        "hip_rod_force_structure": ("F_hip = (tau_h - c F_k)/a = (tau_h - (c/b) tau_k)/a with c/b = %.3f at the default "
                                    "pose: F_hip tracks the (near-)invariant tau_h - tau_k, so it does NOT depend on how "
                                    "the stance's internal horizontal forces split the load between hip and knee (A-mask "
                                    "and R differ in that split by ~27 N*m but agree on F_hip within 1.5 %%); the knee rod "
                                    "force does depend on it" % c_over_b),
        "reading": ("Per leg, the measured torques regressed on the measured foot load: hip slope %.3f m vs a geometric "
                    "lever %.3f m, knee slope %.3f vs %.3f m. The hip/knee SPLIT differs by controller (R: more hip, less "
                    "knee) as an in-plane horizontal stance force would make it, but tau_hip - tau_knee is the same for "
                    "both controllers (%s N*m) and is %.0f %% / %.0f %% above the model (no / with leg weight: %.1f / %.1f "
                    "N*m); per unit foot load the slope sum is %.3f m vs the %.3f m that ANY foot force must give. The "
                    "four stamps agree on tau_hip - tau_knee to 0.2 N*m (it is fixed by the total weight: the six-leg "
                    "mean knee-point load x Lf cos th), so the excess (~45 N*m per leg, ~ +12 %% of the weight-only "
                    "value, +22 %% with leg weight) is not a foot force the static model omits and not noise: either "
                    "the sim's implicit-PD torque reading (Isaac Lab's k (q_target - q) estimate) differs from the "
                    "joints' true loads, or a load path outside the leg-plane statics exists. Which joint carries it "
                    "cannot be separated from the split; the hip ROD force tracks the sum (see hip_rod_force_structure), "
                    "so the whole excess lands on F_hip. Unexplained on CPU; needs a Kit-side check (e.g. the articulation's "
                    "measured joint reaction forces vs applied_torque on a settled stand) before the measured hip "
                    "number is trusted as a force." % (
                        float(np.mean([r[4] for r in rows])), float(np.mean([r[5] for r in rows])),
                        float(np.mean([r[6] for r in rows])), float(np.mean([r[7] for r in rows])),
                        {c: [round(v, 1) for v in vs] for c, vs in per_ctrl.items()},
                        100 * (P["six_leg_mean_tau_hip_Nm"] - P["six_leg_mean_tau_knee_left_Nm"]) /
                        (m0["tau_hip_Nm"] - m0["tau_knee_left_Nm"]) - 100,
                        100 * (P["six_leg_mean_tau_hip_Nm"] - P["six_leg_mean_tau_knee_left_Nm"]) /
                        (m1["tau_hip_Nm"] - m1["tau_knee_left_Nm"]) - 100,
                        m0["tau_hip_Nm"] - m0["tau_knee_left_Nm"], m1["tau_hip_Nm"] - m1["tau_knee_left_Nm"],
                        meas_sum, geom_sum)),
    }


# ===================================================================================== (c) table and ground geometry
def _lowest_parts(hip: float, knee: float) -> dict:
    """Lowest leg-plane z (body frame, root at 0) of each leg part for a pose (hull geometry of sn_table_core)."""
    tib = STC.tibia_hull(hip, knee)
    lp = STC.leg_plane_points(hip, knee)
    plate_bottom = -(STC.PLATE_CORE_HALF_Z_M + STC.PLATE_RADIUS_M)
    femur_low = min(lp["pivot"][1], lp["knee"][1]) - STC.FEMUR_RADIUS_M
    tib_low = min(p[1] for p in tib)
    parts = {"tibia_or_pad": tib_low, "femur": femur_low, "hip_plate": plate_bottom, "belly": -STC.BODY_HALF_M[2]}
    low = min(parts.values())
    first = "+".join(sorted(k for k, v in parts.items() if v <= low + 1e-3))   # parts within 1 mm of the lowest
    return {"parts_z": parts, "lowest_z": low, "first_contact": first}


def table_ground_facts(h_min: float, h_max: float, h_max_alt: float | None) -> dict:
    poses = {p: STC.retracted_pose(p) for p in STC.POSES}
    out = {"poses": {}}
    belly_half = STC.BODY_HALF_M[2]
    for p, pose in poses.items():
        lowp = _lowest_parts(pose["hip_rad"], pose["knee_left_rad"])
        # table: every leg part >= 0.02 m above the floor at spawn root z = t + 0.15875 + 0.005
        t_min = TOE_FLOOR_CLEAR_M - lowp["lowest_z"] - belly_half - SPAWN_CLEARANCE_M
        leg_low = min(v for k, v in lowp["parts_z"].items() if k != "belly")
        t_min_legs = TOE_FLOOR_CLEAR_M - leg_low - belly_half - SPAWN_CLEARANCE_M
        ground_root = -lowp["lowest_z"] + SPAWN_CLEARANCE_M
        rz635 = STC.spawn_root_z(0.635)
        clr = STC.table_leg_clearance(pose, rz635)
        out["poses"][p] = {
            "hip_deg": pose["hip_deg"], "knee_left_deg": pose["knee_left_deg"],
            "lowest_part_z_body_m": lowp["parts_z"], "lowest_leg_part": lowp["first_contact"],
            "table_top_min_m_legs_0p02_above_floor": t_min_legs,
            "table_lateral_clearance_m": clr["lateral_clearance_m"],
            "any_toe_inside_table_footprint": clr["any_toe_inside_table_footprint"],
            "ground": {"first_contact": lowp["first_contact"], "spawn_root_z_m": ground_root,
                       "h_meas_at_spawn_m": ground_root,
                       "belly_height_at_spawn_m": ground_root - belly_half,
                       "below_h_min": bool(ground_root < h_min)},
            "link_clearances": pose["link_clearances"],
            "hold_actions": {"hip": pose["hip_hold_action"], "knee_left": pose["knee_hold_action_left"]},
        }
        del t_min
    # random-fold family: hip in [stop + 0.5 deg, default], knee in [knee at the knee-rod minimum (opened), the largest
    # plate-clear fold] at that hip
    lim = STC.joint_limits_rad()
    m = math.radians(STC.LIMIT_MARGIN_DEG)
    hips = np.linspace(lim["hip"][0] + m, RANDOM_FOLD_HIP_MAX, 9)
    fam = []
    for hp in hips:
        k_lo = max(float(K.knee_angle(K.KNEE_LEN_MIN_M, hp)), lim["knee_left"][0] + m)
        k_hi = STC.max_clear_knee_fold(float(hp), step_rad=2e-3)["knee_rad"]
        for kn in np.linspace(k_lo, k_hi, 9):
            lp = _lowest_parts(float(hp), float(kn))
            leg_low = min(v for k, v in lp["parts_z"].items() if k != "belly")
            fam.append({"hip": float(hp), "knee": float(kn), "first": lp["first_contact"], "root": -lp["lowest_z"] + SPAWN_CLEARANCE_M,
                        "t_min": TOE_FLOOR_CLEAR_M - leg_low - belly_half - SPAWN_CLEARANCE_M})
    firsts = {}
    for f in fam:
        firsts[f["first"]] = firsts.get(f["first"], 0) + 1
    on_plates = [f for f in fam if "hip_plate" in f["first"]]
    out["random_fold_family"] = {
        "share_resting_on_hip_plates": len(on_plates) / len(fam),
        "share_spawn_below_h_min": float(np.mean([f["root"] < h_min for f in fam])),
        "definition": "hip uniform in [-44.5 deg, -30 deg] (femur raised: a retracted family), knee uniform in [max(knee at "
                      "the knee-rod minimum, knee "
                      "stop + 0.5 deg), the largest fold keeping 0.02 m tibia/pad clearance to the hip plate and body "
                      "(sn_table_core.max_clear_knee_fold)] at that hip; Body_Hip / cams 0; same pose on all six legs. "
                      "Evaluated on a 9 x 9 grid.",
        "n_grid": len(fam),
        "ground_first_contact_counts": firsts,
        "ground_spawn_root_z_m": [min(f["root"] for f in fam), float(np.median([f["root"] for f in fam])),
                                  max(f["root"] for f in fam)],
        "table_top_min_for_legs_clear_m": [min(f["t_min"] for f in fam), float(np.median([f["t_min"] for f in fam])),
                                           max(f["t_min"] for f in fam)],
        "rule_for_tables": "on a table of top t, draw a random fold only if its table_top_min <= t (legs >= 0.02 m above "
                           "the floor); otherwise it is a floor-touching start",
    }
    # table range
    t_restore = K.H_SETTLED - belly_half - BELLY_RESTORE_GAP_M
    t_min_all = max(out["poses"][p]["table_top_min_m_legs_0p02_above_floor"] for p in poses)

    def table_range(hmax):
        t_cmd_train = hmax - TABLE_CMD_MARGIN_TRAIN
        t_cmd_eval = hmax - TABLE_CMD_MARGIN_EVAL
        top_max = min(t_restore, t_cmd_train)
        top_max_eval = min(t_restore, t_cmd_eval)
        return {"h_max_m": hmax, "top_min_m": t_min_all, "top_max_train_m": top_max,
                "top_max_train_terms_m": {"restore_belly_gap": t_restore, "h_max_minus_0p264": t_cmd_train},
                "top_max_eval_m": top_max_eval, "feasible": bool(top_max >= t_min_all)}
    primary = table_range(h_max)
    cand = [0.4572, 0.5334, 0.635, 0.6858, 0.7112, 0.762]   # 18, 21, 25, 27, 28, 30 in
    ok = [t for t in cand if t_min_all <= t <= primary["top_max_eval_m"]]
    k3 = None
    if 0.635 in ok and len(ok) >= 3:
        k3 = [ok[0], 0.635, ok[-1]] if ok[-1] != 0.635 else [ok[0], ok[1], 0.635]
    k3_rows = None
    if k3:
        k3_rows = [{"top_m": t, "top_in": round(t / 0.0254, 1), "spawn_root_z_m": STC.spawn_root_z(t),
                    "h_cmd_min_m": max(h_min, t + TABLE_CMD_MARGIN_EVAL), "h_cmd_max_m": h_max,
                    "belly_gap_at_1p064_m": K.H_SETTLED - belly_half - t} for t in k3]
    alt = table_range(h_max_alt) if h_max_alt else None
    k3_alt = None
    if alt:
        ok2 = [t for t in cand if t_min_all <= t <= alt["top_max_eval_m"]]
        if 0.635 in ok2 and len(ok2) >= 3:
            k3_alt = [ok2[0], 0.635, ok2[-1]]
    out["table_range"] = {
        "rule": ("spawn root z = top + 0.15875 (belly) + 0.005; top_min = every leg part of the retracted pose >= 0.02 m "
                 "above the floor at spawn (knee_folded binds: its toes); legs clear the table laterally by the "
                 "sn_table_core lateral clearance (the 0.90 x 0.70 m table, +-3 cm / +-10 deg jitter, is inside the leg "
                 "yaw axes at |y| 0.6096; independent of the top height); top_max (training) = min(restore: belly at "
                 "1.064 - 0.15875 clears the top by >= 0.10 m, h_max - 0.264 so h_cmd >= top + 0.264 is reachable); "
                 "K3 eval tops: >= 3 incl. 0.635 with top + 0.31 <= h_max"),
        "primary": primary,
        "k3_eval_tops_m": k3, "k3_rows": k3_rows,
        "alternative_if_h_max_from_foot_load_model": alt, "k3_eval_tops_alternative_m": k3_alt,
    }
    out["ground_note"] = ("ground starts: h_meas at spawn (root z - floor) is below h_min for rod_min, knee_folded and "
                          "most random folds (the rest spawn above it but below 1.064), so the "
                          "limiter state starts at the measured height and both q(h) and the limiter CLAMP to [h_min, "
                          "h_max] (q is evaluated at max(h, h_min)). rod_min rests on the hip plates' bottoms (the femur "
                          "pivot hull sits at the same z): the hip-contact > 500 N clause fires at spawn -> the PLAN H5 "
                          "wrapper; knee_folded rests on its toes (pads). Random folds: %.0f %% of the grid rest on the "
                          "plates, the rest on the toes; spawn root z must be computed per draw (lowest leg-plane hull "
                          "point, sn_table_core.tibia_hull / plate / femur stadia, + 0.005 m)" %
                          (100 * out["random_fold_family"]["share_resting_on_hip_plates"]))
    return out


def coupling_share_1064() -> float:
    """Share of the modelled hip rod force at 1.064 that comes from the knee-coupling term (exploration model)."""
    tau_h, tau_k = K.stance_joint_torques(K.TH0, K.TK0, K.foot_load_n(), leg_weight=False)
    fh, fk = K.rod_forces(K.TH0, K.TK0, tau_h, tau_k)
    return float(-K.dLk_dth(K.TH0, K.TK0) * fk / K.dLh_dth(K.TH0) / fh)


# ===================================================================================== main
def _existing_frozen() -> dict | None:
    """The CHARTER-frozen block of the current ``facts.json`` (status ``FROZEN ...``), or None (no file / not frozen)."""
    try:
        fr = json.loads((CAMPAIGN / "facts.json").read_text()).get("frozen")
    except (OSError, ValueError):
        return None
    return fr if isinstance(fr, dict) and str(fr.get("status", "")).startswith("FROZEN") else None


def main() -> dict:
    now = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    ver = verify_kin()
    post = posture_facts()
    ff = force_facts(post)
    ff["excess_hip_torque_diagnostic"] = excess_diagnostic(ff)
    hm = ff["h_max_by_curve_m"]
    h_max_primary = hm["exploration_model_scaled"]
    h_min = post["h_min"]["frozen_m"]
    tg = table_ground_facts(h_min, h_max_primary if h_max_primary else h_min, hm.get("foot_load_model"))
    # ---- STOP flags
    errs = ff["model_vs_measured_error_pct"]
    e_expl = errs["exploration_model_no_leg_weight"]
    e_lw = errs["model_with_leg_weight"]
    worst_expl = max(abs(e_expl["tau_hip_pct"]), abs(e_expl["tau_knee_pct"]))
    worst_lw = max(abs(e_lw["tau_hip_pct"]), abs(e_lw["tau_knee_pct"]))
    stop_model = bool(min(worst_expl, worst_lw) > 100 * MODEL_TOL)
    stop_hmax = bool(h_max_primary is None or h_max_primary < H_MAX_FLOOR)
    fd = ver["finite_difference_max_abs_err"]
    sg = ver["sign_over_full_rom_grid"]
    h_top = post["h_top_constant_reach_m"] or 1.5
    h_stance = np.arange(h_min, h_top + 1e-9, 0.001)
    th_rng, tk_rng, _ = K.stance_angles(h_stance)
    coup_stance = K.dLk_dth(th_rng, tk_rng)
    h_all = np.arange(post["h_kinematic_floor_m"], h_top + 1e-9, 0.0005)
    c_all = K.dLk_dth(*K.stance_angles(h_all)[:2])
    h_coup_zero = float(h_all[int(np.argmax(c_all > 0))]) if (c_all <= 0).any() else None
    meas_signs = [(s["sign_share_row_six_leg_mean_tau_hip_pos"], s["sign_share_row_six_leg_mean_tau_knee_neg"])
                  for s in ff["per_stamp"].values()]
    meas_signs_loaded = [(s["sign_share_tau_hip_pos_loaded_legs"], s["sign_share_tau_knee_neg_loaded_legs"])
                         for s in ff["per_stamp"].values()]
    coupling_ok = (fd["dLk_dth"] < 1e-6 and fd["dLk_dtk"] < 1e-6 and fd["dLh_dth"] < 1e-6 and sg["dLh_dth_min"] > 0
                   and sg["dLk_dtk_min"] > 0 and bool((coup_stance > 0).all())
                   and ver["max_abs_diff_vs_torch_linkage"]["dLk_dth_vs_knee_hip_coupling"] < 1e-9
                   and min(a for a, _ in meas_signs) > 0.99 and min(b for _, b in meas_signs) > 0.90)
    stops = {
        "stop_model_error": {
            "value": stop_model,
            "evidence": ("pooled four walkstop stamps, settled stands at h_meas %.4f m: six-leg mean measured tau_hip %.1f "
                         "N*m, tau_knee %.1f N*m; exploration static model (no leg weight) at the measured posture %.1f / "
                         "%.1f N*m -> errors %+.1f %% hip / %+.1f %% knee; with leg weight %.1f / %.1f -> %+.1f %% / %+.1f "
                         "%%. Tolerance 10 %% on the joint torques: exceeded on BOTH joints by both models. The split "
                         "between hip and knee is controller-dependent (internal horizontal stance forces), but the "
                         "weight-fixed sum tau_hip - tau_knee is 12-22 %% above statics on every stamp, which no foot "
                         "force can produce (per unit foot load: measured hip + |knee| slope sum %.3f m vs the geometric "
                         "%.3f m) -- an unexplained term in the sim torque reading that lands entirely on F_hip." % (
                             ff["pooled_four_stamps_equal_weight"]["h_meas_mean_m"],
                             ff["pooled_four_stamps_equal_weight"]["six_leg_mean_tau_hip_Nm"],
                             ff["pooled_four_stamps_equal_weight"]["six_leg_mean_tau_knee_left_Nm"],
                             ff["pooled_four_stamps_equal_weight"]["model_at_measured_posture_no_leg_weight"]["tau_hip_Nm"],
                             ff["pooled_four_stamps_equal_weight"]["model_at_measured_posture_no_leg_weight"]["tau_knee_left_Nm"],
                             e_expl["tau_hip_pct"], e_expl["tau_knee_pct"],
                             ff["pooled_four_stamps_equal_weight"]["model_at_measured_posture_with_leg_weight"]["tau_hip_Nm"],
                             ff["pooled_four_stamps_equal_weight"]["model_at_measured_posture_with_leg_weight"]["tau_knee_left_Nm"],
                             e_lw["tau_hip_pct"], e_lw["tau_knee_pct"],
                             ff["excess_hip_torque_diagnostic"]["mean_measured_slope_sum_m"],
                             ff["excess_hip_torque_diagnostic"]["mean_geometric_sum_m"])),
            "worst_abs_error_pct_exploration_model": worst_expl, "worst_abs_error_pct_leg_weight_model": worst_lw,
        },
        "stop_h_max_low": {
            "value": stop_hmax,
            "evidence": ("measured six-leg mean |F_hip| %.0f N (pooled) already exceeds the 2000 N cap at the walking "
                         "height; model scaled to the measurement (x%.3f) -> h_max %s m (< %.2f). Unscaled exploration model "
                         "h_max %s m; leg-weight model unscaled %s m; foot-load-based model %s m." % (
                             ff["pooled_four_stamps_equal_weight"]["six_leg_mean_abs_F_hip_N"],
                             ff["scale_factors"]["exploration_model_hip"],
                             None if h_max_primary is None else "%.3f" % h_max_primary, H_MAX_FLOOR,
                             hm["exploration_model_unscaled"], hm["leg_weight_model_unscaled"], hm["foot_load_model"])),
        },
        "stop_coupling_sign_ambiguous": {
            "value": not coupling_ok,
            "evidence": ("dLh/dth, dLk/dtk and dLk/dth (signed) match central finite differences of the rod-length "
                         "functions to %.1e / %.1e / %.1e m/rad and the torch linkage module (knee_hip_coupling) to %.1e; "
                         "dLh/dth in [%.4f, %.4f] and dLk/dtk in [%.4f, %.4f] m/rad over the whole ROM grid (both > 0, "
                         "the action term's own feedforward relies on it); coupling dLk/dth in [%.4f, %.4f] m/rad over "
                         "the stance range h_min..%.3f m (> 0; it crosses zero only at %s m, below h_min, near the "
                         "kinematic floor); the six-leg mean measured tau_hip > 0 and tau_knee (left conv.) < 0 on >= %.4f "
                         "/ %.4f of settled rows per stamp (loaded legs only: >= %.3f / %.3f) -> F_hip pushes, F_knee "
                         "pulls; with the coupling > 0 the knee pull ADDS to the hip rod force (%.0f %% of it at 1.064)." % (
                             fd["dLh_dth"], fd["dLk_dtk"], fd["dLk_dth"],
                             ver["max_abs_diff_vs_torch_linkage"]["dLk_dth_vs_knee_hip_coupling"],
                             sg["dLh_dth_min"], sg["dLh_dth_max"], sg["dLk_dtk_min"], sg["dLk_dtk_max"],
                             float(coup_stance.min()), float(coup_stance.max()), h_top,
                             "none" if h_coup_zero is None else "%.3f" % h_coup_zero,
                             min(a for a, _ in meas_signs), min(b for _, b in meas_signs),
                             min(a for a, _ in meas_signs_loaded), min(b for _, b in meas_signs_loaded),
                             100.0 * coupling_share_1064())),
            "coupling_zero_crossing_h_m": h_coup_zero,
        },
    }
    derived_frozen = {"h_min_m": h_min, "h_max_m": h_max_primary,
                      "h_max_rule": "exploration static model scaled by the pooled measured / model six-leg mean "
                                    "|F_hip| (x%.3f), highest h with <= 2000 N" % ff["scale_factors"]["exploration_model_hip"],
                      "status": "PROVISIONAL -- step-1 STOP raised; the user decides before anything is frozen in the "
                                "CHARTER",
                      "top_min_m": tg["table_range"]["primary"]["top_min_m"],
                      "top_max_train_m": tg["table_range"]["primary"]["top_max_train_m"],
                      "k3_eval_tops_m": tg["table_range"]["k3_eval_tops_m"]}
    # a re-run never overwrites the CHARTER-frozen block (the user decisions at the step-1 and step-4 stops): an existing
    # FROZEN block is carried over verbatim and this run's derivation is written beside it (``frozen_rederived``)
    kept = _existing_frozen()
    facts = {
        "campaign": "2026-10-13_stand_height", "step": "1 (facts and premises)", "generated": now,
        "generator": {"script": str(Path(__file__).relative_to(REPO)), "sha256": _sha(Path(__file__)),
                      "sh_kin_sha256": _sha(HERE / "sh_kin.py"),
                      "sources_sha256": {"crab_hex_linkage.py": _sha(K.FWD_MDP / "crab_hex_linkage.py"),
                                         "crab_hex_dimensions.py": _sha(K.DIMS_PATH),
                                         "crab_hex_foot_fk.py": _sha(K.FWD_MDP / "crab_hex_foot_fk.py"),
                                         "parkour_actions.py": _sha(K.FWD_MDP / "parkour_actions.py"),
                                         "sn_table_core.py": _sha(SN_SCRIPTS / "sn_table_core.py")},
                      "run": "CUDA_VISIBLE_DEVICES='' isaac_venv python scripts/sh_facts.py (CPU, no Kit)"},
        "stop": {k: v["value"] for k, v in stops.items()},
        "stop_evidence": stops,
        "frozen": derived_frozen if kept is None else kept,
        **({} if kept is None else {"frozen_rederived": derived_frozen}),
        "verification": ver,
        "a_posture": post,
        "b_forces": ff,
        "c_table_ground": tg,
    }
    (CAMPAIGN / "facts.json").write_text(json.dumps(facts, indent=1, default=lambda o: _r(o, 6)) + "\n")
    write_md(facts)
    return facts


def write_md(f: dict) -> None:
    a, b, c = f["a_posture"], f["b_forces"], f["c_table_ground"]
    P = b["pooled_four_stamps_equal_weight"]
    e = b["model_vs_measured_error_pct"]
    L = []
    L.append("# Stand height -- step 1 facts (CPU, existing raws)\n")
    L.append(f"Generated {f['generated']} by `scripts/sh_facts.py` (posture map: `scripts/sh_kin.py`). Full record with "
             "methods: `facts.json`.\n")
    L.append("## STOP flags\n")
    L.append("| flag | value | evidence (short) |\n|---|---|---|")
    for k, v in f["stop_evidence"].items():
        L.append(f"| `{k}` | **{v['value']}** | {v['evidence']} |")
    L.append("")
    L.append("## (a) Posture map q(h)\n")
    an = a["anchor"]
    L.append(f"- Anchor: h_model = h - H_OFFSET, H_OFFSET = {an['H_OFFSET_m']:+.4f} m (tilted-pad rigid model gives "
             f"{an['rigid_tilted_pad_h_at_a0_m']:.4f} m at a = 0; the flat-pad reading gives {an['rigid_flat_pad_h_at_a0_m']:.4f} m). "
             f"q(1.064) = ({an['q_at_1p064'][0]}, {an['q_at_1p064'][1]}). Toe reach U0 = {an['U0_toe_reach_m']:.4f} m.")
    hm = a["h_min"]
    L.append(f"- Kinematic floor (stance hip at the -45 deg stop, constant reach): {a['h_kinematic_floor_m']:.4f} m. "
             f"Top of the constant-reach family: {a['h_top_constant_reach_m']} m ({', '.join(a['h_top_binding'] or [])}).")
    L.append(f"- **h_min = {hm['frozen_m']:.3f} m** = max(floor + 0.02 = {hm['term1_floor_plus_0p02_m']:.4f}, hip room 0.125 rad "
             f"= {hm['term2_hip_room_0p125_rad_m']:.4f}) -> binding: {hm['binding']}; q(h_min) = "
             f"({hm['q_at_h_min'][0]:+.3f}, {hm['q_at_h_min'][1]:+.3f}).")
    qr = a["q_range_reachable"]
    L.append(f"- q over {qr['grid_m'][0]:.3f}-{qr['grid_m'][1]:.3f} m (1 mm grid): hip strictly rising "
             f"{qr['hip_monotone_increasing']}; knee strictly falling up to {qr['knee_turning_point_m']} m (it turns only "
             f"there, far above any h_max candidate), so q is monotone and smooth on [h_min, h_max]; inside +-4.8: "
             f"{qr['within_raw_clip']}; max second difference {qr['max_abs_second_diff_hip']:.1e} / "
             f"{qr['max_abs_second_diff_knee']:.1e} units per mm^2.")
    lim = a["action_limits"]
    L.append(f"- Limits (action units): hip stop {lim['hip_stop_lo_action']:+.3f} / {lim['hip_stop_hi_action']:+.3f}; hip rod "
             f"window {lim['hip_rod_min_action']:+.3f} / {lim['hip_rod_max_action']:+.3f}; knee rod window (left) "
             f"{lim['knee_rod_min_action_left']:+.3f} / {lim['knee_rod_max_action_left']:+.3f}; raw clip +-4.8.\n")
    L.append("| h (m) | hip (deg) | knee L (deg) | q_hip | q_knee L | hip room (rad) | toe rise hip-only (cm) | toe rise R lift (cm) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in a["table"]:
        L.append(f"| {r['h']:.3f} | {r['hip_deg']:.2f} | {r['knee_left_deg']:.2f} | {r['q_hip']:+.3f} | {r['q_knee_left']:+.3f} | "
                 f"{r['hip_room_to_stop_rad']:.3f} | {100 * r['toe_rise_hip_only_m']:.1f} | {100 * r['toe_rise_R_lift_m']:.1f} |")
    L.append("")
    L.append("## (b) Forces (coupled linkage conversion)\n")
    L.append("| stamp | rows | h_meas | tau_hip | tau_knee | 6-leg mean abs F_hip | p90 | worst hip leg | 6-leg mean abs F_knee | worst knee leg |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for k, s in b["per_stamp"].items():
        fh = s["six_leg_mean_abs_F_hip_N"]
        L.append(f"| {k} | {s['n_rows']} | {s['h_meas_mean_m']:.4f} | {s['six_leg_mean_tau_hip_Nm']:.1f} | "
                 f"{s['six_leg_mean_tau_knee_left_Nm']:.1f} | {fh['mean']:.0f} | {fh['p90']:.0f} | {s['worst_leg_hip']['leg']} "
                 f"{s['worst_leg_hip']['mean_abs_F_N']:.0f} | {s['six_leg_mean_abs_F_knee_N']['mean']:.0f} | "
                 f"{s['worst_leg_knee']['leg']} {s['worst_leg_knee']['mean_abs_F_N']:.0f} |")
    m0 = P["model_at_measured_posture_no_leg_weight"]
    m1 = P["model_at_measured_posture_with_leg_weight"]
    L.append(f"| **pooled** | | {P['h_meas_mean_m']:.4f} | {P['six_leg_mean_tau_hip_Nm']:.1f} | {P['six_leg_mean_tau_knee_left_Nm']:.1f} | "
             f"**{P['six_leg_mean_abs_F_hip_N']:.0f}** | | | {P['six_leg_mean_abs_F_knee_N']:.0f} | |")
    L.append(f"| model (exploration, no leg weight) | | | {m0['tau_hip_Nm']:.1f} | {m0['tau_knee_left_Nm']:.1f} | {abs(m0['F_hip_N']):.0f} | | | {abs(m0['F_knee_N']):.0f} | |")
    L.append(f"| model (with leg weight) | | | {m1['tau_hip_Nm']:.1f} | {m1['tau_knee_left_Nm']:.1f} | {abs(m1['F_hip_N']):.0f} | | | {abs(m1['F_knee_N']):.0f} | |")
    L.append("")
    L.append(f"- Model vs measured joint torques: exploration model hip {e['exploration_model_no_leg_weight']['tau_hip_pct']:+.1f} %, "
             f"knee {e['exploration_model_no_leg_weight']['tau_knee_pct']:+.1f} %; with leg weight hip "
             f"{e['model_with_leg_weight']['tau_hip_pct']:+.1f} %, knee {e['model_with_leg_weight']['tau_knee_pct']:+.1f} % "
             "(tolerance 10 %).")
    d = b["excess_hip_torque_diagnostic"]
    L.append(f"- Diagnostic: per-leg regression of torque on foot load -- hip slope {d['mean_hip_slope_m']:.3f} m vs geometric lever "
             f"{d['mean_geom_hip_lever_m']:.3f}; knee slope {d['mean_knee_slope_m']:.3f} vs {d['mean_geom_knee_lever_m']:.3f}; "
             f"hip + abs(knee) {d['mean_measured_slope_sum_m']:.3f} vs the geometric {d['mean_geometric_sum_m']:.3f} m that any foot "
             f"force must satisfy; tau_hip - tau_knee per stamp {json.dumps({c: [round(v, 1) for v in vs] for c, vs in d['tau_hip_minus_tau_knee_by_controller_Nm'].items()})} N*m vs model "
             f"{d['tau_hip_minus_tau_knee_model_Nm']['no_leg_weight']:.1f} (no leg weight) / "
             f"{d['tau_hip_minus_tau_knee_model_Nm']['with_leg_weight']:.1f} -> an unexplained term in the sim torque reading, "
             "not a missing foot force; F_hip tracks this sum, so the excess lands on the hip rod force (needs a Kit-side check).")
    ref = b["exploration_reference_at_1p064"]
    L.append(f"- Exploration reference reproduced at 1.064: F_hip {ref['F_hip_N']:.0f} N, F_knee {ref['F_knee_N']:.0f} N "
             f"(with leg weight {ref['with_leg_weight_F_hip_N']:.0f} / {ref['with_leg_weight_F_knee_N']:.0f}).")
    L.append(f"- h_max by curve (2000 N cap): {json.dumps({k: (None if v is None else round(v, 3)) for k, v in b['h_max_by_curve_m'].items()})}\n")
    L.append("| h (m) | F_hip model | F_hip scaled | F_hip leg-wt | F_hip leg-wt scaled | F_hip foot-load model | F_knee model | F_knee scaled |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in b["curves"]:
        L.append(f"| {r['h']:.3f} | {r['F_hip_expl_N']:.0f} | {r['F_hip_expl_scaled_N']:.0f} | {r['F_hip_lw_N']:.0f} | "
                 f"{r['F_hip_lw_scaled_N']:.0f} | {r['F_hip_foot_load_model_N']:.0f} | {r['F_knee_expl_N']:.0f} | {r['F_knee_expl_scaled_N']:.0f} |")
    L.append("")
    L.append("## (c) Table and ground starts\n")
    tr = c["table_range"]
    pr = tr["primary"]
    L.append(f"- Primary (h_max {pr['h_max_m']}): top_min {pr['top_min_m']:.3f} m, top_max (train) {pr['top_max_train_m']:.3f} m "
             f"(restore bound {pr['top_max_train_terms_m']['restore_belly_gap']:.3f}, h_max - 0.264 = "
             f"{pr['top_max_train_terms_m']['h_max_minus_0p264']:.3f}); eval max {pr['top_max_eval_m']:.3f}; K3 tops "
             f"{tr['k3_eval_tops_m']}.")
    if tr["alternative_if_h_max_from_foot_load_model"]:
        al = tr["alternative_if_h_max_from_foot_load_model"]
        L.append(f"- If h_max came from the foot-load model ({al['h_max_m']}): top_max (train) {al['top_max_train_m']:.3f}, eval "
                 f"max {al['top_max_eval_m']:.3f}, K3 tops {tr['k3_eval_tops_alternative_m']}.")
    L.append("\n| pose | min table top (legs >= 0.02 m above floor) | lowest leg part | ground: first contact | ground spawn root z = h_meas | below h_min |")
    L.append("|---|---|---|---|---|---|")
    for p, v in c["poses"].items():
        g = v["ground"]
        L.append(f"| {p} | {v['table_top_min_m_legs_0p02_above_floor']:.3f} | {v['lowest_leg_part']} | {g['first_contact']} | "
                 f"{g['spawn_root_z_m']:.3f} | {g['below_h_min']} |")
    rf = c["random_fold_family"]
    L.append(f"| random folds (9x9 grid) | {rf['table_top_min_for_legs_clear_m'][0]:.3f}-{rf['table_top_min_for_legs_clear_m'][2]:.3f} | | "
             f"{rf['ground_first_contact_counts']} | {rf['ground_spawn_root_z_m'][0]:.3f}-{rf['ground_spawn_root_z_m'][2]:.3f} | "
             f"{rf['share_spawn_below_h_min']:.0%} of the grid |")
    L.append("")
    L.append(f"- {c['ground_note']}.")
    (CAMPAIGN / "facts.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    out = main()
    print(json.dumps(out["stop"], indent=1))
    print(json.dumps(out["frozen"], indent=1, default=str))
