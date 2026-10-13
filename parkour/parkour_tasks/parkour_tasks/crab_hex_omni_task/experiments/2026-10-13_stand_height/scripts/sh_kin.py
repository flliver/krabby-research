#!/usr/bin/env python3
"""Stand-height campaign (``2026-10-13_stand_height``) -- the POSTURE MAP and the linkage STATICS, numpy only.

This is the module the later build (R_h, the router, the trainer, the readers) imports. It is a numpy re-statement of the
plant's own pitch-linkage and action-term arithmetic, so it can be loaded before Kit and on CPU without torch:

* ``crab_hex_forward_task/mdp/crab_hex_linkage.py`` -- rod length <-> angle for the hip and the knee, the signed partial
  derivatives (the linkage module returns |dL/dtheta| for the two moment arms; the SIGNED forms are here and are pinned
  against the torch module and against finite differences by ``sh_facts.py``);
* ``crab_hex_forward_task/mdp/parkour_actions.py:313-387`` (``_apply_pitch_linkage``) -- hip target = default + 0.25 a,
  converted to a rod length clamped to [0.450, 0.650] m; knee action = a rod length computed at the DEFAULT hip from
  ``knee_default + 0.25 a`` (left convention), clamped, and re-solved at the hip TARGET angle;
* ``crab_hex_forward_task/mdp/crab_hex_foot_fk.py`` / ``sn_table_core.leg_plane_points`` -- the leg-plane chain (pivot
  (0.0635, -0.2413) from the yaw axis / body centre, femur 0.5842, tibia 0.8255);
* ``assets/crab.usda`` -- the footpad (0.06 x 0.06 x 0.04 m box centred on the toe, its 0.04 axis along the tibia) and
  the femur / tibia centres of mass (femur: mid hinge-to-hinge; tibia: 0.2667 m below the knee along the tibia).

CONVENTIONS. Angles are the LEFT-leg sim convention (hip: 0 = femur horizontal, + = femur down; knee: + folds the toe
inboard). Right legs (FR/MR/RR) carry a 180-deg frame flip on the knee: right joint angle = -left, right knee torque =
-left, and the right knee POLICY action = -(left-convention knee action) (``SIDE``; R's ``knee = -rho * side * hip`` is
the same rule). Hip angles, torques and actions have the same sign on every leg. Leg plane: u = outboard of the leg's yaw
axis (the body side face), z = up from the body centre (= the root frame origin).

HEIGHT. ``h`` is the root (``chassis/body`` centre) height above the floor under the robot, i.e. the shared ``h_meas``
definition: root z minus the mean terrain height under the six feet (:func:`h_meas`). The rigid model height for a leg
pose is :func:`h_geom` = -(toe z) + the footpad's lowest-corner drop (the pad is a box tilted with the tibia). The
settled simulation reads :data:`H_SETTLED` = 1.064 m at the zero action; :data:`H_OFFSET` = H_SETTLED - h_geom(defaults)
is a CONSTANT anchor (joint sag under load, pad penetration, terrain roughness), so the map is
``h_model = h - H_OFFSET`` and :func:`q` (h) == (0.0, 0.0) exactly at 1.064 (the residual of the solver at the anchor is
subtracted too). The older flat-pad reading (pad bottom 0.02 below the toe centre, 1.056 m at a = 0, offset 0.008 m) is
kept as :func:`h_geom_flat_pad` for comparison only.

POSTURE. ``q(h) = (a_hip, a_knee_left)`` is the stance posture in policy action units that puts the toe at the default
reach :data:`U0` (constant stance width, cams 0, body level, feet on a flat floor) with the root at height ``h``:
two-link inverse kinematics for the hip / knee joint angles, then the action-term inverse (hip: (theta_h -
default)/0.25; knee: the knee rod length at the solved pose, re-read at the DEFAULT hip, minus the knee default, /0.25).
``q`` is monotone (hip rising, knee falling with h) and smooth over [:func:`h_kinematic_floor`, ...]; the caller clamps
``h`` to the campaign's frozen [h_min, h_max] (``facts.json``) BEFORE calling (:func:`clamp_h`).
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
#: scripts -> 2026-10-13_stand_height -> experiments -> crab_hex_omni_task -> parkour_tasks
TASKS = HERE.parents[3]
FWD_MDP = TASKS / "crab_hex_forward_task" / "mdp"
DIMS_PATH = FWD_MDP / "crab_hex_dimensions.py"


def _load_dims():
    spec = importlib.util.spec_from_file_location("_sh_crab_hex_dimensions", str(DIMS_PATH))
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"sh_kin: cannot load {DIMS_PATH}")
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


dims = _load_dims()

# ===================================================================================== constants
IN_TO_M = dims.IN_TO_M
G = 9.81
LEGS = ("FL", "FR", "ML", "MR", "RL", "RR")
#: +1 left, -1 right: right-leg knee angle / torque / policy action = SIDE * (left-convention value)
SIDE = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
ROD_SCALE = 0.25            #: rad per action unit, hip and knee channels
RAW_CLIP = 4.8              #: the env action term's raw clip under preset 2a (crab_hex_env_cfg.py:272)
WRAPPER_CLIP = 1.0          #: the runner's clip_actions (S1's rows; the stand net's residual)
H_SETTLED = 1.064           #: settled sim root height above the terrain at the zero action (walkstop raws, table test)
HEIGHT_OBS_SCALE = 0.4      #: stand-net input column 15 = (h_cmd - H_SETTLED) / HEIGHT_OBS_SCALE (PLAN H2)
HEIGHT_RATE_M_S = 0.08      #: the h_cmd rate limiter (PLAN H2)
R_LIFT = -0.5               #: R's D3 hip lift (action units); its knee lift is -rho*side*hip = +0.5 left-convention
R_RHO = 1.0

# linkage geometry (crab_hex_linkage.py, same formulas)
FEMUR_ATTACH_R_M = dims.FEMUR_ACT_ATTACH_FROM_HIP_PIVOT_IN * IN_TO_M
FEMUR_LEN_M = dims.FEMUR_HINGE_TO_HINGE_M
TIBIA_LEN_M = dims.TIBIA_KNEE_TO_TOE_M
KNEE_LEVER_M = dims.TIBIA_ACT_LEVER_IN * IN_TO_M
HIP_ANCHOR_U_M = 0.0
HIP_ANCHOR_D_M = -21.0 * IN_TO_M
KNEE_ANCHOR_U_M = 2.333 * IN_TO_M
KNEE_ANCHOR_D_M = -(dims.HIP_PLATE_WIDTH_IN / 2.0) * IN_TO_M
HIP_LEN_MIN_M = dims.HIP_ACTUATOR["retracted_len_m"]
HIP_LEN_MAX_M = HIP_LEN_MIN_M + dims.HIP_ACTUATOR["stroke_m"]
KNEE_LEN_MIN_M = dims.KNEE_ACTUATOR["retracted_len_m"]
KNEE_LEN_MAX_M = KNEE_LEN_MIN_M + dims.KNEE_ACTUATOR["stroke_m"]
HIP_FORCE_N = dims.HIP_ACTUATOR["force_n"]
KNEE_FORCE_N = dims.KNEE_ACTUATOR["force_n"]
HIP_SPEED_M_S = dims.HIP_ACTUATOR["speed_loaded_m_s"]
KNEE_SPEED_M_S = dims.KNEE_ACTUATOR["speed_loaded_m_s"]

# leg-plane chain
PIVOT_U_M = dims.FEMUR_PIVOT_OUTBOARD_M     # 0.0635
PIVOT_Z_M = dims.FEMUR_PIVOT_Z_M            # -0.2413
BODY_HALF_Z_M = dims.BODY_SIZE_M[2] / 2.0   # 0.15875 (belly below the root)
PAD_HALF_AXIAL_M = 0.02                     # footpad half extent along the tibia (USD scale z 0.04)
PAD_HALF_PERP_M = 0.03                      # footpad half extent across the tibia in the leg plane (USD scale y 0.06)

# joint stops (USD hard limits, rad)
HIP_STOP_LO = math.radians(dims.HIP_SIM_LIMITS_DEG[0])    # -45 deg
HIP_STOP_HI = math.radians(dims.HIP_SIM_LIMITS_DEG[1])    # +60 deg
KNEE_STOP_LO = math.radians(dims.KNEE_SIM_LIMITS_LEFT_DEG[0])
KNEE_STOP_HI = math.radians(dims.KNEE_SIM_LIMITS_LEFT_DEG[1])

# masses (USD / dims): every leg's femur, tibia, footpad; the whole robot
FEMUR_MASS_KG = dims.LEG_LINK_MASSES_KG["femur"]
TIBIA_MASS_KG = dims.LEG_LINK_MASSES_KG["tibia"]
PAD_MASS_KG = dims.FOOTPAD_MASS_KG
TOTAL_MASS_KG = dims.TOTAL_MASS_KG
#: tibia centre of mass below the knee along the tibia (assets/crab.usda: knee at local z 0.326009, COM at 0.0593196)
TIBIA_COM_BELOW_KNEE_M = 0.326009 - 0.05931956492
#: femur centre of mass from the hip pivot along the femur (USD: COM at the femur's mid hinge-to-hinge)
FEMUR_COM_FROM_PIVOT_M = 0.5 * FEMUR_LEN_M


def _f(x):
    return np.asarray(x, dtype=np.float64)


# ===================================================================================== hip linkage
def hip_len(th):
    """Hip rod pin-to-pin length (m) at hip angle ``th`` (rad)."""
    th = _f(th)
    pu = FEMUR_ATTACH_R_M * np.cos(th) - HIP_ANCHOR_U_M
    pd = FEMUR_ATTACH_R_M * np.sin(th) - HIP_ANCHOR_D_M
    return np.sqrt(pu * pu + pd * pd)


def hip_angle(length):
    """Inverse of :func:`hip_len` on the operating branch (the linkage module's law-of-cosines form)."""
    a = FEMUR_ATTACH_R_M
    mag_a = math.hypot(HIP_ANCHOR_U_M, HIP_ANCHOR_D_M)
    beta = math.atan2(HIP_ANCHOR_D_M, HIP_ANCHOR_U_M)
    cos_arg = (a * a + mag_a * mag_a - _f(length) ** 2) / (2.0 * a * mag_a)
    return beta + np.arccos(np.clip(cos_arg, -1.0, 1.0))


def dLh_dth(th):
    """SIGNED dL_hip/dtheta_h (m/rad); > 0 over the ROM (rod extends as the femur goes down)."""
    th = _f(th)
    pu = FEMUR_ATTACH_R_M * np.cos(th) - HIP_ANCHOR_U_M
    pd = FEMUR_ATTACH_R_M * np.sin(th) - HIP_ANCHOR_D_M
    return (pu * (-FEMUR_ATTACH_R_M * np.sin(th)) + pd * (FEMUR_ATTACH_R_M * np.cos(th))) / np.sqrt(pu * pu + pd * pd)


# ===================================================================================== knee linkage
def _knee_vec(th, tk):
    th, tk = _f(th), _f(tk)
    phi = th + tk
    lu = FEMUR_LEN_M * np.cos(th) + KNEE_LEVER_M * np.sin(phi)
    ld = FEMUR_LEN_M * np.sin(th) - KNEE_LEVER_M * np.cos(phi)
    return lu - KNEE_ANCHOR_U_M, ld - KNEE_ANCHOR_D_M


def knee_len(th, tk):
    """Knee rod length (m) at hip ``th`` and knee ``tk`` (left convention)."""
    ru, rd = _knee_vec(th, tk)
    return np.sqrt(ru * ru + rd * rd)


def knee_angle(length, th):
    """Knee angle (left convention) that gives rod ``length`` at hip ``th`` (the linkage's closed form)."""
    th = _f(th)
    ku = FEMUR_LEN_M * np.cos(th) - KNEE_ANCHOR_U_M
    kd = FEMUR_LEN_M * np.sin(th) - KNEE_ANCHOR_D_M
    mag_w = np.sqrt(ku * ku + kd * kd)
    psi = np.arctan2(kd, ku)
    r = KNEE_LEVER_M
    sin_arg = (_f(length) ** 2 - mag_w * mag_w - r * r) / (2.0 * r * mag_w)
    return psi + np.arcsin(np.clip(sin_arg, -1.0, 1.0)) - th


def dLk_dtk(th, tk):
    """SIGNED dL_knee/dtheta_k at fixed hip (m/rad); > 0 on the operating branch (folding extends the knee rod)."""
    phi = _f(th) + _f(tk)
    ru, rd = _knee_vec(th, tk)
    return (ru * KNEE_LEVER_M * np.cos(phi) + rd * KNEE_LEVER_M * np.sin(phi)) / np.sqrt(ru * ru + rd * rd)


def dLk_dth(th, tk):
    """SIGNED dL_knee/dtheta_h at fixed knee (m/rad) -- the coupling (= ``crab_hex_linkage.knee_hip_coupling``)."""
    th = _f(th)
    phi = th + _f(tk)
    ru, rd = _knee_vec(th, tk)
    dlu = -FEMUR_LEN_M * np.sin(th) + KNEE_LEVER_M * np.cos(phi)
    dld = FEMUR_LEN_M * np.cos(th) + KNEE_LEVER_M * np.sin(phi)
    return (ru * dlu + rd * dld) / np.sqrt(ru * ru + rd * rd)


def hip_default() -> float:
    """Hip angle at mid-stroke (= ``crab_hex_linkage.hip_default_rad``)."""
    return float(hip_angle(0.5 * (HIP_LEN_MIN_M + HIP_LEN_MAX_M)))


def knee_default() -> float:
    """Left knee angle at mid-stroke at the default hip (= ``crab_hex_linkage.knee_default_left_rad``)."""
    return float(knee_angle(0.5 * (KNEE_LEN_MIN_M + KNEE_LEN_MAX_M), hip_default()))


TH0 = hip_default()
TK0 = knee_default()


# ===================================================================================== action layer
def angles_from_action(a_h, a_k, *, clamp_rods: bool = True):
    """The action term's TARGET angles (left convention) for raw hip / knee actions (``parkour_actions.py:345-376``):
    hip rod = clamp(L_h(default + 0.25 a_h)); knee rod = clamp(L_k(hip_default, knee_default + 0.25 a_k)), re-solved at
    the hip target. The USD joint stops are NOT applied here (:func:`action_limits` reports them)."""
    lh = hip_len(TH0 + ROD_SCALE * _f(a_h))
    lk = knee_len(TH0, TK0 + ROD_SCALE * _f(a_k))
    if clamp_rods:
        lh = np.clip(lh, HIP_LEN_MIN_M, HIP_LEN_MAX_M)
        lk = np.clip(lk, KNEE_LEN_MIN_M, KNEE_LEN_MAX_M)
    th = hip_angle(lh)
    return th, knee_angle(lk, th)


def action_from_angles(th, tk):
    """Inverse of :func:`angles_from_action` inside the rod windows: ``(a_h, a_k_left)``."""
    a_h = (_f(th) - TH0) / ROD_SCALE
    a_k = (knee_angle(knee_len(th, tk), TH0) - TK0) / ROD_SCALE
    return a_h, a_k


def action18(q_h, q_k):
    """Per-row 18-wide action (cams 0, hips ``q_h`` on all legs, knees ``SIDE * q_k``) for arrays of q."""
    q_h, q_k = np.atleast_1d(_f(q_h)), np.atleast_1d(_f(q_k))
    out = np.zeros(q_h.shape + (18,))
    out[..., 6:12] = q_h[..., None]
    out[..., 12:18] = q_k[..., None] * SIDE
    return out


# ===================================================================================== leg-plane kinematics / height
def toe_uz(th, tk):
    """Toe (footpad centre) in the leg plane: (u outboard of the yaw axis, z up from the body centre)."""
    th, tk = _f(th), _f(tk)
    phi = th + tk
    ku = PIVOT_U_M + FEMUR_LEN_M * np.cos(th)
    kz = PIVOT_Z_M - FEMUR_LEN_M * np.sin(th)
    return ku - TIBIA_LEN_M * np.sin(phi), kz - TIBIA_LEN_M * np.cos(phi)


def pad_drop(th, tk):
    """How far the footpad box's lowest corner sits below the toe centre (the box tilts with the tibia)."""
    phi = _f(th) + _f(tk)
    return PAD_HALF_AXIAL_M * np.abs(np.cos(phi)) + PAD_HALF_PERP_M * np.abs(np.sin(phi))


def h_geom(th, tk):
    """Rigid model root height above a flat floor for a leg pose (body level, toe pad corner on the floor)."""
    return -toe_uz(th, tk)[1] + pad_drop(th, tk)


def h_geom_flat_pad(th, tk):
    """The older reading (pad bottom a constant 0.02 below the toe centre): 1.056 m at the defaults. Comparison only."""
    return -toe_uz(th, tk)[1] + PAD_HALF_AXIAL_M


#: the constant anchor: settled sim height minus the rigid tilted-pad model at the default pose (h_model = h - H_OFFSET)
H_OFFSET = H_SETTLED - float(h_geom(TH0, TK0))
#: default toe reach from the yaw axis (u of the toe centre at the default pose)
U0 = float(toe_uz(TH0, TK0)[0])


def h_from_angles(th, tk):
    """Predicted sim root height (anchored) for target angles."""
    return h_geom(th, tk) + H_OFFSET


def h_from_action(a_h, a_k):
    th, tk = angles_from_action(a_h, a_k)
    return h_from_angles(th, tk)


def h_meas(root_z, terrain_z6):
    """The shared measured height: root z minus the mean terrain height under the six feet (``(..., 6)``)."""
    return _f(root_z) - np.mean(_f(terrain_z6), axis=-1)


# ===================================================================================== inverse kinematics
def ik_toe(u, z):
    """Hip / knee angles (left convention) that put the toe centre at leg-plane (u, z); the default (knee-outboard)
    elbow branch. Returns (th, tk, reachable) arrays."""
    u, z = np.broadcast_arrays(_f(u), _f(z))
    du, dz = u - PIVOT_U_M, z - PIVOT_Z_M
    d2 = du * du + dz * dz
    d = np.sqrt(d2)
    reachable = (d <= FEMUR_LEN_M + TIBIA_LEN_M) & (d >= abs(TIBIA_LEN_M - FEMUR_LEN_M))
    a = (FEMUR_LEN_M ** 2 - TIBIA_LEN_M ** 2 + d2) / (2.0 * d)
    hgt = np.sqrt(np.clip(FEMUR_LEN_M ** 2 - a * a, 0.0, None))
    mu, mz = PIVOT_U_M + a * du / d, PIVOT_Z_M + a * dz / d
    # two knee candidates; the default branch is the one with the knee ABOVE / outboard of the pivot->toe line
    # (pivot->toe points down-and-in; the knee sits on the +u side of that line: rotate (du, dz) by +90 deg -> (-dz, du))
    ku = mu + hgt * (-dz) / d
    kz = mz + hgt * du / d
    th = np.arctan2(-(kz - PIVOT_Z_M), ku - PIVOT_U_M)
    phi = np.arctan2(-(u - ku), -(z - kz))
    return th, phi - th, reachable


def stance_angles(h, u=None, *, iters: int = 30):
    """Target angles whose anchored height is ``h`` with the toe centre at reach ``u`` (default :data:`U0`). The pad
    drop depends on the tibia tilt, so the toe z is solved by a fixed-point iteration (converges to < 1e-12 m)."""
    h = _f(h)
    u = U0 if u is None else u
    h_model = h - H_OFFSET
    z = -(h_model - PAD_HALF_AXIAL_M)
    th = tk = None
    for _ in range(iters):
        th, tk, ok = ik_toe(u, z)
        z = -(h_model - pad_drop(th, tk))
    th, tk, ok = ik_toe(u, z)
    return th, tk, ok


def q_raw(h, u=None):
    th, tk, _ = stance_angles(h, u)
    return action_from_angles(th, tk)


_Q_ANCHOR = tuple(float(x) for x in q_raw(H_SETTLED))


def q(h, u=None):
    """The posture map ``(a_hip, a_knee_left)`` in action units, anchored so ``q(1.064) == (0.0, 0.0)`` exactly
    (the solver residual at the anchor, ~1e-15, is subtracted). Not clamped: callers clamp ``h`` to [h_min, h_max]
    first (:func:`clamp_h`); the result then lies far inside +-RAW_CLIP (``facts.json`` q table)."""
    a_h, a_k = q_raw(h, u)
    if u is None:
        a_h, a_k = a_h - _Q_ANCHOR[0], a_k - _Q_ANCHOR[1]
    return np.clip(a_h, -RAW_CLIP, RAW_CLIP), np.clip(a_k, -RAW_CLIP, RAW_CLIP)


def q_table(h_grid):
    """``(h, a_hip, a_knee_left)`` rows for a monotone grid (for a torch-side ``interp`` in the build)."""
    h_grid = _f(h_grid)
    a_h, a_k = q(h_grid)
    return np.stack([h_grid, a_h, a_k], axis=-1)


def q_interp(h, table):
    """Linear interpolation in a :func:`q_table` (numpy); exact at the grid nodes (1.064 must be a node)."""
    t = np.asarray(table, dtype=np.float64)
    return np.interp(_f(h), t[:, 0], t[:, 1]), np.interp(_f(h), t[:, 0], t[:, 2])


def clamp_h(h, h_min, h_max):
    return np.clip(_f(h), float(h_min), float(h_max))


def encode_height(h_cmd):
    """Stand-net input column 15 (PLAN H2)."""
    return (_f(h_cmd) - H_SETTLED) / HEIGHT_OBS_SCALE


def rate_limit_step(h_state, h_cmd, dt, rate=HEIGHT_RATE_M_S):
    """One step of the h_cmd rate limiter."""
    step = float(rate) * float(dt)
    return _f(h_state) + np.clip(_f(h_cmd) - _f(h_state), -step, step)


# ===================================================================================== limits
def action_limits() -> dict:
    """Hip / knee joint stops and rod windows expressed in action units (left convention)."""
    lh_lo_th = float(hip_angle(HIP_LEN_MIN_M))
    lh_hi_th = float(hip_angle(HIP_LEN_MAX_M))
    k_lo_rod = float(knee_angle(KNEE_LEN_MIN_M, TH0))
    k_hi_rod = float(knee_angle(KNEE_LEN_MAX_M, TH0))
    return {
        "hip_stop_lo_action": (HIP_STOP_LO - TH0) / ROD_SCALE,
        "hip_stop_hi_action": (HIP_STOP_HI - TH0) / ROD_SCALE,
        "hip_rod_min_action": (lh_lo_th - TH0) / ROD_SCALE,
        "hip_rod_max_action": (lh_hi_th - TH0) / ROD_SCALE,
        "hip_rod_min_angle_rad": lh_lo_th, "hip_rod_max_angle_rad": lh_hi_th,
        "knee_rod_min_action_left": (k_lo_rod - TK0) / ROD_SCALE,
        "knee_rod_max_action_left": (k_hi_rod - TK0) / ROD_SCALE,
        "knee_stop_lo_action_left_at_default_hip": (KNEE_STOP_LO - TK0) / ROD_SCALE,
        "knee_stop_hi_action_left_at_default_hip": (KNEE_STOP_HI - TK0) / ROD_SCALE,
        "raw_clip": RAW_CLIP,
        "note": "hip: target = default + 0.25 a -> stop / rod-window angles map 1:1; knee: the action is a rod length "
                "read at the DEFAULT hip, so the knee rod window maps to fixed knee actions, while a knee joint stop is "
                "a different action at every hip (the 'at_default_hip' values only)",
    }


def hip_room_rad(h):
    """Room between the commanded stance hip and the -45 deg stop (rad)."""
    th, _, _ = stance_angles(h)
    return th - HIP_STOP_LO


# ===================================================================================== re-step clearance
def relift_toe_rise(h, *, knee_lift: float = 0.0, hip_lift: float = R_LIFT):
    """Toe rise (m) when a stance leg at q(h) is commanded q + (hip_lift, knee_lift) with the body held: ``knee_lift`` =
    0 holds the knee rod command (hip-only lift); ``knee_lift`` = -rho*side*hip_lift (= +0.5 left-convention) is R's
    D3 LIFT / HOME rod target. Measured at the footpad's lowest corner."""
    a_h, a_k = q(h)
    th0, tk0 = angles_from_action(a_h, a_k)
    th1, tk1 = angles_from_action(a_h + hip_lift, a_k + knee_lift)
    low0 = toe_uz(th0, tk0)[1] - pad_drop(th0, tk0)
    low1 = toe_uz(th1, tk1)[1] - pad_drop(th1, tk1)
    return low1 - low0


# ===================================================================================== statics
def foot_load_n(n_loaded: int = 6) -> float:
    """Even share of the whole robot's weight per loaded foot (N)."""
    return TOTAL_MASS_KG * G / float(n_loaded)


def stance_joint_torques(th, tk, f_toe, *, leg_weight: bool = True, f_u=0.0):
    """Static ACTUATOR joint torques (left convention, N*m) holding a leg at (th, tk) under a toe reaction
    ``(f_u, f_toe)`` (leg plane: u outboard, z up; the floor pushes the toe up with ``f_toe``) and, with ``leg_weight``,
    the femur / tibia / footpad gravity. tau_act = -J_toe^T F_toe + sum_links m g dz_com/dtheta."""
    th, tk = _f(th), _f(tk)
    phi = th + tk
    # toe Jacobian: d(u, z)/d(th, tk)
    du_dth = -FEMUR_LEN_M * np.sin(th) - TIBIA_LEN_M * np.cos(phi)
    du_dtk = -TIBIA_LEN_M * np.cos(phi)
    dz_dth = -FEMUR_LEN_M * np.cos(th) + TIBIA_LEN_M * np.sin(phi)
    dz_dtk = TIBIA_LEN_M * np.sin(phi)
    tau_h = -(f_u * du_dth + f_toe * dz_dth)
    tau_k = -(f_u * du_dtk + f_toe * dz_dtk)
    if leg_weight:
        # femur COM: z = pz - c_f sin th ; tibia COM: z = pz - Lf sin th - c_t cos phi ; pad at the toe
        c_f, c_t = FEMUR_COM_FROM_PIVOT_M, TIBIA_COM_BELOW_KNEE_M
        tau_h = tau_h + G * (FEMUR_MASS_KG * (-c_f * np.cos(th))
                             + TIBIA_MASS_KG * (-FEMUR_LEN_M * np.cos(th) + c_t * np.sin(phi))
                             + PAD_MASS_KG * dz_dth)
        tau_k = tau_k + G * (TIBIA_MASS_KG * (c_t * np.sin(phi)) + PAD_MASS_KG * dz_dtk)
    return tau_h, tau_k


def rod_forces(th, tk, tau_h, tau_k):
    """COUPLED conversion of joint torques to rod forces (N; + = the rod pushes / extends, - = it pulls):
    [tau_h, tau_k] = J^T [F_h, F_k] with J = [[dLh/dth, 0], [dLk/dth, dLk/dtk]], so
    F_k = tau_k / (dLk/dtk) and F_h = (tau_h - (dLk/dth) F_k) / (dLh/dth). Virtual work: tau . dtheta = F . dL."""
    fk = _f(tau_k) / dLk_dtk(th, tk)
    fh = (_f(tau_h) - dLk_dth(th, tk) * fk) / dLh_dth(th)
    return fh, fk


def rod_forces_uncoupled(th, tk, tau_h, tau_k):
    """The same without the knee-rod coupling term (F_h = tau_h / (dLh/dth)) -- comparison only."""
    return _f(tau_h) / dLh_dth(th), _f(tau_k) / dLk_dtk(th, tk)


def static_rod_forces_at_h(h, *, leg_weight: bool = True, n_loaded: int = 6, scale_h: float = 1.0,
                           scale_k: float = 1.0):
    """Modelled per-leg static rod forces (N) at the posture q(h) (target angles), even load share. ``scale_*`` are the
    measurement calibration factors (measured / model at 1.064, ``facts.json``)."""
    th, tk, _ = stance_angles(h)
    tau_h, tau_k = stance_joint_torques(th, tk, foot_load_n(n_loaded), leg_weight=leg_weight)
    fh, fk = rod_forces(th, tk, tau_h, tau_k)
    return fh * scale_h, fk * scale_k
