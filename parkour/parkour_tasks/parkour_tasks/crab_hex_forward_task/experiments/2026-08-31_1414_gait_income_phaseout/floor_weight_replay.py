"""Offline replay of the REMAINING reward stack at gait-income floor weights (PLAN G pre-GPU gate).

Question under test: with clock/apex/airtime/stride at epsilon, does any known
degenerate-gait family out-earn a healthy gait in the remaining economy? This is the
hacking surface the phase-out re-opens (the historical #1 hack is tracking-abandonment
creep; the family most favored by a tracking-dominated economy is the v3 skate).

Replays the economically live remaining terms directly from saved gait-eval npz traces
(same fixture set as the tripod-era replay gate). Formulas mirror
parkour_isaaclab/envs/mdp/rewards.py + Isaac Lab stock exp-tracking; weights are the
graduated r1-window env stack. Terms not replayable from traces and economically small
in the r1 income table are listed in NOT_REPLAYED.

CAVEAT (recorded in CHANGELOG): fixtures are old-plant traces (2026-08-10, position
era). Absolute incomes are not cross-plant transferable; the gate checks the
healthy-vs-degenerate ORDERING of the floor-weight economy, which is behavior-level.

Usage: python floor_weight_replay.py
"""
import glob
import os

import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "2026-08-10_0058_tripod_stability"))
RUNS = {
    "H2000-healthy": f"{BASE}/healthy_refs/gait_eval_h2000",
    "H3000-healthy": f"{BASE}/healthy_refs/gait_eval_h3000",
    "v1-lunge": f"{BASE}/fromscratch_tripod_reward_w0.15/gait_eval",
    "v2-tiprock": f"{BASE}/fromscratch_tripod_v2_w0.15/gait_eval_midtrain",
    "v3-skate": f"{BASE}/fromscratch_tripod_v3_w0.15_short/gait_eval_3000",
    "v3b-drag": f"{BASE}/fromscratch_tripod_v3b_slide_short/gait_eval_3000",
}

# Graduated r1-window weights (campaign env stack); gait income at eps ~ 0 by design.
W_TRACK_LIN = 1.25       # track_lin_vel_xy_exp, sigma^2 = KRABBY_TRACK_SIGMA2 = 0.1
SIGMA2_LIN = 0.1
W_TRACK_ANG = 1.0        # track_ang_vel_z_exp, std^2 = 0.25
SIGMA2_ANG = 0.25
W_TRACK_L1 = -1.0        # KRABBY_TRACK_L1_W
W_PROGRESS = 0.60        # forward progress, clamp [0, 1.05], gate ||cmd||>0.12
PROGRESS_CAP = 1.05
W_LIN_VEL_Y = -3.0       # lateral vel sq, gated no-lateral-cmd & forward-cmd
W_LIN_VEL_Z = -0.15      # vz^2
W_ORIENT = -0.7          # sum sq gravity_b xy
W_ACTION_RATE = -0.3     # sum sq(a_t - a_{t-1})
W_DELTA_TORQUE = -1e-6   # sum sq(tau_t - tau_{t-1})
W_FOOT_IDLE = -0.12      # relu(idle-90) summed, gated forward
IDLE_MAX = 90
W_EXCESS_CONTACT = -0.20  # relu(nfeet-4), gated forward
FORCE_THRESH = 1.0        # N, same engagement threshold as the tripod replay gate

NOT_REPLAYED = "tibia_deviation (-0.018/step live: no joint_pos in traces), yaw/goal/clearance/edge/stumble (flat traces, ~0 in r1 window), collision (~-0.010)"


def gravity_b_xy_sq(quat_wxyz):
    """sum of squared xy components of gravity projected into the body frame."""
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    # world gravity direction (0,0,-1) rotated into body frame: R^T @ g
    gx = -(2.0 * (x * z - w * y))
    gy = -(2.0 * (y * z + w * x))
    return gx * gx + gy * gy


def replay_episode(d):
    contact = d["foot_force_norm"] > FORCE_THRESH          # (T, 6)
    cmd = d["cmd_applied"].astype(np.float64)              # (T, 3) vx, vy, wz
    v_b = d["root_lin_vel_b"].astype(np.float64)           # (T, 3)
    w_b = d["root_ang_vel_b"].astype(np.float64)           # (T, 3)
    quat = d["root_quat_w"].astype(np.float64)             # (T, 4) wxyz
    actions = d["actions"].astype(np.float64)              # (T, A)
    torque = d["applied_torque"].astype(np.float64)        # (T, J)
    steady = d["steady_mask"].astype(bool)
    dt = float(d["dt"])
    T = contact.shape[0]

    err_xy_sq = np.sum((cmd[:, :2] - v_b[:, :2]) ** 2, axis=1)
    track_lin = np.exp(-err_xy_sq / SIGMA2_LIN)
    track_ang = np.exp(-((cmd[:, 2] - w_b[:, 2]) ** 2) / SIGMA2_ANG)
    track_l1 = np.linalg.norm(cmd[:, :2] - v_b[:, :2], axis=1)

    cmd_norm = np.linalg.norm(cmd[:, :2], axis=1)
    cmd_active = cmd_norm > 0.12
    dir_xy = cmd[:, :2] / (cmd_norm[:, None] + 1e-8)
    progress = np.clip(np.sum(v_b[:, :2] * dir_xy, axis=1), 0.0, PROGRESS_CAP) * cmd_active

    no_lat_cmd = np.abs(cmd[:, 1]) < 0.05
    fwd_cmd = np.abs(cmd[:, 0]) > 0.12
    lin_vel_y = (v_b[:, 1] ** 2) * (no_lat_cmd & fwd_cmd)
    lin_vel_z = v_b[:, 2] ** 2
    orient = gravity_b_xy_sq(quat)

    action_rate = np.zeros(T)
    action_rate[1:] = np.sum((actions[1:] - actions[:-1]) ** 2, axis=1)
    delta_tau = np.zeros(T)
    delta_tau[1:] = np.sum((torque[1:] - torque[:-1]) ** 2, axis=1)

    idle = np.zeros(6)
    foot_idle = np.zeros(T)
    for t in range(T):
        idle = np.where(contact[t], 0.0, idle + 1.0)
        foot_idle[t] = np.sum(np.maximum(idle - IDLE_MAX, 0.0))
    excess = np.maximum(contact.sum(axis=1) - 4.0, 0.0)

    per_step = {
        "track_lin": W_TRACK_LIN * track_lin,
        "track_ang": W_TRACK_ANG * track_ang,
        "track_L1": W_TRACK_L1 * track_l1,
        "progress": W_PROGRESS * progress,
        "lin_vel_y": W_LIN_VEL_Y * lin_vel_y,
        "lin_vel_z": W_LIN_VEL_Z * lin_vel_z,
        "orient": W_ORIENT * orient,
        "action_rate": W_ACTION_RATE * action_rate,
        "delta_torque": W_DELTA_TORQUE * delta_tau,
        "foot_idle": W_FOOT_IDLE * foot_idle * fwd_cmd,
        "excess_contact": W_EXCESS_CONTACT * excess * fwd_cmd,
    }
    seconds = steady.sum() * dt
    sums = {k: float(np.sum(v[steady]) * dt) for k, v in per_step.items()}
    return sums, seconds


# Action-stream regularizers are era-nontransferable (fixtures carry position-era
# action/torque streams; the current plant uses velocity actions with different
# statistics) AND are unchanged by the phase-out, so they cancel in any
# candidate-vs-control comparison. The gate therefore ranks the BEHAVIORAL economy;
# the regularizer columns are printed separately for the record.
BEHAVIORAL = ("track_lin", "track_ang", "track_L1", "progress", "lin_vel_y",
              "lin_vel_z", "orient", "foot_idle", "excess_contact")
ACTION_STREAM = ("action_rate", "delta_torque")


def main():
    behav_net = {}
    detail = {}
    for name, run_dir in RUNS.items():
        files = sorted(glob.glob(os.path.join(run_dir, "**/raw/episode_*.npz"), recursive=True))
        if not files:
            print(f"{name}: NO NPZ FOUND under {run_dir}")
            continue
        total = {}
        secs = 0.0
        for f in files:
            sums, s = replay_episode(np.load(f))
            secs += s
            for k, v in sums.items():
                total[k] = total.get(k, 0.0) + v
        mins = secs / 60.0
        detail[name] = {k: v / mins for k, v in total.items()}
        behav_net[name] = sum(detail[name][t] for t in BEHAVIORAL)

    header = f"{'run':<16}" + "".join(f"{t:>13s}" for t in BEHAVIORAL) + f"{'BEHAV NET/min':>14s}"
    print(header)
    for name in RUNS:
        if name not in behav_net:
            continue
        row = f"{name:<16}" + "".join(f"{detail[name][t]:>13.3f}" for t in BEHAVIORAL)
        print(row + f"{behav_net[name]:>14.3f}")
    print("\naction-stream regularizers (era-nontransferable, phase-out-invariant; excluded from gate):")
    for name in RUNS:
        if name not in behav_net:
            continue
        print(f"  {name:<16}" + "".join(f"{t} {detail[name][t]:>12.3f}  " for t in ACTION_STREAM))

    healthy = min(v for k, v in behav_net.items() if "healthy" in k)
    degen_name, degen = max(((k, v) for k, v in behav_net.items() if "healthy" not in k),
                            key=lambda kv: kv[1])
    print(f"\nbehavioral healthy-min {healthy:.3f}/min vs degenerate-max {degen:.3f}/min ({degen_name})")
    print(f"not replayed: {NOT_REPLAYED}")
    ok = healthy > degen
    print("GATE:", "PASS — healthy gait out-earns every degenerate family at floor weights"
          if ok else f"FAIL — {degen_name} out-earns healthy under the floor-weight stack")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
