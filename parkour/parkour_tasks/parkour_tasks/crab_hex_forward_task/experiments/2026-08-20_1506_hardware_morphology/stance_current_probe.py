# SPDX-License-Identifier: BSD-3-Clause
"""Stance-detection study: motor-current proxy vs ground-truth foot contact.

The physical robot has no foot sensors; stance is to be inferred from actuator current.
This probe drives a scripted gait-like motion (cam spin + pitch/knee oscillation at rates
the rod-speed-limited linkage can actually follow), logs per leg per env-step:

  - the current-sense proxy exactly as the policy observation computes it
    (CrabHexParkourObservations._get_contact_fill), and
  - ground-truth footpad contact from the privileged sim ContactSensor,

then reports precision/recall/F1 over proxy thresholds, plus the worm-drive dead-zone
characterization: what fraction of true-stance time the proxy is blind because no
actuator on that leg is being driven (lead screws draw ~no current while parked).

Also records the settled root height above the terrain -- the measured value that sets
``ground_offset_from_root_m`` and validates spawn_z.

Run (from ~/krabby/krabby-research/parkour):
    OMNI_KIT_ACCEPT_EULA=yes .../isaac_venv/bin/python \
        ../parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-20_1506_hardware_morphology/stance_current_probe.py --headless
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Current-sense stance proxy vs true contact.")
parser.add_argument("--task", type=str, default="Isaac-Crab-Hex-Flat-Walk-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--settle_steps", type=int, default=200)
parser.add_argument("--gait_steps", type=int, default=800, help="Scripted-motion steps (16 s).")
parser.add_argument("--output", type=str, default=None)

_PARKOUR_ROOT = Path("/home/nickmagus/krabby/krabby-research/parkour")
sys.path.insert(0, str(_PARKOUR_ROOT / "scripts" / "rsl_rl"))
import cli_args as _cli_args  # isort: skip

_cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, str(_PARKOUR_ROOT))
sys.path.insert(0, str(_PARKOUR_ROOT / "parkour_tasks"))

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg

import parkour_tasks  # noqa: F401

FOOT_NAMES = ["FL_Footpad", "FR_Footpad", "ML_Footpad", "MR_Footpad", "RL_Footpad", "RR_Footpad"]
CONTACT_FORCE_N = 5.0  # ground-truth stance threshold


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]
    contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
    foot_ids, _ = contact_sensor.find_bodies(FOOT_NAMES, preserve_order=True)

    # Recompute the proxy with the same primitives the obs class uses.
    from parkour_tasks.crab_hex_forward_task.mdp import crab_hex_linkage as linkage
    from parkour_tasks.crab_hex_forward_task.mdp.observations import (
        _CURRENT_SENSE_GATE_FRACTION,
        _CURRENT_SENSE_MAX,
        _CURRENT_SENSE_NO_LOAD,
    )

    pitch_ids, pitch_names = robot.find_joints([".*_Hip_Femur_RevoluteJoint"], preserve_order=True)
    knee_ids, _ = robot.find_joints(
        [n.replace("_Hip_Femur_", "_Femur_Tibia_") for n in pitch_names], preserve_order=True
    )

    def current_proxy(action_term) -> torch.Tensor:
        tau = robot.data.applied_torque
        theta_h = robot.data.joint_pos[:, pitch_ids]
        theta_k = robot.data.joint_pos[:, knee_ids]
        ma_h = linkage.hip_moment_arm(theta_h).clamp_min(1e-4)
        ma_k = linkage.knee_moment_arm(theta_h, theta_k.abs()).clamp_min(1e-4)
        force_h = tau[:, pitch_ids].abs() / ma_h / linkage.HIP_FORCE_N
        force_k = tau[:, knee_ids].abs() / ma_k / linkage.KNEE_FORCE_N
        gate_h = (
            action_term.hip_rod_speed.abs() / (_CURRENT_SENSE_GATE_FRACTION * linkage.HIP_SPEED_M_S)
        ).clamp(0.0, 1.0)
        gate_k = (
            action_term.knee_rod_speed.abs() / (_CURRENT_SENSE_GATE_FRACTION * linkage.KNEE_SPEED_M_S)
        ).clamp(0.0, 1.0)
        cur_h = gate_h * (_CURRENT_SENSE_NO_LOAD + force_h).clamp(0.0, _CURRENT_SENSE_MAX)
        cur_k = gate_k * (_CURRENT_SENSE_NO_LOAD + force_k).clamp(0.0, _CURRENT_SENSE_MAX)
        return torch.maximum(cur_h, cur_k), gate_h, gate_k

    obs, _ = env.reset()
    action_dim = env.action_space.shape[1]
    zero = torch.zeros((args_cli.num_envs, action_dim), device=device)

    # Column layout of the action space (18): resolve by joint name via the action term.
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    names = action_term._joint_names
    cam_cols = [i for i, n in enumerate(names) if "CamShaft" in n]
    hip_cols = [i for i, n in enumerate(names) if "Hip_Femur" in n]
    knee_cols = [i for i, n in enumerate(names) if "Femur_Tibia" in n]

    with torch.inference_mode():
        for _ in range(args_cli.settle_steps):
            env.step(zero)

    terrain_z = 0.0
    root_z = float(robot.data.root_pos_w[0, 2].item()) - terrain_z
    settle_contact = contact_sensor.data.net_forces_w[0, foot_ids].norm(dim=-1)

    # Scripted gait-ish motion: cams at 60% speed, pitch/knee sine at 0.4 Hz within the
    # rod-speed envelope (hip rod full stroke takes ~7 s; use small amplitudes).
    env_dt = float(env.unwrapped.step_dt)
    proxies, truths, gates_any = [], [], []
    with torch.inference_mode():
        for i in range(args_cli.gait_steps):
            t = i * env_dt
            act = zero.clone()
            for c in cam_cols:
                act[:, c] = 0.6
            hip_a = 0.6 * math.sin(2.0 * math.pi * 0.4 * t)
            knee_a = 0.6 * math.sin(2.0 * math.pi * 0.4 * t + math.pi / 2.0)
            for c in hip_cols:
                act[:, c] = hip_a
            for c in knee_cols:
                act[:, c] = knee_a
            env.step(act)
            proxy, gate_h, gate_k = current_proxy(action_term)
            truth = contact_sensor.data.net_forces_w[:, foot_ids].norm(dim=-1) > CONTACT_FORCE_N
            proxies.append(proxy[0].clone())
            truths.append(truth[0].clone())
            gates_any.append(torch.maximum(gate_h, gate_k)[0].clone())

    P = torch.stack(proxies)  # (T, 6)
    T_ = torch.stack(truths)  # (T, 6) bool
    G = torch.stack(gates_any)  # (T, 6)

    report = {
        "settled_root_height_m": root_z,
        "settled_foot_forces_N": [round(float(v), 1) for v in settle_contact],
        "gait_steps": args_cli.gait_steps,
        "true_stance_fraction": float(T_.float().mean()),
        "dead_zone_fraction_of_stance": float(((G < 0.05) & T_).float().sum() / T_.float().sum()),
        "thresholds": {},
    }
    for thr in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
        pred = P > thr
        tp = float((pred & T_).sum())
        fp = float((pred & ~T_).sum())
        fn = float((~pred & T_).sum())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        report["thresholds"][str(thr)] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }
    # Driving-phase-only metrics (where the signal physically exists):
    driving = G > 0.05
    for thr in (0.1, 0.2):
        pred = (P > thr) & driving
        t_driving = T_ & driving
        tp = float((pred & t_driving).sum())
        fp = float((pred & ~T_).sum())
        fn = float((~pred & t_driving).sum())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        report["thresholds"][f"{thr}_driving_only"] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
        }

    out = args_cli.output or str(Path(__file__).parent / "stance_current_report.json")
    Path(out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
