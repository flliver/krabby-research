# SPDX-License-Identifier: BSD-3-Clause
"""PLAN E Phase 0: scripted-gait feasibility probe on the collision-fixed plant.

Drives the mechanism's DESIGNED gait open-loop — constant cam angular velocity with the
two tripod sets pi out of phase — and measures whether the as-built machine (slew-limited
linkage actuators, real leg collision) locomotes at all. No policy, no reward.

Phase offsetting is done physically, not by state-writing: tripod set B starts spinning
first; set A holds for Delta_t = pi/omega and then ramps to the same omega. Identical
acceleration transients cancel, locking a pi offset at steady state without ever
teleporting a cam (which would snap the slaved yaw targets).

Per omega segment: settle -> stagger-start -> hold ~20 s. Records root kinematics, cam
phases, per-foot contact forces, and terminations to an npz (the on-plant reference
dataset for replay gates / RSI / AMP), plus one continuous video across all segments.

Run (from krabby-research/parkour):
  <venv-python> ../parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-22_1200_gait_formation_v2/scripted_gait_probe.py --headless
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Crab-Hex-Flat-Walk-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--settle_steps", type=int, default=150)
parser.add_argument("--hold_steps", type=int, default=1000, help="steps at steady offset per omega (20 s at 50 Hz)")
parser.add_argument("--omegas", type=str, default="0.5,0.75,1.0", help="cam speeds as fractions of CAM_VEL_SCALE (pi rad/s)")
parser.add_argument("--out_dir", type=str, default=str(Path(__file__).parent))
_PARKOUR_ROOT = Path("/home/nickmagus/krabby/krabby-research/parkour")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, str(_PARKOUR_ROOT))
sys.path.insert(0, str(_PARKOUR_ROOT / "parkour_tasks"))

import math

import gymnasium as gym
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg

import parkour_tasks  # noqa: F401

TRIPOD_A = ("FL", "MR", "RL")
TRIPOD_B = ("FR", "ML", "RR")
FOOT_NAMES = ["FL_Footpad", "FR_Footpad", "ML_Footpad", "MR_Footpad", "RL_Footpad", "RR_Footpad"]


def main() -> None:
    omegas = [float(x) for x in args_cli.omegas.split(",")]
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    # Upper bound: per segment = settle + stagger (<= 2 s / 0.02) + hold.
    total = sum(args_cli.settle_steps + int(math.ceil(math.pi / (w * math.pi) / 0.02)) + args_cli.hold_steps for w in omegas)
    env = gym.wrappers.RecordVideo(
        env, video_folder=str(Path(args_cli.out_dir) / "videos"),
        step_trigger=lambda step: step == 0, video_length=total, disable_logger=True,
    )
    env.reset()
    uenv = env.unwrapped
    device = uenv.device
    action_dim = env.action_space.shape[1]
    action_term = uenv.action_manager.get_term("joint_pos")
    names = action_term._joint_names
    cam_cols_A = [i for i, n in enumerate(names) if "CamShaft" in n and n.split("_")[0] in TRIPOD_A]
    cam_cols_B = [i for i, n in enumerate(names) if "CamShaft" in n and n.split("_")[0] in TRIPOD_B]
    robot = uenv.scene["robot"]
    cam_joint_ids, cam_joint_names = robot.find_joints([".*_Body_CamShaft_RevoluteJoint"], preserve_order=True)
    contact = uenv.scene.sensors["contact_forces"]
    foot_ids, _ = contact.find_bodies(FOOT_NAMES, preserve_order=True)
    dt = uenv.step_dt
    print(f"[probe] cam cols A={cam_cols_A} B={cam_cols_B} | joints={cam_joint_names} | dt={dt}")

    with torch.inference_mode():
        for w in omegas:
            omega = w * math.pi
            stagger = int(round((math.pi / omega) / dt))
            act = torch.zeros((args_cli.num_envs, action_dim), device=device)
            env.reset()
            rec: dict[str, list] = {k: [] for k in (
                "root_lin_vel_b", "root_ang_vel_b", "root_pos_w", "root_quat_w",
                "cam_pos", "cam_vel", "joint_pos", "joint_vel", "foot_force_N", "done")}
            for _ in range(args_cli.settle_steps):
                env.step(act)
            # B first, A joins after half a period -> pi offset at equal speeds.
            for c in cam_cols_B:
                act[:, c] = w
            for _ in range(stagger):
                env.step(act)
            for c in cam_cols_A:
                act[:, c] = w
            for t in range(args_cli.hold_steps):
                _, _, terminated, truncated, _ = env.step(act)
                d = robot.data
                rec["root_lin_vel_b"].append(d.root_lin_vel_b[0].cpu().numpy().copy())
                rec["root_ang_vel_b"].append(d.root_ang_vel_b[0].cpu().numpy().copy())
                rec["root_pos_w"].append(d.root_pos_w[0].cpu().numpy().copy())
                rec["root_quat_w"].append(d.root_quat_w[0].cpu().numpy().copy())
                rec["cam_pos"].append(d.joint_pos[0, cam_joint_ids].cpu().numpy().copy())
                rec["cam_vel"].append(d.joint_vel[0, cam_joint_ids].cpu().numpy().copy())
                rec["joint_pos"].append(d.joint_pos[0].cpu().numpy().copy())
                rec["joint_vel"].append(d.joint_vel[0].cpu().numpy().copy())
                rec["foot_force_N"].append(
                    contact.data.net_forces_w[0, foot_ids].norm(dim=-1).cpu().numpy().copy())
                rec["done"].append(bool(terminated[0] or truncated[0]))
            arrays = {k: np.asarray(v) for k, v in rec.items()}
            arrays["dt"] = np.asarray(dt)
            arrays["omega_rad_s"] = np.asarray(omega)
            arrays["cam_joint_names"] = np.asarray(cam_joint_names)
            out = Path(args_cli.out_dir) / f"scripted_gait_w{w:.2f}.npz"
            np.savez_compressed(out, **arrays)
            # Headline per segment, printed for the log parser.
            vx = arrays["root_lin_vel_b"][:, 0]
            done = arrays["done"]
            fell = bool(done.any())
            first_done = int(np.argmax(done)) if fell else len(done)
            print(f"[probe] omega={omega:.3f} rad/s ({2*math.pi/omega:.2f}s cycle): "
                  f"mean vx {float(vx[:first_done].mean()):+.3f} m/s over {first_done} steps"
                  f" | fell={fell}", flush=True)
    env.close()
    print("[probe] DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
