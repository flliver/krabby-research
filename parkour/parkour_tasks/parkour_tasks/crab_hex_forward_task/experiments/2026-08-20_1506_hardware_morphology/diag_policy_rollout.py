# SPDX-License-Identifier: BSD-3-Clause
"""Roll out a trained checkpoint and capture the state trajectory before each termination.

Diagnoses the smoke-test 2-second wall: logs pitch/roll/root_z, cam shaft speeds, pitch
joint angles vs targets, and per-foot contact each step; on termination, prints the
preceding window so the tip mechanism is visible.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Crab-Hex-Flat-Walk-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=600)
# NOTE: --checkpoint comes from cli_args.add_rsl_rl_args below.
_PARKOUR_ROOT = Path("/home/nickmagus/krabby/krabby-research/parkour")
sys.path.insert(0, str(_PARKOUR_ROOT / "scripts" / "rsl_rl"))
import cli_args as _cli_args  # isort: skip

_cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.experiment_name = "crab_hex_flat_walk"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, str(_PARKOUR_ROOT))
sys.path.insert(0, str(_PARKOUR_ROOT / "parkour_tasks"))

import gymnasium as gym
import torch
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
from isaaclab_tasks.utils import parse_env_cfg

import parkour_tasks  # noqa: F401
from scripts.rsl_rl.runner_factory import agent_cfg_to_train_dict, make_on_policy_runner
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper

FOOT_NAMES = ["FL_Footpad", "FR_Footpad", "ML_Footpad", "MR_Footpad", "RL_Footpad", "RR_Footpad"]


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = _cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env = gym.make(args_cli.task, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
    foot_ids, _ = contact_sensor.find_bodies(FOOT_NAMES, preserve_order=True)
    wrapped = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = make_on_policy_runner(
        wrapped, agent_cfg_to_train_dict(agent_cfg), log_dir=None, device=agent_cfg.device
    )
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    estimator = runner.get_estimator_inference_policy(device=wrapped.device)
    est = agent_cfg.to_dict()["estimator"]
    num_prop, num_scan, num_priv = est["num_prop"], est["num_scan"], est["num_priv_explicit"]

    shaft_ids, _ = robot.find_joints([".*_Body_CamShaft_RevoluteJoint"], preserve_order=True)
    pitch_ids, _ = robot.find_joints([".*_Hip_Femur_RevoluteJoint"], preserve_order=True)
    knee_ids, _ = robot.find_joints([".*_Femur_Tibia_RevoluteJoint"], preserve_order=True)

    obs, extras = wrapped.get_observations()
    window: list[dict] = []
    episodes = 0
    for step in range(args_cli.steps):
        with torch.inference_mode():
            obs[:, num_prop + num_scan : num_prop + num_scan + num_priv] = estimator.inference(
                obs[:, :num_prop]
            )
            actions = policy(obs, hist_encoding=True)
        obs, _, dones, extras = wrapped.step(actions)
        roll, pitch, yaw = euler_xyz_from_quat(robot.data.root_quat_w)
        feet = contact_sensor.data.net_forces_w[0, foot_ids].norm(dim=-1)
        row = {
            "t": step,
            "pitch": round(float(wrap_to_pi(pitch)[0]), 3),
            "roll": round(float(wrap_to_pi(roll)[0]), 3),
            "z": round(float(robot.data.root_pos_w[0, 2]), 3),
            "vx": round(float(robot.data.root_lin_vel_b[0, 0]), 3),
            "cam_w": [round(float(v), 2) for v in robot.data.joint_vel[0, shaft_ids]],
            "hip": [round(float(v), 3) for v in robot.data.joint_pos[0, pitch_ids]],
            "knee": [round(float(v), 3) for v in robot.data.joint_pos[0, knee_ids]],
            "feet_N": [round(float(v)) for v in feet],
        }
        window.append(row)
        if len(window) > 30:
            window.pop(0)
        if bool(dones[0]):
            episodes += 1
            print(f"=== TERMINATION #{episodes} at step {step} — last {len(window)} steps ===", flush=True)
            for r in window:
                print(json.dumps(r), flush=True)
            window.clear()
            if episodes >= 3:
                break
    if episodes == 0:
        print("no termination within horizon; last rows:", flush=True)
        for r in window[-10:]:
            print(json.dumps(r), flush=True)
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
