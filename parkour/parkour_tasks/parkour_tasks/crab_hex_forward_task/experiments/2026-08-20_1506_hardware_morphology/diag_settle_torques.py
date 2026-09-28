# SPDX-License-Identifier: BSD-3-Clause
"""Diagnose the slow zero-action pitch tip: log per-joint state/targets/torques while settling."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Crab-Hex-Flat-Walk-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=250)
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
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
from isaaclab_tasks.utils import parse_env_cfg

import parkour_tasks  # noqa: F401


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]
    env.reset()
    zero = torch.zeros(env.action_space.shape, device=device)

    names = robot.joint_names
    watch = [i for i, n in enumerate(names) if ("Hip_Femur" in n or "Femur_Tibia" in n or "Body_Hip" in n)]
    print("watched:", [names[i] for i in watch], flush=True)
    with torch.inference_mode():
        for step in range(args_cli.steps):
            env.step(zero)
            if step % 25 == 0 or step == args_cli.steps - 1:
                roll, pitch, yaw = euler_xyz_from_quat(robot.data.root_quat_w)
                q = robot.data.joint_pos[0]
                qd = robot.data.joint_vel[0]
                tau = robot.data.applied_torque[0]
                tgt = robot.data.joint_pos_target[0]
                row = {
                    "step": step,
                    "pitch": round(float(wrap_to_pi(pitch)[0]), 4),
                    "root_z": round(float(robot.data.root_pos_w[0, 2]), 4),
                }
                for i in watch:
                    row[names[i]] = (
                        round(float(q[i]), 4),
                        round(float(tgt[i]), 4),
                        round(float(tau[i]), 1),
                    )
                print(json.dumps(row), flush=True)
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
