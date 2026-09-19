# SPDX-License-Identifier: BSD-3-Clause
"""Record a checkpoint-free geometry review video: zero-action settle, then a scripted
cam sweep. For the user's visual sign-off of the vertical-plate hip correction."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Crab-Hex-Flat-Walk-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--settle_steps", type=int, default=150)
parser.add_argument("--sweep_steps", type=int, default=250)
parser.add_argument("--out_dir", type=str, default=str(Path(__file__).parent / "videos"))
_PARKOUR_ROOT = Path("/home/nickmagus/krabby/krabby-research/parkour")
sys.path.insert(0, str(_PARKOUR_ROOT / "scripts" / "rsl_rl"))
import cli_args as _cli_args  # isort: skip

_cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, str(_PARKOUR_ROOT))
sys.path.insert(0, str(_PARKOUR_ROOT / "parkour_tasks"))

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg

import parkour_tasks  # noqa: F401


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    total = args_cli.settle_steps + args_cli.sweep_steps
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=args_cli.out_dir,
        step_trigger=lambda step: step == 0,
        video_length=total,
        disable_logger=True,
    )
    env.reset()
    device = env.unwrapped.device
    action_dim = env.action_space.shape[1]
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    cam_cols = [i for i, n in enumerate(action_term._joint_names) if "CamShaft" in n]

    with torch.inference_mode():
        act = torch.zeros((args_cli.num_envs, action_dim), device=device)
        for _ in range(args_cli.settle_steps):
            env.step(act)
        for c in cam_cols:
            act[:, c] = 0.5
        for _ in range(args_cli.sweep_steps):
            env.step(act)
    env.close()
    print("video written under", args_cli.out_dir, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
