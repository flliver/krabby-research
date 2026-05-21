# SPDX-License-Identifier: BSD-3-Clause
#
# Parkour fork of Isaac Lab scripts/environments/zero_agent.py.
# Registers parkour_tasks so crab / parkour gym ids resolve.

"""Run a parkour env with zero actions (hold default joint targets only; no checkpoint)."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Zero-action agent for parkour_tasks environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Crab-Hex-Teacher-Play-v0",
    help="Gym task id (e.g. Isaac-Crab-Hex-Teacher-Play-v0).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
from pathlib import Path

# Running this file adds ``parkour/scripts`` to sys.path, which shadows pip ``rsl_rl-lib``
# with ``parkour/scripts/rsl_rl`` and breaks ``import parkour_tasks`` (via isaaclab_rl).
_scripts_dir = Path(__file__).resolve().parent
_parkour_root = _scripts_dir.parent
for _p in (str(_scripts_dir), str(_parkour_root / "parkour_tasks")):
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(_parkour_root))
sys.path.insert(0, str(_parkour_root / "parkour_tasks"))

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import parkour_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[INFO] task={args_cli.task}")
    print(f"[INFO] observation space={env.observation_space}")
    print(f"[INFO] action space={env.action_space}")
    env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
