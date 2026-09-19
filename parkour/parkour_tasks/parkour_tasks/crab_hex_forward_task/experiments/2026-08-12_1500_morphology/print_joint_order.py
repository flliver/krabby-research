"""One-shot: print the articulation joint order (for offline npz analysis)."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Crab-Hex-Flat-Walk-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
_PARKOUR_ROOT = Path("/home/nickmagus/krabby/krabby-research/parkour")
sys.path.insert(0, str(_PARKOUR_ROOT / "scripts" / "rsl_rl"))
import cli_args as _cli_args
_cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app = AppLauncher(args_cli).app
sys.path.insert(0, str(_PARKOUR_ROOT)); sys.path.insert(0, str(_PARKOUR_ROOT / "parkour_tasks"))
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import parkour_tasks  # noqa
env = gym.make(args_cli.task, cfg=parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1))
for i, n in enumerate(env.unwrapped.scene["robot"].joint_names):
    print(f"JOINT[{i:02d}] = {n}", flush=True)
env.close(); app.close()
