# SPDX-License-Identifier: BSD-3-Clause
"""Harvest a sideways-native RSI bank from a trained sideways checkpoint (campaign fallback).

Copied from ``crab_hex_forward_task/experiments/2026-08-22_1200_gait_formation_v2/
harvest_rsi_bank.py`` (2026-09-11); NOT run by default. This is the CHARTER's fallback recipe for
H1 (formation without RSI): if phase 1a does not clear the written gate, this harvests states from
the 1a checkpoint's OWN rollouts -- roll the policy under travel commands, keep states that are
upright and genuinely moving along the travel axis (body +y), record the policy's OWN gait-clock
phase. Because the bank records ``root_quat_w`` verbatim, the -pi/2 spawn yaw is preserved: unlike
the forward (body-x) RSI bank of record, an RSI reset built from this bank will not teleport a
sideways walker back to yaw 0 (see CHARTER.md, Fallback section).

Run (from krabby-research/parkour):
  <venv-python> ../parkour/parkour_tasks/parkour_tasks/crab_hex_sideways_task/experiments/2026-09-11_0414_sideways_formation/harvest_side_rsi_bank.py \
    --headless --task Isaac-Crab-Hex-Side-Walk-Play-v0 \
    --checkpoint logs/rsl_rl/crab_hex_side_flat_walk/<run>/model_4999.pt \
    --num_envs 16 --steps 800
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Crab-Hex-Side-Walk-Play-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=800)
parser.add_argument("--min_vx", type=float, default=0.15)
parser.add_argument("--out", type=str, default=str(Path(__file__).parent / "rsi_bank_side_1a.npz"))
_PARKOUR_ROOT = Path("/home/nickmagus/krabby/krabby-research/parkour")
sys.path.insert(0, str(_PARKOUR_ROOT / "scripts" / "rsl_rl"))
import cli_args as _cli_args  # isort: skip

_cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.experiment_name = "crab_hex_side_flat_walk"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, str(_PARKOUR_ROOT))
sys.path.insert(0, str(_PARKOUR_ROOT / "parkour_tasks"))

import gymnasium as gym
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg

import parkour_tasks  # noqa: F401
from scripts.rsl_rl.runner_factory import agent_cfg_to_train_dict, make_on_policy_runner
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = _cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env = gym.make(args_cli.task, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    wrapped = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = make_on_policy_runner(
        wrapped, agent_cfg_to_train_dict(agent_cfg), log_dir=None, device=agent_cfg.device
    )
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    estimator = runner.get_estimator_inference_policy(device=wrapped.device)
    est = agent_cfg.to_dict()["estimator"]
    num_prop, num_scan, num_priv = est["num_prop"], est["num_scan"], est["num_priv_explicit"]
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    env_origins = env.unwrapped.scene.env_origins

    rows = {k: [] for k in ("joint_pos", "joint_vel", "root_quat_w", "root_z",
                            "root_lin_vel_b", "clock_phase")}
    obs, extras = wrapped.get_observations()
    with torch.inference_mode():
        for step in range(args_cli.steps):
            obs[:, num_prop + num_scan : num_prop + num_scan + num_priv] = estimator.inference(
                obs[:, :num_prop]
            )
            actions = policy(obs.detach())
            obs, _, _, _ = wrapped.step(actions.detach())
            if step < 100:  # let the gait establish
                continue
            d = robot.data
            upright = -d.projected_gravity_b[:, 2] > 0.95
            moving = d.root_lin_vel_b[:, 1] > args_cli.min_vx  # travel speed: body +y (Shape B)
            keep = (upright & moving).nonzero(as_tuple=True)[0]
            if len(keep) == 0 or step % 3 != 0:  # thin the sampling
                continue
            rows["joint_pos"].append(d.joint_pos[keep].cpu().numpy())
            rows["joint_vel"].append(d.joint_vel[keep].cpu().numpy())
            rows["root_quat_w"].append(d.root_quat_w[keep].cpu().numpy())
            rows["root_z"].append((d.root_pos_w[keep, 2] - env_origins[keep, 2]).cpu().numpy())
            rows["root_lin_vel_b"].append(d.root_lin_vel_b[keep].cpu().numpy())
            rows["clock_phase"].append(action_term.clock_phase[keep].cpu().numpy())
    bank = {k: np.concatenate(v).astype(np.float32) for k, v in rows.items()}
    np.savez_compressed(args_cli.out, **bank)
    print(f"[harvest] {bank['joint_pos'].shape[0]} states -> {args_cli.out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
