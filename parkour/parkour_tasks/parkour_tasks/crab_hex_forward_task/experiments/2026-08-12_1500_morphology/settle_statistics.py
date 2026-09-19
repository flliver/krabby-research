# SPDX-License-Identifier: BSD-3-Clause
"""M4b: A-share distribution across N jittered zero-action settles at STOCK defaults.
Answers: is the static B-bias systematic, or a landing lottery?"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Crab-Hex-Flat-Walk-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--settle", type=int, default=150)
parser.add_argument("--trials", type=int, default=20)
_PARKOUR_ROOT = Path("/home/nickmagus/krabby/krabby-research/parkour")
sys.path.insert(0, str(_PARKOUR_ROOT / "scripts" / "rsl_rl"))
import cli_args as _cli_args
_cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
sys.path.insert(0, str(_PARKOUR_ROOT)); sys.path.insert(0, str(_PARKOUR_ROOT / "parkour_tasks"))
import gymnasium as gym
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils import parse_env_cfg
import parkour_tasks  # noqa: F401

FOOT_NAMES = ["FL_Footpad","FR_Footpad","ML_Footpad","MR_Footpad","RL_Footpad","RR_Footpad"]
A_SET=(0,3,4)

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
env = gym.make(args_cli.task, cfg=env_cfg)
uenv = env.unwrapped; robot = uenv.scene["robot"]; cs = uenv.scene.sensors["contact_forces"]
foot_cfg = SceneEntityCfg("contact_forces", body_names=FOOT_NAMES, preserve_order=True)
foot_cfg.resolve(uenv.scene)
actions = torch.zeros(env.action_space.shape, device=uenv.device)
gen = torch.Generator(device="cpu").manual_seed(7)

shares, forces_all = [], []
with torch.inference_mode():
    for trial in range(args_cli.trials):
        env.reset()
        jp = robot.data.joint_pos.clone()
        jitter = (torch.rand(jp.shape, generator=gen) * 0.04 - 0.02).to(jp.device)
        robot.write_joint_state_to_sim(jp + jitter, torch.zeros_like(jp))
        fh = []
        for i in range(args_cli.settle):
            env.step(actions)
            if i >= args_cli.settle - 30:
                fh.append(torch.norm(cs.data.net_forces_w[0, foot_cfg.body_ids], dim=-1).cpu())
        f = torch.stack(fh).mean(0)
        a = float(f[list(A_SET)].sum()); tot = float(f.sum())
        shares.append(a / max(tot, 1e-9)); forces_all.append([float(v) for v in f])
        print(f"[trial {trial:02d}] A-share {100*a/max(tot,1e-9):5.1f}%  forces {[round(float(v),0) for v in f]}", flush=True)

import statistics
mean = statistics.mean(shares); sd = statistics.stdev(shares)
print(f"[RESULT] A-share over {args_cli.trials} jittered settles: mean {100*mean:.1f}%  sd {100*sd:.1f}%  "
      f"min {100*min(shares):.1f}%  max {100*max(shares):.1f}%", flush=True)
Path(__file__).with_name("settle_statistics.json").write_text(json.dumps(
    {"shares": shares, "mean": mean, "sd": sd, "forces": forces_all}, indent=2))
env.close(); simulation_app.close()
