# SPDX-License-Identifier: BSD-3-Clause
"""Training-timeline probe (PLAN H B0c / A0, 2026-09-03).

Rolls a checkpoint in the lineage's ACTUAL training configuration (the env stack comes
from the KRABBY_* env vars the orchestrator sets: band 0.0:0.35, 6-s resample, 20-s
episodes, RSI 0.2 with the window's bank, pushes, heading band, curriculum terrain) and
answers: how often does a training episode reach the platform edge / the obstacles, how
much of it is spent in the field, and WHEN inside the episode does the robot move?

Two policy modes per run: ``stochastic`` (actions sampled from the checkpoint's
distribution, std at the 2.0 clamp -- what training actually sees) and ``deterministic``
(the mean action, what the eval harness sees). The exposure numbers come from the same
ExposureLedger ParkourEvent uses for training telemetry, so A0's stochastic column must
match C0's first-iteration Metrics/base_parkour/* by construction.

Run (from krabby-research/parkour, env vars set):
  <venv-python> ../parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-09-03_1156_obstacle_exposure/training_timeline_probe.py \
    --headless --checkpoint <model.pt> --num_envs 64 --steps 4000 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Crab-Hex-Flat-Walk-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=4000, help="env steps per policy mode (20-s episodes = 1000 steps)")
parser.add_argument("--modes", type=str, default="stochastic,deterministic")
parser.add_argument("--walk_vx", type=float, default=0.05, help="|root vx| above this counts as walking")
FAST_VX = 0.12  # above the zero-command creep speed (~0.08 m/s in the eval stand hold)
parser.add_argument("--out", type=str, required=True, help="output directory (JSON summary + NPZ timeline)")
parser.add_argument("--label", type=str, default="probe")
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
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg

import parkour_tasks  # noqa: F401
from parkour_isaaclab.envs.mdp.parkours.exposure_stats import (
    SPAWN_RSI,
    ExposureLedger,
    motion_profile,
)
from scripts.rsl_rl.runner_factory import agent_cfg_to_train_dict, make_on_policy_runner
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper


def _actor_critic(runner):
    alg = runner.alg
    for name in ("actor_critic", "policy"):
        if hasattr(alg, name):
            return getattr(alg, name)
    raise AttributeError("runner.alg has neither actor_critic nor policy")


def _summarise_timeline(tl: dict, horizon_s: float, bin_s: float, walk_vx: float) -> dict:
    """Motion profile + per-slot statistics from the per-step timeline arrays."""
    t = torch.from_numpy(tl["t_s"])
    walking = torch.from_numpy(np.abs(tl["root_vx"]) > walk_vx)
    cmd_on = torch.from_numpy(tl["cmd_vx"] > 0.0)
    non_rsi = torch.from_numpy(tl["spawn_kind"] != SPAWN_RSI)
    obst = torch.from_numpy(tl["is_obst"].astype(bool))
    out = {
        "walking_frac_all": float(walking.float().mean()),
        "walking_frac_non_rsi": float(walking[non_rsi].float().mean()) if non_rsi.any() else float("nan"),
        "cmd_on_frac": float(cmd_on.float().mean()),
        "walking_given_cmd_on": float(walking[cmd_on].float().mean()) if cmd_on.any() else float("nan"),
        "walking_given_cmd_off": float(walking[~cmd_on].float().mean()) if (~cmd_on).any() else float("nan"),
        "mean_root_vx_walking": float(np.abs(tl["root_vx"])[walking.numpy()].mean()) if walking.any() else float("nan"),
        "mean_root_vx_cmd_on": float(tl["root_vx"][cmd_on.numpy()].mean()) if cmd_on.any() else float("nan"),
        "motion_profile_bins_s": bin_s,
        "motion_profile_walking": motion_profile(t[non_rsi], walking[non_rsi], horizon_s, bin_s),
        "motion_profile_cmd_on": motion_profile(t[non_rsi], cmd_on[non_rsi], horizon_s, bin_s),
        "motion_profile_walking_obst": motion_profile(t[non_rsi & obst], walking[non_rsi & obst], horizon_s, bin_s)
        if (non_rsi & obst).any() else [],
    }
    # stricter "walking" = above the zero-command creep (eval stand-hold achieved_vx ~0.08 m/s)
    fast = torch.from_numpy(np.abs(tl["root_vx"]) > FAST_VX)
    out["fast_vx_threshold"] = FAST_VX
    out["fast_frac_non_rsi"] = float(fast[non_rsi].float().mean()) if non_rsi.any() else float("nan")
    out["fast_given_cmd_on"] = float(fast[cmd_on].float().mean()) if cmd_on.any() else float("nan")
    out["fast_given_cmd_off"] = float(fast[~cmd_on].float().mean()) if (~cmd_on).any() else float("nan")
    out["motion_profile_fast"] = motion_profile(t[non_rsi], fast[non_rsi], horizon_s, bin_s)
    # systematic-late-motion test: walking fraction in the last bin vs the mean of the others
    prof = [p for p in out["motion_profile_walking"] if p == p]
    if len(prof) >= 2:
        out["late_motion_ratio"] = prof[-1] / max(1e-6, float(np.mean(prof[:-1])))
    prof_f = [p for p in out["motion_profile_fast"] if p == p]
    if len(prof_f) >= 2:
        out["late_motion_ratio_fast"] = prof_f[-1] / max(1e-6, float(np.mean(prof_f[:-1])))
    return out


def run_mode(env, wrapped, runner, mode: str, steps: int, walk_vx: float) -> tuple[dict, dict]:
    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    pe = uenv.parkour_manager.get_term("base_parkour")
    ac = _actor_critic(runner)
    policy_det = runner.get_inference_policy(device=uenv.device)
    estimator = runner.get_estimator_inference_policy(device=wrapped.device)
    est = runner_agent_cfg["estimator"]
    num_prop, num_scan, num_priv = est["num_prop"], est["num_scan"], est["num_priv_explicit"]
    dt = float(uenv.step_dt)
    # fresh ledger for this mode (ParkourEvent's own ledger keeps accumulating across modes)
    ledger = ExposureLedger(pe.n_obst, capacity=max(2048, steps), device="cpu")
    pe.exposure = ledger
    cols = {k: [] for k in ("t_s", "cmd_vx", "root_vx", "x_rel", "spawn_kind", "is_obst", "goal_idx", "env",
                            "goal1_x", "edge_x", "failed")}
    env_idx = torch.arange(uenv.num_envs, device=uenv.device)
    # Everything (the reset too) under inference_mode: the parkour term re-creates its metric
    # tensors every step, so after one mode's rollout they are inference tensors and a reset
    # outside the context ("metric_value[env_ids] = 0.0") raises. Training/eval never reset
    # outside their rollout context, so only the probe needs this.
    with torch.inference_mode():
        wrapped.reset()
        obs, _ = wrapped.get_observations()
        for step in range(steps):
            obs[:, num_prop + num_scan: num_prop + num_scan + num_priv] = estimator.inference(obs[:, :num_prop])
            if mode == "stochastic":
                actions = ac.act(obs.detach(), hist_encoding=True)
            else:
                actions = policy_det(obs.detach(), hist_encoding=True)  # eval-harness convention
            # record the state the action was taken in
            cols["t_s"].append((uenv.episode_length_buf.float() * dt).cpu().numpy())
            cols["cmd_vx"].append(uenv.command_manager.get_command("base_velocity")[:, 0].cpu().numpy())
            cols["root_vx"].append(robot.data.root_lin_vel_b[:, 0].cpu().numpy())
            cols["x_rel"].append((robot.data.root_pos_w[:, 0] - pe.env_origins[:, 0]).cpu().numpy())
            cols["spawn_kind"].append(pe.spawn_kind.cpu().numpy())
            cols["is_obst"].append(pe.is_obst_tile.cpu().numpy())
            cols["goal_idx"].append(pe.cur_goal_idx.cpu().numpy())
            cols["env"].append(env_idx.cpu().numpy())
            cols["goal1_x"].append(pe.env_goals[:, 1, 0].cpu().numpy())        # first-obstacle goal x (origin-relative)
            cols["edge_x"].append(np.full(uenv.num_envs, float(pe.edge_x_rel), dtype=np.float32))
            obs, _, _, _ = wrapped.step(actions.detach())
            # termination flag of the step just taken (crab_failure); the env has already reset
            try:
                cols["failed"].append(uenv.termination_manager.get_term("crab_failure").cpu().numpy())
            except Exception:
                cols["failed"].append(np.zeros(uenv.num_envs, dtype=bool))
    tl = {k: np.concatenate(v) for k, v in cols.items()}
    exposure = ledger.means()
    summary = {
        "mode": mode,
        "steps": steps,
        "num_envs": int(uenv.num_envs),
        "episodes_pushed": int(ledger._rings["rsi_frac_actual"].n),
        "exposure": exposure,
        "timeline": _summarise_timeline(tl, float(uenv.cfg.episode_length_s), 6.0, walk_vx),
        "terrain_levels_mean": float(pe.terrain.terrain_levels.float().mean()),
        "obst_tile_frac": float(pe.is_obst_tile.float().mean()),
    }
    return summary, tl


def main() -> None:
    global runner_agent_cfg
    out_dir = Path(args_cli.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = _cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    runner_agent_cfg = agent_cfg.to_dict()
    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = make_on_policy_runner(wrapped, agent_cfg_to_train_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    ac = _actor_critic(runner)
    try:
        std = float(ac.std.mean()) if hasattr(ac, "std") else float("nan")
    except Exception:
        std = float("nan")
    results = {
        "label": args_cli.label,
        "checkpoint": args_cli.checkpoint,
        "task": args_cli.task,
        "episode_length_s": float(env.unwrapped.cfg.episode_length_s),
        "resampling_time_range": list(env.unwrapped.cfg.commands.base_velocity.resampling_time_range),
        "lin_vel_x": list(env.unwrapped.cfg.commands.base_velocity.ranges.lin_vel_x),
        "policy_std_mean": std,
        "env_vars": {k: v for k, v in __import__("os").environ.items() if k.startswith("KRABBY_")},
        "modes": {},
    }
    for mode in [m.strip() for m in args_cli.modes.split(",") if m.strip()]:
        t0 = time.time()
        summary, tl = run_mode(env, wrapped, runner, mode, args_cli.steps, args_cli.walk_vx)
        summary["wall_s"] = time.time() - t0
        results["modes"][mode] = summary
        np.savez_compressed(out_dir / f"timeline_{mode}.npz", **tl)
        print(f"[probe] {mode}: {json.dumps(summary['exposure'])}", flush=True)
        print(f"[probe] {mode} motion: {json.dumps(summary['timeline'])}", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(results, indent=1))
    print(f"[probe] wrote {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
