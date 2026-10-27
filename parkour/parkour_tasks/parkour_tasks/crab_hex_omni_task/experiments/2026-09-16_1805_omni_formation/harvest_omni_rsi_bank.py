# SPDX-License-Identifier: BSD-3-Clause
"""Harvest ONE omni RSI bank frame source from a trained walking checkpoint (plan D6).

Generalises the two existing per-task harvesters into one script driven by ``--axis``/``--source``:

- ``crab_hex_forward_task/experiments/2026-08-22_1200_gait_formation_v2/harvest_rsi_bank.py``
  (body +x, filters on ``root_lin_vel_b[:, 0]``) -- this script's ``--axis x --source 0``.
- ``crab_hex_sideways_task/experiments/2026-09-11_0414_sideways_formation/harvest_side_rsi_bank.py``
  (body +y, "left side leads", filters on ``root_lin_vel_b[:, 1]``) -- ``--axis y --source 1``.

The fixed thresholds and step counters both scripts hardcoded (``upright > 0.95``, ``v > 0.15``,
warm-up ``step < 100``, thin every 3rd step) are CLI flags here (``--min_upright``, ``--min_speed``,
``--min_step``, ``--thin``), plus a new ``|wz| < --max_wz`` filter neither original script had, and
a per-env episode-length gate (``env.episode_length_buf >= --min_step``, robust to mid-rollout
resets) in place of the original single global step counter.

Writes the SAME six arrays the forward RSI reset event reads
(``crab_hex_forward_task/mdp/crab_hex_rsi.py::_load_bank``) -- ``joint_pos``, ``joint_vel``,
``root_quat_w``, ``root_z``, ``root_lin_vel_b``, ``clock_phase`` -- so a bank built by this script
loads there unmodified, plus a ``source`` int8 tag (all rows = ``--source``) and a ``meta`` JSON
string (task, checkpoint path + sha256, axis, filters, counts, a PRE-filter stats pool of
upright/speed/wz at every thinned+episode-eligible step -- the natural distribution the filters cut
into, since angular velocity is never itself stored in the bank -- git HEAD, date).

Two invocations of this one script (D6) produce ``rsi_bank_fwd_raw.npz`` and
``rsi_bank_side_raw.npz``; ``merge_rsi_banks.py`` in this directory combines them into
``rsi_bank_omni.npz``.

Run (from krabby-research/parkour):
  <venv-python> ../parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/\
2026-09-16_1805_omni_formation/harvest_omni_rsi_bank.py --headless \
    --task Isaac-Crab-Hex-Flat-Walk-Play-v0 --checkpoint <ckpt> \
    --axis x --source 0 --out rsi_bank_fwd_raw.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher

AXIS_INDEX = {"x": 0, "y": 1}
AXIS_DEG = {"x": 0.0, "y": 90.0}  # atan2(vy,vx) this axis implies, for the source<->angle check

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
# NOTE: --checkpoint is NOT defined here -- _cli_args.add_rsl_rl_args(parser) below adds it
# (conflicts with argparse if redeclared); args_cli.checkpoint is required at runtime instead.
parser.add_argument(
    "--axis", type=str, choices=("x", "y"), required=True,
    help="body-frame travel axis to filter/tag on: x for the forward head, "
         "+y for the sideways head (left side leads)",
)
parser.add_argument(
    "--source", type=int, choices=(0, 1), required=True,
    help="provenance tag written into the bank's `source` array (0=forward/x, 1=sideways/y)",
)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--frames", type=int, default=2000, help="environment steps to roll the policy")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--min_speed", type=float, default=0.15)
parser.add_argument("--max_wz", type=float, default=0.3)
parser.add_argument("--min_upright", type=float, default=0.95)
parser.add_argument(
    "--min_step", type=int, default=100,
    help="per-env episode step (since last reset) below which frames are dropped",
)
parser.add_argument("--thin", type=int, default=3, help="keep only every Nth rolled step")
_PARKOUR_ROOT = Path("/home/nickmagus/krabby/krabby-research/parkour")
sys.path.insert(0, str(_PARKOUR_ROOT / "scripts" / "rsl_rl"))
import cli_args as _cli_args  # isort: skip

_cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.checkpoint:
    parser.error("--checkpoint is required (added by add_rsl_rl_args, default None)")
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

_BANK_KEYS = ("joint_pos", "joint_vel", "root_quat_w", "root_z", "root_lin_vel_b", "clock_phase")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _stats(x: np.ndarray) -> dict:
    if x.size == 0:
        return {"median": float("nan"), "p10": float("nan"), "p90": float("nan"), "n": 0}
    return {
        "median": float(np.percentile(x, 50)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
        "n": int(x.size),
    }


def main() -> None:
    axis_idx = AXIS_INDEX[args_cli.axis]
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

    rows: dict[str, list[np.ndarray]] = {k: [] for k in _BANK_KEYS}
    # Diagnostic pool: (upright, speed-along-axis, wz) at every THINNED, episode-eligible
    # step for ALL envs -- gated only by --min_step, i.e. BEFORE the quality filters -- so we
    # can report the natural distribution the filters cut into. wz is never itself stored in
    # the bank (root angular velocity is not part of the RSI state), so this is the only place
    # its distribution is ever captured.
    diag_upright: list[np.ndarray] = []
    diag_speed: list[np.ndarray] = []
    diag_wz: list[np.ndarray] = []
    n_candidates = 0

    obs, extras = wrapped.get_observations()
    with torch.inference_mode():
        for step in range(args_cli.frames):
            obs[:, num_prop + num_scan : num_prop + num_scan + num_priv] = estimator.inference(
                obs[:, :num_prop]
            )
            actions = policy(obs.detach())
            obs, _, _, _ = wrapped.step(actions.detach())
            if step % args_cli.thin != 0:
                continue
            ep_len = env.unwrapped.episode_length_buf
            eligible = (ep_len >= args_cli.min_step).nonzero(as_tuple=True)[0]
            if len(eligible) == 0:
                continue
            n_candidates += len(eligible)
            d = robot.data
            upright = -d.projected_gravity_b[eligible, 2]
            speed = d.root_lin_vel_b[eligible, axis_idx]
            wz = d.root_ang_vel_b[eligible, 2]
            diag_upright.append(upright.cpu().numpy())
            diag_speed.append(speed.cpu().numpy())
            diag_wz.append(wz.cpu().numpy())
            keep_mask = (
                (upright > args_cli.min_upright)
                & (speed > args_cli.min_speed)
                & (wz.abs() < args_cli.max_wz)
            )
            keep = eligible[keep_mask.nonzero(as_tuple=True)[0]]
            if len(keep) == 0:
                continue
            rows["joint_pos"].append(d.joint_pos[keep].cpu().numpy())
            rows["joint_vel"].append(d.joint_vel[keep].cpu().numpy())
            rows["root_quat_w"].append(d.root_quat_w[keep].cpu().numpy())
            rows["root_z"].append((d.root_pos_w[keep, 2] - env_origins[keep, 2]).cpu().numpy())
            rows["root_lin_vel_b"].append(d.root_lin_vel_b[keep].cpu().numpy())
            rows["clock_phase"].append(action_term.clock_phase[keep].cpu().numpy())

    assert len(rows["joint_pos"]) > 0 and sum(a.shape[0] for a in rows["joint_pos"]) > 0, (
        f"[harvest] ZERO frames survived the filters (upright>{args_cli.min_upright}, "
        f"speed[axis {args_cli.axis}]>{args_cli.min_speed}, |wz|<{args_cli.max_wz}, "
        f"episode step>={args_cli.min_step}) over {n_candidates} candidate env-steps -- "
        "refusing to write an empty/degenerate bank."
    )

    bank = {k: np.concatenate(v).astype(np.float32) for k, v in rows.items()}
    n = bank["joint_pos"].shape[0]
    if n < 100:
        print(f"[harvest] WARNING: only {n} frames kept (< 100) -- a thin bank", flush=True)
    for k, v in bank.items():
        assert v.shape[0] == n, f"[harvest] array length mismatch: {k} has {v.shape[0]} rows, expected {n}"
        assert np.isfinite(v).all(), f"[harvest] non-finite values in {k}"
    assert bank["joint_pos"].shape == bank["joint_vel"].shape, "joint_pos/joint_vel shape mismatch"
    assert bank["root_quat_w"].shape[1] == 4, f"root_quat_w must be (N,4), got {bank['root_quat_w'].shape}"
    assert bank["root_lin_vel_b"].shape[1] == 3, f"root_lin_vel_b must be (N,3), got {bank['root_lin_vel_b'].shape}"

    diag_upright_np = np.concatenate(diag_upright) if diag_upright else np.zeros(0, dtype=np.float32)
    diag_speed_np = np.concatenate(diag_speed) if diag_speed else np.zeros(0, dtype=np.float32)
    diag_wz_np = np.concatenate(diag_wz) if diag_wz else np.zeros(0, dtype=np.float32)

    ckpt_path = Path(args_cli.checkpoint).resolve()
    meta = {
        "task": args_cli.task,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": _sha256(ckpt_path),
        "axis": args_cli.axis,
        "axis_deg": AXIS_DEG[args_cli.axis],
        "source": int(args_cli.source),
        "filters": {
            "min_speed": args_cli.min_speed,
            "max_wz": args_cli.max_wz,
            "min_upright": args_cli.min_upright,
            "min_step": args_cli.min_step,
            "thin": args_cli.thin,
        },
        "counts": {
            "frames_rolled": int(args_cli.frames),
            "num_envs": int(args_cli.num_envs),
            "candidate_env_steps": int(n_candidates),
            "kept_frames": int(n),
        },
        "stats_prefilter": {
            "upright": _stats(diag_upright_np),
            "speed": _stats(diag_speed_np),
            "wz": _stats(diag_wz_np),
        },
        "git_head": _git_head(),
        "date": datetime.now(timezone.utc).isoformat(),
    }
    bank["source"] = np.full((n,), int(args_cli.source), dtype=np.int8)
    bank["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(args_cli.out, **bank)
    print(f"[harvest] {n} states (of {n_candidates} candidates) -> {args_cli.out}", flush=True)
    print(f"[harvest] meta={json.dumps(meta)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
