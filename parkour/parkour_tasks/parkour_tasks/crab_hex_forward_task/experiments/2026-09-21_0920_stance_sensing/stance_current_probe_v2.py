# SPDX-License-Identifier: BSD-3-Clause
"""Stance-detection study v2: motor-current proxy vs ground-truth foot contact.

Successor to ``experiments/2026-08-20_1506_hardware_morphology/stance_current_probe.py``,
which measured the proxy on the 2026-08-20 hardware plant under a SCRIPTED sine (the open
item it left behind: "rerun the stance study on a real trained gait, not the scripted
sine"). Two changes:

  * ``--mode policy``: a trained checkpoint drives the robot, so the stance/swing split is
    the one the policy actually produces (the scripted sine held the feet down 72-97 % of
    the time, which barely exercised the false-positive side).
  * the report stores AGREEMENT (how often proxy and contact sensor say the same thing)
    and the full confusion split, not only precision/recall/F1 -- plus the always-stance
    baseline, so agreement can be read against the trivial predictor.

``--mode scripted`` reproduces the 2026-08-20 test exactly (same cam/sine drive, env 0
only) so the new plant can be compared against the old numbers on the same motion.

The proxy is read from the LIVE observation term (``CrabHexParkourObservations._get_contact_fill``)
when it can be reached, so there is no second copy of the sensing model to drift; the
recomputed-from-primitives path is kept as a fallback and cross-checked when both exist.
Ground truth is the privileged sim ``ContactSensor`` (>5 N on a footpad = stance).

Run (from ~/krabby/krabby-research/parkour):

    # scripted, current plant
    KRABBY_PLANT=A15+B OMNI_KIT_ACCEPT_EULA=yes /home/nickmagus/krabby/isaac_venv/bin/python \
        .../2026-09-21_0920_stance_sensing/stance_current_probe_v2.py --headless \
        --mode scripted --num_envs 1 --gait_steps 800 --output .../report_scripted.json

    # trained gait (teacher head of record)
    KRABBY_PHASE=2c KRABBY_PLANT=A15+B OMNI_KIT_ACCEPT_EULA=yes /home/nickmagus/krabby/isaac_venv/bin/python \
        .../stance_current_probe_v2.py --headless --mode policy \
        --task Isaac-Crab-Hex-Teacher-Play-v0 --num_envs 32 --gait_steps 1200 \
        --checkpoint .../policy/2c_teacher/model_19996.pt --output .../report_policy_2c.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Current-sense stance proxy vs true contact (v2).")
parser.add_argument("--task", type=str, default="Isaac-Crab-Hex-Flat-Walk-Play-v0")
parser.add_argument("--mode", type=str, default="scripted", choices=("scripted", "policy"))
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--settle_steps", type=int, default=200, help="Scripted mode only.")
parser.add_argument("--gait_steps", type=int, default=800, help="Measured steps.")
parser.add_argument(
    "--skip_after_reset", type=int, default=10, help="Policy mode: steps dropped after each reset."
)
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--save-raw", dest="save_raw", type=str, default=None, help="Optional .npz of the raw traces.")

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
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg

import parkour_tasks  # noqa: F401

FOOT_NAMES = ["FL_Footpad", "FR_Footpad", "ML_Footpad", "MR_Footpad", "RL_Footpad", "RR_Footpad"]
CONTACT_FORCE_N = 5.0  # ground-truth stance threshold
THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.5)
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _live_obs_term(env):
    """The instantiated ``CrabHexParkourObservations`` the policy reads, or ``None``."""
    from parkour_tasks.crab_hex_forward_task.mdp.observations import CrabHexParkourObservations

    om = env.unwrapped.observation_manager
    for cfgs in getattr(om, "_group_obs_class_term_cfgs", {}).values():
        for cfg in cfgs:
            if isinstance(getattr(cfg, "func", None), CrabHexParkourObservations):
                return cfg.func
    return None


def _confusion(pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> dict:
    n = float(mask.sum())
    tp = float((pred & truth & mask).sum())
    fp = float((pred & ~truth & mask).sum())
    fn = float((~pred & truth & mask).sum())
    tn = float((~pred & ~truth & mask).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "agreement": round((tp + tn) / n, 3) if n else 0.0,
        "specificity": round(specificity, 3),
        "balanced_accuracy": round(0.5 * (recall + specificity), 3),
        "fractions": {k: round(v / n, 3) if n else 0.0 for k, v in (("tp", tp), ("fp", fp), ("fn", fn), ("tn", tn))},
    }


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]
    contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
    foot_ids, _ = contact_sensor.find_bodies(FOOT_NAMES, preserve_order=True)

    from parkour_tasks.crab_hex_forward_task.mdp import crab_hex_linkage as linkage
    from parkour_tasks.crab_hex_forward_task.mdp.observations import (
        _CURRENT_SENSE_GATE_FRACTION,
        _CURRENT_SENSE_MAX,
        _CURRENT_SENSE_NO_LOAD,
    )

    pitch_ids, pitch_names = robot.find_joints([".*_Hip_Femur_RevoluteJoint"], preserve_order=True)
    knee_ids, _ = robot.find_joints(
        [n.replace("_Hip_Femur_", "_Femur_Tibia_") for n in pitch_names], preserve_order=True
    )
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    obs_term = _live_obs_term(env)

    def recomputed_proxy():
        """The 2026-08-20 probe's recomputation, in the obs class's own numeric range (+0.5)."""
        tau = robot.data.applied_torque
        theta_h = robot.data.joint_pos[:, pitch_ids]
        theta_k = robot.data.joint_pos[:, knee_ids]
        ma_h = linkage.hip_moment_arm(theta_h).clamp_min(1e-4)
        ma_k = linkage.knee_moment_arm(theta_h, theta_k.abs()).clamp_min(1e-4)
        force_h = tau[:, pitch_ids].abs() / ma_h / linkage.HIP_FORCE_N
        force_k = tau[:, knee_ids].abs() / ma_k / linkage.KNEE_FORCE_N
        gate_h = (
            action_term.hip_rod_speed.abs() / (_CURRENT_SENSE_GATE_FRACTION * linkage.HIP_SPEED_M_S)
        ).clamp(0.0, 1.0)
        gate_k = (
            action_term.knee_rod_speed.abs()
            / (_CURRENT_SENSE_GATE_FRACTION * linkage.KNEE_SPEED_M_S)
        ).clamp(0.0, 1.0)
        cur_h = gate_h * (_CURRENT_SENSE_NO_LOAD + force_h).clamp(0.0, _CURRENT_SENSE_MAX)
        cur_k = gate_k * (_CURRENT_SENSE_NO_LOAD + force_k).clamp(0.0, _CURRENT_SENSE_MAX)
        return torch.maximum(cur_h, cur_k), torch.maximum(gate_h, gate_k)

    def sample():
        """(proxy, gate, truth, force, recomputed) for the current sim state."""
        nonlocal obs_term
        recomp, gate = recomputed_proxy()
        proxy = recomp
        if obs_term is not None:
            try:
                proxy = obs_term._get_contact_fill() + 0.5
            except Exception:  # noqa: BLE001 -- fall back to the local recomputation
                obs_term = None
        force = contact_sensor.data.net_forces_w[:, foot_ids].norm(dim=-1)
        return proxy, gate, force > CONTACT_FORCE_N, force, recomp

    proxies, gates, truths, forces, masks, live_gap = [], [], [], [], [], 0.0
    settle_contact, root_z, gait_stats = None, None, {}

    if args_cli.mode == "scripted":
        obs, _ = env.reset()
        zero = torch.zeros((args_cli.num_envs, env.action_space.shape[1]), device=device)
        names = action_term._joint_names
        cam_cols = [i for i, n in enumerate(names) if "CamShaft" in n]
        hip_cols = [i for i, n in enumerate(names) if "Hip_Femur" in n]
        knee_cols = [i for i, n in enumerate(names) if "Femur_Tibia" in n]
        with torch.inference_mode():
            for _ in range(args_cli.settle_steps):
                env.step(zero)
        root_z = float(robot.data.root_pos_w[0, 2].item())
        settle_contact = contact_sensor.data.net_forces_w[0, foot_ids].norm(dim=-1)
        env_dt = float(env.unwrapped.step_dt)
        with torch.inference_mode():
            for i in range(args_cli.gait_steps):
                t = i * env_dt
                act = zero.clone()
                for c in cam_cols:
                    act[:, c] = 0.6
                hip_a = 0.6 * math.sin(2.0 * math.pi * 0.4 * t)
                knee_a = 0.6 * math.sin(2.0 * math.pi * 0.4 * t + math.pi / 2.0)
                for c in hip_cols:
                    act[:, c] = hip_a
                for c in knee_cols:
                    act[:, c] = knee_a
                env.step(act)
                p, g, tr, f, rc = sample()
                live_gap = max(live_gap, float((p - rc).abs().max()))
                proxies.append(p.clone()), gates.append(g.clone())
                truths.append(tr.clone()), forces.append(f.clone())
    else:
        from scripts.rsl_rl.runner_factory import agent_cfg_to_train_dict, make_on_policy_runner
        from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper

        if "Student" in args_cli.task:
            raise SystemExit("policy mode supports the flat-walk/teacher heads; student needs the depth branch")
        agent_cfg = _cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
        wrapped = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = make_on_policy_runner(
            wrapped, agent_cfg_to_train_dict(agent_cfg), log_dir=None, device=agent_cfg.device
        )
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=device)
        estimator = runner.get_estimator_inference_policy(device=wrapped.device)
        est = agent_cfg.to_dict()["estimator"]
        num_prop, num_scan, num_priv = est["num_prop"], est["num_scan"], est["num_priv_explicit"]

        obs, _ = wrapped.get_observations()
        cooldown = torch.full((args_cli.num_envs,), args_cli.skip_after_reset, device=device)
        vx, dones_total = [], 0
        for _ in range(args_cli.gait_steps):
            with torch.inference_mode():
                obs[:, num_prop + num_scan : num_prop + num_scan + num_priv] = estimator.inference(
                    obs[:, :num_prop]
                )
                actions = policy(obs, hist_encoding=True)
            obs, _, dones, _ = wrapped.step(actions)
            p, g, tr, f, rc = sample()
            live_gap = max(live_gap, float((p - rc).abs().max()))
            masks.append((cooldown <= 0).clone())
            proxies.append(p.clone()), gates.append(g.clone())
            truths.append(tr.clone()), forces.append(f.clone())
            vx.append(float(robot.data.root_lin_vel_b[:, 0].mean()))
            cooldown = torch.where(dones.bool(), torch.full_like(cooldown, args_cli.skip_after_reset), cooldown - 1)
            dones_total += int(dones.sum())
        root_z = float(robot.data.root_pos_w[:, 2].mean().item())
        gait_stats = {
            "checkpoint": args_cli.checkpoint,
            "resets": dones_total,
            "mean_steps_per_episode": round(
                args_cli.gait_steps * args_cli.num_envs / max(dones_total, 1), 1
            ),
            "mean_vx_body_m_s": round(float(np.mean(vx)), 3),
        }

    P = torch.stack(proxies)  # (T, N, 6)
    G = torch.stack(gates)
    T_ = torch.stack(truths)
    F = torch.stack(forces)
    M = torch.stack(masks) if args_cli.mode == "policy" else torch.ones_like(T_[:, :, 0], dtype=torch.bool)
    M = M.unsqueeze(-1).expand_as(T_)
    if args_cli.mode == "scripted":  # 2026-08-20 parity: env 0 only
        P, G, T_, F, M = (x[:, :1] for x in (P, G, T_, F, M))

    prevalence = float((T_ & M).sum() / M.sum())
    driving = G > 0.05
    report = {
        "mode": args_cli.mode,
        "task": args_cli.task,
        "plant": os.environ.get("KRABBY_PLANT", "(config default)"),
        "phase": os.environ.get("KRABBY_PHASE", "(none)"),
        "num_envs": args_cli.num_envs,
        "gait_steps": args_cli.gait_steps,
        "leg_steps_scored": int(M.sum()),
        "proxy_source": "live obs term" if obs_term is not None else "recomputed",
        "live_vs_recomputed_max_abs_gap": round(live_gap, 6),
        "settled_root_height_m": root_z,
        "true_stance_fraction": round(prevalence, 3),
        "always_stance_agreement": round(max(prevalence, 1 - prevalence), 3),
        "dead_zone_fraction_of_stance": round(
            float(((G < 0.05) & T_ & M).sum() / (T_ & M).sum()), 3
        ),
        "mean_proxy_in_stance": round(float(P[T_ & M].mean()), 3),
        "mean_proxy_in_swing": round(float(P[~T_ & M].mean()) if (~T_ & M).any() else 0.0, 3),
        "thresholds": {},
    }
    if settle_contact is not None:
        report["settled_foot_forces_N"] = [round(float(v), 1) for v in settle_contact]
    if gait_stats:
        report["gait"] = gait_stats

    for thr in THRESHOLDS:
        report["thresholds"][str(thr)] = _confusion(P > thr, T_, M)
    # Driving-phase-only (where the signal physically exists), 2026-08-20 definition.
    for thr in (0.1, 0.2):
        pred, t_driving = (P > thr) & driving, T_ & driving
        tp = float((pred & t_driving & M).sum())
        fp = float((pred & ~T_ & M).sum())
        fn = float((~pred & t_driving & M).sum())
        report["thresholds"][f"{thr}_driving_only"] = {
            "precision": round(tp / (tp + fp), 3) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 3) if tp + fn else 0.0,
        }
    report["best_by_f1"] = max(THRESHOLDS, key=lambda t: report["thresholds"][str(t)]["f1"])
    report["best_by_agreement"] = max(
        THRESHOLDS, key=lambda t: report["thresholds"][str(t)]["agreement"]
    )
    report["best_by_balanced_accuracy"] = max(
        THRESHOLDS, key=lambda t: report["thresholds"][str(t)]["balanced_accuracy"]
    )
    # Threshold-free: ROC AUC over a fine sweep (trapezoid over the (1-spec, recall) curve).
    fine = np.linspace(0.0, 1.6, 161)
    pts = []
    for thr in fine:
        c = _confusion(P > float(thr), T_, M)
        pts.append((1.0 - c["specificity"], c["recall"]))
    pts = sorted(set(pts))
    report["roc_auc"] = round(
        float(_trapz([y for _, y in pts], [x for x, _ in pts])) if len(pts) > 1 else 0.0, 3
    )
    thr_best = report["best_by_f1"]
    report["per_leg"] = {
        FOOT_NAMES[i]: _confusion(P[..., i : i + 1] > thr_best, T_[..., i : i + 1], M[..., i : i + 1])
        for i in range(6)
    }

    out = args_cli.output or str(Path(__file__).parent / f"report_{args_cli.mode}.json")
    Path(out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if args_cli.save_raw:
        np.savez_compressed(
            args_cli.save_raw,
            proxy=P.cpu().numpy(),
            gate=G.cpu().numpy(),
            truth=T_.cpu().numpy(),
            force=F.cpu().numpy(),
            mask=M.cpu().numpy(),
        )
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
