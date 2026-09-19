# SPDX-License-Identifier: BSD-3-Clause
"""Rung (ii) statics battery for one plant variant, in ONE Isaac boot (PLAN G, 2026-09-02).

Select the variant with ``KRABBY_HEX_USD_PATH`` (unset = the golden asset). Records:

1. ``--trials`` jittered zero-action settles (+-0.02 rad joint jitter, seed 7 -- the
   settle_statistics.py protocol; the hyperstatic 6-foot load split is a landing lottery,
   so N~20 is mandatory): per trial the last-30-step mean per-foot force, tripod-set share,
   pitch/roll, root height, min footpad z (penetration check), femur/hip net contact force
   (femurs and hip plates never touch the ground at neutral, so force there = leg-leg or
   leg-body contact), and the joint torques at the end of the settle.
2. On the last settle, the settled joint positions, body positions and names (offline
   FK-vs-Isaac cross-check) and the standing support-polygon margins.
3. A scripted cam sweep (all cams at +0.5 for ``--sweep_steps``): hip-yaw range, femur/hip
   contact forces and min footpad z during the sweep, plus per-step joint/foot records.
4. Optionally (``--video``) an mp4 of the last settle + the sweep for the user's sign-off.

Writes ``<out_dir>/statics_<tag>.json`` and ``statics_<tag>.npz``.
Run: isaac_venv python statics_battery.py --headless --video --out_dir ... --tag ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Isaac-Crab-Hex-Flat-Walk-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--settle", type=int, default=150)
parser.add_argument("--trials", type=int, default=20)
parser.add_argument("--sweep_steps", type=int, default=250)
parser.add_argument("--out_dir", type=str, default=str(Path(__file__).parent / "statics"))
parser.add_argument("--tag", type=str, default="base")
parser.add_argument("--video", action="store_true", default=False)
_PARKOUR_ROOT = Path("/home/nickmagus/krabby/krabby-research/parkour")
sys.path.insert(0, str(_PARKOUR_ROOT / "scripts" / "rsl_rl"))
import cli_args as _cli_args  # isort: skip

_cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, str(_PARKOUR_ROOT))
sys.path.insert(0, str(_PARKOUR_ROOT / "parkour_tasks"))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import parkour_tasks  # noqa: F401,E402

FOOT_NAMES = ["FL_Footpad", "FR_Footpad", "ML_Footpad", "MR_Footpad", "RL_Footpad", "RR_Footpad"]
A_SET = (0, 3, 4)
B_SET = (1, 2, 5)


def pitch_roll(q) -> tuple[float, float]:
    w, x, y, z = [float(v) for v in q]
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    return pitch, roll


def main() -> None:
    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    total_video = args_cli.settle + args_cli.sweep_steps
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env, video_folder=str(out_dir / f"video_{args_cli.tag}"),
            step_trigger=lambda s: s == (args_cli.trials - 1) * args_cli.settle,
            video_length=total_video, disable_logger=True,
        )
    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    cs = uenv.scene.sensors["contact_forces"]
    foot_cfg = SceneEntityCfg("contact_forces", body_names=FOOT_NAMES, preserve_order=True)
    foot_cfg.resolve(uenv.scene)
    foot_body_cfg = SceneEntityCfg("robot", body_names=FOOT_NAMES, preserve_order=True)
    foot_body_cfg.resolve(uenv.scene)
    leg_cfg = SceneEntityCfg("contact_forces", body_names=[".*_Hip", ".*_Femur"])
    leg_cfg.resolve(uenv.scene)
    leg_link_names = [cs.body_names[i] for i in leg_cfg.body_ids]
    action_term = uenv.action_manager.get_term("joint_pos")
    cam_cols = [i for i, n in enumerate(action_term._joint_names) if "CamShaft" in n]
    yaw_ids = [i for i, n in enumerate(robot.joint_names) if "Body_Hip" in n]
    masses = robot.root_physx_view.get_masses()[0].to(uenv.device)
    zero = torch.zeros(env.action_space.shape, device=uenv.device)
    gen = torch.Generator(device="cpu").manual_seed(7)

    trials = []
    last_settle = {}
    with torch.inference_mode():
        for trial in range(args_cli.trials):
            env.reset()
            jp = robot.data.joint_pos.clone()
            jitter = (torch.rand(jp.shape, generator=gen) * 0.04 - 0.02).to(jp.device)
            robot.write_joint_state_to_sim(jp + jitter, torch.zeros_like(jp))
            fh, leg_max, min_foot_z, tilt_max = [], 0.0, 1e9, 0.0
            for i in range(args_cli.settle):
                env.step(zero)
                p, r = pitch_roll(robot.data.root_quat_w[0])
                tilt_max = max(tilt_max, math.hypot(p, r))
                leg_f = torch.norm(cs.data.net_forces_w[0, leg_cfg.body_ids], dim=-1)
                leg_max = max(leg_max, float(leg_f.max()))
                min_foot_z = min(min_foot_z, float(robot.data.body_pos_w[0, foot_body_cfg.body_ids, 2].min()))
                if i >= args_cli.settle - 30:
                    fh.append(torch.norm(cs.data.net_forces_w[0, foot_cfg.body_ids], dim=-1).cpu())
            f = torch.stack(fh).mean(0)
            fa, fb = float(f[list(A_SET)].sum()), float(f[list(B_SET)].sum())
            p, r = pitch_roll(robot.data.root_quat_w[0])
            body_pos = robot.data.body_pos_w[0]
            com_w = (body_pos * masses.unsqueeze(1)).sum(0) / masses.sum()
            foot_pos = body_pos[foot_body_cfg.body_ids]
            tau = robot.data.applied_torque[0]
            trials.append({
                "trial": trial, "per_foot_force_N": [float(v) for v in f], "A_share": fa / max(fa + fb, 1e-9),
                "pitch_eq": p, "roll_eq": r, "tilt_max_during_settle": tilt_max,
                "root_z": float(robot.data.root_pos_w[0, 2]), "com_z": float(com_w[2]),
                "com_x_minus_foot_centroid_x": float(com_w[0] - foot_pos[:, 0].mean()),
                "min_foot_z": min_foot_z, "leg_link_contact_max_N": leg_max,
                "feet_loaded": int((f > 1.0).sum()),
                "joint_torque_abs_max": float(tau.abs().max()),
            })
            print(f"[trial {trial:02d}] A {100 * trials[-1]['A_share']:5.1f}% pitch {math.degrees(p):+5.2f} "
                  f"root_z {trials[-1]['root_z']:.4f} legF {leg_max:6.1f}N minfoot {min_foot_z:+.4f}", flush=True)
            if trial == args_cli.trials - 1:
                last_settle = {
                    "joint_pos": robot.data.joint_pos[0].cpu().numpy(),
                    "body_pos_w": body_pos.cpu().numpy(),
                    "root_pos_w": robot.data.root_pos_w[0].cpu().numpy(),
                    "root_quat_w": robot.data.root_quat_w[0].cpu().numpy(),
                    "foot_force_N": f.numpy(),
                    "com_w": com_w.cpu().numpy(),
                }
        # cam sweep from the last settled state
        act = zero.clone()
        for c in cam_cols:
            act[:, c] = 0.5
        sw_jp, sw_foot, sw_root, sw_quat, sw_legf, sw_footz = [], [], [], [], [], []
        for _ in range(args_cli.sweep_steps):
            env.step(act)
            sw_jp.append(robot.data.joint_pos[0].cpu().numpy())
            sw_foot.append(robot.data.body_pos_w[0, foot_body_cfg.body_ids].cpu().numpy())
            sw_root.append(robot.data.root_pos_w[0].cpu().numpy())
            sw_quat.append(robot.data.root_quat_w[0].cpu().numpy())
            sw_legf.append(torch.norm(cs.data.net_forces_w[0, leg_cfg.body_ids], dim=-1).cpu().numpy())
            sw_footz.append(float(robot.data.body_pos_w[0, foot_body_cfg.body_ids, 2].min()))
        sw_jp = np.asarray(sw_jp)
        yaw_range = {robot.joint_names[i]: [float(sw_jp[:, i].min()), float(sw_jp[:, i].max())] for i in yaw_ids}

    shares = [t["A_share"] for t in trials]
    report = {
        "tag": args_cli.tag, "usd": os.environ.get("KRABBY_HEX_USD_PATH", "<golden>"),
        "spawn_z_env": os.environ.get("KRABBY_HEX_SPAWN_Z", "<default>"),
        "trials": trials,
        "A_share_mean": float(np.mean(shares)), "A_share_sd": float(np.std(shares, ddof=1)) if len(shares) > 1 else 0.0,
        "n_upright": int(sum(1 for t in trials if t["tilt_max_during_settle"] < 0.5)),
        "n_penetrating": int(sum(1 for t in trials if t["min_foot_z"] < -0.01)),
        "leg_link_contact_max_N": float(max(t["leg_link_contact_max_N"] for t in trials)),
        "root_z_mean": float(np.mean([t["root_z"] for t in trials])),
        "com_z_mean": float(np.mean([t["com_z"] for t in trials])),
        "pitch_eq_mean_deg": float(np.degrees(np.mean([t["pitch_eq"] for t in trials]))),
        "sweep": {"yaw_range_rad": yaw_range,
                  "leg_link_contact_max_N": float(np.max(sw_legf)),
                  "min_foot_z": float(min(sw_footz))},
        "joint_names": list(robot.joint_names), "body_names": list(robot.body_names),
        "leg_link_names": leg_link_names,
        "masses_kg": [float(m) for m in masses.cpu()],
    }
    (out_dir / f"statics_{args_cli.tag}.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        out_dir / f"statics_{args_cli.tag}.npz",
        sweep_joint_pos=sw_jp, sweep_foot_pos_w=np.asarray(sw_foot), sweep_root_pos_w=np.asarray(sw_root),
        sweep_root_quat_w=np.asarray(sw_quat), sweep_leg_force=np.asarray(sw_legf), **last_settle,
    )
    print(f"[RESULT] {args_cli.tag}: A-share {100 * report['A_share_mean']:.1f}% sd {100 * report['A_share_sd']:.1f}% | "
          f"upright {report['n_upright']}/{args_cli.trials} | penetrating {report['n_penetrating']} | "
          f"leg-link contact max {report['leg_link_contact_max_N']:.1f} N (sweep {report['sweep']['leg_link_contact_max_N']:.1f} N) | "
          f"root_z {report['root_z_mean']:.4f} com_z {report['com_z_mean']:.4f} pitch {report['pitch_eq_mean_deg']:+.2f} deg", flush=True)
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
