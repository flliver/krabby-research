# SPDX-License-Identifier: BSD-3-Clause
"""PLAN E Phase 0, probe v2: scripted gait WITH cam-phase-synchronized knee lift.

Probe v1 result: pure cam spin (all feet always grounded) produces no net locomotion --
expected mechanically: with Coulomb friction and both tripods always loaded, the sets'
instantaneous friction forces cancel regardless of quick-return timing. Drag-gait
platforms locomote because recirculating legs UNLOAD; on this plant that is the knee
actuators' job.

v2 adds a half-cycle knee-lift window per leg, keyed to that leg's own cam angle, and
empirically sweeps the three coordination unknowns instead of deriving them:
  - spin direction: omega in {+w, -w}
  - lift-window center: cam angle 0 (slow/power region) vs pi (fast/return region)
  - knee-lift sign per side: {+, -} (which knee direction lifts the toe)
8 combos x 15 s at w=0.5 (4 s cycle -- the only speed that survived v1). The best
combo IS the mechanism's intended coordination; its npz becomes the reference candidate.

Run (from krabby-research/parkour):
  <venv-python> ../parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-22_1200_gait_formation_v2/scripted_gait_probe_v2.py --headless
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Crab-Hex-Flat-Walk-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--settle_steps", type=int, default=150)
parser.add_argument("--hold_steps", type=int, default=750)
parser.add_argument("--w", type=float, default=0.5, help="cam speed as fraction of CAM_VEL_SCALE")
parser.add_argument("--lift_action", type=float, default=1.0)
parser.add_argument("--duty_cos", type=float, default=0.0,
                    help="lift when cos(cam - phase_center) > this; 0.0 = half-cycle window, 0.5 = third-cycle")
parser.add_argument("--best_combo", action="store_true",
                    help="run only the propulsive combo (dw=+1, ph=0, ks=+1) found by the full sweep")
parser.add_argument("--offset_pattern", type=str, default="setAB",
                    choices=["setAB", "side_corrected", "all_zero", "left_right"],
                    help="per-leg cam phase offsets: setAB = tripod sets 0/pi in cam space (v1/v2); "
                         "side_corrected = tripod offsets PLUS pi on right-side legs (compensates the "
                         "mirrored joint frames under the side-agnostic cam mapping); all_zero / "
                         "left_right = controls")
parser.add_argument("--tag_suffix", type=str, default="")
parser.add_argument("--out_dir", type=str, default=str(Path(__file__).parent))
_PARKOUR_ROOT = Path("/home/nickmagus/krabby/krabby-research/parkour")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, str(_PARKOUR_ROOT))
sys.path.insert(0, str(_PARKOUR_ROOT / "parkour_tasks"))

import math

import gymnasium as gym
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg

import parkour_tasks  # noqa: F401

TRIPOD_A = ("FL", "MR", "RL")
LEFT = ("FL", "ML", "RL")
FOOT_NAMES = ["FL_Footpad", "FR_Footpad", "ML_Footpad", "MR_Footpad", "RL_Footpad", "RR_Footpad"]


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    combos = ([(+1, 0.0, +1)] if args_cli.best_combo
              else [(dw, ph, ks) for dw in (+1, -1) for ph in (0.0, math.pi) for ks in (+1, -1)])
    stagger_steps = int(round((math.pi / (args_cli.w * math.pi)) / 0.02))
    total = len(combos) * (args_cli.settle_steps + stagger_steps + args_cli.hold_steps)
    env = gym.wrappers.RecordVideo(
        env, video_folder=str(Path(args_cli.out_dir) / "videos_v2"),
        step_trigger=lambda step: step == 0, video_length=total, disable_logger=True,
    )
    env.reset()
    uenv = env.unwrapped
    device = uenv.device
    action_dim = env.action_space.shape[1]
    action_term = uenv.action_manager.get_term("joint_pos")
    names = action_term._joint_names
    legs = [n.split("_")[0] for n in names]
    cam_cols = {legs[i]: i for i, n in enumerate(names) if "CamShaft" in n}
    knee_cols = {legs[i]: i for i, n in enumerate(names) if "Femur_Tibia" in n}
    robot = uenv.scene["robot"]
    cam_joint_ids, cam_joint_names = robot.find_joints([".*_Body_CamShaft_RevoluteJoint"], preserve_order=True)
    cam_leg_of_joint = [n.split("_")[0] for n in cam_joint_names]
    contact = uenv.scene.sensors["contact_forces"]
    foot_ids, _ = contact.find_bodies(FOOT_NAMES, preserve_order=True)
    foot_body_ids, _ = robot.find_bodies(FOOT_NAMES, preserve_order=True)
    # PLAN G (2026-09-02): leg-link contact + all-env recording for the morphology probe matrix.
    leg_ids, leg_names = contact.find_bodies([".*_Hip", ".*_Femur", ".*_Tibia"])
    dt = uenv.step_dt
    print(f"[probe2] cams={cam_cols} knees={knee_cols} w={args_cli.w}", flush=True)

    with torch.inference_mode():
        for dw, phase_center, knee_sign in combos:
            tag = f"w{dw * args_cli.w:+.2f}_ph{phase_center:.2f}_ks{knee_sign:+d}{args_cli.tag_suffix}"
            act = torch.zeros((args_cli.num_envs, action_dim), device=device)
            env.reset()
            for _ in range(args_cli.settle_steps):
                env.step(act)
            # Per-leg cam phase offsets, realized by staggered starts: a leg whose offset is
            # pi starts half a period EARLIER than a 0-offset leg (equal ramps cancel).
            if args_cli.offset_pattern == "setAB":
                offsets = {leg: (0.0 if leg in TRIPOD_A else math.pi) for leg in cam_cols}
            elif args_cli.offset_pattern == "side_corrected":
                offsets = {leg: ((0.0 if leg in TRIPOD_A else math.pi)
                                 + (0.0 if leg in LEFT else math.pi)) % (2 * math.pi)
                           for leg in cam_cols}
            elif args_cli.offset_pattern == "all_zero":
                offsets = {leg: 0.0 for leg in cam_cols}
            else:  # left_right
                offsets = {leg: (0.0 if leg in LEFT else math.pi) for leg in cam_cols}
            # start the pi-offset group first, then the 0-offset group after half a period
            for leg, c in cam_cols.items():
                if offsets[leg] > 1e-6:
                    act[:, c] = dw * args_cli.w
            for _ in range(stagger_steps):
                env.step(act)
            for leg, c in cam_cols.items():
                act[:, c] = dw * args_cli.w
            rec = {k: [] for k in ("root_lin_vel_b", "root_pos_w", "root_quat_w", "cam_pos",
                                   "joint_pos", "joint_vel", "foot_force_N", "done",
                                   "foot_pos_w", "foot_lin_vel_w",
                                   "done_all", "root_lin_vel_b_all", "root_pos_w_all", "root_quat_w_all",
                                   "foot_force_N_all", "foot_pos_w_all", "leg_contact_N_all")}
            for t in range(args_cli.hold_steps):
                cam_angles = robot.data.joint_pos[0, cam_joint_ids]
                for j, leg in enumerate(cam_leg_of_joint):
                    in_window = math.cos(float(cam_angles[j]) - phase_center) > args_cli.duty_cos
                    side = 1.0 if leg in LEFT else -1.0
                    act[:, knee_cols[leg]] = (
                        knee_sign * side * args_cli.lift_action if in_window else 0.0
                    )
                _, _, terminated, truncated, _ = env.step(act)
                d = robot.data
                rec["root_lin_vel_b"].append(d.root_lin_vel_b[0].cpu().numpy().copy())
                rec["root_pos_w"].append(d.root_pos_w[0].cpu().numpy().copy())
                rec["root_quat_w"].append(d.root_quat_w[0].cpu().numpy().copy())
                rec["cam_pos"].append(d.joint_pos[0, cam_joint_ids].cpu().numpy().copy())
                rec["joint_pos"].append(d.joint_pos[0].cpu().numpy().copy())
                rec["joint_vel"].append(d.joint_vel[0].cpu().numpy().copy())
                rec["foot_force_N"].append(
                    contact.data.net_forces_w[0, foot_ids].norm(dim=-1).cpu().numpy().copy())
                rec["foot_pos_w"].append(d.body_pos_w[0, foot_body_ids].cpu().numpy().copy())
                rec["foot_lin_vel_w"].append(d.body_lin_vel_w[0, foot_body_ids].cpu().numpy().copy())
                rec["done"].append(bool(terminated[0] or truncated[0]))
                rec["done_all"].append((terminated | truncated).cpu().numpy().copy())
                rec["root_lin_vel_b_all"].append(d.root_lin_vel_b.cpu().numpy().copy())
                rec["root_pos_w_all"].append(d.root_pos_w.cpu().numpy().copy())
                rec["root_quat_w_all"].append(d.root_quat_w.cpu().numpy().copy())
                rec["foot_force_N_all"].append(contact.data.net_forces_w[:, foot_ids].norm(dim=-1).cpu().numpy().copy())
                rec["foot_pos_w_all"].append(d.body_pos_w[:, foot_body_ids].cpu().numpy().copy())
                rec["leg_contact_N_all"].append(contact.data.net_forces_w[:, leg_ids].norm(dim=-1).cpu().numpy().copy())
            arrays = {k: np.asarray(v) for k, v in rec.items()}
            arrays["dt"] = np.asarray(dt)
            arrays["omega_rad_s"] = np.asarray(dw * args_cli.w * math.pi)
            arrays["phase_center"] = np.asarray(phase_center)
            arrays["knee_sign"] = np.asarray(knee_sign)
            arrays["leg_link_names"] = np.asarray(leg_names)
            np.savez_compressed(Path(args_cli.out_dir) / f"probe2_{tag}.npz", **arrays)
            vx = arrays["root_lin_vel_b"][:, 0]
            done = arrays["done"]
            fell = bool(done.any())
            n = int(np.argmax(done)) if fell else len(done)
            print(f"[probe2] {tag}: mean vx {float(vx[:n].mean()):+.3f} m/s over {n} steps | fell={fell}",
                  flush=True)
    env.close()
    print("[probe2] DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
