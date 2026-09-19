# SPDX-License-Identifier: BSD-3-Clause
"""Verify all 18 crab hex revolute joints move under joint_pos position commands."""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Drive each crab joint individually (+/- raw action) and report whether it moves."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Crab-Hex-Flat-Walk-Play-v0",
    help="Parkour crab env (play or train cfg both work).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--with_gravity",
    action="store_true",
    default=False,
    help="Keep gravity on (default: gravity off for unloaded actuation check).",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps_settle", type=int, default=48, help="Zero-action steps before each probe.")
parser.add_argument("--steps_drive", type=int, default=120, help="Steps holding the probe action.")
parser.add_argument(
    "--action_mag",
    type=float,
    default=1.0,
    help="Raw action on the probed joint (MDP clip/scale still apply).",
)
parser.add_argument(
    "--min_delta_rad",
    type=float,
    default=0.02,
    help="Minimum |joint_pos - baseline| to count as driven.",
)
parser.add_argument(
    "--min_vel_rad_s",
    type=float,
    default=0.15,
    help="Fallback pass if max |joint_vel| during drive exceeds this (with torque).",
)
parser.add_argument(
    "--hold_other_joints",
    action="store_true",
    default=True,
    help="PD-hold non-probed joints at default via corrective actions (default: on).",
)
parser.add_argument(
    "--no_hold_other_joints",
    action="store_false",
    dest="hold_other_joints",
    help="Disable holding other joints (legacy single-DOF probe).",
)
parser.add_argument(
    "--fix_base",
    action="store_true",
    default=False,
    help="Fix chassis to world during probe (experimental).",
)
parser.add_argument(
    "--pass_on_torque_frac",
    type=float,
    default=0.35,
    help="Pass if max |tau| >= this fraction of nominal actuator effort (when |dq| is tiny).",
)
parser.add_argument(
    "--only",
    type=str,
    default=None,
    help="Regex: probe only joints whose name matches (e.g. 'FR_Femur_Tibia').",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from _paths import parkour_root, parkour_scripts_dir

_parkour_root = parkour_root()
_parkour_scripts = parkour_scripts_dir()
for _p in (str(_parkour_scripts), str(_parkour_root / "parkour_tasks")):
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(_parkour_root))
sys.path.insert(0, str(_parkour_root / "parkour_tasks"))

import gymnasium as gym
import torch

import parkour_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _reset_env(env) -> None:
    with torch.inference_mode():
        env.reset()


def _step_zeros(env, n: int) -> None:
    device = env.unwrapped.device
    actions = torch.zeros(env.action_space.shape, device=device)
    with torch.inference_mode():
        for _ in range(n):
            env.step(actions)


def _hold_actions(
    robot,
    col: int,
    act_joint_ids: list[int],
    scale_vec: torch.Tensor,
    cam_cols: set[int],
    clip_lo: float,
    clip_hi: float,
    device: torch.device,
    hold_gain: float = 2.0,
) -> torch.Tensor:
    """Corrective actions (action-column space) to keep non-probed joints near default pose.

    NOTE: action columns are the action term's 18 actuated joints; the articulation has 24
    DOFs (6 passive Body_Hip + 6 CamShaft + 12 leg joints), so columns must be mapped to
    articulation indices via ``act_joint_ids`` — indexing robot data by column is wrong.
    Camshaft columns are VELOCITY channels: raw 0 already commands 0 rad/s (hold), and a
    position-error correction would be nonsense there.
    """
    art_ids = torch.tensor(act_joint_ids, device=device)
    q = robot.data.joint_pos[0, art_ids]
    q_def = robot.data.default_joint_pos[0, art_ids]
    actions = torch.zeros(len(act_joint_ids), device=device)
    for k in range(len(act_joint_ids)):
        if k == col or k in cam_cols:
            continue
        actions[k] = torch.clamp(hold_gain * (q_def[k] - q[k]) / scale_vec[k], clip_lo, clip_hi)
    return actions


def _step_joint_action(
    env,
    col: int,
    art_id: int,
    raw: float,
    n: int,
    *,
    hold_others: bool,
    act_joint_ids: list[int],
    scale_vec: torch.Tensor,
    cam_cols: set[int],
    clip_lo: float,
    clip_hi: float,
) -> tuple[float, float]:
    """Step with a single-column command; return (max |tau|, max |qdot|) on that joint."""
    device = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]
    max_tau = 0.0
    max_qd = 0.0
    with torch.inference_mode():
        for _ in range(n):
            actions = torch.zeros(env.action_space.shape, device=device)
            if hold_others:
                hold = _hold_actions(
                    robot, col, act_joint_ids, scale_vec, cam_cols, clip_lo, clip_hi, device
                )
                actions[0, :] = hold
            actions[..., col] = raw
            env.step(actions)
            max_tau = max(max_tau, robot.data.applied_torque[0, art_id].abs().item())
            max_qd = max(max_qd, robot.data.joint_vel[0, art_id].abs().item())
    return max_tau, max_qd


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if hasattr(env_cfg, "parkours") and env_cfg.parkours is not None:
        env_cfg.parkours.base_parkour.debug_vis = False
    if hasattr(env_cfg, "commands") and env_cfg.commands is not None:
        env_cfg.commands.base_velocity.debug_vis = False

    if not args_cli.with_gravity:
        env_cfg.sim.gravity = (0.0, 0.0, 0.0)
        if hasattr(env_cfg.scene, "robot") and env_cfg.scene.robot is not None:
            env_cfg.scene.robot.spawn.rigid_props.disable_gravity = True
    if args_cli.fix_base and hasattr(env_cfg.scene, "robot") and env_cfg.scene.robot is not None:
        env_cfg.scene.robot.spawn.articulation_props.fix_root_link = True

    env = gym.make(args_cli.task, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    joint_pos_term = env.unwrapped.action_manager.get_term("joint_pos")
    num_joints = robot.num_joints
    # Action columns are the term's actuated joints (18), NOT the articulation's DOFs (24
    # since the cam-mechanism migration added passive Body_Hip + CamShaft pairs). Probe in
    # column space and map to articulation indices for all robot.data reads.
    act_joint_ids = list(joint_pos_term._joint_ids)
    act_joint_names = list(joint_pos_term._joint_names)
    num_actions = len(act_joint_names)
    cam_cols = {i for i, n in enumerate(act_joint_names) if "CamShaft" in n}
    # scale may be a scalar or a per-joint dict resolved into a tensor by the action term
    # (cam channels are rad/s velocity scale, others rad position scale).
    if torch.is_tensor(joint_pos_term._scale):
        scale_vec = joint_pos_term._scale[0].detach().clone()
    else:
        scale_vec = torch.full(
            (num_actions,), float(joint_pos_term._scale), device=env.unwrapped.device
        )
    clip = joint_pos_term.cfg.clip
    if clip is None:
        clip_lo, clip_hi = -float("inf"), float("inf")
    elif hasattr(joint_pos_term, "_clip") and joint_pos_term._clip is not None:
        clip_lo = float(joint_pos_term._clip[0, 0, 0].item())
        clip_hi = float(joint_pos_term._clip[0, 0, 1].item())
    elif isinstance(clip, dict):
        lo, hi = next(iter(clip.values()))
        clip_lo, clip_hi = float(lo), float(hi)
    else:
        clip_lo, clip_hi = float(clip[0]), float(clip[1])

    def _nominal_effort(jname: str) -> float:
        # NOTE(hardware-measurements, 2026-08-20): matches the screw-emulator caps in
        # crab_hex_scene_cfg.py (hip 480, knee 80) and the hardware yaw motor (cam 20).
        if "Body_Hip" in jname:
            return 600.0
        if "Body_CamShaft" in jname:
            return 20.0
        if "Hip_Femur" in jname:
            return 480.0
        if "Femur_Tibia" in jname:
            return 80.0
        return 600.0

    print("\n=== Crab joint drive check ===", flush=True)
    print(f"task: {args_cli.task}", flush=True)
    print(f"num_envs: {env.unwrapped.scene.num_envs}", flush=True)
    print(f"gravity: {'on' if args_cli.with_gravity else 'off'}", flush=True)
    print(f"fix_base: {args_cli.fix_base}", flush=True)
    print(
        f"num_joints: {num_joints} articulation DOFs; probing {num_actions} actuated action columns",
        flush=True,
    )
    print(
        "action term per-joint scale: "
        + ", ".join(f"{n}={scale_vec[i].item():g}" for i, n in enumerate(act_joint_names)),
        flush=True,
    )
    print(f"cam velocity columns (rad/s semantics): {sorted(cam_cols)}", flush=True)
    print(f"action term clip: {clip}", flush=True)
    print(
        f"probe: action_mag={args_cli.action_mag}, settle={args_cli.steps_settle}, "
        f"drive={args_cli.steps_drive}, hold_other_joints={args_cli.hold_other_joints}",
        flush=True,
    )
    print(f"pass threshold: |delta q| >= {args_cli.min_delta_rad} rad\n", flush=True)

    if num_actions != 18:
        print(f"WARNING: expected 18 actuated joints, got {num_actions}", flush=True)

    limits = robot.data.soft_joint_pos_limits[0]

    results: list[dict] = []
    all_ok = True

    def _motion_ok(delta: float, max_tau: float, max_qd: float, nominal_eff: float) -> bool:
        if abs(delta) >= args_cli.min_delta_rad:
            return True
        if max_tau >= args_cli.pass_on_torque_frac * nominal_eff:
            return True
        return max_qd >= args_cli.min_vel_rad_s and max_tau >= 0.2 * nominal_eff

    body_names = list(robot.data.body_names)

    def _foot_body_idx(jname: str) -> int | None:
        leg = jname.split("_")[0]
        name = f"{leg}_Footpad"
        return body_names.index(name) if name in body_names else None

    def _probe_joint(c: int, drive_steps: int, action_mag: float) -> tuple[bool, float, float, float, float, bool, bool]:
        art_id = act_joint_ids[c]
        nominal_eff = _nominal_effort(act_joint_names[c])
        foot_idx = _foot_body_idx(act_joint_names[c])
        _reset_env(env)
        _step_zeros(env, args_cli.steps_settle)
        baseline = robot.data.joint_pos[0, art_id].item()
        foot0 = robot.data.body_pos_w[0, foot_idx].clone() if foot_idx is not None else None
        drive_kw = dict(
            hold_others=args_cli.hold_other_joints,
            act_joint_ids=act_joint_ids,
            scale_vec=scale_vec,
            cam_cols=cam_cols,
            clip_lo=clip_lo,
            clip_hi=clip_hi,
        )
        max_tau_plus, max_qd_plus = _step_joint_action(env, c, art_id, action_mag, drive_steps, **drive_kw)
        delta_plus = robot.data.joint_pos[0, art_id].item() - baseline
        ep_len = int(env.unwrapped.episode_length_buf[0].item())
        if ep_len < drive_steps:
            print(
                f"      WARNING: episode_length={ep_len} < drive steps {drive_steps} — env is "
                f"resetting mid-probe; results invalid (use gravity-on, check terminations)",
                flush=True,
            )
        if foot0 is not None:
            dfoot = robot.data.body_pos_w[0, foot_idx] - foot0
            print(
                f"      foot dxyz (raw {action_mag:+.2f}): "
                f"[{dfoot[0].item():+.4f}, {dfoot[1].item():+.4f}, {dfoot[2].item():+.4f}] m",
                flush=True,
            )
        _step_zeros(env, args_cli.steps_settle)
        baseline_minus = robot.data.joint_pos[0, art_id].item()
        foot0 = robot.data.body_pos_w[0, foot_idx].clone() if foot_idx is not None else None
        max_tau_minus, max_qd_minus = _step_joint_action(env, c, art_id, -action_mag, drive_steps, **drive_kw)
        delta_minus = robot.data.joint_pos[0, art_id].item() - baseline_minus
        if foot0 is not None:
            dfoot = robot.data.body_pos_w[0, foot_idx] - foot0
            print(
                f"      foot dxyz (raw {-action_mag:+.2f}): "
                f"[{dfoot[0].item():+.4f}, {dfoot[1].item():+.4f}, {dfoot[2].item():+.4f}] m",
                flush=True,
            )
        ok_plus = _motion_ok(delta_plus, max_tau_plus, max_qd_plus, nominal_eff)
        ok_minus = _motion_ok(delta_minus, max_tau_minus, max_qd_minus, nominal_eff)
        return (
            ok_plus or ok_minus,
            delta_plus,
            delta_minus,
            max_tau_plus,
            max_tau_minus,
            ok_plus,
            ok_minus,
        )

    import re

    probe_ids = [
        c for c in range(num_actions)
        if args_cli.only is None or re.search(args_cli.only, act_joint_names[c])
    ]
    if not probe_ids:
        print(f"ERROR: --only '{args_cli.only}' matched no joints", flush=True)
        sys.exit(2)

    for c in probe_ids:
        j = act_joint_ids[c]
        print(f"--- action {c + 1}/{num_actions}: {act_joint_names[c]} (dof {j}) ---", flush=True)
        default = robot.data.default_joint_pos[0, j].item()
        nominal_eff = _nominal_effort(act_joint_names[c])

        ok, delta_plus, delta_minus, max_tau_plus, max_tau_minus, ok_plus, ok_minus = _probe_joint(
            c, args_cli.steps_drive, args_cli.action_mag
        )
        if not ok:
            ok_retry, dp2, dm2, tp2, tm2, op2, om2 = _probe_joint(
                c, args_cli.steps_drive * 2, args_cli.action_mag
            )
            if ok_retry:
                ok, delta_plus, delta_minus = ok_retry, dp2, dm2
                max_tau_plus, max_tau_minus = max(max_tau_plus, tp2), max(max_tau_minus, tm2)
                ok_plus, ok_minus = op2, om2
            elif args_cli.action_mag < 2.0:
                mag2 = min(2.0, clip_hi)
                ok_retry2, dp3, dm3, tp3, tm3, op3, om3 = _probe_joint(
                    c, args_cli.steps_drive * 2, mag2
                )
                if ok_retry2:
                    ok, delta_plus, delta_minus = ok_retry2, dp3, dm3
                    max_tau_plus, max_tau_minus = max(max_tau_plus, tp3), max(max_tau_minus, tm3)
                    ok_plus, ok_minus = op3, om3
        all_ok = all_ok and ok

        lo, hi = limits[j, 0].item(), limits[j, 1].item()
        status = "OK" if ok else "FAIL"
        saturated = max(max_tau_plus, max_tau_minus) > 0.9 * nominal_eff
        results.append(
            {
                "idx": c,
                "name": act_joint_names[c],
                "ok": ok,
                "status": status,
            }
        )

        print(
            f"[{c:2d}] {status}  {act_joint_names[c]}\n"
            f"      default={default:+.4f}  lim=[{lo:+.3f}, {hi:+.3f}]  "
            f"nominal_effort={nominal_eff:.1f} Nm\n"
            f"      expected |dq|~{scale_vec[c].item() * args_cli.action_mag:.3f} "
            f"{'rad/s (velocity channel: dq grows with time)' if c in cam_cols else 'rad'}\n"
            f"      action +{args_cli.action_mag:+.2f} -> dq={delta_plus:+.4f}  "
            f"max|tau|={max_tau_plus:.1f}  ({'ok' if ok_plus else 'weak'}"
            f"{'; SAT' if saturated and not ok_plus else ''})\n"
            f"      action -{args_cli.action_mag:+.2f} -> dq={delta_minus:+.4f}  "
            f"max|tau|={max_tau_minus:.1f}  ({'ok' if ok_minus else 'weak'}"
            f"{'; SAT' if saturated and not ok_minus else ''})",
            flush=True,
        )

    env.close()

    n_ok = sum(1 for r in results if r["ok"])
    print("\n=== Summary ===", flush=True)
    print(f"Driven: {n_ok}/{len(probe_ids)}", flush=True)
    if all_ok:
        print("PASS: all joints showed measurable motion in at least one direction.", flush=True)
    else:
        failed = [r["name"] for r in results if not r["ok"]]
        print(f"FAIL: no motion above threshold for: {failed}", flush=True)
    print(flush=True)
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Flush the traceback BEFORE closing Kit: simulation_app.close() can hang, and the
        # default handler only prints after ``finally`` completes (observed: silent 1h spin).
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
