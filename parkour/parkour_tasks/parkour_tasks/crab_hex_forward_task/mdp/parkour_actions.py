"""Crab-hex action terms (clip-before-history for last_action alignment; Go2 keeps shared behavior)."""

from __future__ import annotations

import math

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from parkour_isaaclab.envs.mdp.parkour_actions import DelayedJointPositionActionCfg
from parkour_isaaclab.envs.mdp.parkour_actions.joint_actions import DelayedJointPositionAction

from parkour_tasks.crab_hex_forward_task.mdp import crab_hex_linkage as linkage
from parkour_tasks.crab_hex_forward_task.mdp.crab_hex_cam_mapping import cam_shaft_to_hip

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

_LEFT_LEGS = ("FL", "ML", "RL")


class CrabHexDelayedJointPositionAction(DelayedJointPositionAction):
    """Clip policy actions before the delay buffer so ``last_action`` obs matches applied commands.

    Also drives the crab-hex cam mechanism's kinematic coupling (see plan): each leg's
    ``*_Body_CamShaft_RevoluteJoint`` is a real, PD-actuated DOF (part of this term's normal
    joint_names set); each leg's ``*_Body_Hip_RevoluteJoint`` has its own (high-gain) actuator
    (see crab_hex_scene_cfg.py) whose *position target* is computed from the shaft's current
    state via ``crab_hex_cam_mapping.cam_shaft_to_hip`` every physics substep.

    NOTE(cam-mechanism-migration): an earlier version of this coupling used
    ``Articulation.write_joint_state_to_sim`` (direct state teleportation) instead of a position
    target. That froze the *entire* articulation's dynamics -- including the completely separate
    floating-base chassis root, not just the targeted joint -- when called every substep
    (confirmed via A/B test: disabling the write alone restored normal free-fall). Repeatedly
    teleporting DOF state each substep is not how PhysX's articulation solver expects continuous
    control; ``set_joint_position_target`` (the same mechanism every other joint already uses) is
    the correct approach. Tracking is now approximate (real PD tracking error), not exact.

    NOTE(hardware-measurements, 2026-08-20): the hip-pitch and knee joints are driven by
    LINEAR ACTUATORS on the real robot (lead screws: non-backdrivable, rod speed limited),
    so this term now runs their position targets through ``crab_hex_linkage``:

    - The policy's pitch commands are converted to rod-length commands, clamped to the
      physical 450-650 mm window, and SLEW-RATE-LIMITED at the loaded rod speed each
      substep. The joint-level actuator stays a rigid high-gain PD (emulating the
      self-locking screw, exactly like the yaw linkage emulator) -- static holding is
      free on hardware, so speed, not torque, is the binding constraint that must be
      enforced here.
    - The knee actuator spans hip -> tibia across the femur, so the knee's rod length is
      commanded (policy knee channel interpreted at the DEFAULT hip angle) and its joint
      target is re-solved from the hip's COMMANDED trajectory every substep (target-side
      coupling; see NOTE(coupling-stability) below): pitching the femur swings the knee
      as the hardware linkage does. The rod-window clamp also makes the unreachable
      joint-space corner (hip full-down + knee full-fold) physically unreachable in sim.
    - Right legs (FR/MR/RR) carry a 180-deg Z joint-frame flip: their knee sign is
      mirrored into the linkage's left-leg convention and back.
    """

    def __init__(self, cfg: DelayedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # NOTE(cam-mechanism-migration): resolve CamShaft -> Body_Hip joint index pairs, one
        # pair per converted leg (starts at 1 leg in Phase A, grows to 6 in Phase C -- resolved
        # dynamically from whatever *_Body_CamShaft_RevoluteJoint joints exist in the loaded USD,
        # no code change needed as legs are propagated).
        shaft_ids, shaft_names = self._asset.find_joints([".*_Body_CamShaft_RevoluteJoint"], preserve_order=True)
        hip_names = [name.replace("_Body_CamShaft_RevoluteJoint", "_Body_Hip_RevoluteJoint") for name in shaft_names]
        hip_ids, _ = self._asset.find_joints(hip_names, preserve_order=True)
        self._cam_shaft_joint_ids = shaft_ids
        self._cam_hip_joint_ids = hip_ids
        # NOTE(cam-velocity-actions): the 6 camshaft action channels are VELOCITY targets
        # (rad/s), matching the real quick-return linkage whose motor spins continuously in
        # one direction. Their offset must be zero (default joint *velocity*), not the default
        # joint position that use_default_offset injects for the position channels.
        self._cam_action_cols = [
            i for i, name in enumerate(self._joint_names) if name in set(shaft_names)
        ]
        if self._cam_action_cols:
            self._offset[:, self._cam_action_cols] = 0.0
        # NOTE(gait-formation Phase 0, 2026-08-20): commanded cam speed is
        # ACCELERATION-slew-limited (crab_hex_dimensions.CAM_ACCEL_LIMIT_RAD_S2) — the
        # hardware gearmotor ramps, it does not step, and the instant-full-throttle
        # wheelie (both smoke tests) is removed by construction.
        self._cam_vel_applied = torch.zeros(
            (self.num_envs, len(self._cam_action_cols)), device=self.device
        )

        # NOTE(gait-formation-v2 Phase 1, 2026-08-22): the gait clock. A per-env monotonic
        # phase (rad) advanced each env step at a cadence proportional to the commanded
        # forward speed, frozen while the command is a stop (below the env's lin-vel clip).
        # The clock-referenced contact-schedule reward and the clock sin/cos observation both
        # read this buffer, so the schedule the reward pays for and the phase the policy sees
        # are one and the same. Cadence endpoints are anchored to hardware: full command
        # (CLOCK_V_MAX) maps to the cam motor's full speed (CAM_VEL_SCALE = pi rad/s = 0.5
        # rev/s), per the Phase-0 scripted-gait calibration.
        self.clock_phase = torch.zeros(self.num_envs, device=self.device)
        # RSI staging (crab_hex_rsi): events run before action-manager reset, so the RSI
        # event stages the reference clock here and reset() consumes it (NaN = not staged).
        self.rsi_clock_staged = torch.full((self.num_envs,), float("nan"), device=self.device)

        # NOTE(hardware-measurements): resolve the pitch linkage joints, leg-aligned. Each
        # index i below refers to the same leg across all four lists/tensors.
        pitch_ids, pitch_names = self._asset.find_joints([".*_Hip_Femur_RevoluteJoint"], preserve_order=True)
        knee_names = [n.replace("_Hip_Femur_RevoluteJoint", "_Femur_Tibia_RevoluteJoint") for n in pitch_names]
        knee_ids, _ = self._asset.find_joints(knee_names, preserve_order=True)
        self._pitch_joint_ids = pitch_ids
        self._knee_joint_ids = knee_ids
        self._pitch_action_cols = [self._joint_names.index(n) for n in pitch_names]
        self._knee_action_cols = [self._joint_names.index(n) for n in knee_names]
        # +1 for left legs, -1 for right legs (the linkage math is left-leg convention).
        self._knee_sign = torch.tensor(
            [1.0 if n.split("_")[0] in _LEFT_LEGS else -1.0 for n in knee_names],
            device=self.device,
        ).unsqueeze(0)
        # Rod-length state (the physical actuator position), synced from the live joint
        # state on the first substep after each reset (events may randomize joint pos).
        self._hip_rod_len = torch.full(
            (self.num_envs, len(pitch_ids)),
            (linkage.HIP_LEN_MIN_M + linkage.HIP_LEN_MAX_M) / 2.0,
            device=self.device,
        )
        self._knee_rod_len = torch.full(
            (self.num_envs, len(knee_ids)),
            (linkage.KNEE_LEN_MIN_M + linkage.KNEE_LEN_MAX_M) / 2.0,
            device=self.device,
        )
        self._rod_sync_needed = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        # Last-substep rod speeds (m/s), exposed for the motor-current-sense observation
        # proxy (a lead-screw motor draws current only while DRIVING; static holding is
        # free -- the worm-drive dead zone the current-sense obs must model honestly).
        self.hip_rod_speed = torch.zeros((self.num_envs, len(pitch_ids)), device=self.device)
        self.knee_rod_speed = torch.zeros((self.num_envs, len(knee_ids)), device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._rod_sync_needed[env_ids] = True
        self._cam_vel_applied[env_ids] = 0.0
        # Random initial clock phase: decorrelates the schedule across envs (no batch-wide
        # synchronized swing) and makes the policy phase-invariant. RSI-staged envs instead
        # take the reference phase (see crab_hex_rsi).
        fresh = torch.rand(len(env_ids), device=self.device) * (2.0 * math.pi)
        staged = self.rsi_clock_staged[env_ids]
        self.clock_phase[env_ids] = torch.where(torch.isfinite(staged), staged, fresh)
        self.rsi_clock_staged[env_ids] = float("nan")

    def _clip_raw_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.cfg.clip is None:
            return actions
        return torch.clamp(actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1])

    def _apply_pitch_linkage(self):
        """Linear-actuator layer for hip-pitch + knee: rod-length clamp + speed slew +
        hip->knee coupling. Runs every physics substep (dt = env.physics_dt)."""
        dt = self._env.physics_dt
        theta_h_live = self._asset.data.joint_pos[:, self._pitch_joint_ids]
        theta_k_live = self._asset.data.joint_pos[:, self._knee_joint_ids] * self._knee_sign
        if not torch.isfinite(theta_h_live).all() or not torch.isfinite(theta_k_live).all():
            theta_h_live = torch.nan_to_num(theta_h_live, nan=0.0, posinf=0.0, neginf=0.0)
            theta_k_live = torch.nan_to_num(theta_k_live, nan=0.0, posinf=0.0, neginf=0.0)

        # Sync rod state from the live joint state on the first substep after a reset.
        if self._rod_sync_needed.any():
            sync = self._rod_sync_needed
            self._hip_rod_len[sync] = linkage.hip_actuator_length(theta_h_live[sync])
            self._knee_rod_len[sync] = linkage.knee_actuator_length(
                theta_h_live[sync], theta_k_live[sync]
            )
            self._rod_sync_needed[:] = False

        # --- hip pitch: rod-length command, window clamp, loaded-speed slew ---
        theta_h_cmd = self._processed_actions[:, self._pitch_action_cols]
        lam_h_cmd = linkage.hip_actuator_length(theta_h_cmd).clamp(
            linkage.HIP_LEN_MIN_M, linkage.HIP_LEN_MAX_M
        )
        max_step_h = linkage.HIP_SPEED_M_S * dt
        dlam_h = (lam_h_cmd - self._hip_rod_len).clamp(-max_step_h, max_step_h)
        self._hip_rod_len += dlam_h
        self.hip_rod_speed = dlam_h / dt
        theta_h_tgt = linkage.hip_angle_from_length(self._hip_rod_len)
        # Velocity feedforward: rod speed through the (signed-positive) moment arm.
        omega_h_tgt = (dlam_h / dt) / linkage.hip_moment_arm(theta_h_tgt).clamp_min(1e-4)
        self._asset.set_joint_position_target(theta_h_tgt, joint_ids=self._pitch_joint_ids)
        self._asset.set_joint_velocity_target(omega_h_tgt, joint_ids=self._pitch_joint_ids)

        # --- knee: rod command interpreted at the DEFAULT hip angle, re-solved at the
        # hip's TARGET angle (the hardware coupling), window clamp, loaded-speed slew ---
        # NOTE(coupling-stability): the coupling is computed from the hip's COMMANDED
        # trajectory (theta_h_tgt from the rod state above), never the live joint state.
        # A first implementation used theta_h_live and destabilized the zero-action
        # settle: live hip deflection -> instant knee retarget through the knee PD's
        # phase lag -> energy pumped into the body's rocking mode (growing pitch
        # oscillation, worse with stiffer knees). The hardware's rigid kinematic loop is
        # energy-conserving; a lagged live-state PD emulation of it is not. Target-side
        # coupling keeps the mechanism's kinematics on the commanded path and is stable
        # by construction (targets evolve only at rod speed).
        theta_h_default = self._asset.data.default_joint_pos[:, self._pitch_joint_ids]
        theta_k_cmd = self._processed_actions[:, self._knee_action_cols] * self._knee_sign
        lam_k_cmd = linkage.knee_actuator_length(theta_h_default, theta_k_cmd).clamp(
            linkage.KNEE_LEN_MIN_M, linkage.KNEE_LEN_MAX_M
        )
        max_step_k = linkage.KNEE_SPEED_M_S * dt
        dlam_k = (lam_k_cmd - self._knee_rod_len).clamp(-max_step_k, max_step_k)
        self._knee_rod_len += dlam_k
        self.knee_rod_speed = dlam_k / dt
        theta_k_tgt = linkage.knee_angle_from_length(self._knee_rod_len, theta_h_tgt)
        # Velocity feedforward: rod motion plus the hip-coupling term at constant rod
        # length, both evaluated on the commanded trajectory.
        ma_k = linkage.knee_moment_arm(theta_h_tgt, theta_k_tgt).clamp_min(1e-4)
        coupling = linkage.knee_hip_coupling(theta_h_tgt, theta_k_tgt)
        omega_k_tgt = (dlam_k / dt - coupling * omega_h_tgt) / ma_k
        self._asset.set_joint_position_target(
            theta_k_tgt * self._knee_sign, joint_ids=self._knee_joint_ids
        )
        self._asset.set_joint_velocity_target(
            omega_k_tgt * self._knee_sign, joint_ids=self._knee_joint_ids
        )

    def apply_actions(self):
        # Position targets for all 18 columns (inert for the camshaft: its actuator runs
        # stiffness=0, so the position term contributes no torque), then velocity targets on
        # the 6 cam columns — processed = raw * scale + 0 offset = rad/s.
        super().apply_actions()
        if self._pitch_action_cols:
            self._apply_pitch_linkage()
        if self._cam_action_cols:
            cam_cmd = self._processed_actions[:, self._cam_action_cols]
            max_step = linkage.dims.CAM_ACCEL_LIMIT_RAD_S2 * self._env.physics_dt
            self._cam_vel_applied += (cam_cmd - self._cam_vel_applied).clamp(-max_step, max_step)
            self._asset.set_joint_velocity_target(
                self._cam_vel_applied, joint_ids=self._cam_shaft_joint_ids
            )
        if self._cam_shaft_joint_ids:
            theta_shaft = self._asset.data.joint_pos[:, self._cam_shaft_joint_ids]
            omega_shaft = self._asset.data.joint_vel[:, self._cam_shaft_joint_ids]
            # NOTE(cam-mechanism-migration): a freshly-added joint can transiently report
            # non-finite pos/vel from PhysX for the first substep or two before it fully settles
            # (observed at env reset). Sanitizing here is a hard guarantee we never feed NaN into
            # Body_Hip's position-target actuator below -- unlike an observation-level NaN (which
            # the repo already sanitizes and self-heals), a NaN position target would corrupt the
            # PD effort computation for that joint every subsequent step.
            if not torch.isfinite(theta_shaft).all() or not torch.isfinite(omega_shaft).all():
                theta_shaft = torch.nan_to_num(theta_shaft, nan=0.0, posinf=0.0, neginf=0.0)
                omega_shaft = torch.nan_to_num(omega_shaft, nan=0.0, posinf=0.0, neginf=0.0)
            theta_hip, omega_hip = cam_shaft_to_hip(theta_shaft, omega_shaft)
            self._asset.set_joint_position_target(theta_hip, joint_ids=self._cam_hip_joint_ids)
            # NOTE(cam-velocity-actions): also feed the linkage's velocity as a target — the
            # PD's damping term then acts as feedforward instead of braking against the
            # legitimate hip motion. Without it the hip lags/overshoots at the quick-return
            # velocity peak (~0.92x shaft speed) under continuous spin.
            self._asset.set_joint_velocity_target(omega_hip, joint_ids=self._cam_hip_joint_ids)

    def process_actions(self, actions: torch.Tensor):
        # Gait-clock advance (once per env step, before physics): cadence linear in the
        # commanded forward speed, frozen on stop commands. See __init__ NOTE.
        cmd = self._env.command_manager.get_command("base_velocity")
        v = cmd[:, 0].abs()
        freq_rev_s = torch.where(
            v > linkage.dims.CLOCK_CMD_STOP_M_S,
            (v / linkage.dims.CLOCK_V_MAX_M_S).clamp(max=1.0) * linkage.dims.CLOCK_F_MAX_REV_S,
            torch.zeros_like(v),
        )
        self.clock_phase = torch.remainder(
            self.clock_phase + 2.0 * math.pi * freq_rev_s * self._env.step_dt, 2.0 * math.pi
        )
        if self.env.common_step_counter % self._delay_update_global_steps == 0:
            if len(self._action_delay_steps) != 0:
                self.delay = torch.tensor(
                    self._action_delay_steps.pop(0), device=self.device, dtype=torch.float
                )
        clipped_actions = self._clip_raw_actions(actions)
        self._action_history_buf = torch.cat(
            [self._action_history_buf[:, 1:].clone(), clipped_actions[:, None, :].clone()], dim=1
        )
        indices = -1 - self.delay
        if self._use_delay:
            self._raw_actions[:] = self._action_history_buf[:, indices.long()]
        else:
            self._raw_actions[:] = clipped_actions
        self._processed_actions = self._raw_actions * self._scale + self._offset


@configclass
class CrabHexDelayedJointPositionActionCfg(DelayedJointPositionActionCfg):
    class_type: type[ActionTerm] = CrabHexDelayedJointPositionAction
