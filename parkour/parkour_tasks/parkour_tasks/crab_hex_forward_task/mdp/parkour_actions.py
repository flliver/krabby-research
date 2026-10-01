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
from parkour_tasks.crab_hex_forward_task.mdp.crab_hex_clock_reward import (
    clock_command_speed,
    park_clock_phase,
)

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
        # The commanded speed the cadence was last computed from (``clock_command_speed``),
        # kept for telemetry and for sibling tasks that read the gait clock's drive term.
        self.clock_cmd_speed = torch.zeros(self.num_envs, device=self.device)
        # RSI staging (crab_hex_rsi): events run before action-manager reset, so the RSI
        # event stages the reference clock here and reset() consumes it (NaN = not staged).
        self.rsi_clock_staged = torch.full((self.num_envs,), float("nan"), device=self.device)
        # NOTE(PPO-construction plan step 4, 2026-10-01): RSI CAM STAGING (``KRABBY_RSI_CAM_STAGE``,
        # DEFAULT OFF). Same staging pattern as the clock: the omni RSI event writes the bank
        # frame's cam shaft velocities (rad/s, ``_cam_shaft_joint_ids`` order; NaN = not staged)
        # and raises ``rsi_cam_stage_pending``; reset() seeds the applied cam target and the cam
        # columns of the action history from them (see ``_apply_rsi_cam_stage``). While the flag
        # is down -- every run without the knob -- reset() never touches these buffers.
        n_cam = len(self._cam_action_cols)
        self.rsi_cam_vel_staged = torch.full((self.num_envs, n_cam), float("nan"), device=self.device)
        self.rsi_cam_stage_pending = False
        # ``KRABBY_RSI_CAM_STAGE_DEBUG`` (default off): the frame's cam velocities again, for the
        # one-line step-0 read-out only (``_report_rsi_cam_stage``); never applied.
        self.rsi_cam_debug_jv = torch.full((self.num_envs, n_cam), float("nan"), device=self.device)
        self.rsi_cam_debug_pending = False

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

        # NOTE(motion-quality campaign, 2026-09-22): two OPTIONAL, DEFAULT-OFF knobs. Both are
        # read off attributes the shipped cfgs do not carry (``getattr`` -> off), so every
        # recorded run takes the untouched code path below.
        #
        # ``lpf_hz`` (``KRABBY_ACTION_LPF_HZ``): a first-order low-pass on the policy's ROD-LENGTH
        # command, BEFORE the rod slew limiter -- noise and dither on the pitch/knee channels
        # otherwise become a full-rate sawtooth on the rods. Runs every physics substep with
        # ``alpha = 1 - exp(-2 pi f_c physics_dt)``; the filter state is synced to the physical
        # rod length on the first substep after a reset, like the rod state itself.
        # ``rod_cmd_filtered`` exposes the filtered command, normalised to [-1, 1] per stroke
        # window, for the optional observation slot (``KRABBY_ACTION_LPF_OBS``).
        lpf_hz = getattr(cfg, "lpf_hz", None)
        self._lpf_alpha = (
            linkage.lpf_alpha(float(lpf_hz), float(env.physics_dt))
            if lpf_hz is not None and float(lpf_hz) > 0.0 else None
        )
        if self._lpf_alpha is not None:
            self._hip_rod_cmd_lpf = self._hip_rod_len.clone()
            self._knee_rod_cmd_lpf = self._knee_rod_len.clone()
        # ``omni_stand_park`` (``KRABBY_OMNI_STAND_PARK``, an ENV-cfg attribute): while the omni
        # command term reports the STAND slot, the gait clock advances (at full cadence) to its
        # next touchdown and holds there instead of freezing mid-swing; ``clock_parked`` is the
        # per-env flag the clock rewards read to treat all six legs as stance.
        self._stand_park = bool(getattr(env.cfg, "omni_stand_park", False))
        if self._stand_park:
            self.clock_parked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def rod_cmd_filtered(self) -> torch.Tensor:
        """``(N, 12)`` low-passed rod-length command, [hip x6 | knee x6] in leg order, each in
        [-1, 1] over its stroke window. Only exists with ``lpf_hz`` set."""
        if self._lpf_alpha is None:
            raise RuntimeError("rod_cmd_filtered needs the action cfg's lpf_hz (KRABBY_ACTION_LPF_HZ)")
        return torch.cat(
            (
                linkage.hip_rod_normalized(self._hip_rod_cmd_lpf),
                linkage.knee_rod_normalized(self._knee_rod_cmd_lpf),
            ),
            dim=-1,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._rod_sync_needed[env_ids] = True
        self._cam_vel_applied[env_ids] = 0.0
        # RSI cam staging (default off): runs after the event that staged it (env reset order:
        # events, then the action manager) and after the zeroing above, which it overrides for
        # the staged envs only.
        cam_staged = self.rsi_cam_stage_pending
        if cam_staged:
            self._apply_rsi_cam_stage(env_ids)
        if self.rsi_cam_debug_pending:
            self._report_rsi_cam_stage(env_ids, cam_staged)
        # Random initial clock phase: decorrelates the schedule across envs (no batch-wide
        # synchronized swing) and makes the policy phase-invariant. RSI-staged envs instead
        # take the reference phase (see crab_hex_rsi).
        fresh = torch.rand(len(env_ids), device=self.device) * (2.0 * math.pi)
        staged = self.rsi_clock_staged[env_ids]
        self.clock_phase[env_ids] = torch.where(torch.isfinite(staged), staged, fresh)
        self.rsi_clock_staged[env_ids] = float("nan")

    def _cam_raw_bounds(self, env_ids, like: torch.Tensor):
        """``(scale, lo, hi)`` shaped like ``like`` (``(k, n_cam)``) for the cam columns of
        ``env_ids``: rad/s per raw action unit (``CAM_VEL_SCALE`` = pi), and the raw bounds a cam
        action can take (+-1, narrowed by the term's own clip when it has one)."""
        cols = self._cam_action_cols
        if isinstance(self._scale, torch.Tensor):
            scale = self._scale[env_ids][:, cols]
        else:
            scale = torch.full_like(like, float(self._scale))
        lo = torch.full_like(scale, -1.0)
        hi = torch.full_like(scale, 1.0)
        if self.cfg.clip is not None:
            clip = self._clip[env_ids][:, cols]
            lo, hi = torch.maximum(lo, clip[..., 0]), torch.minimum(hi, clip[..., 1])
        return scale, lo, hi

    def _apply_rsi_cam_stage(self, env_ids) -> None:
        """Seed the RSI-staged envs' cam state from the bank frame (``KRABBY_RSI_CAM_STAGE``).

        For each env whose staged row is finite (``jv``, rad/s):

        * the APPLIED cam velocity target (the acceleration-limited state ``apply_actions`` slews
          from) = ``clamp(jv, -pi, +pi)`` -- i.e. +-``CAM_VEL_SCALE``, the full-action speed;
        * the cam columns of EVERY slot of the action history = ``clip(jv / pi, -1, +1)``, the raw
          action that commands that speed. The history holds CLIPPED raw actions, so this is what a
          policy already walking at that speed would have left there: the delayed read in
          ``process_actions`` (``history[:, -1 - delay]``) and the ``last_action`` observation of
          step 0 (``history[:, -1]``) both see it.

        Rod columns, ``_raw_actions`` (overwritten by the first ``process_actions`` before anything
        reads it) and every unstaged env are left exactly as the plain reset left them. The staging
        buffer is then cleared: a staged row is valid for the reset it was staged in only.
        """
        staged = self.rsi_cam_vel_staged[env_ids]
        ok = torch.isfinite(staged).all(dim=1, keepdim=True)
        jv = torch.nan_to_num(staged, nan=0.0)
        scale, lo, hi = self._cam_raw_bounds(env_ids, jv)
        applied = torch.minimum(torch.maximum(jv, lo * scale), hi * scale)
        raw = torch.minimum(torch.maximum(jv / scale, lo), hi)
        self._cam_vel_applied[env_ids] = torch.where(ok, applied, self._cam_vel_applied[env_ids])
        cols = self._cam_action_cols
        hist = self._action_history_buf[env_ids]
        cam_hist = hist[:, :, cols]
        hist[:, :, cols] = torch.where(ok.unsqueeze(1), raw.unsqueeze(1).expand_as(cam_hist), cam_hist)
        self._action_history_buf[env_ids] = hist
        self.rsi_cam_vel_staged.fill_(float("nan"))
        self.rsi_cam_stage_pending = False

    def _report_rsi_cam_stage(self, env_ids, staged: bool) -> None:
        """``KRABBY_RSI_CAM_STAGE_DEBUG``: one tagged line for the RSI-seeded envs of this reset
        (the rung-0b step-0 check, parsed by the campaign's ``scripts/rung0b_read.py``)::

            RSI_CAM_STAGE_DEBUG n_rsi=<n> stage=<on|off> target_err=<e> hist_err=<e>
                target_absmax=<a> hist_absmax=<a> bank_absmax=<a>

        computed PER ENV against the frame's own cam velocities ``jv`` (rad/s):

        * ``target_err`` = max |applied cam target - clamp(jv, lo*scale, hi*scale)| (rad/s);
        * ``hist_err`` = max over envs, history slots and cam columns of
          |history cam column - clip(jv/scale, lo, hi)|;
        * ``*_absmax`` = max |applied target|, |history cam columns|, |bank jv|.

        Staging on: both errors ~0. Staging off: both absmax read 0 while ``bank_absmax`` does not
        (and the errors are then the size of the frame's speeds)."""
        jv = self.rsi_cam_debug_jv[env_ids]
        rows = torch.isfinite(jv).all(dim=1)
        n = int(rows.sum())
        if n:
            jv_r = jv[rows]
            applied = self._cam_vel_applied[env_ids][rows]
            cam_hist = self._action_history_buf[env_ids][rows][:, :, self._cam_action_cols]
            scale, lo, hi = (b[rows] for b in self._cam_raw_bounds(env_ids, jv))
            want_target = torch.minimum(torch.maximum(jv_r, lo * scale), hi * scale)
            want_hist = torch.minimum(torch.maximum(jv_r / scale, lo), hi)
            target_err = float((applied - want_target).abs().max())
            hist_err = float((cam_hist - want_hist.unsqueeze(1)).abs().max())
            print(
                f"RSI_CAM_STAGE_DEBUG n_rsi={n} stage={'on' if staged else 'off'} "
                f"target_err={target_err:.6g} hist_err={hist_err:.6g} "
                f"target_absmax={float(applied.abs().max()):.6g} "
                f"hist_absmax={float(cam_hist.abs().max()):.6g} "
                f"bank_absmax={float(jv_r.abs().max()):.6g}",
                flush=True,
            )
        self.rsi_cam_debug_jv.fill_(float("nan"))
        self.rsi_cam_debug_pending = False

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
            if self._lpf_alpha is not None:
                self._hip_rod_cmd_lpf[sync] = self._hip_rod_len[sync]
                self._knee_rod_cmd_lpf[sync] = self._knee_rod_len[sync]
            self._rod_sync_needed[:] = False

        # --- hip pitch: rod-length command, window clamp, loaded-speed slew ---
        theta_h_cmd = self._processed_actions[:, self._pitch_action_cols]
        lam_h_cmd = linkage.hip_actuator_length(theta_h_cmd).clamp(
            linkage.HIP_LEN_MIN_M, linkage.HIP_LEN_MAX_M
        )
        if self._lpf_alpha is not None:
            self._hip_rod_cmd_lpf = linkage.lpf_step(self._hip_rod_cmd_lpf, lam_h_cmd, self._lpf_alpha)
            lam_h_cmd = self._hip_rod_cmd_lpf
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
        if self._lpf_alpha is not None:
            self._knee_rod_cmd_lpf = linkage.lpf_step(self._knee_rod_cmd_lpf, lam_k_cmd, self._lpf_alpha)
            lam_k_cmd = self._knee_rod_cmd_lpf
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
        # ``omni_command`` is absent on the forward (and sideways) cfg classes -> False, and
        # ``clock_command_speed`` is then exactly ``cmd[:, 0].abs()`` as before.
        v = clock_command_speed(
            cmd,
            bool(getattr(self._env.cfg, "omni_command", False)),
            float(getattr(self._env.cfg, "omni_clock_turn_radius", 0.0)),
        )
        self.clock_cmd_speed = v
        freq_rev_s = torch.where(
            v > linkage.dims.CLOCK_CMD_STOP_M_S,
            (v / linkage.dims.CLOCK_V_MAX_M_S).clamp(max=1.0) * linkage.dims.CLOCK_F_MAX_REV_S,
            torch.zeros_like(v),
        )
        advanced = torch.remainder(
            self.clock_phase + 2.0 * math.pi * freq_rev_s * self._env.step_dt, 2.0 * math.pi
        )
        if self._stand_park:
            # KRABBY_OMNI_STAND_PARK (default off): the STAND slot parks the clock at its next
            # touchdown (full cadence until it gets there) whatever the yaw-rate relay says.
            parked = self._env.command_manager.get_term("base_velocity").stand_mask()
            self.clock_parked = parked
            park_step = 2.0 * math.pi * linkage.dims.CLOCK_F_MAX_REV_S * self._env.step_dt
            advanced = torch.where(parked, park_clock_phase(self.clock_phase, park_step), advanced)
        self.clock_phase = advanced
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
