"""Crab-hex observation terms (Go2-compatible shared stack stays in parkour_isaaclab)."""

from __future__ import annotations

import os

import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

from parkour_isaaclab.envs import ParkourManagerBasedRLEnv
from isaaclab.assets import Articulation
from isaaclab.managers import ObservationTermCfg

from parkour_isaaclab.envs.mdp.observations import ExtremeParkourObservations
from parkour_isaaclab.envs.mdp.parkours import ParkourEvent
from parkour_isaaclab.utils.nonfinite_logging import warn_if_nonfinite

from parkour_tasks.crab_hex_forward_task.mdp import crab_hex_linkage as linkage

# Extra proprio dims vs ``ExtremeParkourObservations``: body-frame planar linear velocity
# (2) + the gait clock's sin/cos (2, gait-formation-v2 Phase 1).
_CRAB_EXTRA_BASE_DIM = 4

# NOTE(hardware-measurements, 2026-08-20): the physical robot has NO foot contact sensors;
# stance is to be inferred from actuator motor current. The 6 per-leg channels that used to
# carry ground-truth footpad contact (deployable only with foot sensors) now carry a
# CURRENT-SENSE PROXY built from what the hardware can actually measure:
#   current ~ (rod force / rated force) while the rod is DRIVEN, ~0 when parked --
# lead screws are non-backdrivable, so a statically loaded but unmoving actuator draws no
# current (the worm-drive dead zone). Rod force = |applied joint torque| / moment arm.
# The drive gate uses the commanded rod speed from the action term's linkage layer.
# These constants are sensing-model parameters; the offline stance-detection study
# (campaign 2026-08-20_1506_hardware_morphology) evaluates them against ground truth.
_CURRENT_SENSE_NO_LOAD = 0.1  # normalized no-load current while driving
_CURRENT_SENSE_GATE_FRACTION = 0.3  # rod speed (fraction of rated) for a full drive gate
_CURRENT_SENSE_MAX = 1.5  # clip: stall current ~1.5x rated


class CrabHexParkourObservations(ExtremeParkourObservations):
    """``ExtremeParkourObservations`` with ``root_lin_vel_b[:, :2] * 2`` (dims 13-14) and the
    gait clock's sin/cos (dims 15-16) in the proprio block."""

    def __init__(self, cfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._obs_buf_dim += _CRAB_EXTRA_BASE_DIM
        self._obs_history_buffer = torch.zeros(
            self.num_envs, self.history_length, self._obs_buf_dim, device=self.device
        )
        # NOTE(cam-velocity-actions): under velocity control the shaft rotates continuously,
        # so its raw joint_pos grows without bound and would saturate the +-100 obs clip,
        # blinding the policy. Wrap those channels to [-pi, pi]: the cam map is 2*pi-periodic
        # (wrapped angle is information-complete), wrap is odd (mirror sign map unchanged),
        # and the obs dim is preserved (checkpoint/history compatible).
        self._cam_shaft_obs_ids, _ = self.asset.find_joints(
            [".*_Body_CamShaft_RevoluteJoint"], preserve_order=True
        )
        # Current-sense proxy joints, leg-aligned (FL, FR, ML, MR, RL, RR — articulation
        # order, matching the old contact fill's footpad-body order).
        pitch_ids, pitch_names = self.asset.find_joints(
            [".*_Hip_Femur_RevoluteJoint"], preserve_order=True
        )
        knee_ids, _ = self.asset.find_joints(
            [n.replace("_Hip_Femur_", "_Femur_Tibia_") for n in pitch_names],
            preserve_order=True,
        )
        self._pitch_joint_ids = pitch_ids
        self._knee_joint_ids = knee_ids

    def _get_contact_fill(self):
        """Motor-current-sense stance proxy (replaces ground-truth footpad contact).

        Returns 6 per-leg channels in roughly the same numeric range as the old contact
        bits (−0.5 idle .. ~+1.0 heavily loaded): max of the leg's hip/knee normalized
        current estimates. Rewards, terminations, and gait metrics keep using the
        privileged sim ContactSensor — only the policy-visible channels change.
        """
        if self.contact_sensor is None:
            return torch.zeros(self.num_envs, self._num_contact, device=self.device) - 0.5
        action_term = self._env.action_manager.get_term("joint_pos")
        tau = self.asset.data.applied_torque
        theta_h = self.asset.data.joint_pos[:, self._pitch_joint_ids]
        theta_k = self.asset.data.joint_pos[:, self._knee_joint_ids]
        # Moment arms are magnitude-symmetric, so the right legs' knee sign flip only
        # matters for the knee angle fed into the linkage; use |theta_k| convention-free.
        ma_h = linkage.hip_moment_arm(theta_h).clamp_min(1e-4)
        ma_k = linkage.knee_moment_arm(theta_h, theta_k.abs()).clamp_min(1e-4)
        force_h = tau[:, self._pitch_joint_ids].abs() / ma_h / linkage.HIP_FORCE_N
        force_k = tau[:, self._knee_joint_ids].abs() / ma_k / linkage.KNEE_FORCE_N
        gate_h = (
            action_term.hip_rod_speed.abs()
            / (_CURRENT_SENSE_GATE_FRACTION * linkage.HIP_SPEED_M_S)
        ).clamp(0.0, 1.0)
        gate_k = (
            action_term.knee_rod_speed.abs()
            / (_CURRENT_SENSE_GATE_FRACTION * linkage.KNEE_SPEED_M_S)
        ).clamp(0.0, 1.0)
        current_h = gate_h * (_CURRENT_SENSE_NO_LOAD + force_h).clamp(0.0, _CURRENT_SENSE_MAX)
        current_k = gate_k * (_CURRENT_SENSE_NO_LOAD + force_k).clamp(0.0, _CURRENT_SENSE_MAX)
        return torch.maximum(current_h, current_k) - 0.5

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        parkour_name: str,
        history_length: int,
    ) -> torch.Tensor:
        terrain_names = self.parkour_event.env_per_terrain_name
        on_flat = torch.as_tensor(
            terrain_names == "parkour_flat", dtype=torch.bool, device=self.device
        ).reshape(self.num_envs)
        invert_env_idx_tensor = on_flat.reshape(self.num_envs, 1)
        env_idx_tensor = (~on_flat).reshape(self.num_envs, 1)
        roll, pitch, yaw = euler_xyz_from_quat(self.asset.data.root_quat_w)
        imu_obs = torch.stack((wrap_to_pi(roll), wrap_to_pi(pitch)), dim=1).to(self.device)
        if env.common_step_counter % 5 == 0:
            self.delta_yaw = (self.parkour_event.target_yaw - wrap_to_pi(yaw)).reshape(self.num_envs)
            self.delta_next_yaw = (self.parkour_event.next_target_yaw - wrap_to_pi(yaw)).reshape(
                self.num_envs
            )
            self.measured_heights = self._get_heights()
        # Flat-walk: velocity command is straight ahead; parkour goal yaw biases crab/drift.
        # NOTE(PLAN F turning school, 2026-08-27): with KRABBY_HEADING set, flat tiles carry
        # the HEADING-COMMAND error in the steering channels instead of zero — otherwise the
        # ang-vel tracking reward pays for turning toward a target the policy cannot observe
        # (blind turning). Same channel the turn_walk_v1 eval probe injects, so training and
        # eval steering semantics match. Strictly env-var-gated: all pre-PLAN-F policies keep
        # their original zeroed-on-flat semantics.
        if os.environ.get("KRABBY_HEADING") is not None:
            cmd_term = env.command_manager.get_term("base_velocity")
            heading_err = wrap_to_pi(cmd_term.heading_target - self.asset.data.heading_w)
            delta_yaw = torch.where(on_flat, heading_err, self.delta_yaw)
            delta_next_yaw = torch.where(on_flat, heading_err, self.delta_next_yaw)
        else:
            delta_yaw = torch.where(on_flat, torch.zeros_like(self.delta_yaw), self.delta_yaw)
            delta_next_yaw = torch.where(on_flat, torch.zeros_like(self.delta_next_yaw), self.delta_next_yaw)
        commands = env.command_manager.get_command("base_velocity")
        clock_phase = env.action_manager.get_term("joint_pos").clock_phase
        joint_pos_delta = self.asset.data.joint_pos - self.asset.data.default_joint_pos
        joint_pos_delta[:, self._cam_shaft_obs_ids] = wrap_to_pi(
            joint_pos_delta[:, self._cam_shaft_obs_ids]
        )
        obs_buf = torch.cat(
            (
                self.asset.data.root_ang_vel_b * 0.25,
                imu_obs,
                0 * delta_yaw[:, None],
                delta_yaw[:, None],
                delta_next_yaw[:, None],
                0 * commands[:, 0:2],
                commands[:, 0:1],
                env_idx_tensor,
                invert_env_idx_tensor,
                self.asset.data.root_lin_vel_b[:, :2] * 2.0,
                # NOTE(gait-formation-v2 Phase 1): the gait clock, as sin/cos. Under the L/R
                # mirror a swapped-handedness gait is the same schedule advanced by pi, so
                # both dims carry mirror sign -1 (see crab_hex_mirror._HEAD_SIGNS).
                torch.sin(clock_phase)[:, None],
                torch.cos(clock_phase)[:, None],
                joint_pos_delta,
                self.asset.data.joint_vel * 0.05,
                env.action_manager.get_term("joint_pos").action_history_buf[:, -1],
                self._get_contact_fill(),
            ),
            dim=-1,
        )
        priv_explicit = self._get_priv_explicit()
        priv_latent = self._get_priv_latent()
        warn_if_nonfinite("observations.history_buffer", self._obs_history_buffer)
        self._obs_history_buffer = torch.nan_to_num(
            self._obs_history_buffer, nan=0.0, posinf=0.0, neginf=0.0
        )
        observations = torch.cat(
            [
                obs_buf,
                self.measured_heights,
                priv_explicit,
                priv_latent,
                self._obs_history_buffer.view(self.num_envs, -1),
            ],
            dim=-1,
        )
        obs_buf[:, 6:8] = 0
        self._obs_history_buffer = torch.where(
            (env.episode_length_buf <= 1)[:, None, None],
            torch.stack([obs_buf] * self.history_length, dim=1),
            torch.cat([self._obs_history_buffer[:, 1:], obs_buf.unsqueeze(1)], dim=1),
        )
        warn_if_nonfinite("observations.concat", observations)
        return torch.nan_to_num(observations, nan=0.0, posinf=0.0, neginf=0.0)


class CrabHexObservationDeltaYawOk(ManagerTermBase):
    """Student distillation gate: ``(num_envs, 1)`` bool for Isaac Lab obs dim probing."""

    def __init__(self, cfg: ObservationTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.delta_yaw = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids=None) -> None:
        pass

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        parkour_name: str,
        threshold: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        if env.common_step_counter % 5 == 0:
            parkour_event: ParkourEvent = env.parkour_manager.get_term(parkour_name)
            asset: Articulation = env.scene[asset_cfg.name]
            _, _, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
            self.delta_yaw = parkour_event.target_yaw - wrap_to_pi(yaw)
        return (self.delta_yaw < threshold).unsqueeze(-1)
