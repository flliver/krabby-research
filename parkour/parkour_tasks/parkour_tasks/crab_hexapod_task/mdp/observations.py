"""Crab-hex observation terms (Go2-compatible shared stack stays in parkour_isaaclab)."""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

from parkour_isaaclab.envs import ParkourManagerBasedRLEnv
from parkour_isaaclab.envs.mdp.observations import ExtremeParkourObservations
from parkour_isaaclab.utils.nonfinite_logging import warn_if_nonfinite

# Extra proprio dims vs ``ExtremeParkourObservations``: body-frame planar linear velocity.
_CRAB_EXTRA_BASE_DIM = 2


class CrabHexParkourObservations(ExtremeParkourObservations):
    """``ExtremeParkourObservations`` with ``root_lin_vel_b[:, :2] * 2`` in the proprio block (dims 13–14)."""

    def __init__(self, cfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._obs_buf_dim += _CRAB_EXTRA_BASE_DIM
        self._obs_history_buffer = torch.zeros(
            self.num_envs, self.history_length, self._obs_buf_dim, device=self.device
        )

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
        delta_yaw = torch.where(on_flat, torch.zeros_like(self.delta_yaw), self.delta_yaw)
        delta_next_yaw = torch.where(on_flat, torch.zeros_like(self.delta_next_yaw), self.delta_next_yaw)
        commands = env.command_manager.get_command("base_velocity")
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
                self.asset.data.joint_pos - self.asset.data.default_joint_pos,
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
