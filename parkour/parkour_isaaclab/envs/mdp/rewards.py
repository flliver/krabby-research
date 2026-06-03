from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation
from isaaclab.utils.math  import euler_xyz_from_quat, wrap_to_pi, quat_apply
from parkour_isaaclab.envs.mdp.parkours import ParkourEvent 
from collections.abc import Sequence

if TYPE_CHECKING:
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

import cv2
import numpy as np 

class reward_feet_edge(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor | None = env.scene.sensors.get(cfg.params["sensor_cfg"].name)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]
        self.parkour_event: ParkourEvent = env.parkour_manager.get_term(cfg.params["parkour_name"])
        self.horizontal_scale = env.scene.terrain.cfg.terrain_generator.horizontal_scale
        size_x, size_y = env.scene.terrain.cfg.terrain_generator.size
        self.rows_offset = (size_x * env.scene.terrain.cfg.terrain_generator.num_rows/2)
        self.cols_offset = (size_y * env.scene.terrain.cfg.terrain_generator.num_cols/2)
        total_x_edge_maskes = torch.from_numpy(self.parkour_event.terrain.terrain_generator_class.x_edge_maskes).to(device = self.device)
        self.x_edge_masks_tensor = total_x_edge_maskes.permute(0, 2, 1, 3).reshape(
            env.scene.terrain.terrain_generator_class.total_width_pixels, env.scene.terrain.terrain_generator_class.total_length_pixels
        )

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        parkour_name: str,
        ) -> torch.Tensor:
        if self.contact_sensor is None:
            return torch.zeros(env.num_envs, device=self.device)
        feet_pos_x = ((self.asset.data.body_state_w[:, self.asset_cfg.body_ids ,0] + self.rows_offset)
                      /self.horizontal_scale).round().long() 
        feet_pos_y = ((self.asset.data.body_state_w[:, self.asset_cfg.body_ids ,1] + self.cols_offset)
                      /self.horizontal_scale).round().long() 
        feet_pos_x = torch.clip(feet_pos_x, 0, self.x_edge_masks_tensor.shape[0]-1)
        feet_pos_y = torch.clip(feet_pos_y, 0, self.x_edge_masks_tensor.shape[1]-1)
        feet_at_edge = self.x_edge_masks_tensor[feet_pos_x, feet_pos_y]
        contact_forces = self.contact_sensor.data.net_forces_w_history[:, 0, self.sensor_cfg.body_ids] #(N, 4, 3)
        previous_contact_forces = self.contact_sensor.data.net_forces_w_history[:, -1, self.sensor_cfg.body_ids] # N, 4, 3
        contact = torch.norm(contact_forces, dim=-1) > 2.
        last_contacts = torch.norm(previous_contact_forces, dim=-1) > 2.
        contact_filt = torch.logical_or(contact, last_contacts) 
        self.feet_at_edge = contact_filt & feet_at_edge
        rew = (self.parkour_event.terrain.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        ## This is for debugging to matching index and x_edge_mask
        # origin = self.x_edge_masks_tensor.detach().cpu().numpy().astype(np.uint8) * 255
        # cv2.imshow('origin',origin)
        # origin[feet_pos_x.detach().cpu().numpy(), feet_pos_y.detach().cpu().numpy()] -= 100
        # cv2.imshow('feet_edge',origin)
        # cv2.waitKey(1)
        return rew

def reward_torques(
    env: ParkourManagerBasedRLEnv,        
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque), dim=1)

def reward_dof_error(    
    env: ParkourManagerBasedRLEnv,        
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)

def reward_hip_pos(
    env: ParkourManagerBasedRLEnv,        
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] \
                                    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1)

def reward_ang_vel_xy(
    env: ParkourManagerBasedRLEnv,        
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:,:2]), dim=1)


def penalty_lin_vel_y_l2(
    env: ParkourManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_threshold: float = 0.05,
    min_forward_speed_cmd: float = 0.12,
) -> torch.Tensor:
    """Penalize lateral body velocity when only forward motion is commanded."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    vy = asset.data.root_lin_vel_b[:, 1]
    no_lateral_cmd = torch.abs(cmd[:, 1]) < cmd_threshold
    forward_cmd = torch.abs(cmd[:, 0]) > min_forward_speed_cmd
    return torch.square(vy) * (no_lateral_cmd & forward_cmd).float()


def penalty_backward_along_command(
    env: ParkourManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_forward_speed_cmd: float = 0.12,
) -> torch.Tensor:
    """Penalize backward body-frame velocity when a forward command is active."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    vx = asset.data.root_lin_vel_b[:, 0]
    forward_cmd = cmd[:, 0] > min_forward_speed_cmd
    backward = torch.relu(-vx)
    return torch.square(backward) * forward_cmd.float()


def penalty_body_heading_error_l2(
    env: ParkourManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_forward_speed_cmd: float = 0.12,
) -> torch.Tensor:
    """Penalize chassis heading error vs the parkour heading target while moving forward."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_term = env.command_manager.get_term(command_name)
    heading_target = cmd_term.heading_target
    heading_error = torch.atan2(
        torch.sin(heading_target - asset.data.heading_w),
        torch.cos(heading_target - asset.data.heading_w),
    )
    forward_cmd = cmd[:, 0] > min_forward_speed_cmd
    return torch.square(heading_error) * forward_cmd.float()


def reward_body_hip_excursion_when_forward(
    env: ParkourManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_forward_speed_cmd: float = 0.12,
) -> torch.Tensor:
    """Reward body-hip yaw motion from default while commanding forward (encourages splay use)."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None:
        return torch.zeros(env.num_envs, device=env.device)
    excursion = torch.mean(
        torch.abs(asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]),
        dim=1,
    )
    forward_cmd = cmd[:, 0] > min_forward_speed_cmd
    return excursion * forward_cmd.float()


def reward_joint_excursion_when_forward(
    env: ParkourManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_forward_speed_cmd: float = 0.12,
) -> torch.Tensor:
    """Reward joint motion from default while commanding forward (body-hip splay or hip–femur lift)."""
    return reward_body_hip_excursion_when_forward(
        env, command_name, asset_cfg, min_forward_speed_cmd=min_forward_speed_cmd
    )


def reward_body_hip_limit_usage_when_forward(
    env: ParkourManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hip_limit_rad: float = 1.309,
    min_forward_speed_cmd: float = 0.12,
) -> torch.Tensor:
    """Reward fraction of body-hip yaw limit used (|q| / limit, capped at 1) while moving forward."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None or hip_limit_rad <= 0.0:
        return torch.zeros(env.num_envs, device=env.device)
    usage = torch.mean(
        torch.clamp(torch.abs(asset.data.joint_pos[:, joint_ids]) / hip_limit_rad, max=1.0),
        dim=1,
    )
    forward_cmd = cmd[:, 0] > min_forward_speed_cmd
    return usage * forward_cmd.float()


class reward_action_rate(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        joint_pos_term = env.action_manager.get_term("joint_pos")
        action_dim = getattr(joint_pos_term, "_num_joints", None)
        if action_dim is None:
            asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
            action_dim = asset.num_joints
        self.previous_actions = torch.zeros(env.num_envs, 2, action_dim, dtype=torch.float, device=self.device)
        
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.previous_actions[env_ids, 0,:] = 0.
        self.previous_actions[env_ids, 1,:] = 0.

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
        self.previous_actions[:, 0, :] = self.previous_actions[:, 1, :]
        self.previous_actions[:, 1, :] = env.action_manager.get_term('joint_pos').raw_actions
        return torch.norm(self.previous_actions[:, 1, :] - self.previous_actions[:,0,:], dim=1)
    
class reward_dof_acc(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.previous_joint_vel = torch.zeros(env.num_envs, 2,  asset.num_joints, dtype= torch.float ,device=self.device)
        self.dt = env.cfg.decimation * env.cfg.sim.dt

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.previous_joint_vel[env_ids, 0,:] = 0.
        self.previous_joint_vel[env_ids, 1,:] = 0.

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        self.previous_joint_vel[:, 0, :] = self.previous_joint_vel[:, 1, :]
        self.previous_joint_vel[:, 1, :] = asset.data.joint_vel
        return torch.sum(torch.square((self.previous_joint_vel[:, 1, :] - self.previous_joint_vel[:,0,:]) / self.dt), dim=1)
        
def reward_lin_vel_z(
    env: ParkourManagerBasedRLEnv,        
    parkour_name:str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    parkour_event: ParkourEvent =  env.parkour_manager.get_term(parkour_name)
    terrain_names = parkour_event.env_per_terrain_name
    asset: Articulation = env.scene[asset_cfg.name]
    rew = torch.square(asset.data.root_lin_vel_b[:, 2])
    rew[(terrain_names !='parkour_flat')[:,-1]] *= 0.5
    return rew


def _parkour_flat_mask(env: ParkourManagerBasedRLEnv, parkour_name: str) -> torch.Tensor:
    parkour_event: ParkourEvent = env.parkour_manager.get_term(parkour_name)
    terrain_names = parkour_event.env_per_terrain_name
    on_flat = torch.as_tensor(terrain_names == "parkour_flat", device=env.device)
    return on_flat[:, -1].float()


def reward_orientation(
    env: ParkourManagerBasedRLEnv,   
    parkour_name:str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    parkour_event: ParkourEvent =  env.parkour_manager.get_term(parkour_name)
    terrain_names = parkour_event.env_per_terrain_name
    asset: Articulation = env.scene[asset_cfg.name]
    rew = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    rew[(terrain_names !='parkour_flat')[:,-1]] = 0.
    return rew


def reward_orientation_upright(
    env: ParkourManagerBasedRLEnv,
    parkour_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize roll/pitch tilt on all terrains (teacher bridge mode)."""
    del parkour_name
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def penalty_base_pitch_forward_l2(
    env: ParkourManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_forward_speed_cmd: float = 0.12,
    max_pitch_rad: float = 0.17,
) -> torch.Tensor:
    """Penalize nose-down pitch above ``max_pitch_rad`` while commanding forward (bridge uses ~0.08 rad)."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    _, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    forward_cmd = cmd[:, 0] > min_forward_speed_cmd
    excess = torch.relu(pitch - max_pitch_rad)
    return torch.square(excess) * forward_cmd.float()


def penalty_base_pitch_forward_linear(
    env: ParkourManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_forward_speed_cmd: float = 0.12,
) -> torch.Tensor:
    """Linear penalty on all nose-down pitch while commanding forward (no dead band; bridge mode)."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    _, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    forward_cmd = cmd[:, 0] > min_forward_speed_cmd
    return torch.relu(pitch) * forward_cmd.float()


def reward_feet_air_time_on_flat(
    env: ParkourManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    parkour_name: str,
    threshold: float,
) -> torch.Tensor:
    """Swing bonus on ``parkour_flat`` only (encourages long strides on flat tiles)."""
    rew = reward_feet_air_time_positive(env, command_name, sensor_cfg, threshold)
    return rew * _parkour_flat_mask(env, parkour_name)


def reward_forward_speed_on_flat(
    env: ParkourManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    parkour_name: str,
    min_forward_speed_cmd: float = 0.12,
    target_speed: float = 0.55,
    max_bonus_speed: float = 0.85,
) -> torch.Tensor:
    """Bonus for forward body speed above ``target_speed`` on flat terrain only."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    forward_cmd = cmd[:, 0] > min_forward_speed_cmd
    vx = asset.data.root_lin_vel_b[:, 0]
    bonus = torch.clamp(vx - target_speed, min=0.0, max=max_bonus_speed - target_speed)
    return bonus * _parkour_flat_mask(env, parkour_name) * forward_cmd.float()


def penalty_low_forward_speed_when_commanded(
    env: ParkourManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_forward_speed_cmd: float = 0.12,
    min_actual_speed: float = 0.35,
) -> torch.Tensor:
    """Linear penalty for forward speed below ``min_actual_speed`` while commanding forward (anti-stall)."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    forward_cmd = cmd[:, 0] > min_forward_speed_cmd
    vx = asset.data.root_lin_vel_b[:, 0]
    shortfall = torch.relu(min_actual_speed - vx)
    return shortfall * forward_cmd.float()


def reward_feet_stumble(
    env: ParkourManagerBasedRLEnv,        
    sensor_cfg: SceneEntityCfg ,
    ) -> torch.Tensor:
    contact_sensor = env.scene.sensors.get(sensor_cfg.name)
    if contact_sensor is None:
        return torch.zeros(env.num_envs, device=env.device)
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
    rew = torch.any(
        torch.norm(net_contact_forces[:, :, :2], dim=2) > 4 * torch.abs(net_contact_forces[:, :, 2]),
        dim=1,
    )
    return rew.float()


def reward_obstacle_clearance(
    env: ParkourManagerBasedRLEnv,
    parkour_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    min_goal_progress: float = 0.15,
    min_forward_speed: float = 0.25,
    min_forward_speed_cmd: float = 0.12,
    max_tilt_gravity_xy_sq: float = 0.02,
) -> torch.Tensor:
    """Bonus on parkour tiles for lift-and-cross: goal progress, forward speed, upright, no stumble."""
    on_parkour = 1.0 - _parkour_flat_mask(env, parkour_name)
    stumble = reward_feet_stumble(env, sensor_cfg)
    asset: Articulation = env.scene[asset_cfg.name]
    goal_prog = torch.clamp(
        reward_tracking_goal_vel(env, parkour_name, asset_cfg),
        min=0.0,
    )
    cmd = env.command_manager.get_command(command_name)
    forward_cmd = cmd[:, 0] > min_forward_speed_cmd
    vx = asset.data.root_lin_vel_b[:, 0]
    tilt_sq = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    progressing = (goal_prog >= min_goal_progress).float()
    moving = ((vx >= min_forward_speed) & forward_cmd).float()
    landed = (tilt_sq <= max_tilt_gravity_xy_sq).float()
    no_stumble = (1.0 - stumble)
    return on_parkour * no_stumble * progressing * moving * landed


def reward_foot_clearance(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    contact_force_threshold: float = 0.1,
    min_clearance_m: float = 0.05,
    max_clearance_m: float = 0.20,
    min_forward_speed_cmd: float = 0.12,
    ground_offset_from_root_m: float = -1.0,
    parkour_name: str | None = "base_parkour",
) -> torch.Tensor:
    """Per-foot swing lift: reward vertical clearance above nominal ground while foot is in swing.

    Complements ``reward_obstacle_clearance`` (global lift-and-cross gate) with dense per-step
    encouragement to lift feet early during swing, before they catch on small steps or hole lips.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors.get(sensor_cfg.name)
    if contact_sensor is None:
        return torch.zeros(env.num_envs, device=env.device)

    foot_ids = sensor_cfg.body_ids
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, foot_ids]
    in_contact = torch.norm(net_contact_forces, dim=-1) > contact_force_threshold
    in_swing = ~in_contact

    foot_z = asset.data.body_pos_w[:, foot_ids, 2]
    ground_z = asset.data.root_pos_w[:, 2].unsqueeze(1) + ground_offset_from_root_m
    height_above_ground = foot_z - ground_z
    clearance = torch.relu(height_above_ground - min_clearance_m)
    clearance = torch.clamp(clearance, max=max_clearance_m)

    per_foot = in_swing.float() * clearance
    rew = torch.mean(per_foot, dim=1)

    cmd = env.command_manager.get_command(command_name)
    forward_cmd = cmd[:, 0] > min_forward_speed_cmd
    rew = rew * forward_cmd.float()

    if parkour_name is not None:
        rew = rew * (1.0 - _parkour_flat_mask(env, parkour_name))
    return rew


def _foot_swing_height_above_ground(
    env: ParkourManagerBasedRLEnv,
    asset: Articulation,
    contact_sensor,
    foot_body_ids,
    contact_force_threshold: float,
    ground_offset_from_root_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-foot in_contact, in_swing, height above nominal ground (N_env, N_feet)."""
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, foot_body_ids]
    in_contact = torch.norm(net_contact_forces, dim=-1) > contact_force_threshold
    in_swing = ~in_contact
    foot_z = asset.data.body_pos_w[:, foot_body_ids, 2]
    ground_z = asset.data.root_pos_w[:, 2].unsqueeze(1) + ground_offset_from_root_m
    height_above_ground = foot_z - ground_z
    return in_contact, in_swing, height_above_ground


def penalty_swing_min_clearance(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    contact_force_threshold: float = 0.1,
    min_clearance_m: float = 0.03,
    min_forward_speed_cmd: float = 0.12,
    ground_offset_from_root_m: float = -1.0,
    parkour_name: str | None = "base_parkour",
) -> torch.Tensor:
    """Penalty for micro-swings: foot in swing but vertical clearance below ``min_clearance_m``."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors.get(sensor_cfg.name)
    if contact_sensor is None:
        return torch.zeros(env.num_envs, device=env.device)

    foot_ids = sensor_cfg.body_ids
    _, in_swing, height_above_ground = _foot_swing_height_above_ground(
        env, asset, contact_sensor, foot_ids, contact_force_threshold, ground_offset_from_root_m
    )
    micro_swing = in_swing.float() * (height_above_ground < min_clearance_m).float()
    penalty = torch.mean(micro_swing, dim=1)

    cmd = env.command_manager.get_command(command_name)
    forward_cmd = cmd[:, 0] > min_forward_speed_cmd
    penalty = penalty * forward_cmd.float()
    if parkour_name is not None:
        penalty = penalty * (1.0 - _parkour_flat_mask(env, parkour_name))
    return penalty


class RewardSwingVerticalVel(ManagerTermBase):
    """Bonus when a foot enters swing with upward vertical velocity (lift early, not when stuck)."""

    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        self.contact_sensor = env.scene.sensors.get(sensor_cfg.name)
        self.sensor_cfg = sensor_cfg
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.command_name = cfg.params["command_name"]
        self.parkour_name = cfg.params["parkour_name"]
        self.contact_force_threshold = float(cfg.params.get("contact_force_threshold", 0.1))
        self.min_forward_speed_cmd = float(cfg.params.get("min_forward_speed_cmd", 0.12))
        self.max_vertical_vel = float(cfg.params.get("max_vertical_vel", 0.5))
        self.ground_offset_from_root_m = float(cfg.params.get("ground_offset_from_root_m", -1.0))
        n_feet = len(sensor_cfg.body_ids)
        self._prev_in_contact = torch.zeros(env.num_envs, n_feet, device=self.device, dtype=torch.bool)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._prev_in_contact[env_ids] = False

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        parkour_name: str,
        command_name: str = "base_velocity",
        contact_force_threshold: float = 0.1,
        min_forward_speed_cmd: float = 0.12,
        max_vertical_vel: float = 0.5,
        ground_offset_from_root_m: float = -1.0,
    ) -> torch.Tensor:
        if self.contact_sensor is None:
            return torch.zeros(env.num_envs, device=env.device)

        foot_ids = self.sensor_cfg.body_ids
        in_contact, in_swing, _ = _foot_swing_height_above_ground(
            env,
            self.asset,
            self.contact_sensor,
            foot_ids,
            contact_force_threshold,
            ground_offset_from_root_m,
        )
        swing_start = in_swing & self._prev_in_contact
        foot_vz = self.asset.data.body_lin_vel_w[:, foot_ids, 2]
        vz_norm = torch.clamp(foot_vz / self.max_vertical_vel, min=0.0, max=1.0)
        per_foot = swing_start.float() * vz_norm
        n_start = torch.sum(swing_start.float(), dim=1).clamp(min=1.0)
        rew = torch.sum(per_foot, dim=1) / n_start

        cmd = env.command_manager.get_command(command_name)
        forward_cmd = cmd[:, 0] > min_forward_speed_cmd
        on_parkour = 1.0 - _parkour_flat_mask(env, parkour_name)
        rew = rew * forward_cmd.float() * on_parkour

        self._prev_in_contact = in_contact.clone()
        return rew


class RewardRecoverFromStall(ManagerTermBase):
    """Dense bonus for lifting a loaded foot while near-stall on parkour (re-step from hole/lip)."""

    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        self.contact_sensor = env.scene.sensors.get(sensor_cfg.name)
        self.sensor_cfg = sensor_cfg
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.command_name = cfg.params["command_name"]
        self.parkour_name = cfg.params["parkour_name"]
        self.min_forward_speed_cmd = float(cfg.params.get("min_forward_speed_cmd", 0.12))
        self.min_actual_speed = float(cfg.params.get("min_actual_speed", 0.15))
        self.stuck_contact_force = float(cfg.params.get("stuck_contact_force", 15.0))
        self.min_other_feet_loaded = int(cfg.params.get("min_other_feet_loaded", 2))
        self.max_tilt_gravity_xy_sq = float(cfg.params.get("max_tilt_gravity_xy_sq", 0.04))
        n_feet = len(sensor_cfg.body_ids)
        self._prev_in_contact = torch.zeros(env.num_envs, n_feet, device=self.device, dtype=torch.bool)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._prev_in_contact[env_ids] = False

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        parkour_name: str,
        command_name: str = "base_velocity",
        min_forward_speed_cmd: float = 0.12,
        min_actual_speed: float = 0.15,
        stuck_contact_force: float = 15.0,
        min_other_feet_loaded: int = 2,
        max_tilt_gravity_xy_sq: float = 0.04,
    ) -> torch.Tensor:
        if self.contact_sensor is None:
            return torch.zeros(env.num_envs, device=env.device)

        foot_ids = self.sensor_cfg.body_ids
        net_contact_forces = self.contact_sensor.data.net_forces_w_history[:, 0, foot_ids]
        force_mag = torch.norm(net_contact_forces, dim=-1)
        in_contact = force_mag > 0.1

        cmd = env.command_manager.get_command(command_name)
        forward_cmd = cmd[:, 0] > min_forward_speed_cmd
        vx = self.asset.data.root_lin_vel_b[:, 0]
        near_stall = (vx < min_actual_speed) & forward_cmd

        stuck_foot = torch.any(force_mag > stuck_contact_force, dim=1)
        tilt_sq = torch.sum(torch.square(self.asset.data.projected_gravity_b[:, :2]), dim=1)
        upright = tilt_sq <= max_tilt_gravity_xy_sq

        lifted_foot = (~in_contact) & self._prev_in_contact
        other_loaded = torch.sum(in_contact.float(), dim=1) >= float(min_other_feet_loaded)
        re_step = torch.any(lifted_foot, dim=1) & other_loaded

        on_parkour = 1.0 - _parkour_flat_mask(env, parkour_name)
        stall_context = on_parkour * near_stall.float() * stuck_foot.float() * upright.float()
        rew = re_step.float() * stall_context

        self._prev_in_contact = in_contact.clone()
        return rew


def reward_tracking_goal_vel(
    env: ParkourManagerBasedRLEnv, 
    parkour_name : str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    parkour_event: ParkourEvent = env.parkour_manager.get_term(parkour_name)
    target_pos_rel = parkour_event.target_pos_rel
    target_vel = target_pos_rel / (torch.norm(target_pos_rel, dim=-1, keepdim=True) + 1e-5)
    cur_vel = asset.data.root_vel_w[:, :2]
    proj_vel = torch.sum(target_vel * cur_vel, dim=-1)
    command_vel = env.command_manager.get_command("base_velocity")[:, 0]
    # Avoid division blow-ups when |command_vel| is tiny (stabilizes logging / value learning).
    v_min = 0.05
    sign = torch.sign(command_vel)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    denom = torch.where(torch.abs(command_vel) < v_min, sign * v_min, command_vel)
    rew_move = torch.minimum(proj_vel, command_vel) / denom
    rew_move = torch.nan_to_num(rew_move, nan=0.0, posinf=0.0, neginf=0.0)
    rew_move = torch.clamp(rew_move, -10.0, 10.0)
    return rew_move


def reward_tracking_goal_vel_on_parkour(
    env: ParkourManagerBasedRLEnv,
    parkour_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Parkour waypoint velocity reward; zero on ``parkour_flat`` (use command velocity on flat)."""
    rew = reward_tracking_goal_vel(env, parkour_name, asset_cfg)
    return rew * (1.0 - _parkour_flat_mask(env, parkour_name))


def reward_tracking_yaw(     
    env: ParkourManagerBasedRLEnv, 
    parkour_name : str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    parkour_event: ParkourEvent =  env.parkour_manager.get_term(parkour_name)
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.root_quat_w
    yaw = torch.atan2(2*(q[:,0]*q[:,3] + q[:,1]*q[:,2]),
                    1 - 2*(q[:,2]**2 + q[:,3]**2))
    return torch.exp(-torch.abs((parkour_event.target_yaw - yaw)))


def reward_tracking_yaw_on_parkour(
    env: ParkourManagerBasedRLEnv,
    parkour_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    min_forward_speed_cmd: float = 0.12,
) -> torch.Tensor:
    """Yaw-alignment bonus on parkour tiles while commanding forward (reduces sideways edge approach)."""
    on_parkour = 1.0 - _parkour_flat_mask(env, parkour_name)
    cmd = env.command_manager.get_command(command_name)
    forward_cmd = cmd[:, 0] > min_forward_speed_cmd
    yaw_align = reward_tracking_yaw(env, parkour_name, asset_cfg)
    return on_parkour * yaw_align * forward_cmd.float()

class reward_delta_torques(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.previous_torque = torch.zeros(env.num_envs, 2,  self.asset.num_joints, dtype= torch.float ,device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.previous_torque[env_ids, 0,:] = 0.
        self.previous_torque[env_ids, 1,:] = 0.

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
        self.previous_torque[:, 0, :] = self.previous_torque[:, 1, :]
        self.previous_torque[:, 1, :] = self.asset.data.applied_torque
        return torch.sum(torch.square((self.previous_torque[:, 1, :] - self.previous_torque[:,0,:])), dim=1)

def reward_collision(
    env: ParkourManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg ,
) -> torch.Tensor:
    contact_sensor = env.scene.sensors.get(sensor_cfg.name)
    if contact_sensor is None:
        return torch.zeros(env.num_envs, device=env.device)
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
    return torch.sum(1.0 * (torch.norm(net_contact_forces, dim=-1) > 0.1), dim=1)


def reward_feet_air_time_positive(
    env: ParkourManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Reward swing duration above ``threshold`` at touchdown (no penalty for short swings).

    Unlike Isaac Lab ``feet_air_time``, uses ``relu(last_air_time - threshold)`` so brief contacts
    do not get a negative contribution at first contact.
    """
    contact_sensor = env.scene.sensors.get(sensor_cfg.name)
    if contact_sensor is None:
        return torch.zeros(env.num_envs, device=env.device)
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    excess = torch.relu(last_air_time - threshold)
    reward = torch.sum(excess * first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def penalty_excess_feet_in_contact_forward(
    env: ParkourManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    max_feet_on_ground: int,
    contact_force_threshold: float = 0.1,
    min_forward_speed_cmd: float = 0.12,
) -> torch.Tensor:
    """Penalize having too many feet on the ground while commanding forward motion (hexapod gait nudge).

    Counts feet with net contact force magnitude above ``contact_force_threshold``. When
    ``|base_velocity command x|`` exceeds ``min_forward_speed_cmd``, returns
    ``relu(count - max_feet_on_ground)`` per env (0 if not commanding forward).
    """
    contact_sensor = env.scene.sensors.get(sensor_cfg.name)
    if contact_sensor is None:
        return torch.zeros(env.num_envs, device=env.device)
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
    in_contact = torch.norm(net_contact_forces, dim=-1) > contact_force_threshold
    num_feet = torch.sum(in_contact.float(), dim=1)
    excess = torch.relu(num_feet - float(max_feet_on_ground))
    cmd = env.command_manager.get_command(command_name)
    moving = torch.abs(cmd[:, 0]) > min_forward_speed_cmd
    return excess * moving.float()


def reward_forward_progress_along_command(
    env: ParkourManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    min_cmd_norm: float = 0.12,
    max_speed_scale: float = 2.0,
) -> torch.Tensor:
    """Dense nonnegative progress: base linear velocity (body frame) along commanded planar direction.

    Matches body-frame velocity commands. Only applies when planar command norm exceeds ``min_cmd_norm``.
    Clips at ``max_speed_scale`` [m/s] along the projection.
    """
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_xy = cmd[:, :2]
    norm = torch.norm(cmd_xy, dim=1)
    active = norm > min_cmd_norm
    dir_xy = cmd_xy / (norm.unsqueeze(-1) + 1e-8)
    vel_b_xy = asset.data.root_lin_vel_b[:, :2]
    progress = torch.sum(vel_b_xy * dir_xy, dim=1)
    progress = torch.clamp(progress, min=0.0, max=max_speed_scale)
    return progress * active.float()


def reward_stance_support_feet_when_forward(
    env: ParkourManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    min_feet_loaded: int = 3,
    contact_force_threshold: float = 0.1,
    min_forward_speed_cmd: float = 0.12,
) -> torch.Tensor:
    """Binary bonus when at least ``min_feet_loaded`` tibias show contact while commanding forward.

    Complements ``penalty_excess_feet_in_contact_forward``: rewards a load-bearing stance for pushing.
    """
    contact_sensor = env.scene.sensors.get(sensor_cfg.name)
    if contact_sensor is None:
        return torch.zeros(env.num_envs, device=env.device)
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
    in_contact = torch.norm(net_contact_forces, dim=-1) > contact_force_threshold
    num_feet = torch.sum(in_contact.float(), dim=1)
    cmd = env.command_manager.get_command(command_name)
    moving = torch.abs(cmd[:, 0]) > min_forward_speed_cmd
    has_support = num_feet.float() >= float(min_feet_loaded)
    return has_support.float() * moving.float()


class PenaltyFootIdleWhenForward(ManagerTermBase):
    """Penalize feet that stay airborne too long while commanding forward (hex duty cycle)."""

    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        self.contact_sensor = env.scene.sensors.get(sensor_cfg.name)
        self.sensor_cfg = sensor_cfg
        self.max_idle_steps = int(cfg.params["max_idle_steps"])
        self.contact_force_threshold = float(cfg.params.get("contact_force_threshold", 0.1))
        self.command_name = cfg.params["command_name"]
        self.min_forward_speed_cmd = float(cfg.params.get("min_forward_speed_cmd", 0.12))
        self.idle_steps = torch.zeros(env.num_envs, len(sensor_cfg.body_ids), device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.idle_steps[env_ids] = 0

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        command_name: str,
        sensor_cfg: SceneEntityCfg,
        max_idle_steps: int,
        contact_force_threshold: float = 0.1,
        min_forward_speed_cmd: float = 0.12,
    ) -> torch.Tensor:
        if self.contact_sensor is None:
            return torch.zeros(env.num_envs, device=self.device)
        net_forces = self.contact_sensor.data.net_forces_w_history[:, 0, self.sensor_cfg.body_ids]
        in_contact = torch.norm(net_forces, dim=-1) > contact_force_threshold
        self.idle_steps = torch.where(in_contact, torch.zeros_like(self.idle_steps), self.idle_steps + 1.0)
        excess = torch.relu(self.idle_steps - float(max_idle_steps))
        cmd = env.command_manager.get_command(command_name)
        moving = torch.abs(cmd[:, 0]) > min_forward_speed_cmd
        return torch.sum(excess, dim=1) * moving.float()


def penalty_joint_deviation_when_in_contact(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    contact_force_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize deviation from default joint pose for loaded legs only."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors.get(sensor_cfg.name)
    if contact_sensor is None:
        return torch.zeros(env.num_envs, device=env.device)

    joint_err = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_sq = torch.square(joint_err)

    net_forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
    in_contact = torch.norm(net_forces, dim=-1) > contact_force_threshold

    num_joints = joint_sq.shape[1]
    num_feet = in_contact.shape[1]
    if num_joints != num_feet:
        load_frac = in_contact.float().sum(dim=1) / float(num_feet)
        return torch.sum(joint_sq, dim=1) * load_frac

    return torch.sum(joint_sq * in_contact.float(), dim=1)
