from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation
from isaaclab.utils.math  import euler_xyz_from_quat, wrap_to_pi, quat_apply
from parkour_isaaclab.envs.mdp.parkours import ParkourEvent
from parkour_tasks.crab_hex_forward_task.mdp.crab_hex_stride_reward import stride_length_reward_step
from parkour_tasks.crab_hex_forward_task.mdp.crab_hex_clock_reward import (
    APEX_SIGMA_M as CLOCK_APEX_SIGMA_M,
    APEX_TARGET_M as CLOCK_APEX_TARGET_M,
    FOOT_ORDER as CLOCK_FOOT_ORDER,
    FORCE_REF_N as CLOCK_FORCE_REF_N,
    VEL_REF_M_S as CLOCK_VEL_REF_M_S,
    clock_schedule_income,
    clock_swing_apex_income,
)
from parkour_tasks.crab_hex_forward_task.mdp.crab_hex_tripod_reward import (
    RESET_T_SINCE,
    S_T_SINCE,
    STATE_DIM,
    TRIPOD_A_IDX,
    TRIPOD_B_IDX,
    tripod_swap_crossing_reward_step,
)
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
    # asset_cfg.joint_ids is slice(None) when no joint_names filter is given (Go2 path),
    # preserving the historical all-joints behavior.
    return torch.sum(
        torch.square(
            asset.data.joint_pos[:, asset_cfg.joint_ids]
            - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )

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


class PenaltyMotorDirectionReversal(ManagerTermBase):
    """Penalize the cam-shaft motor changing rotational direction; encourages sustained
    one-directional spin so the cam geometry -- not motor reversal -- produces the leg's
    back-and-forth yaw motion."""

    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.joint_ids = asset_cfg.joint_ids
        self.prev_dir = torch.zeros(env.num_envs, len(self.joint_ids), device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.prev_dir[env_ids] = 0.0

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        vel_deadzone: float = 0.05,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        vel = asset.data.joint_vel[:, self.joint_ids]
        cur_dir = torch.sign(vel) * (vel.abs() > vel_deadzone).float()
        reversed_mask = (self.prev_dir * cur_dir) < 0
        penalty = reversed_mask.float().sum(dim=1)
        nonzero = cur_dir != 0
        self.prev_dir = torch.where(nonzero, cur_dir, self.prev_dir)
        return penalty


class RewardOneDirectionSpin(ManagerTermBase):
    """Reward sustained one-directional cam-shaft rotation (velocity era, 2026-08-14).

    Five-point penalty dose-response (onedir-spin campaign) showed taxing reversals is
    either absorbed (<= -0.6) or collapses locomotion (-1.0); this term shapes TOWARD the
    spin basin instead. Per shaft it pays the signed-consistency of an EMA'd velocity:
    |ema(v)| / ema(|v|) in [0, 1] — a symmetric oscillation earns ~0, a continuous spin
    earns ~1 — scaled by min(ema(|v|)/speed_ref, 1) so slow/parked shafts cannot farm it,
    and gated on an active velocity command so standing still earns nothing.
    """

    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.joint_ids = asset_cfg.joint_ids
        n = len(self.joint_ids)
        self.ema_signed = torch.zeros(env.num_envs, n, device=self.device)
        self.ema_abs = torch.zeros(env.num_envs, n, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.ema_signed[env_ids] = 0.0
        self.ema_abs[env_ids] = 0.0

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        command_name: str = "base_velocity",
        ema_tau: float = 2.0,
        speed_ref: float = 4.0,
        min_cmd_norm: float = 0.12,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        vel = asset.data.joint_vel[:, self.joint_ids]
        alpha = 1.0 - torch.exp(torch.tensor(-env.step_dt / ema_tau, device=self.device))
        self.ema_signed = self.ema_signed + alpha * (vel - self.ema_signed)
        self.ema_abs = self.ema_abs + alpha * (vel.abs() - self.ema_abs)
        consistency = self.ema_signed.abs() / self.ema_abs.clamp_min(1e-6)
        speed_scale = (self.ema_abs / speed_ref).clamp(max=1.0)
        cmd = env.command_manager.get_command(command_name)
        cmd_active = (torch.norm(cmd[:, :2], dim=1) > min_cmd_norm).float()
        return (consistency * speed_scale).mean(dim=1) * cmd_active


def penalty_tracking_error_l1(
    env: ParkourManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Linear planar velocity-tracking error |cmd_xy - v_xy| (task1-velocity C1).

    The exponential tracking term's gradient dies outside ~+-0.25 m/s (narrow sigma) or
    pays income without tracking (wide sigma). This L1 penalty supplies constant
    gradient pressure at every error magnitude and cannot be satisfied at a fixed
    deficit; as a pure penalty its optimum is exact tracking."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    return torch.norm(cmd[:, :2] - asset.data.root_lin_vel_b[:, :2], dim=1)


def penalty_mechanical_power(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Total mechanical power sum |tau * qdot| (lit review s5, replayed 2026-08-15):
    continuous one-direction spin costs ~738 W vs ~1260-1420 W for shaft-oscillation
    gaits — the saving is in the LEG chain (fewer reversal transients), so this prices
    the oscillation basin ~1.7x harder than spin on physics grounds. Weight scale:
    ~1e-3 puts the differential at a few percent of locomotion income."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=1
    )


class PenaltyCamContactSchedule(ManagerTermBase):
    """Contact-schedule penalty referenced to each leg's OWN cam-shaft phase (round 4,
    lit-review synthesis: Siekmann-style swing/stance windows, but the clock is the
    hardware phase variable the quick-return linkage provides).

    Return stroke (fast hip sweep, |d theta_hip/d s| > g_thresh): contact is penalized —
    the foot should be in swing while the cam snaps the leg back. Power stroke: planar
    foot speed while in contact is penalized — a planted foot must not slide. The target
    behavior (contact only during power stroke, no slide) pays exactly zero; as a pure
    penalty there is no holdable positive-income state to farm.
    """

    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # resolve leg order by name so shaft, footpad-body, and sensor indices agree
        shaft_ids, shaft_names = asset.find_joints([".*_Body_CamShaft_RevoluteJoint"], preserve_order=True)
        legs = [n.split("_")[0] for n in shaft_names]
        self._shaft_ids = shaft_ids
        body_ids, body_names = asset.find_bodies([f"{leg}_Footpad" for leg in legs], preserve_order=True)
        self._foot_body_ids = body_ids
        sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        name_to_sensor = {n: i for i, n in enumerate(sensor.body_names)}
        self._sensor_ids = [name_to_sensor[f"{leg}_Footpad"] for leg in legs]

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        g_thresh: float = 0.55,
        contact_force_threshold: float = 1.0,
    ) -> torch.Tensor:
        from parkour_tasks.crab_hex_forward_task.mdp.crab_hex_cam_mapping import cam_shaft_to_hip

        asset: Articulation = env.scene[asset_cfg.name]
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        theta = asset.data.joint_pos[:, self._shaft_ids]
        _, g = cam_shaft_to_hip(theta, torch.ones_like(theta))
        in_return = g.abs() > g_thresh
        forces = sensor.data.net_forces_w_history[:, :, self._sensor_ids, :].norm(dim=-1).max(dim=1)[0]
        contact = forces > contact_force_threshold
        foot_speed = asset.data.body_lin_vel_w[:, self._foot_body_ids, :2].norm(dim=-1)
        force_pen = (contact & in_return).float().sum(dim=1)
        slide_pen = (foot_speed * (contact & ~in_return).float()).sum(dim=1)
        return force_pen + slide_pen


class RewardClockContactSchedule(ManagerTermBase):
    """Clock-referenced contact-schedule income (gait-formation-v2 Phase 1) -- see
    ``crab_hex_clock_reward`` for the pure math and design rationale. Reads the gait clock
    from ``CrabHexDelayedJointPositionAction.clock_phase``, contact forces from the
    privileged sensor, and foot world velocities from the articulation. Pays only upright:
    a fallen robot has every foot unloaded, which would otherwise be free swing income
    (same gate rationale as RewardCamPhaseLock's round-4 post-mortem)."""

    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        foot_names = [f"{leg}_Footpad" for leg in CLOCK_FOOT_ORDER]
        self._foot_body_ids, _ = asset.find_bodies(foot_names, preserve_order=True)
        sensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self._foot_sensor_ids, _ = sensor.find_bodies(foot_names, preserve_order=True)
        from parkour_tasks.crab_hex_forward_task.mdp import crab_hex_dimensions as _dims

        self._cmd_stop = _dims.CLOCK_CMD_STOP_M_S

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        command_name: str = "base_velocity",
        force_ref: float = CLOCK_FORCE_REF_N,
        vel_ref: float = CLOCK_VEL_REF_M_S,
        min_upright_gz: float = 0.9,
        combine: str = "sum",
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        sensor = env.scene.sensors[sensor_cfg.name]
        phase = env.action_manager.get_term("joint_pos").clock_phase
        force = sensor.data.net_forces_w[:, self._foot_sensor_ids].norm(dim=-1)
        speed_xy = asset.data.body_lin_vel_w[:, self._foot_body_ids, :2].norm(dim=-1)
        cmd = env.command_manager.get_command(command_name)
        clock_running = cmd[:, 0].abs() > self._cmd_stop
        income = clock_schedule_income(
            phase, force, speed_xy, clock_running, force_ref=force_ref, vel_ref=vel_ref,
            combine=combine,
        )
        upright = (-asset.data.projected_gravity_b[:, 2] > min_upright_gz).float()
        return income * upright


class RewardClockSwingApex(ManagerTermBase):
    """Scheduled swing-apex income (gait-formation-v2 Phase 4) -- see
    ``crab_hex_clock_reward.clock_swing_apex_income``. Foot height measured above the
    nominal ground plane (root z + ground offset, as reward_foot_clearance does).
    Upright-gated for the same fallen-farming reason as the schedule term."""

    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        foot_names = [f"{leg}_Footpad" for leg in CLOCK_FOOT_ORDER]
        self._foot_body_ids, _ = asset.find_bodies(foot_names, preserve_order=True)
        from parkour_tasks.crab_hex_forward_task.mdp import crab_hex_dimensions as _dims

        self._cmd_stop = _dims.CLOCK_CMD_STOP_M_S

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        command_name: str = "base_velocity",
        apex_m: float = CLOCK_APEX_TARGET_M,
        sigma_m: float = CLOCK_APEX_SIGMA_M,
        ground_offset_from_root_m: float = -1.05,
        min_upright_gz: float = 0.9,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        phase = env.action_manager.get_term("joint_pos").clock_phase
        foot_z = asset.data.body_pos_w[:, self._foot_body_ids, 2]
        ground_z = asset.data.root_pos_w[:, 2].unsqueeze(1) + ground_offset_from_root_m
        height = foot_z - ground_z
        cmd = env.command_manager.get_command(command_name)
        clock_running = cmd[:, 0].abs() > self._cmd_stop
        income = clock_swing_apex_income(
            phase, height, clock_running, apex_m=apex_m, sigma_m=sigma_m
        )
        upright = (-asset.data.projected_gravity_b[:, 2] > min_upright_gz).float()
        return income * upright


class RewardCamPhaseLock(ManagerTermBase):
    """Reward in-tripod-set cam-phase coherence, gated by one-direction spin (round 4).

    Sets A = {FL, MR, RL}, B = {FR, ML, RR}. Per set: |mean_j exp(i * dir_j * theta_j)|
    in [0, 1] (1 = shafts phase-locked), multiplied by the set's mean spin gate
    (EMA |mean v|/mean |v| x speed scale — the replay-validated one-direction measure),
    so a non-spinning policy cannot farm the coherence of parked shafts (offline replica:
    oscillator 3/min vs spin gait 26/min vs ideal 59/min).
    """

    _SET_A = ("FL", "MR", "RL")
    _SET_B = ("FR", "ML", "RR")

    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        shaft_ids, shaft_names = asset.find_joints([".*_Body_CamShaft_RevoluteJoint"], preserve_order=True)
        legs = [n.split("_")[0] for n in shaft_names]
        self._shaft_ids = shaft_ids
        self._a_cols = [legs.index(leg) for leg in self._SET_A]
        self._b_cols = [legs.index(leg) for leg in self._SET_B]
        n = len(shaft_ids)
        self.ema_signed = torch.zeros(env.num_envs, n, device=self.device)
        self.ema_abs = torch.zeros(env.num_envs, n, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.ema_signed[env_ids] = 0.0
        self.ema_abs[env_ids] = 0.0

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        command_name: str = "base_velocity",
        ema_tau: float = 2.0,
        speed_ref: float = 4.0,
        min_cmd_norm: float = 0.12,
        min_upright_gz: float = 0.9,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        vel = asset.data.joint_vel[:, self._shaft_ids]
        alpha = 1.0 - torch.exp(torch.tensor(-env.step_dt / ema_tau, device=self.device))
        self.ema_signed = self.ema_signed + alpha * (vel - self.ema_signed)
        self.ema_abs = self.ema_abs + alpha * (vel.abs() - self.ema_abs)
        gate = (self.ema_signed.abs() / self.ema_abs.clamp_min(1e-6)) * (
            self.ema_abs / speed_ref
        ).clamp(max=1.0)
        # direction-normalized phase so opposite-spinning sets compare consistently
        theta = asset.data.joint_pos[:, self._shaft_ids] * torch.sign(
            self.ema_signed + 1e-9
        )
        z = torch.exp(1j * theta.to(torch.complex64))
        coh_a = z[:, self._a_cols].mean(dim=1).abs() * gate[:, self._a_cols].mean(dim=1)
        coh_b = z[:, self._b_cols].mean(dim=1).abs() * gate[:, self._b_cols].mean(dim=1)
        # NOTE(round-4 screen post-mortem): without these gates the optimal policy is to
        # FALL OVER and spin — a fallen robot phase-locks trivially (feet off the ground,
        # no contact-schedule pressure, no ground disturbances). 100% crab_failure at ~80
        # steps for 3000 iters. Pay only upright, commanded locomotion.
        cmd = env.command_manager.get_command(command_name)
        cmd_active = (torch.norm(cmd[:, :2], dim=1) > min_cmd_norm).float()
        upright = (-asset.data.projected_gravity_b[:, 2] > min_upright_gz).float()
        return 0.5 * (coh_a + coh_b) * cmd_active * upright


class RewardStrideLength(ManagerTermBase):
    """Reward each leg's stance-phase contribution to real body progress along the commanded
    direction -- only a planted foot can push the robot forward, so only stance counts, and only
    the component of body motion actually moving in the desired direction (motion the wrong way
    earns nothing). Convex in accumulated stance progress so one long productive stance outscores
    several short ones covering the same net range, same anti-tippy-tap rationale as before. See
    ``crab_hex_stride_reward.stride_length_reward_step`` for the pure math and the rationale for
    dropping the earlier swing-phase reward (real training data showed a leg "snapping" through
    its whole joint range in a single physics step to bank reward without moving the robot at
    all -- measuring body progress rather than joint-space movement eliminates that structurally,
    no separate velocity cost needed).

    This is v3 of the term (v4, per-foot touchdown-to-touchdown displacement, was tried and
    reverted back to this design -- see ``parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-09_0106_stride_length_v4/CHANGELOG.md``: v4
    finally beat the target stride-length metric at 2b2 but at the cost of the worst tippy-tap in
    the whole comparison series and broad regressions vs this design on training stability)."""

    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        self.body_ids = sensor_cfg.body_ids
        self.stance_progress = torch.zeros(env.num_envs, len(self.body_ids), device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.stance_progress[env_ids] = 0.0

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        command_name: str,
        power: float = 2.0,
        min_phase_duration: float = 0.1,
        min_cmd_norm: float = 0.12,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        root_lin_vel_b_xy = asset.data.root_lin_vel_b[:, :2]
        command_xy = env.command_manager.get_command(command_name)[:, :2]
        in_contact = contact_sensor.data.current_contact_time[:, self.body_ids] > 0.0
        first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, self.body_ids]
        first_air = contact_sensor.compute_first_air(env.step_dt)[:, self.body_ids]
        last_contact_time = contact_sensor.data.last_contact_time[:, self.body_ids]
        reward, self.stance_progress = stride_length_reward_step(
            root_lin_vel_b_xy, command_xy, in_contact, first_contact, first_air,
            last_contact_time, self.stance_progress, env.step_dt, power,
            min_phase_duration, min_cmd_norm,
        )
        return reward


class RewardTripodSchedule(ManagerTermBase):
    """Event credit for genuine tripod-support alternation (v5 crossing credit). See
    ``crab_hex_tripod_reward.tripod_swap_crossing_reward_step`` for the full math and the
    campaign history (v1-v4 addenda) that led to it. Uses *raw* per-foot contact -- the healthy
    gait's stance bouts are shorter than any useful debounce window."""

    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        self.body_ids = sensor_cfg.body_ids
        self.state = torch.zeros(env.num_envs, STATE_DIM, device=self.device)
        self.state[:, S_T_SINCE] = RESET_T_SINCE

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.state[env_ids] = 0.0
        self.state[env_ids, S_T_SINCE] = RESET_T_SINCE

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        command_name: str,
        min_cmd_norm: float = 0.12,
        ema_tau: float = 0.06,
        corr_tau: float = 0.20,
        min_period: float = 0.10,
        max_period: float = 0.60,
        min_amp: float = 0.15,
        var_min: float = 0.01,
        credit_scale: float = 1.0,
    ) -> torch.Tensor:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        contact = contact_sensor.data.current_contact_time[:, self.body_ids] > 0.0
        command_xy = env.command_manager.get_command(command_name)[:, :2]
        reward, self.state = tripod_swap_crossing_reward_step(
            contact, command_xy, self.state, env.step_dt,
            TRIPOD_A_IDX, TRIPOD_B_IDX, min_cmd_norm, ema_tau, corr_tau,
            min_period, max_period, min_amp, var_min, credit_scale,
        )
        return reward
