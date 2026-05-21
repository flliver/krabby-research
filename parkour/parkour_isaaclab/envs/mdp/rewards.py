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
