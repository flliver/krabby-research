

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal
import omni.usd
from isaaclab.assets import RigidObject,Articulation, AssetBase
from isaaclab.managers import SceneEntityCfg, ManagerTermBase
import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.actuators import DCMotor
from parkour_isaaclab.actuators import ParkourDCMotor
from isaaclab.sensors import RayCasterCamera
from isaaclab.utils.math import quat_from_euler_xyz

if TYPE_CHECKING:
    from isaaclab.envs import  ManagerBasedEnv
    from isaaclab.managers import EventTermCfg

def reset_joints_by_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    # bias these values randomly
    joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)
    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


# ---------------------------------------------------------------------------
# PLAN H (2026-09-03) obstacle-exposure spawn machinery
# ---------------------------------------------------------------------------
from parkour_isaaclab.envs.mdp.parkours.exposure_stats import (  # noqa: E402
    SPAWN_PLATFORM,
    SPAWN_RSI,
    SPAWN_SPREAD,
    global_height_field,
    local_to_world_x,
    patch_is_flat,
    sample_spread_x,
    world_to_pixel,
)

SPREAD_MAX_TRIES = 8
SPREAD_FLAT_RX_M = 0.3
"""Half-window along x for the spawn clearance check (no obstacle face within 0.3 m). The
flat stretches between obstacles are only 0.5-1.35 m long (gap/hurdle spacing 0.8-1.5 m
minus the obstacle, stones 0.7-1.0 m), so a wider window rejected nearly every in-field
draw on obstacle tiles (armed check 2026-09-03: all accepted spread spawns on obstacle
tiles landed on the platform)."""
SPREAD_FLAT_RY_M = 1.2
"""Half-window across y: covers the 1.19 m half-stance so no foot spawns in a side trench."""
SPREAD_FLAT_TOL_M = 0.05
"""Max height range inside the window; above the +-0.02 m tile noise, below the shallowest
gap (0.06) / hurdle (0.032 + noise) so an obstacle face inside the window rejects the draw."""


def note_spawn_to_parkour(env, env_ids: torch.Tensor, x_w: torch.Tensor, kind) -> None:
    """Tell every parkour term that implements ``note_spawn`` where env_ids were placed."""
    pm = getattr(env, "parkour_manager", None)
    if pm is None:
        return
    for name in pm.active_terms:
        term = pm.get_term(name)
        if hasattr(term, "note_spawn"):
            term.note_spawn(env_ids, x_w, kind)


def _height_field_cpu(env) -> torch.Tensor:
    """Global (rows*W, cols*L) int16 height field, cached on the env (CPU)."""
    hf = getattr(env, "_krabby_height_field_cpu", None)
    if hf is None:
        gen = env.scene.terrain.terrain_generator_class
        hf = global_height_field(torch.from_numpy(gen.height_fields)).contiguous()
        env._krabby_height_field_cpu = hf
    return hf


def _apply_spawn_spread(
    env, env_ids: torch.Tensor, positions: torch.Tensor, origin: torch.Tensor, default_z: torch.Tensor,
    spread: tuple[float, float], spread_frac: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move a Bernoulli(spread_frac) subset of spawns to a uniform tile-local x in ``spread``.

    Upper bound clamped per env to the last obstacle's goal x - 0.5 m; z from the height
    field; draws whose clearance window is not flat are re-drawn (up to SPREAD_MAX_TRIES),
    then fall back to the platform spawn. Returns (positions, spawn_kind).
    """
    tg = env.scene.terrain.cfg.terrain_generator
    n = len(env_ids)
    kind = torch.full((n,), SPAWN_PLATFORM, dtype=torch.long, device=env.device)
    pick = torch.rand(n, device=env.device) < spread_frac
    if not bool(pick.any()):
        return positions, kind
    hf = _height_field_cpu(env)
    size_x, size_y = float(tg.size[0]), float(tg.size[1])
    hscale, vscale = float(tg.horizontal_scale), float(tg.vertical_scale)
    rows_offset = size_x * tg.num_rows / 2
    cols_offset = size_y * tg.num_cols / 2
    rx_px, ry_px = int(round(SPREAD_FLAT_RX_M / hscale)), int(round(SPREAD_FLAT_RY_M / hscale))
    tol_px = SPREAD_FLAT_TOL_M / vscale
    lo, hi = float(spread[0]), float(spread[1])
    hi_t = torch.full((n,), hi, device=env.device)
    pm = getattr(env, "parkour_manager", None)
    if pm is not None:
        for name in pm.active_terms:
            term = pm.get_term(name)
            if hasattr(term, "env_goals") and hasattr(term, "num_goals"):
                last_obst_local = term.env_goals[env_ids, term.num_goals - 2, 0] + 0.5 * size_x
                hi_t = torch.minimum(hi_t, last_obst_local - 0.5)
                break
    accepted = torch.zeros(n, dtype=torch.bool, device=env.device)
    x_local = torch.zeros(n, device=env.device)
    z_terr = torch.zeros(n, device=env.device)
    for _ in range(SPREAD_MAX_TRIES):
        todo = pick & ~accepted
        if not bool(todo.any()):
            break
        u = torch.rand(n, device=env.device)
        xl = sample_spread_x(u, lo, hi_t)
        x_w = local_to_world_x(xl, origin[:, 0], size_x)
        ix, iy = world_to_pixel(x_w.cpu(), origin[:, 1].cpu(), rows_offset, cols_offset, hscale, tuple(hf.shape))
        flat = patch_is_flat(hf, ix, iy, rx_px, ry_px, tol_px).to(env.device)
        ok = flat & todo
        x_local[ok] = xl[ok]
        z_terr[ok] = (hf[ix, iy].float() * vscale).to(env.device)[ok]
        accepted |= ok
    positions = positions.clone()
    positions[accepted, 0] = local_to_world_x(x_local, origin[:, 0], size_x)[accepted]
    positions[accepted, 2] = default_z[accepted] + z_terr[accepted]
    kind[accepted] = SPAWN_SPREAD
    return positions, kind


def reset_root_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    offset: float = 3.0,
    spread: tuple[float, float] | None = None,
    spread_frac: float = 0.5,
):
    """Place resetting envs on their tile's platform (tile-local x = size_y+... see below).

    PLAN H (2026-09-03): ``spread``/``spread_frac`` (B3, ``KRABBY_SPAWN_SPREAD``) move a
    fraction of spawns along the course; unset = bit-identical to before. Every spawn is
    reported to the parkour term (``note_spawn``) for the exposure telemetry (B0).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
    root_states = asset.data.default_root_state[env_ids].clone()
    origin = env.scene.env_origins[env_ids].clone()
    origin[:,-1] = 0
    positions = root_states[:, 0:3] + origin - \
        torch.tensor((terrain_gen_cfg.size[1] + offset, 0, 0)).to(env.device)
    spawn_kind = None
    if spread is not None and spread_frac > 0.0:
        positions, spawn_kind = _apply_spawn_spread(
            env, env_ids, positions, origin, root_states[:, 2], spread, spread_frac
        )
    asset.write_root_pose_to_sim(torch.cat([positions, root_states[:, 3:7]], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_states[:, 7:13] , env_ids=env_ids) ## it mush need for init vel
    note_spawn_to_parkour(env, env_ids, positions[:, 0], SPAWN_PLATFORM if spawn_kind is None else spawn_kind)

def randomize_actuator_gains(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    stiffness_distribution_params: tuple[float, float] | None = None,
    damping_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    def randomize(data: torch.Tensor, params: tuple[float, float]) -> torch.Tensor:
        return _randomize_prop_by_op(
            data, params, dim_0_ids=None, dim_1_ids=actuator_indices, operation=operation, distribution=distribution
        )

    for actuator in asset.actuators.values():
        if isinstance(asset_cfg.joint_ids, slice):
            actuator_indices = slice(None)
            if isinstance(actuator.joint_indices, slice):
                global_indices = slice(None)
            else:
                global_indices = torch.tensor(actuator.joint_indices, device=asset.device)
        elif isinstance(actuator.joint_indices, slice):
            global_indices = actuator_indices = torch.tensor(asset_cfg.joint_ids, device=asset.device)
        else:
            actuator_joint_indices = torch.tensor(actuator.joint_indices, device=asset.device)
            asset_joint_ids = torch.tensor(asset_cfg.joint_ids, device=asset.device)
            # the indices of the joints in the actuator that have to be randomized
            actuator_indices = torch.nonzero(torch.isin(actuator_joint_indices, asset_joint_ids)).view(-1)
            if len(actuator_indices) == 0:
                continue
            global_indices = actuator_joint_indices[actuator_indices]
        if stiffness_distribution_params is not None:
            stiffness = actuator.stiffness[env_ids].clone()
            stiffness[:, actuator_indices] = asset.data.default_joint_stiffness[env_ids][:, global_indices].clone()
            randomize(stiffness, stiffness_distribution_params)
            actuator.stiffness[env_ids] = stiffness
            if isinstance(actuator, DCMotor) or isinstance(actuator, ParkourDCMotor):
                asset.write_joint_stiffness_to_sim(stiffness, joint_ids=actuator.joint_indices, env_ids=env_ids)
        # Randomize damping
        if damping_distribution_params is not None:
            damping = actuator.damping[env_ids].clone()
            damping[:, actuator_indices] = asset.data.default_joint_damping[env_ids][:, global_indices].clone()
            randomize(damping, damping_distribution_params)
            actuator.damping[env_ids] = damping
            if isinstance(actuator, DCMotor) or isinstance(actuator, ParkourDCMotor):
                asset.write_joint_damping_to_sim(damping, joint_ids=actuator.joint_indices, env_ids=env_ids)

def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()
    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples
    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)

def push_by_setting_velocity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):  
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    vel_w = asset.data.root_vel_w[env_ids]
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    random_noise = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    vel_w[:,:2] = random_noise[:,:2]
    vel_w[:,2:] += random_noise[:,2:]
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)

def random_camera_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    sensor_cfg: SceneEntityCfg,
    pos_noise_range: dict[str,tuple[float,float]] | None = None,
    rot_noise_range: dict[str,tuple[float,float]] | None = None,
    convention: str = 'ros',
):
    """
    prestartup
    """
    camera_sensor: RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    init_rot = torch.tensor(camera_sensor.cfg.offset.rot).repeat(env.num_envs,1).to(env.device)

    if pos_noise_range is not None: 
        pos_range_list = [pos_noise_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
        pos_ranges = torch.tensor(pos_range_list, device=env.device)
        random_pose = math_utils.sample_uniform(pos_ranges[:,0], pos_ranges[:,1], (env.num_envs,1), device=env.device)
    else:
        random_pose = None
    if rot_noise_range is not None:
        rot_range_list = [rot_noise_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
        rot_ranges = torch.deg2rad(torch.tensor(rot_range_list)).to(env.device)
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(init_rot)
        init_rot = torch.stack([roll, pitch, yaw], dim=-1).to(env.device)
        init_rot += math_utils.sample_uniform(rot_ranges[:,0], rot_ranges[:,1], (env.num_envs,1), device=env.device)
        random_rot = math_utils.quat_from_euler_xyz(init_rot[:,0],init_rot[:,1],init_rot[:,2])
    else:
        random_rot = init_rot 

    camera_sensor.set_world_poses(
        positions=random_pose,
        orientations=random_rot,
        convention=convention,
        env_ids=torch.arange(env.num_envs, dtype=torch.int64, device=env.device),
    )
    
class randomize_rigid_body_material(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            ValueError: If the asset is not a RigidObject or an Articulation.
        """
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if not isinstance(self.asset, (RigidObject, Articulation)):
            raise ValueError(
                f"Randomization term 'randomize_rigid_body_material' not supported for asset: '{self.asset_cfg.name}'"
                f" with type: '{type(self.asset)}'."
            )
        if isinstance(self.asset, Articulation) and self.asset_cfg.body_ids != slice(None):
            self.num_shapes_per_body = []
            for link_path in self.asset.root_physx_view.link_paths[0]:
                link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
                self.num_shapes_per_body.append(link_physx_view.max_shapes)
            num_shapes = sum(self.num_shapes_per_body)
            expected_shapes = self.asset.root_physx_view.max_shapes
            if num_shapes != expected_shapes:
                raise ValueError(
                    "Randomization term 'randomize_rigid_body_material' failed to parse the number of shapes per body."
                    f" Expected total shapes: {expected_shapes}, but got: {num_shapes}."
                )
        else:
            self.num_shapes_per_body = None

        # obtain parameters for sampling friction and restitution values
        friction_range = cfg.params.get("friction_range", (1.0, 1.0))
        restitution_range = cfg.params.get("restitution_range", (0.,0.))
        num_buckets = int(cfg.params.get("num_buckets", 1))
        range_list = [friction_range, (0,0), restitution_range]
        ranges = torch.tensor(range_list, device="cpu")
        self.material_buckets = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_buckets, 3), device="cpu")
        self.material_buckets[:,1] = self.material_buckets[:,0]

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        friction_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        bucket_ids = torch.randint(0, num_buckets, (len(env_ids),), device="cpu")
        material_samples = self.material_buckets[bucket_ids]
        total_num_shapes = self.asset.root_physx_view.max_shapes
        material_samples = material_samples.unsqueeze(1).repeat(1,total_num_shapes,1)
        # retrieve material buffer from the physics simulation
        materials = self.asset.root_physx_view.get_material_properties()
        # update material buffer with new samples
        if self.num_shapes_per_body is not None:
            # sample material properties from the given ranges
            for body_id in self.asset_cfg.body_ids:
                # obtain indices of shapes for the body
                start_idx = sum(self.num_shapes_per_body[:body_id])
                end_idx = start_idx + self.num_shapes_per_body[body_id]
                # assign the new materials
                # material samples are of shape: num_env_ids x total_num_shapes 
                materials[env_ids, start_idx:end_idx] = material_samples[:, start_idx:end_idx]
        else:
            # assign all the materials
            materials[env_ids] = material_samples[:]

        # apply to simulation
        self.asset.root_physx_view.set_material_properties(materials, env_ids)
