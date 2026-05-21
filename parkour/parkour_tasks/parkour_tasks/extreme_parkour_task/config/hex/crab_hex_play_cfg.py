"""Crab Hex (18-DOF hexapod) joystick config: scene loads robot from USD via KRABBY_HEX_USD_PATH."""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from parkour_isaaclab.actuators.parkour_actuator_cfg import ParkourDCMotorCfg
from parkour_tasks.crab_hexapod_task.mdp.observations import CrabHexParkourObservations
from parkour_tasks.extreme_parkour_task.config.go2.parkour_teacher_cfg import (
    ParkourTeacherSceneCfg,
    UnitreeGo2TeacherParkourEnvCfg_PLAY,
    TeacherObservationsCfg,
)
from parkour_tasks.extreme_parkour_task.config.go2.parkour_mdp_cfg import (
    TeacherRewardsCfg,
    RewTerm,
    ActionsCfg,
)
from parkour_isaaclab.envs.mdp import rewards as mdp_rewards
from parkour_isaaclab.envs.mdp.parkour_actions import DelayedJointPositionActionCfg
def _crab_hex_robot_cfg():
    """Build robot ArticulationCfg from KRABBY_HEX_USD_PATH (18 joints)."""
    usd_path = os.environ.get(
        "KRABBY_HEX_USD_PATH",
        "/workspace/assets/crab_hex_ref.usd",
    )
    base_legs_cfg = ParkourDCMotorCfg(
        joint_names_expr=[".*"],
        effort_limit={".*": 40.0},
        saturation_effort={".*": 45.0},
        velocity_limit={".*": 30.1},
        stiffness=40.0,
        damping=1.0,
        friction=0.0,
    )
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=True,
        ),
        actuators={"base_legs": base_legs_cfg},
    )
    if getattr(robot.spawn, "articulation_props", None) is not None:
        robot.spawn.articulation_props.enabled_self_collisions = True
    return robot


@configclass
class CrabHexSceneCfg(ParkourTeacherSceneCfg):
    """Scene with hexapod robot loaded from USD (path from KRABBY_HEX_USD_PATH)."""

    def __post_init__(self):
        super().__post_init__()
        self.robot = _crab_hex_robot_cfg()
        self.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/crab_hex"
        # Crab hex USD has no contact reporter API; disable contact sensor so scene builds.
        self.contact_forces = None


@configclass
class CrabHexObservationsCfg(TeacherObservationsCfg):
    """Observations for crab_hex: sensor_cfg points at robot (no contact_forces scene entity)."""

    @configclass
    class PolicyCfg(ObsGroup):
        extreme_parkour_observations = ObsTerm(
            func=CrabHexParkourObservations,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("robot", body_names=".*_tibia_tip"),
                "parkour_name": "base_parkour",
                "history_length": 10,
            },
            clip=(-100, 100),
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class CrabHexRewardsCfg(TeacherRewardsCfg):
    """Rewards for crab_hex: contact-dependent terms use robot entity (no contact_forces); they return 0 at runtime."""

    reward_collision = RewTerm(
        func=mdp_rewards.reward_collision,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("robot", body_names=["base_link", ".*_tibia", ".*_femur"]),
        },
    )
    reward_feet_edge = RewTerm(
        func=mdp_rewards.reward_feet_edge,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_tibia_tip"),
            "sensor_cfg": SceneEntityCfg("robot", body_names=".*_tibia_tip"),
            "parkour_name": "base_parkour",
        },
    )
    reward_feet_stumble = RewTerm(
        func=mdp_rewards.reward_feet_stumble,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("robot", body_names=".*_tibia_tip"),
        },
    )
    # Crab hex uses .*_hip_yaw / .*_hip_pitch, not .*_hip_joint (Go2).
    reward_hip_pos = RewTerm(
        func=mdp_rewards.reward_hip_pos,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw", ".*_hip_pitch"]),
        },
    )


@configclass
class CrabHexActionsCfg(ActionsCfg):
    """Actions for crab_hex: only 18 actuated joints (USD has 24; exclude .*_knee passive)."""

    joint_pos = DelayedJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_hip_yaw", ".*_hip_pitch", ".*_knee_actuator_joint"],
        scale=0.25,
        use_default_offset=True,
        action_delay_steps=[1, 1],
        delay_update_global_steps=24 * 8000,
        history_length=8,
        use_delay=True,
        clip={".*": (-4.8, 4.8)},
    )


@configclass
class CrabHexJoystickEnvCfg(UnitreeGo2TeacherParkourEnvCfg_PLAY):
    """Crab Hex joystick env: single env, same MDP as Play but scene uses crab_hex_ref.usd."""

    scene: CrabHexSceneCfg = CrabHexSceneCfg(num_envs=1, env_spacing=1.0)
    observations: CrabHexObservationsCfg = CrabHexObservationsCfg()
    rewards: CrabHexRewardsCfg = CrabHexRewardsCfg()
    actions: CrabHexActionsCfg = CrabHexActionsCfg()

    def __post_init__(self):
        # Parent sets self.scene.contact_forces.update_period; we have no contact sensor.
        dummy = type("DummyContactSensor", (), {"update_period": 0.02})()
        orig = self.scene.contact_forces
        self.scene.contact_forces = dummy
        super().__post_init__()
        self.scene.contact_forces = orig
        self.scene.num_envs = 1
        self.episode_length_s = 60.0
        # Crab hex uses base_link, not base; override event asset_cfg so body names resolve.
        base_link_cfg = SceneEntityCfg("robot", body_names="base_link")
        if self.events.base_external_force_torque is not None:
            self.events.base_external_force_torque.params["asset_cfg"] = base_link_cfg
        if self.events.randomize_rigid_body_mass is not None:
            self.events.randomize_rigid_body_mass.params["asset_cfg"] = base_link_cfg
        if self.events.randomize_rigid_body_com is not None:
            self.events.randomize_rigid_body_com.params["asset_cfg"] = base_link_cfg
