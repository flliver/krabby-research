import math

from isaaclab.envs.mdp.rewards import track_ang_vel_z_exp, track_lin_vel_xy_exp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards import feet_slide

from parkour_isaaclab.envs.mdp import rewards as mdp_rewards
from parkour_isaaclab.envs.mdp import terminations as parkour_terminations
from parkour_tasks.crab_hexapod_task.config.crab_hex.crab_hex_mdp_terminations import (
    terminate_crab_hex_failure,
)
from parkour_tasks.crab_hexapod_task.mdp.observations import CrabHexParkourObservations
from parkour_tasks.crab_hexapod_task.mdp.parkour_actions import CrabHexDelayedJointPositionActionCfg
from parkour_tasks.extreme_parkour_task.config.go2.parkour_mdp_cfg import (
    ActionsCfg,
    CommandsCfg,
    EventCfg,
    ParkourEventsCfg,
    StudentObservationsCfg,
)

@configclass
class CrabHexFlatWalkActionsCfg:
    """Flat-walk: scale 0.20 and ±1 raw clip (matches runner clip_actions)."""

    joint_pos = CrabHexDelayedJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.20,
        use_default_offset=True,
        action_delay_steps=[1, 1],
        delay_update_global_steps=24 * 8000,
        history_length=1,
        use_delay=False,
        clip={".*": (-1.0, 1.0)},
    )


@configclass
class CrabHexTeacherObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        extreme_parkour_observations = ObsTerm(
            func=CrabHexParkourObservations,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Footpad"),
                "parkour_name": "base_parkour",
                "history_length": 10,
            },
            clip=(-100, 100),
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class CrabHexStudentObservationsCfg(StudentObservationsCfg):
    @configclass
    class PolicyCfg(ObsGroup):
        extreme_parkour_observations = ObsTerm(
            func=CrabHexParkourObservations,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Footpad"),
                "parkour_name": "base_parkour",
                "history_length": 10,
            },
            clip=(-100, 100),
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class CrabHexRewardsCfg:
    """Parkour rewards aligned with Go2 ``TeacherRewardsCfg`` weights; contact terms use ``contact_forces``."""

    reward_collision = RewTerm(
        func=mdp_rewards.reward_collision,
        weight=-6.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                # Include hip links: otherwise hip-on-terrain is not counted and the policy can
                # minimize torques by sitting on the hip (knees bent) with little collision signal.
                body_names=["body", ".*_Hip", ".*_Femur"],
            ),
        },
    )
    reward_feet_edge = RewTerm(
        func=mdp_rewards.reward_feet_edge,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Footpad"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Footpad"),
            "parkour_name": "base_parkour",
        },
    )
    reward_torques = RewTerm(
        func=mdp_rewards.reward_torques,
        weight=-0.00001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_dof_error = RewTerm(
        func=mdp_rewards.reward_dof_error,
        weight=-0.04,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_hip_pos = RewTerm(
        func=mdp_rewards.reward_hip_pos,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Body_Hip_RevoluteJoint"])},
    )
    reward_ang_vel_xy = RewTerm(
        func=mdp_rewards.reward_ang_vel_xy,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_action_rate = RewTerm(
        func=mdp_rewards.reward_action_rate,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_dof_acc = RewTerm(
        func=mdp_rewards.reward_dof_acc,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_lin_vel_z = RewTerm(
        func=mdp_rewards.reward_lin_vel_z,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": "base_parkour",
        },
    )
    reward_orientation = RewTerm(
        func=mdp_rewards.reward_orientation,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": "base_parkour",
        },
    )
    reward_feet_stumble = RewTerm(
        func=mdp_rewards.reward_feet_stumble,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Footpad")},
    )
    reward_tracking_goal_vel = RewTerm(
        func=mdp_rewards.reward_tracking_goal_vel,
        weight=2.25,
        params={"asset_cfg": SceneEntityCfg("robot"), "parkour_name": "base_parkour"},
    )
    reward_tracking_yaw = RewTerm(
        func=mdp_rewards.reward_tracking_yaw,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "parkour_name": "base_parkour"},
    )
    reward_delta_torques = RewTerm(
        func=mdp_rewards.reward_delta_torques,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class CrabHexFlatWalkRewardsCfg:
    """Flat-walk: velocity tracking + orientation + light gait shaping (forward progress, swing)."""

    track_lin_vel_xy_exp = RewTerm(
        func=track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.02)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    penalty_lin_vel_y = RewTerm(
        func=mdp_rewards.penalty_lin_vel_y_l2,
        weight=-3.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )
    reward_forward_progress_along_command = RewTerm(
        func=mdp_rewards.reward_forward_progress_along_command,
        weight=0.50,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "min_cmd_norm": 0.12,
            "max_speed_scale": 1.65,
        },
    )
    reward_orientation = RewTerm(
        func=mdp_rewards.reward_orientation,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": "base_parkour",
        },
    )
    reward_lin_vel_z = RewTerm(
        func=mdp_rewards.reward_lin_vel_z,
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": "base_parkour",
        },
    )
    reward_ang_vel_xy = RewTerm(
        func=mdp_rewards.reward_ang_vel_xy,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_dof_error = RewTerm(
        func=mdp_rewards.reward_dof_error,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_feet_air_time_positive = RewTerm(
        func=mdp_rewards.reward_feet_air_time_positive,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Footpad"),
            "threshold": 0.05,
        },
    )
    penalty_excess_feet_contact_forward = RewTerm(
        func=mdp_rewards.penalty_excess_feet_in_contact_forward,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Footpad"),
            "max_feet_on_ground": 4,
            "contact_force_threshold": 0.1,
            "min_forward_speed_cmd": 0.12,
        },
    )
    reward_stance_support_feet_when_forward = RewTerm(
        func=mdp_rewards.reward_stance_support_feet_when_forward,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Footpad"),
            "min_feet_loaded": 3,
            "contact_force_threshold": 0.1,
            "min_forward_speed_cmd": 0.12,
        },
    )
    feet_slide = RewTerm(
        func=feet_slide,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Footpad"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Footpad"),
        },
    )
    reward_collision = RewTerm(
        func=mdp_rewards.reward_collision,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["body", ".*_Hip", ".*_Femur"],
            ),
        },
    )


@configclass
class CrabHexFlatWalkTerminationsCfg:
    """Relaxed terminations for flat-walk pretraining."""

    total_terminates = DoneTerm(
        func=parkour_terminations.terminate_episode,
        time_out=True,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    crab_failure = DoneTerm(
        func=terminate_crab_hex_failure,
        time_out=False,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "limit_angle": 0.5,
            "minimum_root_height_z": None,
            "contact_force_threshold": 500.0,
            "hip_contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_Hip"]),
        },
    )


@configclass
class CrabHexStudentRewardsCfg:
    """Same pattern as Go2 ``StudentRewardsCfg``: collision term weight 0; hex collision bodies on ``contact_forces``."""

    reward_collision = RewTerm(
        func=mdp_rewards.reward_collision,
        weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["body", ".*_Hip", ".*_Femur"],
            ),
        },
    )


@configclass
class CrabHexTerminationsCfg:
    """Parkour episode term (timeout / goal / legacy fall) plus crab-specific early failure.

    Tune ``crab_failure.params``: ``limit_angle``, ``contact_force_threshold``, optional
    ``minimum_root_height_z``, ``hip_contact_sensor_cfg``. Env: ``KRABBY_HEX_TRAIN_EASY`` / spawn documented in scene/env cfgs.
    """

    total_terminates = DoneTerm(
        func=parkour_terminations.terminate_episode,
        time_out=True,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    crab_failure = DoneTerm(
        func=terminate_crab_hex_failure,
        time_out=False,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "limit_angle": 1.5,
            "minimum_root_height_z": None,
            "contact_force_threshold": 500.0,
            # Hips only: chassis ``body`` contact was ending episodes during benign brushes.
            "hip_contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_Hip"]),
        },
    )
