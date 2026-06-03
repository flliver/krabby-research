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
from parkour_isaaclab.envs.mdp import observations as mdp_observations
from parkour_tasks.crab_hexapod_task.mdp.observations import (
    CrabHexObservationDeltaYawOk,
    CrabHexParkourObservations,
)
from parkour_tasks.crab_hexapod_task.mdp.parkour_actions import CrabHexDelayedJointPositionActionCfg
from parkour_tasks.extreme_parkour_task.config.go2.parkour_mdp_cfg import (
    ActionsCfg,
    CommandsCfg,
    EventCfg,
    ParkourEventsCfg,
)

# Leg order must match between tibia joints and footpads for stance-gated knee shaping.
_CRAB_TIBIA_JOINT_NAMES = [
    "FL_Femur_Tibia_RevoluteJoint",
    "FR_Femur_Tibia_RevoluteJoint",
    "ML_Femur_Tibia_RevoluteJoint",
    "MR_Femur_Tibia_RevoluteJoint",
    "RL_Femur_Tibia_RevoluteJoint",
    "RR_Femur_Tibia_RevoluteJoint",
]
_CRAB_FOOT_BODY_NAMES = [
    "FL_Footpad",
    "FR_Footpad",
    "ML_Footpad",
    "MR_Footpad",
    "RL_Footpad",
    "RR_Footpad",
]

@configclass
class CrabHexFlatWalkActionsCfg:
    """Flat-walk: scale 0.24 and ±1 raw clip (matches runner clip_actions)."""

    joint_pos = CrabHexDelayedJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.24,
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
class CrabHexStudentActionsCfg(CrabHexFlatWalkActionsCfg):
    """Student distillation: same 0.24 / ±1 scale as 2b2 teacher."""

    def __post_init__(self):
        self.joint_pos.use_delay = True
        self.joint_pos.history_length = 8


@configclass
class CrabHexStudentObservationsCfg:
    """Crab hex student obs (depth + proprio); not inherited from Go2 ``StudentObservationsCfg``."""

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

    @configclass
    class DepthCameraPolicyCfg(ObsGroup):
        depth_cam = ObsTerm(
            func=mdp_observations.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "resize": (58, 87),
                "buffer_len": 2,
                "debug_vis": False,
            },
        )

    @configclass
    class DeltaYawOkPolicyCfg(ObsGroup):
        delta_yaw_ok = ObsTerm(
            func=CrabHexObservationDeltaYawOk,
            params={
                "parkour_name": "base_parkour",
                "threshold": 0.6,
            },
        )

    policy: PolicyCfg = PolicyCfg()
    depth_camera: DepthCameraPolicyCfg = DepthCameraPolicyCfg()
    delta_yaw_ok: DeltaYawOkPolicyCfg = DeltaYawOkPolicyCfg()


@configclass
class CrabHexRewardsCfg:
    """``KRABBY_HEX_TEACHER_MODE=full`` (default): Go2-style parkour — goal velocity primary."""

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
class CrabHexTeacherWarmupRewardsCfg(CrabHexRewardsCfg):
    """Stage-2 bridge: softer contact penalties, parkour goals, and flat-walk velocity tracking."""

    reward_collision = RewTerm(
        func=mdp_rewards.reward_collision,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["body", ".*_Hip", ".*_Femur"],
            ),
        },
    )
    reward_feet_edge = RewTerm(
        func=mdp_rewards.reward_feet_edge,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Footpad"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Footpad"),
            "parkour_name": "base_parkour",
        },
    )
    track_lin_vel_xy_exp = RewTerm(
        func=track_lin_vel_xy_exp,
        weight=1.25,
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


@configclass
class CrabHexTeacherBridgeRewardsCfg(CrabHexTeacherWarmupRewardsCfg):
    """``KRABBY_HEX_TEACHER_MODE=bridge``: easy mixed walk — velocity/posture primary, parkour goal/yaw off."""

    reward_hip_pos = RewTerm(
        func=mdp_rewards.reward_hip_pos,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Body_Hip_RevoluteJoint"])},
    )
    reward_tracking_goal_vel = RewTerm(
        func=mdp_rewards.reward_tracking_goal_vel_on_parkour,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "parkour_name": "base_parkour"},
    )
    reward_tracking_yaw = RewTerm(
        func=mdp_rewards.reward_tracking_yaw,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "parkour_name": "base_parkour"},
    )
    track_lin_vel_xy_exp = RewTerm(
        func=track_lin_vel_xy_exp,
        weight=2.2,
        params={"command_name": "base_velocity", "std": math.sqrt(0.02)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=track_ang_vel_z_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    reward_forward_progress_along_command = RewTerm(
        func=mdp_rewards.reward_forward_progress_along_command,
        weight=0.4,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "min_cmd_norm": 0.12,
            "max_speed_scale": 1.75,
        },
    )
    reward_orientation = RewTerm(
        func=mdp_rewards.reward_orientation_upright,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": "base_parkour",
        },
    )
    penalty_base_pitch_forward_linear = RewTerm(
        func=mdp_rewards.penalty_base_pitch_forward_linear,
        weight=-2.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "min_forward_speed_cmd": 0.12,
        },
    )
    penalty_low_forward_speed_when_commanded = RewTerm(
        func=mdp_rewards.penalty_low_forward_speed_when_commanded,
        weight=-3.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "min_forward_speed_cmd": 0.12,
            "min_actual_speed": 0.35,
        },
    )
    reward_feet_air_time_on_flat = RewTerm(
        func=mdp_rewards.reward_feet_air_time_on_flat,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Footpad"),
            "parkour_name": "base_parkour",
            "threshold": 0.05,
        },
    )
    reward_forward_speed_on_flat = RewTerm(
        func=mdp_rewards.reward_forward_speed_on_flat,
        weight=0.7,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": "base_parkour",
            "min_forward_speed_cmd": 0.12,
            "target_speed": 0.55,
            "max_bonus_speed": 0.85,
        },
    )
    penalty_backward_along_command = RewTerm(
        func=mdp_rewards.penalty_backward_along_command,
        weight=-1.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "min_forward_speed_cmd": 0.12,
        },
    )
    penalty_body_heading_error_l2 = RewTerm(
        func=mdp_rewards.penalty_body_heading_error_l2,
        weight=-1.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "min_forward_speed_cmd": 0.12,
        },
    )


@configclass
class CrabHexStage2BPhase1RewardsCfg(CrabHexTeacherBridgeRewardsCfg):
    """``KRABBY_HEX_TEACHER_MODE=2b1``: hybrid walk — bridge core + weak goal_vel (0.75) / yaw (0.2) aux."""

    reward_tracking_goal_vel = RewTerm(
        func=mdp_rewards.reward_tracking_goal_vel,
        weight=0.75,
        params={"asset_cfg": SceneEntityCfg("robot"), "parkour_name": "base_parkour"},
    )
    reward_tracking_yaw = RewTerm(
        func=mdp_rewards.reward_tracking_yaw,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot"), "parkour_name": "base_parkour"},
    )
    reward_hip_pos = RewTerm(
        func=mdp_rewards.reward_hip_pos,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Body_Hip_RevoluteJoint"])},
    )
    reward_feet_stumble = RewTerm(
        func=mdp_rewards.reward_feet_stumble,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Footpad")},
    )
    reward_lin_vel_z = RewTerm(
        func=mdp_rewards.reward_lin_vel_z,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "parkour_name": "base_parkour"},
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
    reward_dof_error = RewTerm(
        func=mdp_rewards.reward_dof_error,
        weight=-0.04,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_torques = RewTerm(
        func=mdp_rewards.reward_torques,
        weight=-0.00001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_dof_acc = RewTerm(
        func=mdp_rewards.reward_dof_acc,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_delta_torques = RewTerm(
        func=mdp_rewards.reward_delta_torques,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class CrabHexStage2BPhase2RewardsCfg(CrabHexStage2BPhase1RewardsCfg):
    """``KRABBY_HEX_TEACHER_MODE=2b2`` teacher-ready (phase 2): obstacle-walk for distillation.

    Global clearance +1.8, foot swing +2.0, swing-vz +0.4, recover +0.4; forward +0.25.
    Micro-swing penalty −0.2; anti-stall −0.8. Bridge velocity-primary aux zeroed.
    """

    # --- Zero bridge-primary aux (not in teacher-ready stack) ---
    track_lin_vel_xy_exp = RewTerm(
        func=track_lin_vel_xy_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.02)},
    )
    penalty_lin_vel_y = RewTerm(
        func=mdp_rewards.penalty_lin_vel_y_l2,
        weight=0.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )
    penalty_base_pitch_forward_linear = RewTerm(
        func=mdp_rewards.penalty_base_pitch_forward_linear,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "min_forward_speed_cmd": 0.12,
        },
    )
    penalty_low_forward_speed_when_commanded = RewTerm(
        func=mdp_rewards.penalty_low_forward_speed_when_commanded,
        weight=-0.8,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "min_forward_speed_cmd": 0.12,
            "min_actual_speed": 0.35,
        },
    )
    reward_feet_air_time_on_flat = RewTerm(
        func=mdp_rewards.reward_feet_air_time_on_flat,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Footpad"),
            "parkour_name": "base_parkour",
            "threshold": 0.05,
        },
    )
    reward_forward_speed_on_flat = RewTerm(
        func=mdp_rewards.reward_forward_speed_on_flat,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": "base_parkour",
            "min_forward_speed_cmd": 0.12,
            "target_speed": 0.55,
            "max_bonus_speed": 0.85,
        },
    )
    penalty_backward_along_command = RewTerm(
        func=mdp_rewards.penalty_backward_along_command,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "min_forward_speed_cmd": 0.12,
        },
    )
    penalty_body_heading_error_l2 = RewTerm(
        func=mdp_rewards.penalty_body_heading_error_l2,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "min_forward_speed_cmd": 0.12,
        },
    )
    reward_tracking_yaw_on_parkour = RewTerm(
        func=mdp_rewards.reward_tracking_yaw_on_parkour,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": "base_parkour",
            "command_name": "base_velocity",
            "min_forward_speed_cmd": 0.12,
        },
    )

    # --- Teacher-ready stack ---
    reward_forward_progress_along_command = RewTerm(
        func=mdp_rewards.reward_forward_progress_along_command,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "min_cmd_norm": 0.12,
            "max_speed_scale": 1.75,
        },
    )
    reward_obstacle_clearance = RewTerm(
        func=mdp_rewards.reward_obstacle_clearance,
        weight=1.8,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Footpad"),
            "parkour_name": "base_parkour",
            "command_name": "base_velocity",
            "min_goal_progress": 0.15,
            "min_forward_speed": 0.25,
            "min_forward_speed_cmd": 0.12,
            "max_tilt_gravity_xy_sq": 0.02,
        },
    )
    reward_foot_clearance = RewTerm(
        func=mdp_rewards.reward_foot_clearance,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Footpad"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Footpad"),
            "command_name": "base_velocity",
            "contact_force_threshold": 0.1,
            "min_clearance_m": 0.05,
            "max_clearance_m": 0.20,
            "min_forward_speed_cmd": 0.12,
            "ground_offset_from_root_m": -1.0,
            "parkour_name": "base_parkour",
        },
    )
    reward_recover_from_stall = RewTerm(
        func=mdp_rewards.RewardRecoverFromStall,
        weight=0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Footpad"),
            "parkour_name": "base_parkour",
            "command_name": "base_velocity",
            "min_forward_speed_cmd": 0.12,
            "min_actual_speed": 0.15,
            "stuck_contact_force": 15.0,
            "min_other_feet_loaded": 2,
            "max_tilt_gravity_xy_sq": 0.04,
        },
    )
    penalty_swing_min_clearance = RewTerm(
        func=mdp_rewards.penalty_swing_min_clearance,
        weight=-0.4,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Footpad"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Footpad"),
            "command_name": "base_velocity",
            "contact_force_threshold": 0.1,
            "min_clearance_m": 0.03,
            "min_forward_speed_cmd": 0.12,
            "ground_offset_from_root_m": -1.0,
            "parkour_name": "base_parkour",
        },
    )
    reward_swing_vertical_vel = RewTerm(
        func=mdp_rewards.RewardSwingVerticalVel,
        weight=0.8,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Footpad"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Footpad"),
            "parkour_name": "base_parkour",
            "command_name": "base_velocity",
            "contact_force_threshold": 0.1,
            "min_forward_speed_cmd": 0.12,
            "max_vertical_vel": 0.5,
            "ground_offset_from_root_m": -1.0,
        },
    )
    reward_tracking_goal_vel = RewTerm(
        func=mdp_rewards.reward_tracking_goal_vel,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "parkour_name": "base_parkour"},
    )
    reward_tracking_yaw = RewTerm(
        func=mdp_rewards.reward_tracking_yaw,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot"), "parkour_name": "base_parkour"},
    )
    reward_collision = RewTerm(
        func=mdp_rewards.reward_collision,
        weight=-3.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["body", ".*_Hip", ".*_Femur"],
            ),
        },
    )
    reward_feet_edge = RewTerm(
        func=mdp_rewards.reward_feet_edge,
        weight=-0.8,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Footpad"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Footpad"),
            "parkour_name": "base_parkour",
        },
    )
    reward_feet_stumble = RewTerm(
        func=mdp_rewards.reward_feet_stumble,
        weight=-0.8,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Footpad")},
    )
    reward_orientation = RewTerm(
        func=mdp_rewards.reward_orientation,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": "base_parkour",
        },
    )


@configclass
class CrabHexFlatWalkRewardsCfg:
    """Stage 1 **gait** rewards (``Isaac-Crab-Hex-Flat-Walk-v0``): speed + posture + footfall shaping; no parkour goals."""

    track_lin_vel_xy_exp = RewTerm(
        func=track_lin_vel_xy_exp,
        weight=1.25,
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
        weight=0.60,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "min_cmd_norm": 0.12,
            "max_speed_scale": 1.75,
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
        weight=0.40,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Footpad"),
            "threshold": 0.05,
        },
    )
    penalty_tibia_deviation_in_stance = RewTerm(
        func=mdp_rewards.penalty_joint_deviation_when_in_contact,
        weight=-0.28,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=_CRAB_TIBIA_JOINT_NAMES,
                preserve_order=True,
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=_CRAB_FOOT_BODY_NAMES,
                preserve_order=True,
            ),
            "contact_force_threshold": 0.1,
        },
    )
    penalty_foot_idle_when_forward = RewTerm(
        func=mdp_rewards.PenaltyFootIdleWhenForward,
        weight=-0.12,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=_CRAB_FOOT_BODY_NAMES,
                preserve_order=True,
            ),
            "max_idle_steps": 60,
            "contact_force_threshold": 0.1,
            "min_forward_speed_cmd": 0.12,
        },
    )
    penalty_excess_feet_contact_forward = RewTerm(
        func=mdp_rewards.penalty_excess_feet_in_contact_forward,
        weight=-0.20,
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
    ``minimum_root_height_z``, ``hip_contact_sensor_cfg``. Env: ``KRABBY_HEX_TEACHER_MODE`` / spawn documented in scene/env cfgs.
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
