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
from parkour_tasks.crab_hex_forward_task.mdp import crab_hex_clock_reward as clock_reward
from parkour_isaaclab.envs.mdp import terminations as parkour_terminations
from parkour_tasks.crab_hex_forward_task.config.crab_hex.crab_hex_mdp_terminations import (
    terminate_crab_hex_failure,
)
from parkour_isaaclab.envs.mdp import observations as mdp_observations
from parkour_tasks.crab_hex_forward_task.mdp.observations import (
    CrabHexObservationDeltaYawOk,
    CrabHexParkourObservations,
)
from parkour_tasks.crab_hex_forward_task.mdp.parkour_actions import CrabHexDelayedJointPositionActionCfg
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

# NOTE(cam-mechanism-migration): the 18 actually-actuated joints. Excludes
# *_Body_Hip_RevoluteJoint, which is now passive/kinematically-slaved to
# *_Body_CamShaft_RevoluteJoint (see crab_hex_cam_mapping.py) and must not receive a
# policy-commanded position target or appear in an unfiltered-sum reward (reward_dof_error,
# reward_torques).
_CRAB_ACTUATED_JOINT_NAMES = [
    ".*_Body_CamShaft_RevoluteJoint",
    ".*_Hip_Femur_RevoluteJoint",
    ".*_Femur_Tibia_RevoluteJoint",
]

# NOTE(cam-velocity-actions): the 12 position-actuated joints. The camshaft channels are
# velocity targets whose joint POSITION grows without bound under continuous rotation, so any
# position-deviation reward (reward_dof_error) must be restricted to this subset.
_CRAB_POSITION_ACTUATED_JOINT_NAMES = [
    ".*_Hip_Femur_RevoluteJoint",
    ".*_Femur_Tibia_RevoluteJoint",
]

# NOTE(cam-velocity-actions): raw +-1 on a camshaft channel maps to +-CAM_VEL_SCALE rad/s of
# commanded shaft speed (one gait cycle = 2*pi shaft rad).
# NOTE(hardware-measurements, 2026-08-20): the real yaw gearmotor runs ~30 RPM = pi rad/s
# (motor-sourcing doc, user-confirmed), so full-scale action = one gait cycle per 2 s.
# Replaces the 6.0 placeholder. Keep below the cam actuator's velocity_limit=8.0
# (see crab_hex_scene_cfg.py torque-speed note).
CAM_VEL_SCALE = math.pi

# NOTE(hardware-measurements, 2026-08-20): geometry-coupled reward constants, re-derived
# for the measured robot (legs ~35-40% longer than the old model).
# GROUND_OFFSET_FROM_ROOT_M: the nominal terrain height relative to the root, used by the
# clearance rewards; = -(settled root height above the terrain SURFACE). Measured
# (campaign 2026-08-20_1506_hardware_morphology, vertical-plate battery): settled root
# 1.064 above z=0, surface ~0.017 -> root-to-surface ~1.047 -> -1.05.
GROUND_OFFSET_FROM_ROOT_M = -1.05
# Swing clearance bands, scaled from the old 0.05/0.20 (and 0.03 micro-swing) by leg growth.
MIN_CLEARANCE_M = 0.07
MAX_CLEARANCE_M = 0.26
MIN_SWING_CLEARANCE_M = 0.04


def _crab_action_scale(pos_scale: float) -> dict[str, float]:
    """Per-joint action scale: velocity semantics (rad/s) on cam channels, rad on the rest."""
    return {
        ".*_Body_CamShaft_RevoluteJoint": CAM_VEL_SCALE,
        ".*_Hip_Femur_RevoluteJoint": pos_scale,
        ".*_Femur_Tibia_RevoluteJoint": pos_scale,
    }


def _crab_action_clip(pos_clip: tuple[float, float]) -> dict[str, tuple[float, float]]:
    """Per-joint raw-action clip: cam channels stay +-1 (full speed range) at every stage."""
    return {
        ".*_Body_CamShaft_RevoluteJoint": (-1.0, 1.0),
        ".*_Hip_Femur_RevoluteJoint": pos_clip,
        ".*_Femur_Tibia_RevoluteJoint": pos_clip,
    }

@configclass
class CrabHexFlatWalkActionsCfg:
    """Class-level defaults 0.24 / ±1 (re-applied by ``_apply_crab_hex_bridge_actions_and_events`` for
    the legacy bridge / 2b1 / 2b2 modes and inherited by the legacy ``CrabHexStudentActionsCfg``).
    Flat-Walk-v0 (mode ``full``), the phase-2 modes and the phase-3 student override these to 0.25
    (``KRABBY_ACTION_SCALE``) / ±4.8 with action delay via ``_apply_crab_hex_full_actions``; the
    runner's ``clip_actions = 1.0`` bounds the raw policy output.
    """

    joint_pos = CrabHexDelayedJointPositionActionCfg(
        asset_name="robot",
        # NOTE(cam-mechanism-migration): see _CRAB_ACTUATED_JOINT_NAMES — excludes the now-passive
        # FL_Body_Hip_RevoluteJoint. Keeps action_dim == 18.
        joint_names=_CRAB_ACTUATED_JOINT_NAMES,
        scale=_crab_action_scale(0.24),
        use_default_offset=True,
        action_delay_steps=[1, 1],
        delay_update_global_steps=24 * 8000,
        history_length=1,
        use_delay=False,
        clip=_crab_action_clip((-1.0, 1.0)),
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
    # NOTE(cam-mechanism-migration): filtered to the 18 actuated joints -- excludes the now-passive
    # *_Body_Hip_RevoluteJoint (kinematically-slaved to *_Body_CamShaft_RevoluteJoint, see
    # crab_hex_cam_mapping.py), whose "torque"/dof-error is meaningless (not force/PD-commanded
    # by the policy).
    reward_torques = RewTerm(
        func=mdp_rewards.reward_torques,
        weight=-0.00001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_CRAB_ACTUATED_JOINT_NAMES)},
    )
    reward_dof_error = RewTerm(
        func=mdp_rewards.reward_dof_error,
        weight=-0.04,
        # NOTE(cam-velocity-actions): position-actuated joints only — the camshaft position
        # is unbounded under continuous rotation.
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_CRAB_POSITION_ACTUATED_JOINT_NAMES)},
    )
    reward_hip_pos = RewTerm(
        func=mdp_rewards.reward_hip_pos,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Body_Hip_RevoluteJoint"])},
    )
    # NOTE(cam-mechanism-migration): see CrabHexFlatWalkRewardsCfg.penalty_motor_direction_reversal
    # for full rationale -- same medium-weight penalty, added here so it propagates to every
    # teacher-stage subclass (Warmup/Bridge/2b1/2b2) that doesn't re-override it.
    penalty_motor_direction_reversal = RewTerm(
        func=mdp_rewards.PenaltyMotorDirectionReversal,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Body_CamShaft_RevoluteJoint"])},
    )
    # NOTE(stride-length): see CrabHexFlatWalkRewardsCfg.reward_stride_length for full rationale
    # -- same starting weight/power, added here so it propagates to every teacher-stage subclass
    # (Warmup/Bridge/2b1/2b2) that doesn't re-override it.
    reward_stride_length = RewTerm(
        func=mdp_rewards.RewardStrideLength,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_CRAB_FOOT_BODY_NAMES, preserve_order=True),
            "command_name": "base_velocity",
            "power": 2.0,
            # NOTE(stride-length-v3): redefined to reward stance-phase body progress along the
            # commanded direction, not joint-space (hip-yaw) movement, and dropped swing-phase
            # reward entirely -- a foot in the air can't push the robot, so it shouldn't be
            # rewarded for moving. This also structurally eliminates the v2 snap exploit (a leg
            # covering its whole joint range in a single 20ms physics step): that exploit relied on
            # swing-phase joint displacement being rewarded regardless of whether the robot actually
            # moved, and swing is no longer rewarded at all, so the earlier velocity_cost_weight/
            # _power terms are no longer needed. min_phase_duration still guards against a spurious
            # one-step contact reading being trusted as a real stance. (v4, per-foot signed
            # touchdown displacement, was tried and reverted -- see
            # parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-09_0106_stride_length_v4/CHANGELOG.md.)
            "min_phase_duration": 0.1,
            "min_cmd_norm": 0.12,
        },
    )
    reward_ang_vel_xy = RewTerm(
        func=mdp_rewards.reward_ang_vel_xy,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # NOTE(teacher-carry-up): -0.1 -> -0.3, matching the flat-walk campaign's baked winner
    # (parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-09_0920_short_runs/CHANGELOG.md) after the T0/T1/T2 teacher-stack
    # screening study picked T2 (this weight kept, reversal penalty unchanged) over the control
    # and the full-mirror arm -- see parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-09_1526_gait_tuned/CHANGELOG.md.
    reward_action_rate = RewTerm(
        func=mdp_rewards.reward_action_rate,
        weight=-0.3,
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
    # NOTE(teacher-carry-up): -1e-7 -> -1e-6, see reward_action_rate's note above.
    reward_delta_torques = RewTerm(
        func=mdp_rewards.reward_delta_torques,
        weight=-1.0e-6,
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
    """``KRABBY_HEX_TEACHER_MODE=bridge``: easy mixed walk — velocity/posture primary, parkour goal/yaw off.

    NOTE(teacher-handoff, 2026-08-26): the measured-hardware plant broke this stack's
    economics — its positive income is tuned for old-plant speeds (net ≈ −3.2/step with
    the torque tax dominating; hand-off chain #1/#2 collapsed to 100% failure as dying
    early became optimal). Two plant recalibrations, inherited by 2b1/2b2:
    (a) the clock contact-schedule income (the plant's proven locomotion income, dims
        15-16 clock obs are already in the shared observation head) is registered here,
        armed by KRABBY_CLOCK_W exactly as in flat-walk;
    (b) penalty_low_forward_speed's min_actual_speed 0.35 belongs to the old plant
        (this one's steady walk is 0.10-0.20 m/s) — KRABBY_MIN_ACTUAL_SPEED overrides.
    """

    reward_clock_schedule = RewTerm(
        func=mdp_rewards.RewardClockContactSchedule,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_CRAB_FOOT_BODY_NAMES, preserve_order=True),
            "command_name": "base_velocity",
            "force_ref": clock_reward.FORCE_REF_N,
            "vel_ref": clock_reward.VEL_REF_M_S,
            "min_upright_gz": 0.9,
            "combine": "sum",
        },
    )

    def __post_init__(self):
        import os as _os

        _cw = _os.environ.get("KRABBY_CLOCK_W")
        if _cw is not None:
            self.reward_clock_schedule.weight = float(_cw)
        _mas = _os.environ.get("KRABBY_MIN_ACTUAL_SPEED")
        if _mas is not None:
            self.penalty_low_forward_speed_when_commanded.params["min_actual_speed"] = float(_mas)
        # NOTE(teacher-handoff chain #5 post-mortem): 2b1's full-dose goal/yaw income
        # taught the never-turned-before flat policy to attempt maneuvers it cannot
        # survive (failure tracked income upward, 0.0 -> 0.96 in one chunk; critic reset
        # irrelevant). Dose control for a Freitag-style ramp:
        _gv = _os.environ.get("KRABBY_GOAL_VEL_W")
        if _gv is not None and hasattr(self, "reward_tracking_goal_vel"):
            self.reward_tracking_goal_vel.weight = float(_gv)
        _yw = _os.environ.get("KRABBY_YAW_W")
        if _yw is not None and hasattr(self, "reward_tracking_yaw"):
            self.reward_tracking_yaw.weight = float(_yw)

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
            # NOTE(gait-formation Phase 0, 2026-08-20): 1.75 paid the wheelie exploit up
            # to 1.14 m/s (2x the command envelope); both smoke tests rode it into the
            # 0.5 rad tilt termination. 1.05 caps progress income at ~the command.
            # Override: KRABBY_MAX_SPEED_SCALE.
            "max_speed_scale": 1.05,
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
    # NOTE(teacher-carry-up): -0.1 -> -0.3, covers 2b1 and 2b2 (Stage2BPhase2RewardsCfg inherits
    # from this class without re-declaring). See CrabHexRewardsCfg.reward_action_rate's note.
    reward_action_rate = RewTerm(
        func=mdp_rewards.reward_action_rate,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_dof_error = RewTerm(
        func=mdp_rewards.reward_dof_error,
        weight=-0.04,
        # NOTE(cam-velocity-actions): position-actuated joints only — the camshaft position
        # is unbounded under continuous rotation.
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_CRAB_POSITION_ACTUATED_JOINT_NAMES)},
    )
    reward_torques = RewTerm(
        func=mdp_rewards.reward_torques,
        weight=-0.00001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_CRAB_ACTUATED_JOINT_NAMES)},
    )
    reward_dof_acc = RewTerm(
        func=mdp_rewards.reward_dof_acc,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # NOTE(teacher-carry-up): -1e-7 -> -1e-6, see reward_action_rate's note above.
    reward_delta_torques = RewTerm(
        func=mdp_rewards.reward_delta_torques,
        weight=-1.0e-6,
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
            # NOTE(gait-formation Phase 0, 2026-08-20): 1.75 paid the wheelie exploit up
            # to 1.14 m/s (2x the command envelope); both smoke tests rode it into the
            # 0.5 rad tilt termination. 1.05 caps progress income at ~the command.
            # Override: KRABBY_MAX_SPEED_SCALE.
            "max_speed_scale": 1.05,
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
            "min_clearance_m": MIN_CLEARANCE_M,
            "max_clearance_m": MAX_CLEARANCE_M,
            "min_forward_speed_cmd": 0.12,
            "ground_offset_from_root_m": GROUND_OFFSET_FROM_ROOT_M,
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
            "min_clearance_m": MIN_SWING_CLEARANCE_M,
            "min_forward_speed_cmd": 0.12,
            "ground_offset_from_root_m": GROUND_OFFSET_FROM_ROOT_M,
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
            "ground_offset_from_root_m": GROUND_OFFSET_FROM_ROOT_M,
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
            # NOTE(gait-formation Phase 0, 2026-08-20): 1.75 paid the wheelie exploit up
            # to 1.14 m/s (2x the command envelope); both smoke tests rode it into the
            # 0.5 rad tilt termination. 1.05 caps progress income at ~the command.
            # Override: KRABBY_MAX_SPEED_SCALE.
            "max_speed_scale": 1.05,
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
    # NOTE(tripod-stability-campaign): registered at weight 0.0 (inert) so it's Hydra-sweepable.
    # reward_orientation is direction-blind (roll^2+pitch^2); this term is signed and forward-gated,
    # so it's the only lever that can target the sustained ~12deg nose-down lean measured on the
    # baseline checkpoint (see parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-10_0058_tripod_stability/RESULTS.md).
    penalty_base_pitch_forward_linear = RewTerm(
        func=mdp_rewards.penalty_base_pitch_forward_linear,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "min_forward_speed_cmd": 0.12,
        },
    )
    # NOTE(tripod-stability-campaign): the explicit contact-schedule reward from Task 1 §2.3,
    # registered at weight 0.0 (inert). None of the config-only knobs tried in
    # parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-10_0058_tripod_stability/ moved tripod_score, so this term rewards
    # genuine tripod-set alternation directly -- see crab_hex_tripod_reward.py for the full math.
    # v5 pays an event credit per zero-crossing of the smoothed support difference between the
    # two tripod sets, scaled by swing amplitude and anti-correlation quality, inside a 0.10-0.60s
    # period band. Chosen after the offline npz replay gate showed v4's debounced swap detector
    # never fires on the healthy gait (stance bouts ~0.10s < any useful debounce) while paying
    # the slow degenerates -- see the v1-v5 addenda in crab_hex_tripod_reward.py and the campaign
    # RESULTS.md / offline_replay/ for the gate numbers.
    # NOTE(gait-formation-v2 Phase 1, 2026-08-22): the clock-referenced contact-schedule
    # reward — the stability review's ranked recommendation #1 (Siekmann/Walk These Ways
    # family), adopted after the crossing-credit term proved gradient-dead at zero behavior
    # across two campaigns on two plants. The alternating-tripod schedule exists in the
    # reward from step 0 (dense positive income; no holdable state earns while the clock
    # advances). The clock itself lives in CrabHexDelayedJointPositionAction.clock_phase;
    # the policy sees it as sin/cos obs dims. Arm via KRABBY_CLOCK_W (screens use ~1.0).
    reward_clock_schedule = RewTerm(
        func=mdp_rewards.RewardClockContactSchedule,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_CRAB_FOOT_BODY_NAMES, preserve_order=True),
            "command_name": "base_velocity",
            "force_ref": clock_reward.FORCE_REF_N,
            "vel_ref": clock_reward.VEL_REF_M_S,
            "min_upright_gz": 0.9,
            # KRABBY_CLOCK_COMBINE overrides ("sum" | "product") — see the pure function's
            # docstring for the creep-differential rationale (2026-08-22).
            "combine": "sum",
        },
    )
    # NOTE(gait-formation-v2 Phase 4, 2026-08-24): scheduled swing apex (WTW-style
    # commanded footswing height). Income-priced clearance lost to survival economics at
    # every dose (G1-G3/I1); the apex is specified in the clock schedule instead.
    # KRABBY_APEX_W arms it; KRABBY_APEX_M overrides the commanded height.
    reward_clock_swing_apex = RewTerm(
        func=mdp_rewards.RewardClockSwingApex,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "apex_m": clock_reward.APEX_TARGET_M,
            "sigma_m": clock_reward.APEX_SIGMA_M,
            "ground_offset_from_root_m": GROUND_OFFSET_FROM_ROOT_M,
            "min_upright_gz": 0.9,
        },
    )
    reward_tripod_schedule = RewTerm(
        func=mdp_rewards.RewardTripodSchedule,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_CRAB_FOOT_BODY_NAMES, preserve_order=True),
            "command_name": "base_velocity",
            "min_cmd_norm": 0.12,
            # NOTE(gait-formation Phase 0, 2026-08-20): band recalibrated to the pi-rad/s
            # cam (CAM_VEL_SCALE=pi -> 2.0 s gait cycle, A/B crossings ~1.0 s apart). The
            # old 0.10/0.60 band + corr_tau 0.20 were tuned for the 6 rad/s era's ~0.30 s
            # stride and were two independent kills at pi (out-of-band crossings AND
            # var < var_min from the too-fast correlation window). Overrides:
            # KRABBY_TRIPOD_MIN_PERIOD / MAX_PERIOD / CORR_TAU / MIN_AMP.
            "ema_tau": 0.06,
            "corr_tau": 0.70,
            "min_period": 0.30,
            "max_period": 1.40,
            "min_amp": 0.15,
            "var_min": 0.01,
            "credit_scale": 1.0,
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
        # NOTE(cam-velocity-actions): position-actuated joints only (excludes passive Body_Hip
        # AND the velocity-driven camshaft, whose position is unbounded under continuous spin).
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_CRAB_POSITION_ACTUATED_JOINT_NAMES)},
    )
    # NOTE(short-run-campaign): weight raised 0.40 -> 0.8, the winning value from the Milestone 18
    # Task 1 short-run tuning campaign (parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-09_0920_short_runs/CHANGELOG.md) -- threshold=0.05
    # itself was swept (0.10, 0.15) and found not to move gait quality on its own, so it stays at
    # default. Improves tippy_tap and measured stride together vs the untouched v3 baseline.
    reward_feet_air_time_positive = RewTerm(
        func=mdp_rewards.reward_feet_air_time_positive,
        weight=0.8,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Footpad"),
            # NOTE(task1-velocity C2, BAKED 2026-08-16; recalibrated gait-formation
            # Phase 0 2026-08-20): cam-derived swing target — return stroke 2.14 rad at
            # CAM_VEL_SCALE=pi is ~0.68 s, so 0.38 keeps the same ~0.55x selectivity the
            # 0.20 had at 6 rad/s. Override: KRABBY_AIRTIME_THRESH.
            "threshold": 0.38,
        },
    )
    # NOTE(stride-length): rewards |hip-yaw diff|**power across every touchdown<->liftoff
    # transition (both swing- and stance-phase leg movement) -- convex (power=2) so a single
    # long stride outscores several short ones covering the same net range, directly targeting
    # the "tippy-tap" micro-stepping pattern the gait-eval harness flags across every checkpoint
    # tested so far. Starting weight/power, tune against the harness (Milestone 18 Task 1).
    reward_stride_length = RewTerm(
        func=mdp_rewards.RewardStrideLength,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_CRAB_FOOT_BODY_NAMES, preserve_order=True),
            "command_name": "base_velocity",
            "power": 2.0,
            # NOTE(stride-length-v3): redefined to reward stance-phase body progress along the
            # commanded direction, not joint-space (hip-yaw) movement, and dropped swing-phase
            # reward entirely -- a foot in the air can't push the robot, so it shouldn't be
            # rewarded for moving. This also structurally eliminates the v2 snap exploit (a leg
            # covering its whole joint range in a single 20ms physics step): that exploit relied on
            # swing-phase joint displacement being rewarded regardless of whether the robot actually
            # moved, and swing is no longer rewarded at all, so the earlier velocity_cost_weight/
            # _power terms are no longer needed. min_phase_duration still guards against a spurious
            # one-step contact reading being trusted as a real stance. (v4, per-foot signed
            # touchdown displacement, was tried and reverted -- see
            # parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-09_0106_stride_length_v4/CHANGELOG.md.)
            "min_phase_duration": 0.1,
            "min_cmd_norm": 0.12,
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
            # NOTE(gait-formation Phase 0): 60 steps = 1.2 s taxed legitimate cam-timed
            # swings below full throttle (return stroke ~1.36 s at half speed). 90 = 1.8 s.
            "max_idle_steps": 90,
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
    # NOTE(PLAN F gated lineage, 2026-08-26): the teacher elements, registered inert in the
    # flat stack so the entire curriculum runs as one flat-task lineage. Goal terms use the
    # MASKED _on_parkour variants (zero on flat tiles) — the 2b1 collapse came from the
    # unmasked versions at full dose. Armed via KRABBY_GOAL_VEL_W / KRABBY_YAW_W /
    # KRABBY_EDGE_W / KRABBY_STUMBLE_W (collision arms the existing term above).
    reward_tracking_goal_vel = RewTerm(
        func=mdp_rewards.reward_tracking_goal_vel_on_parkour,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "parkour_name": "base_parkour"},
    )
    reward_tracking_yaw = RewTerm(
        func=mdp_rewards.reward_tracking_yaw_on_parkour,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "parkour_name": "base_parkour"},
    )
    reward_feet_edge = RewTerm(
        func=mdp_rewards.reward_feet_edge,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Footpad"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Footpad"),
            "parkour_name": "base_parkour",
        },
    )
    reward_feet_stumble = RewTerm(
        func=mdp_rewards.reward_feet_stumble,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Footpad")},
    )
    # NOTE(cam-mechanism-migration, superseded by short-run-campaign): originally penalized the
    # cam-shaft motor reversing rotational direction at weight=-0.3. The Milestone 18 Task 1
    # short-run campaign (parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-09_0920_short_runs/CHANGELOG.md) ran a 3-arm study testing this
    # mechanism against general action-smoothness terms for the same goal (suppressing
    # high-frequency reversals): retuning this weight alone (-0.15, -0.6) never beat the
    # air-time-only baseline, but turning it OFF and using reward_action_rate/reward_delta_torques
    # instead (below) was the single best result of the whole campaign -- best tippy_tap AND best
    # measured stride simultaneously, at both 1000 and 2000 iterations. Weight zeroed accordingly;
    # left registered (not deleted) so it can be reintroduced if the smoothness-only route proves
    # insufficient once carried up the teacher stack, where reversal count remains a real
    # hardware-longevity concern documented in motor_reversal_on/CHANGELOG.md.
    penalty_motor_direction_reversal = RewTerm(
        func=mdp_rewards.PenaltyMotorDirectionReversal,
        # NOTE(task1-velocity campaign, UNBAKED 2026-08-15): the -0.3 bake (2026-08-14) was
        # rescinded — its justifying evidence was scored on a landscape where the dominant
        # tracking term was inert (velocity-era command-blindness, onedir-spin RESULTS.md).
        # Effective stack returns to the position-era value; re-enable via KRABBY_REVERSAL_W
        # only through the Task-1 campaign's replay-gate + screen discipline.
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Body_CamShaft_RevoluteJoint"])},
    )
    # NOTE(short-run-campaign): weights raised from 0.0 (previously inactive on flat-walk, only
    # used in the teacher-stage rewards above) to the winning values from the 3-arm study
    # described above -- see parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-09_0920_short_runs/CHANGELOG.md for the full comparison.
    # Combined with reward_feet_air_time_positive=0.8 and penalty_motor_direction_reversal=0.0,
    # this config cleared every Task 1 §1c/§1f target at a 2000-iteration validation: tippy_tap
    # 19.09%->13.39% (<= baseline's 13.5%), measured stride 0.178m->0.1975m (best of the entire
    # Task 1 comparison series), schedule_completion_rate steady at 100%. Training-time tracking
    # error was somewhat elevated in testing (policy trades a little velocity-tracking precision
    # for smoother actions) but did not show up as a gait-eval regression on flat-walk -- worth
    # watching if this config is carried up the teacher stack.
    reward_action_rate = RewTerm(
        func=mdp_rewards.reward_action_rate,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_delta_torques = RewTerm(
        func=mdp_rewards.reward_delta_torques,
        weight=-1e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # NOTE(onedir-spin-campaign round 3): positive shaping toward continuous one-direction
    # shaft spin — the penalty route was closed by a 5-point dose-response (absorbed up to
    # -0.6, locomotion collapse at -1.0). Weight 0.0 until screened; KRABBY_SPIN_REWARD_W.
    reward_one_direction_spin = RewTerm(
        func=mdp_rewards.RewardOneDirectionSpin,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Body_CamShaft_RevoluteJoint"]),
            "command_name": "base_velocity",
            # NOTE(gait-formation Phase 0): default speed_ref=4.0 > CAM_VEL_SCALE=pi
            # capped this term at 0.785 of nominal forever; 2.8 = 0.9*pi.
            "speed_ref": 2.8,
        },
    )
    # NOTE(onedir-spin round 4, lit-review synthesis): contact schedule referenced to each
    # leg's own cam phase (stance on power stroke / swing on return stroke) + in-set cam
    # phase locking toward tripod. Both replica-gated offline (ideal pays 0 / earns 59;
    # misaligned gaits pay 23-33 / earn 3-26). Weights 0.0 until screened.
    penalty_cam_contact_schedule = RewTerm(
        func=mdp_rewards.PenaltyCamContactSchedule,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Footpad"),
        },
    )
    reward_cam_phase_lock = RewTerm(
        func=mdp_rewards.RewardCamPhaseLock,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "speed_ref": 2.8},
    )
    # NOTE(staged-ramp): physics-grounded basin selector — spin gaits are ~42% cheaper in
    # total |tau*qdot| than oscillation (see penalty_mechanical_power docstring).
    penalty_mechanical_power = RewTerm(
        func=mdp_rewards.penalty_mechanical_power,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # NOTE(task1-velocity C1, BAKED 2026-08-16): constant-gradient tracking pressure — the
    # change that restored command-following to the velocity era (screen: deficits <=0.08
    # vs the era's 0.25-0.48 command-blindness). Override: KRABBY_TRACK_L1_W.
    penalty_tracking_error_l1 = RewTerm(
        func=mdp_rewards.penalty_tracking_error_l1,
        weight=-0.5,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )

    # NOTE(phased-flat Phase C 2026-08-18): clearance instrument for the light-terrain
    # mode (KRABBY_FLAT_TERRAIN_MODE). Teacher-validated term, same params as the 2b
    # stack; weight 0.0 keeps the flat stack unchanged on 100% flat. Phase-C runs set
    # KRABBY_CLEARANCE_W=0.01 (instrument scale: income <=0.01x raw, negligible vs the
    # ~19-28 locomotion income) purely so Episode_Reward/reward_obstacle_clearance
    # tracks lifting for the C gate; larger weights are an experiment knob if lifting lags.
    # NOTE(C3 gate relaxation 2026-08-18): the teacher gates (speed 0.25 / progress 0.15)
    # exclude slow first crossings — C1/C2 video showed the policy creeps at obstacles,
    # so the term paid ~0 and income *declined* as flat-majority optimization won.
    # Relaxed for the flat-light stage so bootstrap crossings pay; the teacher stack's
    # copy keeps the strict gates.
    reward_obstacle_clearance = RewTerm(
        func=mdp_rewards.reward_obstacle_clearance,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Footpad"),
            "parkour_name": "base_parkour",
            "command_name": "base_velocity",
            "min_goal_progress": 0.05,
            "min_forward_speed": 0.10,
            "min_forward_speed_cmd": 0.12,
            "max_tilt_gravity_xy_sq": 0.02,
        },
    )

    # NOTE(C5 2026-08-18): dense swing-height shaper from the teacher stack — the
    # sparse obstacle_clearance bonus alone plateaued at ~0.004 income (C0-C4); in the
    # teacher stack this +2.0 term is the workhorse that teaches lifting mechanics and
    # obstacle_clearance is only the outcome bonus. Instrument 0.0; KRABBY_FOOT_CLEAR_W.
    reward_foot_clearance = RewTerm(
        func=mdp_rewards.reward_foot_clearance,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Footpad"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Footpad"),
            "command_name": "base_velocity",
            "contact_force_threshold": 0.1,
            "min_clearance_m": MIN_CLEARANCE_M,
            "max_clearance_m": MAX_CLEARANCE_M,
            "min_forward_speed_cmd": 0.12,
            "ground_offset_from_root_m": GROUND_OFFSET_FROM_ROOT_M,
            "parkour_name": "base_parkour",
        },
    )
    # NOTE(gait-formation Phase 0, 2026-08-20): micro-swing penalty from the 2b2 teacher
    # stack, registered here inert for the lift phase. KRABBY_SWING_MIN_CLEAR_W. Like
    # foot_clearance it is masked to zero on parkour_flat tiles; KRABBY_FOOT_CLEAR_FLAT=1
    # lifts that mask on both terms (parkour_name -> None) for pure-flat lift shaping.
    penalty_swing_min_clearance = RewTerm(
        func=mdp_rewards.penalty_swing_min_clearance,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Footpad"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Footpad"),
            "command_name": "base_velocity",
            "contact_force_threshold": 0.1,
            "min_clearance_m": MIN_SWING_CLEARANCE_M,
            "min_forward_speed_cmd": 0.12,
            "ground_offset_from_root_m": GROUND_OFFSET_FROM_ROOT_M,
            "parkour_name": "base_parkour",
        },
    )

    def __post_init__(self):
        # NOTE(onedir-spin-campaign 2026-08-13): env-var override for the camshaft
        # direction-reversal penalty weight, for the velocity-era weight screens. The
        # position-era evidence that zeroed this term does not transfer: under velocity
        # actions it prices literal shaft spin direction (offline replay: the -0.1/-0.3
        # candidates cost the current oscillator 3.3%/9.9% of locomotion income; a true
        # one-direction spin pays exactly 0). Default 0.0 preserves current behavior
        # until the screen winner is baked.
        import os

        # Presence-based overrides: an explicitly-set env var wins even at 0.0 (needed to
        # DISABLE baked defaults, e.g. staged-ramp phase A turns the -0.3 reversal off).
        _overrides = {
            "KRABBY_REVERSAL_W": "penalty_motor_direction_reversal",
            "KRABBY_SPIN_REWARD_W": "reward_one_direction_spin",
            "KRABBY_CAM_SCHED_W": "penalty_cam_contact_schedule",
            "KRABBY_PHASE_LOCK_W": "reward_cam_phase_lock",
            "KRABBY_POWER_W": "penalty_mechanical_power",
            "KRABBY_TRACK_L1_W": "penalty_tracking_error_l1",
            "KRABBY_TRIPOD_W": "reward_tripod_schedule",
            "KRABBY_CLOCK_W": "reward_clock_schedule",
            "KRABBY_APEX_W": "reward_clock_swing_apex",
            # PLAN F gated lineage: teacher elements in the flat stack.
            "KRABBY_GOAL_VEL_W": "reward_tracking_goal_vel",
            "KRABBY_YAW_W": "reward_tracking_yaw",
            "KRABBY_EDGE_W": "reward_feet_edge",
            "KRABBY_STUMBLE_W": "reward_feet_stumble",
            "KRABBY_COLLISION_W": "reward_collision",
            "KRABBY_CLEARANCE_W": "reward_obstacle_clearance",
            "KRABBY_FOOT_CLEAR_W": "reward_foot_clearance",
            # NOTE(gait-formation Phase 0, 2026-08-20): campaign levers, all arms are
            # pure env-var deltas.
            "KRABBY_ANGVEL_W": "reward_ang_vel_xy",
            "KRABBY_ORIENT_W": "reward_orientation",
            "KRABBY_ACTION_RATE_W": "reward_action_rate",
            "KRABBY_DELTA_TORQUE_W": "reward_delta_torques",
            "KRABBY_EXCESS_CONTACT_W": "penalty_excess_feet_contact_forward",
            "KRABBY_STANCE_SUPPORT_W": "reward_stance_support_feet_when_forward",
            "KRABBY_STRIDE_W": "reward_stride_length",
            "KRABBY_AIRTIME_W": "reward_feet_air_time_positive",
            "KRABBY_SWING_MIN_CLEAR_W": "penalty_swing_min_clearance",
        }
        for env_name, term_name in _overrides.items():
            raw = os.environ.get(env_name)
            if raw is not None:
                getattr(self, term_name).weight = float(raw)
        # Tripod band params (gait-formation Phase 0): the crossing-credit band must be
        # retunable per-arm without code edits.
        _tripod_params = {
            "KRABBY_TRIPOD_MIN_PERIOD": "min_period",
            "KRABBY_TRIPOD_MAX_PERIOD": "max_period",
            "KRABBY_TRIPOD_CORR_TAU": "corr_tau",
            "KRABBY_TRIPOD_MIN_AMP": "min_amp",
        }
        for env_name, param_name in _tripod_params.items():
            raw = os.environ.get(env_name)
            if raw is not None:
                self.reward_tripod_schedule.params[param_name] = float(raw)
        _mss = os.environ.get("KRABBY_MAX_SPEED_SCALE")
        if _mss is not None:
            self.reward_forward_progress_along_command.params["max_speed_scale"] = float(_mss)
        _ccm = os.environ.get("KRABBY_CLOCK_COMBINE")
        if _ccm is not None:
            self.reward_clock_schedule.params["combine"] = _ccm
        # NOTE(gait-formation-v2 Phase 4, 2026-08-23): the clearance term's relu floor
        # (min_clearance_m 0.07) pays ZERO below 7 cm while the baked gait lifts 4.4 cm —
        # a dead zone with no gradient between current behavior and the income floor (the
        # same pathology as the crossing term, one level down; G1/G2 measured the plateau).
        # KRABBY_FOOT_CLEAR_MIN lowers the floor so the gradient is dense from the ground up.
        _fcm = os.environ.get("KRABBY_FOOT_CLEAR_MIN")
        if _fcm is not None:
            self.reward_foot_clearance.params["min_clearance_m"] = float(_fcm)
        _apx = os.environ.get("KRABBY_APEX_M")
        if _apx is not None:
            self.reward_clock_swing_apex.params["apex_m"] = float(_apx)
        # KRABBY_FOOT_CLEAR_FLAT=1: lift the parkour_flat mask on the clearance terms so
        # they pay on 100%-flat terrain (they are hard-zeroed there otherwise).
        if os.environ.get("KRABBY_FOOT_CLEAR_FLAT") == "1":
            self.reward_foot_clearance.params["parkour_name"] = None
            self.penalty_swing_min_clearance.params["parkour_name"] = None
        # NOTE(tracking-regression 2026-08-15): sigma^2=0.02 gives the tracking well a
        # ~+-0.25 m/s capture radius; velocity-action exploration lands outside it and the
        # era's dominant reward term contributes zero gradient forever (see onedir-spin
        # RESULTS.md root cause). Override widens the well (legged-gym standard: 0.25).
        _ts2 = os.environ.get("KRABBY_TRACK_SIGMA2")
        if _ts2 is not None:
            import math as _math

            self.track_lin_vel_xy_exp.params["std"] = _math.sqrt(float(_ts2))
        # NOTE(task1-velocity C2): air-time swing threshold, recalibrated for the cam era —
        # the cam return stroke at full speed is 2.14 rad / 6 rad/s ~= 0.36 s; the baked
        # 0.05 s cannot distinguish a cam-timed swing from a micro-tap.
        _at = os.environ.get("KRABBY_AIRTIME_THRESH")
        if _at is not None:
            self.reward_feet_air_time_positive.params["threshold"] = float(_at)


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
