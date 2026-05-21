import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from parkour_isaaclab.actuators.parkour_actuator_cfg import ParkourDCMotorCfg
from parkour_tasks.crab_hexapod_task.sensors import ParkourHexContactSensorCfg
from parkour_tasks.default_cfg import CAMERA_CFG
from parkour_tasks.extreme_parkour_task.config.go2.parkour_student_cfg import ParkourStudentSceneCfg
from parkour_tasks.extreme_parkour_task.config.go2.parkour_teacher_cfg import ParkourTeacherSceneCfg


def _crab_simple_usd_path() -> str:
    """USD for crab_hexapod_task. Override with KRABBY_HEX_USD_PATH; default is repo `crab_simple.usda` only."""
    override = os.environ.get("KRABBY_HEX_USD_PATH")
    if override:
        return override
    # .../krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hexapod_task/config/crab_hex/this_file.py
    repo_root = Path(__file__).resolve().parents[6]
    default = repo_root / "assets" / "crab_simple.usda"
    if default.is_file():
        return str(default)
    return "/workspace/krabby-research/assets/crab_simple.usda"


# All body hips share identical drive (USD joints are symmetric).
# PD for ~104 kg; keep Kd ~0.02×Kp (very large Kd values crash Isaac).
_BODY_HIP_STIFFNESS = {
    "FL_Body_Hip_RevoluteJoint": 495.0,
    "FR_Body_Hip_RevoluteJoint": 495.0,
    "ML_Body_Hip_RevoluteJoint": 495.0,
    "MR_Body_Hip_RevoluteJoint": 495.0,
    "RL_Body_Hip_RevoluteJoint": 495.0,
    "RR_Body_Hip_RevoluteJoint": 495.0,
}
_BODY_HIP_DAMPING = {name: 9.9 for name in _BODY_HIP_STIFFNESS}
_BODY_HIP_EFFORT = {name: 600.0 for name in _BODY_HIP_STIFFNESS}
_BODY_HIP_SATURATION = {name: 738.5 for name in _BODY_HIP_STIFFNESS}

_HIP_FEMUR_STIFFNESS = {name: 675.0 for name in [
    "FL_Hip_Femur_RevoluteJoint",
    "FR_Hip_Femur_RevoluteJoint",
    "ML_Hip_Femur_RevoluteJoint",
    "MR_Hip_Femur_RevoluteJoint",
    "RL_Hip_Femur_RevoluteJoint",
    "RR_Hip_Femur_RevoluteJoint",
]}
_HIP_FEMUR_DAMPING = {name: 14.5 for name in _HIP_FEMUR_STIFFNESS}
_HIP_FEMUR_EFFORT = {name: 1500.0 for name in _HIP_FEMUR_STIFFNESS}
_HIP_FEMUR_SATURATION = {name: 1850.0 for name in _HIP_FEMUR_STIFFNESS}

_FEMUR_TIBIA_STIFFNESS = {name: 912.0 for name in [
    "FL_Femur_Tibia_RevoluteJoint",
    "FR_Femur_Tibia_RevoluteJoint",
    "ML_Femur_Tibia_RevoluteJoint",
    "MR_Femur_Tibia_RevoluteJoint",
    "RL_Femur_Tibia_RevoluteJoint",
    "RR_Femur_Tibia_RevoluteJoint",
]}
_FEMUR_TIBIA_DAMPING = {name: 18.2 for name in _FEMUR_TIBIA_STIFFNESS}
_FEMUR_TIBIA_EFFORT = {name: 600.0 for name in _FEMUR_TIBIA_STIFFNESS}
_FEMUR_TIBIA_SATURATION = {name: 740.0 for name in _FEMUR_TIBIA_STIFFNESS}


def _crab_simple_robot_cfg() -> ArticulationCfg:
    """``crab_simple.usda`` (``defaultPrim = "krabby"``): reference composes into ``{ENV_REGEX_NS}/Robot`` — leave
    ``articulation_root_prim_path`` unset so Isaac Lab discovers the root on ``Robot``. Base link ``chassis/body``."""
    # USD lifts ``krabby`` by +1 m; tune root spawn so feet sit on terrain without huge drop or penetration.
    # Default 1.05 m (override ``KRABBY_HEX_SPAWN_Z``); lower if hover-then-slam, raise if hips scrape or interpenetration.
    spawn_z = float(os.environ.get("KRABBY_HEX_SPAWN_Z", "1.05"))
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_crab_simple_usd_path(),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=20,
                solver_velocity_iteration_count=6,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, spawn_z),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                # Body–hip yaw (Z): front/rear splay; L/R mirrored (right-side signs verified in top view).
                "FR_Body_Hip_RevoluteJoint": 0.6,
                "FL_Body_Hip_RevoluteJoint": -0.6,
                "ML_Body_Hip_RevoluteJoint": 0.0,
                "MR_Body_Hip_RevoluteJoint": 0.0,
                "RR_Body_Hip_RevoluteJoint": -0.6,
                "RL_Body_Hip_RevoluteJoint": 0.6,
                # Hip–femur: same on all legs. Knee: sign flip on FR/MR/RR (180° Z in USD); left −0.07
                # vs right +0.10 balances zero-action roll (~−0.14°) with splay unchanged.
                ".*_Hip_Femur_RevoluteJoint": 0.30,
                "FL_Femur_Tibia_RevoluteJoint": -0.07,
                "ML_Femur_Tibia_RevoluteJoint": -0.07,
                "RL_Femur_Tibia_RevoluteJoint": -0.07,
                "FR_Femur_Tibia_RevoluteJoint": 0.10,
                "MR_Femur_Tibia_RevoluteJoint": 0.10,
                "RR_Femur_Tibia_RevoluteJoint": 0.10,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "body_hip_yaw": ParkourDCMotorCfg(
                joint_names_expr=[".*_Body_Hip_RevoluteJoint"],
                effort_limit=_BODY_HIP_EFFORT,
                saturation_effort=_BODY_HIP_SATURATION,
                velocity_limit=6.0,
                stiffness=_BODY_HIP_STIFFNESS,
                damping=_BODY_HIP_DAMPING,
                friction=0.0,
            ),
            # Femur–tibia stiffer than hip–femur: knee chain dominates collapse under zero-action / gravity.
            "hip_femur": ParkourDCMotorCfg(
                joint_names_expr=[".*_Hip_Femur_RevoluteJoint"],
                effort_limit=_HIP_FEMUR_EFFORT,
                saturation_effort=_HIP_FEMUR_SATURATION,
                velocity_limit=6.0,
                stiffness=_HIP_FEMUR_STIFFNESS,
                damping=_HIP_FEMUR_DAMPING,
                friction=0.0,
            ),
            "femur_tibia": ParkourDCMotorCfg(
                joint_names_expr=[".*_Femur_Tibia_RevoluteJoint"],
                effort_limit=_FEMUR_TIBIA_EFFORT,
                saturation_effort=_FEMUR_TIBIA_SATURATION,
                velocity_limit=6.0,
                stiffness=_FEMUR_TIBIA_STIFFNESS,
                damping=_FEMUR_TIBIA_DAMPING,
                friction=0.0,
            ),
        },
    )


@configclass
class CrabHexTeacherSceneCfg(ParkourTeacherSceneCfg):
    def __post_init__(self):
        super().__post_init__()
        self.robot = _crab_simple_robot_cfg()
        self.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/chassis/body"
        # Aggregate chassis + all leg links (``ParkourHexContactSensor``): default nested ``Robot/krabby/.*/.*``
        # only reports ``chassis/body``; Isaac composes ``krabby`` children flat under ``Robot`` at runtime.
        self.contact_forces = ParkourHexContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=2,
            track_air_time=True,
            debug_vis=False,
            force_threshold=1.0,
        )


@configclass
class CrabHexStudentSceneCfg(ParkourStudentSceneCfg):
    depth_camera = CAMERA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot/chassis/body")

    def __post_init__(self):
        super().__post_init__()
        self.robot = _crab_simple_robot_cfg()
        self.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/chassis/body"
        self.contact_forces = ParkourHexContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=2,
            track_air_time=True,
            debug_vis=False,
            force_threshold=1.0,
        )
