import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from parkour_isaaclab.actuators.parkour_actuator_cfg import ParkourDCMotorCfg
from parkour_tasks.crab_hex_forward_task.mdp.crab_hex_cam_mapping import hip_to_cam_shaft_default
from parkour_tasks.crab_hex_forward_task.mdp.crab_hex_linkage import (
    hip_default_rad,
    knee_default_left_rad,
    knee_default_right_rad,
)

# Hardware-natural neutral: both linear actuators at exact mid-stroke (see crab_hex_linkage).
_HIP_DEFAULT_RAD = hip_default_rad()
_KNEE_DEFAULT_LEFT_RAD = knee_default_left_rad()
_KNEE_DEFAULT_RIGHT_RAD = knee_default_right_rad()
from parkour_tasks.crab_hex_forward_task.sensors import ParkourHexContactSensorCfg
from parkour_tasks.default_cfg import CAMERA_CFG
from parkour_tasks.extreme_parkour_task.config.go2.parkour_student_cfg import ParkourStudentSceneCfg
from parkour_tasks.extreme_parkour_task.config.go2.parkour_teacher_cfg import ParkourTeacherSceneCfg


def _crab_usd_path() -> str:
    """USD of the training plant. Override with ``KRABBY_HEX_USD_PATH`` (``KRABBY_PLANT=<name>``
    exports it for the named plant); the default is the repo's MAIN asset ``assets/crab.usda``
    (plant of record A15+B since 2026-09-09; the legacy golden is ``KRABBY_PLANT=legacy_golden`` =
    ``assets/variants/crab_simple__splay00_axis5p5in.usda``)."""
    override = os.environ.get("KRABBY_HEX_USD_PATH")
    if override:
        return override
    # .../krabby-research/parkour/parkour_tasks/parkour_tasks/<task>/config/crab_hex/this_file.py
    repo_root = Path(__file__).resolve().parents[6]
    default = repo_root / "assets" / "crab.usda"
    if default.is_file():
        return str(default)
    return "/workspace/krabby-research/assets/crab.usda"


# NOTE(cam-mechanism-migration): CamShaft's own PD gains. NOT a reuse of the old Body_Hip
# gains (495/9.9/600/738.5) -- those were tuned for a joint moving an entire ~2.8kg leg with
# ~0.1+ kg*m^2 effective inertia. CamShaft is a tiny placeholder body (0.3kg, 2cm cube,
# I ~= 2e-5 kg*m^2 about its own axis) with no leg physically attached (the leg is only
# coupled via the software position-target link, see parkour_actions.py) -- reusing the old
# stiffness against that much smaller inertia gives a natural frequency (~5000 rad/s) far too
# fast for the simulation timestep to resolve, which was observed directly as wild,
# effectively-chaotic drift in CamShaft's settled position across otherwise-identical runs.
# These values target a natural frequency in the same conservative range the proven-stable
# Femur_Tibia joint operates at (~75 rad/s), scaled down for CamShaft's actual inertia --
# placeholder, tune further if lag/overshoot is visible during verification.
# NOTE(cam-velocity-actions): the camshaft is VELOCITY-driven (policy commands signed shaft
# speed, matching the real continuously-rotating quick-return motor). stiffness=0 turns the
# ParkourDCMotor law tau = k_p*(pos_tgt - pos) + k_d*(vel_tgt - vel) into a pure velocity
# servo; damping is the velocity-tracking gain. DISCRETE STABILITY: the explicit actuator
# requires d < 2*I/dt; the placeholder 2cm cube's auto inertia (~2e-5 kg*m^2) made ANY
# useful gain violently unstable (observed as +-5 N*m bang-bang chaos each substep) -- the
# same root cause the pre-velocity comment fought by crippling gains to 1.0/0.05. The USD
# now pins physics:diagonalInertia = 0.015 kg*m^2 on each camshaft (motor rotor reflected
# through the gear reduction is NOT negligible; value is a placeholder pending hardware
# measurement) -> d_max = 2*0.015/0.005 = 6, so d = 1.0 is comfortably stable and reaches
# commanded speed from rest in tens of ms at the effort cap. velocity_limit MUST comfortably
# exceed the max commanded speed (CAM_VEL_SCALE = pi rad/s in parkour_mdp_cfg.py): the
# DC-motor torque-speed curve zeroes available forward torque as joint_vel -> velocity_limit.
# See the hardware-measurements note below for the current cam effort/velocity values.
_CAM_SHAFT_STIFFNESS = {
    "FL_Body_CamShaft_RevoluteJoint": 0.0,
    "FR_Body_CamShaft_RevoluteJoint": 0.0,
    "ML_Body_CamShaft_RevoluteJoint": 0.0,
    "MR_Body_CamShaft_RevoluteJoint": 0.0,
    "RL_Body_CamShaft_RevoluteJoint": 0.0,
    "RR_Body_CamShaft_RevoluteJoint": 0.0,
}
_CAM_SHAFT_DAMPING = {name: 1.0 for name in _CAM_SHAFT_STIFFNESS}
# NOTE(hardware-measurements, 2026-08-20): the yaw drive is a 24 V gearmotor, ~20 N*m and
# ~30 RPM (pi rad/s) through the quick-return crank (motor-sourcing doc, user-confirmed).
# effort/saturation now use the hardware torque; velocity_limit 8 keeps ~61% of saturation
# torque available at the pi rad/s operating speed (torque-speed curve zeroes at the limit).
# CAM_VEL_SCALE in parkour_mdp_cfg.py is pi to match.
_CAM_SHAFT_EFFORT = {name: 20.0 for name in _CAM_SHAFT_STIFFNESS}
_CAM_SHAFT_SATURATION = {name: 24.0 for name in _CAM_SHAFT_STIFFNESS}
_CAM_SHAFT_VELOCITY_LIMIT = 8.0

# NOTE(cam-mechanism-migration): Body_Hip's own actuator -- it tracks position+velocity
# targets computed from the cam-shaft's state each substep
# (CrabHexDelayedJointPositionAction.apply_actions), not a policy action.
# NOTE(cam-velocity-actions): gains raised 495/9.9 -> 2000/40. This actuator emulates a
# RIGID linkage: under continuous shaft spin the quick-return reversal is fast (hip velocity
# peak ~0.92x shaft speed) and the old gains lagged ~0.09 rad, overshooting past the
# mechanism's asin(K)=28.54 deg into the 32 deg hard stop -- which the real linkage can
# never do. Explicit-actuator stability at physics dt 0.005: omega_n = sqrt(2000/I_leg~3)
# ~= 26 rad/s -> omega_n*dt ~= 0.13, comfortably stable.
_HIP_TRACKING_STIFFNESS = {
    "FL_Body_Hip_RevoluteJoint": 2000.0,
    "FR_Body_Hip_RevoluteJoint": 2000.0,
    "ML_Body_Hip_RevoluteJoint": 2000.0,
    "MR_Body_Hip_RevoluteJoint": 2000.0,
    "RL_Body_Hip_RevoluteJoint": 2000.0,
    "RR_Body_Hip_RevoluteJoint": 2000.0,
}
_HIP_TRACKING_DAMPING = {name: 40.0 for name in _HIP_TRACKING_STIFFNESS}
_HIP_TRACKING_EFFORT = {name: 600.0 for name in _HIP_TRACKING_STIFFNESS}
_HIP_TRACKING_SATURATION = {name: 738.5 for name in _HIP_TRACKING_STIFFNESS}

# NOTE(hardware-measurements, 2026-08-20): the hip-pitch and knee joints are driven by
# LEAD-SCREW linear actuators (hip: SLA08 2000 N / 28 mm/s; knee: YH8-523D 500 N / 33 mm/s;
# both 200 mm stroke). Lead screws are NON-BACKDRIVABLE: static holding is free and the
# joint is structurally near-rigid against external loads. Model:
# - joint drive = IMPLICIT PhysX PD at screw-like stiffness (an explicit actuator cannot
#   reach the required stiffness at dt=0.005: with the explicit-stable k=2000 the stance's
#   net pitch stiffness was NEGATIVE -- gravity's m*g*h ~ 2100 N*m/rad destabilizer vs
#   ~400 N*m/rad of leg-PD restoring through the narrow +-0.216 m front/rear leg rows --
#   and the robot tipped over quasi-statically in ~1 s, measured in the settle diagnostics.
#   Implicit drives are unconditionally stable at high k. But FULLY rigid drives
#   (40000/15000, first attempt) degenerate the hyperstatic 6-leg load sharing on rigid
#   flat ground: the robot stood on the FL-RR diagonal with two feet airborne (measured
#   1061/979/0/0 N). k=16000/6000 sits in the window that has BOTH properties: static
#   pitch stiffness ~1.5x gravity's destabilizer AND enough compliance (~17 kN/m vertical
#   per foot) that mm-scale mismatches redistribute only tens of newtons);
# - the real speed/force constraint lives upstream in CrabHexDelayedJointPositionAction:
#   position targets are converted to rod lengths, clamped to the physical 450-650 mm
#   window, and slew-rate-limited at the LOADED rod speed through crab_hex_linkage --
#   joints move at 0.23-0.47 rad/s (hip) / 0.43-2.3 rad/s (knee) no matter what the
#   policy commands, which is the headline sim-to-real correction (old model: 6 rad/s);
# - effort_limit_sim at holding level (~3x the measured stance transients of 354/152 N*m),
#   finite for solver health, far above the drive-phase F*moment-arm (the screw transmits
#   reaction rigidly; drive force limits are expressed by the rod-speed slew, not torque).
_PITCH_JOINT_NAMES = [
    "FL_Hip_Femur_RevoluteJoint",
    "FR_Hip_Femur_RevoluteJoint",
    "ML_Hip_Femur_RevoluteJoint",
    "MR_Hip_Femur_RevoluteJoint",
    "RL_Hip_Femur_RevoluteJoint",
    "RR_Hip_Femur_RevoluteJoint",
]
_KNEE_JOINT_NAMES = [
    "FL_Femur_Tibia_RevoluteJoint",
    "FR_Femur_Tibia_RevoluteJoint",
    "ML_Femur_Tibia_RevoluteJoint",
    "MR_Femur_Tibia_RevoluteJoint",
    "RL_Femur_Tibia_RevoluteJoint",
    "RR_Femur_Tibia_RevoluteJoint",
]
_HIP_FEMUR_STIFFNESS = {name: 16000.0 for name in _PITCH_JOINT_NAMES}
_HIP_FEMUR_DAMPING = {name: 600.0 for name in _PITCH_JOINT_NAMES}
_HIP_FEMUR_EFFORT = {name: 1000.0 for name in _PITCH_JOINT_NAMES}
_HIP_FEMUR_VELOCITY_LIMIT = 2.0  # joint speed never exceeds ~0.5 rad/s (rod-slew-limited)

_FEMUR_TIBIA_STIFFNESS = {name: 6000.0 for name in _KNEE_JOINT_NAMES}
_FEMUR_TIBIA_DAMPING = {name: 200.0 for name in _KNEE_JOINT_NAMES}
_FEMUR_TIBIA_EFFORT = {name: 400.0 for name in _KNEE_JOINT_NAMES}
_FEMUR_TIBIA_VELOCITY_LIMIT = 3.0  # folded-knee rod speed can reach ~2.3 rad/s


def _crab_robot_cfg() -> ArticulationCfg:
    """The crab plant USDA (``assets/crab.usda`` or the selected variant; ``defaultPrim = "krabby"``): reference composes into ``{ENV_REGEX_NS}/Robot`` — leave
    ``articulation_root_prim_path`` unset so Isaac Lab discovers the root on ``Robot``. Base link ``chassis/body``."""
    # Root spawn height so feet sit on terrain without huge drop or penetration.
    # NOTE(vertical-plate correction, 2026-08-20 evening): the mid-stroke default pose
    # sits lower now (hip 1.6 deg vs 6.3): kinematic toe-bottom ~1.056 m below the root,
    # flat-walk terrain surface ~17 mm ABOVE z=0 -> toe clears when spawn > ~1.074.
    # 1.085 gives ~11 mm true clearance (covers the +-5% reset joint randomization).
    # Penetrating spawns cause a violent depenetration transient (measured earlier);
    # re-derive whenever the default pose changes. Measured settled root: 1.064 over z=0.
    spawn_z = float(os.environ.get("KRABBY_HEX_SPAWN_Z", "1.085"))
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_crab_usd_path(),
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
                # NOTE(Phase E, Whitworth mechanism): the pre-Phase-E defaults (+/-0.6, +/-0.25 rad)
                # were sized against the old, unverified +/-50 deg hard limit. The CAD-derived
                # mechanism limit is only +/-28.54 deg (crab_hex_cam_mapping.THETA_HIP_MAX) --
                # +/-0.6 rad (34.4 deg) alone EXCEEDS that and is not physically reachable. Rescaled
                # by THETA_HIP_MAX / radians(50) = 0.5708 to preserve the original design's relative
                # front/rear-vs-middle splay proportions (and the same fractional margin from the
                # limit) under the corrected geometry, rather than picking new values from scratch.
                # NOTE(perpendicular-mounts, 2026-08-13): every leg is mounted perpendicular
                # to the frame with its yaw range centered on that mount (user-confirmed hardware
                # reality). The previous splay defaults (F/R +/-0.342492, M +/-0.142705 -- the
                # Go2-inherited proportions rescaled at the Phase E limit correction) parked the
                # front/rear legs 8.9 deg from the +/-28.54 deg mechanism limit and centered
                # their +/-0.24 rad action window at 82% cam gear, which is why only the mid
                # legs provided forward thrust (see
                # parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-13_1230_perpendicular_mounts/). Zero defaults center
                # every leg's command window on the mount neutral at 100% cam gear with the full
                # symmetric stroke. The standing stance is the policy's choice within the
                # window, not the default's.
                "FR_Body_Hip_RevoluteJoint": 0.0,
                "FL_Body_Hip_RevoluteJoint": 0.0,
                "ML_Body_Hip_RevoluteJoint": 0.0,
                "MR_Body_Hip_RevoluteJoint": 0.0,
                "RR_Body_Hip_RevoluteJoint": 0.0,
                "RL_Body_Hip_RevoluteJoint": 0.0,
                # NOTE(cam-mechanism-migration): principal-value asin inverse of each leg's
                # Body_Hip default above, computed in code (not hand-typed) to stay exactly
                # consistent with crab_hex_cam_mapping.cam_shaft_to_hip (see plan Phase B).
                "FR_Body_CamShaft_RevoluteJoint": hip_to_cam_shaft_default(0.0),
                "FL_Body_CamShaft_RevoluteJoint": hip_to_cam_shaft_default(0.0),
                "ML_Body_CamShaft_RevoluteJoint": hip_to_cam_shaft_default(0.0),
                "MR_Body_CamShaft_RevoluteJoint": hip_to_cam_shaft_default(0.0),
                "RR_Body_CamShaft_RevoluteJoint": hip_to_cam_shaft_default(0.0),
                "RL_Body_CamShaft_RevoluteJoint": hip_to_cam_shaft_default(0.0),
                # NOTE(hardware-measurements, 2026-08-20): defaults = both linear actuators
                # at exact mid-stroke, computed from the linkage geometry (crab_hex_linkage:
                # hip 0.1105 rad / knee L 0.2341 rad). This retires the hand-tuned
                # 0.30 / -0.07 / +0.10 set -- the -0.07/+0.10 knee asymmetry was a
                # compensation for a ~-0.14 deg zero-action roll on the OLD geometry and has
                # no meaning on the new one. Knee sign flip on FR/MR/RR (180-deg Z in USD).
                ".*_Hip_Femur_RevoluteJoint": _HIP_DEFAULT_RAD,
                "FL_Femur_Tibia_RevoluteJoint": _KNEE_DEFAULT_LEFT_RAD,
                "ML_Femur_Tibia_RevoluteJoint": _KNEE_DEFAULT_LEFT_RAD,
                "RL_Femur_Tibia_RevoluteJoint": _KNEE_DEFAULT_LEFT_RAD,
                "FR_Femur_Tibia_RevoluteJoint": _KNEE_DEFAULT_RIGHT_RAD,
                "MR_Femur_Tibia_RevoluteJoint": _KNEE_DEFAULT_RIGHT_RAD,
                "RR_Femur_Tibia_RevoluteJoint": _KNEE_DEFAULT_RIGHT_RAD,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "body_hip_yaw": ParkourDCMotorCfg(
                joint_names_expr=[".*_Body_CamShaft_RevoluteJoint"],
                effort_limit=_CAM_SHAFT_EFFORT,
                saturation_effort=_CAM_SHAFT_SATURATION,
                velocity_limit=_CAM_SHAFT_VELOCITY_LIMIT,
                stiffness=_CAM_SHAFT_STIFFNESS,
                damping=_CAM_SHAFT_DAMPING,
                friction=0.0,
            ),
            # NOTE(cam-mechanism-migration): tracks the position target computed each substep
            # from the cam-shaft's state (see parkour_actions.py); not policy-actuated.
            # NOTE(cam-velocity-actions): velocity_limit raised 6 -> 15. Under continuous shaft
            # spin the quick-return hip velocity peaks at K/(1-K) ~= 0.92x shaft speed
            # (~5.5 rad/s at CAM_VEL_SCALE=6), and the DC-motor torque-speed curve zeroes
            # torque at velocity_limit -- 6.0 left no authority at the swing peak, observed as
            # hip overshoot into the 32-deg hard stop. This "actuator" emulates a rigid
            # linkage, not a motor: it must never be velocity-starved.
            "body_hip_tracking": ParkourDCMotorCfg(
                joint_names_expr=[".*_Body_Hip_RevoluteJoint"],
                effort_limit=_HIP_TRACKING_EFFORT,
                saturation_effort=_HIP_TRACKING_SATURATION,
                velocity_limit=15.0,
                stiffness=_HIP_TRACKING_STIFFNESS,
                damping=_HIP_TRACKING_DAMPING,
                friction=0.0,
            ),
            # Femur–tibia stiffer than hip–femur: knee chain dominates collapse under zero-action / gravity.
            "hip_femur": ImplicitActuatorCfg(
                joint_names_expr=[".*_Hip_Femur_RevoluteJoint"],
                effort_limit_sim=_HIP_FEMUR_EFFORT,
                velocity_limit_sim=_HIP_FEMUR_VELOCITY_LIMIT,
                stiffness=_HIP_FEMUR_STIFFNESS,
                damping=_HIP_FEMUR_DAMPING,
                friction=0.0,
            ),
            "femur_tibia": ImplicitActuatorCfg(
                joint_names_expr=[".*_Femur_Tibia_RevoluteJoint"],
                effort_limit_sim=_FEMUR_TIBIA_EFFORT,
                velocity_limit_sim=_FEMUR_TIBIA_VELOCITY_LIMIT,
                stiffness=_FEMUR_TIBIA_STIFFNESS,
                damping=_FEMUR_TIBIA_DAMPING,
                friction=0.0,
            ),
        },
    )


def _apply_crab_height_scanner(scene) -> None:
    """Re-fit the Go2-sized height scanner to the crab's footprint (override here, never
    edit the shared Go2 file). NOTE(hardware-measurements, 2026-08-20): the measured robot
    stands ~3.1 m toe-to-toe laterally (yaw axis 0.61 + femur pivot 0.064 outboard +
    femur 0.58 + tibia splay ~0.28, per side; vertical-plate hip correction 2026-08-20
    evening) with toes sweeping fore-aft under ±25° yaw. Grid 2.4 × 3.4 m at 0.3 m
    resolution (9×12 rays ≈ the Go2 count), biased 0.6 m ahead along the travel axis."""
    scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/chassis/body"
    scene.height_scanner.offset = type(scene.height_scanner.offset)(pos=(0.6, 0.0, 20.0))
    scene.height_scanner.pattern_cfg.resolution = 0.3
    scene.height_scanner.pattern_cfg.size = [2.4, 3.4]


@configclass
class CrabHexTeacherSceneCfg(ParkourTeacherSceneCfg):
    def __post_init__(self):
        super().__post_init__()
        self.robot = _crab_robot_cfg()
        _apply_crab_height_scanner(self)
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
    # NOTE(hardware-measurements, 2026-08-20): camera moved from the Go2 offset (0.33, 0,
    # 0.08) to the crab body's front face (half-length 0.356 m, upper edge of the 0.318 m
    # tall chassis). Placeholder pending the real camera mount position on the robot.
    depth_camera = CAMERA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot/chassis/body",
        offset=CAMERA_CFG.offset.replace(pos=(0.37, 0.0, 0.10)),
    )

    def __post_init__(self):
        super().__post_init__()
        self.robot = _crab_robot_cfg()
        _apply_crab_height_scanner(self)
        self.contact_forces = ParkourHexContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=2,
            track_air_time=True,
            debug_vis=False,
            force_threshold=1.0,
        )
