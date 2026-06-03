"""Crab hex student parkour env — not tied to Go2 ``UnitreeGo2StudentParkourEnvCfg``."""

from isaaclab.utils import configclass

from parkour_isaaclab.envs import ParkourManagerBasedRLEnvCfg
from parkour_tasks.crab_hexapod_task.config.crab_hex.agents.parkour_mdp_cfg import (
    CommandsCfg,
    CrabHexStudentActionsCfg,
    CrabHexStudentObservationsCfg,
    CrabHexStudentRewardsCfg,
    CrabHexTerminationsCfg,
    EventCfg,
    ParkourEventsCfg,
)
from parkour_tasks.crab_hexapod_task.config.crab_hex.crab_hex_scene_cfg import (
    CrabHexStudentSceneCfg,
)


@configclass
class CrabHexStudentParkourEnvCfg(ParkourManagerBasedRLEnvCfg):
    """Depth + proprio student MDP for distillation from the 2b2 teacher."""

    scene: CrabHexStudentSceneCfg = CrabHexStudentSceneCfg(num_envs=192, env_spacing=1.0)
    observations: CrabHexStudentObservationsCfg = CrabHexStudentObservationsCfg()
    actions: CrabHexStudentActionsCfg = CrabHexStudentActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: CrabHexStudentRewardsCfg = CrabHexStudentRewardsCfg()
    terminations: CrabHexTerminationsCfg = CrabHexTerminationsCfg()
    parkours: ParkourEventsCfg = ParkourEventsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**18
        self.scene.depth_camera.update_period = self.sim.dt * self.decimation
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        self.scene.contact_forces.update_period = self.sim.dt * self.decimation
        self.actions.joint_pos.use_delay = True
        self.actions.joint_pos.history_length = 8
        # Terrain / commands / DR: ``CrabHexStudentEnvCfg`` → ``_apply_crab_hex_student_2b2_teacher_mdp``.
