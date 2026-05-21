"""Crab Hex (hexapod) task registration."""

import gymnasium as gym

from parkour_tasks.crab_hexapod_task.config.crab_hex import agents as crab_hex_agents
from parkour_tasks.extreme_parkour_task.config.go2 import agents as go2_agents

gym.register(
    id="Isaac-CrabHex-Joystick-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.crab_hex_play_cfg:CrabHexJoystickEnvCfg",
        "rsl_rl_cfg_entry_point": f"{crab_hex_agents.__name__}.rsl_rl_ppo_cfg:CrabHexTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{go2_agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)
