"""Gym registrations for crab hex policy training tasks."""

# Paradigm phases (2026-09-07): KRABBY_PHASE / KRABBY_PLANT expand into the KRABBY_* knobs BEFORE
# any config module reads them (the scene cfg reads the USD path at import time).
from .crab_hex_phases import activate_phase as _activate_phase

_activate_phase()

import gymnasium as gym  # noqa: E402

from . import agents  # noqa: E402
from .crab_hex_env_cfg import (
    CrabHexFlatWalkEnvCfg,
    CrabHexFlatWalkEnvCfgPLAY,
    CrabHexStudentEnvCfg,
    CrabHexStudentEnvCfgPLAY,
    CrabHexTeacherEnvCfg,
    CrabHexTeacherEnvCfgPLAY,
)

gym.register(
    id="Isaac-Crab-Hex-Flat-Walk-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CrabHexFlatWalkEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CrabHexFlatWalkPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Crab-Hex-Flat-Walk-Play-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CrabHexFlatWalkEnvCfgPLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CrabHexFlatWalkPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Crab-Hex-Teacher-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CrabHexTeacherEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CrabHexTeacherPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Crab-Hex-Teacher-Play-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CrabHexTeacherEnvCfgPLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CrabHexTeacherPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Crab-Hex-Student-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CrabHexStudentEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CrabHexStudentPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Crab-Hex-Student-Play-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CrabHexStudentEnvCfgPLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CrabHexStudentPPORunnerCfg,
    },
)
