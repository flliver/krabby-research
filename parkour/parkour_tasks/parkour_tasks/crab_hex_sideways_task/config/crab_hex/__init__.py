# SPDX-License-Identifier: BSD-3-Clause
"""Gym registrations for the crab hex sideways-walk tasks."""

# Sideways paradigm phases: KRABBY_SIDE_PHASE / KRABBY_PLANT expand into the KRABBY_* knobs
# BEFORE any config module reads them (the scene cfg reads the USD path, and the env cfg class
# bodies read the reward/command knobs, at import time). This MUST stay the first statement --
# see the knob-timing note in crab_hex_side_env_cfg.py.
from .crab_hex_side_phases import activate_phase as _activate_phase

_activate_phase()

import gymnasium as gym  # noqa: E402

from . import agents  # noqa: E402
from .crab_hex_side_env_cfg import (  # noqa: E402
    CrabHexSideFlatWalkEnvCfg,
    CrabHexSideFlatWalkEnvCfgPLAY,
    CrabHexSideStudentEnvCfg,
    CrabHexSideStudentEnvCfgPLAY,
    CrabHexSideTeacherEnvCfg,
    CrabHexSideTeacherEnvCfgPLAY,
)

gym.register(
    id="Isaac-Crab-Hex-Side-Walk-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CrabHexSideFlatWalkEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CrabHexSideFlatWalkPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Crab-Hex-Side-Walk-Play-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CrabHexSideFlatWalkEnvCfgPLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CrabHexSideFlatWalkPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Crab-Hex-Side-Teacher-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CrabHexSideTeacherEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CrabHexSideTeacherPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Crab-Hex-Side-Teacher-Play-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CrabHexSideTeacherEnvCfgPLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CrabHexSideTeacherPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Crab-Hex-Side-Student-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CrabHexSideStudentEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CrabHexSideStudentPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Crab-Hex-Side-Student-Play-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CrabHexSideStudentEnvCfgPLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CrabHexSideStudentPPORunnerCfg,
    },
)
