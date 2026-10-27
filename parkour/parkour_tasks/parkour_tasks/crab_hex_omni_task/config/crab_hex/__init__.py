# SPDX-License-Identifier: BSD-3-Clause
"""Gym registrations for the crab hex omnidirectional tasks."""

# Omni paradigm phases: KRABBY_OMNI_PHASE / KRABBY_PLANT expand into the KRABBY_* knobs
# BEFORE any config module reads them (the scene cfg reads the USD path, and the env cfg class
# bodies read the reward/command knobs, at import time). This MUST stay the first statement --
# see the knob-timing note in crab_hex_omni_env_cfg.py. It is also the guard that refuses a
# process carrying two preset variables (import order is forward -> omni -> sideways).
from .crab_hex_omni_phases import activate_phase as _activate_phase

_activate_phase()

import gymnasium as gym  # noqa: E402

from . import agents  # noqa: E402
from .crab_hex_omni_env_cfg import (  # noqa: E402
    CrabHexOmniFlatWalkEnvCfg,
    CrabHexOmniFlatWalkEnvCfgPLAY,
    CrabHexOmniStudentEnvCfg,
    CrabHexOmniStudentEnvCfgPLAY,
    CrabHexOmniTeacherEnvCfg,
    CrabHexOmniTeacherEnvCfgPLAY,
)

gym.register(
    id="Isaac-Crab-Hex-Omni-Walk-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CrabHexOmniFlatWalkEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CrabHexOmniFlatWalkPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Crab-Hex-Omni-Walk-Play-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CrabHexOmniFlatWalkEnvCfgPLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CrabHexOmniFlatWalkPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Crab-Hex-Omni-Teacher-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CrabHexOmniTeacherEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CrabHexOmniTeacherPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Crab-Hex-Omni-Teacher-Play-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CrabHexOmniTeacherEnvCfgPLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CrabHexOmniTeacherPPORunnerCfg,
    },
)

# Task 4 pipeline step 4 (M1b, 2026-10-16): the omni depth student (4 cameras, DEFAULT OFF -- nothing else
# reaches these classes). Train: ``KRABBY_OMNI_PHASE=2a`` + T*'s overlay minus the runner-only keys,
# ``--resume --checkpoint <T*>``. Play/eval: the Walk-Play MDP + the cameras (``--policy-role teacher`` pairs T*).
gym.register(
    id="Isaac-Crab-Hex-Omni-Student-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CrabHexOmniStudentEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CrabHexOmniStudentPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Crab-Hex-Omni-Student-Play-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CrabHexOmniStudentEnvCfgPLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CrabHexOmniStudentPPORunnerCfg,
    },
)
