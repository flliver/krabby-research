# SPDX-License-Identifier: BSD-3-Clause
"""Reference State Initialization for the crab hexapod (gait-formation-v2 Phase 1).

DeepMimic-style RSI (spin review section 5: "the one cheap mechanism class never tried"):
a fraction of env resets start FROM states sampled along the target gait instead of the
default stand, so schedule/phase-conditioned income is live from iteration 0 and the
policy's problem shifts from "find the gait" (exploration) to "keep it" (retention).

A bank is built offline by ``experiments/2026-08-22_1200_gait_formation_v2/harvest_rsi_bank.py``
from rollouts of a checkpoint on the flat-walk play task, keeping upright walking frames: full
joint state, root height/orientation/velocity, and the gait-clock phase consistent with the
reference's own cam angles. The bank of record, ``rsi_bank_P0_null.npz`` (2051 frames), was
harvested from the PLAN F gated-lineage P0_null rung checkpoint
(``crab_hex_flat_walk/2026-08-26_21-41-53/model_26899.pt``); ``rsi_bank_setAB.npz`` is the
Phase-0 scripted-gait bank (setAB tripod, a dynamically-generated reference, not mocap). The
event stages the clock on the action term (``rsi_clock_staged``); the action term's reset
consumes it (events run BEFORE action-manager reset in Isaac Lab's ``_reset_idx``).

Arm via ``KRABBY_RSI_FRAC`` (fraction of resets seeded, e.g. 0.15).

Plant note (2026-09-09): the banks of record (``rsi_bank_P0_null.npz`` and the ``rsi_bank_pg_r*``
banks) were harvested on the LEGACY golden geometry (splay 0, outer axes 5.5 in) and are reused
unchanged on the A15+B plant of record by design (joint-space + root state; the mount transform
is not baked into joint angles -- the leg-mount morphology campaign found no RSI-shaped
signature). Re-harvest only if a future plant changes joint kinematics.

PLAN H B4 (2026-09-03, ``KRABBY_RSI_SPAWN_FIX=1`` -> ``fix_spawn=True``): the original
placement added only the tile origin to the bank's (0, 0) xy, i.e. RSI resets spawned at
the TILE CENTRE (7 m downrange, on whatever obstacle sits there) with a flat-harvested z.
With ``fix_spawn`` they are placed where ``reset_root_state`` places every other reset
(origin - (size_y + offset, 0)), so the bank z is correct. Unarmed = bit-identical.
Every RSI spawn is reported to the parkour term (``note_spawn``, kind RSI) for the
exposure telemetry's non-RSI / RSI split.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

_BANK_CACHE: dict[str, dict[str, torch.Tensor]] = {}


def _load_bank(path: str, device: str) -> dict[str, torch.Tensor]:
    key = f"{path}@{device}"
    if key not in _BANK_CACHE:
        raw = np.load(path)
        _BANK_CACHE[key] = {
            k: torch.as_tensor(raw[k], dtype=torch.float32, device=device)
            for k in ("joint_pos", "joint_vel", "root_quat_w", "root_z",
                      "root_lin_vel_b", "clock_phase")
        }
    return _BANK_CACHE[key]


def reset_from_reference_states(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    bank_path: str,
    fraction: float = 0.15,
    fix_spawn: bool = False,
) -> None:
    """Seed a Bernoulli subset of the resetting envs from the reference bank."""
    bank = _load_bank(bank_path, str(env.device))
    pick = torch.rand(len(env_ids), device=env.device) < fraction
    ids = env_ids[pick]
    if len(ids) == 0:
        return
    rows = torch.randint(0, bank["joint_pos"].shape[0], (len(ids),), device=env.device)

    robot = env.scene["robot"]
    # Root: keep each env's spawn XY (origin-relative), take z/orientation/velocity from
    # the reference. Bank linear velocity is body-frame; rotate to world.
    root_state = robot.data.default_root_state[ids].clone()
    root_state[:, 2] = bank["root_z"][rows]
    root_state[:, 3:7] = bank["root_quat_w"][rows]
    lin_w = quat_apply(bank["root_quat_w"][rows], bank["root_lin_vel_b"][rows])
    root_state[:, 7:10] = lin_w
    root_state[:, 10:13] = 0.0  # ang vel not recorded in the probe; ~0 in steady walking
    if fix_spawn:
        tg = env.scene.terrain.cfg.terrain_generator
        offset = float(env.event_manager.get_term_cfg("reset_root_state").params.get("offset", 3.0))
        root_state[:, 0] += env.scene.env_origins[ids, 0] - (tg.size[1] + offset)
        root_state[:, 1] += env.scene.env_origins[ids, 1]
    else:
        root_state[:, :2] += env.scene.env_origins[ids, :2]
    robot.write_root_pose_to_sim(root_state[:, :7], env_ids=ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=ids)
    robot.write_joint_state_to_sim(
        bank["joint_pos"][rows], bank["joint_vel"][rows], env_ids=ids
    )
    # Stage the matching gait-clock phase; CrabHexDelayedJointPositionAction.reset consumes
    # it (and randomizes the non-staged envs as usual).
    action_term = env.action_manager.get_term("joint_pos")
    action_term.rsi_clock_staged[ids] = bank["clock_phase"][rows]
    # PLAN H B0: flag these resets as RSI-seeded for the exposure telemetry split.
    from parkour_isaaclab.envs.mdp.events import note_spawn_to_parkour
    from parkour_isaaclab.envs.mdp.parkours.exposure_stats import SPAWN_RSI

    note_spawn_to_parkour(env, ids, root_state[:, 0], SPAWN_RSI)
