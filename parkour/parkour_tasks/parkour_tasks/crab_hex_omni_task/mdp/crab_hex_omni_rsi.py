# SPDX-License-Identifier: BSD-3-Clause
"""Reference State Initialization for the omni task (plan D6, spec A7).

The forward RSI event (``crab_hex_forward_task/mdp/crab_hex_rsi.py``) drops a fraction of
resetting envs into a recorded walking frame: bank joint state, bank root height and
orientation, bank body velocity rotated to world, and the bank's gait-clock phase staged on
the action term. That works because the forward task has ONE travel direction (body ``+x``,
course ``+x``) and one facing (square to the course): a frame replayed at the spawn is
automatically on-command.

Two things change for the omni task, and both are the difference between "the reset seeds
the gait" and "the reset throws the robot at a command it is not moving towards":

1. **The spawn yaw is random.** The frame is re-yawed to a sampled ``psi`` while its ROLL
   and PITCH are kept (the recorded body attitude is part of the gait cycle -- flattening it
   would make the first step a recovery step):

       ``q_new = q_z(psi - yaw(q_bank)) * q_bank``

   Pre-multiplying by a WORLD-frame yaw rotation leaves the body's tilt relative to gravity
   untouched and sets the yaw to exactly ``psi``. The bank's body-frame velocity is then
   rotated by the NEW attitude (``v_w = q_new * v_b``), so the frame keeps its momentum in
   the body sense and merely points somewhere else in the world.

2. **The command is staged.** Reset events run BEFORE the command manager's resample
   (``_reset_idx``: parkour manager -> events -> ... -> command manager), so the event
   writes ``(theta, v, psi)`` into ``CrabHexOmniCommand.rsi_cmd_staged`` and the very first
   command of the episode continues the frame's own momentum with zero facing error:
   ``theta = atan2(v_w_y, v_w_x)``, ``v = clamp(||v_b_xy||, walk_floor, ellipse_cap(phi_b))``,
   facing ``= psi``. Without this the sampler would draw an unrelated direction and the
   seeded frame would be a disturbance rather than a demonstration.

THE BANK. The six forward arrays (``joint_pos``, ``joint_vel``, ``root_quat_w``, ``root_z``,
``root_lin_vel_b``, ``clock_phase``) plus ``source`` (int8 provenance tag: 0 = forward-bank
frames, 1 = sideways-bank frames) and an optional ``meta`` JSON string. A bank WITHOUT
``source`` -- i.e. any existing forward bank -- is accepted with a warning and treated as
all-source-0, so phase 1a can run against the forward bank of record before the merged omni
bank exists.

No Isaac imports: the quaternion helpers come from :mod:`omni_events` (transcriptions of
``isaaclab.utils.math``) and the ellipse from :mod:`crab_hex_omni_command_math`, so this
module is exercised on CPU by ``tests/unit/test_crab_hex_omni_rsi.py``.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from .crab_hex_omni_command_math import ellipse_cap
from .omni_events import note_spawn_xy_to_parkour, quat_apply, quat_from_yaw, quat_mul, yaw_from_quat
from .omni_exposure import SPAWN_RSI

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = ["reset_from_reference_states_omni", "BANK_ARRAYS", "SOURCE_FORWARD", "SOURCE_SIDEWAYS"]

BANK_ARRAYS = ("joint_pos", "joint_vel", "root_quat_w", "root_z", "root_lin_vel_b", "clock_phase")
"""The six arrays every bank carries (the forward harvest format, unchanged)."""

SOURCE_FORWARD = 0
SOURCE_SIDEWAYS = 1

_BANK_CACHE: dict[str, dict[str, torch.Tensor]] = {}
"""Per ``path@device`` bank cache -- the forward event's pattern: a bank is a few megabytes
and is re-read on every reset otherwise."""


def _load_bank(path: str, device: str) -> dict[str, torch.Tensor]:
    """Load (and cache) a bank; a bank without ``source`` becomes all-zeros with a warning."""
    key = f"{path}@{device}"
    if key not in _BANK_CACHE:
        raw = np.load(path)
        bank = {
            k: torch.as_tensor(raw[k], dtype=torch.float32, device=device) for k in BANK_ARRAYS
        }
        n = int(bank["joint_pos"].shape[0])
        if "source" in getattr(raw, "files", []):
            src = torch.as_tensor(np.asarray(raw["source"]).reshape(-1), dtype=torch.long, device=device)
            if int(src.numel()) != n:
                raise ValueError(
                    f"RSI bank '{path}': 'source' has {int(src.numel())} entries for {n} frames"
                )
        else:
            print(
                f"[rsi-omni] bank '{path}' has no 'source' array; treating all {n} frames as "
                f"source {SOURCE_FORWARD} (the forward-bank convention).",
                flush=True,
            )
            src = torch.zeros(n, dtype=torch.long, device=device)
        bank["source"] = src
        _BANK_CACHE[key] = bank
    return _BANK_CACHE[key]


def _source_rows(bank: dict[str, torch.Tensor], source) -> torch.Tensor:
    """Row indices whose provenance tag matches ``source`` (``0`` / ``1`` / ``"all"``).

    Accepts an int or a string (the knob ``KRABBY_OMNI_RSI_SOURCE`` arrives as text), so
    ``"0"``, ``0``, ``"1"``, ``1`` and ``"all"`` all work. An empty selection is an error,
    not a silent no-op: it would mean a stage is running with RSI armed and nothing to draw
    from, which looks exactly like RSI being off.
    """
    if isinstance(source, str):
        key = source.strip().lower()
        if key in ("all", "", "any", "*"):
            return torch.arange(bank["source"].numel(), device=bank["source"].device)
        try:
            tag = int(key)
        except ValueError as exc:
            raise ValueError(f"RSI source must be 0, 1 or 'all', got {source!r}") from exc
    elif source is None:
        return torch.arange(bank["source"].numel(), device=bank["source"].device)
    else:
        tag = int(source)
    rows = torch.nonzero(bank["source"] == tag, as_tuple=False).reshape(-1)
    if rows.numel() == 0:
        raise ValueError(
            f"RSI bank has no frames with source tag {tag} "
            f"(tags present: {sorted(set(bank['source'].tolist()))})"
        )
    return rows


def reset_from_reference_states_omni(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    bank_path: str,
    fraction: float = 0.2,
    yaw_range: tuple[float, float] = (-math.pi, math.pi),
    walk_floor: float = 0.21,
    source="all",
) -> None:
    """Seed a Bernoulli(``fraction``) subset of the resetting envs from the omni bank.

    Args:
        env: the environment.
        env_ids: envs being reset.
        bank_path: ``.npz`` with :data:`BANK_ARRAYS` (+ optional ``source`` / ``meta``).
        fraction: share of ``env_ids`` to seed (0 -> exact no-op, nothing is written and no
            random numbers beyond the Bernoulli draw are consumed).
        yaw_range: ``(lo, hi)`` uniform band for the spawn yaw ``psi``, radians.
        walk_floor: lowest speed the staged command may carry, m/s. Must equal the command
            term's walk floor (``clips.lin_vel_clip + WALK_MARGIN_M_S``, 0.21 with the stock
            clip): below it the gait clock stalls and the "seeded frame" would be told to
            stand still.
        source: ``0``, ``1`` or ``"all"`` -- which provenance tag to draw frames from.
    """
    bank = _load_bank(bank_path, str(env.device))
    pick = torch.rand(len(env_ids), device=env.device) < fraction
    ids = env_ids[pick]
    if len(ids) == 0:
        return
    pool = _source_rows(bank, source)
    rows = pool[torch.randint(0, int(pool.numel()), (len(ids),), device=env.device)]

    robot = env.scene["robot"]
    root_state = robot.data.default_root_state[ids].clone()

    # -- attitude: keep the frame's roll/pitch, set the yaw to psi exactly.
    q_bank = bank["root_quat_w"][rows]
    lo, hi = float(yaw_range[0]), float(yaw_range[1])
    if not lo <= hi:
        raise ValueError(f"yaw_range must satisfy lo <= hi, got ({lo}, {hi})")
    psi = torch.rand(len(ids), device=env.device) * (hi - lo) + lo
    q_new = quat_mul(quat_from_yaw(psi - yaw_from_quat(q_bank)), q_bank)

    # -- position: tile centre (the ring spawn), bank height. The harvest stored
    #    root_z relative to the tile origin, and the clear disc is the tile's flat middle,
    #    so no terrain-height correction is applied -- the forward event's convention.
    root_state[:, :2] += env.scene.env_origins[ids, :2]
    root_state[:, 2] = bank["root_z"][rows]
    root_state[:, 3:7] = q_new

    # -- velocity: the bank's BODY velocity expressed in the new attitude. Angular velocity
    #    is not recorded by the harvest (~0 in steady walking) and is zeroed.
    v_b = bank["root_lin_vel_b"][rows]
    v_w = quat_apply(q_new, v_b)
    root_state[:, 7:10] = v_w
    root_state[:, 10:13] = 0.0

    robot.write_root_pose_to_sim(root_state[:, :7], env_ids=ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=ids)
    robot.write_joint_state_to_sim(bank["joint_pos"][rows], bank["joint_vel"][rows], env_ids=ids)

    # -- gait clock: staged on the action term, consumed by its own reset (events run
    #    before the action manager's reset), exactly as the forward event does.
    action_term = env.action_manager.get_term("joint_pos")
    action_term.rsi_clock_staged[ids] = bank["clock_phase"][rows]

    # -- command: stage the frame's own momentum so the first resample continues it.
    cmd_term = env.command_manager.get_term("base_velocity")
    a, b, c = (float(x) for x in cmd_term.cfg.speed_ellipse)
    phi_b = torch.atan2(v_b[:, 1], v_b[:, 0])
    cap = ellipse_cap(phi_b, a, b, c)
    speed = torch.minimum(torch.clamp(torch.norm(v_b[:, :2], dim=1), min=float(walk_floor)), cap)
    theta_w = torch.atan2(v_w[:, 1], v_w[:, 0])
    cmd_term.rsi_cmd_staged[ids] = torch.stack((theta_w, speed, psi), dim=1)

    # -- exposure telemetry: RSI spawns are a separate group in the ledger.
    note_spawn_xy_to_parkour(env, ids, root_state[:, :2], SPAWN_RSI)
