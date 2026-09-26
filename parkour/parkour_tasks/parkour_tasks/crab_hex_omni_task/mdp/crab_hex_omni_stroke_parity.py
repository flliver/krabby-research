# SPDX-License-Identifier: BSD-3-Clause
"""Wrapper -> replay parity recorder for the stroke/posture plan (step 12(ii), 2026-09-26). Pure
torch/numpy; never imports ``isaaclab``, never touches the GPU itself, never mutates a tensor it is
handed.

WHAT THIS IS. ``RewardFootStroke`` and ``RewardBodyCentring``
(``crab_hex_omni_stroke_rewards.py``) each call one ``record_*`` method here with the exact tensors
they pass to ``crab_hex_omni_stroke_math`` and the reward each returns. When
``KRABBY_OMNI_STROKE_PARITY_DUMP`` names a path, this module accumulates the first ``K`` envs of the
first ``M`` env steps from BOTH wrappers into one ``.npz`` in
``experiments/tools/replay_stroke_posture.py``'s ``parity`` dump format (its module docstring, "PARITY
DUMP FORMAT"), then saves it once both terms have a full row for every step, or at process exit if
fewer. **Unset or blank, every method below is a single ``if`` and returns -- zero cost, no state, no
import-time allocation.**

WHY ONE SHARED SINGLETON. The two wrappers are separate ``ManagerTermBase`` instances with no
reference to each other, but the parity dump is keyed by ENV STEP (``env.common_step_counter``, the
same integer ``_OncePerStep`` in ``crab_hex_omni_stroke_rewards.py`` uses to catch a double call) --
a row is only complete once both terms have written it. :func:`get_recorder` returns the one instance
for the process, built from the environment on its first call (i.e. at the first wrapper's
construction), so whichever knob values were exported before Isaac's launch are what is read; nothing
here re-reads the environment later.

WHY ``dt`` / ``turn_radius`` and not ``reset``-per-call. ``dt`` and ``turn_radius`` are process-wide
constants, recorded whenever a row is written. Env resets arrive out of band, through
``ManagerTermBase.reset(env_ids)`` -- :func:`ParityRecorder.note_reset` stages them, and the first
NEW step recorded after a reset carries it in the dump's ``reset`` column, matching the parity dump
format's own note ("the env was reset before this call").

A NOTE FOR THE WINDOW-0 SMOKE. ``KRABBY_OMNI_STROKE_PARITY_*`` are read by this module only -- they
are not ``omni_knobs``/``stroke_knobs`` cfg knobs, and ``run_candidates.py``'s knob whitelist never
sees them (that validator only checks ``candidates.yaml`` overlay keys, and the smoke runs by hand
through ``run_phases.train``, outside the ladder). But ``run_phases._subenv`` builds the training
child's environment by COPYING ``os.environ`` and then deleting every ``KRABBY_*`` key before
re-applying only the caller's explicit ``env`` / ``extra_env`` dict (``run_phases.py``, ``_subenv``
and ``train``). Exporting ``KRABBY_OMNI_STROKE_PARITY_DUMP`` in the parent shell is therefore silently
STRIPPED before the training subprocess starts -- not refused, just dropped, so nothing records and
no error says why. The smoke must instead pass the three ``KRABBY_OMNI_STROKE_PARITY_*`` variables
in the ``extra_env`` dict given to ``run_phases.train`` (they are not in ``TRAIN_ENV_RESERVED``, so
that is accepted).
"""

from __future__ import annotations

import atexit
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

#: Env vars this module reads (once, at the singleton's construction -- see :func:`get_recorder`).
DUMP_PATH_VAR = "KRABBY_OMNI_STROKE_PARITY_DUMP"
ENVS_VAR = "KRABBY_OMNI_STROKE_PARITY_ENVS"
STEPS_VAR = "KRABBY_OMNI_STROKE_PARITY_STEPS"
DEFAULT_ENVS = 16
DEFAULT_STEPS = 300

#: dtype for each dump column, matching the tool's ``parity`` dump-format table.
_DTYPES: dict[str, np.dtype] = {
    "force": np.float32, "foot_pos_w": np.float32, "root_pos_w": np.float32, "heading": np.float32,
    "cmd": np.float32, "stand": np.bool_, "root_lin_vel_b_xy": np.float32, "root_ang_vel_b_z": np.float32,
    "turning": np.bool_, "lat_share": np.float32, "resampled": np.bool_, "reward": np.float32,
    "b": np.float32, "reset": np.bool_, "pay": np.float32, "o": np.float32,
}


def _to_numpy(value) -> np.ndarray:
    """``.detach().cpu().numpy()`` for a tensor; ``np.asarray`` for anything else. Never mutates."""
    if isinstance(value, torch.Tensor):
        value = value.detach().to("cpu")
        return value.numpy()
    return np.asarray(value)


class ParityRecorder:
    """Accumulates the parity dump for one process. See the module docstring. Construct via
    :func:`get_recorder`, not directly, except in tests that want an isolated instance."""

    def __init__(self, dump_path: str | None, num_envs: int, num_steps: int):
        self.enabled = bool(dump_path and dump_path.strip())
        self.path = Path(dump_path.strip()) if self.enabled else None
        self.k = max(1, int(num_envs))
        self.m = max(1, int(num_steps))
        self._store: dict[str, np.ndarray] = {}
        self._scalars: dict[str, float] = {}
        self._step_slot: dict[int, int] = {}
        self._n_slots = 0
        self._pending_reset: set[int] = set()
        self._saved = False
        if self.enabled:
            self._stroke_written = np.zeros(self.m, dtype=bool)
            self._centring_written = np.zeros(self.m, dtype=bool)
            atexit.register(self.close)

    @classmethod
    def from_env(cls) -> "ParityRecorder":
        def _int(var: str, default: int) -> int:
            raw = os.environ.get(var, "").strip()
            return int(raw) if raw else default

        dump = os.environ.get(DUMP_PATH_VAR)
        if not (dump and dump.strip()):
            # Unset/blank: a complete no-op -- never parse the size vars, so a stray value in them
            # can never raise inside a reward term's constructor (i.e. crash a training run).
            return cls(None, DEFAULT_ENVS, DEFAULT_STEPS)
        return cls(dump, _int(ENVS_VAR, DEFAULT_ENVS), _int(STEPS_VAR, DEFAULT_STEPS))

    # ---------------------------------------------------------------------------------- recording
    def note_reset(self, env_ids) -> None:
        """Stage a reset (from ``ManagerTermBase.reset``); applied to the next NEW step recorded."""
        if not self.enabled or self._saved:
            return
        if env_ids is None:
            ids: Iterable[int] = range(self.k)
        elif isinstance(env_ids, torch.Tensor):
            ids = env_ids.detach().to("cpu").tolist()
        elif isinstance(env_ids, slice):
            ids = range(*env_ids.indices(self.k))
        else:
            ids = list(env_ids)
        self._pending_reset.update(int(i) for i in ids if int(i) < self.k)

    def record_stroke(
        self, step: int, *, force, foot_pos_w, root_pos_w, heading, cmd, stand, root_lin_vel_b_xy,
        root_ang_vel_b_z, turning, resampled, reward, pay, dt: float, turn_radius: float,
    ) -> None:
        """One env step of ``RewardFootStroke``'s own inputs and output. No-op unless recording."""
        slot = self._slot_for(step)
        if slot is None:
            return
        for name, value in (
            ("force", force), ("foot_pos_w", foot_pos_w), ("root_pos_w", root_pos_w), ("heading", heading),
            ("cmd", cmd), ("stand", stand), ("root_lin_vel_b_xy", root_lin_vel_b_xy),
            ("root_ang_vel_b_z", root_ang_vel_b_z), ("turning", turning), ("resampled", resampled),
            ("reward", reward), ("pay", pay),
        ):
            self._put(name, slot, value)
        self._scalars["dt"] = float(dt)
        self._scalars["turn_radius"] = float(turn_radius)
        self._stroke_written[slot] = True
        self._maybe_autosave()

    def record_centring(
        self, step: int, *, foot_pos_w, root_pos_w, heading, cmd, stand, lat_share, resampled, b, o, dt: float,
    ) -> None:
        """One env step of ``RewardBodyCentring``'s own inputs and output. No-op unless recording."""
        slot = self._slot_for(step)
        if slot is None:
            return
        for name, value in (
            ("foot_pos_w", foot_pos_w), ("root_pos_w", root_pos_w), ("heading", heading), ("cmd", cmd),
            ("stand", stand), ("lat_share", lat_share), ("resampled", resampled), ("b", b), ("o", o),
        ):
            self._put(name, slot, value)
        self._scalars["dt"] = float(dt)
        self._centring_written[slot] = True
        self._maybe_autosave()

    # ---------------------------------------------------------------------------------- internals
    def _slot_for(self, step: int) -> int | None:
        if not self.enabled or self._saved:
            return None
        slot = self._step_slot.get(step)
        if slot is not None:
            return slot
        if self._n_slots >= self.m:
            return None
        slot = self._n_slots
        self._step_slot[step] = slot
        self._n_slots += 1
        if self._pending_reset:
            arr = self._store.get("reset")
            if arr is None:
                arr = np.zeros((self.m, self.k), dtype=np.bool_)
                self._store["reset"] = arr
            for env_id in self._pending_reset:
                arr[slot, env_id] = True
            self._pending_reset.clear()
        return slot

    def _put(self, name: str, slot: int, value) -> None:
        v = _to_numpy(value)[: self.k]
        dtype = _DTYPES.get(name, np.float32)
        arr = self._store.get(name)
        if arr is None:
            arr = np.zeros((self.m, self.k) + v.shape[1:], dtype=dtype)
            self._store[name] = arr
        arr[slot, : v.shape[0]] = v.astype(dtype, copy=False)

    def _maybe_autosave(self) -> None:
        if self._n_slots >= self.m and self._stroke_written.all() and self._centring_written.all():
            self._finalize()

    def _finalize(self) -> None:
        if not self.enabled or self._saved:
            return
        self._saved = True  # set first: an exception below must not be retried by a later call
        if self._n_slots == 0:
            return
        n = self._n_slots
        data = {name: arr[:n] for name, arr in self._store.items()}
        data.update({name: np.asarray(value, dtype=np.float64) for name, value in self._scalars.items()})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.path, **data)

    def close(self) -> None:
        """Save now, with whatever has been recorded so far. Idempotent; the ``atexit`` hook for a
        run that stops before ``M`` steps complete both terms."""
        self._finalize()


_SINGLETON: ParityRecorder | None = None


def get_recorder() -> ParityRecorder:
    """The one recorder for this process, built from the environment on first call."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = ParityRecorder.from_env()
    return _SINGLETON


def _reset_singleton_for_tests() -> None:
    """Test-only: forget the cached singleton so the next :func:`get_recorder` re-reads the env."""
    global _SINGLETON
    _SINGLETON = None
