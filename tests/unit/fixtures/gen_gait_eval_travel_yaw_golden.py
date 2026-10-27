# SPDX-License-Identifier: BSD-3-Clause
"""Deterministic golden-fixture generator for the gait_eval travel-frame threading (Verification 0
of the omni-task plan, step 1 "Baseline captures on the clean base commit").

Pure numpy + the existing ``gait_eval`` harness modules (``gait_eval.report``, ``.schedule``,
``.metrics``) -- no Isaac Sim import anywhere in this file or in anything it imports, so it runs in
plain ``python`` / ``pytest`` with no GPU.

Builds a small synthetic multi-episode ``raw`` dict in the exact shape
``gait_eval.report.score_run`` consumes (the array names/shapes below mirror
``tests/unit/test_crab_hex_gait_metrics.py::test_report_default_path_identical``, the existing
precedent for this shape), and scores it once at ``travel_yaw=0.0`` (the forward task's convention)
and once at ``travel_yaw=math.pi / 2`` (the sideways task's convention -- see
``gait_eval.report.score_episode``'s docstring: "0.0 for the forward task, +pi/2 for the sideways
task"). ``travel_yaw`` is the argument that carries the travel frame today.

The two resulting episode-metric lists are written verbatim (numpy converted via
``gait_eval.metrics.json_safe``; ``json.dumps`` already emits the shortest round-tripping repr for
Python floats, i.e. full precision) to ``gait_eval_travel_yaw_golden.json`` alongside the generator
parameters, so a later test can assert new/refactored code reproduces this JSON exactly.

Importable surface for that later test:
    ``build_raw(seed: int = SEED) -> dict[str, np.ndarray]``
    ``score_both(raw: dict[str, np.ndarray]) -> dict[str, list[dict]]``  (keys ``"yaw_0"``, ``"yaw_half_pi"``)

Run directly to (re)write the golden JSON:
    ``python tests/unit/fixtures/gen_gait_eval_travel_yaw_golden.py``

Regeneration history. Written 2026-09-16 on the clean base commit. Regenerated ONCE, 2026-10-26, by the
user's decision "Make scoring BLAS-free" (Task 4 CHARTER): the first write came from an AVX-512 machine, whose
OpenBLAS ``ddot`` (behind ``np.linalg.norm``) rounds differently from the kernel GitHub's non-AVX-512 runners
load, and ``gait_eval.metrics`` now does its vector reductions without BLAS. This generator was not changed;
the new JSON is byte-identical under every OpenBLAS kernel that runs on the dev machine and on numpy 1.26 and
2.5. Rewriting the fixture to make ``test_crab_hex_gait_eval_golden.py`` pass is otherwise forbidden (see that
module's docstring).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = (
    _REPO_ROOT
    / "parkour"
    / "parkour_tasks"
    / "parkour_tasks"
    / "crab_hex_forward_task"
    / "scripts"
)
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from gait_eval import metrics as M  # noqa: E402
from gait_eval import report as R  # noqa: E402
from gait_eval.schedule import Hold, Scenario, compile_schedule  # noqa: E402

# --- generator parameters (fixed; only the noise below is seed-dependent) -----------------------
SEED = 20260916
DT = 0.02
N_ENVS = 3
HALF_PERIOD = 5  # steps per tripod half-cycle (matches the precedent test's _tripod_contact)
FALL_ENV = 2
FALL_FRAC = 0.75  # crab_failure goes true at this fraction of the schedule for FALL_ENV
FALL_GRACE_STEPS = 20  # steps the "episode" keeps running past the fall onset before term
TRAVEL_YAWS = {"yaw_0": 0.0, "yaw_half_pi": math.pi / 2}

FIXTURE_PATH = Path(__file__).resolve().parent / "gait_eval_travel_yaw_golden.json"


def _schedule() -> list[Hold]:
    # Not seed-dependent: two holds, one pure-translation and one pure-turn, so both the linear
    # and angular tracking paths and both travel-frame rotations get exercised.
    return [
        Hold(hold_s=3.0, vx=0.3, vy=0.0, wz=0.0, label="fwd"),
        Hold(hold_s=2.0, vx=0.0, vy=0.0, wz=0.5, label="turn"),
    ]


def _scenario() -> Scenario:
    return Scenario(
        id="golden_travel_yaw",
        task="Isaac-Crab-Hex-Flat-Walk-v0",
        checkpoint="",
        schedule=_schedule(),
        episodes=N_ENVS,
        env_seed=1,
    )


def _tripod_contact(n_steps: int, half_period: int) -> np.ndarray:
    """Perfectly alternating tripod contact, (n_steps, 6) bool -- ported from the precedent test."""
    contact = np.zeros((n_steps, 6), dtype=bool)
    for t in range(n_steps):
        a_down = (t // half_period) % 2 == 0
        for i in M.TRIPOD_A_IDX:
            contact[t, i] = a_down
        for i in M.TRIPOD_B_IDX:
            contact[t, i] = not a_down
    return contact


def _term_reason_and_n_steps(total_steps: int) -> tuple[list[str], list[int]]:
    term_reason = ["schedule_complete"] * N_ENVS
    n_steps_env = [total_steps] * N_ENVS
    fall_start = int(total_steps * FALL_FRAC)
    term_reason[FALL_ENV] = "fall"
    n_steps_env[FALL_ENV] = min(total_steps, fall_start + FALL_GRACE_STEPS)
    return term_reason, n_steps_env


def build_raw(seed: int = SEED) -> dict[str, np.ndarray]:
    """A small synthetic multi-episode ``raw`` dict, in the exact shape ``score_run`` consumes.

    env 0: clean alternating tripod exactly tracking the commanded velocity, schedule-complete.
    env 1: the same gait plus small seeded noise on force/velocity/foot position, schedule-complete.
    env 2: undertracking (half the commanded velocity) and a fall three-quarters through, to
           exercise the fall-trim / fall-direction / support-polygon code paths.
    """
    rng = np.random.default_rng(seed)
    compiled = compile_schedule(_scenario(), dt=DT, num_envs=N_ENVS)
    T = compiled.n_steps

    contact = np.stack([_tripod_contact(T, HALF_PERIOD) for _ in range(N_ENVS)], axis=1)  # (T, E, 6)
    foot_force_norm = np.where(contact, 400.0, 0.0)
    foot_force_norm[:, 1, :] *= rng.uniform(0.9, 1.1, size=(T, 6))  # env1 jitter, stays > threshold

    identity_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T, N_ENVS, 1))

    root_lin_vel_b = np.zeros((T, N_ENVS, 3))
    root_lin_vel_b[:, 0, :] = compiled.cmd[:, 0, :]
    root_lin_vel_b[:, 1, :] = compiled.cmd[:, 1, :] + rng.normal(0.0, 0.01, size=(T, 3))
    root_lin_vel_b[:, 2, :] = compiled.cmd[:, 2, :] * 0.5

    root_ang_vel_b = np.zeros((T, N_ENVS, 3))
    root_ang_vel_b[:, 0, 2] = compiled.cmd[:, 0, 2]
    root_ang_vel_b[:, 1, 2] = compiled.cmd[:, 1, 2] + rng.normal(0.0, 0.005, size=T)
    root_ang_vel_b[:, 2, 2] = compiled.cmd[:, 2, 2] * 0.5

    foot_pos_w = np.zeros((T, N_ENVS, 6, 3))
    for e in range(N_ENVS):
        foot_pos_w[:, e, :, 0] = (np.cumsum(root_lin_vel_b[:, e, 0]) * DT)[:, None]
    foot_pos_w[..., :2] += rng.normal(0.0, 0.001, size=foot_pos_w[..., :2].shape)

    foot_lin_vel_w = np.zeros((T, N_ENVS, 6, 3))
    foot_lin_vel_w[..., 0] = root_lin_vel_b[:, :, 0:1]

    crab_failure = np.zeros((T, N_ENVS), dtype=bool)
    fall_start = int(T * FALL_FRAC)
    crab_failure[fall_start:, FALL_ENV] = True

    return {
        "foot_force_norm": foot_force_norm,
        "cmd_applied": compiled.cmd.astype(np.float64),
        "root_lin_vel_b": root_lin_vel_b,
        "root_ang_vel_b": root_ang_vel_b,
        "root_quat_w": identity_quat,
        "root_pos_w": np.zeros((T, N_ENVS, 3)),
        "foot_pos_w": foot_pos_w,
        "foot_lin_vel_w": foot_lin_vel_w,
        "foot_ang_vel_w": np.zeros((T, N_ENVS, 6, 3)),
        "terrain_z": np.zeros((T, N_ENVS, 6)),
        "actions": rng.normal(0.0, 0.05, size=(T, N_ENVS, 4)),
        "joint_vel": rng.normal(0.0, 0.1, size=(T, N_ENVS, 4)),
        "crab_failure": crab_failure,
    }


def score_both(raw: dict[str, np.ndarray]) -> dict[str, list[dict]]:
    """Score ``raw`` with ``gait_eval.report.score_run`` at both travel-frame conventions.

    Returns ``{"yaw_0": [...], "yaw_half_pi": [...]}``, each a list of per-episode metric dicts
    (the same shape ``score_run`` returns), run through ``metrics.json_safe`` so the result is
    directly JSON-serialisable.
    """
    scenario = _scenario()
    compiled = compile_schedule(scenario, dt=DT, num_envs=N_ENVS)
    term_reason, n_steps_env = _term_reason_and_n_steps(compiled.n_steps)

    results: dict[str, list[dict]] = {}
    for key, travel_yaw in TRAVEL_YAWS.items():
        episodes = R.score_run(
            raw,
            compiled=compiled,
            scenario=scenario,
            dt=DT,
            term_reason=term_reason,
            n_steps_env=n_steps_env,
            travel_yaw=travel_yaw,
        )
        results[key] = [M.json_safe(ep) for ep in episodes]
    return results


def main() -> None:
    raw = build_raw(SEED)
    scored = score_both(raw)
    payload = {
        "generator": {
            "module": "tests/unit/fixtures/gen_gait_eval_travel_yaw_golden.py",
            "seed": SEED,
            "dt": DT,
            "n_envs": N_ENVS,
            "half_period": HALF_PERIOD,
            "fall_env": FALL_ENV,
            "fall_frac": FALL_FRAC,
            "fall_grace_steps": FALL_GRACE_STEPS,
            "travel_yaws": TRAVEL_YAWS,
            "schedule": [
                {"hold_s": h.hold_s, "vx": h.vx, "vy": h.vy, "wz": h.wz, "label": h.label}
                for h in _schedule()
            ],
        },
        "results": scored,
    }
    FIXTURE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"wrote {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
