"""The clock-economy replay tool (clock-deprecation plan step 5, 2026-09-23), on CPU, without
Isaac.

``OMNI/experiments/2026-09-23_clock_deprecation/replay_clock_economy.py`` prices every replayed
reward term of X08:c6's own economy under two settings of the two clock rewards (clock-held
1.0/1.0, clock-free 0.001/0.001) on saved rollouts, and reports RG1 (healthy > degenerate
ordering), RG2 (per-step pre-clip-total zero-income fraction) and RG3 (per-category income share
removed, report only). It is loaded by path (pure numpy/pyyaml at module scope), which itself
loads the OF tool (``2026-09-16_1805_omni_formation/replay_reward_economy.py``) and the RMT tool
(``experiments/tools/replay_motion_terms.py``) the same way.

The money test is the reproduction one: fed the OF tool's own weights and raw canary episodes,
:func:`term_incomes`/:func:`head_hold_income` reproduce the OF tool's own per-hold table to float
precision -- proof that reusing (not re-deriving) the shipped-formula primitives worked. Tests
that need the OF tool's canary raws or X08:c6's params/env.yaml (both gitignored local artifacts)
skip cleanly when they are not on disk, following the precedent in
``test_crab_hex_omni_clock_hide.py``.
"""

from __future__ import annotations

import glob as _glob
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "parkour/parkour_tasks/parkour_tasks"
OMNI = TASKS / "crab_hex_omni_task"
TOOL_PY = OMNI / "experiments/2026-09-23_clock_deprecation/replay_clock_economy.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


m = _load(TOOL_PY, "replay_clock_economy__test")
OF = m.OF
RMT = m.RMT

X08_C6_ENV_YAML = Path(m.DEFAULT_WEIGHTS_ENV_YAML)
HAS_X08_C6_ENV = X08_C6_ENV_YAML.is_file()
HAS_OF_RAWS = bool(_glob.glob(OF.DEFAULT_RAWS["walker(1a)"]))

skip_no_env = pytest.mark.skipif(not HAS_X08_C6_ENV, reason="X08:c6 params/env.yaml not on disk (gitignored)")
skip_no_of_raws = pytest.mark.skipif(not HAS_OF_RAWS, reason="OF tool's canary raws not on disk (gitignored)")


# ============================================================================ Economy loading
@skip_no_env
def test_economy_from_env_yaml_reads_x08_c6():
    econ = m.economy_from_env_yaml(X08_C6_ENV_YAML, clock=m.CLOCK_HELD)
    assert (econ.l1_w, econ.combine) == (-2.0, "product")
    assert (econ.track_lin_std, econ.track_ang_std) == (0.2, 0.5)
    assert (econ.track_lin_w, econ.track_ang_w, econ.progress_w) == (1.25, 1.0, 0.6)
    assert (econ.air_w, econ.cross_w, econ.action_rate_w) == (0.8, -3.0, -0.3)
    assert (econ.orient_w, econ.lin_vel_z_w, econ.delta_torque_w) == (-0.7, -0.15, -1e-6)
    assert (econ.foot_idle_w, econ.excess_feet_w) == (-0.12, -0.2)
    assert econ.clock == m.CLOCK_HELD
    assert any("reward_stride_length" in n for n in econ.notes)
    assert any("penalty_tibia_deviation_in_stance" in n for n in econ.notes)


@skip_no_env
def test_economy_with_clock_overrides_only_the_two_clock_weights():
    held = m.economy_from_env_yaml(X08_C6_ENV_YAML, clock=m.CLOCK_HELD)
    free = held.with_clock(m.CLOCK_FREE)
    assert held.clock == (1.0, 1.0) and free.clock == (0.001, 0.001)
    for field in ("track_lin_w", "l1_w", "combine", "progress_w", "action_rate_w"):
        assert getattr(held, field) == getattr(free, field)


def test_economy_flags_an_unreplayed_nonzero_term_but_not_a_zero_weight_one(tmp_path):
    env_yaml = tmp_path / "env.yaml"
    env_yaml.write_text(
        "rewards:\n"
        "  track_lin_vel_xy_exp: {weight: 1.0, params: {std: 0.2}}\n"
        "  reward_clock_schedule: {weight: 1.0, params: {combine: sum}}\n"
        "  reward_clock_swing_apex: {weight: 1.0, params: {}}\n"
        "  some_new_reward_nobody_replays: {weight: 0.42, params: {}}\n"
        "  some_other_reward_at_zero: {weight: 0.0, params: {}}\n"
    )
    econ = m.economy_from_env_yaml(env_yaml, clock=m.CLOCK_HELD)
    assert any("some_new_reward_nobody_replays" in n and "0.42" in n for n in econ.notes)
    assert not any("some_other_reward_at_zero" in n for n in econ.notes)


def test_not_replayed_notes_names_the_of_tools_two_gaps():
    assert "reward_stride_length" in m.NOT_REPLAYED_NOTES
    assert "penalty_tibia_deviation_in_stance" in m.NOT_REPLAYED_NOTES


# ============================================================================ the clock phase
def _synthetic_tripod_episode(T: int = 300, dt: float = 0.02, freq: float = 0.333, seed: int = 0) -> tuple[dict, np.ndarray]:
    """A clean tripod gait at a KNOWN phase (no clock_phase field -- callers add it)."""
    phase = np.cumsum(np.full(T, 2 * math.pi * freq * dt))
    rng = np.random.default_rng(seed)
    cmd = np.zeros((T, 3)); cmd[:, 0] = 0.181
    v_b = np.zeros((T, 3)); v_b[:, 0] = 0.181
    quat = np.zeros((T, 4)); quat[:, 0] = 1.0
    force = np.zeros((T, 6))
    foot_p = np.zeros((T, 6, 3))
    root_p = np.zeros((T, 3)); root_p[:, 0] = np.cumsum(v_b[:, 0]) * dt
    for f in range(6):
        phi = phase + OF.FOOT_OFFSETS[f]
        stance = np.sin(phi) < 0
        force[:, f] = np.where(stance, 120.0, 0.0)
        foot_p[:, f, 2] = np.where(stance, 0.0, 0.08) + root_p[:, 2] + OF.GROUND_OFFSET_FROM_ROOT_M
    z = {
        "cmd_applied": cmd, "root_lin_vel_b": v_b, "root_ang_vel_b": np.zeros((T, 3)),
        "root_quat_w": quat, "foot_force_norm": force, "foot_lin_vel_w": np.zeros((T, 6, 3)),
        "foot_pos_w": foot_p, "root_pos_w": root_p, "actions": rng.normal(0, 0.1, (T, 18)),
        "applied_torque": rng.normal(0, 1, (T, 24)), "heading_err": np.zeros(T),
        "segment_id": np.zeros(T, dtype=np.int16), "steady_mask": np.ones(T, dtype=bool),
        "done": np.zeros(T, dtype=bool),
    }
    return z, phase


def test_episode_primitives_uses_the_recorded_clock_phase_when_present():
    z, phase = _synthetic_tripod_episode()
    z_recorded = dict(z, clock_phase=phase)
    p_recorded = m.episode_primitives(z_recorded, action_clip=1.0, n_offsets=36)
    p_fitted = m.episode_primitives(dict(z), action_clip=1.0, n_offsets=36)
    assert p_recorded["phase_source"] == "recorded"
    assert p_fitted["phase_source"].startswith("fitted")
    assert p_recorded["offset_best"] is None and p_recorded["offset_scores"] is None
    # the fit recovers the true phase on a clean synthetic trace, so both paths agree closely
    np.testing.assert_allclose(p_recorded["clock_sum"], p_fitted["clock_sum"], atol=0.05)
    np.testing.assert_allclose(p_recorded["clock_prod"], p_fitted["clock_prod"], atol=0.05)


def test_episode_primitives_fallback_is_exactly_the_of_tools_own_fit():
    """Absent ``clock_phase``, :func:`episode_primitives` calls ``OF.episode_primitives`` directly
    (not a reimplementation) -- every non-phase-source key must be byte-identical."""
    z, _ = _synthetic_tripod_episode(seed=3)
    ours = m.episode_primitives(dict(z), action_clip=None, n_offsets=12)   # None = OF's model
    of_direct = OF.episode_primitives(dict(z), 12)
    for key in ("clock_sum", "clock_prod", "apex", "err_xy_sq", "progress", "facing", "arate", "dtau"):
        np.testing.assert_array_equal(ours[key], of_direct[key])
    assert ours["offset_best"] == of_direct["offset_best"]


# ============================================================================ term scoring
def test_pre_clip_total_is_the_sum_of_term_values():
    z, phase = _synthetic_tripod_episode(seed=1)
    p = m.episode_primitives(dict(z, clock_phase=phase), action_clip=1.0, n_offsets=8)
    econ = m.Economy()
    vals = m.term_values(p, econ)
    total = m.pre_clip_total(p, econ)
    np.testing.assert_allclose(total, sum(vals.values()))


@skip_no_of_raws
def test_term_incomes_reproduces_the_of_tools_clock_held_table():
    """The reproduction test (plan step 5): fed the OF tool's own OLD-variant weights and its own
    canary raws, :func:`term_incomes` / :func:`head_hold_income` reproduce OF's own
    ``term_incomes`` / ``head_hold_income`` output, term for term, to float precision -- because
    both ultimately run the SAME transcribed formulas on the SAME primitives (this module's
    ``episode_primitives`` falls back to ``OF.episode_primitives`` directly whenever, as here,
    ``clock_phase`` is absent from the npz). ``action_clip=None`` is OF's own runner-less clip, so
    OF's ``action_rate`` is reproduced as OF prices it, not as the runner does."""
    n_offsets, limit = 8, 5
    econ = m.Economy(
        track_lin_w=OF.WEIGHTS["track_lin_vel_xy_exp"], track_lin_std=math.sqrt(OF.OLD.sigma2),
        track_ang_w=OF.WEIGHTS["track_ang_vel_z_exp"], track_ang_std=OF.OLD.ang_std,
        l1_w=OF.OLD.l1_w,
        clock=(OF.WEIGHTS["reward_clock_schedule"], OF.WEIGHTS["reward_clock_swing_apex"]),
        combine=OF.OLD.combine, progress_w=OF.OLD.progress_w, yaw_progress_w=OF.OLD.yaw_progress_w,
        facing_w=OF.OLD.yaw_w, facing_turn_w=OF.OLD.facing_turn_w,
        air_w=OF.WEIGHTS["reward_feet_air_time_positive"], cross_w=OF.WEIGHTS["penalty_lin_vel_y"],
        action_rate_w=OF.WEIGHTS["reward_action_rate"], orient_w=OF.WEIGHTS["reward_orientation"],
        lin_vel_z_w=OF.WEIGHTS["reward_lin_vel_z"],
        foot_idle_w=OF.WEIGHTS["penalty_foot_idle_when_forward"],
        excess_feet_w=OF.WEIGHTS["penalty_excess_feet_contact_forward"],
        delta_torque_w=OF.WEIGHTS["reward_delta_torques"],
    )
    for head_name in ("walker(1a)", "creeper(1c)"):
        for hold in ("fwd", "diag"):
            of_eps = OF.load_head(OF.DEFAULT_RAWS[head_name], n_offsets, limit)
            of_inc, of_n, _ = OF.head_hold_income(of_eps, hold, OF.OLD, upright_only=False)
            our_eps = [m.episode_primitives(dict(z), action_clip=None, n_offsets=n_offsets) for z in
                      _load_raw_episodes(OF.DEFAULT_RAWS[head_name], limit)]
            hold_idx = OF.HOLD_LABELS.index(hold)
            our_inc, our_n = m.head_hold_income(our_eps, hold_idx, econ)
            assert our_n == of_n
            assert set(our_inc) == set(of_inc)
            for key in of_inc:
                assert our_inc[key] == pytest.approx(of_inc[key], abs=1e-9)


def _load_raw_episodes(raw_glob: str, limit: int):
    import glob
    import os

    d = sorted(glob.glob(raw_glob))[-1]
    files = sorted(glob.glob(os.path.join(d, "episode_*.npz")))[:limit]
    out = []
    for f in files:
        with np.load(f) as npz:
            out.append({k: npz[k] for k in npz.files})
    return out


# ============================================================================ suite loader
JOINTS = ([f"{l}_Body_Hip_RevoluteJoint" for l in ("FL", "FR", "ML", "MR", "RL", "RR")]
          + [f"{l}_Body_CamShaft_RevoluteJoint" for l in ("FL", "FR", "ML", "MR", "RL", "RR")]
          + [f"{l}_Hip_Femur_RevoluteJoint" for l in ("FL", "FR", "ML", "MR", "RL", "RR")]
          + [f"{l}_Femur_Tibia_RevoluteJoint" for l in ("FL", "FR", "ML", "MR", "RL", "RR")])
HOLDS = [("stand", "stand", 0.0, 0.0), ("fwd", "fwd", 0.25, 0.0), ("side", "side", 0.22, 0.0),
        ("back", "back", 0.22, 0.0), ("diag", "diag", 0.25, 0.0), ("turn", "turn", 0.0, 0.3),
        ("move_turn", "move_turn", 0.22, 0.3)]


#: The REAL ``suite__omni`` schedule (run_meta of the X08:c6 MQ stamp 2026-09-23_05-43-42): 11
#: labels, both hands on side/turn/move_turn, segment ids up to 10.
SUITE_HOLDS = [("stand", "stand", 0.0, 0.0), ("fwd", "fwd", 0.25, 0.0), ("side", "lat_l", 0.22, 0.0),
               ("side", "lat_r", 0.22, 0.0), ("back", "back", 0.22, 0.0), ("diag", "diag", 0.25, 0.0),
               ("turn", "turn_l90", 0.0, 0.3), ("turn", "turn_r90", 0.0, -0.3),
               ("move_turn", "move_turn", 0.22, 0.3), ("move_turn", "move_turn_r", 0.22, -0.3)]
SUITE_LABELS = ["settle"] + [h[1] for h in SUITE_HOLDS]


def _make_suite_stamp(root: Path, *, per_cat: int = 2, T: int = 80, settle: int = 20,
                      speed_scale: float = 1.0, seed0: int = 0, label_order=None,
                      with_clock_phase: bool = False, holds=None, clip_actions="1.0",
                      action_scale: float = 0.1) -> Path:
    """A synthetic ``suite__omni`` stamp (RMT's own directory contract), following the pattern in
    ``test_crab_hex_omni_replay_motion.py``'s ``_synthetic_stamp``. ``label_order`` lets a test
    deliberately reorder the hold labels relative to OF's own ``HOLD_LABELS`` (the mislabeling
    hazard the module docstring's REUSE section names). ``run_meta.checkpoint`` points at a fake
    head whose ``params/agent.yaml`` carries ``clip_actions: <clip_actions>`` (``None`` writes no
    agent.yaml at all)."""
    run = root / "suite__omni/seed001/2026-01-01_00-00-00"
    (run / "raw").mkdir(parents=True)
    head = root / "head"
    (head / "params").mkdir(parents=True)
    if clip_actions is not None:
        (head / "params" / "agent.yaml").write_text(f"seed: 3\nclip_actions: {clip_actions}\n")
    holds = holds or HOLDS
    labels = label_order or (["settle"] + [h[1] for h in holds])
    emap, n_steps = [], []
    idx = 0
    for cat, label, speed, wz in holds:
        for k in range(per_cat):
            rng = np.random.default_rng(seed0 + 1000 + idx)
            seg = np.zeros(T, dtype=np.int16)
            seg[settle:] = labels.index(label)
            cmd = np.zeros((T, 3))
            cmd[settle:, 0] = speed
            cmd[settle:, 2] = wz
            v = (cmd * speed_scale) + rng.normal(0, 0.01, (T, 3))
            yaw = np.cumsum(v[:, 2]) * 0.02
            pos = np.zeros((T, 3)); pos[:, 0] = np.cumsum(v[:, 0]) * 0.02
            quat = np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], axis=1)
            force = rng.uniform(0, 150, (T, 6))
            z = {"actions": rng.normal(0, action_scale, (T, 18)), "cmd_applied": cmd,
                "foot_force_norm": force, "foot_pos_w": rng.normal(0, 0.02, (T, 6, 3)),
                "foot_lin_vel_w": rng.normal(0, 0.02, (T, 6, 3)), "root_pos_w": pos,
                "root_quat_w": quat, "root_lin_vel_b": v,
                "root_ang_vel_b": np.stack([0 * yaw, 0 * yaw, v[:, 2]], axis=1),
                "joint_vel": rng.normal(0, 1, (T, 24)), "applied_torque": rng.normal(0, 3, (T, 24)),
                "joint_pos": rng.normal(0, 0.2, (T, 24)), "heading_err": rng.normal(0, 0.05, T),
                "segment_id": seg, "steady_mask": np.ones(T, dtype=bool),
                "done": np.zeros(T, dtype=bool), "dt": np.float64(0.02)}
            if with_clock_phase:
                z["clock_phase"] = np.cumsum(np.full(T, 2 * math.pi * 0.333 * 0.02))
                z["clock_obs"] = np.stack([np.sin(z["clock_phase"]), np.cos(z["clock_phase"])], axis=1)
            np.savez_compressed(run / "raw" / f"episode_{idx:02d}.npz", **z)
            emap.append({"category": cat, "env": idx, "env_index": idx, "hand": None,
                        "label": label, "wave": 0})
            n_steps.append(T)
            idx += 1
    meta = {"checkpoint": str(head / "model_0.pt"), "checkpoint_sha256": "0" * 64, "scenario_id": "suite__omni",
            "dt": 0.02, "joint_names": JOINTS, "hold_labels": labels, "env_index_map": emap,
            "episodes_total": idx, "n_steps_per_env": n_steps,
            "termination_reason": ["schedule_complete"] * idx,
            "suite": {"hold_s": {h[1]: (T - settle) * 0.02 for h in holds}}}
    (run / "run_meta.json").write_text(json.dumps(meta))
    (run / "complete.json").write_text(json.dumps({"episodes_total": idx}))
    return run


def test_load_suite_rollout_refuses_a_partial_stamp(tmp_path):
    run = _make_suite_stamp(tmp_path, per_cat=1)
    (run / "complete.json").unlink()
    with pytest.raises(SystemExit, match="complete.json"):
        m.load_suite_rollout(run)


def test_load_suite_rollout_resolves_holds_from_run_meta_not_a_fixed_tuple(tmp_path):
    """A run whose ``hold_labels`` order does NOT match OF's own ``HOLD_LABELS`` must still
    resolve every category to ITS OWN hold index -- the mislabeling hazard the OF tool's own
    ``episode_primitives`` guards against and this loader avoids structurally."""
    reordered = ["settle", "stand", "diag", "back", "fwd", "side", "move_turn", "turn"]
    assert reordered != (["settle"] + [h[1] for h in HOLDS])       # actually different from OF-style order
    run = _make_suite_stamp(tmp_path, per_cat=1, label_order=reordered)
    rollout = m.load_suite_rollout(run)
    for ep in rollout["episodes"]:
        assert ep["hold_idx"] == reordered.index(ep["hold"])
        # the segment array in the raw npz must actually carry that index on the hold's own rows
        with np.load(run / "raw" / f"episode_{ep['env_index']:02d}.npz") as npz:
            seg = npz["segment_id"]
        assert (seg[seg != 0] == ep["hold_idx"]).all() or ep["hold_idx"] == 0


def test_score_rollout_reports_recorded_vs_fitted_phase_sources(tmp_path):
    run_recorded = _make_suite_stamp(tmp_path / "rec", per_cat=1, with_clock_phase=True)
    run_fitted = _make_suite_stamp(tmp_path / "fit", per_cat=1, with_clock_phase=False)
    scored_recorded = m.score_rollout(m.load_suite_rollout(run_recorded), n_offsets=6)
    scored_fitted = m.score_rollout(m.load_suite_rollout(run_fitted), n_offsets=6)
    assert m.phase_source_counts(scored_recorded) == {"recorded": len(scored_recorded)}
    assert m.phase_source_counts(scored_fitted) == {"fitted": len(scored_fitted)}


# ============================================================================ RG1 / RG2 / RG3
def _fake_primitives(T: int, hold_idx: int, **term_overrides) -> dict:
    """A minimal stand-in primitives dict (the shape :func:`term_values` reads), every field a
    constant array -- NOT from a real episode; used to pin RG1/RG2/RG3's arithmetic against known
    numbers, the same style as ``test_crab_hex_omni_replay_motion.py``'s ``_ep`` helper."""
    ones = np.ones(T)
    zeros = np.zeros(T)
    base = {"clock_sum": zeros, "clock_prod": zeros, "apex": zeros, "err_xy_sq": zeros,
           "err_wz_sq": np.full(T, 1e6), "err_l1": zeros, "progress": zeros, "yaw_prog": zeros,
           "facing": zeros, "turn_hold": zeros, "air": zeros, "cross": zeros, "arate": zeros,
           "orient": zeros, "vz2": zeros, "idle": zeros, "excess": zeros, "dtau": zeros}
    for k, v in term_overrides.items():
        base[k] = ones * v if np.isscalar(v) else v
    base.update({"T": T, "seg": np.full(T, hold_idx, dtype=np.int32),
                "steady": np.ones(T, dtype=bool), "valid": np.ones(T, dtype=bool),
                "phase_source": "synthetic-stand-in"})
    return base


def _fake_scored(category: str, hold_idx: int, p: dict) -> dict:
    return {"env_index": 0, "category": category, "hold": category, "hold_idx": hold_idx, "p": p}


def test_rg1_reports_instead_of_dividing_when_a_degenerate_total_is_not_positive():
    """Audit item 11: RG1's ratio is undefined (reported, not divided) when a degenerate's total
    is <= 0."""
    econ = m.Economy(track_lin_w=1.0, track_ang_w=0.0, l1_w=-1.0, clock=(1.0, 1.0), combine="sum",
                     progress_w=0, yaw_progress_w=0, facing_w=0, facing_turn_w=0, air_w=0,
                     cross_w=0, action_rate_w=0, orient_w=0, lin_vel_z_w=0, foot_idle_w=0,
                     excess_feet_w=0, delta_torque_w=0)
    T = 20
    # healthy: lin ~= 1 (err_xy_sq=0), no L1 -> net = +1.0
    healthy = {"E1": [_fake_scored("fwd", 1, _fake_primitives(T, 1, err_xy_sq=0.0, err_l1=0.0))]}
    # degenerate: lin ~= 0 (err_xy_sq huge), L1 = 1.0 -> net = -1.0 (<= 0: the guarded case)
    degenerate = {"E6": [_fake_scored("fwd", 1, _fake_primitives(T, 1, err_xy_sq=1e6, err_l1=1.0))]}
    rg1 = m.rg1_ordering(healthy, degenerate, econ, econ, categories=("fwd",))
    assert len(rg1["rows"]) == 1
    row = rg1["rows"][0]
    assert row["ratio_held"] is None and row["ratio_free"] is None
    assert "not divided" in row["note"] or "reported" in row["note"]
    assert row["held"]["ordering"] is True and row["free"]["ordering"] is True
    assert row["pass"] is True                     # ordering alone decides the guarded case


def test_rg1_computes_a_ratio_and_floor_when_the_degenerate_total_is_positive():
    econ = m.Economy(track_lin_w=1.0, track_ang_w=0.0, l1_w=0.0, clock=(1.0, 1.0), combine="sum",
                     progress_w=0, yaw_progress_w=0, facing_w=0, facing_turn_w=0, air_w=0,
                     cross_w=0, action_rate_w=0, orient_w=0, lin_vel_z_w=0, foot_idle_w=0,
                     excess_feet_w=0, delta_torque_w=0)
    T = 20
    healthy = {"E1": [_fake_scored("fwd", 1, _fake_primitives(T, 1, err_xy_sq=0.0))]}       # lin = 1.0
    degenerate = {"E6": [_fake_scored("fwd", 1, _fake_primitives(T, 1, err_xy_sq=1.386294))]}  # lin = 0.5^(1/std^2*...)
    rg1 = m.rg1_ordering(healthy, degenerate, econ, econ, categories=("fwd",))
    row = rg1["rows"][0]
    assert row["ratio_held"] is not None and row["ratio_free"] is not None
    assert row["ratio_held"] == pytest.approx(row["ratio_free"])   # clock plays no role here
    assert row["ratio_floor"] == pytest.approx(min(m.RG1_RATIO_TARGET, row["ratio_held"]))
    assert row["pass"] is True


def test_rg2_reports_the_pooled_zero_income_fraction_and_its_rise():
    T = 100
    econ = m.Economy(track_lin_w=0.0, track_ang_w=0.0, l1_w=-1.0, clock=(1.0, 1.0), combine="sum",
                     progress_w=0, yaw_progress_w=0, facing_w=0, facing_turn_w=0, air_w=0,
                     cross_w=0, action_rate_w=0, orient_w=0, lin_vel_z_w=0, foot_idle_w=0,
                     excess_feet_w=0, delta_torque_w=0)
    # constant -0.1 penalty every step; clock_sum alternates 0.5/0.0 so exactly half the steps
    # are net-positive under CLOCK-HELD (weight 1.0: 0.5 - 0.1 = +0.4) and none under CLOCK-FREE
    # (weight 0.001: 0.0005 - 0.1 < 0)
    clock_sum = np.tile([0.5, 0.0], T // 2)
    p = _fake_primitives(T, 1, err_l1=0.1, clock_sum=clock_sum)
    healthy = {"E1": [_fake_scored("fwd", 1, p)]}
    econ_free = econ.with_clock(m.CLOCK_FREE)
    rg2 = m.rg2_zero_income_fraction(healthy, econ, econ_free, categories=("fwd",))
    assert rg2["frac_held"] == pytest.approx(0.5)
    assert rg2["frac_free"] == pytest.approx(1.0)
    assert rg2["rise"] == pytest.approx(0.5)
    assert rg2["pass"] is False                    # rise 50 pts > 10-pt cap, and free share > 0.25


def test_rg2_passes_within_the_rise_and_share_caps():
    T = 100
    econ = m.Economy(track_lin_w=1.0, track_ang_w=0.0, l1_w=0.0, clock=(1.0, 1.0), combine="sum",
                     progress_w=0, yaw_progress_w=0, facing_w=0, facing_turn_w=0, air_w=0,
                     cross_w=0, action_rate_w=0, orient_w=0, lin_vel_z_w=0, foot_idle_w=0,
                     excess_feet_w=0, delta_torque_w=0)
    p = _fake_primitives(T, 1, err_xy_sq=0.0)       # lin ~= 1.0 every step, always positive
    healthy = {"E1": [_fake_scored("fwd", 1, p)]}
    econ_free = econ.with_clock(m.CLOCK_FREE)
    rg2 = m.rg2_zero_income_fraction(healthy, econ, econ_free, categories=("fwd",))
    assert rg2["frac_held"] == rg2["frac_free"] == 0.0
    assert rg2["pass"] is True


def test_rg3_flags_the_stand_hold_at_the_30_percent_threshold_but_not_other_categories():
    T = 50
    econ = m.Economy(track_lin_w=1.0, track_ang_w=0.0, l1_w=0.0, clock=(1.0, 1.0), combine="sum",
                     progress_w=0, yaw_progress_w=0, facing_w=0, facing_turn_w=0, air_w=0,
                     cross_w=0, action_rate_w=0, orient_w=0, lin_vel_z_w=0, foot_idle_w=0,
                     excess_feet_w=0, delta_torque_w=0)
    # pos_held = 1.0 (lin) + 1.0 (clock @ 1.0) = 2.0; pos_free = 1.0 + 0.001 = 1.001
    # share = (2.0 - 1.001) / 2.0 = 0.4995 >= 0.30
    p_stand = _fake_primitives(T, 0, err_xy_sq=0.0, clock_sum=1.0)
    p_fwd = _fake_primitives(T, 1, err_xy_sq=0.0, clock_sum=1.0)
    healthy = {"E1": [_fake_scored("stand", 0, p_stand), _fake_scored("fwd", 1, p_fwd)]}
    econ_free = econ.with_clock(m.CLOCK_FREE)
    rg3 = m.rg3_income_share_removed(healthy, econ, econ_free, categories=("stand", "fwd"))
    assert rg3["stand"]["share"] == pytest.approx(0.4995, abs=1e-3)
    assert "flag" in rg3["stand"]
    assert rg3["fwd"]["share"] == pytest.approx(0.4995, abs=1e-3)
    assert "flag" not in rg3["fwd"]                 # the pre-registered flag is stand-only


def test_rg3_reports_instead_of_dividing_when_held_positive_income_is_not_positive():
    T = 30
    econ = m.Economy(track_lin_w=0.0, track_ang_w=0.0, l1_w=-1.0, clock=(1.0, 1.0), combine="sum",
                     progress_w=0, yaw_progress_w=0, facing_w=0, facing_turn_w=0, air_w=0,
                     cross_w=0, action_rate_w=0, orient_w=0, lin_vel_z_w=0, foot_idle_w=0,
                     excess_feet_w=0, delta_torque_w=0)
    p = _fake_primitives(T, 0, err_l1=1.0)          # every term <= 0 -> pos_held == 0
    healthy = {"E1": [_fake_scored("stand", 0, p)]}
    econ_free = econ.with_clock(m.CLOCK_FREE)
    rg3 = m.rg3_income_share_removed(healthy, econ, econ_free, categories=("stand",))
    assert rg3["stand"]["share"] is None
    assert "not divided" in rg3["stand"]["note"] or "reported" in rg3["stand"]["note"]


# ============================================================================ end to end
@pytest.mark.skipif(not HAS_X08_C6_ENV, reason="X08:c6 params/env.yaml not on disk (gitignored)")
def test_main_runs_end_to_end_on_two_synthetic_stamps_and_writes_rg1_rg2_rg3(tmp_path):
    healthy_dir = _make_suite_stamp(tmp_path / "healthy", per_cat=2, T=80, speed_scale=1.0, seed0=0)
    degen_dir = _make_suite_stamp(tmp_path / "degen", per_cat=2, T=80, speed_scale=0.02, seed0=500)
    out = tmp_path / "out.json"
    rc = m.main(["--healthy", f"E1={healthy_dir}", "--degenerate", f"E6={degen_dir}",
                "--n-offsets", "6", "--out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert set(payload) >= {"RG1", "RG2", "RG3", "economy_source", "not_replayed",
                            "known_simplifications", "phase_sources"}
    assert payload["RG1"]["pass"] in (True, False)
    assert payload["RG2"]["pass"] in (True, False)
    assert set(payload["RG3"]) == set(m.ALL_CATEGORIES)
    assert payload["action_clips"]["E1"]["clip"] == payload["action_clips"]["E6"]["clip"] == 1.0
    assert payload["action_clips"]["E1"]["source"].endswith("head/params/agent.yaml")


# ============================================================================ step-6 review fixes
def test_the_real_11_label_suite_schedule_scores_end_to_end(tmp_path):
    """Step-6 fix 1: OF's slow__omni 8-hold guard must never see suite data. A stamp with the
    REAL suite__omni schedule (11 labels, segment ids up to 10) loads, scores and runs RG1-RG3;
    ``turn_hold`` comes from the stamp's own labels, and both hands are read."""
    healthy_dir = _make_suite_stamp(tmp_path / "h", per_cat=1, holds=SUITE_HOLDS, seed0=0)
    degen_dir = _make_suite_stamp(tmp_path / "d", per_cat=1, holds=SUITE_HOLDS, speed_scale=0.02,
                                  seed0=500)
    healthy_roll = m.load_suite_rollout(healthy_dir)
    assert healthy_roll["meta"]["hold_labels"] == SUITE_LABELS
    assert max(ep["hold_idx"] for ep in healthy_roll["episodes"]) == 10
    healthy = {"E1": m.score_rollout(healthy_roll, n_offsets=4)}
    degenerate = {"E6": m.score_rollout(m.load_suite_rollout(degen_dir), n_offsets=4)}
    assert len(healthy["E1"]) == len(SUITE_HOLDS)
    for e in healthy["E1"]:
        p = e["p"]
        assert int(p["seg"].max()) == e["hold_idx"]                  # real ids restored
        on_hold = p["seg"] == e["hold_idx"]
        want = 1.0 if e["hold"] in m.TURN_HOLD_LABELS else 0.0
        assert (p["turn_hold"][on_hold] == want).all()
        assert (p["turn_hold"][~on_hold] == 0.0).all()
    econ = m.Economy()
    econ_free = econ.with_clock(m.CLOCK_FREE)
    rg1 = m.rg1_ordering(healthy, degenerate, econ, econ_free)
    rows = rg1["rows"] + rg1["held_reversed"]
    assert {r["category"] for r in rows} == set(m.MOVING_CATEGORIES)
    n_by_cat = {r["category"]: r["n_healthy"] for r in rows}
    assert n_by_cat["side"] == n_by_cat["turn"] == n_by_cat["move_turn"] == 2 * n_by_cat["fwd"]
    rg2 = m.rg2_zero_income_fraction(healthy, econ, econ_free)
    assert rg2["n"] > 0 and rg2["pass"] in (True, False)
    rg3 = m.rg3_income_share_removed(healthy, econ, econ_free)
    assert set(rg3) == set(m.ALL_CATEGORIES)
    assert rg3["side"]["n"] == 2 * rg3["fwd"]["n"]


def test_episode_primitives_idle_reset_matches_of_across_the_segment_remap():
    """The boundary-preserving remap handed to OF leaves ``idle_reset`` (OF's only other
    ``segment_id`` reader) exactly as OF computes it on the real ids."""
    z, _ = _synthetic_tripod_episode(seed=5)
    seg = np.zeros(len(z["segment_id"]), dtype=np.int16)
    seg[100:] = 3
    seg[200:] = 5
    z = dict(z, segment_id=seg)
    z["foot_force_norm"] = z["foot_force_norm"].copy()
    z["foot_force_norm"][:, 0] = 0.0                                  # one foot always idle
    ours = m.episode_primitives(dict(z), action_clip=1.0, n_offsets=4)
    of_direct = OF.episode_primitives(dict(z), 4)
    np.testing.assert_array_equal(ours["idle_reset"], of_direct["idle_reset"])
    np.testing.assert_array_equal(ours["seg"], of_direct["seg"])
    np.testing.assert_array_equal(ours["turn_hold"], of_direct["turn_hold"])


def _progress_econ() -> "m.Economy":
    return m.Economy(track_lin_w=0.0, track_ang_w=0.0, l1_w=0.0, clock=(1.0, 1.0), combine="sum",
                     progress_w=1.0, yaw_progress_w=0, facing_w=0, facing_turn_w=0, air_w=0,
                     cross_w=0, action_rate_w=0, orient_w=0, lin_vel_z_w=0, foot_idle_w=0,
                     excess_feet_w=0, delta_torque_w=0)


def test_rg1_judges_only_triples_whose_clock_held_ordering_holds():
    """Step-6 fix 2: a triple already reversed under clock-held is reported as 'no clock-held
    ordering: not an RG1 criterion' and does not fail RG1."""
    T = 20
    econ = _progress_econ()
    healthy = {"E3": [_fake_scored("fwd", 2, _fake_primitives(T, 2, progress=1.0)),
                      _fake_scored("diag", 6, _fake_primitives(T, 6, progress=0.2))]}
    degenerate = {"E7": [_fake_scored("fwd", 2, _fake_primitives(T, 2, progress=0.5)),
                         _fake_scored("diag", 6, _fake_primitives(T, 6, progress=1.0))]}
    rg1 = m.rg1_ordering(healthy, degenerate, econ, econ.with_clock(m.CLOCK_FREE),
                         categories=("fwd", "diag"))
    assert rg1["pass"] is True
    assert [r["category"] for r in rg1["rows"]] == ["fwd"]
    (rev,) = rg1["held_reversed"]
    assert rev["category"] == "diag" and rev["pass"] is None
    assert rev["note"] == "no clock-held ordering: not an RG1 criterion"


def test_rg1_and_rg3_pool_both_hands_of_a_category():
    """Step-6 fix 3: side = lat_l (idx 3) + lat_r (idx 4); both hands contribute to RG1 and RG3."""
    T = 20
    econ = _progress_econ()
    econ_free = econ.with_clock(m.CLOCK_FREE)
    healthy = {"E1": [_fake_scored("side", 3, _fake_primitives(T, 3, progress=1.0)),
                      _fake_scored("side", 4, _fake_primitives(T, 4, progress=3.0))]}
    degenerate = {"E6": [_fake_scored("side", 3, _fake_primitives(T, 3, progress=0.5)),
                         _fake_scored("side", 4, _fake_primitives(T, 4, progress=0.5))]}
    rg1 = m.rg1_ordering(healthy, degenerate, econ, econ_free, categories=("side",))
    (row,) = rg1["rows"]
    assert row["n_healthy"] == 2 * T
    assert row["held"]["healthy_total"] == pytest.approx(2.0)        # mean of 1.0 and 3.0
    rg3 = m.rg3_income_share_removed(healthy, econ, econ_free, categories=("side",))
    assert rg3["side"]["n"] == 2 * T
    assert rg3["side"]["pos_held"] == pytest.approx(2.0)


# ============================================================================ the runner's action clip
# 2026-09-25 fix: the runner clamps every action to +-clip_actions (the head's params/agent.yaml)
# before the action term's own +-1 / +-4.8 clip, and reward_action_rate reads the result. OF's
# primitives apply only the second clamp, which overstated action_rate 2.9-3.4x on the step-0 walkers.
OMNI_RUNS = X08_C6_ENV_YAML.parents[2]                     # .../logs/rsl_rl/crab_hex_omni_flat_walk
#: The step-0 replay heads (STEP0_READOUT.md): E1 and E6's donor X08:c6, E3 X08:t3, E7 the 1c creeper.
STEP0_HEAD_AGENT_YAMLS = {
    "X08:c6": X08_C6_ENV_YAML.parent / "agent.yaml",
    "X08:t3": OMNI_RUNS / "2026-09-22_09-10-29_seed3_X08_t3_A15pB/params/agent.yaml",
    "1c creeper": OMNI_RUNS / "2026-09-17_00-13-40/params/agent.yaml",
}


def _expected_arate(actions: np.ndarray, clip: float) -> np.ndarray:
    a = np.clip(np.asarray(actions, dtype=np.float64), -clip, clip)
    out = np.zeros(a.shape[0])
    out[1:] = np.linalg.norm(np.diff(a, axis=0), axis=1)
    return out


def test_load_suite_rollout_takes_the_action_clip_from_the_heads_agent_yaml(tmp_path):
    run = _make_suite_stamp(tmp_path, per_cat=1, clip_actions="0.5", action_scale=3.0)
    rollout = m.load_suite_rollout(run)
    assert rollout["action_clip"] == 0.5
    assert rollout["action_clip_source"] == f"agent: {tmp_path / 'head/params/agent.yaml'}"
    scored = m.score_rollout(rollout, n_offsets=4)
    for e, ep in zip(scored, rollout["episodes"]):
        assert e["p"]["action_clip"] == 0.5
        np.testing.assert_allclose(e["p"]["arate"], _expected_arate(ep["z"]["actions"], 0.5))
        assert e["p"]["arate"].max() <= 2 * 0.5 * math.sqrt(18) + 1e-12
        # OF's own (runner-less) arate on the same saturated actions is larger: the bug
        assert OF.episode_primitives(dict(ep["z"]), 4)["arate"].sum() > 2 * e["p"]["arate"].sum()


def test_an_explicit_action_clip_overrides_agent_yaml_and_must_be_positive(tmp_path):
    run = _make_suite_stamp(tmp_path, per_cat=1, clip_actions="0.5")
    rollout = m.load_suite_rollout(run, action_clip=0.25)
    assert rollout["action_clip"] == 0.25
    assert rollout["action_clip_source"].startswith("explicit")
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="action clip"):
            m.load_suite_rollout(run, action_clip=bad)


def test_a_missing_agent_yaml_key_or_checkpoint_is_refused_not_defaulted(tmp_path):
    run = _make_suite_stamp(tmp_path / "nofile", per_cat=1, clip_actions=None)
    with pytest.raises(SystemExit, match="--action-clip"):
        m.load_suite_rollout(run)
    run = _make_suite_stamp(tmp_path / "nokey", per_cat=1)
    (tmp_path / "nokey/head/params/agent.yaml").write_text("seed: 3\n")
    with pytest.raises(SystemExit, match="no clip_actions key"):
        m.load_suite_rollout(run)
    run = _make_suite_stamp(tmp_path / "nockpt", per_cat=1)
    meta = json.loads((run / "run_meta.json").read_text())
    (run / "run_meta.json").write_text(json.dumps(dict(meta, checkpoint=None)))
    with pytest.raises(SystemExit, match="no checkpoint"):
        m.load_suite_rollout(run)


def test_clip_actions_null_means_no_runner_clamp_which_is_ofs_own_arate(tmp_path):
    run = _make_suite_stamp(tmp_path, per_cat=1, clip_actions="null", action_scale=3.0)
    rollout = m.load_suite_rollout(run)
    assert rollout["action_clip"] is None
    ep = rollout["episodes"][0]
    p = m.episode_primitives(dict(ep["z"]), action_clip=rollout["action_clip"], n_offsets=4,
                             hold_labels=rollout["meta"]["hold_labels"])
    np.testing.assert_array_equal(p["arate"], OF.episode_primitives(dict(ep["z"]), 4)["arate"])


def test_bang_bang_action_rate_is_bounded_by_the_runner_clip():
    """Every column flipping +-8 each step (the step-0 heads are this saturated: median |mu| 5-8).
    Runner clip 1.0: 2 * sqrt(18) per step. OF's term-only clip (1.0 cam, 4.8 hip/knee):
    sqrt(6 * 2^2 + 12 * 9.6^2), ~3.96x more."""
    a = np.tile(np.array([[8.0] * 18, [-8.0] * 18]), (10, 1))
    np.testing.assert_allclose(m.action_rate_primitive(a, 1.0)[1:], 2 * math.sqrt(18))
    of_model = m.action_rate_primitive(a, None)[1:]
    np.testing.assert_allclose(of_model, math.sqrt(6 * 2.0**2 + 12 * 9.6**2))
    assert of_model[0] / (2 * math.sqrt(18)) == pytest.approx(3.96, abs=0.01)


@pytest.mark.parametrize("head", sorted(STEP0_HEAD_AGENT_YAMLS))
def test_the_step0_heads_agent_yaml_pins_clip_actions_to_1(head):
    path = STEP0_HEAD_AGENT_YAMLS[head]
    if not path.is_file():
        pytest.skip(f"{path} not on disk (gitignored run dir)")
    assert m.action_clip_from_agent_yaml(path) == 1.0
