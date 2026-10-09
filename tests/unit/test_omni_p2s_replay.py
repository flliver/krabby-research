# SPDX-License-Identifier: BSD-3-Clause
"""``p2s_replay.py`` (CHARTER 11 / P2S_PLAN step 4: the p2s offline replay gate). CPU only; nothing here reads a real
eval stamp or imports Isaac (the S1 s18 ``params/env.yaml`` dump is read where a test says so, skipped when absent).

Pinned on synthetic fixtures:

* the LANDED kernels through this module's stub env: six-down +3, a statue (one grounded 0-N foot) +2, a tripod
  hover 0, a loaded sliding foot priced and an airborne moving foot not, zero off the slot and when tipped;
* fidelity: the landed kernels (float64) equal this module's numpy mirror per step, with values placed on every
  threshold (50 N, upright 0.9), and :func:`stand_episode` equals ``p2s_stand``'s own numpy T1/T2 on a synthetic
  episode -- including that a facing-control yaw-rate row stays in the slot gate and only leaves the strict one;
* the gate arithmetic: :func:`sig2`, :func:`auc`, :func:`solve_doses` (w_slide exact, w_support solved then 2 s.f.,
  the realised margin), :func:`g3_counterfactuals` against hand-derived hover / shuffle / crouch values;
* :func:`ast_gate_check` on the real source (pass) and on a doctored one (fail); :func:`emulation_check`;
* :func:`pit_mask` (a hole under one foot vs a raised foot over flat ground);
* :func:`existing_terms` on a still synthetic stand at the S1 dump's weights (gated terms 0, tracking at its weight);
* :func:`merge_replay_json` keeps every other key.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-07_omni_phase2/scripts"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load(SCRIPTS / "p2s_replay.py", "p2s_replay__test")
ST = R.ST
KW = {"load_ref_n": 50.0, "upright_min": 0.9}
UP = -1.0                                   # projected_gravity_b z of an upright body


def _rows(force, vel=None, gz=None, stand=None):
    force = np.atleast_2d(np.asarray(force, dtype=np.float64))
    n = force.shape[0]
    vel = np.zeros((n, 6, 3)) if vel is None else np.asarray(vel, dtype=np.float64)
    gz = np.full(n, UP) if gz is None else np.asarray(gz, dtype=np.float64)
    stand = np.ones(n, dtype=bool) if stand is None else np.asarray(stand, dtype=bool)
    return force, vel, gz, stand


# ================================================================================================ landed kernels
def test_landed_kernels_values_through_the_stub():
    six = [700.0] * 6
    statue = [700.0, 700.0, 700.0, 700.0, 0.0, 700.0]
    hover = [700.0, 0.0, 0.0, 700.0, 700.0, 0.0]
    t1, t2 = R.landed_terms(*_rows([six, statue, hover]), **KW)
    assert t1.tolist() == [3.0, 2.0, 0.0]
    assert t2.tolist() == [0.0, 0.0, 0.0]
    # a loaded sliding foot (0.2 m/s) is priced at s_f * |v|; an airborne foot moving at 1 m/s is not
    vel = np.zeros((2, 6, 3))
    vel[0, 0] = (0.2, 0.0, 0.0)
    vel[1, 1] = (0.0, 1.0, 0.0)
    _, t2 = R.landed_terms(*_rows([six, hover], vel=vel), **KW)
    assert t2 == pytest.approx([0.2, 0.0])
    # half load (25 N) counts half: s_f = 0.5
    t1, t2 = R.landed_terms(*_rows([[25.0] + [700.0] * 5], vel=np.full((1, 6, 3), 0.0)), **KW)
    assert t1.tolist() == [2.5]


def test_landed_kernels_zero_off_slot_and_tipped():
    f, v, gz, _ = _rows([[700.0] * 6] * 3, vel=np.full((3, 6, 3), 0.1))
    t1, t2 = R.landed_terms(f, v, gz, np.array([False, False, False]), **KW)
    assert not t1.any() and not t2.any()
    t1, t2 = R.landed_terms(f, v, np.array([-0.5, -0.9, -0.91]), np.ones(3, dtype=bool), **KW)
    # strict > 0.9: -g_z 0.9 is not upright
    assert t1.tolist() == [0.0, 0.0, 3.0] and t2[:2].tolist() == [0.0, 0.0] and t2[2] > 0


def test_fidelity_landed_vs_numpy_mirror_on_thresholds():
    rng = np.random.default_rng(3)
    n = 400
    force = rng.choice([0.0, 0.5, 49.999, 50.0, 50.001, 300.0, 900.0], size=(n, 6))
    vel = rng.normal(0.0, 0.2, (n, 6, 3))
    gz = rng.choice([-1.0, -0.95, -0.9, -0.9000001, -0.5, 0.2], size=n)
    stand = rng.random(n) > 0.3
    t1, t2 = R.landed_terms(force, vel, gz, stand, **KW)
    n1, n2 = R.np_terms(force, vel, gz, stand, **KW)
    assert np.abs(t1 - n1).max() < 1e-12 and np.abs(t2 - n2).max() < 1e-12


# ================================================================================================ one episode
def _episode(n=200, hold_rows=(50, 200), wz_rows=(), slide_rows=(), zero_foot=4):
    """Settle hold 0 then a stand hold 1 (rows ``hold_rows``); steady from 50 rows into the hold."""
    force = np.full((n, 6), 400.0)
    force[:, zero_foot] = 0.0
    vel = np.zeros((n, 6, 3))
    for r in slide_rows:
        vel[r, 0] = (0.3, 0.0, 0.0)
    seg = np.zeros(n, dtype=np.int64)
    seg[hold_rows[0]:hold_rows[1]] = 1
    steady = np.zeros(n, dtype=bool)
    steady[hold_rows[0] + 50:hold_rows[1]] = True
    cmd = np.zeros((n, 3))
    for r in wz_rows:
        cmd[r, 2] = 0.05
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    return {
        "foot_force_norm": force.astype(np.float32), "foot_lin_vel_w": vel.astype(np.float32),
        "foot_pos_w": np.concatenate([np.zeros((n, 6, 2)), np.full((n, 6, 1), 0.03)], axis=-1),
        "terrain_z": np.zeros((n, 6)), "root_pos_w": np.tile([0.0, 0.0, 1.06], (n, 1)), "root_quat_w": quat,
        "root_lin_vel_b": np.zeros((n, 3)), "cmd_applied": cmd, "segment_id": seg, "steady_mask": steady,
        "crab_failure": np.zeros(n, dtype=bool), "heading_err": np.zeros(n),
    }


def test_stand_episode_matches_p2s_stand_and_keeps_facing_rows():
    p = {"dt": 0.02, "load_ref_n": 50.0, "upright_min": 0.9, "tripod_offset": 3.0}
    z = _episode(wz_rows=(120, 121, 122), slide_rows=(130, 131))
    e = R.stand_episode(z, 1, p, with_existing=False)
    m = ST.episode_stand_metrics(z, 1, 0.02)
    # gate rows = steady rows of the hold (100: post rows 100..199), the wz rows stay in the slot gate
    assert e["n"] == 100 and e["n_hold_wz_nonzero"] == 3 and e["n_strict"] == 97
    assert e["t1"] == pytest.approx(m["t1_per_s"], abs=1e-6)
    assert e["t1"] == pytest.approx(2.0)
    assert e["t2"] == pytest.approx(m["t2_per_s"], abs=1e-6)
    assert e["t2"] == pytest.approx(2 * 0.3 / 100, rel=1e-5)
    assert e["off_gate_nonzero"] == 0 and e["finite"] and e["step_fid"] < 1e-12
    assert e["pit_feet"] == 0


def test_pit_mask_hole_vs_raised_foot():
    n = 3
    fz = np.full((n, 6), 0.03)
    tz = np.zeros((n, 6))
    loaded = np.ones((n, 6), dtype=bool)
    tz[0, 2] = -0.2                       # a hole under foot 2 (the foot rests on the edge)
    fz[1, 3], loaded[1, 3] = 0.25, False  # a raised foot over flat ground: not a pit
    loaded[2] = False                     # no loaded foot: fall back to the median terrain
    pit = R.pit_mask(fz, tz, loaded, 0.03)
    assert pit[0].tolist() == [False, False, True, False, False, False]
    assert not pit[1].any() and not pit[2].any()


# ================================================================================================ arithmetic
def test_sig2_and_auc():
    assert R.sig2(1.0677) == 1.1 and R.sig2(0.32955) == 0.33 and R.sig2(0.0) == 0.0 and R.sig2(-0.0123) == -0.012
    assert R.auc([2.0, 2.0], [0.1, 0.2]) == 1.0
    assert R.auc([1.0], [1.0]) == 0.5
    assert R.auc([1.0, 0.0], [0.5]) == 0.5
    assert R.auc([], [1.0]) is None


def test_solve_doses_m_rule():
    B, dT1, dT2 = 0.3, 1.8, 0.16
    d = R.solve_doses(B, dT1, dT2)
    for dose, M in (("ST1", 1.0), ("ST2", 2.5)):
        w_sl = -0.5 * M
        exact = (M - B - abs(w_sl) * dT2) / dT1
        assert d[dose]["w_slide"] == w_sl                     # set by the rule, kept exact (-1.25 not rounded)
        assert d[dose]["w_support_exact"] == pytest.approx(exact)
        assert d[dose]["w_support"] == float(f"{exact:.2g}")
        assert d[dose]["realised_margin"] == pytest.approx(B + d[dose]["w_support"] * dT1 + abs(w_sl) * dT2)
        assert d[dose]["feasible"]
    assert R.solve_doses(5.0, 1.0, 0.1)["ST1"]["feasible"] is False


def _statue_ep(n=50, speed=0.0):
    f = np.tile([300.0, 300.0, 300.0, 300.0, 0.0, 300.0], (n, 1))
    v = np.zeros((n, 6, 3))
    v[..., 1] = speed
    return {"g3": {"force": f, "vel": v, "gz": np.full(n, UP), "root_z": np.full(n, 1.06),
                   "foot_z": np.full((n, 6), 0.03)}}


def test_g3_counterfactuals_hand_derived():
    p = {"load_ref_n": 50.0, "upright_min": 0.9}
    ws, wsl, M = 1.0, -1.25, 2.5
    r = R.g3_counterfactuals([_statue_ep(speed=0.01)], ws, wsl, M, p)
    # statue: T1 = 5 - 3 = 2, T2 = 5 loaded feet x 0.01
    assert r["statue"]["t1"] == pytest.approx(2.0) and r["statue"]["t2"] == pytest.approx(0.05)
    # hover: three feet unloaded, their weight on the other three -> T1 0, T2 = 3 x 0.01
    for h in r["hover"]["per_tripod"].values():
        assert h["t1"] == pytest.approx(0.0) and h["t2"] == pytest.approx(0.03)
        assert h["cost_vs_statue"] == pytest.approx(ws * 2.0 + wsl * (0.05 - 0.03))
    assert r["hover"]["pass"] == (r["hover"]["cost"] >= 0.5 * M)
    # shuffle: six loaded feet; +0.1 m/s along x on a 0.01 m/s y velocity -> |v| = sqrt(0.1^2 + 0.01^2)
    sh = r["shuffle"]
    assert sh["still_six_down"]["t1"] == pytest.approx(3.0) and sh["still_six_down"]["t2"] == pytest.approx(0.06)
    assert sh["shuffle"]["t2"] == pytest.approx(6 * np.hypot(0.1, 0.01))
    assert sh["cost"] == pytest.approx(-wsl * (6 * np.hypot(0.1, 0.01) - 0.06))
    assert sh["ideal_cost"] == pytest.approx(0.75)
    assert r["crouch"]["max_abs_delta"] == 0.0 and r["crouch"]["pass"]


def test_ast_gate_check_real_and_doctored():
    assert R.ast_gate_check()["pass"]
    src = R.MOTION_REWARDS_PY.read_text()
    doctored = src.replace("return torch.sum(support * speed, dim=1) * gate", "return torch.sum(support * speed, dim=1)")
    assert doctored != src
    r = R.ast_gate_check(doctored)
    assert not r["pass"] and not r["penalty_stand_slide"]["returns_times_gate"] and r["reward_stand_support"]["pass"]
    no_mask = src.replace(".stand_mask()", ".slot_mask()")
    assert not R.ast_gate_check(no_mask)["_stand_gate"]["reads_stand_mask"]


def test_emulation_check():
    assert R.emulation_check({"load_ref_n": 50.0, "upright_min": 0.9})["pass"]


# ================================================================================================ existing terms
S1_ENV = R.S1_RUN / "params/env.yaml"


@pytest.mark.skipif(not S1_ENV.is_file(), reason="S1 s18 dump not on this machine")
def test_existing_terms_still_stand_at_s1_weights():
    p = R.s1_params()
    assert all(p["checks"].values()), p["checks"]
    n = 120
    z = _episode(n=n, hold_rows=(0, n))
    jn = ([f"{l}_Body_Hip_RevoluteJoint" for l in R.FEET] + [f"{l}_Body_CamShaft_RevoluteJoint" for l in R.FEET]
          + [f"{l}_Hip_Femur_RevoluteJoint" for l in R.FEET] + [f"{l}_Femur_Tibia_RevoluteJoint" for l in R.FEET])
    q0 = R.RT.resolve_default_pos(p["tp"].default_joint_pos, jn)
    z.update({"actions": np.zeros((n, 18)), "joint_vel": np.zeros((n, 24)), "joint_pos": np.tile(q0, (n, 1)),
              "applied_torque": np.ones((n, 24)), "root_ang_vel_b": np.zeros((n, 3)),
              "leg_contact_force": np.zeros((n, 18)), "clock_phase": np.zeros(n),
              "_joint_names": jn, "_leg_link_names": [f"{l}_{b}" for l in R.FEET for b in ("Femur", "Hip", "Tibia")]})
    ex = R.existing_terms(z, p)
    w = p["weights"]
    assert all(v.shape == (n - 1,) for v in ex.values())
    assert ex["track_lin_vel_xy_exp"] == pytest.approx(np.full(n - 1, w["track_lin_vel_xy_exp"]))
    assert ex["track_ang_vel_z_exp"] == pytest.approx(np.full(n - 1, w["track_ang_vel_z_exp"]))
    assert ex["reward_tracking_yaw"] == pytest.approx(np.full(n - 1, w["reward_tracking_yaw"]))
    for gated in ("reward_feet_air_time_positive", "penalty_foot_idle_when_forward", "penalty_lin_vel_y",
                  "reward_forward_progress_along_command", "penalty_excess_feet_contact_forward",
                  "reward_yaw_progress", "reward_clock_swing_apex"):
        assert not ex[gated].any(), gated
    for still in ("penalty_tracking_error_l1", "reward_orientation", "reward_lin_vel_z", "reward_action_rate",
                  "penalty_tibia_deviation_in_stance", "reward_collision"):
        assert not ex[still].any(), still
    # delta torques: the term's buffer starts at 0, so only the first step pays (24 joints x 1^2)
    assert ex["reward_delta_torques"][0] == pytest.approx(w["reward_delta_torques"] * 24)
    assert not ex["reward_delta_torques"][1:].any()
    # stopped clock: stance quality = mean over feet of loaded x exp(-(v/0.1)^2), 5 of 6 feet loaded at 400 N
    loaded = 1.0 - np.exp(-((400.0 / 50.0) ** 2))
    assert ex["reward_clock_schedule"] == pytest.approx(np.full(n - 1, w["reward_clock_schedule"] * 5 * loaded / 6))


# ================================================================================================ json
def test_merge_replay_json_keeps_other_keys(tmp_path):
    path = tmp_path / "replay_gate.json"
    path.write_text(json.dumps({"E1": {"episode_reward": -0.02}, "F1": {"x": 1}, "p2b": {"CL": {}}}))
    R.merge_replay_json(path, {"verdict": "PASS"})
    d = json.loads(path.read_text())
    assert d["E1"] == {"episode_reward": -0.02} and d["F1"] == {"x": 1} and d["p2b"] == {"CL": {}}
    assert d["p2s"] == {"verdict": "PASS"}
