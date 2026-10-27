# SPDX-License-Identifier: BSD-3-Clause
"""``p2b_replay.py`` (CHARTER 10 / P2B_PLAN step 2: the p2b offline replay gate). CPU only; nothing here reads a real
eval stamp or imports Isaac.

Pinned on synthetic fixtures:

* the kernels against hand-derived values: the omni command gate (strict >, a turn-only command gates on), the
  terrain-relative clearance income (swing mask, 0.03 floor, 0.26 clamp, mean over six feet), the micro-swing
  indicator (strict < 0.04), tilt^2 from a quaternion, the tracking kernel;
* the crouch counterfactual: the terrain-relative height never reads the root; the library root form does;
* the kernels against the REAL landed twins (``mdp/omni_clearance.py`` with the library gate exec'd from
  ``rewards.py``) on a random synthetic episode with values placed on every threshold (``fidelity_check``);
* step alignment, the walking / pre-fall windows and the per-swing income of :func:`replay_episode`;
* :func:`redose` (in band, moved up / down with inward 2-s.f. rounding, infeasible), :func:`rankdata` /
  :func:`spearman`;
* :func:`p2b_params` applying the CL overlay to a p2a-style dump;
* an end-to-end run on a tiny on-disk campaign (three heads, both field rows, the suite): ``replay_gate.json`` keeps
  its E1/F1 keys, gains ``p2b`` in the shape ``p2_live.py`` reads, and the report is written.
"""

from __future__ import annotations

import importlib.util
import json
import math
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


R = _load(SCRIPTS / "p2b_replay.py", "p2b_replay__test")

P = {
    "dt": 0.02, "episode_length_s": 40.0, "omni_command": True, "turn_r": 1.2,
    "fc_contact": 0.1, "fc_min": 0.03, "fc_max": 0.26, "fc_gate": 0.12, "fc_parkour_name": None,
    "sm_contact": 0.1, "sm_min": 0.04, "sm_gate": 0.12, "sm_parkour_name": None,
    "tg_func": "reward_orientation_upright", "p2a_orient_w": -0.7, "p2a_orient_func": "x",
    "track_lin_w": 1.25, "track_lin_std": 0.2, "track_ang_w": 1.0, "track_ang_std": 0.5,
    "p2a_fc_min": 0.07, "p2a_fc_parkour_name": "base_parkour", "p2a_fc_w": 0.0, "p2a_sm_w": 0.0,
    "ground_offset_from_root_m": -1.05,
}


# ================================================================================================ kernels
def test_cmd_gate_is_the_omni_clock_speed_strict():
    cmd = np.array([[0.0, 0.0, 0.0], [0.12, 0.0, 0.0], [0.121, 0.0, 0.0], [0.0, -0.2, 0.0],
                    [0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [-0.3, 0.0, 0.0]])
    g = R.cmd_gate(cmd, 0.12, 1.2)
    # exactly 0.12 is not > 0.12; a turn-only command gates on through r_turn |wz| (1.2 * 0.2)
    want = [False, False, True, True, bool(1.2 * 0.1 > 0.12), True, True]
    assert g.tolist() == want
    # the forward literal when omni_command is off: signed cmd[:, 0]
    assert R.cmd_gate(cmd, 0.12, 1.2, omni=False).tolist() == [False, False, True, False, False, False, False]


def test_clearance_kernel_floor_clamp_swing_mean():
    h = np.array([[0.0, 0.03, 0.05, 0.20, 0.40, 0.10]], dtype=np.float32)
    swing = np.array([[True, True, True, True, True, False]])
    k = R.k_foot_clearance(h, swing, np.array([1.0]), 0.03, 0.26)
    want = (0.0 + 0.0 + 0.02 + 0.17 + 0.26 + 0.0) / 6.0
    assert k[0] == pytest.approx(want, abs=1e-7)
    assert R.k_foot_clearance(h, swing, np.array([0.0]), 0.03, 0.26)[0] == 0.0


def test_swing_min_indicator_is_strict():
    h = np.array([[0.0, 0.039, 0.04, 0.05, -0.02, 0.0]], dtype=np.float32)
    swing = np.array([[True, True, True, True, True, False]])
    k = R.k_swing_min(h, swing, np.array([1.0]), 0.04)
    assert k[0] == pytest.approx(3.0 / 6.0)          # 0.0, 0.039, -0.02 (0.04 itself is not < 0.04)


def test_in_swing_is_not_above_threshold():
    f = np.array([[0.0, 0.1, 0.1001, 50.0]])
    assert R.in_swing(f, 0.1).tolist() == [[True, True, False, False]]


def test_orientation_and_tracking_kernels():
    th = 0.3
    roll = np.array([[math.cos(th / 2), math.sin(th / 2), 0.0, 0.0]])
    assert R.k_orientation(roll)[0] == pytest.approx(math.sin(th) ** 2, abs=1e-12)
    assert R.k_orientation(np.array([[1.0, 0, 0, 0]]))[0] == 0.0
    cmd = np.array([[0.25, 0.0, 0.0]])
    v = np.array([[0.05, 0.0, 0.0]])
    assert R.k_track_lin(cmd, v, 0.2)[0] == pytest.approx(math.exp(-0.04 / 0.04))
    assert R.k_track_ang(np.array([[0, 0, 0.3]]), np.array([[0, 0, 0.0]]), 0.5)[0] == pytest.approx(math.exp(-0.09 / 0.25))


def test_crouch_moves_the_root_form_only():
    fz = np.array([[0.05, 0.10, 0.2, 0.03, 0.03, 0.03]], dtype=np.float32)
    tz = np.zeros_like(fz)
    rz = np.array([1.08], dtype=np.float32)
    a, b = R.foot_height_terrain(fz, tz, rz), R.foot_height_terrain(fz, tz, rz - 0.01)
    assert np.array_equal(a, b)
    ra, rb = R.foot_height_root(fz, rz, -1.05), R.foot_height_root(fz, rz - 0.01, -1.05)
    assert np.allclose(rb - ra, 0.01, atol=1e-6)


def test_rankdata_and_spearman():
    assert R.rankdata([3, 1, 2, 2]).tolist() == [4.0, 1.0, 2.5, 2.5]
    rho, n = R.spearman(np.arange(10), np.arange(10) ** 2)
    assert n == 10 and rho == pytest.approx(1.0)
    rho, n = R.spearman(np.arange(10), -np.arange(10))
    assert rho == pytest.approx(-1.0)
    assert R.spearman([1, 2, 3], [1, 2, 3]) == (None, 3)
    assert R.spearman(np.ones(10), np.arange(10))[0] is None


# ================================================================================================ re-dose
def test_redose_keeps_an_in_band_start():
    d = R.redose(1.0, (0.01, 0.10), [0.035, 0.033, 0.032])
    assert d["feasible"] and d["weight"] == 1.0 and d["moved"] is False


def test_redose_moves_up_to_the_nearest_2sf_inside():
    d = R.redose(-0.7, (0.01, 0.05), [0.004 / 0.7, 0.0041 / 0.7, 0.0031 / 0.7])
    lo = 0.01 / (0.0031 / 0.7)
    assert d["feasible"] and d["moved"] and d["weight"] < 0
    assert abs(d["weight"]) >= lo and abs(d["weight"]) == pytest.approx(float(f"{math.ceil(lo * 10) / 10:.10g}"))
    for r in (0.004 / 0.7, 0.0041 / 0.7, 0.0031 / 0.7):
        assert 0.01 <= abs(d["weight"]) * r <= 0.05


def test_redose_moves_down_and_zero_floor_band():
    d = R.redose(-0.4, (0.0, 0.05), [0.2, 0.1])            # shares 8 % / 4 % at 0.4 -> |w| <= 0.25
    assert d["feasible"] and d["weight"] == -0.25 and d["moved"]
    d = R.redose(1.0, (0.01, 0.10), [0.2, 0.0005])        # intervals [0.05, 0.5] and [20, 200]: no overlap
    assert d["feasible"] is False and d["weight"] is None


# ================================================================================================ one episode
def _episode(T=400, fall_at=None, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    force = np.where((t[:, None] // 10 + np.arange(6)[None]) % 2 == 0, 300.0, 0.0).astype(np.float32)
    foot = np.zeros((T, 6, 3), dtype=np.float32)
    foot[..., 2] = np.where(force > 0, 0.03, 0.03 + 0.08 * rng.random((T, 6))).astype(np.float32)
    terrain = np.zeros((T, 6), dtype=np.float32)
    terrain[:, 0] = -0.06                                   # foot 0 always over a pit (relative to the others)
    foot[:, 0, 2] -= 0.06
    seg = np.zeros(T, dtype=np.int16)
    seg[100:] = 1                                           # hold 0 = stand, hold 1 = fwd
    cmd = np.zeros((T, 3), dtype=np.float32)
    cmd[100:, 0] = 0.25
    root = np.zeros((T, 3), dtype=np.float32)
    root[:, 2] = 1.06
    quat = np.tile(np.array([1.0, 0, 0, 0], dtype=np.float32), (T, 1))
    cf = np.zeros(T, dtype=bool)
    if fall_at is not None:
        cf[fall_at] = True
        quat[fall_at - 50:fall_at + 1] = [math.cos(0.15), math.sin(0.15), 0.0, 0.0]
        force, foot, terrain, seg, cmd, root, quat, cf = (x[:fall_at + 1] for x in
                                                          (force, foot, terrain, seg, cmd, root, quat, cf))
    vb = np.zeros_like(cmd)
    vb[:, 0] = 0.2
    return {"foot_force_norm": force, "foot_pos_w": foot, "terrain_z": terrain, "root_pos_w": root,
            "root_quat_w": quat, "root_lin_vel_b": vb, "root_ang_vel_b": np.zeros_like(cmd), "cmd_applied": cmd,
            "segment_id": seg, "crab_failure": cf}


def test_replay_episode_windows_and_alignment():
    z = _episode()
    r = R.replay_episode(z, ["stand", "fwd"], P)
    T = 400
    assert r["S"] == T - 1 and not r["fell"]
    # reward step s uses cmd[s]: the fwd command starts at s = 100 -> walking steps 100 .. T-2
    assert r["n_walk"] == T - 1 - 100
    # tracking kernel 1.0 * exp(-(0.05^2)/0.04) on walking steps, at dt
    assert r["walk"]["trk_lin"] == pytest.approx((T - 101) * 0.02 * math.exp(-0.0025 / 0.04), rel=1e-6)   # float32 inputs
    assert r["walk"]["tg"] == 0.0 and r["walk"]["cl"] > 0.0
    assert r["cf"]["cl_crouch"] == 0.0 and r["cf"]["sm_crouch"] == 0.0
    assert r["cf"]["cl_lift"] > 0.0 and r["cf"]["cl_root_crouch"] > 0.0
    assert set(r["walk_by_hold"]) == {"fwd"}
    sw = r["swings"]
    assert sw.shape[1] == 5 and (sw[:, 2] == 1.0).any() and (sw[:, 2] == 0.0).any()   # pit and ordinary swings
    walking = sw[sw[:, 3] > 0]
    assert (walking[:, 1] > 0).all()                       # every walking swing above the floor pays
    # stand-hold swings: the gate is off, no income (bar the <= 6 that lift off on the last stand step and swing on
    # into the fwd hold)
    assert ((sw[:, 3] == 0) & (sw[:, 1] > 0)).sum() <= 6 and ((sw[:, 3] == 0) & (sw[:, 1] == 0)).sum() > 0


def test_replay_episode_prefall_window():
    z = _episode(T=400, fall_at=300)
    r = R.replay_episode(z, ["stand", "fwd"], P)
    assert r["fell"] and r["i_fail"] == 300 and r["S"] == 300
    assert r["n_prefall"] == 100                           # reward steps 200 .. 299
    assert r["n_walk"] == 200 - 100                        # fwd from 100, cut at the pre-fall window
    assert r["prefall"]["tg"] == pytest.approx(51 * 0.02 * math.sin(0.3) ** 2, rel=1e-5)


# ================================================================================================ fidelity
def test_kernels_match_the_landed_twins(tmp_path):
    pytest.importorskip("torch")
    T = 300
    rng = np.random.default_rng(7)
    z = _episode(T=T, seed=3)
    # values ON the thresholds: forces at 0.1, heights at the floor / micro-swing / clamp edges, gate edges
    z["foot_force_norm"][::7, 1] = 0.1
    z["foot_force_norm"][::11, 2] = 0.1001
    z["foot_pos_w"][::5, 3, 2] = np.float32(0.04)
    z["foot_pos_w"][::9, 4, 2] = np.float32(0.03)
    z["foot_pos_w"][::13, 5, 2] = np.float32(0.35)
    z["cmd_applied"][::3] = rng.uniform(-0.3, 0.3, size=(len(z["cmd_applied"][::3]), 3)).astype(np.float32)
    z["cmd_applied"][::17] = [0.12, 0.0, 0.0]
    z["root_quat_w"][:] = rng.normal(size=(T, 4)).astype(np.float32)
    z["root_quat_w"] /= np.linalg.norm(z["root_quat_w"], axis=1, keepdims=True)
    path = tmp_path / "episode_00.npz"
    np.savez(path, **z)
    f = R.fidelity_check([str(path)], P, ["stand", "fwd"])
    assert f["ok"], f
    assert f["n_steps"] == T - 1


# ================================================================================================ params
def _env_yaml_dict():
    lib = "parkour_isaaclab.envs.mdp.rewards:"
    return {
        "decimation": 4, "sim": {"dt": 0.005}, "episode_length_s": 40.0, "omni_command": True,
        "omni_clock_turn_radius": 1.2,
        "rewards": {
            "reward_foot_clearance": {"func": lib + "reward_foot_clearance", "weight": 0.0, "params": {
                "contact_force_threshold": 0.1, "min_clearance_m": 0.07, "max_clearance_m": 0.26,
                "min_forward_speed_cmd": 0.12, "ground_offset_from_root_m": -1.05, "parkour_name": "base_parkour"}},
            "penalty_swing_min_clearance": {"func": lib + "penalty_swing_min_clearance", "weight": 0.0, "params": {
                "contact_force_threshold": 0.1, "min_clearance_m": 0.04, "min_forward_speed_cmd": 0.12,
                "ground_offset_from_root_m": -1.05, "parkour_name": "base_parkour"}},
            "reward_orientation": {"func": lib + "reward_orientation", "weight": -0.7,
                                   "params": {"parkour_name": "base_parkour"}},
            "track_lin_vel_xy_exp": {"func": "isaaclab.envs.mdp.rewards:track_lin_vel_xy_exp", "weight": 1.25,
                                     "params": {"std": 0.2}},
            "track_ang_vel_z_exp": {"func": "isaaclab.envs.mdp.rewards:track_ang_vel_z_exp", "weight": 1.0,
                                    "params": {"std": 0.5}},
        },
    }


def test_p2b_params_apply_the_cl_overlay():
    p = R.p2b_params(_env_yaml_dict())
    assert p["fc_min"] == 0.03 and p["p2a_fc_min"] == 0.07       # KRABBY_FOOT_CLEAR_MIN moves the income floor only
    assert p["sm_min"] == 0.04 and p["fc_max"] == 0.26
    assert p["fc_parkour_name"] is None and p["sm_parkour_name"] is None   # KRABBY_FOOT_CLEAR_FLAT 1
    assert p["dt"] == pytest.approx(0.02) and p["turn_r"] == 1.2
    assert all(p["checks"].values()), p["checks"]
    assert set(p["sources"]) >= {"max_clearance_m (clamp)", "swing-min floor", "TG func", "gate", "tracking"}


# ================================================================================================ end to end
def _write_stamp(d: Path, eps: list[dict], hold_labels, reasons, *, suite=False):
    (d / "raw").mkdir(parents=True)
    for i, z in enumerate(eps):
        np.savez(d / "raw" / f"episode_{i:02d}.npz", **z)
    meta = {"hold_labels": hold_labels, "termination_reason": reasons, "dt": 0.02, "terrain_source": "raycast_mesh",
            "checkpoint_sha256": "x", "env_overrides": {"episode_length_s": 80.0}}
    if suite:
        meta["env_index_map"] = [{"env_index": i, "category": "fwd", "label": hold_labels[1]} for i in range(len(eps))]
        (d / "complete.json").write_text("{}")
    else:
        (d / "p2_complete.json").write_text("{}")
    (d / "run_meta.json").write_text(json.dumps(meta))


def test_end_to_end_on_a_tiny_campaign(tmp_path, monkeypatch):
    import yaml

    lr, arms = tmp_path / "lr", tmp_path / "arms"
    arms.mkdir(parents=True)
    for k, head in enumerate(R.HEADS):
        run_dir = tmp_path / "runs" / head
        (run_dir / "params").mkdir(parents=True)
        (run_dir / "params" / "env.yaml").write_text(yaml.safe_dump(_env_yaml_dict()))
        for row in R.FIELD_ROWS:
            for seed in R.FIELD_SEEDS:
                eps = [_episode(T=400, seed=k * 10 + seed), _episode(T=400, fall_at=330, seed=k * 10 + seed + 1)]
                _write_stamp(lr / head / row / f"seed{seed:03d}" / "2026-10-08_00-00-00", eps, ["stand", "fwd"],
                             ["schedule_complete", "fall"])
        suite = tmp_path / "suite" / head / "2026-10-08_00-00-00"
        _write_stamp(suite, [_episode(T=200, seed=k), _episode(T=200, seed=k + 5)], ["settle", "fwd"],
                     ["schedule_complete", "schedule_complete"], suite=True)
        (arms / f"{head}.record.json").write_text(json.dumps({"run_dir": str(run_dir), "driver_stamp": str(suite)}))
    gate = tmp_path / "replay_gate.json"
    gate.write_text(json.dumps({"E1": {"episode_reward": -0.04, "verdict": "FAIL"}, "F1": {"verdict": "FAIL"}}))
    monkeypatch.setattr(R, "LROUT", lr)
    monkeypatch.setattr(R, "ARMS", arms)
    monkeypatch.setattr(R, "REPLAY_JSON", gate)
    monkeypatch.setattr(R, "REPORT_MD", tmp_path / "REPLAY_GATE_P2B.md")
    res = R.run(workers=2, tb=False)
    R.write_outputs(res)
    g = json.loads(gate.read_text())
    assert g["E1"] == {"episode_reward": -0.04, "verdict": "FAIL"} and g["F1"] == {"verdict": "FAIL"}
    assert set(g["p2b"]) >= {"CL", "TG", "tracking_income", "meta"}
    live = _load(SCRIPTS / "p2_live.py", "p2_live__p2b_replay_test")
    for arm, terms in R.ARM_TERMS.items():
        assert g["p2b"][arm]["verdict"] in ("PASS", "FAIL")
        for t in terms:
            er = live.p2b_replay_magnitude(g, arm, t)
            assert er is not None and er != 0.0
            assert (er > 0) == (live.P2B_TERM_SIGN[t] > 0)          # sign-aware, as p2_live's liveness reads it
    assert g["p2b"]["CL"]["checks"]["b_no_crouch_payoff"]["pass"] is True
    assert res["fidelity"]["ok"], res["fidelity"]
    assert (tmp_path / "REPLAY_GATE_P2B.md").read_text().startswith("# Replay gate p2b")


def test_episode_reward_units():
    """rate * L / 40: a schedule-complete episode stands for a full 40-s episode at its own rate; a fall shortens it."""
    eps = {"suite__omni": [{"duration_s": 10.0, "fell": False, "all": {"cl": 2.0}}],
           "field_hard__omni": [{"duration_s": 80.0, "fell": False, "all": {"cl": 8.0}},
                                {"duration_s": 20.0, "fell": True, "all": {"cl": 2.0}}]}
    er = R.episode_reward(eps, 0.5, "cl")
    assert er["per_row"]["suite__omni"] == pytest.approx(0.5 * 2.0 / 10.0)                 # rate 0.1/s * 40 / 40
    assert er["per_row"]["field_hard__omni"] == pytest.approx(np.mean([0.5 * 8 / 80, 0.5 * 2 / 20 * 20 / 40]))
    wsum = R.ROW_MIX["suite__omni"] + R.ROW_MIX["field_hard__omni"]
    assert er["mixed"] == pytest.approx((R.ROW_MIX["suite__omni"] * 0.1 + R.ROW_MIX["field_hard__omni"]
                                         * er["per_row"]["field_hard__omni"]) / wsum)
