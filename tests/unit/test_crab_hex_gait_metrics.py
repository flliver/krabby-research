"""Unit tests for the crab-hex gait eval metrics (Milestone 18, Task 0).

Pure numpy -- no Isaac Sim, so these run in plain ``make test``. The point of this file is that the
gateable tripod score behaves correctly on *known* gaits, including the degenerate ones that would
otherwise silently produce NaN or a flattering number: later milestone tasks gate on it, and a gate
that scores a frozen policy 1.0 is worse than no gate.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "parkour"
    / "parkour_tasks"
    / "parkour_tasks"
    / "crab_hex_forward_task"
    / "scripts"
)
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from gait_eval import metrics as M  # noqa: E402

DT = 0.02


def _tripod_contact(n_steps: int, half_period: int) -> np.ndarray:
    """Perfectly alternating tripod: set A down while set B is up, swapping every half period."""
    contact = np.zeros((n_steps, 6), dtype=bool)
    for t in range(n_steps):
        a_down = (t // half_period) % 2 == 0
        for i in M.TRIPOD_A_IDX:
            contact[t, i] = a_down
        for i in M.TRIPOD_B_IDX:
            contact[t, i] = not a_down
    return contact


# ---------------------------------------------------------------------------
# tripod score
# ---------------------------------------------------------------------------
def test_perfect_tripod_scores_one():
    res = M.tripod_window_metrics(_tripod_contact(400, 10), dt=DT)
    assert res["valid"]
    assert res["tripod_score"] == pytest.approx(1.0, abs=1e-6)
    assert res["corr_A_B"] == pytest.approx(-1.0, abs=1e-6)
    assert res["coherence_A"] == pytest.approx(1.0)
    assert res["coherence_B"] == pytest.approx(1.0)
    assert not res["degenerate_anti_phase"]


def test_statue_scores_zero_and_flags_degenerate():
    """All six feet planted forever: coherence is a perfect 1.0, so only the zero-variance guard
    stops this from scoring 1.0 * NaN. This is the failure mode the gate exists to catch."""
    res = M.tripod_window_metrics(np.ones((400, 6), dtype=bool), dt=DT)
    assert res["tripod_score"] == 0.0
    assert res["corr_A_B"] is None
    assert res["degenerate_anti_phase"] is True
    assert len(res["foot_never_lifts"]) == 6
    # A confident zero, not an uncertain one: a degenerate window must still reach the aggregate,
    # or a frozen policy reports "unscored" instead of failing the gate.
    assert res["low_confidence"] is False


def test_all_airborne_scores_zero():
    res = M.tripod_window_metrics(np.zeros((400, 6), dtype=bool), dt=DT)
    assert res["tripod_score"] == 0.0
    assert res["degenerate_anti_phase"] is True
    assert len(res["foot_never_lands"]) == 6


def test_random_contact_scores_near_zero():
    rng = np.random.default_rng(0)
    res = M.tripod_window_metrics(rng.random((600, 6)) > 0.5, dt=DT)
    assert res["tripod_score"] < 0.1


def test_short_window_is_discarded_not_scored():
    """A 5-step window can trivially produce corr == -1; refusing beats flattering."""
    res = M.tripod_window_metrics(_tripod_contact(5, 2), dt=DT)
    assert res["valid"] is False
    assert res["tripod_score"] is None
    assert "window_too_short" in res["discard_reason"]


def test_too_few_cycles_flagged_low_confidence():
    res = M.tripod_window_metrics(_tripod_contact(60, 30), dt=DT, min_window_steps=50)
    assert res["low_confidence"] is True


def test_one_dragging_foot_degrades_but_stays_finite():
    contact = _tripod_contact(400, 10)
    contact[:, M.FOOT_ORDER.index("RL_Footpad")] = True  # one A-set foot never lifts
    res = M.tripod_window_metrics(contact, dt=DT)
    assert res["valid"]
    assert 0.0 < res["tripod_score"] < 1.0
    assert "RL_Footpad" in res["foot_never_lifts"]


def test_tripod_score_is_json_safe_in_every_branch():
    for contact in (
        _tripod_contact(400, 10),
        np.ones((400, 6), dtype=bool),
        np.zeros((400, 6), dtype=bool),
    ):
        payload = M.json_safe(M.tripod_window_metrics(contact, dt=DT))
        # allow_nan=False is what a downstream consumer should use; NaN/Infinity are not valid JSON.
        text = json.dumps(payload, allow_nan=False)
        assert "NaN" not in text and "Infinity" not in text


# ---------------------------------------------------------------------------
# tippy-tap: tripod and air time must be independent signals
# ---------------------------------------------------------------------------
def test_tippy_tap_scores_high_tripod_but_is_caught_by_air_time():
    """The whole reason air time is a separate required metric.

    A policy doing clean 40 ms micro-hops alternates its tripod sets perfectly, so the tripod score
    alone would bless it. The air-time distribution is what exposes it.
    """
    contact = _tripod_contact(400, 2)  # 2 steps == 40 ms at 50 Hz
    tri = M.tripod_window_metrics(contact, dt=DT)
    air = M.air_time_metrics(contact, dt=DT)
    assert tri["tripod_score"] > 0.9
    assert air["tippy_tap_fraction"] == pytest.approx(1.0)
    assert air["pooled"]["p50"] == pytest.approx(0.04, abs=1e-9)


def test_air_time_histogram_bins_are_step_quantized():
    air = M.air_time_metrics(_tripod_contact(400, 3), dt=DT)
    # every air interval is exactly 3 steps, so all mass must land in bin 3 and nowhere else
    assert set(k for k, v in air["histogram_step_counts"].items() if v > 0) == {3}


def test_air_time_excludes_window_censored_intervals():
    contact = np.ones((100, 6), dtype=bool)
    contact[:10, :] = False  # open at the start -- true duration unknown
    air = M.air_time_metrics(contact, dt=DT)
    assert air["censored_intervals"] == 6
    assert air["n_intervals"] == 0


# ---------------------------------------------------------------------------
# foot slip / stance stability
# ---------------------------------------------------------------------------
def _slip_case(mode: str, n_steps: int = 200):
    contact = np.zeros((n_steps, 6), dtype=bool)
    pos = np.zeros((n_steps, 6, 3))
    vel = np.zeros((n_steps, 6, 3))
    for t in range(n_steps):
        contact[t, :] = (t % 40) < 30  # 30 steps stance, 10 swing
    for i in range(6):
        x = 0.0
        for t in range(n_steps):
            if contact[t, i]:
                if mode == "drag":
                    x += 0.002
                    vel[t, i, 0] = 0.1
                elif mode == "jitter":
                    x += 0.002 * (1 if t % 2 else -1)
                    vel[t, i, 0] = 0.1 * (1 if t % 2 else -1)
            else:
                x += 0.02
                vel[t, i, 0] = 1.0
            pos[t, i, 0] = x
    return M.slip_metrics(contact, pos, vel, dt=DT)


def test_planted_foot_has_zero_slip():
    foot = _slip_case("planted")["per_foot"]["FL_Footpad"]
    assert foot["slip_path_len"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert foot["slip_ratio"]["mean"] == pytest.approx(0.0, abs=1e-9)


def test_dragged_foot_has_path_equal_to_net_displacement():
    """Monotonic slide: every step moves the same direction, so path length == net displacement."""
    foot = _slip_case("drag")["per_foot"]["FL_Footpad"]
    assert foot["slip_path_len"]["mean"] > 0.01
    assert foot["slip_path_len"]["mean"] == pytest.approx(foot["slip_net_disp"]["mean"], rel=1e-6)
    assert foot["slip_ratio"]["mean"] > 0.0


def test_jittering_foot_separates_path_length_from_net_displacement():
    """Vibrating in place: high abrasion path, near-zero net travel. Reporting only one would
    conflate this with a foot that is actually skating forward."""
    foot = _slip_case("jitter")["per_foot"]["FL_Footpad"]
    assert foot["slip_path_len"]["mean"] > 10 * foot["slip_net_disp"]["mean"]


def test_never_lifting_foot_reports_null_ratio_but_keeps_rate():
    n = 200
    contact = np.ones((n, 6), dtype=bool)
    pos = np.zeros((n, 6, 3))
    vel = np.zeros((n, 6, 3))
    for t in range(n):
        pos[t, :, 0] = 0.001 * t
        vel[t, :, 0] = 0.05
    res = M.slip_metrics(contact, pos, vel, dt=DT)["per_foot"]["FL_Footpad"]
    assert res["never_lifts"] is True
    # no touchdown => no stride => ratio undefined; the per-second rate keeps it visible
    assert res["slip_ratio"]["count"] == 0
    assert res["slip_path_len_per_s"] > 0.0


def test_slip_work_proxy_weights_by_normal_force():
    n = 200
    contact = np.ones((n, 6), dtype=bool)
    pos = np.zeros((n, 6, 3))
    vel = np.zeros((n, 6, 3))
    vel[:, :, 0] = 0.1
    light = M.slip_metrics(contact, pos, vel, dt=DT, foot_force_norm=np.full((n, 6), 10.0))
    heavy = M.slip_metrics(contact, pos, vel, dt=DT, foot_force_norm=np.full((n, 6), 100.0))
    lw = light["per_foot"]["FL_Footpad"]["slip_work_proxy"]["mean"]
    hw = heavy["per_foot"]["FL_Footpad"]["slip_work_proxy"]["mean"]
    assert hw == pytest.approx(10.0 * lw, rel=1e-6)


# ---------------------------------------------------------------------------
# supporting metrics
# ---------------------------------------------------------------------------
def test_pearson_returns_none_on_constant_series():
    assert M.pearson(np.ones(50), np.arange(50)) is None
    assert M.pearson(np.arange(50), np.arange(50)) == pytest.approx(1.0)


def test_contiguous_runs():
    mask = np.array([0, 1, 1, 0, 0, 1, 0], dtype=bool)
    assert M.contiguous_runs(mask) == [(1, 3), (5, 6)]
    assert M.contiguous_runs(np.zeros(5, dtype=bool)) == []


def test_debounce_removes_short_runs():
    mask = np.array([1, 0, 1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    assert not M.debounce(mask, min_true=2)[0]


def test_json_safe_converts_nan_to_none():
    assert M.json_safe({"a": float("nan"), "b": np.float32(1.5), "c": np.int64(2)}) == {
        "a": None,
        "b": 1.5,
        "c": 2,
    }


def test_tracking_metrics_signed_mean_exposes_undershoot():
    """mean_abs alone hides systematic undershoot, which is the suspected speed-forcing pathology."""
    n = 100
    cmd = np.tile(np.array([0.8, 0.0, 0.0]), (n, 1))
    lin = np.tile(np.array([0.5, 0.0, 0.0]), (n, 1))
    res = M.tracking_metrics(cmd, lin, np.zeros((n, 3)))
    assert res["vx"]["signed_mean"] == pytest.approx(0.3)
    assert res["vx"]["mean_abs"] == pytest.approx(0.3)


def test_tracking_ratio_flags_creep_and_gates_on_command_floor():
    """The creep-audit metric: a 0.92-completion policy moving at 20% of command must score
    ratio ~0.2, and sub-clip (stop-by-construction) commands must report None, not a huge or
    divide-by-near-zero ratio."""
    n = 100
    creep = M.tracking_metrics(
        np.tile(np.array([0.35, 0.0, 0.0]), (n, 1)),
        np.tile(np.array([0.07, 0.0, 0.0]), (n, 1)),
        np.zeros((n, 3)),
        ratio_min_cmd=0.2,
    )
    assert creep["vx"]["ratio"] == pytest.approx(0.2)
    stand = M.tracking_metrics(
        np.tile(np.array([0.1, 0.0, 0.0]), (n, 1)),
        np.tile(np.array([0.05, 0.0, 0.0]), (n, 1)),
        np.zeros((n, 3)),
        ratio_min_cmd=0.2,
    )
    assert stand["vx"]["ratio"] is None
    # Without the floor argument the key is absent entirely (backward-compatible shape).
    legacy = M.tracking_metrics(
        np.tile(np.array([0.35, 0.0, 0.0]), (n, 1)),
        np.tile(np.array([0.07, 0.0, 0.0]), (n, 1)),
        np.zeros((n, 3)),
    )
    assert "ratio" not in legacy["vx"]


def test_aggregate_surfaces_tracking_ratio_and_by_hold():
    """Aggregate must carry the tracking ratio so gates can read it from scenario_metrics.json
    without re-touching raw npz."""
    from gait_eval import report as R

    def _ep(idx, ratio, achieved):
        return {
            "env_index": idx,
            "termination_reason": "schedule_complete",
            "completed_schedule": True,
            "n_steps": 100,
            "tripod_score": None,
            "tippy_tap_fraction": None,
            "slip_ratio_mean": None,
            "tracking_ratio": ratio,
            "holds": {
                "low": {
                    "tripod": {"tripod_score": None, "low_confidence": True},
                    "tracking": {
                        "vx": {"cmd_mean": 0.35, "actual_mean": achieved, "ratio": ratio}
                    },
                }
            },
        }

    agg = R.aggregate([_ep(0, 0.2, 0.07), _ep(1, 0.4, 0.14)])
    assert agg["tracking_ratio"]["median"] == pytest.approx(0.3)
    hold = agg["tracking_by_hold"]["low"]
    assert hold["cmd_vx_mean"] == pytest.approx(0.35)
    assert hold["achieved_vx"]["mean"] == pytest.approx(0.105)
    assert hold["ratio"]["n"] == 2


def test_action_metrics_reports_reversals_per_joint_group():
    """CamShaft reversals are a distinct pathology from knee reversals (the cam is meant to spin one
    way), so a flat 18-DOF mean would wash the signal out."""
    n = 100
    actions = np.zeros((n, 4))
    actions[:, 0] = np.linspace(0, 1, n)  # monotonic: no reversals
    actions[:, 1] = 0.5 * (-1) ** np.arange(n)  # alternating: many reversals
    res = M.action_metrics(actions, joint_groups={"camshaft": [0], "femur_tibia": [1]})
    assert res["per_group"]["camshaft"]["action_sign_reversals"] == 0
    assert res["per_group"]["femur_tibia"]["action_sign_reversals"] > 50


def test_swing_clearance_interior_min_detects_drag():
    """min_over_swing is ~0 for any gait (swings start and end at ground level); the interior
    minimum is the number that actually distinguishes a lifted foot from a dragged one."""
    n = 120
    contact = np.zeros((n, 6), dtype=bool)
    contact[(np.arange(n) % 40) < 30, :] = True
    pos = np.zeros((n, 6, 3))
    lifted = pos.copy()
    swing = ~contact[:, 0]
    lifted[swing, :, 2] = 0.10
    terrain = np.zeros((n, 6))
    res_flat = M.swing_clearance_metrics(contact, pos, terrain)
    res_lift = M.swing_clearance_metrics(contact, lifted, terrain)
    fl_flat = res_flat["per_foot"]["FL_Footpad"]["min_over_swing_interior"]["mean"]
    fl_lift = res_lift["per_foot"]["FL_Footpad"]["min_over_swing_interior"]["mean"]
    assert fl_flat == pytest.approx(0.0, abs=1e-9)
    assert fl_lift > 0.05


# --- shaft_spin_metrics (velocity-era one-direction spin gate) ---


def test_spin_metrics_pure_one_direction():
    """Constant positive omega on all shafts: ratio 1.0, zero reversals, correct revolutions."""
    n, omega = 500, 6.0
    jv = np.zeros((n, 24))
    shaft_ids = [1, 3, 5, 7, 9, 11]
    jv[:, shaft_ids] = omega
    out = M.shaft_spin_metrics(jv, shaft_ids, dt=DT)
    assert out["one_direction_ratio_median"] == pytest.approx(1.0)
    assert out["reversals_per_s_median"] == 0.0
    assert out["mean_abs_vel_median"] == pytest.approx(omega)
    expected_revs = omega * n * DT / (2 * np.pi)
    assert out["net_revolutions"][0] == pytest.approx(expected_revs)


def test_spin_metrics_symmetric_oscillation():
    """Sine-wave shaft velocity: ratio ~0, reversal rate ~= 2x the oscillation frequency."""
    n, freq_hz, amp = 1000, 1.4, 6.0
    t = np.arange(n) * DT
    jv = np.zeros((n, 24))
    shaft_ids = [1, 3, 5, 7, 9, 11]
    jv[:, shaft_ids] = amp * np.sin(2 * np.pi * freq_hz * t)[:, None]
    out = M.shaft_spin_metrics(jv, shaft_ids, dt=DT)
    assert out["one_direction_ratio_median"] < 0.05
    assert out["reversals_per_s_median"] == pytest.approx(2 * freq_hz, rel=0.15)
    assert abs(out["net_revolutions"][0]) < 0.5


def test_spin_metrics_deadzone_dwell_no_reversal():
    """Coasting into the deadzone and resuming the SAME direction pays no reversal
    (sticky rule, matching PenaltyMotorDirectionReversal)."""
    jv = np.zeros((300, 24))
    shaft_ids = [1, 3, 5, 7, 9, 11]
    jv[:100, shaft_ids] = 5.0
    jv[100:200, shaft_ids] = 0.01  # inside 0.05 deadzone
    jv[200:, shaft_ids] = 5.0
    out = M.shaft_spin_metrics(jv, shaft_ids, dt=DT)
    assert out["reversals_per_s_median"] == 0.0
    # genuine flip across a dwell still pays exactly once
    jv[200:, shaft_ids] = -5.0
    out2 = M.shaft_spin_metrics(jv, shaft_ids, dt=DT)
    assert out2["reversals_per_s_median"] == pytest.approx(1.0 / (300 * DT))
