"""Unit tests for the crab-hex gait eval metrics (Milestone 18, Task 0).

Pure numpy -- no Isaac Sim, so these run in plain ``make test``. The point of this file is that the
gateable tripod score behaves correctly on *known* gaits, including the degenerate ones that would
otherwise silently produce NaN or a flattering number: later milestone tasks gate on it, and a gate
that scores a frozen policy 1.0 is worse than no gate.
"""

import json
import math
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


# ---------------------------------------------------------------------------
# travel frame (sideways-task harness sharing, 2026-09-11): one harness serves the forward task
# (travel_yaw absent/0.0, byte-identical) and the sideways task (travel_yaw = +pi/2, travel axis
# = body +y). See test_crab_hex_stability_metrics.py for the fall_direction/orientation/
# support_polygon travel-frame tests.
# ---------------------------------------------------------------------------
def test_lin_vel_to_travel_frame():
    v = np.array([0.4, -0.1, 0.05])
    same = M.lin_vel_to_travel_frame(v, 0.0)
    assert same is v
    same_default = M.lin_vel_to_travel_frame(v)
    assert same_default is v

    rotated = M.lin_vel_to_travel_frame(v, math.pi / 2)
    np.testing.assert_allclose(rotated, [-0.1, -0.4, 0.05], atol=1e-9)

    # The tracking-ratio creep-audit check itself: a robot moving at body +y (0, 0.3, 0) is
    # exactly on-command for a travel-frame vx command of 0.3 once rotated into the travel frame.
    n = 50
    cmd = np.tile(np.array([0.3, 0.0, 0.0]), (n, 1))
    lin_b = np.tile(np.array([0.0, 0.3, 0.0]), (n, 1))
    lin_travel = M.lin_vel_to_travel_frame(lin_b, math.pi / 2)
    res = M.tracking_metrics(cmd, lin_travel, np.zeros((n, 3)), ratio_min_cmd=0.2)
    assert res["vx"]["ratio"] == pytest.approx(1.0, abs=1e-6)


def test_stride_along_command_uses_travel_yaw():
    """Feet displacing along world +x, root at body yaw -pi/2, cmd_xy=(0.3, 0) (travel-frame vx).

    ``stride_metrics``' ``root_yaw`` parameter must be the TRAVEL frame's world yaw
    (``body_yaw + travel_yaw``, the report.py convention) for ``along_command`` to read the
    stride along the commanded travel direction -- passing the raw body yaw instead measures the
    stride along the (unrelated) body +x axis and reads ~0.
    """
    n = 30
    contact = np.zeros((n, 6), dtype=bool)
    cycle = (np.arange(n) % 10) < 5
    contact[:, 0] = cycle  # touchdowns at t=10 and t=20 (one stride pair)
    pos = np.zeros((n, 6, 3))
    pos[:, 0, 0] = np.arange(n) * 0.02  # foot moves steadily along world +x
    cmd_xy = np.tile(np.array([0.3, 0.0]), (n, 1))

    root_yaw_raw = np.full(n, -math.pi / 2)
    res_raw = M.stride_metrics(contact, pos, dt=DT, cmd_xy=cmd_xy, root_yaw=root_yaw_raw)
    fl_raw = res_raw["per_foot"]["FL_Footpad"]
    assert fl_raw["n_strides"] == 1
    magnitude = fl_raw["magnitude"]["mean"]
    assert magnitude > 0.0
    assert fl_raw["along_command"]["mean"] == pytest.approx(0.0, abs=1e-9)

    root_yaw_travel = root_yaw_raw + math.pi / 2
    res_travel = M.stride_metrics(contact, pos, dt=DT, cmd_xy=cmd_xy, root_yaw=root_yaw_travel)
    fl_travel = res_travel["per_foot"]["FL_Footpad"]
    assert fl_travel["along_command"]["mean"] == pytest.approx(magnitude, rel=1e-6)


def test_report_default_path_identical():
    """``score_episode``/``score_run`` at ``travel_yaw=0.0`` (explicit or default) are identical."""
    from gait_eval import report as R
    from gait_eval.schedule import Hold, Scenario, compile_schedule

    scenario = Scenario(
        id="synthetic",
        task="Isaac-Crab-Hex-Flat-Walk-v0",
        checkpoint="",
        schedule=[Hold(hold_s=3.0, vx=0.3, vy=0.0, wz=0.0, label="hold0")],
        episodes=1,
        env_seed=1,
    )
    compiled = compile_schedule(scenario, dt=DT, num_envs=1)
    T = compiled.n_steps
    contact = _tripod_contact(T, 5)
    identity_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T, 1))

    raw = {
        "foot_force_norm": np.where(contact, 400.0, 0.0)[:, None, :],
        "cmd_applied": compiled.cmd.astype(np.float64),
        "root_lin_vel_b": np.tile(np.array([0.3, 0.0, 0.0]), (T, 1))[:, None, :],
        "root_ang_vel_b": np.zeros((T, 1, 3)),
        "root_quat_w": identity_quat[:, None, :],
        "root_pos_w": np.zeros((T, 1, 3)),
        "foot_pos_w": np.zeros((T, 1, 6, 3)),
        "foot_lin_vel_w": np.zeros((T, 1, 6, 3)),
        "foot_ang_vel_w": np.zeros((T, 1, 6, 3)),
        "terrain_z": np.zeros((T, 1, 6)),
        "actions": np.zeros((T, 1, 4)),
        "joint_vel": np.zeros((T, 1, 4)),
        "crab_failure": np.zeros((T, 1), dtype=bool),
    }
    episode_kwargs = dict(
        env_idx=0,
        compiled=compiled,
        scenario=scenario,
        dt=DT,
        term_reason="schedule_complete",
        n_steps=T,
    )
    default_ep = R.score_episode(raw, **episode_kwargs)
    explicit_ep = R.score_episode(raw, travel_yaw=0.0, **episode_kwargs)
    assert default_ep == explicit_ep
    # Sanity: this raw shape actually exercises the code paths under test (not vacuously equal).
    assert default_ep["tripod_score"] is not None and default_ep["tripod_score"] > 0.9
    assert default_ep["fall_direction"]["fall_class"] == "none"

    run_kwargs = dict(
        compiled=compiled,
        scenario=scenario,
        dt=DT,
        term_reason=["schedule_complete"],
        n_steps_env=[T],
    )
    default_run = R.score_run(raw, **run_kwargs)
    explicit_run = R.score_run(raw, travel_yaw=0.0, **run_kwargs)
    assert default_run == explicit_run


# ---------------------------------------------------------------------------
# per-step travel frame (omni task, 2026-09-16): travel_yaw may be a (T,) array -- the direction of
# the body command at each step. The scalar paths above must stay bit-identical, which is what the
# "constant array == float result" tests below actually pin.
# ---------------------------------------------------------------------------
def test_lin_vel_to_travel_frame_constant_array_equals_the_float_result_bitwise():
    rng = np.random.default_rng(7)
    v = rng.normal(size=(40, 3))
    for angle in (math.pi / 2, -0.37, 2.1):
        scalar = M.lin_vel_to_travel_frame(v, angle)
        array = M.lin_vel_to_travel_frame(v, np.full(v.shape[0], angle))
        assert np.array_equal(scalar, array)  # bitwise, not approx


def test_lin_vel_to_travel_frame_rotates_per_row():
    v = np.tile(np.array([0.3, 0.0, 0.05]), (3, 1))
    angles = np.array([0.0, math.pi / 2, math.pi])
    out = M.lin_vel_to_travel_frame(v, angles)
    np.testing.assert_allclose(out[0], [0.3, 0.0, 0.05], atol=1e-12)
    np.testing.assert_allclose(out[1], [0.0, -0.3, 0.05], atol=1e-12)
    np.testing.assert_allclose(out[2], [-0.3, 0.0, 0.05], atol=1e-12)
    # A robot travelling along its own command direction reads pure +x in the travel frame at
    # every step, however the command turns -- the omni tracking ratio's whole premise.
    dirs = np.linspace(-math.pi, math.pi, 25)
    moving = np.stack([0.24 * np.cos(dirs), 0.24 * np.sin(dirs), np.zeros_like(dirs)], axis=-1)
    travel = M.lin_vel_to_travel_frame(moving, dirs)
    np.testing.assert_allclose(travel[:, 0], 0.24, atol=1e-12)
    np.testing.assert_allclose(travel[:, 1], 0.0, atol=1e-12)


def test_lin_vel_to_travel_frame_identity_is_scalar_zero_only():
    v = np.array([0.4, -0.1, 0.05])
    assert M.lin_vel_to_travel_frame(v, 0.0) is v  # same object, as today
    # An all-zero ARRAY is not the identity fast path (it is a real per-step frame), but it must
    # still produce the same numbers.
    out = M.lin_vel_to_travel_frame(np.tile(v, (4, 1)), np.zeros(4))
    assert out is not v
    np.testing.assert_allclose(out, np.tile(v, (4, 1)), atol=0.0)


def test_rotate_xy_and_quat_helpers_accept_per_step_angles():
    rng = np.random.default_rng(11)
    xy = rng.normal(size=(15, 2))
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (15, 1))
    for angle in (0.0, 0.8, -math.pi / 2):
        assert np.array_equal(M._rotate_xy(xy, angle), M._rotate_xy(xy, np.full(15, angle)))
        assert np.array_equal(
            M._yaw_rotate_quat_wxyz(quat, angle), M._yaw_rotate_quat_wxyz(quat, np.full(15, angle))
        )
    per_step = np.linspace(0.0, math.pi, 15)
    rotated = M._rotate_xy(xy, per_step)
    for t, a in enumerate(per_step):
        np.testing.assert_allclose(rotated[t], M._rotate_xy(xy[t], float(a)), atol=1e-15)


# ---------------------------------------------------------------------------
# facing_metrics (omni task): the second command channel.
# ---------------------------------------------------------------------------
def test_facing_metrics_on_a_settling_error():
    n = 200
    err = np.concatenate([np.linspace(1.5, 0.0, 100), np.zeros(100)])
    wz = np.full(n, 0.2)
    wz_cmd = np.concatenate([np.full(100, 0.3), np.zeros(100)])
    out = M.facing_metrics(err, wz, DT, wz_cmd_t=wz_cmd)
    assert out["n_steps"] == n
    assert out["heading_error_max_abs"] == pytest.approx(1.5)
    assert out["heading_error_final_abs"] == pytest.approx(0.0)
    assert out["heading_error_rms"] == pytest.approx(float(np.sqrt(np.mean(err**2))))
    # Last step outside the 0.15 rad band is index 90 (err 1.5 -> 0 over 100 steps).
    last_outside = int(np.flatnonzero(np.abs(err) >= 0.15)[-1])
    assert out["settle_time_s"] == pytest.approx((last_outside + 1) * DT)
    assert out["n_turn_steps"] == 100
    assert out["yaw_rate_ratio"] == pytest.approx(0.2 / 0.3)


def test_facing_metrics_never_settling_is_none_not_zero():
    n = 50
    out = M.facing_metrics(np.full(n, 0.9), np.zeros(n), DT, wz_cmd_t=np.full(n, 0.3))
    assert out["settle_time_s"] is None
    assert out["heading_error_final_abs"] == pytest.approx(0.9)
    assert out["yaw_rate_ratio"] == pytest.approx(0.0)  # commanded but not turning: the record
    assert out["n_turn_steps"] == n


def test_facing_metrics_already_on_target_settles_at_zero():
    out = M.facing_metrics(np.full(30, 0.01), np.zeros(30), DT)
    assert out["settle_time_s"] == 0.0
    assert out["yaw_rate_ratio"] is None and out["n_turn_steps"] == 0  # no wz command given


def test_facing_metrics_ignores_steps_without_a_yaw_command():
    wz = np.array([0.0, 0.0, 0.3, 0.3])
    wz_cmd = np.array([0.0, 0.0, 0.3, 0.3])
    out = M.facing_metrics(np.zeros(4), wz, DT, wz_cmd_t=wz_cmd)
    assert out["n_turn_steps"] == 2
    assert out["yaw_rate_ratio"] == pytest.approx(1.0)


def test_facing_metrics_empty_window():
    out = M.facing_metrics(np.array([]), np.array([]), DT, wz_cmd_t=np.array([]))
    assert out["n_steps"] == 0 and out["heading_error_rms"] is None
    assert out["settle_time_s"] is None and out["yaw_rate_ratio"] is None
    assert out["yaw_progress_rate"] is None and out["net_yaw_rate_toward"] is None


# ---------------------------------------------------------------------------
# yaw_progress_rate / net_yaw_rate_toward (2026-09-17): the GATED turning quantities. The retired
# proxy (yaw_rate_ratio x ang_vel_max) divided two magnitudes, so it scored the gait's yaw wobble as
# turning and would have passed a non-turning head on the yaw-capability smoke. These tests pin the
# sign conventions that failure turned on.
# ---------------------------------------------------------------------------
N_HOLD = 400            # 8 s at DT -- the omni schedule's hold length
HOLD_S = N_HOLD * DT


def test_yaw_progress_rate_is_the_rate_toward_the_command():
    """Constant rotation with the command: progress == the achieved rate, both signs of command."""
    out = M.facing_metrics(np.zeros(N_HOLD), np.full(N_HOLD, 0.25), DT, wz_cmd_t=np.full(N_HOLD, 0.30))
    assert out["yaw_progress_rate"] == pytest.approx(0.25)
    # Commanded right turn, body turns right: still POSITIVE progress (toward the command).
    out = M.facing_metrics(np.zeros(N_HOLD), np.full(N_HOLD, -0.25), DT, wz_cmd_t=np.full(N_HOLD, -0.30))
    assert out["yaw_progress_rate"] == pytest.approx(0.25)


def test_yaw_progress_rate_cancels_wobble_where_the_ratio_does_not():
    """The 2026-09-17 ground truth, synthesised: |wz| large every step, net rotation zero.

    ``wz`` alternates +-0.284 rad/s (mean |wz| 0.284) against a command pinned at +0.30, which is
    ``yaw_rate_ratio`` 0.946 -- the exact number the retired proxy read on ``turn_l90`` (x 0.3 =
    0.284 rad/s, 5.7x the 0.05 floor). The signed progress is 0.
    """
    wz = 0.284 * np.where(np.arange(N_HOLD) % 2 == 0, 1.0, -1.0)
    out = M.facing_metrics(np.zeros(N_HOLD), wz, DT, wz_cmd_t=np.full(N_HOLD, 0.30))
    assert out["yaw_progress_rate"] == pytest.approx(0.0, abs=1e-12)
    assert out["yaw_rate_ratio"] == pytest.approx(0.284 / 0.30)      # the wobble-blind number
    assert abs(out["yaw_progress_rate"]) < 0.05 < out["yaw_rate_ratio"] * 0.3


def test_yaw_progress_rate_is_negative_when_the_body_turns_the_wrong_way():
    out = M.facing_metrics(np.zeros(N_HOLD), np.full(N_HOLD, -0.12), DT, wz_cmd_t=np.full(N_HOLD, 0.30))
    assert out["yaw_progress_rate"] == pytest.approx(-0.12)
    # Mirror: commanded right, body turns left.
    out = M.facing_metrics(np.zeros(N_HOLD), np.full(N_HOLD, 0.12), DT, wz_cmd_t=np.full(N_HOLD, -0.30))
    assert out["yaw_progress_rate"] == pytest.approx(-0.12)


def test_yaw_progress_rate_ignores_steps_without_a_yaw_command():
    """Only commanded-turn steps count; a hold that never commands a turn has no progress number."""
    wz = np.array([0.5, -0.5, 0.2, 0.2])          # the zero-command steps would drag a plain mean to 0.1
    cmd = np.array([0.0, 0.0, 0.3, 0.3])
    out = M.facing_metrics(np.zeros(4), wz, DT, wz_cmd_t=cmd)
    assert out["n_turn_steps"] == 2 and out["yaw_progress_rate"] == pytest.approx(0.2)
    out = M.facing_metrics(np.zeros(4), wz, DT, wz_cmd_t=np.zeros(4))
    assert out["n_turn_steps"] == 0 and out["yaw_progress_rate"] is None
    assert M.facing_metrics(np.zeros(4), wz, DT)["yaw_progress_rate"] is None   # no command logged


@pytest.mark.parametrize("err0, err1, expect_sign", [
    (1.5, 0.0, +1),      # left turn, error closes  -> turned TOWARD the target
    (0.5, 2.0, -1),      # left turn, error opens    -> turned AWAY
    (-1.5, 0.0, +1),     # right turn, error closes  -> toward (the sign(err[0]) factor)
    (-0.5, -2.0, -1),    # right turn, error opens   -> away
])
def test_net_yaw_rate_toward_signs_on_a_linear_error(err0, err1, expect_sign):
    err = np.linspace(err0, err1, N_HOLD)
    out = M.facing_metrics(err, np.zeros(N_HOLD), DT, wz_cmd_t=np.full(N_HOLD, 0.30))
    expect = -(err1 - err0) / HOLD_S * np.sign(err0)
    assert out["net_yaw_rate_toward"] == pytest.approx(expect)
    assert np.sign(out["net_yaw_rate_toward"]) == expect_sign
    assert abs(out["net_yaw_rate_toward"]) == pytest.approx(abs(err1 - err0) / HOLD_S)


def test_net_yaw_rate_toward_unwraps_across_pi():
    """A wrapped error series must not read as a huge rotation: the body drifts 0.6 rad AWAY from the
    target and the error crosses +pi, so the raw end-minus-start would be -6.08 rad (a spurious
    +0.76 rad/s TOWARD); ``unwrap`` recovers the true -0.075 rad/s."""
    raw = np.linspace(3.0, 3.6, N_HOLD)
    err = (raw + np.pi) % (2 * np.pi) - np.pi                 # wrap_to_pi, as the harness logs it
    assert err.min() < -3.0 < err.max()                       # the series really does wrap
    out = M.facing_metrics(err, np.zeros(N_HOLD), DT, wz_cmd_t=np.full(N_HOLD, 0.30))
    assert out["net_yaw_rate_toward"] == pytest.approx(-0.6 / HOLD_S)
    naive = -(err[-1] - err[0]) / HOLD_S * np.sign(err[0])
    assert naive > 0.7 > 0.0 > out["net_yaw_rate_toward"]      # what the un-unwrapped form would say


def test_net_yaw_rate_toward_guards_short_windows_and_a_moving_target():
    one = M.facing_metrics(np.array([0.5]), np.array([0.1]), DT, wz_cmd_t=np.array([0.3]))
    assert one["net_yaw_rate_toward"] is None and one["yaw_progress_rate"] == pytest.approx(0.1)
    err = np.linspace(1.5, 0.0, N_HOLD)
    spans_holds = M.facing_metrics(err, np.zeros(N_HOLD), DT, wz_cmd_t=np.full(N_HOLD, 0.3),
                                   constant_target=False)
    assert spans_holds["net_yaw_rate_toward"] is None          # episode-level: target is not constant
    assert spans_holds["yaw_progress_rate"] == pytest.approx(0.0)   # progress needs no constant target


def test_net_yaw_rate_toward_matches_a_real_turn_geometry():
    """A left 90 deg turn at 0.2 rad/s: err goes pi/2 -> 0 in 7.85 s of the 8 s hold, then holds."""
    n_turn = int(round((np.pi / 2) / 0.2 / DT))
    err = np.concatenate([np.linspace(np.pi / 2, 0.0, n_turn), np.zeros(N_HOLD - n_turn)])
    wz = np.concatenate([np.full(n_turn, 0.2), np.zeros(N_HOLD - n_turn)])
    out = M.facing_metrics(err, wz, DT, wz_cmd_t=np.full(N_HOLD, 0.30))
    # Net rate = the whole turn spread over the whole hold; progress = the mean achieved rate.
    assert out["net_yaw_rate_toward"] == pytest.approx((np.pi / 2) / HOLD_S, rel=1e-3)
    assert out["yaw_progress_rate"] == pytest.approx(0.2 * n_turn / N_HOLD, rel=1e-6)
    assert out["net_yaw_rate_toward"] > 0.05 and out["yaw_progress_rate"] > 0.05


# ---------------------------------------------------------------------------
# report.py on a WORLD-frame row (omni): the command is rotated into the travel frame so the
# unchanged tracking_metrics still answers "what fraction of the commanded speed did it make",
# and each hold gains a whole-hold facing entry.
# ---------------------------------------------------------------------------
def _world_raw_and_scenario(travel_dir: float = math.pi / 4):
    """A synthetic omni episode: one moving hold travelling at ``travel_dir``, one 90 deg turn."""
    from gait_eval import report as R  # noqa: PLC0415
    from gait_eval.schedule import Hold, Scenario, compile_schedule  # noqa: PLC0415

    scenario = Scenario(
        id="synthetic_world",
        task="Isaac-Crab-Hex-Omni-Walk-Play-v0",
        checkpoint="",
        schedule=[
            Hold(hold_s=6.0, speed=0.24, dir=travel_dir, face=0.0, label="move"),
            Hold(hold_s=7.0, speed=0.0, dir=0.0, face=math.pi / 2, label="turn_l90"),
        ],
        episodes=1,
        env_seed=1,
        speed_ellipse={"fwd": 0.35, "lat": 0.25, "back": 0.25},
    )
    compiled = compile_schedule(scenario, dt=DT, num_envs=1)
    T = compiled.n_steps
    moving = compiled.world_speed[:, 0] > 0.0

    # Body command: the moving hold travels at `travel_dir` (body == world, the robot faces 0);
    # the turn hold has no translation and a yaw-rate command that decays as the error closes.
    cmd = np.zeros((T, 1, 3))
    cmd[moving, 0, 0] = 0.24 * math.cos(travel_dir)
    cmd[moving, 0, 1] = 0.24 * math.sin(travel_dir)
    heading_err = np.zeros((T, 1))
    turn_idx = np.flatnonzero(~moving)
    # error decays from pi/2 to ~0 over the first 5 s of the turn hold, then holds.
    ramp = np.clip(np.arange(turn_idx.size) / (5.0 / DT), 0.0, 1.0)
    heading_err[turn_idx, 0] = (math.pi / 2) * (1.0 - ramp)
    cmd[turn_idx, 0, 2] = np.clip(heading_err[turn_idx, 0] * 0.8, -0.3, 0.3)

    lin = np.zeros((T, 1, 3))
    lin[:, 0, :2] = cmd[:, 0, :2]  # perfectly on command
    ang = np.zeros((T, 1, 3))
    ang[:, 0, 2] = cmd[:, 0, 2]

    contact = _tripod_contact(T, 5)
    raw = {
        "foot_force_norm": np.where(contact, 400.0, 0.0)[:, None, :],
        "cmd_applied": cmd,
        "root_lin_vel_b": lin,
        "root_ang_vel_b": ang,
        "root_quat_w": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T, 1))[:, None, :],
        "root_pos_w": np.zeros((T, 1, 3)),
        "foot_pos_w": np.zeros((T, 1, 6, 3)),
        "foot_lin_vel_w": np.zeros((T, 1, 6, 3)),
        "foot_ang_vel_w": np.zeros((T, 1, 6, 3)),
        "terrain_z": np.zeros((T, 1, 6)),
        "actions": np.zeros((T, 1, 4)),
        "joint_vel": np.zeros((T, 1, 4)),
        "crab_failure": np.zeros((T, 1), dtype=bool),
        "heading_err": heading_err,
        # what the harness logs: the direction of the body command at each step
        "travel_yaw": np.arctan2(cmd[:, :, 1], cmd[:, :, 0]),
    }
    return R, raw, compiled, scenario, T


def test_world_row_tracking_is_measured_along_the_commanded_direction():
    """A robot travelling exactly on a 45 deg command must read ratio 1.0, not cos(45 deg)."""
    R, raw, compiled, scenario, T = _world_raw_and_scenario()
    ep = R.score_run(
        raw, compiled=compiled, scenario=scenario, dt=DT,
        term_reason=["schedule_complete"], n_steps_env=[T], travel_yaw=raw["travel_yaw"],
    )[0]
    move = ep["holds"]["move"]["tracking"]
    assert move["vx"]["cmd_mean"] == pytest.approx(0.24, abs=1e-9)
    assert move["vx"]["actual_mean"] == pytest.approx(0.24, abs=1e-9)
    assert move["vx"]["ratio"] == pytest.approx(1.0, abs=1e-9)
    assert move["vy"]["actual_mean"] == pytest.approx(0.0, abs=1e-9)
    # ratio.n > 0 iff speed > clip: the standing turn hold is a commanded stop.
    assert ep["holds"]["turn_l90"]["tracking"]["vx"]["ratio"] is None
    assert ep["tracking_ratio"] == pytest.approx(1.0, abs=1e-9)


def test_world_row_facing_is_scored_over_the_whole_hold():
    R, raw, compiled, scenario, T = _world_raw_and_scenario()
    episodes = R.score_run(
        raw, compiled=compiled, scenario=scenario, dt=DT,
        term_reason=["schedule_complete"], n_steps_env=[T], travel_yaw=raw["travel_yaw"],
    )
    ep = episodes[0]
    turn = ep["holds"]["turn_l90"]["facing"]
    # Whole hold, not steady-masked: the settle window is 1 s, the hold 7 s.
    assert turn["n_steps"] == int(round(7.0 / DT))
    assert turn["heading_error_max_abs"] == pytest.approx(math.pi / 2, abs=1e-6)
    assert turn["heading_error_final_abs"] == pytest.approx(0.0, abs=1e-6)
    assert 4.0 < turn["settle_time_s"] < 6.0
    assert turn["n_turn_steps"] > 0
    assert turn["yaw_rate_ratio"] == pytest.approx(1.0, abs=1e-9)
    # The gated turning numbers on a head that DOES turn: the body tracks the yaw command exactly
    # here, so progress == the mean commanded rate over the turning steps, and the net rotation is
    # the whole pi/2 turn spread over the 7 s hold.
    wz_cmd = raw["cmd_applied"][:, 0, 2]
    turning = np.abs(wz_cmd) > 1e-6
    assert turn["yaw_progress_rate"] == pytest.approx(float(wz_cmd[turning].mean()), abs=1e-9)
    assert turn["net_yaw_rate_toward"] == pytest.approx((math.pi / 2) / 7.0, abs=1e-6)
    assert ep["holds"]["move"]["facing"]["heading_error_rms"] == pytest.approx(0.0)
    assert ep["holds"]["move"]["facing"]["yaw_progress_rate"] is None       # no yaw command commanded
    assert ep["facing_error_rms"] == pytest.approx(ep["facing"]["heading_error_rms"])
    # Episode level spans two holds with different facing targets: net is refused, not guessed.
    assert ep["facing"]["net_yaw_rate_toward"] is None
    assert ep["facing"]["yaw_progress_rate"] is not None

    agg = R.aggregate(episodes)
    assert set(agg["facing_by_hold"]) == {"move", "turn_l90"}
    assert agg["facing_by_hold"]["turn_l90"]["heading_error_final_abs"]["median"] == pytest.approx(0.0, abs=1e-6)
    assert agg["facing_by_hold"]["turn_l90"]["yaw_progress_rate"]["median"] == pytest.approx(
        turn["yaw_progress_rate"], abs=1e-9)
    assert agg["facing_by_hold"]["turn_l90"]["net_yaw_rate_toward"]["median"] == pytest.approx(
        turn["net_yaw_rate_toward"], abs=1e-9)
    assert agg["facing_by_hold"]["move"]["yaw_progress_rate"]["n"] == 0     # unmeasured, not 0.0
    assert agg["facing_error_rms"]["n"] == 1
    text = R.summary_text({"scenario_id": "s", "task": "t", "command_override_max_deviation": 0.0}, episodes)
    assert "facing_error_rms" in text and "turn_l90" in text and "yaw_rate_ratio" in text
    assert "progress=" in text and "net=" in text


def test_world_row_stride_reads_along_the_body_command():
    """cmd_applied is a true body vector on world rows, so the stride frame is the body yaw."""
    R, raw, compiled, scenario, T = _world_raw_and_scenario(travel_dir=math.pi / 2)
    # feet displace along world +y, the commanded direction, while the body faces world +x
    raw = dict(raw)
    pos = np.zeros((T, 1, 6, 3))
    pos[:, 0, 0, 1] = np.arange(T) * 0.004
    raw["foot_pos_w"] = pos
    ep = R.score_run(
        raw, compiled=compiled, scenario=scenario, dt=DT,
        term_reason=["schedule_complete"], n_steps_env=[T], travel_yaw=raw["travel_yaw"],
    )[0]
    fl = ep["stride"]["per_foot"]["FL_Footpad"]
    assert fl["n_strides"] > 0
    assert fl["along_command"]["mean"] == pytest.approx(fl["magnitude"]["mean"], rel=1e-6)


def test_body_rows_are_unaffected_by_the_world_branch():
    """A body scenario scores identically whether or not heading_err happens to be in raw."""
    from gait_eval import report as R
    from gait_eval.schedule import Hold, Scenario, compile_schedule

    scenario = Scenario(
        id="body", task="Isaac-Crab-Hex-Flat-Walk-v0", checkpoint="",
        schedule=[Hold(hold_s=3.0, vx=0.3, label="hold0")], episodes=1, env_seed=1,
    )
    compiled = compile_schedule(scenario, dt=DT, num_envs=1)
    T = compiled.n_steps
    contact = _tripod_contact(T, 5)
    raw = {
        "foot_force_norm": np.where(contact, 400.0, 0.0)[:, None, :],
        "cmd_applied": compiled.cmd.astype(np.float64),
        "root_lin_vel_b": np.tile(np.array([0.3, 0.0, 0.0]), (T, 1))[:, None, :],
        "root_ang_vel_b": np.zeros((T, 1, 3)),
        "root_quat_w": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T, 1))[:, None, :],
        "root_pos_w": np.zeros((T, 1, 3)),
        "foot_pos_w": np.zeros((T, 1, 6, 3)),
        "foot_lin_vel_w": np.zeros((T, 1, 6, 3)),
        "foot_ang_vel_w": np.zeros((T, 1, 6, 3)),
        "terrain_z": np.zeros((T, 1, 6)),
        "actions": np.zeros((T, 1, 4)),
        "joint_vel": np.zeros((T, 1, 4)),
        "crab_failure": np.zeros((T, 1), dtype=bool),
    }
    kwargs = dict(compiled=compiled, scenario=scenario, dt=DT,
                  term_reason=["schedule_complete"], n_steps_env=[T])
    base = R.score_run(raw, **kwargs)
    with_extra = R.score_run(dict(raw, heading_err=np.zeros((T, 1))), **kwargs)
    assert base == with_extra
    assert "facing_by_hold" not in R.aggregate(base)


# ---------------------------------------------------------------------------
# facing_closed / heading_error_initial_abs (2026-09-18): the SCALE-FREE facing quantity.
# heading_error_final_abs answers "where did it end up", which is only comparable between holds
# that started at the same error -- the candidate ladder compares holds and recipes that do not.
# ---------------------------------------------------------------------------
def test_facing_closed_is_the_fraction_of_the_initial_error_closed():
    n = 200
    # 1.5 rad -> 0.3 rad: four fifths of the turn closed.
    out = M.facing_metrics(np.linspace(1.5, 0.3, n), np.full(n, 0.2), DT, wz_cmd_t=np.full(n, 0.3))
    assert out["heading_error_initial_abs"] == pytest.approx(1.5)
    assert out["facing_closed"] == pytest.approx(1.0 - 0.3 / 1.5)
    assert out["n_uninformative"] == 0
    # A right turn is the mirror image, not a sign flip: both sides are magnitudes.
    mirror = M.facing_metrics(np.linspace(-1.5, -0.3, n), np.full(n, -0.2), DT, wz_cmd_t=np.full(n, -0.3))
    assert mirror["facing_closed"] == pytest.approx(out["facing_closed"])


def test_facing_closed_is_none_when_there_was_no_turn_to_close():
    """A hold that starts on target measures nothing: dividing a wobble by ~0 manufactures a
    verdict. UNINFORMATIVE and counted, never 0.0 (which reads as "closed nothing")."""
    on_target = M.facing_metrics(np.full(50, 0.02), np.zeros(50), DT, wz_cmd_t=np.zeros(50))
    assert on_target["facing_closed"] is None and on_target["n_uninformative"] == 1
    assert on_target["heading_error_initial_abs"] == pytest.approx(0.02)
    # Exactly at the floor is informative; a hair under is not.
    at = M.facing_metrics(np.full(50, M.FACING_CLOSED_MIN_INITIAL_RAD), np.zeros(50), DT)
    under = M.facing_metrics(np.full(50, M.FACING_CLOSED_MIN_INITIAL_RAD - 1e-9), np.zeros(50), DT)
    assert at["facing_closed"] == pytest.approx(0.0) and at["n_uninformative"] == 0
    assert under["facing_closed"] is None and under["n_uninformative"] == 1
    empty = M.facing_metrics(np.array([]), np.array([]), DT)
    assert empty["facing_closed"] is None and empty["heading_error_initial_abs"] is None
    assert empty["n_uninformative"] == 0            # nothing was measured, so nothing was DISCARDED


def test_facing_closed_is_clipped_and_refused_across_a_moving_target():
    """A body that spins away from the target would otherwise own a median on its own; and across
    holds the target MOVES, so "the initial error" belongs to a different command than the final
    one -- the same reason net_yaw_rate_toward is refused there."""
    opened = M.facing_metrics(np.linspace(0.5, 3.0, N_HOLD), np.zeros(N_HOLD), DT)
    assert opened["facing_closed"] == pytest.approx(-1.0)          # clipped, not -5.0
    held = M.facing_metrics(np.full(N_HOLD, 0.8), np.zeros(N_HOLD), DT)
    assert held["facing_closed"] == pytest.approx(0.0)             # ended where it started
    episode = M.facing_metrics(np.linspace(1.5, 0.0, N_HOLD), np.zeros(N_HOLD), DT,
                               constant_target=False)
    assert episode["facing_closed"] is None and episode["n_uninformative"] == 0
    assert episode["heading_error_initial_abs"] == pytest.approx(1.5)   # still recorded


def test_world_row_carries_the_closed_fraction_through_to_the_aggregate():
    """End to end on a world row: the turn hold closes its whole error, the travel hold has none to
    close, and both keys reach ``aggregate``'s facing block (body rows still have no facing at all).
    """
    R, raw, compiled, scenario, T = _world_raw_and_scenario()
    episodes = R.score_run(
        raw, compiled=compiled, scenario=scenario, dt=DT,
        term_reason=["schedule_complete"], n_steps_env=[T], travel_yaw=raw["travel_yaw"],
    )
    turn = episodes[0]["holds"]["turn_l90"]["facing"]
    assert turn["heading_error_initial_abs"] == pytest.approx(math.pi / 2, abs=1e-6)
    assert turn["facing_closed"] == pytest.approx(1.0, abs=1e-6)     # error pi/2 -> ~0
    move = episodes[0]["holds"]["move"]["facing"]
    assert move["facing_closed"] is None and move["n_uninformative"] == 1
    agg = R.aggregate(episodes)
    assert agg["facing_by_hold"]["turn_l90"]["facing_closed"]["median"] == pytest.approx(1.0, abs=1e-6)
    assert agg["facing_by_hold"]["turn_l90"]["heading_error_initial_abs"]["median"] == pytest.approx(
        math.pi / 2, abs=1e-6)
    # the uninformative hold: measured for error, no closed fraction (n 0, not a 0.0 median)
    assert agg["facing_by_hold"]["move"]["facing_closed"]["n"] == 0
    assert agg["facing_by_hold"]["move"]["facing_closed"]["median"] is None
    assert agg["facing_by_hold"]["move"]["heading_error_rms"]["n"] == 1
    text = R.summary_text({"scenario_id": "s", "task": "t", "command_override_max_deviation": 0.0}, episodes)
    assert "closed=1.000" in text and "initial=1.571" in text


# ---------------------------------------------------------------------------
# per-category eval (plan D3, 2026-09-21): per-hold motion / travel direction on world rows, and
# the category-suite aggregate -- own-hold survival and by_category pooled over both hands.
# ---------------------------------------------------------------------------
def test_travel_direction_metrics_reads_the_angle_the_drift_and_the_net_speed():
    v = np.tile(np.array([0.2, 0.05]), (100, 1))
    out = M.travel_direction_metrics(v, 0.25)
    assert out["direction_error_rad"] == pytest.approx(math.atan2(0.05, 0.2))
    assert out["cross_track_ratio"] == pytest.approx(0.05 / 0.25)
    assert out["travel_speed_m_s"] == pytest.approx(math.hypot(0.2, 0.05))
    # A drift to the right is the same size of error (a magnitude), and a per-step command is averaged.
    mirror = M.travel_direction_metrics(v * np.array([1.0, -1.0]), np.full(100, 0.25))
    assert mirror["direction_error_rad"] == pytest.approx(out["direction_error_rad"])
    assert mirror["cross_track_ratio"] == pytest.approx(out["cross_track_ratio"])
    backwards = M.travel_direction_metrics(np.tile(np.array([-0.2, 0.0]), (10, 1)), 0.2)
    assert backwards["direction_error_rad"] == pytest.approx(math.pi)
    # Wobble around a standstill has no direction: None, never a noisy angle.
    still = M.travel_direction_metrics(np.tile(np.array([0.001, 0.005]), (10, 1)), 0.22)
    assert still["direction_error_rad"] is None
    assert still["cross_track_ratio"] == pytest.approx(0.005 / 0.22)
    assert M.travel_direction_metrics(v, 0.0)["cross_track_ratio"] is None
    assert M.travel_direction_metrics(np.zeros((0, 2)), 0.25) is None


def _yaw_quat(yaw: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)], axis=-1)


def test_hold_motion_metrics_on_a_known_trajectory():
    n = 101
    # out 1 m along +x, then back half a metre: net drift 0.5 m over a 1.5 m path
    x = np.concatenate([np.linspace(0.0, 1.0, 51), np.linspace(1.0, 0.5, 51)[1:]])
    pos = np.stack([x, np.zeros(n), np.full(n, 0.3)], axis=-1)
    yaw = np.linspace(0.0, 3.0 * math.pi, n)          # > pi: the drift must be unwrapped
    vel = np.tile(np.array([0.3, 0.4, 0.1]), (n, 1))  # planar speed 0.5 (vz ignored)
    contact = np.ones((n, 6), dtype=bool)
    contact[10:20, 0] = False                         # FL lifts twice, touches down twice
    contact[40:50, 0] = False
    contact[60:, 3] = False                           # MR lifts once, never lands
    out = M.hold_motion_metrics(pos, _yaw_quat(yaw), vel, contact, DT)
    assert out["speed_abs_mean_m_s"] == pytest.approx(0.5)
    assert out["pos_drift_m"] == pytest.approx(0.5)
    assert out["path_len_m"] == pytest.approx(1.5)
    assert out["yaw_drift_rad"] == pytest.approx(3.0 * math.pi)
    assert out["foot_lift_count"] == 3
    assert out["contact_churn_per_s"] == pytest.approx(5 / (n * DT))
    assert out["n_steps"] == n
    back = M.hold_motion_metrics(pos, _yaw_quat(-yaw), vel, contact, DT)
    assert back["yaw_drift_rad"] == pytest.approx(-3.0 * math.pi)    # signed: + = left
    one = M.hold_motion_metrics(pos[:1], _yaw_quat(yaw[:1]), vel[:1], contact[:1], DT)
    assert one["path_len_m"] == 0.0 and one["foot_lift_count"] == 0 and one["contact_churn_per_s"] == 0.0
    assert M.hold_motion_metrics(pos[:0], _yaw_quat(yaw[:0]), vel[:0], contact[:0], DT) is None


SUITE_E = 4
# per-env tracking ratio (achieved/commanded speed) of the synthetic suite, env_index = wave*E + env:
# wave 0 fwd, wave 1 lat_l, wave 2 lat_r, wave 3 turn_l90.
SUITE_RATIOS = [1.0] * 4 + [0.1, 0.2, 0.3, 0.4] + [0.9] * 4 + [0.0] * 4


def _suite_run(n_steps_of=None, term_of=None, trained=("fwd", "side")):
    """A synthetic E = 4, W = 4 category suite (fwd, lat_l, lat_r, turn_l90) scored end to end."""
    from gait_eval import report as R  # noqa: PLC0415
    from gait_eval import schedule as S  # noqa: PLC0415

    suite = S.CategorySuite(
        settle_hold_s=2.0,
        settle_label="settle",
        hold_s=8.0,
        categories=[
            S.Category("fwd", "translate", [S.CategoryHold("fwd", 0.25, 0.0, 0.0)]),
            S.Category("side", "translate", [
                S.CategoryHold("lat_l", 0.22, math.pi / 2, 0.0, "l"),
                S.CategoryHold("lat_r", 0.22, -math.pi / 2, 0.0, "r"),
            ]),
            S.Category("turn", "turn", [S.CategoryHold("turn_l90", 0.0, 0.0, math.pi / 2)]),
        ],
    )
    scenario = S.Scenario(
        id="synthetic_suite", task="Isaac-Crab-Hex-Omni-Walk-Play-v0", checkpoint="",
        schedule=S.suite_schedule(suite), episodes=SUITE_E, kind=S.SUITE_KIND, suite=suite, waves=4,
        speed_ellipse={"fwd": 0.35, "lat": 0.25, "back": 0.25},
    )
    S.validate_scenario(scenario)
    compiled = S.stack_waves([S.compile_wave(scenario, k, dt=DT) for k in range(4)])
    T, N = compiled.n_steps, 4 * SUITE_E
    speed, direction = compiled.world_speed, compiled.world_dir     # the body faces 0: body == world
    cmd = np.zeros((T, N, 3))
    cmd[..., 0] = speed * np.cos(direction)
    cmd[..., 1] = speed * np.sin(direction)
    heading_err = np.vectorize(lambda a: math.remainder(a, 2 * math.pi))(compiled.world_face)
    cmd[..., 2] = np.clip(heading_err * 0.8, -0.3, 0.3)
    lin = cmd * np.asarray(SUITE_RATIOS)[None, :, None]
    lin[..., 2] = 0.0
    pos = np.cumsum(lin, axis=0) * DT
    contact = _tripod_contact(T, 5)
    raw = {
        "foot_force_norm": np.repeat(np.where(contact, 400.0, 0.0)[:, None, :], N, axis=1),
        "cmd_applied": cmd,
        "root_lin_vel_b": lin,
        "root_ang_vel_b": np.zeros((T, N, 3)),
        "root_quat_w": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T, N, 1)),
        "root_pos_w": pos,
        "foot_pos_w": np.zeros((T, N, 6, 3)),
        "foot_lin_vel_w": np.zeros((T, N, 6, 3)),
        "foot_ang_vel_w": np.zeros((T, N, 6, 3)),
        "terrain_z": np.zeros((T, N, 6)),
        "actions": np.zeros((T, N, 4)),
        "joint_vel": np.zeros((T, N, 4)),
        "crab_failure": np.zeros((T, N), dtype=bool),
        "heading_err": heading_err,
        "travel_yaw": np.arctan2(cmd[..., 1], cmd[..., 0]),
    }
    n_steps = [T] * N if n_steps_of is None else [n_steps_of.get(i, T) for i in range(N)]
    terms = ["schedule_complete"] * N if term_of is None else [term_of.get(i, "schedule_complete") for i in range(N)]
    episodes = R.score_run(raw, compiled=compiled, scenario=scenario, dt=DT, term_reason=terms,
                           n_steps_env=n_steps, travel_yaw=raw["travel_yaw"])
    run_meta = {
        "scenario_id": scenario.id, "task": scenario.task, "command_override_max_deviation": 0.0,
        **S.suite_run_meta(scenario, dt=DT, num_envs=SUITE_E, wave_seeds=[S.wave_seed(scenario, k) for k in range(4)],
                           trained_categories=list(trained)),
    }
    return R, episodes, run_meta, compiled


def test_world_row_holds_carry_direction_and_motion():
    # A legacy world row: every hold carries motion; direction only on the moving hold.
    R, raw, compiled, scenario, T = _world_raw_and_scenario(travel_dir=math.pi / 4)
    ep = R.score_run(raw, compiled=compiled, scenario=scenario, dt=DT,
                     term_reason=["schedule_complete"], n_steps_env=[T], travel_yaw=raw["travel_yaw"])[0]
    move, turn = ep["holds"]["move"], ep["holds"]["turn_l90"]
    assert move["motion"]["n_steps"] == int(round(6.0 / DT))           # the WHOLE hold, not the steady window
    assert turn["motion"]["n_steps"] == int(round(7.0 / DT))
    assert move["motion"]["speed_abs_mean_m_s"] == pytest.approx(0.24)
    assert "direction" not in turn                                     # commanded speed 0: none at all
    assert move["direction"]["direction_error_rad"] == pytest.approx(0.0, abs=1e-9)   # dead on the 45-deg line
    assert move["direction"]["cross_track_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert move["direction"]["travel_speed_m_s"] == pytest.approx(0.24)
    agg = R.aggregate([ep])
    assert set(agg["motion_by_hold"]) == {"move", "turn_l90"} and set(agg["direction_by_hold"]) == {"move"}
    assert agg["motion_by_hold"]["move"]["speed_abs_mean_m_s"]["median"] == pytest.approx(0.24)
    assert "suite" not in agg                                          # no suite argument, no suite block

    # A suite row: the settle and the standing turn carry no direction; a hold the episode never
    # visited carries an entry with nothing measured (motion None), never numbers from another wave.
    R, episodes, _, _ = _suite_run()
    fwd_ep = episodes[0]
    assert "direction" not in fwd_ep["holds"]["settle"] and fwd_ep["holds"]["settle"]["motion"]["n_steps"] == 100
    assert "direction" not in fwd_ep["holds"]["turn_l90"]
    assert fwd_ep["holds"]["fwd"]["direction"]["direction_error_rad"] == pytest.approx(0.0, abs=1e-9)
    # first to last sample of the 400-step hold: 399 steps of 0.25 m/s
    assert fwd_ep["holds"]["fwd"]["motion"]["pos_drift_m"] == pytest.approx(0.25 * 399 * DT, rel=1e-9)
    assert fwd_ep["holds"]["lat_l"]["motion"] is None and fwd_ep["holds"]["lat_l"]["direction"] is None
    lat_ep = episodes[SUITE_E]                                          # wave 1 env 0, ratio 0.1
    assert lat_ep["holds"]["lat_l"]["direction"]["travel_speed_m_s"] == pytest.approx(0.022)


def test_suite_aggregate_pools_both_hands_and_scores_own_hold_survival():
    T = 75 + 100 + 400
    # env 12 = turn_l90 env 0 falls mid-hold; env 13 = turn_l90 env 1 falls ON the wave's last step;
    # env 1 = fwd env 1 falls ONE step before its own hold ends (pins both sides of the >= boundary)
    R, episodes, run_meta, compiled = _suite_run(
        n_steps_of={1: T - 1, 12: 300, 13: T}, term_of={1: "fall", 12: "fall", 13: "fall"}
    )
    assert compiled.n_steps == T
    assert [(e["wave"], e["hold"], e["category"], e["hand"]) for e in episodes[::SUITE_E]] == [
        (0, "fwd", "fwd", None), (1, "lat_l", "side", "l"), (2, "lat_r", "side", "r"), (3, "turn_l90", "turn", None),
    ]
    agg = R.aggregate(episodes, suite=R.suite_aggregate_arg(run_meta))
    sb = agg["suite"]
    assert set(sb) == {"waves", "wave_seeds", "episodes_total", "dt", "prologue_s", "settle_hold_s", "hold_s",
                       "hold_labels", "hold_of_wave", "category_of_hold", "hand_of_hold", "n_by_hold",
                       "survival_by_hold", "trained_categories", "by_category"}
    assert sb["waves"] == 4 and sb["episodes_total"] == 16 and sb["wave_seeds"] == [1_000_003 + k for k in range(4)]
    assert sb["hold_labels"] == ["settle", "fwd", "lat_l", "lat_r", "turn_l90"]
    assert sb["hold_of_wave"] == ["fwd", "lat_l", "lat_r", "turn_l90"]
    assert sb["category_of_hold"] == {"fwd": "fwd", "lat_l": "side", "lat_r": "side", "turn_l90": "turn"}
    assert sb["hand_of_hold"]["lat_r"] == "r" and sb["hand_of_hold"]["turn_l90"] is None
    assert sb["n_by_hold"] == {"fwd": 4, "lat_l": 4, "lat_r": 4, "turn_l90": 4}
    assert sb["trained_categories"] == ["fwd", "side"]
    # OWN-hold survival: each fall touches no other hold, a fall on the last step survived, and a
    # fall one step short of the own hold's end did not.
    assert sb["survival_by_hold"] == {"fwd": 0.75, "lat_l": 1.0, "lat_r": 1.0, "turn_l90": 0.75}
    # ...while the legacy pooled completion counts all three falls (WATCH-only on a suite row).
    assert agg["schedule_completion_rate"] == pytest.approx(13 / 16)
    # side: the POOLED median of 2 x 4 episodes (0.65), not the mean of the hand medians (0.575).
    side = sb["by_category"]["side"]
    assert side["ratio"]["n"] == 8
    assert side["ratio"]["median"] == pytest.approx(0.65)
    assert side["by_hand"]["l"]["ratio"]["median"] == pytest.approx(0.25)
    assert side["by_hand"]["r"]["ratio"]["median"] == pytest.approx(0.9)
    assert side["achieved_speed"]["median"] == pytest.approx(0.65 * 0.22)
    assert side["direction_error"]["n"] == 8 and side["tripod"]["n"] == 8
    assert sb["by_category"]["fwd"]["ratio"]["median"] == pytest.approx(1.0)
    assert sb["by_category"]["fwd"]["by_hand"] is None
    turn = sb["by_category"]["turn"]
    assert turn["ratio"]["n"] == 0 and turn["direction_error"]["n"] == 0     # a standing turn: no speed keys
    assert turn["facing_closed"]["n"] == 4 and turn["heading_final"]["n"] == 4
    assert set(turn) == set(R.CATEGORY_METRIC_SOURCES) | {"by_hand"}
    # the per-hold view is keyed by label and never mixes in the unvisited holds of other waves
    assert agg["tracking_by_hold"]["lat_l"]["ratio"]["n"] == 4
    text = R.summary_text(run_meta, episodes)
    assert "by category" in text and "trained: fwd, side" in text and "turn_l90=0.75" in text
    json.dumps(M.json_safe(agg))


def test_category_value_excludes_low_confidence_tripod():
    from gait_eval import report as R

    assert R._category_value({"tripod": {"tripod_score": 0.9, "low_confidence": True}}, "tripod") is None
    assert R._category_value({"tripod": {"tripod_score": 0.9, "low_confidence": False}}, "tripod") == 0.9
    assert R._category_value({}, "tripod") is None


def test_aggregate_has_no_suite_block_on_body_rows():
    from gait_eval import report as R
    from gait_eval.schedule import Hold, Scenario, compile_schedule

    scenario = Scenario(
        id="body", task="Isaac-Crab-Hex-Flat-Walk-v0", checkpoint="",
        schedule=[Hold(hold_s=3.0, vx=0.3, label="hold0")], episodes=1, env_seed=1,
    )
    compiled = compile_schedule(scenario, dt=DT, num_envs=1)
    T = compiled.n_steps
    contact = _tripod_contact(T, 5)
    raw = {
        "foot_force_norm": np.where(contact, 400.0, 0.0)[:, None, :],
        "cmd_applied": compiled.cmd.astype(np.float64),
        "root_lin_vel_b": np.tile(np.array([0.3, 0.0, 0.0]), (T, 1))[:, None, :],
        "root_ang_vel_b": np.zeros((T, 1, 3)),
        "root_quat_w": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T, 1))[:, None, :],
        "root_pos_w": np.zeros((T, 1, 3)),
        "foot_pos_w": np.zeros((T, 1, 6, 3)),
        "foot_lin_vel_w": np.zeros((T, 1, 6, 3)),
        "foot_ang_vel_w": np.zeros((T, 1, 6, 3)),
        "terrain_z": np.zeros((T, 1, 6)),
        "actions": np.zeros((T, 1, 4)),
        "joint_vel": np.zeros((T, 1, 4)),
        "crab_failure": np.zeros((T, 1), dtype=bool),
    }
    episodes = R.score_run(raw, compiled=compiled, scenario=scenario, dt=DT,
                           term_reason=["schedule_complete"], n_steps_env=[T])
    agg = R.aggregate(episodes)
    # ABSENT, not empty -- a body row's aggregate key set is exactly what it always was.
    for key in ("suite", "direction_by_hold", "motion_by_hold", "facing_by_hold"):
        assert key not in agg
    for key in ("wave", "hold", "category", "hand"):
        assert key not in episodes[0]
    assert "direction" not in episodes[0]["holds"]["hold0"] and "motion" not in episodes[0]["holds"]["hold0"]
    assert R.suite_aggregate_arg({"scenario_id": "body"}) is None
    text = R.summary_text({"scenario_id": "body", "task": "t", "command_override_max_deviation": 0.0}, episodes)
    assert "by category" not in text
