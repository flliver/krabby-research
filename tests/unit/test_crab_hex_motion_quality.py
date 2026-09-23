"""Motion-quality metrics of the crab-hex gait eval (gait-motion-quality plan, step 1, 2026-09-22).

Pure numpy on synthetic traces -- no Isaac. Every metric's math is pinned on a trace whose answer is
known in closed form, the category applicability (N/A = ``None``, never 0) is pinned per key, and the
new keys are cross-pinned across the four places that must agree: ``gait_eval.metrics``,
``gait_eval.report.CATEGORY_METRIC_SOURCES``, ``eval_categories.CAT_KEYS``,
``run_phases.CAT_METRIC_KEYS`` and ``resolution.CAT_COLUMN_KEYS``.
"""

from __future__ import annotations

import importlib.util
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "parkour/parkour_tasks/parkour_tasks"
SCRIPTS_DIR = TASKS / "crab_hex_forward_task/scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from gait_eval import metrics as M  # noqa: E402
from gait_eval import report as R  # noqa: E402
from gait_eval import schedule as S  # noqa: E402

DT = 0.02
IDENT_Q = np.array([1.0, 0.0, 0.0, 0.0])


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _yaw_quat(yaw: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)], axis=-1)


def _trace(n=200, *, force=400.0, speed=0.0, yaw_rate=0.0, **over):
    """A static six-down window of ``n`` steps; ``speed`` m/s along +x, ``yaw_rate`` rad/s."""
    t = np.arange(n) * DT
    root = np.zeros((n, 3))
    root[:, 0] = speed * t
    kw = dict(
        dt=DT,
        actions=np.zeros((n, 4)),
        foot_force_norm=np.full((n, 6), force),
        foot_pos_w=np.zeros((n, 6, 3)),
        foot_lin_vel_w=np.zeros((n, 6, 3)),
        root_pos_w=root,
        root_quat_w=_yaw_quat(yaw_rate * t),
        cmd_wz=np.zeros(n),
    )
    kw.update(over)
    return kw


# ------------------------------------------------------------------------------ action metrics
def test_action_jitter_is_the_mean_absolute_second_difference():
    assert M.action_jitter(np.ones((10, 3))) == 0.0
    assert M.action_jitter(np.linspace(0, 1, 50)[:, None]) == pytest.approx(0.0, abs=1e-12)   # a ramp
    saw = 0.3 * (-1.0) ** np.arange(40)                                    # +-a at the step rate
    assert M.action_jitter(saw[:, None]) == pytest.approx(4 * 0.3)
    two = np.stack([saw, np.zeros(40)], axis=1)                             # mean over dims too
    assert M.action_jitter(two) == pytest.approx(2 * 0.3)
    assert M.action_jitter(np.zeros((2, 3))) is None


def test_action_fft_freq_is_the_power_weighted_mean_frequency():
    n = 200                                                                # 4 s at 50 Hz: 0.25 Hz bins
    t = np.arange(n) * DT
    a2 = np.sin(2 * np.pi * 2.0 * t)
    assert M.action_fft_freq(a2[:, None], DT) == pytest.approx(2.0, abs=1e-9)
    # power pooled over dims: 5 Hz at amplitude 1 and 1 Hz at amplitude 2 -> (1*5 + 4*1) / 5
    both = np.stack([np.sin(2 * np.pi * 5.0 * t), 2 * np.sin(2 * np.pi * 1.0 * t)], axis=1)
    assert M.action_fft_freq(both, DT) == pytest.approx((5.0 + 4.0) / 5.0, abs=1e-9)
    # an offset carries no power; a constant has none at all
    assert M.action_fft_freq((a2 + 3.0)[:, None], DT) == pytest.approx(2.0, abs=1e-9)
    assert M.action_fft_freq(np.full((n, 2), 0.7), DT) is None
    assert M.action_fft_freq(a2[: M.MIN_FFT_STEPS - 1, None], DT) is None


def test_action_sm_matches_the_christmann_formula_on_a_sinusoid():
    n = 200
    t = np.arange(n) * DT
    a = 0.5 * np.sin(2 * np.pi * 2.0 * t)
    # one bin at f = 2 Hz with |rfft| = A n / 2 -> M = A/2; Sm = 2 / (n_bins fs) * M f
    n_bins, fs = n // 2 + 1, 1 / DT
    assert M.action_sm(a[:, None], DT) == pytest.approx(2 / (n_bins * fs) * 0.25 * 2.0, rel=1e-9)
    assert M.action_sm(np.zeros((n, 1)), DT) == 0.0


# ------------------------------------------------------------------------------ contacts / slip
def test_contact_switch_hz_and_swings_count_transitions_per_foot_per_second():
    n = 200
    force = np.full((n, 6), 400.0)
    force[50:60, 0] = 10.0          # foot 0: one swing (lift at 50, touch at 60)
    force[100:120, 3] = 49.0        # below the 50 N loaded threshold: also a swing
    force[150:160, 5] = 51.0        # above it: NOT a swing
    q = M.motion_quality_metrics("stand", **_trace(n, foot_force_norm=force))
    assert (q["liftoffs"], q["touchdowns"]) == (2, 2)
    assert q["contact_switch_hz"] == pytest.approx(4 / (6 * n * DT))
    assert q["touch_ref_n"] == M.MOTION_TOUCH_REF_N == 50.0


def test_foot_slip_reads_loaded_feet_only():
    n = 100
    vel = np.zeros((n, 6, 3))
    vel[:, :, 0] = 0.1
    vel[:, :, 2] = 5.0              # vertical speed is not slip
    force = np.full((n, 6), 400.0)
    force[:, 5] = 0.0               # an unloaded foot swinging fast is not slip
    vel[:, 5, 0] = 3.0
    q = M.motion_quality_metrics("fwd", **_trace(n, speed=0.25, foot_force_norm=force, foot_lin_vel_w=vel))
    assert q["foot_slip"] == pytest.approx(0.1)
    # slip distance = 5 loaded feet * 0.1 m/s * n dt; per metre over the net displacement
    dist = 0.25 * (n - 1) * DT
    assert q["slip_m"] == pytest.approx(5 * 0.1 * n * DT)
    assert q["slip_per_m"] == pytest.approx(5 * 0.1 * n * DT / dist)
    none = M.motion_quality_metrics("fwd", **_trace(n, speed=0.25, force=0.0))
    assert none["foot_slip"] is None


# ------------------------------------------------------------------------------ command / clock
def test_wz_active_frac_and_clock_restarts():
    n = 10
    wz = np.array([0, 0, 0.2, 0.3, 0, 0, -0.2, 0, 0, 0], dtype=float)
    clock = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 1], dtype=bool)
    q = M.motion_quality_metrics("fwd", **_trace(n, cmd_wz=wz, clock_running=clock))
    assert q["wz_active_frac"] == pytest.approx(0.3)
    assert q["clock_restarts"] == 3
    assert M.motion_quality_metrics("fwd", **_trace(n))["clock_restarts"] is None   # not recorded
    # running from the first step is no restart inside the window
    assert M.motion_quality_metrics("fwd", **_trace(n, clock_running=np.ones(n, bool)))["clock_restarts"] == 0


# ------------------------------------------------------------------------------ joints / work
NAMES = ["FL_Body_CamShaft_RevoluteJoint", "FL_Body_Hip_RevoluteJoint", "FL_Hip_Femur_RevoluteJoint",
         "FL_Femur_Tibia_RevoluteJoint", "FR_Body_CamShaft_RevoluteJoint", "FR_Hip_Femur_RevoluteJoint"]


def test_motion_joint_groups_split_rods_cams_and_the_action_set():
    g = M.motion_joint_groups(NAMES)
    assert g == {"rod": [2, 3, 5], "cam": [0, 1, 4], "actuated": [0, 2, 3, 4, 5]}


def test_work_is_positive_rod_power_and_absolute_cam_power_per_metre():
    n = 100
    g = M.motion_joint_groups(NAMES)
    jv = np.zeros((n, 6))
    tau = np.zeros((n, 6))
    jv[:, 2], tau[:, 2] = 1.0, 2.0          # rod +2 W
    jv[:, 3], tau[:, 3] = 1.0, -3.0         # rod -3 W: braking, excluded from positive work
    jv[:, 0], tau[:, 0] = 2.0, -0.5         # cam |-1 W|
    jv[:, 1], tau[:, 1] = 1.0, 0.5          # cam (body hip) 0.5 W
    kw = _trace(n, speed=0.25, joint_vel=jv, applied_torque=tau, groups=g)
    q = M.motion_quality_metrics("fwd", **kw)
    dist = 0.25 * (n - 1) * DT
    assert q["rod_work_pos_j"] == pytest.approx(2.0 * n * DT)
    assert q["cam_work_abs_j"] == pytest.approx(1.5 * n * DT)
    assert q["work_per_m"] == pytest.approx(2.0 * n * DT / dist)
    assert q["cam_work_per_m"] == pytest.approx(1.5 * n * DT / dist)
    assert q["joint_speed_mean"] == pytest.approx((2 + 1 + 1 + 0 + 0) / 5)   # the 5 actuated here
    assert q["rod_speed_mean"] is None                                        # not recorded
    rods = np.full((n, 12), -0.004)
    assert M.motion_quality_metrics("fwd", **{**kw, "rod_speed": rods})["rod_speed_mean"] == pytest.approx(0.004)
    stand = M.motion_quality_metrics("stand", **_trace(n, joint_vel=jv, applied_torque=tau, groups=g))
    assert stand["work_per_s"] == pytest.approx(2.0 + 1.5)                    # W
    assert stand["work_per_m"] is None
    # no torque recorded (an older npz): work is unmeasured, never 0
    q0 = M.motion_quality_metrics("fwd", **_trace(n, speed=0.25, joint_vel=jv, groups=g))
    assert q0["work_per_m"] is None and q0["cam_work_per_m"] is None


def test_actions_are_clipped_before_the_action_metrics():
    saw = 3.0 * (-1.0) ** np.arange(40)
    q = M.motion_quality_metrics("fwd", **_trace(40, actions=saw[:, None], action_clip=1.0))
    assert q["action_jitter"] == pytest.approx(4.0)
    assert M.motion_quality_metrics("fwd", **_trace(40, actions=saw[:, None]))["action_jitter"] == pytest.approx(12.0)


# ------------------------------------------------------------------------------ normalisers
def test_per_metre_needs_real_travel_and_per_radian_real_yaw():
    n = 100
    force = np.full((n, 6), 400.0)
    force[40:50, 1] = 0.0                     # one swing
    fwd = M.motion_quality_metrics("fwd", **_trace(n, speed=0.25, foot_force_norm=force))
    assert fwd["distance_m"] == pytest.approx(0.25 * (n - 1) * DT)
    assert fwd["swings_per_m"] == pytest.approx(1 / fwd["distance_m"])
    drift = M.motion_quality_metrics("fwd", **_trace(n, speed=0.01, foot_force_norm=force))
    assert drift["distance_m"] < M.MIN_TRAVEL_M and drift["swings_per_m"] is None
    turn = M.motion_quality_metrics("turn", **_trace(n, yaw_rate=-0.3, foot_force_norm=force))
    assert turn["net_yaw_rad"] == pytest.approx(0.3 * (n - 1) * DT)         # |net yaw|, sign dropped
    assert turn["swings_per_rad"] == pytest.approx(1 / turn["net_yaw_rad"])
    still = M.motion_quality_metrics("turn", **_trace(n, yaw_rate=0.01, foot_force_norm=force))
    assert still["swings_per_rad"] is None
    # yaw is unwrapped across +-pi
    wrap = M.motion_quality_metrics("turn", **_trace(400, yaw_rate=1.0))
    assert wrap["net_yaw_rad"] == pytest.approx(399 * DT)


# ------------------------------------------------------------------------------ stand
def test_six_down_settle_time_with_dwell_and_censoring():
    n = 200
    force = np.full((n, 6), 400.0)
    force[:30, 2] = 0.0                       # six-down from step 30 ...
    force[40:45, 4] = 0.0                     # ... broken at 40 (a 10-step run < 0.5 s dwell)
    q = M.motion_quality_metrics("stand", **_trace(n, foot_force_norm=force), hold_s_nominal=12.0)
    assert q["settle_six_down_s"] == pytest.approx(45 * DT)
    assert q["settle_six_down_censored"] is False
    assert q["six_down_frac"] == pytest.approx((n - 30 - 5) / n)
    force[:, 0] = 0.0                         # never six-down: censored at the nominal hold
    c = M.motion_quality_metrics("stand", **_trace(n, foot_force_norm=force), hold_s_nominal=12.0)
    assert c["settle_six_down_s"] == 12.0 and c["settle_six_down_censored"] is True
    assert c["six_down_frac"] == 0.0
    c2 = M.motion_quality_metrics("stand", **_trace(n, foot_force_norm=force))
    assert c2["settle_six_down_s"] == pytest.approx(n * DT)        # no nominal: the window length
    # a run that reaches the window end but is shorter than the dwell is not settled
    assert M.settle_time_s(np.r_[np.zeros(10, bool), np.ones(20, bool)], DT) is None
    assert M.settle_time_s(np.r_[np.zeros(10, bool), np.ones(25, bool)], DT) == pytest.approx(10 * DT)


def test_foot_disp_end_is_mean_planar_foothold_displacement():
    n = 50
    fp = np.zeros((n, 6, 3))
    fp[-1, 0, :2] = (0.03, 0.04)              # 5 cm
    fp[-1, 1, 2] = 0.2                        # vertical only: 0
    fp[10, 2, 0] = 1.0                        # moved and came back: 0
    q = M.motion_quality_metrics("stand", **_trace(n, foot_pos_w=fp))
    assert q["foot_disp_end"] == pytest.approx(0.05 / 6)


# ------------------------------------------------------------------------------ applicability
@pytest.mark.parametrize("category", ["stand", "fwd", "side", "back", "diag", "turn", "move_turn"])
def test_na_keys_are_none_on_every_category(category):
    n = 200
    g = M.motion_joint_groups(NAMES)
    t = np.arange(n) * DT
    kw = _trace(n, speed=0.25, yaw_rate=0.3, actions=np.sin(2 * np.pi * t)[:, None],
                joint_vel=np.ones((n, 6)), applied_torque=np.ones((n, 6)), groups=g,
                clock_running=np.ones(n, bool), rod_speed=np.ones((n, 12)))
    q = M.motion_quality_metrics(category, **kw, hold_s_nominal=12.0)
    assert set(M.MOTION_QUALITY_KEYS) <= set(q)
    for key in M.MOTION_QUALITY_KEYS:
        if category in M.MOTION_QUALITY_APPLIES[key]:
            assert q[key] is not None, (category, key)
        else:
            assert q[key] is None, (category, key)


def test_the_applicability_table_is_the_plans():
    A = M.MOTION_QUALITY_APPLIES
    assert set(A) == set(M.MOTION_QUALITY_KEYS)
    every = {"stand", "fwd", "side", "back", "diag", "turn", "move_turn"}
    for k in ("action_jitter", "action_fft_freq", "joint_speed_mean", "rod_speed_mean",
              "contact_switch_hz", "foot_slip", "wz_active_frac", "clock_restarts"):
        assert A[k] == every
    for k in ("swings_per_m", "slip_per_m", "work_per_m", "cam_work_per_m"):
        assert A[k] == {"fwd", "side", "back", "diag"}
    for k in ("swings_per_rad", "slip_per_rad", "work_per_rad", "cam_work_per_rad"):
        assert A[k] == {"turn", "move_turn"}
    for k in ("six_down_frac", "settle_six_down_s", "foot_disp_end", "work_per_s"):
        assert A[k] == {"stand"}


def test_touch_ref_mirrors_the_clock_reward_constant():
    src = (TASKS / "crab_hex_forward_task/mdp/crab_hex_clock_reward.py").read_text()
    m = re.search(r"^TOUCH_REF_N\s*=\s*([0-9.]+)\s*$", src, flags=re.M)
    assert m and float(m.group(1)) == M.MOTION_TOUCH_REF_N


# ------------------------------------------------------------------------------ registration
def test_new_keys_are_registered_everywhere_they_must_agree():
    ec = _load(TASKS / "crab_hex_omni_task/eval_categories.py", "eval_categories_mq_test")
    res = _load(TASKS / "crab_hex_omni_task/experiments/tools/resolution.py", "resolution_mq_test")
    mq = tuple(M.MOTION_QUALITY_KEYS)
    # the rigidity keys (clock-deprecation plan step 3) follow the quality keys everywhere; the
    # per-episode columns carry only rigidity's per-episode keys (the pooled two have no column).
    # Their own cross-pins are in tests/unit/test_crab_hex_rigidity_metrics.py.
    rig, rig_ep = len(M.RIGIDITY_KEYS), len(M.RIGIDITY_EPISODE_KEYS)
    assert tuple(ec.MOTION_QUALITY_KEYS) == mq
    assert tuple(ec.CAT_KEYS[-len(mq) - rig:-rig]) == mq
    assert tuple(res.CAT_QUALITY_KEYS) == mq and tuple(res.CAT_COLUMN_KEYS[-len(mq) - rig_ep:-rig_ep]) == mq
    assert tuple(res.rp.CAT_METRIC_KEYS[-len(mq) - rig:-rig]) == mq
    for key in mq:
        assert R.CATEGORY_METRIC_SOURCES[key] == ("quality", key)
        assert res.CAT_COLUMN_SOURCES[key] == ("quality", key)
        assert ec.parse_cat_id(f"cat.stand.{key}") == ("stand", None, key)
    # the per-episode column reader: None (N/A) is NaN, a number is itself
    assert math.isnan(res._category_value({"quality": {"work_per_m": None}}, "work_per_m", closed=0.0, survived=1.0))
    assert res._category_value({"quality": {"work_per_m": 3.5}}, "work_per_m", closed=0.0, survived=1.0) == 3.5
    assert R._category_value({"quality": {"six_down_frac": 0.8}}, "six_down_frac") == 0.8
    assert R._category_value({}, "six_down_frac") is None


# ------------------------------------------------------------------------------ end to end
def _mini_suite():
    suite = S.CategorySuite(
        settle_hold_s=1.0, settle_label="settle", hold_s=7.0,
        categories=[
            S.Category("stand", "stand", [S.CategoryHold("stand", 0.0, 0.0, 0.0)]),
            S.Category("fwd", "translate", [S.CategoryHold("fwd", 0.25, 0.0, 0.0)]),
            S.Category("turn", "turn", [S.CategoryHold("turn_l90", 0.0, 0.0, math.pi / 2)]),
        ],
    )
    return S.Scenario(
        id="mq_suite", task="Isaac-Crab-Hex-Omni-Walk-Play-v0", checkpoint="",
        schedule=S.suite_schedule(suite), episodes=2, kind=S.SUITE_KIND, suite=suite, waves=3,
        speed_ellipse={"fwd": 0.35, "lat": 0.25, "back": 0.25},
    )


def test_suite_scoring_carries_quality_on_the_own_hold_only_and_pools_by_category():
    scenario = _mini_suite()
    S.validate_scenario(scenario)
    E = 2
    compiled = S.stack_waves([S.compile_wave(scenario, k, dt=DT) for k in range(3)])
    T, N = compiled.n_steps, 3 * E
    speed, face = compiled.world_speed, compiled.world_face
    cmd = np.zeros((T, N, 3))
    cmd[..., 0] = speed
    cmd[..., 2] = np.clip(face * 0.8, -0.3, 0.3)
    vel = cmd.copy()
    vel[..., 2] = 0.0
    pos = np.cumsum(vel, axis=0) * DT
    yaw = np.cumsum(cmd[..., 2], axis=0) * DT
    n_j = len(NAMES)
    raw = {
        "foot_force_norm": np.full((T, N, 6), 400.0),
        "cmd_applied": cmd, "root_lin_vel_b": vel, "root_ang_vel_b": np.zeros((T, N, 3)),
        "root_quat_w": _yaw_quat(yaw), "root_pos_w": pos,
        "foot_pos_w": np.zeros((T, N, 6, 3)), "foot_lin_vel_w": np.zeros((T, N, 6, 3)),
        "foot_ang_vel_w": np.zeros((T, N, 6, 3)), "terrain_z": np.zeros((T, N, 6)),
        "actions": np.zeros((T, N, 4)), "joint_vel": np.ones((T, N, n_j)),
        "applied_torque": np.ones((T, N, n_j)), "crab_failure": np.zeros((T, N), dtype=bool),
        "heading_err": face - yaw, "travel_yaw": np.zeros((T, N)),
        "clock_running": np.abs(cmd[..., 2]) > 0, "rod_speed": np.full((T, N, 12), 0.002),
    }
    eps = R.score_run(raw, compiled=compiled, scenario=scenario, dt=DT, term_reason=["schedule_complete"] * N,
                      n_steps_env=[T] * N, travel_yaw=raw["travel_yaw"],
                      motion_groups=M.motion_joint_groups(NAMES), action_clip=1.0)
    for e in eps:
        own = e["hold"]
        assert e["holds"][own]["quality"]["category"] == e["category"]
        assert all("quality" not in h for lab, h in e["holds"].items() if lab != own)
    stand_q = eps[0]["holds"]["stand"]["quality"]
    assert stand_q["six_down_frac"] == 1.0 and stand_q["settle_six_down_s"] == 0.0
    assert stand_q["swings_per_m"] is None and stand_q["work_per_s"] is not None
    fwd_q = eps[E]["holds"]["fwd"]["quality"]
    assert fwd_q["work_per_m"] is not None and fwd_q["six_down_frac"] is None
    assert fwd_q["rod_speed_mean"] == pytest.approx(0.002)

    run_meta = {"scenario_id": scenario.id, "task": scenario.task, "command_override_max_deviation": 0.0,
                **S.suite_run_meta(scenario, dt=DT, num_envs=E,
                                   wave_seeds=[S.wave_seed(scenario, k) for k in range(3)],
                                   trained_categories=["stand", "fwd", "turn"])}
    agg = R.aggregate(eps, suite=R.suite_aggregate_arg(run_meta))
    bc = agg["suite"]["by_category"]
    assert bc["stand"]["six_down_frac"]["n"] == E and bc["stand"]["six_down_frac"]["median"] == 1.0
    assert bc["fwd"]["six_down_frac"]["n"] == 0 and bc["fwd"]["six_down_frac"]["median"] is None
    assert bc["turn"]["work_per_rad"]["n"] == E and bc["turn"]["work_per_m"]["median"] is None
    assert bc["turn"]["wz_active_frac"]["median"] > 0.0 and bc["fwd"]["wz_active_frac"]["median"] == 0.0
    assert bc["turn"]["clock_restarts"]["median"] == 0          # running from its first step on
    text = R.summary_text(run_meta, eps)
    assert "motion quality by category" in text
    stand_line = next(ln for ln in text.splitlines() if ln.strip().startswith("stand:") and "six_down_frac" in ln)
    assert "swings_per_m" not in stand_line
