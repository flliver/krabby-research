"""Rigidity metrics of the crab-hex gait eval (clock-deprecation plan, step 3, 2026-09-23).

Pure numpy on synthetic contacts -- no Isaac. Pins the step-3 definitions on traces whose answer is
known in closed form (a clean 0.333-Hz tripod at 0.181 m/s steps 11.0 times per metre on 3 loaded
feet, a wave gait stands on 5, 1-4-step lifts are taps, every key is ``None`` under its floor,
``offset_R`` is ~1 for a clock-locked gait and at the Rayleigh null for a clock-blind one), the
category applicability (N/A = ``None``), the registration across the places that must agree
(``gait_eval.metrics``, ``gait_eval.report``, ``eval_categories``, ``run_phases.CAT_METRIC_KEYS``,
``resolution``), the metric senses, and -- end to end through the harness's own scoring -- that
``summarize()`` and resolution's per-episode columns carry every RIG, guard and flag key of the
plan's Decision rule.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math
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
N12 = 600                                # the suite's 12-s own hold at dt 0.02
CYCLE = 150                              # 0.333 Hz: one tripod cycle = 3 s = 150 steps
ALL7 = ("stand", "fwd", "side", "back", "diag", "turn", "move_turn")
TRANSLATE = ("fwd", "side", "back", "diag")

#: The plan's Decision rule, per category: RIG's ten members ...
RIG_MEMBERS = {**{c: ("steps_per_m", "support_feet") for c in TRANSLATE},
               "turn": ("steps_per_rad", "support_feet")}
#: ... its 23 guards (the ``@guard`` copies read the same ``summarize()`` value as the plain id) ...
GUARD_KEYS = {"stand": ("survival", "steps_per_s"),
              **{c: ("survival", "ratio", "foot_slip", "steps_per_m") for c in TRANSLATE},
              "turn": ("survival", "foot_slip", "steps_per_rad"),
              "move_turn": ("survival", "ratio")}
#: ... and the keys its flags read (taps, lift, support, the speed decomposition).
FLAG_KEYS = {**{c: ("taps_per_m", "swing_apex", "step_rate_hz", "support_feet") for c in TRANSLATE},
             "turn": ("taps_per_rad", "swing_apex", "step_rate_hz", "support_feet"),
             "stand": ("support_feet",), "move_turn": ("support_feet",)}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ec():
    return _load(TASKS / "crab_hex_omni_task/eval_categories.py", "eval_categories_rig_test")


@pytest.fixture(scope="module")
def res():
    return _load(TASKS / "crab_hex_omni_task/experiments/tools/resolution.py", "resolution_rig_test")


def _yaw_quat(yaw: np.ndarray) -> np.ndarray:
    yaw = np.asarray(yaw, dtype=np.float64)
    z = np.zeros_like(yaw)
    return np.stack([np.cos(yaw / 2), z, z, np.sin(yaw / 2)], axis=-1)


def _tripod(n: int, *, half: int = CYCLE // 2, shift: int = 38) -> np.ndarray:
    """``(n, 6)`` loaded mask of a 50/50 tripod: set A unloaded on the odd half-cycles of ``t + shift``."""
    a_up = ((np.arange(n) + shift) // half) % 2 == 1
    loaded = np.ones((n, 6), dtype=bool)
    loaded[:, list(M.TRIPOD_A_IDX)] = ~a_up[:, None]
    loaded[:, list(M.TRIPOD_B_IDX)] = a_up[:, None]
    return loaded


def _locked_tripod(clock_phase: np.ndarray) -> np.ndarray:
    """A tripod that FOLLOWS the clock: set A loaded while ``sin(phase) >= 0``."""
    a_down = np.sin(clock_phase) >= 0.0
    loaded = np.ones((clock_phase.size, 6), dtype=bool)
    loaded[:, list(M.TRIPOD_A_IDX)] = a_down[:, None]
    loaded[:, list(M.TRIPOD_B_IDX)] = ~a_down[:, None]
    return loaded


def _rig(category: str, loaded: np.ndarray, *, speed: float = 0.0, yaw_rate: float = 0.0, **over):
    """``rigidity_metrics`` of a window: loaded -> 400 N / else 0 N, root at ``speed`` m/s along +x."""
    n = loaded.shape[0]
    t = np.arange(n) * DT
    root = np.zeros((n, 3))
    root[:, 0] = speed * t
    kw = dict(dt=DT, foot_force_norm=np.where(loaded, 400.0, 0.0), root_pos_w=root,
              root_quat_w=_yaw_quat(yaw_rate * t))
    kw.update(over)
    return M.rigidity_metrics(category, **kw)


# ------------------------------------------------------------------------------ the plan's traces
def test_clean_tripod_steps_eleven_times_per_metre_on_three_loaded_feet():
    """6 feet x 0.333 Hz / 0.181 m/s = 11.05 steps per metre; a 50/50 tripod always has 3 feet down."""
    q = _rig("fwd", _tripod(N12), speed=0.181)
    assert q["steps"] == 24 and q["taps"] == 0 and q["runs_censored"] == 0   # 4 cycles x 6 feet
    assert q["distance_m"] == pytest.approx(0.181 * (N12 - 1) * DT)
    assert q["steps_per_m"] == pytest.approx(11.0, rel=0.02)
    assert q["support_feet"] == 3.0
    assert q["step_rate_hz"] == pytest.approx(24 / 12.0)                      # per foot: 1/3 Hz
    assert q["steps_per_s"] is None and q["steps_per_rad"] is None           # stand / turning keys: N/A
    # gait_hz within one natural FFT bin (1 / 12 s) of the 1/3-Hz cycle
    assert q["gait_hz"] == pytest.approx(1 / 3, abs=1 / (N12 * DT))


def test_the_loaded_threshold_is_50_newtons():
    """Step-6 fix 8: ``loaded`` is foot force > 50 N -- pinned on the default itself and on a trace
    that swings between 60 N (loaded) and 40 N (unloaded) instead of 400 N / 0 N."""
    loaded = _tripod(N12)
    q = _rig("fwd", loaded, speed=0.181, foot_force_norm=np.where(loaded, 60.0, 40.0))
    assert q["touch_ref_n"] == 50.0
    assert q["steps"] == 24 and q["taps"] == 0 and q["support_feet"] == 3.0
    # 40 N is not loaded: at 40 N the whole window no foot is ever down
    assert _rig("fwd", loaded, speed=0.181, foot_force_norm=np.full((N12, 6), 40.0))["support_feet"] == 0.0
    # 60 N is loaded: all six feet are always down, nothing steps
    six = _rig("fwd", loaded, speed=0.181, foot_force_norm=np.full((N12, 6), 60.0))
    assert six["support_feet"] == 6.0 and six["steps"] == 0


def test_wave_gait_stands_on_five_feet():
    """One leg at a time: each foot unloaded for 1/6 of a 1.2-s cycle, so 5 feet are always down."""
    n, per = N12, 10
    loaded = np.ones((n, 6), dtype=bool)
    for t in range(n):
        loaded[t, (t // per) % 6] = False
    q = _rig("fwd", loaded, speed=0.1)
    assert q["support_feet"] == pytest.approx(5.0)
    # every 10-step lift is a step; foot 0's opening lift began before the window and is not one
    assert q["taps"] == 0 and q["steps"] == n // per - 1


def test_lifts_of_one_to_four_steps_are_taps_and_window_edges_are_handled():
    n = 200
    loaded = np.ones((n, 6), dtype=bool)
    for start, length in ((20, 1), (40, 2), (60, 3), (80, 4), (100, 5)):
        loaded[start:start + length, 0] = False       # foot 0: four taps, then one 5-step step
    loaded[:10, 1] = False                            # foot 1: airborne at the first step -> neither
    loaded[n - 3:, 2] = False                         # foot 2: a 3-step run cut by the window -> censored
    loaded[n - 6:, 3] = False                         # foot 3: 6 steps and still up -> already a step
    steps, taps, censored = M.step_tap_runs(loaded)
    assert sorted((f, s) for f, s, _e in steps) == [(0, 100), (3, n - 6)]
    assert [(f, e - s) for f, s, e in taps] == [(0, 1), (0, 2), (0, 3), (0, 4)]
    assert censored == 1
    q = _rig("fwd", loaded, speed=0.25)
    assert (q["steps"], q["taps"]) == (2, 4)
    assert q["taps_per_m"] == pytest.approx(4 / q["distance_m"])
    assert q["steps_per_m"] == pytest.approx(2 / q["distance_m"])
    # the tap boundary is the knob, not a literal
    assert len(M.step_tap_runs(loaded, min_step_steps=4)[0]) == 3


def test_every_metric_is_none_under_its_floor():
    loaded = _tripod(N12)
    # travel floor (MIN_TRAVEL_M) on the per-metre keys; yaw floor (MIN_NET_YAW_RAD) per radian
    still = _rig("fwd", loaded, speed=0.05 / (N12 * DT) * 0.5)
    assert still["distance_m"] < M.MIN_TRAVEL_M
    assert still["steps_per_m"] is None and still["taps_per_m"] is None
    assert still["support_feet"] == 3.0                                    # no floor: always measured
    spun = _rig("turn", loaded, yaw_rate=0.04 / (N12 * DT), heading_err=np.linspace(1.0, 0.0, N12))
    assert spun["net_yaw_rad"] < M.MIN_NET_YAW_RAD
    for key in ("steps_per_rad", "taps_per_rad", "steps_per_rad_pre", "steps_per_rad_post"):
        assert spun[key] is None, key
    # no clock recorded -> no clock keys; a stopped clock -> clock_hz 0 and no ratio / phase
    assert still["clock_hz"] is None and still["cadence_ratio"] is None
    stopped = _rig("fwd", loaded, speed=0.2, clock_phase=np.full(N12, 1.0))
    assert stopped["clock_hz"] == 0.0 and stopped["cadence_ratio"] is None
    assert stopped["phase_offset_rad"] is None and stopped["phase_plv"] is None
    # no terrain (the root-offset fallback) -> no swing apex; so too with no step that ended
    assert _rig("fwd", loaded, speed=0.2, foot_pos_w=np.zeros((N12, 6, 3)))["swing_apex"] is None
    none_closed = _rig("fwd", np.ones((N12, 6), dtype=bool), speed=0.2, foot_pos_w=np.zeros((N12, 6, 3)),
                       terrain_z=np.zeros((N12, 6)))
    assert none_closed["swing_apex"] is None and none_closed["swing_apex_n"] == 0
    # gait_hz: too short a window, or no alternation at all (a statue)
    assert _rig("fwd", loaded[:M.MIN_FFT_STEPS - 1], speed=0.2)["gait_hz"] is None
    assert none_closed["gait_hz"] is None
    # the pooled keys under MIN_POOLED_EPISODES episodes
    two = [{"speed_m_s": 0.1, "gait_hz": 0.3, "phase_offset_rad": 0.0},
           {"speed_m_s": 0.2, "gait_hz": 0.4, "phase_offset_rad": 0.0}]
    pooled = M.rigidity_pooled_metrics("fwd", two)
    assert pooled["hz_speed_slope"]["value"] is None and pooled["offset_R"]["value"] is None
    # an empty window scores nothing
    assert M.rigidity_metrics("fwd", dt=DT, foot_force_norm=np.zeros((0, 6)), root_pos_w=np.zeros((0, 3)),
                              root_quat_w=np.zeros((0, 4))) is None


# ------------------------------------------------------------------------------ applicability
def test_the_applicability_table_is_the_plans():
    A = M.RIGIDITY_APPLIES
    assert set(A) == set(M.RIGIDITY_KEYS)
    for k in ("steps_per_m", "taps_per_m"):
        assert A[k] == set(TRANSLATE)
    for k in ("steps_per_rad", "taps_per_rad"):
        assert A[k] == {"turn", "move_turn"}
    for k in ("support_feet", "gait_hz", "clock_hz", "cadence_ratio"):
        assert A[k] == set(ALL7)
    assert A["steps_per_s"] == {"stand"}
    for k in ("step_rate_hz", "swing_apex", "offset_R"):
        assert A[k] == set(ALL7) - {"stand"}
    assert A["steps_per_rad_pre"] == A["steps_per_rad_post"] == {"turn"}
    assert A["hz_speed_slope"] == set(TRANSLATE) | {"move_turn"}


@pytest.mark.parametrize("category", ALL7)
def test_na_keys_are_none_on_every_category(category):
    n = N12
    phase = np.remainder(0.3 + 2 * np.pi * (1 / 3) * np.arange(n) * DT, 2 * np.pi)
    loaded = _locked_tripod(phase)
    fp = np.zeros((n, 6, 3))
    fp[..., 2] = np.where(loaded, 0.02, 0.07)
    err = np.linspace(1.5, 0.0, n)                  # closes at the end of the hold
    q = _rig(category, loaded, speed=0.2, yaw_rate=0.25, foot_pos_w=fp, terrain_z=np.zeros((n, 6)),
             clock_phase=phase, heading_err=err)
    assert set(M.RIGIDITY_EPISODE_KEYS) <= set(q)
    assert not set(M.RIGIDITY_POOLED_KEYS) & set(q)             # across-episode: never per episode
    for key in M.RIGIDITY_EPISODE_KEYS:
        if category in M.RIGIDITY_APPLIES[key]:
            assert q[key] is not None, (category, key)
        else:
            assert q[key] is None, (category, key)


# ------------------------------------------------------------------------------ definitions
def test_swing_apex_is_the_median_step_peak_above_the_calibrated_stance():
    n = N12
    loaded = _tripod(n)
    fp = np.zeros((n, 6, 3))
    fp[..., 2] = 0.03                                 # the pad origin sits 3 cm above the ground in stance
    for f, s, e in M.step_tap_runs(loaded)[0]:
        k = np.arange(e - s)
        fp[s:e, f, 2] += 0.08 * np.sin(np.pi * (k + 0.5) / (e - s))
    assert loaded[150:152, 0].all()
    loaded[150:152, 0] = False                        # a tap inside foot 0's stance: never an apex
    tz = np.zeros((n, 6))
    q = _rig("fwd", loaded, speed=0.2, foot_pos_w=fp, terrain_z=tz)
    assert q["stance_z_offset_m"] == pytest.approx(0.03)
    assert q["swing_apex"] == pytest.approx(0.08, abs=1e-3)
    # the one step still up at the window's end has no known peak and is left out of the median
    assert q["swing_apex_n"] == q["steps"] - 3
    # the calibration follows the terrain, not the world z
    q2 = _rig("fwd", loaded, speed=0.2, foot_pos_w=fp + np.array([0.0, 0.0, 0.5]), terrain_z=tz + 0.5)
    assert q2["swing_apex"] == pytest.approx(q["swing_apex"])


def test_clock_hz_and_cadence_ratio_follow_the_recorded_phase():
    n = N12
    phase = np.remainder(1.0 + 2 * np.pi * (1 / 3) * np.arange(n) * DT, 2 * np.pi)   # wrapped, as recorded
    assert M.clock_frequency_hz(phase, DT) == pytest.approx(1 / 3, rel=1e-9)
    q = _rig("fwd", _locked_tripod(phase), speed=0.181, clock_phase=phase)
    assert q["clock_hz"] == pytest.approx(1 / 3, rel=1e-9)
    assert q["cadence_ratio"] == pytest.approx(1.0, abs=(1 / (n * DT)) / (1 / 3))
    assert q["phase_plv"] > 0.9                        # locked within the window


def test_the_turn_split_sums_to_steps_per_rad_around_closure():
    n = N12
    loaded = _tripod(n)
    t_c = 250
    err = np.where(np.arange(n) < t_c, 1.2, 0.05)      # |err| first below 0.15 rad at t_c
    q = _rig("turn", loaded, yaw_rate=0.1, heading_err=err)
    steps = M.step_tap_runs(loaded)[0]
    pre = sum(1 for _f, s, _e in steps if s < t_c)
    assert q["closure_s"] == pytest.approx(t_c * DT)
    assert q["steps_per_rad_pre"] == pytest.approx(pre / q["net_yaw_rad"])
    assert q["steps_per_rad_post"] == pytest.approx((len(steps) - pre) / q["net_yaw_rad"])
    assert q["steps_per_rad_pre"] + q["steps_per_rad_post"] == pytest.approx(q["steps_per_rad"])
    # a turn that never closes: every step is before closure and there is no after
    open_ = _rig("turn", loaded, yaw_rate=0.1, heading_err=np.full(n, 1.2))
    assert open_["closure_s"] is None and open_["steps_per_rad_post"] is None
    assert open_["steps_per_rad_pre"] == open_["steps_per_rad"]


def test_offset_R_is_one_for_a_clock_locked_gait_and_at_the_null_for_a_clock_blind_one():
    rng = np.random.default_rng(3)
    n, n_ep = N12, 100
    t = np.arange(n) * DT
    locked, blind = [], []
    for _ in range(n_ep):
        clock = np.remainder(rng.uniform(0, 2 * np.pi) + 2 * np.pi * (1 / 3) * t, 2 * np.pi)
        own = np.remainder(rng.uniform(0, 2 * np.pi) + 2 * np.pi * (1 / 3) * t, 2 * np.pi)
        locked.append(_rig("fwd", _locked_tripod(clock), speed=0.18, clock_phase=clock))
        # same cadence, its own phase: the clock's cadence without the clock's phase
        blind.append(_rig("fwd", _locked_tripod(own), speed=0.18, clock_phase=clock))
    got = M.rigidity_pooled_metrics("fwd", locked)["offset_R"]
    assert got["n"] == n_ep and got["value"] > 0.95 and got["rayleigh_p"] < 1e-6
    null = M.rigidity_pooled_metrics("fwd", blind)["offset_R"]
    assert null["n"] == n_ep
    assert null["value"] < math.sqrt(4.6 / n_ep)            # under the 99 % critical value (~0.21)
    assert null["rayleigh_p"] > 0.01
    # within each episode the blind gait is still cadence-locked: phase freedom is ACROSS episodes
    assert np.median([b["phase_plv"] for b in blind]) > 0.9
    # the test itself: identical angles are R = 1, two opposite pairs are R = 0
    assert M.rayleigh_test([0.4] * 5)[0] == pytest.approx(1.0)
    assert M.rayleigh_test([0.0, 0.0, np.pi, np.pi])[0] == pytest.approx(0.0, abs=1e-12)


def test_hz_speed_slope_is_the_ols_slope_of_gait_hz_on_achieved_speed():
    blocks = [{"speed_m_s": s, "gait_hz": 0.1 + 2.0 * s} for s in (0.10, 0.15, 0.20, 0.25)]
    got = M.rigidity_pooled_metrics("fwd", blocks + [None, {"speed_m_s": 0.3, "gait_hz": None}])
    assert got["hz_speed_slope"]["value"] == pytest.approx(2.0) and got["hz_speed_slope"]["n"] == 4
    # a command-locked cadence (every episode the same gait_hz) reads 0
    flat = [{"speed_m_s": s, "gait_hz": 1 / 3} for s in (0.1, 0.2, 0.3)]
    assert M.rigidity_pooled_metrics("fwd", flat)["hz_speed_slope"]["value"] == pytest.approx(0.0, abs=1e-12)
    # no speed spread -> undefined; a category with no speed command -> N/A
    same = [{"speed_m_s": 0.2, "gait_hz": g} for g in (0.3, 0.4, 0.5)]
    assert M.rigidity_pooled_metrics("fwd", same)["hz_speed_slope"]["value"] is None
    assert M.rigidity_pooled_metrics("turn", blocks)["hz_speed_slope"]["value"] is None
    assert M.rigidity_pooled_metrics("stand", blocks)["offset_R"]["value"] is None


# ------------------------------------------------------------------------------ registration / signs
def test_rigidity_keys_are_registered_everywhere_they_must_agree(ec, res):
    rig, ep, pooled = tuple(M.RIGIDITY_KEYS), tuple(M.RIGIDITY_EPISODE_KEYS), tuple(M.RIGIDITY_POOLED_KEYS)
    assert rig == ep + pooled
    assert tuple(ec.RIGIDITY_KEYS) == rig and tuple(ec.CAT_KEYS[-len(rig):]) == rig
    assert tuple(res.rp.CAT_METRIC_KEYS[-len(rig):]) == rig
    assert tuple(res.CAT_RIGIDITY_KEYS) == ep and tuple(res.CAT_POOLED_KEYS) == pooled
    assert tuple(res.CAT_COLUMN_KEYS[-len(ep):]) == ep and not set(pooled) & set(res.CAT_COLUMN_KEYS)
    assert tuple(R.CATEGORY_POOLED_KEYS) == pooled
    for key in ep:
        assert R.CATEGORY_METRIC_SOURCES[key] == ("rigidity", key)
        assert res.CAT_COLUMN_SOURCES[key] == ("rigidity", key)
    for key in pooled:
        assert key not in R.CATEGORY_METRIC_SOURCES and key not in res.CAT_COLUMN_SOURCES
    for key in rig:                                       # incl. offset_R's capital
        assert ec.parse_cat_id(f"cat.turn.{key}") == ("turn", None, key)
        assert ec.parse_cat_id(f"cat.side.by_hand.l.{key}") == ("side", "l", key)
    assert ec.parse_cat_id("cat.fwd.offset_r") is None    # membership still decides
    # the per-episode column reader: None (N/A) is NaN, a number is itself
    assert math.isnan(res._category_value({"rigidity": {"steps_per_m": None}}, "steps_per_m",
                                          closed=0.0, survived=1.0))
    assert res._category_value({"rigidity": {"support_feet": 3.2}}, "support_feet", closed=0.0, survived=1.0) == 3.2
    assert R._category_value({"rigidity": {"support_feet": 3.2}}, "support_feet") == 3.2


def test_stepping_counts_are_lower_is_better_and_support_and_apex_higher(ec):
    for key in ("steps_per_m", "steps_per_rad", "steps_per_s", "taps_per_m", "taps_per_rad"):
        assert key in ec.LOWER_IS_BETTER_KEYS
        assert ec.metric_sign(f"cat.fwd.{key}") == -1
        assert ec.metric_sign(f"cat.fwd.{key}@guard") == -1
    assert ec.metric_sign("cat.side.by_hand.r.steps_per_m") == -1
    # support_feet / swing_apex keep +1, and so does every WATCH key (no "better" direction)
    for key in ("support_feet", "swing_apex", "step_rate_hz", "gait_hz", "clock_hz", "cadence_ratio",
                "hz_speed_slope", "offset_R", "steps_per_rad_pre", "steps_per_rad_post"):
        assert key not in ec.LOWER_IS_BETTER_KEYS
        assert ec.metric_sign(f"cat.turn.{key}") == 1, key


def test_the_pooled_keys_never_resolve_and_say_why(res):
    labels = res._manifest_scenario("suite__omni")["hold_labels"]
    watch = ["cat.fwd.offset_R", "cat.move_turn.hz_speed_slope", "cat.side.by_hand.l.offset_R",
             "cat.fwd.steps_per_m", "cat.turn.support_feet"]
    rule = {"rule_version": 2, "expanded": True, "blocks": {},
            "guards": {"level": 0.99, "metrics": ["cat.stand.steps_per_s"]}, "watch": watch}
    specs, unresolved = res.specs_from_rule(rule, labels, suite=True)
    assert unresolved == watch[:3]
    assert {s.mid for s in specs} == {"cat.stand.steps_per_s", "cat.fwd.steps_per_m", "cat.turn.support_feet"}
    assert next(s for s in specs if s.mid == "cat.stand.steps_per_s").sign == -1
    for mid in unresolved:
        assert "one statistic across the category's episodes" in res.why_unresolved(mid, labels, suite=True)


# ------------------------------------------------------------------------------ end to end
def _suite_scenario(res):
    sc = next(s for s in S.load_manifest(res.rp.OMNI_MANIFEST)[0] if s.id == "suite__omni")
    return dataclasses.replace(sc, episodes=3)


def _suite_raw(compiled, rng):
    """A clock-locked 0.333-Hz tripod walking at 0.6-0.8 of each episode's command, turning toward its face."""
    T, N = compiled.n_steps, compiled.world_speed.shape[1]
    t = np.arange(T) * DT
    speed, direction, face = compiled.world_speed, compiled.world_dir, compiled.world_face
    ratio = 0.6 + 0.1 * (np.arange(N) % 3)
    yaw = np.zeros((T, N))
    wz = np.zeros((T, N))
    for k in range(T - 1):
        wz[k] = np.clip(0.8 * M.wrap_to_pi(face[k] - yaw[k]), -0.3, 0.3)
        yaw[k + 1] = yaw[k] + wz[k] * DT
    heading_err = M.wrap_to_pi(face - yaw)
    v_w = np.stack([speed * np.cos(direction), speed * np.sin(direction)], axis=-1)       # (T, N, 2)
    c, s = np.cos(-yaw), np.sin(-yaw)
    v_b = np.stack([c * v_w[..., 0] - s * v_w[..., 1], s * v_w[..., 0] + c * v_w[..., 1]], axis=-1)
    cmd = np.concatenate([v_b, wz[..., None]], axis=-1)
    lin = np.concatenate([v_b * ratio[None, :, None], np.zeros((T, N, 1))], axis=-1)
    pos = np.zeros((T, N, 3))
    pos[..., :2] = np.cumsum(v_w * ratio[None, :, None], axis=0) * DT
    clock = np.remainder(rng.uniform(0, 2 * np.pi, N)[None, :] + 2 * np.pi * (1 / 3) * t[:, None], 2 * np.pi)
    loaded = np.stack([_locked_tripod(clock[:, e]) for e in range(N)], axis=1)            # (T, N, 6)
    foot = np.zeros((T, N, 6, 3))
    foot[..., 2] = np.where(loaded, 0.03, 0.09)
    ang = np.zeros((T, N, 3))
    ang[..., 2] = wz
    return {
        "foot_force_norm": np.where(loaded, 400.0, 0.0), "cmd_applied": cmd, "root_lin_vel_b": lin,
        "root_ang_vel_b": ang, "root_quat_w": _yaw_quat(yaw), "root_pos_w": pos, "foot_pos_w": foot,
        "foot_lin_vel_w": np.full((T, N, 6, 3), 0.01 / math.sqrt(2)) * np.array([1.0, 1.0, 0.0]),
        "foot_ang_vel_w": np.zeros((T, N, 6, 3)), "terrain_z": np.zeros((T, N, 6)),
        "actions": np.zeros((T, N, 4)), "joint_vel": np.zeros((T, N, 4)),
        "crab_failure": np.zeros((T, N), dtype=bool),
        "heading_err": heading_err, "travel_yaw": np.arctan2(cmd[..., 1], cmd[..., 0]), "clock_phase": clock,
    }


def test_summarize_and_resolution_columns_carry_every_rig_guard_and_flag_key(res):
    """Through the harness's own scoring: every RIG member, guard and flag key of the plan's Decision
    rule is a finite number in ``summarize()['cat']``, and resolution's per-episode ``cat`` columns
    reproduce it -- so the step-10 spread check can never read one of them as absent."""
    assert sum(len(v) for v in RIG_MEMBERS.values()) == 10
    assert sum(len(v) for v in GUARD_KEYS.values()) == 23
    sc = _suite_scenario(res)
    compiled = S.stack_waves([S.compile_wave(sc, k, dt=DT) for k in range(sc.waves)])
    raw = _suite_raw(compiled, np.random.default_rng(11))
    N = raw["cmd_applied"].shape[1]
    eps = R.score_run(raw, compiled=compiled, scenario=sc, dt=DT, term_reason=["schedule_complete"] * N,
                      n_steps_env=[compiled.n_steps] * N, travel_yaw=raw["travel_yaw"], terrain_mesh=True)
    for e in eps:
        own = e["hold"]
        assert e["holds"][own]["rigidity"]["category"] == e["category"]
        assert all("rigidity" not in h for lab, h in e["holds"].items() if lab != own)
    run_meta = {"scenario_id": sc.id, "task": sc.task, "command_override_max_deviation": 0.0,
                **S.suite_run_meta(sc, dt=DT, num_envs=sc.episodes,
                                   wave_seeds=[S.wave_seed(sc, k) for k in range(sc.waves)],
                                   trained_categories=list(ALL7))}
    agg = R.aggregate(eps, suite=R.suite_aggregate_arg(run_meta))
    cat = res.rp.summarize(agg)["cat"]

    want: dict[str, set] = {}
    for table in (RIG_MEMBERS, GUARD_KEYS, FLAG_KEYS):
        for c, keys in table.items():
            want.setdefault(c, set()).update(keys)
    for c, keys in want.items():
        for key in keys:
            v = cat[c][key]
            assert v is not None and math.isfinite(v), (c, key, v)
    # the numbers are the traces' own: a 50/50 tripod, turns that closed, a clock-locked phase
    for c in ALL7:
        assert cat[c]["support_feet"] == pytest.approx(3.0)
    assert cat["turn"]["steps_per_rad_pre"] is not None and cat["turn"]["steps_per_rad_post"] is not None
    for c in ("fwd", "side", "move_turn"):
        assert cat[c]["offset_R"] > 0.95 and cat[c]["hz_speed_slope"] is not None
    for h in ("l", "r"):
        for key in ("steps_per_rad", "support_feet", "offset_R"):
            assert cat["turn"]["by_hand"][h][key] is not None, (h, key)

    # resolution's per-episode columns: every per-episode key, reducing to summarize()'s number
    idx = res._suite_index(Path("synthetic"), run_meta, agg, eps)
    end_steps = res._survival_end_steps(sc.id, DT, suite=agg["suite"])
    cols = res._columns(eps, compiled.hold_labels, {}, end_steps, suite_index=idx)
    for c, keys in want.items():
        for key in keys:
            col = cols[f"cat:{c}:{key}"]
            assert np.isfinite(col).any(), (c, key)
            got = np.nanmean(col) if key == "survival" else np.nanmedian(col)
            assert got == pytest.approx(cat[c][key], abs=1e-12), (c, key)

    text = R.summary_text(run_meta, eps)
    assert "rigidity by category" in text
    stand_line = next(ln for ln in text.splitlines() if ln.strip().startswith("stand:") and "support_feet" in ln)
    assert "steps_per_s=" in stand_line and "steps_per_m" not in stand_line


def test_rigidity_off_the_mesh_has_no_swing_apex_and_body_rows_carry_no_block(res):
    sc = _suite_scenario(res)
    compiled = S.stack_waves([S.compile_wave(sc, k, dt=DT) for k in range(2)])
    raw = _suite_raw(compiled, np.random.default_rng(5))
    N = raw["cmd_applied"].shape[1]
    kw = dict(compiled=compiled, scenario=sc, dt=DT, term_reason=["schedule_complete"] * N,
              n_steps_env=[compiled.n_steps] * N, travel_yaw=raw["travel_yaw"])
    on = R.score_run(raw, **kw)                           # the default: the harness's mesh
    off = R.score_run(raw, terrain_mesh=False, **kw)
    for a, b in zip(on, off):
        ra, rb = a["holds"][a["hold"]]["rigidity"], b["holds"][b["hold"]]["rigidity"]
        assert rb["swing_apex"] is None and rb["stance_z_offset_m"] is None
        assert {k: v for k, v in ra.items() if k not in ("swing_apex", "stance_z_offset_m", "swing_apex_n")} == \
               {k: v for k, v in rb.items() if k not in ("swing_apex", "stance_z_offset_m", "swing_apex_n")}
    assert any(a["holds"][a["hold"]]["rigidity"]["swing_apex"] is not None for a in on)
    # an older npz without clock_phase leaves the clock keys None and scores everything else
    old = R.score_run({k: v for k, v in raw.items() if k != "clock_phase"}, **kw)
    fwd = next(e for e in old if e["hold"] == "fwd")["holds"]["fwd"]["rigidity"]
    assert fwd["clock_hz"] is None and fwd["phase_offset_rad"] is None and fwd["steps_per_m"] is not None
