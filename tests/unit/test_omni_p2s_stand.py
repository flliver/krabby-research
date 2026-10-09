# SPDX-License-Identifier: BSD-3-Clause
"""``p2s_stand.py`` (P2S_PLAN.md step 1: the ONE stand-estimator set for stage p2s). CPU only, pure numpy --
nothing here reads a real eval stamp (the module's own ``--selfcheck`` is what validates against the real E6
statue / TG raws: STILL on every clause, >= 9 hover episodes).

Pinned on synthetic fixtures:

* :func:`hold_masks` / :func:`fall_trim_upper`'s fall-trim tail, reproducing ``gait_eval/report.py``'s own
  ``_window_mask`` convention;
* :func:`stance_offset` pooled across the six feet, not self-referenced per foot (a foot that never lands is
  excluded from the pool rather than corrupting it or going unmeasured against it);
* :func:`episode_stand_metrics` on four hand-built episodes: a still six-down stance (zero steps, zero
  off-ground, no hover), a still stance with one grounded 0-N foot (unloaded but NOT hover -- force alone is
  not enough), a tripod hover (unloaded AND raised -- hover True), and a march (a known step count);
* :func:`clearance_stats`'s median-over-feet vs report-only max-over-feet;
* :func:`stand_hold_index` / :func:`stand_onset_pairs` against literal field-row schedules (turn holds, also
  commanding zero speed, must never be read as a "stand");
* :func:`episode_onset_latency`'s reach-and-hold rule and its censored (never reaches) case;
* :func:`still_clauses`'s thresholds, including the "<=" boundary;
* :func:`head_summary` / the CLI end to end on a tiny on-disk stamp, including the PENDING case (no stamp on
  disk -> None, not a crash).
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL_PY = (REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-07_omni_phase2/"
                   "scripts/p2s_stand.py")
N_FEET = 6


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def P():
    return _load(TOOL_PY, "p2s_stand__test")


# ================================================================================================ episode fixtures
def _episode(n: int, *, force=None, foot_z=None, terrain_z=None, foot_vel=None, segment_id=None,
            steady=None, crab_failure=None, heading_err=None, root_lin_vel_b=None, root_z=1.0,
            cmd_applied=None) -> dict:
    """A minimal synthetic episode (one hold, ``hold_idx`` always 0 unless ``segment_id`` is given)."""
    force = np.full((n, N_FEET), 600.0) if force is None else force
    foot_z = np.full((n, N_FEET), 0.03) if foot_z is None else foot_z
    terrain_z = np.zeros((n, N_FEET)) if terrain_z is None else terrain_z
    foot_vel = np.zeros((n, N_FEET, 3)) if foot_vel is None else foot_vel
    foot_pos_w = np.zeros((n, N_FEET, 3))
    foot_pos_w[:, :, 2] = foot_z
    root_pos_w = np.zeros((n, 3))
    root_pos_w[:, 2] = root_z
    return {
        "foot_force_norm": force, "foot_pos_w": foot_pos_w, "terrain_z": terrain_z,
        "foot_lin_vel_w": foot_vel, "root_pos_w": root_pos_w,
        "root_lin_vel_b": np.zeros((n, 3)) if root_lin_vel_b is None else root_lin_vel_b,
        "heading_err": np.zeros(n) if heading_err is None else heading_err,
        "segment_id": np.zeros(n, dtype=np.int16) if segment_id is None else segment_id,
        "steady_mask": np.ones(n, dtype=bool) if steady is None else steady,
        "crab_failure": np.zeros(n, dtype=bool) if crab_failure is None else crab_failure,
        "cmd_applied": np.zeros((n, 3)) if cmd_applied is None else cmd_applied,
    }


# ================================================================================================ windows / fall trim
def test_hold_masks_full_vs_steady(P):
    n = 40
    seg = np.array([0] * 10 + [1] * 20 + [2] * 10, dtype=np.int16)
    steady = np.ones(n, dtype=bool)
    steady[10:14] = False  # the first 4 steps of hold 1 are "settling"
    d = _episode(n, segment_id=seg, steady=steady)
    full, steadym = P.hold_masks(d, 1, dt=0.02)
    assert full.sum() == 20 and int(full.nonzero()[0][0]) == 10
    assert steadym.sum() == 16


def test_fall_trim_drops_the_last_half_second_only_on_a_fall(P):
    n = 40
    cf = np.zeros(n, dtype=bool)
    d_clean = _episode(n, crab_failure=cf)
    assert P.fall_trim_upper(d_clean, dt=0.02) == n  # no fall: nothing trimmed

    cf2 = np.zeros(n, dtype=bool)
    cf2[-1] = True  # the terminal logged step is the failure
    d_fallen = _episode(n, crab_failure=cf2)
    assert P.fall_trim_upper(d_fallen, dt=0.02) == n - 25  # 0.5 s / 0.02 s = 25 steps


def test_fell_in_hold_uses_the_untrimmed_segment(P):
    n = 40
    seg = np.array([0] * 20 + [1] * 20, dtype=np.int16)
    cf = np.zeros(n, dtype=bool)
    cf[-1] = True  # fails inside hold 1, in the last row the trim would otherwise cut
    d = _episode(n, segment_id=seg, crab_failure=cf)
    assert P.fell_in_hold(d, 1) is True
    assert P.fell_in_hold(d, 0) is False


# ================================================================================================ stance offset,
# pooled across feet (never a per-foot self-reference)
def test_stance_offset_is_pooled_and_excludes_a_never_loaded_foot(P):
    n = 30
    force = np.full((n, N_FEET), 600.0)
    force[:, 5] = 0.0  # foot 5 never loaded (a hovering foot) -- must not corrupt or define the offset
    foot_z = np.full((n, N_FEET), 0.03)
    foot_z[:, 5] = 0.23  # raised, consistent with hovering -- irrelevant to the (loaded-only) offset
    d = _episode(n, force=force, foot_z=foot_z)
    full = np.ones(n, dtype=bool)
    offset = P.stance_offset(d, full)
    assert offset == pytest.approx(0.03, abs=1e-9)  # pooled over the five loaded feet only


# ================================================================================================ the four hand-built
# stand episodes
def test_still_six_down_stance(P):
    d = _episode(100)  # all six feet loaded, flat, motionless -- the default fixture
    m = P.episode_stand_metrics(d, 0, dt=0.02)
    assert m["steps_per_s"] == 0.0 and m["taps_per_s"] == 0.0
    assert m["off_ground_share"] == 0.0
    assert m["hover"] is False
    assert m["clearance_m"] == pytest.approx(0.0, abs=1e-9)
    assert m["fall"] is False


def test_still_stance_with_one_grounded_zero_newton_foot_is_not_hover(P):
    n = 100
    force = np.full((n, N_FEET), 600.0)
    force[:, 3] = 0.0  # grounded (flat, height unchanged) but reads 0 N -- force alone must not flag this
    d = _episode(n, force=force)
    m = P.episode_stand_metrics(d, 0, dt=0.02)
    assert m["hover"] is False
    assert m["off_ground_share"] == 0.0  # the 0-N foot is not raised, so it is not off-ground either


def test_tripod_hover_unloaded_and_raised(P):
    n = 100
    force = np.full((n, N_FEET), 600.0)
    foot_z = np.full((n, N_FEET), 0.03)
    hover_feet = (0, 2, 4)
    for f in hover_feet:
        force[:, f] = 0.0
        foot_z[:, f] = 0.03 + 0.10  # 10 cm above the stance offset, well past the 3 cm tolerance
    d = _episode(n, force=force, foot_z=foot_z)
    m = P.episode_stand_metrics(d, 0, dt=0.02)
    assert m["hover"] is True
    hov = P.hover_feet(d, np.ones(n, dtype=bool), m["stance_offset_m"])
    assert list(np.flatnonzero(hov)) == list(hover_feet)


def test_march_counts_steps_not_taps(P):
    """One foot: 5 steps loaded, then a clean 10-step swing (>= MIN_STEP_STEPS), repeated 3x -- exactly 3
    steps, 0 taps, on a 45-step hold (the pattern's own length; no trailing partial cycle)."""
    n = 45
    force = np.full((n, N_FEET), 600.0)
    for cycle in range(3):
        start = cycle * 15 + 5
        force[start:start + 10, 0] = 0.0  # a 10-step swing on foot 0 (>= MIN_STEP_STEPS = 5)
    d = _episode(n, force=force)
    m = P.episode_stand_metrics(d, 0, dt=0.02)
    assert m["steps_per_s"] == pytest.approx(3 / (n * 0.02))
    assert m["taps_per_s"] == 0.0


def test_a_short_tap_is_not_counted_as_a_step(P):
    n = 45
    force = np.full((n, N_FEET), 600.0)
    force[10:13, 0] = 0.0  # a 3-step blip, under MIN_STEP_STEPS (5) -- a tap
    d = _episode(n, force=force)
    m = P.episode_stand_metrics(d, 0, dt=0.02)
    assert m["steps_per_s"] == 0.0
    assert m["taps_per_s"] == pytest.approx(1 / (n * 0.02))


# ================================================================================================ clearance:
# median-over-feet (the STILL/REF statistic) vs report-only max-over-feet
def test_clearance_median_vs_max_over_feet(P):
    n = 60
    rng = np.random.default_rng(0)
    foot_z = np.full((n, N_FEET), 0.03)
    spreads = []
    for f in range(N_FEET):
        amp = 0.01 * (f + 1)  # foot f's own p95-p5 grows with f (foot 5 is the widest)
        foot_z[:, f] += amp * np.sin(np.linspace(0, 8 * np.pi, n))
        spreads.append(float(np.percentile(foot_z[:, f], 95) - np.percentile(foot_z[:, f], 5)))
    d = _episode(n, foot_z=foot_z)
    med, mx = P.clearance_stats(d, np.ones(n, dtype=bool))
    assert med == pytest.approx(float(np.median(spreads)), abs=1e-9)
    assert mx == pytest.approx(float(np.max(spreads)), abs=1e-9)
    assert mx > med  # the max-over-feet figure is never smaller than the median it is reported beside


def test_clearance_none_under_the_minimum_row_floor(P):
    d = _episode(10)  # under MIN_EP_ROWS (25)
    med, mx = P.clearance_stats(d, np.ones(10, dtype=bool))
    assert med is None and mx is None


# ================================================================================================ hold identification
def test_stand_hold_index_never_matches_a_zero_speed_turn(P):
    labels = ["stand", "fwd", "lat_l", "diag", "back", "turn_l90", "turn_r90", "move_turn"]
    assert P.stand_hold_index(labels) == 0
    assert P.stand_hold_index(["fwd", "turn_l90"]) is None  # no "stand" label at all -> None, not a guess


def test_stand_onset_pairs_field_hard_and_field_turn(P):
    labels = ["stand", "fwd", "lat_l", "diag", "back", "turn_l90", "turn_r90", "move_turn"]
    speeds = [0.0, 0.25, 0.22, 0.25, 0.22, 0.0, 0.0, 0.22]
    pairs = P.stand_onset_pairs(labels, speeds)
    assert pairs == [(0, 1, "fwd")]  # the opening stand leads straight into fwd

    ft_labels = ["fwd", "stand", "turn_l90", "turn_r90"]
    ft_speeds = [0.25, 0.0, 0.0, 0.0]
    assert P.stand_onset_pairs(ft_labels, ft_speeds) == []  # field_turn's stand leads into a turn: no onset pair


# ================================================================================================ onset latency
def test_onset_latency_reach_and_hold(P):
    n = 100
    dt = 0.02
    cmd = np.zeros((n, 3))
    cmd[:, 0] = 0.25  # commanded speed 0.25 m/s along body x
    vel = np.zeros((n, 3))
    vel[20:, 0] = 0.25  # the body instantly reaches full speed at step 20 and holds it
    seg = np.ones(n, dtype=np.int16)
    d = _episode(n, segment_id=seg, cmd_applied=cmd, root_lin_vel_b=vel)
    got = P.episode_onset_latency(d, dt, move_idx=1)
    # the 5-step trailing mean needs 3/5 steps >= 20 to clear 0.5 x cmd (0.125 m/s): first true at t = 22
    assert got == pytest.approx(22 * dt, abs=1e-9)


def test_onset_latency_censored_when_never_reached(P):
    n = 60
    dt = 0.02
    cmd = np.zeros((n, 3))
    cmd[:, 0] = 0.25
    vel = np.zeros((n, 3))
    vel[:, 0] = 0.05  # well under half the commanded speed, forever
    seg = np.ones(n, dtype=np.int16)
    d = _episode(n, segment_id=seg, cmd_applied=cmd, root_lin_vel_b=vel)
    assert P.episode_onset_latency(d, dt, move_idx=1) is None


def test_onset_latency_zero_command_is_undefined(P):
    n = 20
    d = _episode(n, segment_id=np.ones(n, dtype=np.int16))  # cmd_applied defaults to all zero
    assert P.episode_onset_latency(d, 0.02, move_idx=1) is None


# ================================================================================================ still clauses
def test_still_clauses_boundary_is_inclusive(P):
    summary = {"steps_per_s": 0.5, "clearance_m": 0.05, "off_ground_share": 0.10,
              "hover_episodes_per_100": 5.0, "taps_per_s": 1.0, "ks_m_s": 0.5}
    clauses = P.still_clauses(summary)
    assert all(ok for _v, _t, ok in clauses.values())  # every value sits exactly AT its threshold -> pass
    summary["steps_per_s"] = 0.50001
    assert P.still_clauses(summary)["steps_per_s"][2] is False


def test_still_clauses_missing_value_fails_closed(P):
    clauses = P.still_clauses(None)
    assert all(ok is False for _v, _t, ok in clauses.values())
    assert set(clauses) == set(P.STILL_THRESHOLDS)


# ================================================================================================ end to end on a
# tiny on-disk stamp (head_summary / the CLI), including PENDING
def _write_stamp(tmp_path: Path, n_episodes: int = 2, n_steps: int = 100) -> Path:
    stamp = tmp_path / "stamp"
    (stamp / "raw").mkdir(parents=True)
    (stamp / "metrics").mkdir()
    (stamp / "complete.json").write_text("{}")
    hold_labels = ["settle", "stand", "fwd"]
    env_index_map = [{"env_index": i, "label": "stand", "wave": 0} for i in range(n_episodes)]
    run_meta = {"hold_labels": hold_labels, "hold_speeds": [0.0, 0.0, 0.25], "dt": 0.02,
                "env_index_map": env_index_map}
    (stamp / "run_meta.json").write_text(json.dumps(run_meta))
    for i in range(n_episodes):
        seg = np.array([0] * 10 + [1] * (n_steps - 10), dtype=np.int16)
        np.savez(stamp / "raw" / f"episode_{i:02d}.npz", foot_force_norm=np.full((n_steps, N_FEET), 600.0),
                foot_pos_w=np.zeros((n_steps, N_FEET, 3)) + np.array([0, 0, 0.03]),
                foot_lin_vel_w=np.zeros((n_steps, N_FEET, 3)), terrain_z=np.zeros((n_steps, N_FEET)),
                root_pos_w=np.full((n_steps, 3), [0.0, 0.0, 1.0]), root_lin_vel_b=np.zeros((n_steps, 3)),
                heading_err=np.zeros(n_steps), segment_id=seg, steady_mask=np.ones(n_steps, dtype=bool),
                crab_failure=np.zeros(n_steps, dtype=bool), cmd_applied=np.zeros((n_steps, 3)))
    return stamp


def test_head_summary_end_to_end_and_pending(P, tmp_path):
    stamp = _write_stamp(tmp_path)
    summary = P.head_summary(str(stamp))
    assert summary["n"] == 2
    assert summary["steps_per_s"] == 0.0
    assert summary["hover_episodes"] == 0
    assert P.head_summary(None) is None
    assert P.head_summary(str(tmp_path / "does_not_exist")) is None  # PENDING, not a crash


def test_cli_stamp_writes_json(P, tmp_path):
    stamp = _write_stamp(tmp_path)
    out = tmp_path / "out.json"
    rc = subprocess.run([sys.executable, str(TOOL_PY), "--stamp", str(stamp), "--json", str(out)],
                        capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    result = json.loads(out.read_text())
    assert result["summary"]["n"] == 2
    assert "steps_per_s" in result["still_clauses"]
