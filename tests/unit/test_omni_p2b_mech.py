# SPDX-License-Identifier: BSD-3-Clause
"""``p2_mech.py`` (P2B_PLAN step 1/6: the field_hard fall/swing mechanism classifier). CPU only, pure numpy --
nothing here imports Isaac or reads a real eval stamp.

Pinned on synthetic fixtures, not real data (the module's own ``--self-check`` is what validates against the real
``P2__S1_s{16,17,18}`` field_hard raws and the plan's saved 36/34/17/7/6 category split):

* :func:`projected_gravity_b` / :func:`rel_terrain_to_neighbours` against hand-derived values for simple
  single-axis quaternions and height arrays;
* :func:`classify_fall`'s onset duration, its five-category cascade (one synthetic fall per category, each
  isolating the ONE geometric/force cue that should decide it), its axis label and the travel-frame swap the
  module docstring describes (a lateral-travel fall with a large BODY pitch reads as travel-frame "roll");
* :func:`extract_swings`'s apex (terrain-relative, not root-relative) and its pit-vs-ordinary split at the
  liftoff margin;
* :func:`walking_root_heights`'s steady + moving-hold mask;
* :func:`newest_stamp` / :func:`stamp_complete` and :func:`mech_report` end to end on a tiny on-disk stamp (two
  seeds, one fall, one clean episode), including the PENDING case (no stamp on disk -> an empty report, not a
  crash);
* the CLI (``main``) writing a JSON summary file.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL_PY = (REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-07_omni_phase2/"
                   "scripts/p2_mech.py")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def M():
    return _load(TOOL_PY, "p2_mech__test")


# ================================================================================================ quaternion /
# gravity helpers (hand-derivable: a pure rotation about one body axis by theta tilts the gravity reading by
# exactly theta, entirely onto the OTHER horizontal axis)
def _pitch_quat(theta: float) -> np.ndarray:
    return np.array([np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0])


def _roll_quat(theta: float) -> np.ndarray:
    return np.array([np.cos(theta / 2), np.sin(theta / 2), 0.0, 0.0])


def test_projected_gravity_b_identity_and_known_tilts(M):
    g0 = M.projected_gravity_b(np.array([[1.0, 0.0, 0.0, 0.0]]))[0]
    assert np.allclose(g0, [0.0, 0.0, -1.0])

    g_pitch = M.projected_gravity_b(_pitch_quat(0.2)[None, :])[0]
    assert g_pitch[0] == pytest.approx(np.sin(0.2), abs=1e-9)
    assert g_pitch[1] == pytest.approx(0.0, abs=1e-9)

    g_roll = M.projected_gravity_b(_roll_quat(-0.3)[None, :])[0]
    assert g_roll[1] == pytest.approx(np.sin(0.3), abs=1e-9)  # roll(-0.3) -> gy = -sin(-0.3) = +sin(0.3)
    assert g_roll[0] == pytest.approx(0.0, abs=1e-9)

    tilt = np.arccos(np.clip(-g_pitch[2], -1, 1))
    assert tilt == pytest.approx(0.2, abs=1e-9)


def test_rel_terrain_to_neighbours_basic(M):
    # one foot (index 0) dips 0.1 m below the other five, which are all flat at 0.0
    terrain_z = np.array([[-0.1, 0.0, 0.0, 0.0, 0.0, 0.0]])
    rel = M.rel_terrain_to_neighbours(terrain_z)
    assert rel[0, 0] == pytest.approx(-0.1, abs=1e-9)
    assert np.allclose(rel[0, 1:], 0.02, atol=1e-9)  # each flat foot reads +0.1/5 above the OTHER five's mean


def test_leg_force_masks_excludes_hip_links(M):
    leg_of, femur_tibia = M.leg_force_masks(M.DEFAULT_LEG_LINK_NAMES)
    assert leg_of.shape == (18,)
    assert femur_tibia.sum() == 12  # 6 legs x (Femur, Tibia), Hip excluded
    names = M.DEFAULT_LEG_LINK_NAMES
    for i, n in enumerate(names):
        assert femur_tibia[i] == (not n.endswith("Hip"))


# ================================================================================================ a minimal
# per-episode fixture: 6 feet at fixed body-frame xy (two clearly "tip-side" front feet, index 0 the more extreme
# of the two so argmax is unambiguous), a root that does not translate, and a caller-supplied quat/terrain/force
# trajectory over N steps with crab_failure true only on the last.
FOOT_XY = np.array([[0.31, 0.2], [0.30, -0.2], [0.0, 0.25], [0.0, -0.25], [-0.3, 0.2], [-0.3, -0.2]])
N_FEET = 6


def _episode(theta: list[float], dt: float = 0.1, terrain_z_tip=None, terrain_z_other=None,
             leg_force_hot: tuple[int, float] | None = None, tip_mode: str = "pitch") -> dict:
    """``tip_mode`` picks which single-axis quaternion ramp drives the tilt (so the tip-side foot ends up the
    front-left foot, index 0, for every category test -- only the TERRAIN/FORCE cue under test differs)."""
    n = len(theta)
    quat_fn = _pitch_quat if tip_mode == "pitch" else _roll_quat
    root_quat_w = np.stack([quat_fn(t) for t in theta])
    root_pos_w = np.zeros((n, 3))
    root_pos_w[:, 2] = 1.0
    foot_pos_w = np.zeros((n, N_FEET, 3))
    foot_pos_w[:, :, :2] = FOOT_XY[None, :, :]
    terrain_z = np.zeros((n, N_FEET))
    if terrain_z_other is not None:
        # ONE trailing foot only (index 4): raising all five other feet would drag down the tip foot's OWN
        # relative baseline too (rel_terrain_to_neighbours compares each foot against the other five's mean), so
        # a single outlier keeps the tip foot's baseline inside the no-feature margin while still tripping the
        # trailing foot's own check.
        terrain_z[:, 4] = terrain_z_other
    if terrain_z_tip is not None:
        terrain_z[:, 0] = terrain_z_tip
    leg_contact_force = np.full((n, 18), 10.0)
    if leg_force_hot is not None:
        col, val = leg_force_hot
        leg_contact_force[:, col] = val
    crab_failure = np.zeros(n, dtype=bool)
    crab_failure[-1] = True
    return {
        "dt": np.asarray(dt), "segment_id": np.zeros(n, dtype=np.int16),
        "root_quat_w": root_quat_w, "root_pos_w": root_pos_w, "foot_pos_w": foot_pos_w,
        "terrain_z": terrain_z, "leg_contact_force": leg_contact_force,
        "travel_yaw": np.zeros(n), "crab_failure": crab_failure, "done": crab_failure,
        # extract_swings / walking_root_heights read these too; a fixture that only exercises classify_fall can
        # ignore them, but every episode written to disk for mech_report / the CLI needs the full set.
        "foot_force_norm": np.full((n, N_FEET), 300.0), "steady_mask": np.ones(n, dtype=bool),
    }


THETA_RAMP = [0.0, 0.05, 0.10, 0.14, 0.16, 0.20, 0.25, 0.30, 0.40, 0.50]  # crosses 0.15 between steps 3 and 4
HOLD_LABELS = ["only_hold"]


def test_classify_fall_onset_duration_from_explicit_tilt_crossing(M):
    ep = _episode(THETA_RAMP, dt=0.1)
    rec = M.classify_fall(ep, HOLD_LABELS)
    assert rec is not None
    # last index with tilt < 0.15 is 3 (theta 0.14); i_fail is 9 -> onset_s = (9-3)*0.1
    assert rec["onset_s"] == pytest.approx(0.6, abs=1e-9)
    assert rec["hold"] == "only_hold"


def test_classify_fall_returns_none_without_crab_failure(M):
    ep = _episode(THETA_RAMP)
    ep["crab_failure"] = np.zeros(len(THETA_RAMP), dtype=bool)
    assert M.classify_fall(ep, HOLD_LABELS) is None


@pytest.mark.parametrize("theta, expect_sign", [(list(THETA_RAMP), "pitch_fwd"),
                                                 ([-t for t in THETA_RAMP], "pitch_back")])
def test_classify_fall_pitch_axis_label_and_sign(M, theta, expect_sign):
    rec = M.classify_fall(_episode(theta), HOLD_LABELS)
    assert rec["cls"] == expect_sign


def test_classify_fall_roll_axis_label(M):
    rec = M.classify_fall(_episode(THETA_RAMP, tip_mode="roll"), HOLD_LABELS)
    assert rec["cls"] == "roll"


def test_classify_fall_travel_frame_swaps_axis_label(M):
    """A BODY-frame pitch fall (large gx, gy=0) with the commanded travel direction lateral (travel_yaw = pi/2)
    reads as travel-frame "roll" -- the lat_l / turn_* phenomenon the module docstring describes."""
    ep = _episode(THETA_RAMP, tip_mode="pitch")
    ep["travel_yaw"] = np.full(len(THETA_RAMP), np.pi / 2)
    rec = M.classify_fall(ep, HOLD_LABELS)
    assert rec["cls"] == "roll"


def test_classify_fall_category_pit_tip(M):
    ep = _episode(THETA_RAMP, terrain_z_tip=-0.5)  # far below the flat (0.0) neighbours
    rec = M.classify_fall(ep, HOLD_LABELS)
    assert rec["cat"] == "pit (tip-side foot)"
    assert rec["pit_tip"] is True


def test_classify_fall_category_block_tip(M):
    ep = _episode(THETA_RAMP, terrain_z_tip=+0.5)
    rec = M.classify_fall(ep, HOLD_LABELS)
    assert rec["cat"] == "tip-side foot on/at block"
    assert rec["blk_tip"] is True


def test_classify_fall_category_leg_strike(M):
    ep = _episode(THETA_RAMP, leg_force_hot=(0, 300.0))  # column 0 = FL_Femur, well over the 200 N floor
    rec = M.classify_fall(ep, HOLD_LABELS)
    assert rec["cat"] == "leg strike (femur/tibia>200N)"
    assert rec["leg_max_n"] == pytest.approx(300.0)


def test_classify_fall_category_leg_strike_ignores_hip_links(M):
    # a Hip link (index 1 = FL_Hip) spiking past 200 N must NOT trigger leg strike
    ep = _episode(THETA_RAMP, leg_force_hot=(1, 300.0))
    rec = M.classify_fall(ep, HOLD_LABELS)
    assert rec["cat"] == "no feature (all feet flat 1.5s)"


def test_classify_fall_category_trailing_feature(M):
    ep = _episode(THETA_RAMP, terrain_z_other=0.15)  # a block under ONE trailing foot, not the tip
    rec = M.classify_fall(ep, HOLD_LABELS)
    assert rec["cat"] == "feature under trailing feet"
    assert rec["pit_tip"] is False and rec["blk_tip"] is False
    assert rec["blk_any"] is True


def test_classify_fall_category_no_feature(M):
    rec = M.classify_fall(_episode(THETA_RAMP), HOLD_LABELS)
    assert rec["cat"] == "no feature (all feet flat 1.5s)"


def test_classify_fall_leg_link_names_reorders_the_mask(M):
    """Passing a leg_link_names whose column 0 is a Hip link (instead of the default's Femur) must change which
    column the leg-strike check reads."""
    reordered = ["FL_Hip"] + M.DEFAULT_LEG_LINK_NAMES[1:]
    ep = _episode(THETA_RAMP, leg_force_hot=(0, 300.0))
    rec_default = M.classify_fall(ep, HOLD_LABELS)
    rec_reordered = M.classify_fall(ep, HOLD_LABELS, leg_link_names=reordered)
    assert rec_default["cat"] == "leg strike (femur/tibia>200N)"
    assert rec_reordered["cat"] == "no feature (all feet flat 1.5s)"


# ================================================================================================ swings
def _swing_episode(contact_pattern: list[bool], liftoff_rel: float = 0.0,
                    apex_row: int | None = None, apex_clear: float = 0.0) -> dict:
    """One foot (index 0) only; the other five stay in constant contact throughout, flat at terrain_z = 0.
    ``terrain_z`` for foot 0 is CONSTANT at ``liftoff_rel`` over time, so the liftoff-margin check reads the same
    value regardless of exactly which row the swing starts on. ``apex_row`` separately elevates ``foot_pos_w`` at
    one chosen row (clearance = ``apex_clear``) so the apex test isolates the max-over-the-window behaviour from
    the liftoff-margin behaviour."""
    n = len(contact_pattern)
    foot_force_norm = np.full((n, N_FEET), 300.0)
    foot_force_norm[:, 0] = np.where(contact_pattern, 300.0, 0.0)
    foot_pos_w = np.zeros((n, N_FEET, 3))
    terrain_z = np.zeros((n, N_FEET))
    terrain_z[:, 0] = liftoff_rel
    if apex_row is not None:
        foot_pos_w[apex_row, 0, 2] = apex_clear + liftoff_rel
    return {"foot_force_norm": foot_force_norm, "foot_pos_w": foot_pos_w, "terrain_z": terrain_z,
            "segment_id": np.zeros(n, dtype=np.int16)}


def test_extract_swings_apex_is_terrain_relative_and_bounded_both_sides(M):
    # contact, swing (3 steps), contact -- one complete, bounded swing on foot 0; the apex sits mid-swing (row 2),
    # well above the window's other rows (clearance 0 there), so picking it out proves a real max, not a fluke
    pattern = [True, False, False, False, True]
    ep = _swing_episode(pattern, liftoff_rel=0.0, apex_row=2, apex_clear=0.08)
    swings = M.extract_swings(ep, HOLD_LABELS)
    assert len(swings) == 1
    s = swings[0]
    assert s["foot"] == 0 and s["start"] == 1 and s["end"] == 4
    assert s["apex_m"] == pytest.approx(0.08, abs=1e-5)
    assert s["kind"] == "ordinary"


def test_extract_swings_pit_vs_ordinary_by_liftoff_margin(M):
    pattern = [True, False, False, True]
    pit = M.extract_swings(_swing_episode(pattern, liftoff_rel=-0.05), HOLD_LABELS)[0]
    ordinary = M.extract_swings(_swing_episode(pattern, liftoff_rel=-0.01), HOLD_LABELS)[0]
    assert pit["kind"] == "pit"
    assert ordinary["kind"] == "ordinary"


def test_extract_swings_drops_an_unbounded_run_at_episode_edges(M):
    # airborne from step 0 (no recorded liftoff) and still airborne at the last step (no recorded touchdown)
    pattern = [False, False, True, True, False]
    swings = M.extract_swings(_swing_episode(pattern), HOLD_LABELS)
    assert swings == []


# ================================================================================================ walking root
# height
def test_walking_root_heights_filters_steady_and_moving_holds(M):
    n = 6
    ep = {
        "segment_id": np.array([0, 0, 1, 1, 0, 1], dtype=np.int16),  # hold 0 = stand (speed 0), hold 1 = moving
        "steady_mask": np.array([True, False, True, True, True, False]),
        "root_pos_w": np.zeros((n, 3)),
        "terrain_z": np.zeros((n, N_FEET)),
    }
    ep["root_pos_w"][:, 2] = 1.0
    heights = M.walking_root_heights(ep, ["stand", "fwd"], hold_speeds=[0.0, 0.25])
    # only step index 2 and 3 are both steady AND on the moving hold (index 1)
    assert heights == [1.0, 1.0]


# ================================================================================================ on-disk stamp /
# mech_report / CLI end to end
def _write_stamp(mod, root: Path, name: str, row: str, seed: int, ts: str, episodes: list[dict], terms: list[str]):
    stamp = root / "lr" / name / row / f"seed{seed:03d}" / ts
    (stamp / "raw").mkdir(parents=True)
    (stamp / "p2_complete.json").write_text("{}")
    meta = {"hold_labels": HOLD_LABELS, "hold_speeds": [0.25], "termination_reason": terms,
            "leg_link_names": mod.DEFAULT_LEG_LINK_NAMES}
    (stamp / "run_meta.json").write_text(json.dumps(meta))
    for i, ep in enumerate(episodes):
        np.savez(stamp / "raw" / f"episode_{i:02d}.npz", **ep)
    return stamp


def test_newest_stamp_and_stamp_complete(M, tmp_path):
    name = "T_head"
    assert M.newest_stamp(name, "field_hard__omni", 1, lrout=tmp_path / "lr") is None
    stamp = _write_stamp(M, tmp_path, name, "field_hard__omni", 1, "2026-01-01_00-00-00",
                          episodes=[_episode(THETA_RAMP)], terms=["fall"])
    assert M.stamp_complete(stamp) is True
    got = M.newest_stamp(name, "field_hard__omni", 1, lrout=tmp_path / "lr")
    assert got == str(stamp)
    # an incomplete later stamp (no p2_complete.json / complete.json) is not picked up
    incomplete = tmp_path / "lr" / name / "field_hard__omni" / "seed001" / "2026-01-02_00-00-00" / "raw"
    incomplete.mkdir(parents=True)
    assert M.newest_stamp(name, "field_hard__omni", 1, lrout=tmp_path / "lr") == str(stamp)


def test_mech_report_pending_when_stamp_missing(M, tmp_path):
    rep = M.mech_report("NOPE", lrout=tmp_path / "lr")
    assert rep["n_episodes"] == 0 and rep["n_falls"] == 0 and rep["stamps"] == []
    assert rep["onset_median_s"] is None


def test_mech_report_end_to_end_two_seeds(M, tmp_path):
    fall_ep = _episode(THETA_RAMP, terrain_z_tip=-0.5)  # a pit fall
    clean_ep = dict(_episode([0.0] * len(THETA_RAMP)))
    clean_ep["crab_failure"] = np.zeros(len(THETA_RAMP), dtype=bool)
    clean_ep["done"] = clean_ep["crab_failure"]
    clean_ep["foot_force_norm"] = np.full((len(THETA_RAMP), N_FEET), 300.0)
    clean_ep["steady_mask"] = np.ones(len(THETA_RAMP), dtype=bool)

    _write_stamp(M, tmp_path, "T_head", "field_hard__omni", 1, "2026-01-01_00-00-00",
                 episodes=[fall_ep, clean_ep], terms=["fall", "schedule_complete"])
    _write_stamp(M, tmp_path, "T_head", "field_hard__omni", 2, "2026-01-01_00-00-00",
                 episodes=[clean_ep], terms=["schedule_complete"])

    rep = M.mech_report("T_head", seeds=(1, 2), lrout=tmp_path / "lr")
    assert rep["n_episodes"] == 3
    assert rep["n_falls"] == 1
    assert rep["falls_by_category"] == {"pit (tip-side foot)": 1}
    assert rep["onset_median_s"] == pytest.approx(0.6, abs=1e-9)
    assert len(rep["stamps"]) == 2


def test_main_cli_writes_json_summary(M, tmp_path):
    fall_ep = _episode(THETA_RAMP, terrain_z_tip=-0.5)
    _write_stamp(M, tmp_path, "T_head", "field_hard__omni", 1, "2026-01-01_00-00-00",
                 episodes=[fall_ep], terms=["fall"])
    out = tmp_path / "out.json"
    rc = M.main(["--heads", "T_head", "--seeds", "1", "--out-root", str(tmp_path), "--out", str(out)])
    assert rc == 0
    doc = json.loads(out.read_text())
    assert doc["T_head"]["n_falls"] == 1
    assert doc["T_head"]["falls_by_category"] == {"pit (tip-side foot)": 1}


def test_self_check_handles_no_data_without_crashing(M, tmp_path):
    assert M.self_check(lrout=tmp_path / "lr") == 1
