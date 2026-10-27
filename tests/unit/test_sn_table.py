"""Stand-network campaign (``2026-10-11_stand_network``): the TABLE STAND-UP test's pure pieces on CPU (no Isaac / Kit /
GPU).

Pinned here:

* the retracted start poses (``sn_table_core.retracted_pose``): ``rod_min`` = both lead screws at their minimum rod
  length -- the hip's screw minimum lies past the USD hard stop, so the stop (+ margin) binds; the knee's screw minimum
  re-solved at that hip OPENS the knee; ``knee_folded`` = the largest fold whose tibia hull keeps 2 cm from the hip
  plate (the knee screw then near its maximum); mirroring (right knees negated, hips equal, Body_Hip / cam 0), every
  joint inside the USD limits, the rod round trips through ``crab_hex_linkage``;
* the collision-hull constants against ``assets/crab.usda`` (hip-plate stadium, femur stadium, tibia outline, knee
  joint offset, body box, joint limits, the coplanar leg links that make the hardware fold unrepresentable);
* the spawn-height arithmetic, the oriented-box lowest point / tilt, the table / leg clearances, the toes at spawn;
* the STOOD classifier on synthetic traces (each clause, its boundaries, the 95 % feet tolerance for post-settle load
  flickers with the gap clause still per-step, stood_strict, standing-since, lift-off with a bounce, feet counting),
  and the merge table's near-miss columns (strict / w5 share / w min);
* controller routing: ``stand_net`` = ``sn_common.stand_net_input`` on the harness obs (S1's estimate in place) with
  mask 1, routed by ``sn_common.route_actions`` -- bitwise ``sn_dagger_core.launcher_actions``; ``r`` = R's target with
  the D3 string on stand rows, S1 elsewhere; ``s1`` / ``zero``;
* R at the retracted start: SEAT on all six legs, target exactly zero;
* the reset event with fake assets; the seeded jitter; the kinematic slew timeline;
* ``sn_table_test.py``: compiles, imports nothing of Isaac before Kit, its pre-Kit refusals, ``--emit-facts`` and
  ``--merge`` end to end; ``sn_table_test.sh``: ``bash -n``, its busy pattern, ``--dry-run`` launches and writes nothing,
  and its queue runner (``--foreground``, a fake isaac python + ``systemd-run``): jobs serial with the right args,
  sentinels, facts first, the merge last, the default smoke includes a 1-env video job, the default full run includes
  the zero baseline, the queue stops at the first failing job, and a failed merge ends the queue with rc=6, not 0.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import math
import os
import py_compile
import re
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[2]
TASK = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
SCRIPTS = TASK / "experiments/2026-10-11_stand_network/scripts"
CORE_PATH = SCRIPTS / "sn_table_core.py"
TEST_PATH = SCRIPTS / "sn_table_test.py"
SH_PATH = SCRIPTS / "sn_table_test.sh"
COMMON_PATH = SCRIPTS / "sn_common.py"
DAGGER_CORE_PATH = SCRIPTS / "sn_dagger_core.py"
USD = REPO / "assets/crab.usda"
W, P0, P1, NP = 1149, 207, 216, 75
DT = 0.02


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


@pytest.fixture(scope="module")
def C():
    return _load(CORE_PATH, "_t_snt_core")


@pytest.fixture(scope="module")
def SN():
    return _load(COMMON_PATH, "_t_snt_common")


@pytest.fixture(scope="module")
def DC():
    return _load(DAGGER_CORE_PATH, "_t_snt_dagger_core")


@pytest.fixture(scope="module")
def T():
    return _load(TEST_PATH, "_t_snt_test")


@pytest.fixture(scope="module")
def poses(C):
    return {p: C.retracted_pose(p) for p in C.POSES}


# ===================================================================================== USD helpers
def _block(text: str, header: str) -> str:
    i = text.index(header)
    j = text.index("xformOpOrder", i) if "Mesh" in header or "Cube" in header else text.index("}", i)
    return text[i:j]


def _points(block: str):
    m = re.search(r"point3f\[\] points = \[(.*?)\]", block, re.S)
    return [tuple(float(v) for v in t.split(",")) for t in re.findall(r"\(([^()]*)\)", m.group(1))]


def _f(block: str, key: str):
    m = re.search(rf"{re.escape(key)} = (\([^)]*\)|[-0-9.e]+)", block)
    v = m.group(1)
    return tuple(float(x) for x in v.strip("()").split(",")) if v.startswith("(") else float(v)


@pytest.fixture(scope="module")
def usd():
    return USD.read_text()


# ===================================================================================== the retracted poses
def test_rod_min_pose_is_both_screws_at_minimum_hip_stop_binds(C, poses):
    lk = C.linkage()
    p = poses["rod_min"]
    lim = C.joint_limits_rad()
    m = math.radians(C.LIMIT_MARGIN_DEG)
    t = lambda x: torch.tensor(float(x), dtype=torch.float64)  # noqa: E731
    # the hip screw's minimum angle lies PAST the hard stop -> the stop (+ margin) binds
    assert abs(float(lk.hip_actuator_length(t(p["hip_rod_min_angle_rad"]))) - lk.HIP_LEN_MIN_M) < 1e-9
    assert p["hip_rod_min_angle_rad"] < lim["hip"][0]
    assert p["hip_binding"] == "joint_stop"
    assert p["hip_rad"] == pytest.approx(lim["hip"][0] + m, abs=1e-12)
    assert p["hip_deg"] == pytest.approx(-44.5, abs=1e-9)
    assert lk.HIP_LEN_MIN_M < p["hip_rod_m"] < lk.HIP_LEN_MIN_M + 0.01           # 7 mm off the screw's end
    # the knee screw at its minimum, re-solved at that hip: OPENS the knee (negative = toe outboard)
    assert p["knee_binding"] == "rod_min"
    assert p["knee_rod_m"] == pytest.approx(lk.KNEE_LEN_MIN_M, abs=1e-9)
    assert float(lk.knee_actuator_length(t(p["hip_rad"]), t(p["knee_left_rad"]))) == pytest.approx(0.45, abs=1e-9)
    assert p["knee_left_rad"] < 0.0 and p["knee_interior_deg"] > 90.0
    assert lim["knee_left"][0] < p["knee_left_rad"] < lim["knee_left"][1]
    # outside the action window: no controller can hold it (|a| > 1 on both rods)
    assert p["hip_hold_action"] < -1.0 and abs(p["knee_hold_action_left"]) > 1.0


def test_knee_folded_pose_is_the_largest_clear_fold(C, poses):
    lk = C.linkage()
    p = poses["knee_folded"]
    lim = C.joint_limits_rad()
    assert p["hip_rad"] == poses["rod_min"]["hip_rad"]
    assert p["knee_binding"] == "plate_clearance"                                # the plate binds before the stop
    k = p["knee_left_rad"]
    assert C.defaults_rad()["knee_left"] < k < lim["knee_left"][1]
    assert C.link_clearances(p["hip_rad"], k)["min_m"] >= C.PLATE_CLEARANCE_M
    step = p["fold_search"]["step_rad"]
    assert C.link_clearances(p["hip_rad"], k + step)["tibia_plate_m"] < C.PLATE_CLEARANCE_M
    # the hardware fold (the +85-deg stop) would put the tibia hull INTO the hip plate
    assert C.link_clearances(p["hip_rad"], lim["knee_left"][1])["tibia_plate_m"] == 0.0
    # folding the knee = EXTENDING the knee screw (past mid-stroke)
    assert p["knee_rod_m"] > 0.5 * (lk.KNEE_LEN_MIN_M + lk.KNEE_LEN_MAX_M)


def test_pose_mirroring_and_limits(C, poses, usd):
    lim = C.joint_limits_rad()
    for name, p in poses.items():
        jp = p["joint_pos"]
        assert len(jp) == 24
        for leg in C.LEGS:
            assert jp[f"{leg}_Body_Hip_RevoluteJoint"] == 0.0 and jp[f"{leg}_Body_CamShaft_RevoluteJoint"] == 0.0
            assert jp[f"{leg}_Hip_Femur_RevoluteJoint"] == p["hip_rad"]
            k = jp[f"{leg}_Femur_Tibia_RevoluteJoint"]
            assert k == (p["knee_left_rad"] if leg in C.LEFT else -p["knee_left_rad"])
            klim = lim["knee_left"] if leg in C.LEFT else lim["knee_right"]
            assert klim[0] < k < klim[1], (name, leg)
            assert lim["hip"][0] < p["hip_rad"] < lim["hip"][1]
    # the limits the pose is checked against ARE the USD's (left and mirrored right)
    for joint, want in (("FL_Hip_Femur_RevoluteJoint", (-45.0, 60.0)), ("FR_Hip_Femur_RevoluteJoint", (-45.0, 60.0)),
                        ("FL_Femur_Tibia_RevoluteJoint", (-50.0, 85.0)), ("FR_Femur_Tibia_RevoluteJoint", (-85.0, 50.0))):
        b = _block(usd, f'def PhysicsRevoluteJoint "{joint}"')
        assert (_f(b, "float physics:lowerLimit"), _f(b, "float physics:upperLimit")) == want, joint
    assert tuple(math.degrees(x) for x in lim["knee_right"]) == pytest.approx((-85.0, 50.0))


def test_the_joint_order_matches_the_eval_stamp(C, T):
    rm = json.loads((Path(T.SPEC.REFS_ENV_STAMP) / "run_meta.json").read_text()) if \
        (Path(T.SPEC.REFS_ENV_STAMP) / "run_meta.json").is_file() else None
    if rm is None:
        pytest.skip("REFS stamp not present")
    assert set(rm["joint_names"]) == set(C.retracted_pose("rod_min")["joint_pos"])


# ===================================================================================== the USD collision hulls
def test_hull_constants_match_the_usd(C, usd):
    for leg in ("FL", "MR", "RR"):
        hip = _points(_block(usd, f'def Mesh "{leg}_Hip"'))
        # every hull point inside the stadium of radius PLATE_RADIUS_M around the core (the conservative radius)
        d = [math.hypot(y, max(0.0, abs(z) - C.PLATE_CORE_HALF_Z_M)) for _x, y, z in hip]
        assert max(d) <= C.PLATE_RADIUS_M and max(d) > C.PLATE_RADIUS_M - 5e-4, leg
        assert max(abs(y) for _x, y, _z in hip) == pytest.approx(2 * C.PLATE_CORE_U_M / 2, abs=1e-6)  # 0.127 wide
        assert max(abs(z) for _x, _y, z in hip) == pytest.approx(0.3048, abs=1e-4)
        fem = _points(_block(usd, f'def Mesh "{leg}_Femur"'))
        d = [math.hypot(max(0.0, abs(y) - 0.2921), z) for _x, y, z in fem]   # around the pivot -> knee segment
        assert max(d) <= C.FEMUR_RADIUS_M and max(d) > C.FEMUR_RADIUS_M - 5e-4, leg
    hb = _block(usd, 'def PhysicsRevoluteJoint "FL_Hip_Femur_RevoluteJoint"')
    assert _f(hb, "point3f physics:localPos0") == pytest.approx((0.0, 0.0, -C.PLATE_CORE_HALF_Z_M), abs=1e-4)
    tib = _points(_block(usd, 'def Mesh "FL_Tibia"'))
    hull = C._convex_hull([(y, z) for y, z in C.TIBIA_OUTLINE_YZ] + [(-y, z) for y, z in C.TIBIA_OUTLINE_YZ])
    for _x, y, z in tib:                                                 # the symmetrised outline covers the mesh
        assert C._inside_convex(hull, (y * 0.999999, z * 0.999999)) or any(
            abs(y - a) < 1e-6 and abs(z - b) < 1e-6 for a, b in hull), (y, z)
    assert min(z for _x, _y, z in tib) == pytest.approx(min(z for _y, z in C.TIBIA_OUTLINE_YZ), abs=1e-7)
    kb = _block(usd, 'def PhysicsRevoluteJoint "FL_Femur_Tibia_RevoluteJoint"')
    assert _f(kb, "point3f physics:localPos1") == pytest.approx((0.0, 0.0, C.TIBIA_KNEE_Z_M))
    body = _block(usd, 'def Cube "body"')
    assert _f(body, "double3 xformOp:scale") == pytest.approx(tuple(2 * h for h in C.BODY_HALF_M))
    assert tuple(2 * h for h in C.BODY_HALF_M) == pytest.approx(C.dims().BODY_SIZE_M)


def test_leg_links_are_coplanar_in_the_usd(usd):
    """Why the hardware fold cannot be modelled: hip plate, femur and tibia share one leg-plane offset."""
    xs = [_f(_block(usd, f'def Mesh "MR_{n}"'), "double3 xformOp:translate")[0] for n in ("Hip", "Femur", "Tibia")]
    assert xs == [0.0, 0.0, 0.0]
    c, s = math.cos(math.radians(15.0)), math.sin(math.radians(15.0))
    ref = _f(_block(usd, 'def Mesh "FL_Hip"'), "double3 xformOp:translate")
    for n in ("Femur", "Tibia"):
        tx, ty, _ = _f(_block(usd, f'def Mesh "FL_{n}"'), "double3 xformOp:translate")
        dx, dy = tx - ref[0], ty - ref[1]
        assert abs(c * dx - s * dy) < 1e-3, n                         # leg-frame x offset 0 after undoing the splay


# ===================================================================================== spawn / geometry
def test_spawn_height_arithmetic(C):
    assert C.TABLE_TOP_M == pytest.approx(25 * 0.0254)
    assert C.spawn_root_z() == pytest.approx(0.635 + 0.15875 + 0.005)
    assert C.TABLE_CENTER_Z_M == pytest.approx(0.5 * (C.TABLE_TOP_M + C.TABLE_BASE_Z_M))
    assert C.TABLE_HEIGHT_M == pytest.approx(C.TABLE_TOP_M - C.TABLE_BASE_Z_M)
    low, cz = C.lowest_z_and_cos_tilt(C.spawn_root_z(), 1.0, 0.0, 0.0, 0.0)
    assert low == pytest.approx(C.TABLE_TOP_M + C.SPAWN_CLEARANCE_M) and cz == pytest.approx(1.0)


def test_lowest_point_and_tilt_match_brute_force(C):
    g = np.random.default_rng(0)
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * np.array(C.BODY_HALF_M)
    for _ in range(50):
        q = g.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        rz = 0.8
        want = rz + (corners @ R.T)[:, 2].min()
        low, cz = C.lowest_z_and_cos_tilt(rz, w, x, y, z)
        assert low == pytest.approx(want, abs=1e-12) and cz == pytest.approx(R[2, 2], abs=1e-12)
    tq = torch.tensor([[1.0, 0, 0, 0], [math.cos(0.25), math.sin(0.25), 0, 0]])
    low, cz = C.lowest_z_and_cos_tilt(torch.tensor([1.0, 1.0]), tq[:, 0], tq[:, 1], tq[:, 2], tq[:, 3])
    assert torch.allclose(torch.acos(cz), torch.tensor([0.0, 0.5]), atol=1e-6)        # torch tensors work too


def test_table_blocks_no_leg_and_the_toes_at_spawn(C, poses):
    rz = C.spawn_root_z()
    for name, p in poses.items():
        tl = C.table_leg_clearance(p, rz)
        assert tl["lateral_clearance_m"] > 0.10, name                    # >= 0.15 m to the worst jittered corner
        assert not tl["any_toe_inside_table_footprint"], name
        assert not tl["toes_touch_floor_at_spawn"], name
        assert tl["hip_plate_bottom_world_z_m"] > 0.0
        for leg, toe in tl["toes"].items():
            assert toe["lateral_clear_of_worst_table_m"] > 0.15, (name, leg)
            assert toe["above_table_top"] == (name == "rod_min"), (name, leg)
    assert C.table_max_abs_y() == pytest.approx(max(abs(y) for _, y in C.table_corners(0.0, 0.03, math.radians(10))),
                                                abs=1e-12)
    assert C.in_table(0.44, 0.34) and not C.in_table(0.0, 0.36) and C.in_table(0.0, 0.36, margin=0.02)


def test_foot_fk_at_the_default_matches_the_stated_stance(C):
    d = C.defaults_rad()
    f = C.foot_points_body(d["hip"], d["knee_left"])
    assert abs(f["ML"]["toe"][1]) == pytest.approx(0.98, abs=0.01)                    # feet ~0.98 m lateral
    assert abs(f["FL"]["toe"][0]) == pytest.approx(0.39, abs=0.01)                    # +-0.39 m fore / aft
    assert f["ML"]["toe"][2] - C.FOOTPAD_HALF_M[2] == pytest.approx(-1.056, abs=0.002)  # toe bottom 1.056 below root


def test_slew_timeline(C, poses):
    lk = C.linkage()
    for name, p in poses.items():
        s = C.slew_timeline(p)
        t_hip = abs(float(lk.hip_actuator_length(torch.tensor(C.defaults_rad()["hip"], dtype=torch.float64)))
                    - p["hip_rod_m"]) / lk.HIP_SPEED_M_S
        assert s["hip_rod_arrives_s"] == pytest.approx(t_hip, abs=0.006), name
        assert 0.0 < s["toe_reaches_floor_s"] < s["hip_rod_arrives_s"], name
        assert s["hip_enters_action_window_s"] < s["hip_rod_arrives_s"]
        assert s["end_hip_rad"] == pytest.approx(C.defaults_rad()["hip"], abs=1e-9)
        assert 0.2 < s["belly_gap_at_end_if_lifted_m"] < 0.3


# ===================================================================================== STOOD classifier
def _trace(T=750, gap=0.26, feet=6, tilt=0.0, root=1.06, cam=0.0, term=False):
    def arr(v, dtype=float):
        return np.full(T, v, dtype=dtype) if np.isscalar(v) else np.asarray(v, dtype=dtype)

    return {"lowest_z": arr(gap) + 0.635, "cos_tilt": np.cos(arr(tilt)), "feet_on_floor": arr(feet, int),
            "root_z": arr(root), "cam_abs_max": arr(cam), "terminated": arr(term, bool)}


def test_classifier_clean_standup(C):
    T = 750
    t = (np.arange(T) + 1) * DT
    gap = np.where(t < 1.0, 0.005, np.minimum(0.26, 0.005 + (t - 1.0) * 0.1))
    feet = np.where(t < 1.5, 0, 6)
    r = C.classify(_trace(T, gap=gap, feet=feet), DT)
    assert r["stood"] and not r["reasons"]
    assert r["lift_off_s"] == pytest.approx(round((1.0 + 0.015 / 0.1) / DT) * DT, abs=DT + 1e-9)
    assert r["final_feet_on_floor"] == 6 and r["final_gap_m"] == pytest.approx(0.26)
    assert r["cams_homed_end"] and not r["on_table_end"]


@pytest.mark.parametrize("case,kw,reason", [
    ("never_lifts", dict(gap=0.005, feet=0), "belly gap"),
    ("four_feet", dict(feet=4), "feet on the floor"),
    ("tilt_spike", dict(tilt=np.where(np.arange(750) == 250, 0.6, 0.0)), "FALL"),
    ("fell_off", dict(gap=np.where(np.arange(750) < 300, 0.26, -0.30), feet=6), "FELL OFF"),
    ("sinks_late", dict(gap=np.where(np.arange(750) > 740, 0.05, 0.26)), "belly gap"),
    ("terminated", dict(term=np.arange(750) == 100), "terminated"),
])
def test_classifier_fail_clauses(C, case, kw, reason):
    r = C.classify(_trace(**kw), DT)
    assert not r["stood"], case
    assert any(reason in x for x in r["reasons"]), (case, r["reasons"])


def test_classifier_boundaries_and_window(C):
    w = int(round(C.STOOD_WINDOW_S / DT))
    assert C.classify(_trace(gap=C.STOOD_GAP_M, feet=C.STOOD_MIN_FEET), DT)["stood"]          # >= on both
    assert not C.classify(_trace(gap=C.STOOD_GAP_M - 1e-6), DT)["stood"]
    assert C.classify(_trace(tilt=C.FALL_TILT_RAD - 1e-6), DT)["stood"]
    assert not C.classify(_trace(tilt=C.FALL_TILT_RAD + 1e-6), DT)["stood"]
    T = 750
    early = np.where(np.arange(T) < T - w - 1, 0.0, 0.26)              # only the last window counts for (1)
    feet = np.where(np.arange(T) < T - w - 1, 0, 6)
    r = C.classify(_trace(gap=early, feet=feet), DT)
    assert r["stood"] and r["standing_since_s"] == pytest.approx((T - w) * DT)
    assert not C.classify(_trace(T=w - 1), DT)["stood"]                 # a smoke-length episode never STOOD
    assert C.classify(_trace(gap=-C.FELL_OFF_DROP_M + 1e-6, feet=6), DT)["fell_off_table"] is False


def test_classifier_feet_flicker_tolerance(C):
    """Post-settle load flickers: two feet unloading in the same step (4 feet) for up to 7 of the 150 window steps
    still STOOD (not strict); 8 such steps do not; the gap clause stays per-step."""
    T = 750
    w = int(round(C.STOOD_WINDOW_S / DT))
    assert w == 150 and C.stood_feet_steps_needed(w) == 143
    for n_bad, stood in ((0, True), (1, True), (7, True), (8, False), (20, False)):
        feet = np.full(T, 6)
        feet[T - w + 3: T - w + 3 + 2 * n_bad: 2] = 4                  # scattered single-step dips to 4 feet
        assert int((feet[T - w:] < 5).sum()) == n_bad
        r = C.classify(_trace(T, feet=feet), DT)
        assert r["stood"] is stood, (n_bad, r["reasons"])
        assert r["stood_strict"] is (n_bad == 0), n_bad
        assert r["feet_share_last_window"] == pytest.approx((w - n_bad) / w)
        assert r["min_feet_last_window"] == (6 if n_bad == 0 else 4)
        if not stood:
            assert any("feet on the floor" in x and f"{w - n_bad}/{w}" in x for x in r["reasons"]), r["reasons"]
    # dips BEFORE the window do not count, a dip to 0 feet counts like any other step
    feet = np.full(T, 6)
    feet[: T - w] = 0
    feet[T - 5] = 0
    r = C.classify(_trace(T, feet=feet), DT)
    assert r["stood"] and not r["stood_strict"] and r["min_feet_last_window"] == 0
    # the gap clause has no tolerance: ONE step back on the table inside the window fails
    gap = np.full(T, 0.26)
    gap[T - 10] = 0.0
    r = C.classify(_trace(T, gap=gap), DT)
    assert not r["stood"] and not r["stood_strict"] and any("belly gap" in x for x in r["reasons"])


def test_standing_since_with_flickers(C):
    T = 750
    w = int(round(C.STOOD_WINDOW_S / DT))
    gap = np.where(np.arange(T) < 300, 0.0, 0.26)
    feet = np.where(np.arange(T) < 310, 0, 6)
    feet[400] = feet[600] = feet[T - 2] = 4                              # flickers after standing
    r = C.classify(_trace(T, gap=gap, feet=feet), DT)
    assert r["stood"] and not r["stood_strict"]
    assert r["standing_since_s"] == pytest.approx(311 * DT)             # the first loaded step with the gap held
    assert r["standing_since_s"] <= (T - w + 1) * DT


def test_lift_off_ignores_a_bounce(C):
    T = 750
    i = np.arange(T)
    gap = np.full(T, 0.005)
    gap[100:110] = 0.05                                                    # 0.2 s bounce: not a lift-off
    gap[200:] = 0.2
    r = C.classify(_trace(T, gap=gap, feet=6), DT)
    assert r["first_gap_above_lift_s"] == pytest.approx(101 * DT)
    assert r["lift_off_s"] == pytest.approx(201 * DT)
    assert i.size == T


def test_feet_counting(C):
    force = np.array([[30.0, 30.0, 19.9, 300.0, 30.0, 30.0]])
    z = np.array([[0.02, 0.15, 0.02, 0.66, 0.151, 0.03]])           # 0.66 m: a foot on the TABLE does not count
    assert C.feet_on_floor(force, z).tolist() == [3]


def test_aggregate(C):
    rs = [C.classify(_trace(), DT), C.classify(_trace(gap=0.005, feet=0), DT),
          C.classify(_trace(tilt=0.7), DT)]
    a = C.aggregate(rs)
    assert a["n_envs"] == 3 and a["n_stood"] == 1 and a["stood_share"] == pytest.approx(1 / 3)
    assert a["falls"] == 1 and a["fell_off_table"] == 0 and a["n_lifted_off"] == 2
    assert a["final_feet_on_floor_hist"] == {"0": 1, "6": 2}
    assert a["n_stood_strict"] == 1 and a["min_feet_last_window"]["min"] == 0
    assert a["feet_share_last_window"]["min"] == pytest.approx(0.0) and a["feet_share_last_window"]["max"] == 1.0


def test_format_table_shows_a_feet_near_miss(C):
    T, w = 750, int(round(C.STOOD_WINDOW_S / DT))
    flick = np.full(T, 6)
    flick[T - 20] = flick[T - 40] = 4                                     # STOOD, not strict
    near = np.full(T, 6)
    near[T - w: T - w + 16: 2] = 4                                        # 8 dips: NOT STOOD (142/150)
    rs = [C.classify(_trace(), DT), C.classify(_trace(feet=flick), DT), C.classify(_trace(feet=near), DT)]
    assert [r["stood"] for r in rs] == [True, True, False] and [r["stood_strict"] for r in rs] == [True, False, False]
    merged = {"stats": {"rod_min": {"stand_net": C.aggregate(rs)}}}
    lines = C.format_table(merged).splitlines()
    head, row = lines[0].split(), lines[1].split()
    for col in ("STOOD", "strict", "share", "min"):
        assert col in head, head
    assert row[:6] == ["rod_min", "stand_net", "3", "2/3", "1", f"{142 / 150:.3f}"] and row[6] == "4", row
    # an older summary without the new keys still prints (dashes)
    old = dict(C.aggregate(rs))
    for k in ("n_stood_strict", "feet_share_last_window", "min_feet_last_window"):
        old.pop(k)
    assert "-" in C.format_table({"stats": {"r": {"r": old}}}).splitlines()[1].split()


# ===================================================================================== routing
class _Est:
    def __init__(self, k: int, scale: float):
        self.k, self.scale = k, scale

    def inference(self, prop):
        return (prop.sum(dim=1, keepdim=True) * self.scale + torch.arange(self.k, dtype=prop.dtype)).to(prop.dtype)


class _Net:
    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def act_inference(self, x, hist_encoding=False):
        self.calls.append((x.clone(), hist_encoding))
        return self.fn(x)


def _obs(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, W, generator=g) * 0.3


def test_stand_net_route_is_sn_common_and_launcher_actions(C, SN, DC):
    n = 4
    s1_est, st_est = _Est(P1 - P0, 0.01), _Est(P1 - P0, -0.02)
    obs = _obs(n, 1)
    obs0 = obs.clone()
    s1 = _Net(lambda x: torch.tanh(x[:, :18]))
    st = _Net(lambda x: torch.tanh(x[:, 65:83] + x[:, P0:P0 + 1]))      # reads the masked block and the estimate
    stand = torch.tensor([True, False, True, True])
    a, a_s1, extra = C.controller_actions("stand_net", obs, stand, SN=SN, s1=s1, s1_est=s1_est, p0=P0, p1=P1,
                                          num_prop=NP, st_pol=st, st_est=st_est, mask=1)
    assert torch.equal(obs, obs0)                                         # the env's obs is never written
    h = obs0.clone()
    h[:, P0:P1] = s1_est.inference(h[:, :NP])
    (x_s1, h1), = s1.calls
    (x_st, h2), = st.calls
    assert h1 is True and h2 is True
    assert torch.equal(x_s1, h)                                           # S1: the harness path
    assert torch.equal(x_st, SN.stand_net_input(h, st_est, p0=P0, p1=P1, num_prop=NP, mask=1))
    assert torch.equal(a, SN.route_actions(stand, extra["a_stand"], a_s1))
    assert torch.equal(a[~stand], a_s1[~stand]) and not torch.equal(a[stand], a_s1[stand])
    s1b, stb = _Net(s1.fn), _Net(st.fn)
    want = DC.launcher_actions(obs0, stand, s1b, s1_est, stb, st_est, p0=P0, p1=P1, num_prop=NP, mask=1)
    assert torch.equal(a, want)                                           # bitwise the pinned launcher path
    with pytest.raises(ValueError):
        C.controller_actions("stand_net", obs, stand, SN=SN, s1=s1, s1_est=s1_est, p0=P0, p1=P1, num_prop=NP)


def test_r_route_is_the_d3_restance_target(C, SN):
    n = 3
    s1_est = _Est(P1 - P0, 0.01)
    obs = _obs(n, 2)
    obs[:, [29 + i for i in range(6)]] = -0.8                            # a retracted hip read
    s1 = _Net(lambda x: torch.full((x.shape[0], 18), 0.7))
    stand = torch.tensor([True, True, False])
    a, a_s1, extra = C.controller_actions("r", obs, stand, SN=SN, s1=s1, s1_est=s1_est, p0=P0, p1=P1, num_prop=NP)
    rst = SN.load_rst()
    tgt, br = rst.restance_target(obs, rst.parse_restance(SN.RESTANCE_D3))
    assert rst.format_restance(SN.r_params()) == SN.RESTANCE_D3 and SN.r_params() != rst.RestanceParams()
    assert torch.equal(extra["r_target"], tgt) and torch.equal(extra["r_branch"], br)
    assert torch.equal(a[stand], tgt[stand]) and torch.equal(a[~stand], a_s1[~stand])


def test_s1_and_zero_routes(C, SN):
    s1_est = _Est(P1 - P0, 0.01)
    obs = _obs(2, 3)
    s1 = _Net(lambda x: x[:, :18] * 2.0)
    stand = torch.tensor([True, False])
    a, a_s1, _ = C.controller_actions("s1", obs, stand, SN=SN, s1=s1, s1_est=s1_est, p0=P0, p1=P1, num_prop=NP)
    assert torch.equal(a, a_s1)
    z, _, _ = C.controller_actions("zero", obs, stand, SN=SN, s1=s1, s1_est=s1_est, p0=P0, p1=P1, num_prop=NP)
    assert torch.equal(z, torch.zeros_like(a_s1))
    with pytest.raises(ValueError):
        C.controller_actions("walk", obs, stand, SN=SN, s1=s1, s1_est=s1_est, p0=P0, p1=P1, num_prop=NP)


def test_r_at_the_retracted_start_is_seat_and_zero(C, SN, poses):
    for name, p in poses.items():
        r = C.r_start_analysis(p, SN)
        assert set(r["branches"].values()) == {"seat"}, name
        assert r["target_is_zero"], name
        assert r["hip_obs_rad"] < r["p_up_rad"] < r["p_down_rad"] < 0.0


# ===================================================================================== the reset event / jitter
class _Asset:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def rec(*a, **k):
            self.calls.append((name, [x.clone() if torch.is_tensor(x) else x for x in a],
                               {kk: (v.clone() if torch.is_tensor(v) else v) for kk, v in k.items()}))
        return rec


class _Scene(dict):
    def __init__(self, n):
        super().__init__(robot=_Asset(), sn_table=_Asset())
        self.env_origins = torch.tensor([[10.0 * i, -5.0 * i, 0.03] for i in range(n)])


class _Env:
    def __init__(self, n):
        self.num_envs, self.device, self.scene = n, "cpu", _Scene(n)


def test_reset_event_writes_the_start(C):
    n = 3
    env = _Env(n)
    jp = [0.1 * i for i in range(24)]
    offs = [[0.0, 0.0, 0.0], [0.02, -0.01, 0.1], [-0.03, 0.03, -0.17]]
    C.sn_table_start(env, torch.tensor([1, 2]), joint_pos=jp, root_z=0.79875, table_center_z=0.2925,
                     table_offsets=offs)
    calls = {c[0]: c for c in env.scene["robot"].calls}
    pose = calls["write_root_pose_to_sim"][1][0]
    assert calls["write_root_pose_to_sim"][2]["env_ids"].tolist() == [1, 2]
    assert torch.allclose(pose[:, :2], env.scene.env_origins[1:, :2]) and torch.allclose(pose[:, 2], torch.tensor(0.79875))
    assert torch.equal(pose[:, 3:], torch.tensor([[1.0, 0, 0, 0]] * 2))
    q, v = calls["write_joint_state_to_sim"][1]
    assert torch.allclose(q, torch.tensor([jp] * 2)) and torch.equal(v, torch.zeros_like(v))
    assert torch.allclose(calls["set_joint_position_target"][1][0], q)
    tcalls = {c[0]: c for c in env.scene["sn_table"].calls}
    tp = tcalls["write_root_pose_to_sim"][1][0]
    want_xy = env.scene.env_origins[1:, :2] + torch.tensor([o[:2] for o in offs[1:]])
    assert torch.allclose(tp[:, :2], want_xy) and torch.allclose(tp[:, 2], torch.tensor(0.2925))
    assert torch.allclose(tp[:, 3], torch.cos(torch.tensor([0.05, -0.085])), atol=1e-6)   # (w, x, y, z), half yaw
    assert torch.allclose(tp[:, 6], torch.sin(torch.tensor([0.05, -0.085])), atol=1e-6)
    assert torch.equal(tp[:, 4:6], torch.zeros(2, 2))
    assert "write_root_velocity_to_sim" in calls                         # the robot's velocity is zeroed
    assert set(tcalls) == {"write_root_pose_to_sim"}                     # the kinematic table: pose only
    n_calls = len(env.scene["robot"].calls) + len(env.scene["sn_table"].calls)
    C.sn_table_start(env, torch.tensor([], dtype=torch.long), joint_pos=jp, root_z=0.8, table_center_z=0.29,
                     table_offsets=offs)                                  # an empty reset is a no-op
    assert len(env.scene["robot"].calls) + len(env.scene["sn_table"].calls) == n_calls


def test_jitter_draws(C):
    a = C.jitter_draws(32, 1)
    assert a[0] == [0.0, 0.0, 0.0]
    arr = np.asarray(a)
    assert (np.abs(arr[:, :2]) <= C.JITTER_XY_M).all() and (np.abs(arr[:, 2]) <= math.radians(C.JITTER_YAW_DEG)).all()
    assert np.abs(arr[1:]).min() > 0.0
    assert a == C.jitter_draws(32, 1)                                     # seeded
    assert C.jitter_draws(2, 1)[1] == a[1]                                # env i's draw does not depend on n
    assert a != C.jitter_draws(32, 2)
    assert C.jitter_draws(5, 1, enabled=False) == [[0.0, 0.0, 0.0]] * 5
    assert C.jitter_draws(1, 1) == [[0.0, 0.0, 0.0]]


# ===================================================================================== the scripts (static / Kit-free)
def test_script_compiles_and_imports_no_isaac_before_kit(tmp_path):
    py_compile.compile(str(TEST_PATH), cfile=str(tmp_path / "t.pyc"), doraise=True)
    py_compile.compile(str(CORE_PATH), cfile=str(tmp_path / "c.pyc"), doraise=True)
    for path in (TEST_PATH, CORE_PATH):
        tree = ast.parse(path.read_text())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = {a.name.split(".")[0] for n in top for a in n.names} | {
            (n.module or "").split(".")[0] for n in top if isinstance(n, ast.ImportFrom)}
        assert not names & {"torch", "numpy", "isaaclab", "gymnasium", "parkour_tasks", "imageio"}, (path, names)
    src = TEST_PATH.read_text()
    for needle in ("dcore.load_strict(", "SN.state_dicts_equal(", "P2L.arm_slot_latch(", "_install_schedule_override_world",
                   "_apply_env_overrides", "CORE.controller_actions(", "save_rng()", "restore_rng(before)",
                   "rng_equal(before", "il_time_out", "kinematic_enabled=True", "imageio.get_writer(",
                   "\"Isaac-Crab-Hex-Omni-Walk-Play-v0\""):
        assert needle in src, needle
    assert "RecordVideo" not in src


def test_script_pre_kit_refusals(T, tmp_path):
    o, rest = T.parse(["--controller", "stand_net", "--out", "relative/x", "--video", "--episode-s", "2"])
    bad = "\n".join(T.pre_kit_reasons(o, rest))
    assert "not absolute" in bad and "--enable_cameras" in bad and "STOOD window" in bad
    o, rest = T.parse(["--controller", "r", "--out", str(tmp_path), "--episode-s", "2", "--smoke", "--headless"])
    assert not [r for r in T.pre_kit_reasons(o, rest) if "STOOD" in r or "--out" in r]
    assert o.prefix == "r_rod_min" and rest == ["--headless"]
    with pytest.raises(SystemExit):
        T.parse(["--controller", "r", "--out", str(tmp_path), "--pose", "folded"])
    if T.DEFAULT_S1.is_file() and T.DEFAULT_STAND.is_file() and (Path(T.SPEC.REFS_ENV_STAMP) / "run_meta.json").is_file():
        o, rest = T.parse(["--controller", "stand_net", "--out", str(tmp_path), "--headless", "--video",
                           "--enable_cameras"])
        assert T.pre_kit_reasons(o, rest) == []                           # the adopted files, the REFS env
        (tmp_path / f"{o.prefix}_summary.json").write_text("{}")
        assert any("exists" in r for r in T.pre_kit_reasons(o, rest))
    assert T.main(["sn_table_test.py", "--controller", "zero", "--out", "rel"]) == T.REFUSED_RC


def test_emit_facts_and_merge_end_to_end(T, C, tmp_path):
    assert T.main(["sn_table_test.py", "--emit-facts", "--out", str(tmp_path / "f")]) == 0
    facts = json.loads((tmp_path / "f" / "facts.json").read_text())
    assert set(C.POSES) <= set(facts)
    assert facts["rod_min"]["spawn_root_z_m"] == pytest.approx(C.spawn_root_z())
    assert facts["knee_folded"]["r_at_start"]["target_is_zero"]
    assert facts["_sources"]["r_knob"] == T.SN.RESTANCE_D3
    d = tmp_path / "m"
    d.mkdir()
    for ctrl, pose, video in (("stand_net", "rod_min", False), ("r", "rod_min", False), ("r", "rod_min", True)):
        per = [C.classify(_trace(), DT), C.classify(_trace(gap=0.005, feet=0), DT)]
        (d / f"{ctrl}_{pose}_{int(video)}_summary.json").write_text(json.dumps(
            {"meta": {"controller": ctrl, "pose": pose, "video": video, "num_envs": 2, "mp4": "x.mp4" if video else None},
             "aggregate": C.aggregate(per), "per_env": per}))
    assert T.main(["sn_table_test.py", "--merge", str(d)]) == 0
    m = json.loads((d / "table_test_summary.json").read_text())
    assert set(m["stats"]["rod_min"]) == {"stand_net", "r"} and len(m["videos"]) == 1
    assert m["stats"]["rod_min"]["r"]["n_stood"] == 1 and m["stood_definition"] == C.STOOD_DEFINITION


# ===================================================================================== the launcher
def _pat(name: str) -> str:
    m = re.search(rf"^{name}='([^']*)'$", SH_PATH.read_text(), flags=re.M)
    assert m, f"{name} not found"
    return m.group(1)


def test_launcher_busy_pattern():
    pat, nonkit = re.compile(_pat("ISAAC_PAT")), re.compile(_pat("NONKIT_PAT"))
    py = "/home/nickmagus/krabby/isaac_venv/bin/python"
    busy = [f"{py} /x/sn_table_test.py --headless --controller r --prefix a", f"{py} /x/sn_record_video.py --headless",
            f"{py} -u /x/scripts/rsl_rl/train.py --task X", f"{py} /x/eval_crab_hex_gait.py", f"{py} /x/sn_dagger.py --mode dagger",
            f"{py} /x/play.py"]
    idle = [f"systemd-run --user --scope -q timeout 1 {py} /x/sn_table_test.py --headless", f"bash -c {py} /x/sn_table_test.py",
            "grep -E ^[^ ]*[i]saac_venv/bin/python", f"{py} -m pytest -q"]
    for line in busy:
        assert pat.search(line) and not nonkit.search(line), line
    for line in idle:
        assert not pat.search(line), line
    for line in (f"{py} -B /x/sn_table_test.py --emit-facts --out /y", f"{py} -B /x/sn_table_test.py --merge /y",
                 f"{py} -B /x/sn_dagger.py --read /y"):
        assert pat.search(line) and nonkit.search(line), line              # Kit-free: never counted busy


def test_launcher_syntax_and_dry_run_launches_nothing(tmp_path):
    subprocess.run(["bash", "-n", str(SH_PATH)], check=True)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "LAUNCHED"
    for tool in ("systemd-run", "setsid", "nohup"):
        (fake_bin / tool).write_text(f"#!/bin/sh\ntouch {marker}\nexit 1\n")
        (fake_bin / tool).chmod(0o755)
    out = tmp_path / "out"
    env = dict(os.environ, PATH=f"{fake_bin}:{os.environ.get('PATH', '')}", SN_DDC=str(tmp_path / "ddc"),
               SN_TABLE_OUT=str(out), CUDA_VISIBLE_DEVICES="", SN_PY=sys.executable)
    r = subprocess.run(["bash", str(SH_PATH), "--smoke", "--dry-run"], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    for needle in ("OMNI_KIT_ACCEPT_EULA=yes", "MemoryMax=24G", "MemorySwapMax=0", "--headless",
                   "timeout --signal=TERM -k 120 1800", "--num-envs 2 --episode-s 2", "--controller stand_net --pose rod_min",
                   "--controller r --pose knee_folded", "--smoke", "--merge"):
        assert needle in r.stdout, (needle, r.stdout)
    jobs = [ln for ln in r.stdout.splitlines() if ln.startswith("systemd-run")]
    # the default smoke exercises the render path: 2 stats jobs without cameras + 1 one-env 2-s video job
    assert len(jobs) == 3 and all("--episode-s 2 " in j and "--smoke" in j for j in jobs)
    assert ["--enable_cameras" in j for j in jobs] == [False, False, True]
    assert "--controller s1 --pose rod_min --num-envs 1 --episode-s 2" in jobs[2] and "--video" in jobs[2]
    assert "--prefix smoke_video_s1_rod_min" in jobs[2]
    for flag in (["--smoke-video"], []):                                 # --smoke-video is the (accepted) default
        r2 = subprocess.run(["bash", str(SH_PATH), "--smoke", "--dry-run", *flag], env=env, capture_output=True,
                            text=True, timeout=60)
        assert r2.returncode == 0 and r2.stdout.count("--enable_cameras") == 1, (flag, r2.stdout)
    r = subprocess.run(["bash", str(SH_PATH), "--smoke", "--smoke-no-video", "--dry-run"], env=env, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "--enable_cameras" not in r.stdout and "--video" not in r.stdout
    assert len([ln for ln in r.stdout.splitlines() if ln.startswith("systemd-run")]) == 2
    r = subprocess.run(["bash", str(SH_PATH), "--dry-run"], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    jobs = [ln for ln in r.stdout.splitlines() if ln.startswith("systemd-run")]
    assert len(jobs) == 14                                               # 2 poses x 4 stats + 2 poses x 3 videos
    assert sum("--enable_cameras" in j and "--video --out" in j and "--num-envs 1 " in j for j in jobs) == 6
    stats = [j for j in jobs if "--num-envs 32 --episode-s 15 --seed 1 --jitter" in j]
    assert len(stats) == 8
    for pose in ("rod_min", "knee_folded"):                              # zero = the no-controller baseline
        for c in ("stand_net", "r", "s1", "zero"):
            assert sum(f"--controller {c} --pose {pose} " in j and f"--prefix stats_{c}_{pose}" in j
                       for j in stats) == 1, (c, pose)
    assert not any("--controller zero" in j for j in jobs if "--enable_cameras" in j)
    assert not marker.exists() and not out.exists()                     # a dry run launches and writes nothing
    for bad in (["--pose", "folded"], ["--controllers", "walk"], ["--num-envs", "0"], ["--bogus"]):
        r = subprocess.run(["bash", str(SH_PATH), "--dry-run", *bad], env=env, capture_output=True, text=True, timeout=60)
        assert r.returncode == 5, (bad, r.stderr)
    r = subprocess.run(["bash", str(SH_PATH), "--dry-run"], env=dict(env, SN_TABLE_TIMEOUT="0"), capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 5 and "SN_TABLE_TIMEOUT" in r.stderr


def _fake_tools(tmp_path, fail_first: bool = False, fail_merge: bool = False):
    """A fake ``systemd-run`` (drops its own options, execs from ``timeout`` on) and a fake isaac python: the Kit-free
    modes write their files, an Isaac job records its argv and writes ``<prefix>_summary.json`` (or fails)."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "systemd-run").write_text('#!/bin/bash\nwhile [ $# -gt 0 ] && [ "$1" != timeout ]; do shift; done\n'
                                          'exec "$@"\n')
    (fake_bin / "systemd-run").chmod(0o755)
    calls = tmp_path / "calls.txt"
    py = tmp_path / "fakepy"
    py.write_text(f"""#!/bin/bash
args="$*"
out=""; prefix=""
while [ $# -gt 0 ]; do
  case "$1" in --out) out=$2; shift 2 ;;
    --merge) if [ "{int(fail_merge)}" = 1 ]; then echo "merge boom" >&2; exit 2; fi
             echo merged > "$2/table_test_summary.json"; exit 0 ;;
    --prefix) prefix=$2; shift 2 ;; *) shift ;; esac
done
case "$args" in *--emit-facts*) echo '{{}}' > "$out/facts.json"; exit 0 ;; esac
echo "$args" >> {calls}
if [ "{int(fail_first)}" = 1 ] && [ "$(grep -c . {calls})" = 1 ]; then exit 9; fi
echo '{{}}' > "$out/${{prefix}}_summary.json"
""")
    py.chmod(0o755)
    return fake_bin, py, calls


def _queue_env(tmp_path, fake_bin, py):
    return dict(os.environ, PATH=f"{fake_bin}:{os.environ.get('PATH', '')}", SN_PY=str(py), SN_DDC=str(tmp_path / "ddc"),
                SN_TABLE_OUT=str(tmp_path / "out"), CUDA_VISIBLE_DEVICES="", SN_TABLE_TIMEOUT="60",
                SN_TABLE_KILL_AFTER="5")


def _isaac_busy() -> bool:
    pat, nonkit = re.compile(_pat("ISAAC_PAT")), re.compile(_pat("NONKIT_PAT"))
    lines = subprocess.run(["ps", "-ww", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()
    return any(pat.search(ln) and not nonkit.search(ln) for ln in lines)


def test_launcher_queue_runs_jobs_serially_and_merges(tmp_path):
    if _isaac_busy():
        pytest.skip("an Isaac process is running on this machine (the launcher would refuse)")
    fake_bin, py, calls = _fake_tools(tmp_path)
    env = _queue_env(tmp_path, fake_bin, py)
    r = subprocess.run(["bash", str(SH_PATH), "--smoke", "--foreground"], env=env, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, (r.stdout, r.stderr)
    stamp = Path((tmp_path / "out" / "LATEST").read_text().strip())
    assert stamp.parent == tmp_path / "out" and stamp.name.endswith("_smoke")
    q = (stamp / "queue.log").read_text()
    names = ("smoke_stand_net_rod_min", "smoke_r_knee_folded", "smoke_video_s1_rod_min")
    for name in names:
        assert f"SNTABLE JOB START {name}" in q and f"SNTABLE JOB END {name} rc=0" in q
        assert (stamp / f"{name}_summary.json").is_file() and (stamp / f"{name}.unit").is_file()
    assert [q.index(f"SNTABLE JOB START {n}") for n in names] == sorted(q.index(f"SNTABLE JOB START {n}") for n in names)
    assert "SNTABLE QUEUE END rc=0" in q and (stamp / "facts.json").is_file()
    assert (stamp / "table_test_summary.json").read_text().strip() == "merged"
    jobs = calls.read_text().splitlines()
    assert len(jobs) == 3 and all("--headless" in j and f"--out {stamp}" in j for j in jobs)
    assert "--controller stand_net --pose rod_min --num-envs 2 --episode-s 2" in jobs[0]
    assert "--prefix smoke_r_knee_folded" in jobs[1] and "--enable_cameras" not in jobs[1]
    assert "--enable_cameras" in jobs[2] and "--video" in jobs[2] and "--num-envs 1 --episode-s 2" in jobs[2]


def test_launcher_merge_failure_is_not_success(tmp_path):
    if _isaac_busy():
        pytest.skip("an Isaac process is running on this machine (the launcher would refuse)")
    fake_bin, py, calls = _fake_tools(tmp_path, fail_merge=True)
    env = _queue_env(tmp_path, fake_bin, py)
    r = subprocess.run(["bash", str(SH_PATH), "--smoke", "--smoke-no-video", "--foreground"], env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 6, (r.stdout, r.stderr)
    stamp = Path((tmp_path / "out" / "LATEST").read_text().strip())
    q = (stamp / "queue.log").read_text()
    assert "SNTABLE JOB END smoke_r_knee_folded rc=0" in q and "merge failed" in q
    assert "SNTABLE QUEUE END rc=6" in q and "SNTABLE QUEUE END rc=0" not in q
    assert not (stamp / "table_test_summary.json").exists()
    # a failed job keeps its own code (1) when the merge also fails
    (tmp_path / "b").mkdir()
    fb2, py2, _ = _fake_tools(tmp_path / "b", fail_first=True, fail_merge=True)
    r = subprocess.run(["bash", str(SH_PATH), "--smoke", "--foreground"], env=_queue_env(tmp_path / "b", fb2, py2),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 1, (r.stdout, r.stderr)


def test_launcher_queue_stops_at_the_first_failure(tmp_path):
    if _isaac_busy():
        pytest.skip("an Isaac process is running on this machine (the launcher would refuse)")
    fake_bin, py, calls = _fake_tools(tmp_path, fail_first=True)
    env = _queue_env(tmp_path, fake_bin, py)
    r = subprocess.run(["bash", str(SH_PATH), "--smoke", "--foreground"], env=env, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 1, (r.stdout, r.stderr)
    stamp = Path((tmp_path / "out" / "LATEST").read_text().strip())
    q = (stamp / "queue.log").read_text()
    assert "SNTABLE JOB END smoke_stand_net_rod_min rc=9" in q and "the queue stops" in q
    assert "SNTABLE JOB START smoke_r_knee_folded" not in q and "SNTABLE QUEUE END rc=1" in q
    assert len(calls.read_text().splitlines()) == 1
