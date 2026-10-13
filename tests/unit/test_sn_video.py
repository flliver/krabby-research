"""Stand-network campaign (``2026-10-11_stand_network``): the random-command VIDEO tools on CPU (no Isaac / Kit / GPU).

Pinned here:

* ``scripts/sn_video_schedule.py`` (the seeded generator): settle stand, then COMMAND / STAND alternating, ending on a
  stand; video length (prologue + holds) == ``--total-s`` exactly; command holds 6-14 s, stands 3-12 s (random, not
  constant), settle 2-3 s; every schedule passes the harness's own validator (``gait_eval.schedule``) and re-loads
  bitwise through ``load_manifest``; speeds within the minimum ellipse cap along a move_turn's sweep and >= 0.12; a
  facing change gets ``settle_s + |dface| / 0.15`` s; seeded determinism; the committed manifest IS the generator's
  output for its seed (where written: videos/ is git-ignored);
* ``scripts/sn_video_core.py`` (the stand-ness rule): STAND iff speed 0 and the face of the previous hold (spawn pin
  for hold 0); turn holds (speed 0, facing change) are NOT stand, a stand after a turn IS; the per-step mask from a
  compiled schedule is the hard switch (prologue = hold 0, entry on a stand hold's first step, exit on the step the
  hold changes); the homing statistics of one stand; the eval's latched-slot PREDICTION (TURN on every stand after
  the first facing change until a reset: 7 of the default schedule's 10 stands); the per-hold yaw-command marks
  (``post_fall_turn_cmd``: the stand net on a live turn command after a respawn); the caption names derived from
  the checkpoints;
* ``route_step``: the stand rows get the stand net's action and every other row S1's, both networks run on every
  row, S1 sees the harness's obs, the stand net sees ``sn_common.stand_net_input`` (its own estimate on the UNMASKED
  prop, masked iff mask), ``hist_encoding=True`` on both, the harness's obs is never mutated;
* ``scripts/sn_record_video.py`` compiles (never imported: it boots Kit) and its defaults name the evaluated
  deployment; the P2L functions it calls exist; ``scripts/sn_record_video.sh`` passes ``bash -n``, its busy pattern
  matches the Isaac command lines and not its own helpers, and ``--dry-run`` launches nothing.
"""
from __future__ import annotations

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
yaml = pytest.importorskip("yaml")
np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[2]
TASK = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
CAMP = TASK / "experiments/2026-10-11_stand_network"
SCRIPTS = CAMP / "scripts"
VIDEOS = CAMP / "videos"
GEN_PATH = SCRIPTS / "sn_video_schedule.py"
CORE_PATH = SCRIPTS / "sn_video_core.py"
COMMON_PATH = SCRIPTS / "sn_common.py"
REC_PATH = SCRIPTS / "sn_record_video.py"
SH_PATH = SCRIPTS / "sn_record_video.sh"
P2L_PATH = TASK / "experiments/2026-10-07_omni_phase2/scripts/p2r_mode_eval_crab_hex_gait.py"
W, P0, P1, NP = 1149, 207, 216, 75
SEEDS = (20261013, 0, 1, 2, 3, 7, 42, 99, 123, 2024, 31337, 777)


def _load(path: Path, name: str):
    """By path, never writing ``__pycache__`` next to the campaign / phase-2 scripts."""
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
def GEN():
    return _load(GEN_PATH, "_t_snv_gen")


@pytest.fixture(scope="module")
def CORE():
    return _load(CORE_PATH, "_t_snv_core")


@pytest.fixture(scope="module")
def SN():
    return _load(COMMON_PATH, "_t_snv_common")


@pytest.fixture(scope="module")
def S(GEN):
    return GEN.schedule_module()


@pytest.fixture(scope="module")
def HDR(GEN):
    return GEN.reference_header()


@pytest.fixture(scope="module")
def gens(GEN, HDR):
    return {s: GEN.generate(s, 180.0, hdr=HDR) for s in SEEDS}


def _ds(x: float) -> int:
    v = round(x * 10)
    assert abs(v / 10 - x) < 1e-9, f"{x} is not a whole number of deciseconds"
    return int(v)


# ===================================================================================== generator
def test_alternation_settle_command_stand(gens, CORE):
    for seed, g in gens.items():
        recs = g["records"]
        kinds = [r["kind"] for r in recs]
        assert kinds[0] == "stand" and kinds[-1] == "stand", seed
        assert len(recs) % 2 == 1 and len(recs) >= 3, seed
        for i, k in enumerate(kinds):
            if i % 2 == 0:
                assert k == "stand", (seed, i, k)
            else:
                assert k in CORE.COMMAND_KINDS, (seed, i, k)
        assert [r["stand"] for r in recs] == [k == "stand" for k in kinds]
        assert CORE.hold_stand_flags(g["holds"], 0.0) == [r["stand"] for r in recs]
        for i, r in enumerate(recs):
            assert CORE.LABEL_RE.match(r["label"]), r["label"]
            assert r["label"].startswith(f"{i:02d}_")


def test_totals_and_lengths(gens, GEN):
    for seed, g in gens.items():
        sm = g["summary"]
        holds = g["holds"]
        assert sum(_ds(h.hold_s) for h in holds) + 15 == 1800, seed      # prologue 1.5 s + holds == 180 s exactly
        assert sm["video_s"] == 180.0 and abs(sm["holds_s"] - 178.5) < 1e-9
        recs = g["records"]
        assert 20 <= _ds(recs[0]["hold_s"]) <= 30, seed
        for r in recs[1:]:
            ds = _ds(r["hold_s"])
            if r["kind"] == "stand":
                assert 30 <= ds <= 120, (seed, r)
            else:
                assert 60 <= ds <= 140, (seed, r)
        assert sm["n_commands"] == sm["n_stands_between_commands"] == (len(recs) - 1) // 2
        assert 6 <= sm["n_commands"] <= 17
        assert abs(recs[-1]["t1_s"] - 180.0) < 1e-6 and abs(recs[0]["t0_s"] - 1.5) < 1e-9


def test_every_schedule_passes_the_harness_validator(gens, GEN, S, HDR, tmp_path):
    for seed, g in gens.items():
        S.validate_scenario(GEN._scenario(S, HDR, g["holds"], g["row_id"]))
        p = tmp_path / f"m{seed}.yaml"
        p.write_text(GEN.manifest_text(g))
        GEN.check_written(p, g)                               # load_manifest (validates as written) + bitwise holds
        sc = {s.id: s for s in S.load_manifest(p)[0]}[g["row_id"]]
        assert sc.spawn_yaw_rad() == 0.0 and sc.terrain_label == "flat" and sc.env_seed == 1
        assert sc.task == "Isaac-Crab-Hex-Omni-Walk-Play-v0" and not sc.env_vars
        assert sc.get_speed_ellipse() == (0.35, 0.25, 0.25) and sc.ang_vel_max == 0.3
        assert sc.episode_length_s >= sc.total_duration_s


def test_speed_and_turn_constraints(gens, GEN, CORE, S):
    ell = (0.35, 0.25, 0.25)
    for seed, g in gens.items():
        prev = 0.0
        for h, r in zip(g["holds"], g["records"]):
            dface = CORE.wrap_to_pi(h.face - prev)
            if r["kind"] in ("move", "move_turn"):
                cap = GEN.min_cap_along_turn(h.dir, prev, h.face, ell, S)
                assert GEN.SPEED_MIN - 1e-12 <= h.speed <= cap + 1e-12, (seed, r)
                # the validator's own end-of-hold cap, independently
                assert h.speed <= S.ellipse_cap(S.wrap_to_pi(h.dir - h.face), *ell) + 1e-9
                assert round(h.speed, 2) == h.speed
            else:
                assert h.speed == 0.0
            if r["kind"] in ("turn", "move_turn"):
                lo, hi = GEN.TURN_DEG if r["kind"] == "turn" else GEN.MOVE_TURN_DEG
                assert lo - 1e-3 <= abs(math.degrees(dface)) <= hi + 1e-3, (seed, r)
                assert h.hold_s >= 1.0 + abs(dface) / 0.15 - 1e-9, (seed, r)    # the plant-rate planning rule
            else:
                assert abs(dface) <= CORE.FACE_TOL, (seed, r)
            prev = h.face


def test_min_cap_along_turn_is_the_sweep_minimum(GEN, S):
    ell = (0.35, 0.25, 0.25)
    # moving along world +x while turning from face 0 to +90 deg: body direction sweeps 0 -> -90 deg (fwd -> lat)
    assert GEN.min_cap_along_turn(0.0, 0.0, math.pi / 2, ell, S) == pytest.approx(0.25)
    # no turn: the one cap
    assert GEN.min_cap_along_turn(0.0, 0.0, 0.0, ell, S) == pytest.approx(0.35)
    # the START of the sweep binds too (the validator checks only the end): travelling along world +y while turning
    # from face 0 (body direction +90 deg, lateral cap 0.25) to face +90 deg (body direction 0, forward cap 0.35)
    assert S.ellipse_cap(S.wrap_to_pi(math.pi / 2 - math.pi / 2), *ell) == pytest.approx(0.35)
    assert GEN.min_cap_along_turn(math.pi / 2, 0.0, math.pi / 2, ell, S) == pytest.approx(0.25)
    # a sweep from body +30 deg to -30 deg through 0: the minimum is at the ends
    assert GEN.min_cap_along_turn(0.0, -math.pi / 6, math.pi / 6, ell, S) == pytest.approx(
        S.ellipse_cap(math.pi / 6, *ell))


def test_seeded_determinism(GEN, HDR):
    a = GEN.generate(20261013, 180.0, hdr=HDR)
    b = GEN.generate(20261013, 180.0, hdr=HDR)
    c = GEN.generate(20261014, 180.0, hdr=HDR)
    key = lambda g: [(h.label, h.hold_s, h.as_world()) for h in g["holds"]]   # noqa: E731
    assert key(a) == key(b)
    assert key(a) != key(c)
    assert GEN.manifest_text(a) == GEN.manifest_text(b)


def test_stand_lengths_are_random(gens):
    allstands = []
    for seed, g in gens.items():
        st = [r["hold_s"] for r in g["records"][1:] if r["kind"] == "stand"]
        assert len(set(st)) >= max(2, len(st) // 2), (seed, st)       # not one constant length
        allstands += st
    assert min(allstands) < 4.5 and max(allstands) > 10.5
    assert len(set(allstands)) > 40


def test_kinds_restriction_and_smoke(GEN, HDR):
    g = GEN.generate(20261013, 20.0, kinds=["move_turn"], hdr=HDR, row_id="sn_random_smoke__omni")
    assert [r["kind"] for r in g["records"]] == ["stand", "move_turn", "stand"]
    assert g["summary"]["video_s"] == 20.0
    for k in ("move", "turn"):
        gg = GEN.generate(5, 120.0, kinds=[k], hdr=HDR)
        assert {r["kind"] for r in gg["records"][1::2]} == {k}
    with pytest.raises(ValueError):
        GEN.generate(1, 10.0, hdr=HDR)                       # no room for settle + one pair
    with pytest.raises(ValueError):
        GEN.generate(1, 180.0, kinds=["jump"], hdr=HDR)
    with pytest.raises(ValueError):
        GEN.generate(1, 180.05, hdr=HDR)                     # not whole deciseconds


def test_cli_writes_and_reloads(GEN, S, tmp_path):
    assert GEN.main(["--seed", "5", "--total-s", "60", "--out-dir", str(tmp_path), "--tag", "t"]) == 0
    yml, js = tmp_path / "scenarios_sn_video_t.yaml", tmp_path / "sn_video_schedule_t.json"
    scen = {s.id: s for s in S.load_manifest(yml)[0]}
    assert list(scen) == ["sn_random_t__omni"]
    d = json.loads(js.read_text())
    assert d["seed"] == 5 and d["summary"]["video_s"] == 60.0
    assert [r["label"] for r in d["records"]] == [h.label for h in scen["sn_random_t__omni"].schedule]
    assert GEN.main(["--seed", "5", "--total-s", "60", "--out-dir", str(tmp_path / "dry"), "--dry-run"]) == 0
    assert not (tmp_path / "dry").exists()


def test_written_manifests_are_the_generators_output(GEN, HDR):
    # videos/ is git-ignored (experiments/.gitignore ``**/videos/``): the files exist only where the generator (or
    # sn_record_video.sh, which regenerates them every launch) has run
    if not (VIDEOS / "scenarios_sn_video.yaml").is_file() or not (VIDEOS / "scenarios_sn_video_smoke.yaml").is_file():
        pytest.skip("videos/ manifests not generated in this checkout")
    full = GEN.generate(GEN.DEFAULT_SEED, 180.0, hdr=HDR)
    assert (VIDEOS / "scenarios_sn_video.yaml").read_text() == GEN.manifest_text(full)
    smoke = GEN.generate(GEN.DEFAULT_SEED, 20.0, kinds=["move_turn"], hdr=HDR, row_id="sn_random_smoke__omni")
    assert (VIDEOS / "scenarios_sn_video_smoke.yaml").read_text() == GEN.manifest_text(smoke)
    js = json.loads((VIDEOS / "sn_video_schedule.json").read_text())
    assert js["summary"] == json.loads(json.dumps(full["summary"]))


# ===================================================================================== the stand-ness rule
def test_stand_rule_synthetic(CORE):
    holds = [
        (0.0, 0.0, 0.0),            # stand (face == spawn)
        (0.2, 1.0, 0.0),            # move
        (0.0, 0.0, 0.0),            # stand
        (0.0, 0.0, 1.2),            # turn: speed 0 + facing change -> NOT stand
        (0.0, 0.0, 1.2),            # stand AFTER a turn -> stand
        (0.15, 0.3, 0.5),           # move_turn
        (0.0, 0.0, 0.5),            # stand after a move_turn
        (0.0, 0.0, 0.5 + 2 * math.pi),   # same facing modulo 2 pi -> still stand
        (0.0, 0.0, -0.5),           # turn
    ]
    assert CORE.hold_kinds(holds, 0.0) == ["stand", "move", "stand", "turn", "stand", "move_turn", "stand", "stand",
                                           "turn"]
    assert CORE.hold_stand_flags(holds, 0.0) == [True, False, True, False, True, False, True, True, False]
    # spawn pin != hold 0's face: hold 0 is a turn; None = first_face
    assert CORE.hold_stand_flags([(0.0, 0.0, 1.0)], 0.0) == [False]
    assert CORE.hold_stand_flags([(0.0, 0.0, 1.0)], None) == [True]
    with pytest.raises(ValueError):
        CORE.hold_kind(-0.1, 0.0, 0.0)


def test_eval_slot_prediction_latches_turn_until_reset(CORE):
    holds = [
        (0.0, 0.0, 0.0),            # stand at the spawn pin: latch clear -> STAND
        (0.2, 1.0, 0.0),            # move
        (0.0, 0.0, 0.0),            # stand (no facing change yet) -> STAND
        (0.0, 0.0, 1.2),            # turn: fresh target, error > dead band -> latch TURN
        (0.0, 0.0, 1.2),            # stand right after the turn -> TURN (latched)
        (0.2, 0.3, 1.2),            # move at the same face: NOT a fresh target, latch kept
        (0.0, 0.0, 1.2),            # stand after turn -> move -> still TURN (the corrected claim)
        (0.15, 0.3, 0.5),           # move_turn: fresh, latched again
        (0.0, 0.0, 0.5 + 2 * math.pi),   # same facing modulo 2 pi -> not fresh -> TURN
    ]
    assert CORE.eval_slot_stand_flags(holds, 0.0) == [True, False, True, False, False, False, False, False, False]
    assert CORE.hold_stand_flags(holds, 0.0) == [True, False, True, False, True, False, True, False, True]
    # a facing change inside the dead band re-arms the latch CLEAR
    small = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.1), (0.0, 0.0, 0.1)]
    assert CORE.eval_slot_stand_flags(small, 0.0) == [True, True, True]
    assert CORE.eval_slot_stand_flags(small, 0.0, dead_band_rad=0.05) == [True, False, False]
    # hold 0 is re-derived against the spawn pin; None = first_face
    assert CORE.eval_slot_stand_flags([(0.0, 0.0, 1.0)], 0.0) == [False]
    assert CORE.eval_slot_stand_flags([(0.0, 0.0, 1.0)], None) == [True]


def test_eval_slot_turn_stands_in_generated_schedules(gens, GEN, HDR):
    for seed, g in gens.items():
        recs = g["records"]
        first_turn = next((i for i, r in enumerate(recs) if r["kind"] in ("turn", "move_turn")), len(recs))
        want = [r["label"] for i, r in enumerate(recs) if r["stand"] and i > first_turn]
        assert g["summary"]["stands_eval_slot_reads_turn"] == want, seed
        assert len(want) >= g["summary"]["stands_after_a_facing_change"], seed
    full = GEN.generate(GEN.DEFAULT_SEED, 180.0, hdr=HDR)
    assert full["summary"]["stands_eval_slot_reads_turn"] == [
        "06_stand5.2s", "08_stand9.1s", "10_stand6.7s", "12_stand8.1s", "14_stand9.6s", "16_stand3.9s",
        "18_stand9.6s"]
    assert full["summary"]["stands_after_a_facing_change"] == 3


def test_yaw_cmd_stats_marks_post_fall_turn_commands(CORE):
    T = 20
    routed = np.zeros(T)
    routed[5:15] = 1.0
    wz = np.zeros(T)
    wz[3:8] = 0.2
    wz[12:14] = -0.2
    idx = np.arange(5, 15)
    s = CORE.yaw_cmd_stats(routed, wz, idx, [10])
    assert s == {"yaw_cmd_at_entry": True, "stand_net_on_yaw_cmd_steps": 5, "post_fall": True,
                 "post_fall_turn_cmd": True}
    s = CORE.yaw_cmd_stats(routed, wz, idx, [])
    assert s["post_fall"] is False and s["post_fall_turn_cmd"] is False and s["stand_net_on_yaw_cmd_steps"] == 5
    # a fall on the hold's last step: the respawn is inside that step, no step of the hold comes after it
    s = CORE.yaw_cmd_stats(routed, wz, idx, [14])
    assert s["post_fall"] is False and s["post_fall_turn_cmd"] is False
    # post-fall, but the robot is back at the face (no yaw command) on every later step
    s = CORE.yaw_cmd_stats(routed, wz, idx, [13])
    assert s["post_fall"] is True and s["post_fall_turn_cmd"] is False
    # S1 acting (not routed): no stand-net step on a yaw command, whatever the command
    s = CORE.yaw_cmd_stats(np.zeros(T), wz, idx, [10])
    assert s["stand_net_on_yaw_cmd_steps"] == 0 and s["post_fall_turn_cmd"] is False and s["post_fall"] is True
    assert CORE.yaw_cmd_stats(routed, wz, idx[:0], [1])["yaw_cmd_at_entry"] is None


def test_caption_tags_come_from_the_checkpoints(CORE):
    run = "/x/stand_network/runs/SN_Amask_s17"
    sd = {"arm": "mask", "seed": 17, "iterations": 4000, "smoke": False}
    assert CORE.stand_tag_from_ckpt(sd, f"{run}/model_4000.pt") == "A-mask s17"
    assert CORE.stand_tag_from_ckpt(dict(sd, arm="full", seed=18), "/x/SN_Afull_s18/model_4000.pt") == "A-full s18"
    assert CORE.stand_tag_from_ckpt(sd, f"{run}/model_2000.pt") == "A-mask s17 (model_2000)"
    assert CORE.stand_tag_from_ckpt(dict(sd, smoke=True, iterations=4), "/x/SN_SMOKE/model_4.pt") == "A-mask s17 SMOKE"
    assert CORE.stand_tag_from_ckpt(None, "/x/model_19994.pt") == "S1 identity"
    s1 = "/x/crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB/model_19994.pt"
    assert CORE.s1_tag_from_ckpt(s1, is_stand_nets_s1=True) == "S1 s18"
    assert CORE.s1_tag_from_ckpt(s1, is_stand_nets_s1=False) == "walker s18"
    assert CORE.s1_tag_from_ckpt("/x/some_run/model_1.pt", is_stand_nets_s1=True) == "S1 some_run"
    real = REPO / "parkour/outputs/omni_plan/stand_network/runs/SN_Amask_s17/model_4000.pt"
    if real.is_file():                                   # the default checkpoint names itself as the old fixed default
        ck = torch.load(str(real), map_location="cpu", weights_only=False)
        assert CORE.stand_tag_from_ckpt(ck["infos"]["sn_dagger"], real) == "A-mask s17"


def test_step_mask_is_the_hard_switch(CORE, GEN, S, HDR):
    mk = lambda lab, s, d, f, t: S.Hold(hold_s=t, label=lab, speed=s, dir=d, face=f)   # noqa: E731
    holds = [mk("00_stand3.0s", 0.0, 0.0, 0.0, 3.0), mk("01_turn+57", 0.0, 0.0, 1.0, 9.0),
             mk("02_stand4.0s", 0.0, 0.0, 1.0, 4.0), mk("03_move0.20", 0.2, 0.5, 1.0, 6.0),
             mk("04_stand3.0s", 0.0, 0.0, 1.0, 3.0)]
    sc = GEN._scenario(S, HDR, holds, "t__omni")
    S.validate_scenario(sc)
    comp = S.compile_schedule(sc, dt=0.02, num_envs=1)
    seg = np.asarray(comp.segment_id[:, 0]).astype(np.int64)
    flags = CORE.hold_stand_flags(sc.schedule, sc.spawn_yaw_rad())
    assert flags == [True, False, True, False, True]
    m = CORE.step_stand_mask(seg, flags)
    assert m.shape == (comp.n_steps,) and m.dtype == bool
    assert m[seg == -1].all()                                        # the prologue holds hold 0 (a stand)
    for i, fl in enumerate(flags):
        assert (m[seg == i] == fl).all(), i                          # a turn hold stays S1's to its last step
    starts = [int(np.argmax(seg == i)) for i in range(len(holds))]
    assert not m[starts[2] - 1] and m[starts[2]]                     # entry on the stand hold's first step
    assert m[starts[3] - 1] and not m[starts[3]]                     # exit on the step the hold changes
    with pytest.raises(ValueError):
        CORE.step_stand_mask(np.array([0, 5]), flags)


def test_stand_hold_stats(CORE):
    dt = 0.02
    T = 300
    cam = np.concatenate([np.linspace(0.4, 0.12, 100), np.full(50, 0.05), np.full(5, 0.2), np.full(145, 0.03)])
    idx = np.arange(T)
    s = CORE.stand_hold_stats(cam, idx, dt)
    assert s["rows"] == T and s["homed_at_end"] is True and s["cam_max_end_rad"] == pytest.approx(0.03)
    assert s["settled"] and s["settle_s"] == pytest.approx(100 * dt)     # first 0.5-s homed run starts at row 100
    assert s["cam_max_last5s_rad"] == pytest.approx(cam[-250:].max())    # a 6-s hold: its last 5 s = rows 50..299
    assert s["homed_last5s"] is False
    s2 = CORE.stand_hold_stats(cam, idx, dt, first_fall_step=120)
    assert s2["rows"] == 120 and not s2["settled"] and s2["cam_max_end_rad"] == pytest.approx(0.05)
    s3 = CORE.stand_hold_stats(cam, idx[:0], dt)
    assert s3["rows"] == 0 and s3["homed_at_end"] is None


# ===================================================================================== routing
class _Est:
    """A deterministic estimator double: ``inference(prop) -> (N, k)``; reads the WHOLE prop (cols 65-74 included)."""

    def __init__(self, k: int, scale: float):
        self.k, self.scale = k, scale

    def inference(self, prop):
        return (prop.sum(dim=1, keepdim=True) * self.scale + torch.arange(self.k, dtype=prop.dtype)).to(prop.dtype)


class _Pol:
    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def __call__(self, x, hist_encoding=False):
        self.calls.append((x.clone(), hist_encoding))
        return self.fn(x)


def _harness_obs(n, w, p0, p1, num_prop, s1_est, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, w, generator=g)
    obs[:, p0:p1] = s1_est.inference(obs[:, :num_prop])              # the harness writes S1's estimate in place
    return obs


def test_route_step_rows_and_inputs(CORE, SN):
    n, w, p0, p1, num_prop = 5, 40, 15, 18, 10
    s1_est, st_est = _Est(3, 0.5), _Est(3, -2.0)                     # different estimators: the stand net uses ITS own
    obs = _harness_obs(n, w, p0, p1, num_prop, s1_est)
    obs0 = obs.clone()
    s1 = _Pol(lambda x: x[:, :4] * 1.0)
    st = _Pol(lambda x: x[:, p0:p0 + 3].sum(1, keepdim=True).expand(-1, 4) + 100.0)
    rows = torch.tensor([True, False, True, False, False])
    actions, a_s1, a_st = CORE.route_step(SN, obs, s1, st, st_est, rows, p0=p0, p1=p1, num_prop=num_prop, mask=0)
    assert torch.equal(obs, obs0)                                    # the harness's obs is never written
    assert torch.equal(actions[rows], a_st[rows]) and torch.equal(actions[~rows], a_s1[~rows])
    assert torch.equal(actions, SN.route_actions(rows, a_st, a_s1))
    assert not torch.equal(a_st, a_s1)
    (x1, h1), = s1.calls
    (x2, h2), = st.calls
    assert h1 is True and h2 is True                                 # hist_encoding=True on both (the D1 path)
    assert torch.equal(x1, obs0)                                     # S1: the harness's obs, S1's estimate in place
    want = torch.cat([obs0[:, :p0], st_est.inference(obs0[:, :num_prop]), obs0[:, p1:]], dim=1)
    assert torch.equal(x2, want)                                     # the stand net: its own estimate, a new tensor
    assert torch.equal(x2, SN.stand_net_input(obs0, st_est, p0=p0, p1=p1, num_prop=num_prop, mask=0))
    # every row stand / no row stand
    a_all, _, _ = CORE.route_step(SN, obs, s1, st, st_est, torch.ones(n, dtype=torch.bool), p0=p0, p1=p1,
                                  num_prop=num_prop, mask=0)
    a_none, _, _ = CORE.route_step(SN, obs, s1, st, st_est, torch.zeros(n, dtype=torch.bool), p0=p0, p1=p1,
                                   num_prop=num_prop, mask=0)
    assert torch.equal(a_all, a_st) and torch.equal(a_none, a_s1)
    with pytest.raises(ValueError):
        CORE.route_step(SN, obs, s1, st, st_est, torch.tensor([1, 0, 1, 0, 0]), p0=p0, p1=p1, num_prop=num_prop,
                        mask=0)


def test_route_step_mask1_on_the_real_layout(CORE, SN):
    n = 4
    est = _Est(P1 - P0, 0.01)
    obs = _harness_obs(n, W, P0, P1, NP, est, seed=3)
    obs0 = obs.clone()
    s1 = _Pol(lambda x: x[:, :18] * 0.5)
    st = _Pol(lambda x: torch.tanh(x[:, :18] + x[:, 65:83]))          # reads the masked last-action block
    rows = torch.tensor([True, True, False, False])
    actions, a_s1, a_st = CORE.route_step(SN, obs, s1, st, est, rows, p0=P0, p1=P1, num_prop=NP, mask=1)
    assert torch.equal(obs, obs0)
    (x_st, h), = st.calls
    assert h is True
    assert torch.equal(x_st, SN.stand_net_input(obs0, est, p0=P0, p1=P1, num_prop=NP, mask=1))
    x_unmasked = SN.stand_net_input(obs0, est, p0=P0, p1=P1, num_prop=NP, mask=0)
    assert torch.equal(x_st[:, P0:P1], est.inference(obs0[:, :NP]))   # the estimate is taken on the UNMASKED prop
    diff = (x_st != x_unmasked).any(dim=0).nonzero().reshape(-1).tolist()
    assert diff and all(c not in range(P0, P1) for c in diff)        # the mask writes other columns only
    assert torch.equal(actions[rows], a_st[rows]) and torch.equal(actions[~rows], a_s1[~rows])


# ===================================================================================== recorder / launcher (static)
def test_recorder_compiles_and_names_the_evaluated_deployment(tmp_path):
    py_compile.compile(str(REC_PATH), cfile=str(tmp_path / "rec.pyc"), doraise=True)
    src = REC_PATH.read_text()
    for needle in ("crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB/model_19994.pt",
                   "stand_network/runs/SN_Amask_s17/model_4000.pt",
                   "phase2/lr/SN_Amask_s17/suite__omni/seed001",
                   'parser.add_argument("--stand-mask", type=int, choices=(0, 1), default=1',
                   'parser.add_argument("--stand-rule", choices=CORE.STAND_RULES, default="schedule"',
                   "CORE.route_step(SN, obs, ctx.policy, ctx.stand_policy, ctx.stand_estimator, rows",
                   "P2L.build_with_template(make_on_policy_runner", "P2L.StandInEnv(env, template)",
                   "P2L.save_rng()", "P2L.restore_rng(before)", "P2L.arm_slot_latch(", "SN.env_reasons(WANT_ENV)",
                   "imageio.get_writer(", "CORE.yaw_cmd_stats(", "CORE.eval_slot_stand_flags(",
                   "CORE.stand_tag_from_ckpt(", "CORE.s1_tag_from_ckpt("):
        assert needle in src, needle
    assert "RecordVideo" not in src.replace("gym ``RecordVideo``", "")
    # caption names are derived from the loaded checkpoints, never a fixed default string
    assert re.search(r'"--stand-tag", type=str, default=None', src) and re.search(r'"--label", type=str, default=None', src)
    assert "args_cli.stand_tag}" not in src and 'f"{args_cli.label}' not in src


def test_p2l_functions_the_recorder_calls_exist():
    p2l = _load(P2L_PATH, "_t_snv_p2l")
    for name in ("build_with_template", "StandInEnv", "_load_extra", "save_rng", "restore_rng", "rng_equal",
                 "arm_slot_latch"):
        assert callable(getattr(p2l, name)), name
    assert callable(getattr(p2l.P2RMode, "rederive_on_reset", None))


def _isaac_pat() -> str:
    m = re.search(r"^ISAAC_PAT='([^']*)'$", SH_PATH.read_text(), flags=re.M)
    assert m, "ISAAC_PAT not found"
    return m.group(1)


def test_launcher_busy_pattern():
    pat = re.compile(_isaac_pat())
    py = "/home/nickmagus/krabby/isaac_venv/bin/python"
    busy = [f"{py} /x/sn_record_video.py --headless --prefix a", f"{py} -u /x/scripts/rsl_rl/train.py --task X",
            f"{py} /x/sn_stand_net_eval_crab_hex_gait.py --a", f"{py} /x/eval_crab_hex_gait.py",
            f"{py} /x/record_categories.py --headless", f"{py} /x/play_stream_video.py", f"{py} /x/play.py",
            f"{py} /x/sn_dagger.py --mode dagger", f"{py} /x/run_phases.py"]
    idle = [f"{py} -B /x/sn_video_schedule.py --seed 1", f"{py} -m pytest -q", f"bash -c {py} /x/sn_record_video.py",
            f"timeout --signal=TERM 5400 {py} /x/sn_record_video.py", "grep -E ^[^ ]*[i]saac_venv/bin/python",
            f"systemd-run --user --scope -q timeout 1 {py} /x/sn_record_video.py"]
    for line in busy:
        assert pat.search(line), line
    for line in idle:
        assert not pat.search(line), line


def test_launcher_syntax_and_dry_run_launches_nothing(tmp_path):
    subprocess.run(["bash", "-n", str(SH_PATH)], check=True)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "LAUNCHED"
    for tool in ("systemd-run", "setsid"):
        (fake_bin / tool).write_text(f"#!/bin/sh\ntouch {marker}\nexit 1\n")
        (fake_bin / tool).chmod(0o755)
    before = {p.name: p.read_bytes() for p in VIDEOS.iterdir()} if VIDEOS.is_dir() else {}
    env = dict(os.environ, PATH=f"{fake_bin}:{os.environ.get('PATH', '')}", SN_DDC=str(tmp_path / "ddc"),
               SN_VIDEO_OUT=str(tmp_path / "out"), CUDA_VISIBLE_DEVICES="")
    for extra in (["--smoke"], ["--seed", "7", "--", "--no-arrows"]):
        r = subprocess.run(["bash", str(SH_PATH), "--dry-run", *extra], env=env, capture_output=True, text=True,
                           timeout=300)
        assert r.returncode == 0, r.stderr
        out = r.stdout
        for needle in ("OMNI_KIT_ACCEPT_EULA=yes", "PYTHONUNBUFFERED=1", "MemoryMax=24G", "MemorySwapMax=0",
                       "--headless", "--enable_cameras", "timeout --signal=TERM -k 120 ", "sn_record_video.py"):
            assert needle in out, (needle, out)
        assert ("--sequence sn_random_smoke__omni" in out) == (extra == ["--smoke"])
    assert not marker.exists()
    assert not (tmp_path / "out").exists()
    after = {p.name: p.read_bytes() for p in VIDEOS.iterdir()} if VIDEOS.is_dir() else {}
    assert after == before                                                         # a dry run writes nothing
    r = subprocess.run(["bash", str(SH_PATH), "--dry-run", "--", "--manifest", "x"], env=env, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 5 and "set by this launcher" in r.stderr
    r = subprocess.run(["bash", str(SH_PATH), "--dry-run"], env=dict(env, SN_VIDEO_KILL_AFTER="0"),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 5 and "SN_VIDEO_KILL_AFTER" in r.stderr                 # the hang bound always kills
