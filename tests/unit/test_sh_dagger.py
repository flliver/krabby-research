"""Stand-height campaign (``2026-10-13_stand_height``) step 2: the height DAgger trainer's torch-free spec
(``sh_dagger_spec.py``), its env composition (``sh_starts.py``: start modes, the table, the command / push / termination
wrappers, the Kit probe) driven with stubs, the entry point (``sh_dagger.py``, pre-Kit path) and its launcher
(``sh_dagger.sh``, dry run / busy pattern / detach) -- CPU only, no Isaac / Kit.

Pinned:

* torch-free imports (rule L6) and no bytecode; the process env is the predecessor trainer's, verbatim (reused, not
  re-stated); the constants agree with facts.json / PLAN H5 (top_max follows an h_max re-freeze) and the step-4
  re-scope (CHARTER section 3: start shares 0.50 / 0.40 / 0 / 0.10, poses 0.75 / 0 / 0.25, tops U[0.635, 0.805]); the
  frozen block loads identically into sh_core / sh_table_core / sh_dagger_spec / sh_starts; the trainer never draws a
  ground start or a rod_min pose and draws every top in [0.635, 0.805];
* the cfg asserts (raw +-4.8 rod clip, scales, the wrapped terms present, compose once);
* the step-6 premise read (homed / STOOD ratios, falls, sigma-0 restore timeouts) and the smoke verdict (fail closed);
* the pose bank == facts.json's retracted poses and random-fold family; the start event's geometry per kind (table top
  by z placement under the robot's xy / yaw, ground spawn above the lowest hull point, random-height stance), the fold
  rule on tables, the h_cmd draw (35 % at 1.064, table bound), the forced first stand and its expiry (the slot change),
  the rise latches (up, hip re-arm), the table park, the termination combine, the push restore, the wrappers keep the
  signature the IsaacLab manager checks, the cfg composition and its fail-closed reasons, the Kit probe's h_meas;
* ``sh_dagger.py``: every pre-Kit refusal is exit code 2, ``--emit-env`` composes an accepted env, ``--read``;
  ``sh_dagger.sh``: the dry-run command (memory cap, SIGTERM timeout), refusals, the busy pattern (both campaigns'
  trainers and table tests count; the Kit-free calls do not), ``--detach`` returns only once the trainer is visible;
* the residual deadband (CHARTER 2026-10-14): ``--deadband`` reaches RunOpts and the provenance, bad values are refused
  pre-Kit, ``--read`` reports a run's recorded value (a run.json without it = 0), ``sh_dagger.sh`` passes it through.
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import math
import os
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
UNIT = Path(__file__).resolve().parent
TASK = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
CAMP = TASK / "experiments/2026-10-13_stand_height"
SCRIPTS = CAMP / "scripts"
PRED = TASK / "experiments/2026-10-11_stand_network/scripts"
ENTRY = SCRIPTS / "sh_dagger.py"
SH_SCRIPT = SCRIPTS / "sh_dagger.sh"
TERM_PY = TASK.parent / "crab_hex_forward_task/config/crab_hex/crab_hex_mdp_terminations.py"
CMD_MATH = TASK / "mdp/crab_hex_omni_command_math.py"


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
def SPEC():
    return _load(SCRIPTS / "sh_dagger_spec.py", "_t_sh_spec")


@pytest.fixture(scope="module")
def ST():
    return _load(SCRIPTS / "sh_starts.py", "_t_sh_starts")


@pytest.fixture(scope="module")
def ENT():
    return _load(ENTRY, "_t_sh_entry")


@pytest.fixture(scope="module")
def L():
    """The loop test's fakes (FakeEnv / FakeCmdBase / FakeRobot ...)."""
    return _load(UNIT / "test_sh_dagger_loop.py", "_t_sh_dagger_fakes")


@pytest.fixture(scope="module")
def bank(ST):
    return ST.pose_bank_rows()


@pytest.fixture(scope="module")
def stance(ST):
    return ST.stance_table()


def gens(seed=1):
    out = {}
    for k, name in enumerate(("starts", "hcmd")):
        g = torch.Generator()
        g.manual_seed(1000 * seed + k)
        out[name] = g
    return out


# ===================================================================================== imports / constants
def test_spec_starts_and_entry_are_torch_free_and_write_no_bytecode():
    code = textwrap.dedent(f"""\
        import importlib.util, sys
        for p in ({str(SCRIPTS / 'sh_dagger_spec.py')!r}, {str(SCRIPTS / 'sh_starts.py')!r}, {str(ENTRY)!r}):
            s = importlib.util.spec_from_file_location("_x", p)
            m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
        bad = [k for k in ("torch", "numpy", "yaml", "isaaclab") if k in sys.modules]
        print("BAD", bad); sys.exit(1 if bad else 0)
        """)
    before = set(SCRIPTS.iterdir())
    r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=120,
                       env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert set(SCRIPTS.iterdir()) == before and not (SCRIPTS / "__pycache__").exists()


def test_process_env_is_the_predecessor_trainers_verbatim(SPEC):
    P = _load(PRED / "sn_dagger_spec.py", "_t_sh_pred_spec")
    assert SPEC.expected_krabby_env() == P.expected_krabby_env()
    assert SPEC.train_env({"PATH": "/x", "KRABBY_FOO": "1"}) == P.train_env({"PATH": "/x", "KRABBY_FOO": "1"})
    assert (SPEC.S1_SHA256, SPEC.TASK_ID, SPEC.PRESET, SPEC.PLANT) == (P.S1_SHA256, P.TASK_ID, P.PRESET, P.PLANT)
    assert SPEC.env_reasons(SPEC.train_env({})) == []
    assert (SPEC.SIGMA_TABLE, SPEC.CONTROL_SHARE, SPEC.LR, SPEC.ONSET_SHARE, SPEC.W_MAX) == (
        P.SIGMA_TABLE, P.CONTROL_SHARE, P.LR, P.ONSET_SHARE, P.W_MAX)
    assert set(P.RECORD_KEYS) <= set(SPEC.RECORD_KEYS)


def test_constants_follow_facts_and_plan(SPEC):
    fr = json.loads((CAMP / "facts.json").read_text())["frozen"]
    assert (SPEC.H_MIN, SPEC.H_MAX) == (0.719, 1.346) == (fr["h_min_m"], fr["h_max_m"])
    assert SPEC.TOP_MIN_M == fr["top_min_m"] == 0.635 and SPEC.TOP_MAX_M == pytest.approx(0.805)
    assert SPEC.top_max_for(1.02, 0.805) == pytest.approx(0.756)          # an h_max re-freeze drops the tops
    assert SPEC.top_max_for(1.346, 0.805) == 0.805
    assert SPEC.START_KINDS == ("walk", "table", "ground", "random") == SPEC.SH.START_KINDS
    assert sum(SPEC.START_SHARES) == pytest.approx(1.0) and sum(SPEC.POSE_SHARES) == pytest.approx(1.0)
    assert SPEC.START_SHARES == (0.50, 0.40, 0.0, 0.10)                   # CHARTER step-4 decision: no ground starts
    assert SPEC.POSES == ("knee_folded", "rod_min", "random_fold") and SPEC.POSE_SHARES == (0.75, 0.0, 0.25)
    assert SPEC.DRAWN_START_KINDS == ("walk", "table", "random") and SPEC.DRAWN_POSES == ("knee_folded", "random_fold")
    assert (SPEC.SK_WALK, SPEC.SK_TABLE, SPEC.SK_GROUND, SPEC.SK_RANDOM) == (0, 1, 2, 3)   # the codes did not shift
    assert SPEC.TABLE_CMD_MARGIN_M == 0.264 and SPEC.H_CMD_ATOM == 1.064 and SPEC.H_CMD_ATOM_P == 0.35
    assert SPEC.first_stand_seconds(1.064, 0.719, margin=False) == pytest.approx(0.345 / 0.08 + 7.0)
    assert SPEC.first_stand_seconds(1.0, 1.0) == pytest.approx(7.5)
    lo, hi, atom = SPEC.table_cmd_bounds(0.805)
    assert lo == pytest.approx(1.069) and hi == 1.346 and atom is False
    lo, hi, atom = SPEC.table_cmd_bounds(0.5)
    assert lo == pytest.approx(0.764) and atom is True
    assert SPEC.MODES == ("dagger",) and SPEC.ARMS == {"mask": 1} and SPEC.HARNESS_CLIP == 4.8


def _cfg_ok(SPEC, ST):
    """A duck-typed cfg that passes the predecessor's and this spec's asserts."""
    NS = types.SimpleNamespace

    def CrabHexOmniCommand():  # noqa: N802 -- a stand-in carrying the class name the assert reads
        pass

    def push_by_setting_velocity(env, env_ids, velocity_range=None):
        pass

    def terminate_crab_hex_failure(env, limit_angle=0.5):
        pass

    cmd = NS(resampling_time_range=(3.0, 12.0), cmd_pool=SPEC.POOL, slot_probs=(0.85, 0.55, 0.1, 0.2),
             dir_band=None, speed_ellipse=(0.35, 0.35, 0.25), eval_slot_latch=False, class_type=CrabHexOmniCommand)
    env_cfg = NS(
        commands=NS(base_velocity=cmd), episode_length_s=40.0,
        parkours=NS(base_parkour=NS(move_up_frac=0.7, move_down_frac=0.4, freeze_terrain_levels=False)),
        terminations=NS(crab_failure=NS(func=terminate_crab_hex_failure, params={
            "limit_angle": 0.5, "contact_force_threshold": 500.0, "minimum_root_height_z": None}),
            total_terminates=NS(func=lambda env: None)),
        observations=NS(policy=NS(enable_corruption=False, extreme_parkour_observations=NS(params={
            "clock_hide": True, "facing_obs_clip": 0.6, "facing_obs_encoding": "raw"}))),
        actions=NS(joint_pos=NS(action_delay_steps=[1, 1], use_delay=True, clip={
            ".*_Body_CamShaft_RevoluteJoint": (-1.0, 1.0), ".*_Hip_Femur_RevoluteJoint": (-4.8, 4.8),
            ".*_Femur_Tibia_RevoluteJoint": (-4.8, 4.8)}, scale={
            ".*_Body_CamShaft_RevoluteJoint": math.pi, ".*_Hip_Femur_RevoluteJoint": 0.25,
            ".*_Femur_Tibia_RevoluteJoint": 0.25})),
        events=NS(rsi_reference_reset=None, push_by_setting_velocity=NS(func=push_by_setting_velocity,
                                                                        mode="interval")),
        sim=NS(dt=0.005), decimation=4, scene=NS())
    agent = NS(clip_actions=1.0, policy=NS(noise_std_type="soft", init_noise_std=0.8))
    return env_cfg, agent


def test_cfg_reasons_pass_and_refuse(SPEC, ST):
    env_cfg, agent = _cfg_ok(SPEC, ST)
    assert SPEC.cfg_reasons(env_cfg, agent) == []
    env_cfg.actions.joint_pos.clip[".*_Hip_Femur_RevoluteJoint"] = (-1.0, 1.0)
    assert any("4.8" in r for r in SPEC.cfg_reasons(env_cfg, agent))
    env_cfg, agent = _cfg_ok(SPEC, ST)
    env_cfg.decimation = 2
    assert any("0.02" in r for r in SPEC.cfg_reasons(env_cfg, agent))
    env_cfg, agent = _cfg_ok(SPEC, ST)
    env_cfg.events.push_by_setting_velocity = None
    assert any("push" in r for r in SPEC.cfg_reasons(env_cfg, agent))
    env_cfg, agent = _cfg_ok(SPEC, ST)
    env_cfg.scene.sh_table = object()
    assert any("compose once" in r for r in SPEC.cfg_reasons(env_cfg, agent))
    env_cfg, agent = _cfg_ok(SPEC, ST)
    agent.clip_actions = 4.8                                         # the CONFIG keeps 1.0 (runtime clip only)
    assert any("clip_actions" in r for r in SPEC.cfg_reasons(env_cfg, agent))


# ===================================================================================== premise / smoke verdict
def _rec(kind, ex, *, sigma=0.0, homed=True, stood=None, fall=False, first=False, dh=0.01, sched=8.0, ramp=1.0,
         it=5):
    return {"start_kind": kind, "kind": ex, "sigma": sigma, "sched_len0_s": sched, "cut0": False,
            "entry_kind": "spawn" if first else "handover", "end0": "fall" if fall else "reached",
            "homed0": homed, "first_stand": first, "stood0": stood, "abs_dh0": dh, "ramp0_s": ramp, "it_end": it}


def test_premise_read_rules(SPEC):
    recs = ([_rec("walk", "control") for _ in range(10)] + [_rec("walk", "student", homed=i < 3) for i in range(10)]
            + [_rec("table", "control", first=True, stood=True) for _ in range(10)]
            + [_rec("table", "student", first=True, stood=i < 8) for i in range(10)]
            + [_rec("ground", "student", first=True, stood=True, sigma=0.0) for _ in range(5)])  # an absent kind
    log = [{"it": 5, "restore": {
        "control": {"sigma0": {"walk": {"engaged": 10, "completed": 10, "timed_out": 0, "merged": 0, "reset_cut": 0}}},
        "student": {"sigma0": {"walk": {"engaged": 10, "completed": 9, "timed_out": 1, "merged": 0, "reset_cut": 0}},
                    "sigma_pos": {"walk": {"engaged": 50, "completed": 0, "timed_out": 50, "merged": 0,
                                           "reset_cut": 0}}}}}]
    pr = SPEC.premise_read(recs, log)
    w = pr["by_kind"]["walk"]
    assert w["control"]["homed"]["p"] == 1.0 and w["student"]["homed"]["p"] == 0.3
    t = pr["by_kind"]["table"]
    assert t["student"]["stood"]["p"] == 0.8 and t["control"]["stood"]["p"] == 1.0
    # the step-4 decision: no ground start type (share 0) -- its records are counted, never read, never a STOP
    assert set(pr["by_kind"]) == {"walk", "table", "random"} and "ground" not in pr["by_kind"]
    assert pr["absent_kinds"] == {"ground": {"share": 0.0, "records": 5}}
    by = {c["check"]: c for c in pr["checks"]}
    assert by["walk homed student >= 0.5 x R_h"]["stop"] is True
    assert by["table stood student >= 0.5 x R_h"]["stop"] is False
    assert not any(c["check"].startswith("ground") for c in pr["checks"]) and len(pr["checks"]) == 5
    assert SPEC.premise_read([r for r in recs if r["start_kind"] == "ground"], log)["stop"] in (True, False, None)
    r = by["restore-timeout share (sigma-0 episodes) student <= 2 x R_h + 0.01"]
    assert r["student"] == pytest.approx(0.1) and r["control"] == 0.0 and r["stop"] is True   # sigma_pos ignored
    assert pr["restore_pooled_all_sigma"]["student"]["timed_out"] == 51
    assert pr["stop"] is True and "walk homed student >= 0.5 x R_h" in pr["stop_reasons"]
    # the window: nothing in [6, 9)
    empty = SPEC.premise_read(recs, log, 6, 9)
    assert empty["stop"] is None
    # |h - h_cmd| on holds long enough for their ramp + 2 s only
    recs2 = [_rec("walk", "control", dh=0.01, sched=8.0, ramp=1.0), _rec("walk", "control", dh=0.5, sched=3.0,
                                                                          ramp=2.0)]
    d = SPEC.premise_read(recs2)["by_kind"]["walk"]["control"]["abs_dh"]
    assert d["n"] == 1 and d["median"] == 0.01


def test_smoke_verdict_fails_closed(SPEC):
    names = [n for n, _ in SPEC.SMOKE_DAGGER]
    ok = [{"name": n, "pass": True} for n in names]
    assert SPEC.smoke_verdict(ok)["verdict"] == "PASS"
    assert SPEC.smoke_verdict(ok[1:])["verdict"] == "FAIL"
    assert SPEC.smoke_verdict(ok + [{"name": "extra", "pass": True}])["verdict"] == "FAIL"
    assert SPEC.smoke_verdict(ok + [ok[0]])["verdict"] == "FAIL"
    bad = [dict(c) for c in ok]
    bad[3]["pass"] = None
    assert SPEC.smoke_verdict(bad)["verdict"] == "FAIL"
    bad[3]["pass"] = False
    v = SPEC.smoke_verdict(bad)
    assert v["verdict"] == "FAIL" and v["failed"] == [names[3]]
    # PLAN step 5's list is covered
    for need in ("start_mode_shares", "table_tops", "ground_spawns", "h_cmd_distribution", "limiter",
                 "residual_labels_within_1", "restores_complete", "table_first_stand", "table_parked_at_slot_change",
                 "pushes_off_until_up", "ground_episodes_hip_rearm", "restore_rows_stand_net_inputs",
                 "restore_rows_in_reservoir_with_rh_labels", "s1_rows_pm1", "stand_hip_targets_below_minus1",
                 "forward_parity"):
        assert need in names


# ===================================================================================== the pose bank / stance
def test_pose_bank_is_facts_json(bank):
    fx = json.loads((CAMP / "facts.json").read_text())["c_table_ground"]
    assert [r["name"] for r in bank[:2]] == ["knee_folded", "rod_min"] and len(bank) == 2 + 81
    for r in bank[:2]:
        f = fx["poses"][r["name"]]
        assert r["ground_root_z"] == pytest.approx(f["ground"]["spawn_root_z_m"], abs=1e-9)
        assert r["table_top_min"] == pytest.approx(f["table_top_min_m_legs_0p02_above_floor"], abs=1e-9)
        assert math.degrees(r["hip"]) == pytest.approx(f["hip_deg"]) and \
            math.degrees(r["knee_left"]) == pytest.approx(f["knee_left_deg"])
    fam = bank[2:]
    rf = fx["random_fold_family"]
    assert sum("hip_plate" in r["first_contact"] for r in fam) / 81 == pytest.approx(rf["share_resting_on_hip_plates"])
    roots = sorted(r["ground_root_z"] for r in fam)
    assert [roots[0], roots[40], roots[-1]] == pytest.approx(rf["ground_spawn_root_z_m"], abs=1e-9)
    tm = sorted(r["table_top_min"] for r in fam)
    assert [tm[0], tm[40], tm[-1]] == pytest.approx(rf["table_top_min_for_legs_clear_m"], abs=1e-9)


def test_stance_table(ST, stance):
    K = ST.kin()
    hs, th, tk, hg = stance
    assert hs[0] == 0.719 and hs[-1] == pytest.approx(1.346)
    i = int(abs(hs - 1.064).argmin())
    assert abs(hs[i] - 1.064) < 1e-9
    assert th[i] == pytest.approx(K.TH0, abs=1e-12) and tk[i] == pytest.approx(K.TK0, abs=1e-12)
    assert hg[i] == pytest.approx(1.064 - K.H_OFFSET, abs=1e-9)
    import numpy as np
    assert np.all(np.diff(th) > 0) and np.all(np.diff(hg) > 0)


def test_slot_stand_and_lowest_point_match_their_sources(ST):
    tree = ast.parse(CMD_MATH.read_text())
    lit = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
           for t in n.targets if isinstance(t, ast.Name) and t.id == "SLOT_STAND"}
    assert lit == {"SLOT_STAND": ST.SLOT_STAND}
    assert ST.BODY_HALF_Z_M == ST.stc().BODY_HALF_M[2]
    pos = torch.tensor([[0.0, 0.0, 1.0]])
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    low, cz = ST.lowest_and_cos(pos, q)
    assert float(low) == pytest.approx(1.0 - 0.15875) and float(cz) == pytest.approx(1.0)


# ===================================================================================== the start event
def _env(L, ST, bank, stance, n=200, seed=3, shares=None, pose_shares=None, **kw):
    """The loop test's FakeEnv (the REAL start event / command subclass / wrappers), optionally with other shares."""
    g = gens(seed)
    if shares is None and pose_shares is None:
        return L.FakeEnv(ST, g, bank=bank, stance=stance, n=n, seed=seed, **kw)

    class Shared(ST.StartState):
        def __init__(self, *a, **k):
            k["shares"] = shares or ST.SPEC.START_SHARES
            k["pose_shares"] = pose_shares or ST.SPEC.POSE_SHARES
            super().__init__(*a, **k)

    mod = types.SimpleNamespace(**{k: getattr(ST, k) for k in dir(ST) if not k.startswith("__")})
    mod.StartState = Shared
    return L.FakeEnv(mod, g, bank=bank, stance=stance, n=n, seed=seed, **kw)


def test_start_event_geometry_per_kind(L, ST, bank, stance):
    env = _env(L, ST, bank, stance, n=200)
    st, rd, td = env.state, env.robot.data, env.table.data
    kind = st.kind
    assert set(kind.tolist()) == {0, 1, 3}                                 # ground share 0: never drawn (step 4)
    z = rd.root_link_pose_w[:, 2].double()
    bidx = torch.where(st.pose == 2, 2 + st.fold, st.pose)
    walk, table, ground, rand = (kind == 0), (kind == 1), (kind == 2), (kind == 3)
    # walk: the ring spawn untouched, the table parked
    assert torch.allclose(z[walk], torch.full_like(z[walk], 1.085))
    assert torch.all(td.root_link_pose_w[~table, 2] == -20.0)
    # table: belly 5 mm above the top (floor 0), the cuboid under the root, x along the body x (same yaw)
    top = st.top[table]
    assert torch.allclose(z[table], top + 0.15875 + 0.005, atol=1e-6)
    tp = td.root_link_pose_w[table]
    assert torch.allclose(tp[:, 2].double(), top - 1.0, atol=1e-6)
    assert torch.allclose(tp[:, :2], rd.root_link_pose_w[table, :2])
    yaw_r = env.robot.data.heading_w[table]
    yaw_t = 2 * torch.atan2(tp[:, 6], tp[:, 3])
    assert torch.allclose(torch.cos(yaw_r - yaw_t), torch.ones_like(yaw_r), atol=1e-5)
    assert torch.all(top >= ST.SPEC.TOP_MIN_M) and torch.all(top <= ST.SPEC.TOP_MAX_M)
    # retracted joints = the bank pose (knees mirrored on the right legs), targets = joints
    for rows in (table, ground):
        jp = rd.joint_pos[rows]
        b = bidx[rows]
        hip = torch.tensor([bank[int(i)]["hip"] for i in b])
        knee = torch.tensor([bank[int(i)]["knee_left"] for i in b])
        assert torch.allclose(jp[:, L.HIP_J], hip[:, None].float().expand(-1, 6))
        assert torch.allclose(jp[:, L.KNEE_J], (knee[:, None] * torch.tensor([1., -1, 1, -1, 1, -1])).float())
        assert torch.equal(env.robot.targets[rows], jp)
    # table rows: every random fold allowed at its top (legs >= 2 cm above the floor)
    tf = table & (st.pose == 2)
    for i in torch.nonzero(tf).flatten().tolist():
        assert bank[2 + int(st.fold[i])]["table_top_min"] <= float(st.top[i]) + 1e-12
    # ground: root z = max floor (0) + the pose's lowest hull point + 5 mm
    gz = torch.tensor([bank[int(i)]["ground_root_z"] for i in bidx[ground]], dtype=torch.float64)
    assert torch.allclose(z[ground], gz, atol=1e-6)
    # random height: stance joints at h, root z = the rigid height + 5 mm, analytic h_spawn
    hs, th, tk, hg = stance
    for i in torch.nonzero(rand).flatten().tolist()[:10]:
        hip = float(rd.joint_pos[i, L.HIP_J[0]])
        j = int(abs(th - hip).argmin())
        assert float(z[i]) == pytest.approx(hg[j] + 0.005, abs=2e-3)
    assert torch.all(st.pose[walk | rand] == -1)
    # per-env flags
    assert torch.equal(st.hip_disarmed, ground) and torch.equal(st.up, walk | rand)
    assert torch.equal(st.table_active, table)
    assert torch.allclose(st.h_spawn, z, atol=1e-6)                                  # floor 0
    # the command reset forced the first stand on every non-walk row (pending consumed)
    assert not st.first_pending.any() and torch.equal(st.first_active, ~walk)
    nw = ~walk
    assert torch.all(env.cmd.slot_id[nw] == 0) and torch.all(env.cmd.speed_cmd[nw] == 0)
    assert torch.all(env.cmd.type_id[nw] == 1)                                     # the fake pool's "stand" index
    flen = (st.h_cmd - st.h_spawn.clamp(0.719, 1.346)).abs() / 0.08 + 7.5
    assert torch.allclose(env.cmd.time_left[nw].double(), flen[nw], atol=1e-5)
    assert torch.allclose(st.first_len_s[nw], flen[nw])
    assert torch.all(st.h_cmd[table] >= st.top[table] + 0.264 - 1e-9)
    assert torch.all((st.h_cmd >= 0.719) & (st.h_cmd <= 1.346))
    c = st.counts()
    assert c["floor_fallbacks"] == 0 and sum(c[f"episodes_{k}"] for k in ST.SPEC.START_KINDS) == 200
    assert c["forced_walk"] == 0 and c["forced_table"] == c["episodes_table"]


def test_ground_spawn_reads_the_floor_under_the_poses_own_points(L, ST, bank, stance):
    """Review fix: a ground start's root z = the HIGHEST floor under the drawn pose's own probe points (every leg's
    pivot / knee / toe + the body corners / centre: sh_table_core.probe_points_body, the K3 rule) + the pose's lowest
    hull point + 5 mm -- not the max over the root and the default toes, where a folded pad / hip plate does not rest.
    h_spawn keeps the default-toe mean."""
    TC = _load(SCRIPTS / "sh_table_core.py", "_t_sh_dagger_tc")
    for row in (bank[0], bank[1], bank[2 + 40]):
        assert ST.pose_probe_points(row["hip"], row["knee_left"]) == TC.probe_points_body(row["hip"], row["knee_left"])

    def bumpy(starts, dirs, mesh=None):
        hits = starts.clone()
        hits[:, 2] = 0.03 * torch.sin(9.0 * starts[:, 0]) * torch.cos(7.0 * starts[:, 1])
        return (hits,)

    env = _env(L, ST, bank, stance, n=64, seed=12, shares=(0.0, 0.0, 1.0, 0.0))
    st = env.state
    st.raycast_fn = bumpy
    env._reset(torch.arange(64))
    assert torch.all(st.kind == ST.SPEC.SK_GROUND) and st.counts()["floor_fallbacks"] == 0
    rd = env.robot.data
    z = rd.root_link_pose_w[:, 2].double()
    xy = rd.root_link_pose_w[:, :2].double()
    yaw = rd.heading_w.double()
    bidx = torch.where(st.pose == 2, 2 + st.fold, st.pose)
    differ = 0
    for i in range(64):
        row = bank[int(bidx[i])]
        c, s = math.cos(float(yaw[i])), math.sin(float(yaw[i]))

        def floor_at(pts):
            w = torch.tensor([[float(xy[i, 0]) + c * x - s * y, float(xy[i, 1]) + s * x + c * y, 50.0]
                              for x, y in pts], dtype=torch.float64).float()
            return bumpy(w, None)[0][:, 2].double()

        f_pose = float(floor_at(ST.pose_probe_points(row["hip"], row["knee_left"])).max())
        f7 = floor_at(ST.floor_points_body())
        assert float(z[i]) == pytest.approx(f_pose + row["ground_root_z"], abs=2e-6), i
        assert float(st.h_spawn[i]) == pytest.approx(float(z[i]) - float(f7[1:].mean()), abs=2e-6)
        differ += abs(f_pose - float(f7.max())) > 1e-3
    assert differ > 32                                                     # the old 7-point rule would differ


def test_start_draws_are_seeded_and_private(L, ST, bank, stance):
    torch.manual_seed(5)
    before = torch.get_rng_state()
    a = _env(L, ST, bank, stance, n=64, seed=9)
    b = _env(L, ST, bank, stance, n=64, seed=9)
    assert torch.equal(torch.get_rng_state(), before)                     # the global stream never moved
    for k in ("kind", "pose", "fold", "top", "h_cmd", "h_spawn"):
        x, y = getattr(a.state, k), getattr(b.state, k)
        assert torch.equal(torch.nan_to_num(x.double(), nan=-9), torch.nan_to_num(y.double(), nan=-9)), k
    c = _env(L, ST, bank, stance, n=64, seed=10)
    assert not torch.equal(a.state.kind, c.state.kind)


def test_fold_rule_and_fallback_on_tables(L, ST, stance):
    S = ST.stc()
    kf, rm = S.retracted_pose("knee_folded"), S.retracted_pose("rod_min")
    rows = [ST.pose_row("knee_folded", kf["hip_rad"], kf["knee_left_rad"]),
            ST.pose_row("rod_min", rm["hip_rad"], rm["knee_left_rad"])]
    for tmin in (0.67, 0.71, 0.75):                                      # synthetic folds with chosen floor bounds
        rows.append(dict(rows[0], name="random_fold", table_top_min=tmin))  # (inside the tops U[0.635, 0.805])
    env = _env(L, ST, rows, stance, n=300, seed=4, shares=(0.0, 1.0, 0.0, 0.0), pose_shares=(0.0, 0.0, 1.0))
    st = env.state
    assert torch.all(st.kind == 1)
    low = st.top < 0.67
    assert low.any() and (~low).any()
    assert torch.all(st.pose[low] == 0)                                  # no fold allowed -> knee_folded, counted
    assert st.counts()["fold_fallbacks"] == float(low.sum())
    hi = ~low
    assert torch.all(st.pose[hi] == 2)
    tmins = torch.tensor([0.67, 0.71, 0.75], dtype=torch.float64)
    assert torch.all(tmins[st.fold[hi]] <= st.top[hi] + 1e-12)
    top_hi = st.top > 0.75
    assert len(set(st.fold[top_hi].tolist())) == 3                       # uniform over the allowed cells


def test_trainer_never_draws_ground_or_rod_min_and_tops_are_tall(L, ST, bank, stance):
    """CHARTER step-4 decision: the start event (the trainer's real StartState) never draws a ground start or a rod_min
    pose (share 0: an empty searchsorted interval), draws every table top in [0.635, 0.805], and keeps the table /
    knee_folded / random-fold shares (0.40 / 0.75 / 0.25)."""
    env = _env(L, ST, bank, stance, n=2000, seed=31)
    st = env.state
    kinds, poses, tops = [st.kind.clone()], [st.pose.clone()], [st.top.clone()]
    for _ in range(4):
        env._reset(torch.arange(2000))
        kinds.append(st.kind.clone())
        poses.append(st.pose.clone())
        tops.append(st.top.clone())
    kind, pose, top = torch.cat(kinds), torch.cat(poses), torch.cat(tops)
    n = kind.numel()
    assert not bool((kind == ST.SPEC.SK_GROUND).any()) and not bool((pose == 1).any())          # never drawn
    c = st.counts()
    assert c["episodes_ground"] == 0 and sum(c["poses_ground"]) == 0 and c["poses_table"][1] == 0
    table = kind == ST.SPEC.SK_TABLE
    assert abs(float(table.double().mean()) - 0.40) < 0.02 and abs(float((kind == 0).double().mean()) - 0.50) < 0.02
    t = top[table]
    assert float(t.min()) >= 0.635 - 1e-12 and float(t.max()) <= 0.805 + 1e-12 and float(t.min()) < 0.65
    assert float(t.max()) > 0.79 and torch.isnan(top[~table]).all()
    tp = pose[table]
    assert set(tp.tolist()) == {0, 2} and abs(float((tp == 0).double().mean()) - 0.75) < 0.03
    assert torch.all(pose[~table] == -1) and c["fold_fallbacks"] == 0      # every top >= 0.635 admits a fold
    assert not st.hip_disarmed.any()                                       # no ground start: the clause stays armed
    # the categorical rule itself: the u values at and around the empty intervals
    cum = torch.cumsum(torch.tensor(ST.SPEC.START_SHARES, dtype=torch.float64), 0)
    u = torch.tensor([0.0, 0.4999999, 0.5, 0.8999999, 0.9, 0.9000001, 0.9999999], dtype=torch.float64)
    assert torch.searchsorted(cum, u, right=True).clamp(max=3).tolist() == [0, 0, 1, 1, 3, 3, 3]
    cp = torch.cumsum(torch.tensor(ST.SPEC.POSE_SHARES, dtype=torch.float64), 0)
    u = torch.tensor([0.0, 0.7499999, 0.75, 0.9999999], dtype=torch.float64)
    assert torch.searchsorted(cp, u, right=True).clamp(max=2).tolist() == [0, 0, 2, 2]


def test_frozen_facts_load_consistently_into_every_module(SPEC, ST):
    """facts.json ``frozen`` (re-scoped at the step-4 stop) is the ONE source: sh_core / sh_table_core / sh_dagger_spec
    / sh_starts load the same block and derive the same table range and K3 tops."""
    fr = json.loads((CAMP / "facts.json").read_text())["frozen"]
    TC = _load(SCRIPTS / "sh_table_core.py", "_t_sh_dagger_tc_frozen")
    SHC = _load(SCRIPTS / "sh_core.py", "_t_sh_dagger_core_frozen")
    for mod in (SHC, TC.SH, SPEC.SH, ST.SPEC.SH):
        assert mod.FROZEN == fr and (mod.H_MIN, mod.H_MAX) == (fr["h_min_m"], fr["h_max_m"]) == (0.719, 1.346)
    assert str(fr["status"]).startswith("FROZEN")
    assert fr["top_min_m"] == 0.635 and fr["top_max_train_m"] == 0.805
    assert fr["k3_eval_tops_m"] == [0.635, 0.6985, 0.762] and fr["k3_reported_tops_m"] == [0.4572]
    assert fr["k3_gated_poses"] == ["knee_folded", "random"]
    old = fr["previous_frozen_step4"]
    assert old["top_min_m"] == pytest.approx(0.41239, abs=1e-5) and old["k3_eval_tops_m"] == [0.4572, 0.635, 0.762]
    assert "step-4" in old["reason"] and "CHARTER" in old["reason"]
    assert SPEC.TOP_MIN_M == ST.SPEC.TOP_MIN_M == TC.TOP_MIN_M == fr["top_min_m"]
    assert SPEC.TOP_MAX_M == ST.SPEC.TOP_MAX_M == TC.TOP_MAX_TRAIN_M == fr["top_max_train_m"]
    assert list(TC.K3_TOPS_M) == fr["k3_eval_tops_m"] and list(TC.K3_REPORTED_TOPS_M) == fr["k3_reported_tops_m"]
    assert list(TC.K3_GATED_POSES) == fr["k3_gated_poses"]
    assert all(TC.TOP_MIN_M <= t <= TC.TOP_MAX_TRAIN_M for t in TC.K3_TOPS_M)   # every gated top is a trained top
    # sh_facts never overwrites the frozen block on a re-run (it carries an existing FROZEN block over)
    FA = _load(SCRIPTS / "sh_facts.py", "_t_sh_dagger_facts_frozen")
    assert FA._existing_frozen() == fr


def test_h_cmd_draw_distribution_and_table_bound(ST, bank, stance):
    big = ST.StartState(20000, "cpu", gen_start=gens(2)["starts"], gen_hcmd=gens(2)["hcmd"], bank=bank,
                        stance=stance)
    ids = torch.arange(20000)
    h = big.draw_h_cmd(ids)
    atom = h == 1.064
    assert abs(float(atom.double().mean()) - 0.35) < 0.015
    rest = h[~atom]
    assert float(rest.min()) >= 0.719 and float(rest.max()) <= 1.346
    assert abs(float(rest.mean()) - (0.719 + 1.346) / 2) < 0.01
    big.table_active[:] = True
    big.top[:] = 0.6
    h = big.draw_h_cmd(ids)
    assert float(h.min()) >= 0.864 - 1e-12 and abs(float((h == 1.064).double().mean()) - 0.35) < 0.015
    big.top[:] = 0.805                                                 # 1.064 < top + 0.264: no atom
    h = big.draw_h_cmd(ids)
    assert float(h.min()) >= 1.069 - 1e-12 and not bool((h == 1.064).any())
    c = big.counts()
    assert c["hcmd_table_violations"] == 0 and c["hcmd_table_n"] == 40000


def test_ground_first_command_is_bounded_like_k3_and_never_up_is_counted(L, ST, bank, stance):
    """Review fix: a ground episode's FIRST command draws >= max(h_min, h_spawn + 0.14625) (K3's ground rule), so a
    fold that spawns high still rises >= 0.10 m and latches "up" / the hip re-arm; later commands are U[h_min, h_max]
    again. Episodes that end without ever latching "up" are counted per start kind (env_counts.json)."""
    n = 6
    st = ST.StartState(n, "cpu", gen_start=gens(2)["starts"], gen_hcmd=gens(2)["hcmd"], bank=bank, stance=stance)
    spawn = torch.tensor([0.311, 0.561, 0.60, 0.80, 0.95, math.nan], dtype=torch.float64)
    lo = torch.nan_to_num(spawn + ST.SPEC.GROUND_FIRST_CMD_RISE_M, nan=0.719).clamp(min=0.719)
    assert ST.SPEC.GROUND_FIRST_CMD_RISE_M == pytest.approx(0.14625)
    st.kind[:] = ST.SPEC.SK_GROUND
    st.h_spawn[:] = spawn
    ids = torch.arange(n)
    hs = []
    for _ in range(400):
        st.first_pending[:] = True
        hs.append(st.draw_h_cmd(ids))
    h = torch.stack(hs)
    assert torch.all(h >= lo - 1e-12) and torch.all(h <= 1.346)
    assert float(h[:, 0].min()) < 0.73 and float(h[:, 1].min()) < 0.73      # knee_folded / rod_min spawns: h_min
    assert not bool((h[:, 4] == 1.064).any())                                # lo 1.096 > 1.064: no atom
    c = st.counts()
    assert c["hcmd_ground_first_n"] == 400 * n and c["hcmd_ground_first_bounded"] == 400 * 3
    assert c["hcmd_ground_first_violations"] == 0 and c["hcmd_free_n"] == 400 * (n - 3)
    st.first_pending[:] = False                                              # later commands: unbounded again
    later = torch.stack([st.draw_h_cmd(ids) for _ in range(400)])
    assert float(later[:, 4].min()) < 0.9
    # never-up episodes: counted when the episode ends (the reset event snapshots the outgoing episode)
    env = _env(L, ST, bank, stance, n=8, seed=3)
    s = env.state
    s.kind[:] = torch.tensor([2, 2, 1, 1, 0, 0, 3, 3])
    s.up[:] = torch.tensor([False, True, False, True, True, True, True, True])
    before = s.counts()
    env._reset(torch.arange(8))
    after = s.counts()
    d = {k: after[k] - before[k] for k in ("ended_ground", "never_up_ground", "ended_table", "never_up_table")}
    assert d == {"ended_ground": 2, "never_up_ground": 1, "ended_table": 2, "never_up_table": 1}


def test_forced_first_stand_and_its_expiry(L, ST, bank, stance):
    env = _env(L, ST, bank, stance, n=12, seed=6)
    st, cmd = env.state, env.cmd
    # a twin base command on the same generator state: non-pending rows must keep the base draws
    twin = L.FakeCmdBase(env.unwrapped, 12, torch.Generator().set_state(cmd.g.get_state()))
    st.first_pending[:] = torch.tensor([True, False] * 6)
    st.first_active[:] = False
    st.h_spawn[:] = 0.80
    ids = torch.arange(12)
    env.unwrapped.episode_length_buf[:] = 7
    counter = cmd.command_counter.clone()
    cmd._resample(ids)
    twin._resample(ids)
    pend = torch.tensor([True, False] * 6)
    assert torch.all(cmd.slot_id[pend] == 0) and torch.all(cmd.speed_cmd[pend] == 0)
    assert torch.allclose(cmd.heading_target[pend], env.robot.data.heading_w[pend])
    flen = (st.h_cmd[pend] - 0.80).abs() / 0.08 + 7.5
    assert torch.allclose(cmd.time_left[pend].double(), flen, atol=1e-5)
    for k in ("slot_id", "speed_cmd", "heading_target", "time_left", "type_id"):
        assert torch.equal(getattr(cmd, k)[~pend], getattr(twin, k)[~pend]), k
    assert torch.equal(cmd.command_counter, counter + 1)                  # the base increments after the wrapper
    assert not st.first_pending.any() and torch.equal(st.first_active, pend) and not st.first_done.any()
    # the forced command expires at its next resample (merge or not): the slot change the table park keys on
    env.unwrapped.episode_length_buf[:] = 90
    cmd._resample(torch.tensor([0, 1, 2]))
    assert st.first_done.tolist()[:3] == [True, False, True] and st.change_ep.tolist()[:3] == [90, -1, 90]
    assert not st.first_active[[0, 2]].any() and not st.first_done[3:].any()


def test_post_physics_rise_latches_and_table_park(L, ST, bank, stance):
    env = _env(L, ST, bank, stance, n=4, seed=8)
    st, rd = env.state, env.robot.data
    st.kind[:] = torch.tensor([2, 1, 1, 0])
    st.spawn_lowest[:] = torch.tensor([0.15, 0.485, 0.485, 0.9], dtype=torch.float64)   # table: top_w + 5 mm
    st.hip_disarmed[:] = torch.tensor([True, False, False, False])
    st.up[:] = torch.tensor([False, False, False, True])
    st.table_active[:] = torch.tensor([False, True, True, False])
    st.top_w[:] = torch.tensor([math.nan, 0.48, 0.48, math.nan], dtype=torch.float64)
    st.first_done[:] = torch.tensor([False, True, False, False])
    st.first_checked[:] = False
    st.change_ep[:] = torch.tensor([-1, 40, -1, -1])
    rd.root_link_pose_w[:, 3:] = torch.tensor([1.0, 0, 0, 0])
    env.unwrapped.episode_length_buf[:] = 41
    rd.root_link_pose_w[:, 2] = torch.tensor([0.15 + 0.15875 + 0.03, 0.485 + 0.15875 + 0.03, 0.9, 1.0])
    st.post_physics(env.unwrapped)                                   # rise 0.03: nothing latches; table not clear
    assert st.hip_disarmed[0] and not st.up[0] and st.table_active[1] and st.clear_first[1] == 0
    rd.root_link_pose_w[:, 2] = torch.tensor([0.15 + 0.15875 + 0.12, 0.485 + 0.15875 + 0.12, 0.9, 1.0])
    env.unwrapped.episode_length_buf[:] = 42
    st.post_physics(env.unwrapped)
    assert not st.hip_disarmed[0] and st.rearm_ep[0] == 42 and st.up[0] and st.up_ep[0] == 42
    assert st.up[1] and st.up_ep[1] == 42
    # row 1: first command expired + belly 0.12 above the top -> parked; row 2 (first command still active) stays
    assert not st.table_active[1] and st.park_ep[1] == 42 and float(env.table.data.root_link_pose_w[1, 2]) == -20.0
    assert st.table_active[2]
    c = st.counts()
    assert c["parks"] == 1 and c["parks_before_change"] == 0 and c["park_delay_min"] == 2.0
    assert c["first_checks"] == 1 and c["first_checks_clear"] == 0             # the first check was not clear


def test_termination_combine_and_cause(ST, bank, stance):
    st = ST.StartState(4, "cpu", gen_start=gens()["starts"], gen_hcmd=gens()["hcmd"], bank=bank, stance=stance)
    st.hip_disarmed[:] = torch.tensor([True, True, False, False])
    full = torch.tensor([True, True, True, False])
    tilt = torch.tensor([False, True, False, False])
    out = st.combine_termination(full, tilt)
    assert out.tolist() == [False, True, True, False]
    assert st.term_cause.tolist() == [0, 1, 2, 0]
    c = st.counts()
    assert c["hip_would_fire_disarmed"] == 1 and c["hip_term_disarmed"] == 0 and c["hip_term_armed"] == 1


def test_push_restores_envs_not_up(L, ST, bank, stance):
    env = _env(L, ST, bank, stance, n=6, seed=2)
    st, rd = env.state, env.robot.data
    st.up[:] = torch.tensor([True, False, True, False, True, True])
    rd.root_vel_w[:] = 0.25
    before = rd.root_vel_w.clone()
    env.push(env.unwrapped, None, velocity_range={"x": (-0.5, 0.5)})
    down = ~st.up
    assert torch.equal(rd.root_vel_w[down], before[down])
    assert not torch.equal(rd.root_vel_w[~down][:, :2], before[~down][:, :2])
    env.push(env.unwrapped, torch.tensor([0, 1]), velocity_range={"x": (-0.5, 0.5)})
    assert torch.equal(rd.root_vel_w[1], before[1])
    c = st.counts()
    assert c["push_events"] == 2 and c["push_suppressed"] == 3 and c["push_applied"] == 5
    assert c["push_restore_mismatch"] == 0


def _manager_check(func, params: dict, min_argc: int):
    """IsaacLab ManagerBase._resolve_common_term_cfg's static parameter check (manager_base.py:361-374)."""
    args = inspect.signature(func).parameters
    with_d = [a for a in args if args[a].default is not inspect.Parameter.empty]
    without = [a for a in args if args[a].default is inspect.Parameter.empty]
    names = without + with_d
    if len(names) > min_argc and set(names[min_argc:]) != set(list(params) + with_d):
        raise ValueError(f"{names} vs {params}")


def test_wrappers_keep_the_signature_the_manager_checks(ST, bank, stance):
    tree = ast.parse(TERM_PY.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "terminate_crab_hex_failure")
    src_args = [a.arg for a in fn.args.args]

    def terminate_crab_hex_failure(env, asset_cfg=None, limit_angle=1.05, minimum_root_height_z=None,
                                   contact_force_threshold=15.0, hip_contact_sensor_cfg=None):
        return contact_force_threshold

    assert list(inspect.signature(terminate_crab_hex_failure).parameters) == src_args
    st = ST.StartState(2, "cpu", gen_start=gens()["starts"], gen_hcmd=gens()["hcmd"], bank=bank, stance=stance)
    w = ST.wrap_termination(terminate_crab_hex_failure, st)
    assert inspect.signature(w) == inspect.signature(terminate_crab_hex_failure)
    assert w.__name__ == "sh_wrapped_terminate_crab_hex_failure" and w.__module__ == ST.__name__   # dumps say so
    params = {"asset_cfg": None, "limit_angle": 0.5, "minimum_root_height_z": None, "contact_force_threshold": 500.0,
              "hip_contact_sensor_cfg": None}
    _manager_check(w, params, 1)
    with pytest.raises(ValueError):
        _manager_check(lambda env, **kw: None, params, 1)            # what an unwrapped **kwargs wrapper would do

    def push_by_setting_velocity(env, env_ids, velocity_range, asset_cfg=None):
        pass

    p = ST.wrap_push(push_by_setting_velocity, st)
    _manager_check(p, {"velocity_range": {}}, 2)
    _manager_check(ST.make_reset_event(st), {}, 2)
    with pytest.raises(RuntimeError, match="already wrapped"):
        ST.wrap_termination(w, st)
    with pytest.raises(RuntimeError, match="already wrapped"):
        ST.wrap_push(p, st)


class _Cfg:
    """configclass-like attribute holder (vars() order = insertion order)."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_compose_env_cfg_and_reasons(L, ST, bank, stance):
    NS = types.SimpleNamespace

    def push_by_setting_velocity(env, env_ids, velocity_range=None):
        pass

    def terminate_crab_hex_failure(env, contact_force_threshold=500.0):
        pass

    def reset_root_state_ring(env, env_ids):
        pass

    env_cfg = NS(scene=_Cfg(terrain=NS(physics_material=NS(friction=1.0))),
                 events=_Cfg(reset_root_state=NS(func=reset_root_state_ring, mode="reset"),
                             reset_robot_joints=NS(func=reset_root_state_ring, mode="reset"),
                             push_by_setting_velocity=NS(func=push_by_setting_velocity, mode="interval")),
                 terminations=_Cfg(crab_failure=NS(func=terminate_crab_hex_failure, params={})),
                 commands=NS(base_velocity=NS(class_type=L.FakeCmdBase)))
    rec = lambda **kw: NS(**kw)  # noqa: E731
    sim = NS(CuboidCfg=rec, RigidBodyPropertiesCfg=rec, MassPropertiesCfg=rec, CollisionPropertiesCfg=rec,
             PreviewSurfaceCfg=rec)

    class RigidObjectCfg:
        InitialStateCfg = staticmethod(rec)

        def __init__(self, **kw):
            self.__dict__.update(kw)

    st = ST.StartState(2, "cpu", gen_start=gens()["starts"], gen_hcmd=gens()["hcmd"], bank=bank, stance=stance)
    ch = ST.compose_env_cfg(env_cfg, st, sim_utils=sim, RigidObjectCfg=RigidObjectCfg,
                            EventTermCfg=lambda **kw: NS(**kw))
    assert [c["what"] for c in ch] == ["scene.sh_table", "events.sh_start", "events.push_by_setting_velocity.func",
                                       "terminations.crab_failure.func", "commands.base_velocity.class_type"]
    assert ST.composition_reasons(env_cfg) == []
    assert list(vars(env_cfg.events))[-1] == "sh_start"
    assert env_cfg.scene.sh_table.spawn.rigid_props.kinematic_enabled is True
    assert env_cfg.scene.sh_table.spawn.size == ST.SPEC.TABLE_SIZE_M
    assert issubclass(env_cfg.commands.base_velocity.class_type, L.FakeCmdBase)
    # fail closed: a reset term after the start event, an unwrapped push
    env_cfg.events.late = NS(func=reset_root_state_ring, mode="reset")
    assert any("sh_start must be last" in r for r in ST.composition_reasons(env_cfg))
    del env_cfg.events.late
    env_cfg.events.push_by_setting_velocity.func = push_by_setting_velocity
    assert any("push" in r for r in ST.composition_reasons(env_cfg))


def test_kit_probe_with_stubs(ST):
    n = 3
    feet = [f"{l}_Footpad" for l in ST.LEGS]
    body_pos = torch.zeros(n, 8, 3)
    body_pos[:, 2:8, 2] = torch.tensor([0.05, 0.05, 0.05, 0.05, 0.30, 0.05])     # one foot in the air
    pos = torch.tensor([[0.0, 0, 1.0], [0, 0, 0.8], [0, 0, 1.2]])

    class Robot:
        data = types.SimpleNamespace(root_pos_w=pos, root_quat_w=torch.tensor([[1.0, 0, 0, 0]] * n),
                                     body_pos_w=body_pos)

        def find_bodies(self, names, preserve_order=True):
            return list(range(2, 8)), list(names)

    class Contact:
        data = types.SimpleNamespace(net_forces_w=torch.zeros(n, 10, 3))

        def find_bodies(self, names, preserve_order=True):
            return list(range(4, 10)), list(names)

    contact = Contact()
    contact.data.net_forces_w[:, 4:10, 2] = torch.tensor([100.0, 100, 100, 5, 100, 100])
    table = types.SimpleNamespace(data=types.SimpleNamespace(root_pos_w=torch.tensor([[0, 0, -0.4]] * n)))
    scene = {"robot": Robot(), "sh_table": table}
    sc = type("Scene", (), {"__getitem__": lambda self, k: scene[k], "sensors": {"contact_forces": contact}})()
    u = types.SimpleNamespace(scene=sc, action_manager=types.SimpleNamespace(action=torch.ones(n, 18)))

    def ray(starts, dirs, mesh=None):
        h = starts.clone()
        h[:, 2] = 0.02
        return (h,)

    p = ST.KitProbe(types.SimpleNamespace(unwrapped=u), mesh="m", raycast_fn=ray)
    m = p()
    assert torch.allclose(m["h_meas"], pos[:, 2].double() - 0.02)
    assert m["feet_loaded"].tolist() == [4, 4, 4]            # one foot unloaded (5 N), one too high above the floor
    assert torch.allclose(m["table_top_w"], torch.full((n,), 0.6, dtype=torch.float64))
    assert torch.allclose(m["lowest_z"], pos[:, 2].double() - 0.15875)
    with pytest.raises(RuntimeError, match="mesh"):
        ST.KitProbe(types.SimpleNamespace(unwrapped=u), mesh=None, raycast_fn=ray)


# ===================================================================================== sh_dagger.py (pre-Kit)
def test_emit_env_composes_an_accepted_env(SPEC):
    dirty = {k: v for k, v in os.environ.items() if not k.startswith(("KRABBY_", "P2R_", "SN_STAND_"))}
    dirty.update(KRABBY_FOO="1", KRABBY_OMNI_PHASE="2c", P2R_MODE="st2", SN_STAND_CKPT="/x", CUDA_VISIBLE_DEVICES="")
    script = f'eval "$({sys.executable} -B {ENTRY} --emit-env)" && env -0'
    r = subprocess.run(["bash", "-c", script], capture_output=True, timeout=120, env=dirty)
    assert r.returncode == 0, r.stderr.decode()
    env = dict(x.split("=", 1) for x in r.stdout.decode().split("\0") if "=" in x)
    assert SPEC.env_reasons(env) == []
    assert "KRABBY_FOO" not in env and "P2R_MODE" not in env and "SN_STAND_CKPT" not in env
    assert env["KRABBY_OMNI_PHASE"] == "2a" and env["KRABBY_OMNI_STAND_OBS"] == "0"


def _good_args(out):
    return ["--mode", "dagger", "--seed", "18", "--iterations", "300", "--out", str(out)]


def test_entry_refusals_fire_before_kit(ENT, SPEC, tmp_path, monkeypatch, capsys):
    called = []
    monkeypatch.setattr(ENT, "run_kit", lambda o, rest: called.append((o, rest)) or 0)
    good_env = SPEC.train_env({"PATH": os.environ.get("PATH", "")})
    out = tmp_path / "run"
    assert ENT.main([str(ENTRY)] + _good_args(out) + ["--headless"], environ=good_env) == 0
    assert len(called) == 1 and called[0][1] == ["--headless"] and called[0][0].arm == "mask"
    called.clear()
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "model_3.pt").write_bytes(b"x")
    cases = [
        (_good_args(out)[2:], good_env, "required"),                                            # no --mode
        (["--mode", "recover_probe", "--seed", "1", "--iterations", "3", "--out", str(out)], good_env, None),
        (_good_args(out) + ["--arm", "full"], good_env, None),
        (_good_args("rel/run"), good_env, "not absolute"),
        (_good_args(tmp_path / "a b"), good_env, "whitespace"),
        (_good_args(out) + ["--smoke", "--smoke-iters", "500"], good_env, "smoke-iters"),
        (_good_args(tmp_path / "busy"), good_env, "already holds"),
        (_good_args(out), dict(good_env, KRABBY_OMNI_STAND_OBS="clock"), "STAND_OBS"),
        (_good_args(out), dict(good_env, P2R_BANK_MAX="5"), "P2R_"),
        (_good_args(out), dict(good_env, KRABBY_RESTANCE="default"), "KRABBY_RESTANCE"),
        (_good_args(out) + ["--iterationsx", "3"], good_env, "malformed"),
    ]
    capsys.readouterr()
    for argv, env, needle in cases:
        if needle is None:                                               # argparse choices: SystemExit(2)
            with pytest.raises(SystemExit) as ei:
                ENT.main([str(ENTRY)] + argv, environ=env)
            assert ei.value.code == 2, argv
            capsys.readouterr()
            continue
        assert ENT.main([str(ENTRY)] + argv, environ=env) == ENT.REFUSED_RC == 2, argv
        err = capsys.readouterr().err
        assert needle in err, (argv, err)
    assert called == []


def test_entry_refusal_exit_code_in_a_process(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KRABBY_", "P2R_", "SN_", "SH_"))}
    env["CUDA_VISIBLE_DEVICES"] = ""
    r = subprocess.run([sys.executable, "-B", str(ENTRY)] + _good_args(tmp_path / "run"), capture_output=True,
                       text=True, timeout=120, env=env)
    assert r.returncode == 2 and "refused before Kit" in r.stderr, (r.returncode, r.stderr)
    assert not (tmp_path / "run").exists()


def test_entry_read_mode(ENT, SPEC, tmp_path, capsys):
    d = tmp_path / "rr"
    d.mkdir()
    rec = {k: None for k in SPEC.RECORD_KEYS}
    rec.update(_rec("walk", "control"))
    (d / "stands.jsonl").write_text(json.dumps(rec) + "\n")
    (d / "train_log.jsonl").write_text(json.dumps({"it": 5, "restore": {}}) + "\n")
    (d / "episodes.jsonl").write_text(json.dumps({"it": 5, "kind": "ground", "run_start": False, "end": "fall",
                                                  "cause": "tilt"}) + "\n")
    assert ENT.main([str(ENTRY), "--read", str(d), "--premise", "0:9"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["premise"]["window"] == [0, 9] and out["n_records"] == 1
    assert out["episodes_by_kind"]["ground"] == {"episodes": 1, "falls": 1, "hip_ends": 0}


def test_entry_provenance_names_every_file(ENT):
    o = types.SimpleNamespace(mode="dagger", arm="mask", seed=18, iterations=3000, smoke=False, num_envs=1024,
                              buffer_gb=4.0)
    p = ENT.provenance(o, "ab" * 32)
    sh = p["sh_dagger"]
    for k in ("sh_dagger", "sh_dagger_core", "sh_dagger_spec", "sh_starts", "sh_core", "sh_kin", "sn_common",
              "sn_dagger_spec_pred", "sn_table_core", "restance_module", "facts_json"):
        assert len(sh[f"{k}_sha256"]) == 64, k
    assert p["sn_dagger"]["stand_mask"] == 1 and type(p["sn_dagger"]["stand_mask"]) is int
    assert sh["core"]["h_min"] == 0.719 and sh["core"]["clip"]["harness_clip_actions"] == 4.8


# ===================================================================================== sh_dagger.sh
def _sh(args, tmp_path, extra_env=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KRABBY_", "P2R_", "SN_", "SH_"))}
    env.update({"SH_OUT": str(tmp_path / "shout"), "SH_DDC": str(tmp_path / "ddc"), "CUDA_VISIBLE_DEVICES": "",
                "SH_PY": sys.executable, **(extra_env or {})})
    (tmp_path / "ddc").mkdir(exist_ok=True)
    return subprocess.run(["bash", str(SH_SCRIPT)] + args, capture_output=True, text=True, timeout=120, env=env)


def test_sh_dry_run_composes_the_command(tmp_path):
    r = _sh(["--dry-run", "SH_SMOKE_s18", "--", "--mode", "dagger", "--seed", "18", "--iterations", "3000",
             "--smoke"], tmp_path)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "systemd-run --user --scope -q --unit=sh-dagger-SH_SMOKE_s18-" in out
    assert "-p MemoryMax=48G -p MemorySwapMax=0" in out
    assert f"MemorySwapMax=0 timeout --signal=TERM 3600 {sys.executable} {ENTRY} " in out
    assert "--kill-after" not in out and " -k " not in out
    assert f"{ENTRY} --headless --mode dagger --seed 18 --iterations 3000 --smoke --out " in out
    assert str(tmp_path / "shout/runs/SH_SMOKE_s18") in out
    assert "export KRABBY_OMNI_STAND_OBS=0" in out and "export OMNI_KIT_ACCEPT_EULA=yes" in out
    assert not (tmp_path / "shout/runs").exists()
    full = _sh(["--dry-run", "SH_s18", "--", "--mode", "dagger", "--seed", "18", "--iterations", "900"], tmp_path)
    assert full.returncode == 0 and "timeout --signal=TERM 14400 " in full.stdout, full.stderr
    for bad in ("0", "1h", "-5"):
        r = _sh(["--dry-run", "SH_s18", "--", "--mode", "dagger"], tmp_path, {"SH_TIMEOUT": bad})
        assert r.returncode == 5 and "SH_TIMEOUT" in r.stderr, (bad, r.stderr)


@pytest.mark.parametrize("args, needle", [
    (["bad-name", "--", "--mode", "dagger"], "NAME"),
    (["N1", "--mode", "dagger"], "expected '--'"),
    (["N1", "--", "--out", "/x"], "not allowed"),
    (["N1", "--", "--read", "/x"], "not allowed"),
])
def test_sh_refusals(tmp_path, args, needle):
    r = _sh(["--dry-run"] + args, tmp_path)
    assert r.returncode == 5 and needle in r.stderr, r.stderr


def test_sh_refuses_a_non_empty_run_dir_and_stop_queue(tmp_path):
    (tmp_path / "shout/runs/N2").mkdir(parents=True)
    (tmp_path / "shout/runs/N2/model_1.pt").write_bytes(b"x")
    r = _sh(["--dry-run", "N2", "--", "--mode", "dagger"], tmp_path)
    assert r.returncode == 5 and "not empty" in r.stderr
    (tmp_path / "shout/STOP_QUEUE").write_text("")
    r = _sh(["--dry-run", "N3", "--", "--mode", "dagger"], tmp_path)
    assert r.returncode == 4


def _var(src: str, name: str) -> str:
    return src.split(f"{name}='", 1)[1].split("'", 1)[0]


def _busy(pat: str, nonkit: str, lines: list[str]) -> bool:
    r = subprocess.run(["bash", "-c", 'printf "%s\\n" "$@" | grep -E "$P" | grep -qvE "$N"', "_", *lines],
                       env={"P": pat, "N": nonkit, "PATH": os.environ.get("PATH", "")}, text=True)
    return r.returncode == 0


def test_sh_busy_pattern():
    src = SH_SCRIPT.read_text()
    pat, nonkit = _var(src, "ISAAC_PAT"), _var(src, "NONKIT_PAT")
    assert 'busy() { ps -ww -eo args | grep -E "$ISAAC_PAT" | grep -qvE "$NONKIT_PAT"; }' in src
    py = "/home/nickmagus/krabby/isaac_venv/bin/python"
    yes = [f"{py} {ENTRY} --headless --mode dagger --out /r/N",
           f"{py} {PRED}/sn_dagger.py --headless --mode dagger --out /r/N",
           f"{py} /x/crab_hex_forward_task/scripts/eval_crab_hex_gait.py --headless",
           f"{py} {SCRIPTS}/sh_stand_net_eval_crab_hex_gait.py --headless",
           f"{py} {PRED}/sn_table_test.py --controller r --headless",
           f"{py} {SCRIPTS}/sh_table_test.py --controller r --headless",
           f"{py} {PRED}/sn_record_video.py --headless",
           f"{py} /home/x/parkour/scripts/rsl_rl/train.py --task T",
           f"{py} /x/tools/run_candidates.py --campaign-dir c"]
    no = [f"{py} -B {ENTRY} --emit-env", f"{py} {ENTRY} --read /r/N --premise 0:9", f"{py} {ENTRY} --read=/r/N",
          f"{py} {PRED}/sn_dagger.py --emit-env", f"grep -qE {pat}", f"bash {SH_SCRIPT} N -- --mode dagger",
          f"systemd-run --user --scope -q {py} {ENTRY}", f"timeout --signal=TERM 3600 {py} {ENTRY} --headless"]
    for line in yes:
        assert _busy(pat, nonkit, [line]), line
    for line in no:
        assert not _busy(pat, nonkit, [line]), line
    assert _busy(pat, nonkit, [no[1], yes[0]])


def _isaac_running() -> bool:
    src = SH_SCRIPT.read_text()
    out = subprocess.run(["ps", "-ww", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()
    return _busy(_var(src, "ISAAC_PAT"), _var(src, "NONKIT_PAT"), out) if out else False


FAKE_SYSTEMD_RUN = textwrap.dedent('''\
    #!/bin/bash
    # a fake systemd-run for the detach test: skip its options and the timeout prefix, then show the trainer's
    # command line under a harmless short sleep (never runs the trainer, never starts Kit)
    while [ $# -gt 0 ]; do case "$1" in -p) shift 2 ;; -*) shift ;; *) break ;; esac; done
    [ "$1" = timeout ] && shift 3
    echo "$*" >> "$FAKE_SDR_LOG"
    [ "${FAKE_SDR_FAIL:-0}" = 1 ] && exit 9
    exec -a "$*" sleep "${FAKE_SDR_SLEEP:-3}"
    ''')


@pytest.mark.skipif(sys.platform != "linux", reason="ps / setsid / exec -a")
def test_sh_detach_returns_only_once_the_trainer_is_visible(tmp_path):
    if _isaac_running():
        pytest.skip("an Isaac process is running on this machine (sh_dagger.sh would refuse: busy)")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    sdr = bindir / "systemd-run"
    sdr.write_text(FAKE_SYSTEMD_RUN)
    sdr.chmod(0o755)
    venv = tmp_path / "isaac_venv/bin"
    venv.mkdir(parents=True)
    py = venv / "python"
    py.write_text(f'#!/bin/bash\nexec {sys.executable} "$@"\n')
    py.chmod(0o755)
    log = tmp_path / "sdr.log"
    extra = {"PATH": f"{bindir}:{os.environ.get('PATH', '')}", "SH_PY": str(py), "FAKE_SDR_LOG": str(log),
             "SH_UP_WAIT": "20", "FAKE_SDR_SLEEP": "2"}
    r = _sh(["--detach", "SH_DET", "--", "--mode", "dagger", "--seed", "1", "--iterations", "3"], tmp_path, extra)
    assert r.returncode == 0 and "trainer visible after" in r.stdout, (r.returncode, r.stdout, r.stderr)
    shown = log.read_text()
    assert f"{py} {ENTRY} --headless --mode dagger" in shown and f"--out {tmp_path / 'shout/runs/SH_DET'}" in shown
    for _ in range(100):
        if not _isaac_running():
            break
        time.sleep(0.1)
    assert not _isaac_running()
    r2 = _sh(["--detach", "SH_DET2", "--", "--mode", "dagger", "--seed", "1", "--iterations", "3"], tmp_path,
             {**extra, "FAKE_SDR_FAIL": "1"})
    assert r2.returncode == 7 and "ended before it became visible" in r2.stderr and "rc=9" in r2.stderr, r2.stderr


# ===================================================================================== the residual deadband (entry)
def test_entry_deadband_option_records_and_read(ENT, SPEC, tmp_path, monkeypatch, capsys):
    """CHARTER 2026-10-14: ``--deadband D`` (default 0) reaches RunOpts and every run record; bad values are refused
    before Kit; ``--read`` reports the run's recorded deadband (a run.json without it -- SH_SMOKE / SH_P18 -- is 0)."""
    called = []
    monkeypatch.setattr(ENT, "run_kit", lambda o, rest: called.append((o, rest)) or 0)
    good_env = SPEC.train_env({"PATH": os.environ.get("PATH", "")})
    out = tmp_path / "run"
    assert ENT.main([str(ENTRY)] + _good_args(out) + ["--deadband", "0.25", "--headless"], environ=good_env) == 0
    assert called[-1][0].deadband == 0.25 and called[-1][1] == ["--headless"]
    assert ENT.main([str(ENTRY)] + _good_args(out) + ["--deadband=0.25"], environ=good_env) == 0
    assert called[-1][0].deadband == 0.25
    assert ENT.main([str(ENTRY)] + _good_args(out), environ=good_env) == 0
    assert called[-1][0].deadband == 0.0 and type(called[-1][0].deadband) is float
    n = len(called)
    capsys.readouterr()
    for bad in ("-0.1", "1", "1.5", "nan", "inf"):
        assert ENT.main([str(ENTRY)] + _good_args(out) + ["--deadband", bad], environ=good_env) == ENT.REFUSED_RC
        assert "--deadband" in capsys.readouterr().err, bad
    assert ENT.main([str(ENTRY)] + _good_args(out) + ["--deadbandx", "0.2"], environ=good_env) == ENT.REFUSED_RC
    assert len(called) == n
    o = types.SimpleNamespace(mode="dagger", arm="mask", seed=18, iterations=5000, smoke=False, num_envs=1024,
                              buffer_gb=4.0, deadband=0.25)
    p = ENT.provenance(o, "ab" * 32)["sh_dagger"]
    assert p["residual_deadband"] == 0.25 and p["core"]["clip"]["residual_deadband"] == 0.25
    assert ENT.SH.recorded_deadband({**p, "clip": ENT.SH.clip_record(0.25)}) == 0.25
    o.deadband = 0.0
    assert ENT.provenance(o, "ab" * 32)["sh_dagger"]["residual_deadband"] == 0.0
    # --read: the run's recorded deadband (no run.json / no key -> 0)
    d = tmp_path / "rr"
    d.mkdir()
    (d / "train_log.jsonl").write_text(json.dumps({"it": 5, "restore": {}}) + "\n")
    (d / "episodes.jsonl").write_text("")
    assert ENT.read_run(str(d), None)["residual_deadband"] == 0.0
    (d / "run.json").write_text(json.dumps({"opts": {"seed": 18}, "sh_dagger": {"clip": {"harness_clip_actions": 4.8}}}))
    assert ENT.read_run(str(d), None)["residual_deadband"] == 0.0                       # a run before the option
    (d / "run.json").write_text(json.dumps({"opts": {"deadband": 0.25}, "sh_dagger": {
        "residual_deadband": 0.25, "clip": ENT.SH.clip_record(0.25)}}))
    assert ENT.read_run(str(d), None)["residual_deadband"] == 0.25
    (d / "run.json").write_text(json.dumps({"opts": {"deadband": 0.0}, "sh_dagger": {"residual_deadband": 0.25}}))
    with pytest.raises(ValueError):
        ENT.read_run(str(d), None)


def test_sh_dry_run_passes_the_deadband_through(tmp_path):
    r = _sh(["--dry-run", "SH_DB_s18", "--", "--mode", "dagger", "--seed", "18", "--iterations", "5000",
             "--deadband", "0.25"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert f"{ENTRY} --headless --mode dagger --seed 18 --iterations 5000 --deadband 0.25 --out " in r.stdout
