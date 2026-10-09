"""Stage p2s (P2S_PLAN.md steps 3 and 6 / CHARTER 11): the ``scripts/p2_live.py`` additions. CPU only: nothing here
starts Isaac/Kit or a GPU.

* the termination guards REPLAYED on every s18 sibling pair from the REAL TensorBoard logs -- p2a X06 / X08 / X09 and
  p2b X12 / X13 / X14, both orders (12 pairs), polled every 25 iterations from +300 to the end exactly as run_arm.sh
  polls them: ZERO kills (the episode-length clause, and the sigma clause beside it); the notify-only crab_failure flag
  is allowed to fire (sibling noise reaches +0.116) but never as a kill;
* the liveness at +200 (both terms non-zero, the slide term within 3x of its SAMPLED replay magnitude, the support term's
  implied sum s_f in [2.5, 6] at the pinned stand share 0.15), the generalised stand-only AST check on the real
  motion-rewards module, the sigma / termination guards, the preflight (replay PASS or a user overrule, the solved
  weights, the reference) and the reference resolution on the real arms records.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CAMP = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-07_omni_phase2"
SCRIPTS = CAMP / "scripts"
RUNS = REPO / "parkour/logs/rsl_rl/crab_hex_omni_teacher"
ARMS = REPO / "parkour/outputs/omni_plan/phase2/arms"
MOTION_KNOBS = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/config/crab_hex/motion_knobs.py"
P2A, P2B = ("X06", "X08", "X09"), ("X12", "X13", "X14")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def live():
    return _load(SCRIPTS / "p2_live.py", "_t_p2s_live")


def test_constants_match_the_knob_module(live):
    mk = _load(MOTION_KNOBS, "_t_p2s_live_mk")
    assert (live.P2S_SUPPORT, live.P2S_SLIDE) == (mk.STAND_SUPPORT_TERM, mk.STAND_SLIDE_TERM)
    assert live.P2S_WEIGHT_KNOBS == {mk.STAND_SUPPORT_TERM: mk.STAND_SUPPORT_W_VAR,
                                     mk.STAND_SLIDE_TERM: mk.STAND_SLIDE_W_VAR}
    assert live.P2S_FUNC_MOD == mk._REWARDS_MOD
    assert (REPO / live.MOTION_REWARDS_PY).is_file()
    assert live.P2S_STAND_SHARE == 0.15 and live.P2S_SUPPORT_SUM_BAND == (2.5, 6.0)
    assert (live.P2S_LEN_FRAC, live.P2S_STD_FRAC, live.P2S_CF_MARGIN, live.P2S_CF_AT) == (0.9, 0.8, 0.08, 500)
    assert live.RC_NOTIFY not in (live.RC_PASS, live.RC_WAIT, live.RC_FAIL, live.RC_REFUSE)


def test_self_test_passes(live, capsys):
    assert live._self_test() == 0
    out = capsys.readouterr().out
    assert "[FAIL]" not in out and "p2s guards: crab_failure +0.09 over C0's from +500 is NOTIFY-ONLY" in out


# ================================================================================== the sibling-pair replay (real logs)
def _run_dir(x: str) -> Path | None:
    hits = sorted(RUNS.glob(f"*_seed18_{x}_*"))
    return hits[-1] if hits else None


@pytest.fixture(scope="module")
def sibling_series(live):
    out = {}
    for x in P2A + P2B:
        rd = _run_dir(x)
        if rd is None:
            pytest.skip(f"run {x} not in this checkout (logs are untracked)")
        ser = live.read_tb(rd)
        out[x] = {t: ser[t] for t in live.P2S_GUARD_TAGS}
    return out


def _replay(live, arm: dict, ref: dict, step: int = 25) -> dict:
    """Poll the guards on ``arm`` against ``ref`` (both full runs) every ``step`` iterations from +300 to the end, the
    arm's series cut at each poll exactly as a live run's TensorBoard would read it."""
    import bisect  # noqa: PLC0415

    el = live.TAGS["ep_len"]
    first = arm[el][0][0]
    rref = {"kind": "run", "name": "sibling", "first": ref[el][0][0], "ser": ref}
    steps = {t: [s for s, _ in v] for t, v in arm.items()}
    rcs, ratios, worst_cf = [], [], -1.0
    for last in range(first + live.P2S_LEN_AT, arm[el][-1][0] + 1, step):
        cut = {t: v[:bisect.bisect_right(steps[t], last)] for t, v in arm.items()}
        res = live.judge_p2s_guards(cut, first, rref)
        rcs.append(res["rc"])
        if res.get("ep_len") and res.get("ref_ep_len"):
            ratios.append(res["ep_len"] / res["ref_ep_len"])
        if res.get("crab_failure") is not None and res.get("ref_crab_failure") is not None:
            worst_cf = max(worst_cf, res["crab_failure"] - res["ref_crab_failure"])
    return {"rcs": rcs, "min_len_ratio": min(ratios), "worst_cf": worst_cf}


PAIRS = [p for grp in (P2A, P2B) for p in itertools.permutations(grp, 2)]


@pytest.fixture(scope="module")
def replays(live, sibling_series):
    return {(a, b): _replay(live, sibling_series[a], sibling_series[b]) for a, b in PAIRS}


@pytest.mark.parametrize("a,b", PAIRS)
def test_the_termination_guards_never_kill_a_sibling_pair(live, replays, a, b):
    """P2S_PLAN step 6: 'A unit test replays both clauses on every s18 sibling pair (X06 / X08 / X09 and X12 / X13 /
    X14, both orders): zero kills.' The kill clause (episode length) and the sigma clause never fire; the notify-only
    crab_failure clause may (rc 5), and is never a kill."""
    r = replays[(a, b)]
    assert live.RC_FAIL not in r["rcs"], (a, b, r["min_len_ratio"])
    assert r["min_len_ratio"] >= live.P2S_LEN_FRAC
    assert set(r["rcs"]) <= {live.RC_PASS, live.RC_NOTIFY}


def test_the_sibling_noise_the_plan_describes(live, replays):
    """The replay reproduces the plan's premises: the minimum episode-length ratio stays above 0.9 over all 12 pairs
    (~0.92 at 25-it polls), and the crab_failure clause passes +0.08 on some pairs (why it is notify-only: sibling noise
    reaches ~+0.116)."""
    assert len(replays) == 12
    assert 0.9 < min(r["min_len_ratio"] for r in replays.values()) < 0.95
    assert any(live.RC_NOTIFY in r["rcs"] for r in replays.values())
    assert max(r["worst_cf"] for r in replays.values()) > live.P2S_CF_MARGIN


def test_the_s16_s17_constant_references_from_the_real_records(live):
    if not (ARMS / "P2__S1_s17.record.json").is_file():
        pytest.skip("phase-2 arms records not in this checkout")
    for seed in (16, 17):
        ref = live.load_p2s_reference(live.p2s_reference(ARMS, seed))
        ok, why = live.p2s_reference_ok(ref)
        assert ok and ref["kind"] == "tail" and ref["name"] == f"P2__S1_s{seed}", why
        assert set(ref["tail"]) == set(live.P2S_GUARD_TAGS)
    s17 = live.load_p2s_reference(live.p2s_reference(ARMS, 17))
    assert 0.25 < s17["tail"][live.TAGS["crab_failure"]] < 0.33         # the plan's "s17 crab_failure tail 0.29"
    s18 = live.load_p2s_reference(live.p2s_reference(ARMS, 18))
    assert s18["kind"] == "run" and s18["name"] == "P2__C0_s18" and "_X12_" in s18["run_dir"]
    assert live.p2s_reference(ARMS, None).get("error")


def test_reference_resolution_fails_closed(live, tmp_path):
    assert live.p2s_reference(tmp_path, 18)["error"].startswith("no readable P2__C0_s18.record.json")
    (tmp_path / "P2__C0_s18.record.json").write_text(json.dumps({"run_dir": str(tmp_path), "killed": "x"}))
    assert "was killed" in live.p2s_reference(tmp_path, 18)["error"]
    (tmp_path / "P2__S1_s16.record.json").write_text(json.dumps({"run_dir": str(tmp_path / "nope")}))
    assert "is not a directory" in live.p2s_reference(tmp_path, 16)["error"]
    (tmp_path / "P2__S1_s17.record.json").write_text(json.dumps({"run_dir": str(tmp_path)}))
    ref = live.load_p2s_reference(live.p2s_reference(tmp_path, 17), reader=lambda rd: {})
    assert not live.p2s_reference_ok(ref)[0] and "lacks" in ref["error"]


# ================================================================================== liveness / sigma / termination units
F0 = 19994
BASE = {"Metrics/clock_hidden_frac": 1.0, "Metrics/base_parkour/reach_obst_frac": 0.3,
        "Episode_Reward/reward_action_rate": -0.004}
OV = {"KRABBY_FLAT_TERRAIN_GEOM": "field", "KRABBY_ACTION_RATE_W": "-0.05", "KRABBY_OMNI_SLOTS": "0.15:0.55:0.1:0.2",
      "KRABBY_STAND_SUPPORT_W": "1.12", "KRABBY_STAND_SLIDE_W": "-1.25"}
REPLAY = {"p2s": {"verdict": "PASS", "ST2": {
    "weights": {"KRABBY_STAND_SUPPORT_W": "1.12", "KRABBY_STAND_SLIDE_W": "-1.25"},
    "terms": {"reward_stand_support": {"episode_reward": 0.05, "episode_reward_sampled": 0.02},
              "penalty_stand_slide": {"episode_reward": -0.06, "episode_reward_sampled": -0.04}}}}}


def _facts(live, **over):
    f = {"have_env": True, "curriculum_null": True,
         "weights": {"reward_clock_schedule": 0.001, "reward_clock_swing_apex": 0.001, "reward_action_rate": -0.05,
                     **live.ELEMENT_WEIGHTS},
         "term_funcs": {t: f"{live.P2S_FUNC_MOD}:{t}" for t in live.P2S_TERMS},
         "dr_events": {e: True for e in live.DR_EVENTS}, "push_x": [-0.5, 0.5], "terrain_fns": ["omni_field_terrain"],
         "num_rows": 10, "dropped_ppo_keys": []}
    f.update(over)
    return f


def _ser(sup, sli, n=215):
    d = dict(BASE, **{"Episode_Reward/reward_stand_support": sup, "Episode_Reward/penalty_stand_slide": sli})
    return {t: [(F0 + i, v) for i in range(n)] for t, v in d.items()}


def _wire(live, sup, sli, ov=OV, replay=REPLAY, gate=None, facts=None):
    gate = gate if gate is not None else {t: (True, "gated") for t in live.P2S_TERMS}
    return live.judge_wiring("ST2", ov, facts or _facts(live), _ser(sup, sli), F0, [("overlay x", True)], replay,
                             p2s_gate=gate)


@pytest.mark.parametrize("sup,sli,rc", [
    (0.168, -0.05, 0),        # sum s_f 4.0; slide 1.25x the sampled replay
    (0.0, -0.05, 3),          # a dead T1 (sum s_f would read exactly 3, inside the band): the non-zero rule fails it
    (0.168, 0.0, 3),          # a dead T2
    (0.168, 0.01, 3),         # T2 positive
    (-0.1, -0.05, 3),         # sum s_f 2.40 < 2.5
    (0.6, -0.05, 3),          # sum s_f 6.57 > 6
    (0.168, -0.12, 0),        # T2 at exactly 3x
    (0.168, -0.13, 3),        # T2 beyond 3x
    (0.168, -0.0133, 3),      # T2 under 1/3x
])
def test_p2s_liveness_cases(live, sup, sli, rc):
    assert _wire(live, sup, sli)["rc"] == rc


def test_p2s_liveness_reports_the_implied_sum_and_the_support_ratio(live):
    r = _wire(live, 0.168, -0.05)
    assert r["liveness"]["implied_sum_s_f"] == pytest.approx(4.0)
    assert r["liveness"]["stand_share"] == 0.15
    assert r["liveness"]["terms"]["reward_stand_support"]["ratio_reported"] == pytest.approx(8.4)   # not gated
    assert r["liveness"]["terms"]["penalty_stand_slide"]["ratio"] == pytest.approx(1.25)


def test_p2s_liveness_fails_closed_without_the_sampled_magnitudes(live):
    det_only = {"p2s": {"verdict": "PASS", "ST2": {"terms": {
        "reward_stand_support": {"episode_reward": 0.05}, "penalty_stand_slide": {"episode_reward": -0.06}}}}}
    r = _wire(live, 0.168, -0.05, replay=det_only)
    assert r["rc"] == live.RC_FAIL and any("sampled-probe" in f for f in r["fails"])
    assert _wire(live, 0.168, -0.05, replay=None)["rc"] == live.RC_FAIL
    assert live.p2s_replay_magnitude(det_only, "ST2", "reward_stand_support") is None


def test_p2s_liveness_pins_the_stand_share_from_the_chain_env(live):
    r = _wire(live, 0.168, -0.05, ov=dict(OV, KRABBY_OMNI_SLOTS="0.2:0.5:0.1:0.2"))
    assert r["rc"] == live.RC_FAIL and any("pinned 0.15" in f for f in r["fails"])
    assert _wire(live, 0.168, -0.05, ov={k: v for k, v in OV.items() if k != "KRABBY_OMNI_SLOTS"})["rc"] == live.RC_FAIL


def test_p2s_construction_needs_the_dumped_funcs_and_the_gate(live):
    assert _wire(live, 0.168, -0.05, gate={})["rc"] == live.RC_FAIL
    bad = _facts(live, term_funcs={"reward_stand_support": "parkour_isaaclab.envs.mdp.rewards:reward_stand_support",
                                   "penalty_stand_slide": f"{live.P2S_FUNC_MOD}:penalty_stand_slide"})
    assert _wire(live, 0.168, -0.05, facts=bad)["rc"] == live.RC_FAIL


def test_dump_facts_reads_the_p2s_funcs(live):
    env = {"rewards": {"reward_stand_support": {"func": f"{live.P2S_FUNC_MOD}:reward_stand_support", "weight": 1.12}},
           "curriculum": None, "events": {}, "scene": {}}
    f = live.dump_facts(env, None, "")
    assert f["term_funcs"]["reward_stand_support"] == f"{live.P2S_FUNC_MOD}:reward_stand_support"
    assert f["term_funcs"]["penalty_stand_slide"] is None


def test_the_generalised_stand_only_check_on_the_real_modules(live, tmp_path):
    """Both p2s funcs gate on stand_mask() through the module helper _stand_gate; F1's check is unchanged; an
    ungated function and a gate that is not a conjunct both read False."""
    g = live.p2s_gate_structural(REPO / live.MOTION_REWARDS_PY)
    assert all(ok for ok, _ in g.values()), g
    assert all("_stand_gate" in why for _ok, why in g.values())
    ok, why = live.stand_gate_structural(REPO / live.REWARDS_PY)
    assert ok and "reward_stand_stance_quality" in why
    assert not live.stand_gate_structural(REPO / live.MOTION_REWARDS_PY, "penalty_loaded_foot_slip")[0]
    src = tmp_path / "m.py"
    src.write_text("def _g(env):\n    stand = env.c.stand_mask()\n    return stand | env.other\n\n"
                   "def reward_stand_support(env):\n    gate = _g(env)\n    return env.x * gate\n\n"
                   "def _h(env):\n    stand = env.c.stand_mask()\n    return (stand & env.up).to(float)\n\n"
                   "def penalty_stand_slide(env):\n    gate = _h(env)\n    if env.y:\n        return env.z\n"
                   "    return env.x * gate\n")
    g = live.p2s_gate_structural(src)
    assert not g["reward_stand_support"][0]                  # an OR is not a gate
    assert not g["penalty_stand_slide"][0]                   # one return escapes the gate
    assert not live.stand_gate_structural(tmp_path / "missing.py", "x")[0]


def _gser(n, el=1800.0, cf=0.2, sd=0.82):
    return {t: [(F0 + i, v) for i in range(n)] for t, v in (("Train/mean_episode_length", el),
                                                            ("Episode_Termination/crab_failure", cf),
                                                            ("PPO/effective_std", sd))}


def _rref(el=1800.0, cf=0.2, sd=0.82, first=F0):
    s = _gser(5000, el, cf, sd)
    if first != F0:
        s = {t: [(first + (k - F0), v) for k, v in vs] for t, vs in s.items()}
    return {"kind": "run", "name": "P2__C0_s18", "first": first, "ser": s}


@pytest.mark.parametrize("n,kw,rc", [
    (250, {"el": 900.0}, 0),                          # before +300: n/a
    (400, {"el": 1621.0}, 0),
    (400, {"el": 1619.0}, 3),                         # episode length < 0.9 x C0's -> KILL
    (999, {"sd": 0.5}, 0),                            # sigma not read before +1000
    (1100, {"sd": 0.657}, 0),                         # 0.801 x
    (1100, {"sd": 0.655}, 3),                         # 0.799 x -> KILL
    (499, {"cf": 0.5}, 0),                            # the flag is armed from +500 only
    (600, {"cf": 0.279}, 0),                          # +0.079
    (600, {"cf": 0.281}, 5),                          # +0.081 -> NOTIFY-ONLY
    (600, {"cf": 0.9, "el": 1000.0}, 3),              # a kill outranks the flag
])
def test_p2s_guard_cases(live, n, kw, rc):
    assert live.judge_p2s_guards(_gser(n, **kw), F0, _rref())["rc"] == rc


def test_p2s_guards_use_the_reference_s_own_fork_relative_window(live):
    """The reference's window is taken at the SAME offset from ITS first logged iteration, not the same absolute
    step: a reference logging from 14995 is read 300 its into its own run."""
    ref = _rref(first=14995)
    ref["ser"]["Train/mean_episode_length"] = [(14995 + i, 1800.0 if i < 1000 else 900.0) for i in range(5000)]
    assert live.judge_p2s_guards(_gser(400, el=1700.0), F0, ref)["rc"] == live.RC_PASS
    assert live.judge_p2s_guards(_gser(1300, el=1700.0), F0, ref)["rc"] == live.RC_PASS      # ref drops later too
    short = _rref()
    short["ser"] = {t: v[:300] for t, v in short["ser"].items()}
    assert live.judge_p2s_guards(_gser(600), F0, short)["rc"] == live.RC_WAIT                 # no window: wait


def test_p2s_guards_constant_tail(live):
    tail = {"kind": "tail", "name": "P2__S1_s17",
            "tail": {"Train/mean_episode_length": 1726.0, "Episode_Termination/crab_failure": 0.29,
                     "PPO/effective_std": 0.808}}
    assert live.judge_p2s_guards(_gser(1200, el=1700.0, cf=0.31, sd=0.80), F0, tail)["rc"] == live.RC_PASS
    # the tail reference kills only below 0.85 (fable review 2026-10-08): 1550 / 1726 = 0.898 passes, 1400 = 0.811 kills
    assert live.judge_p2s_guards(_gser(1200, el=1550.0, cf=0.31, sd=0.80), F0, tail)["rc"] == live.RC_PASS
    assert live.judge_p2s_guards(_gser(1200, el=1400.0, cf=0.31, sd=0.80), F0, tail)["rc"] == live.RC_FAIL
    assert live.judge_p2s_guards(_gser(1200, el=1700.0, cf=0.38, sd=0.80), F0, tail)["rc"] == live.RC_NOTIFY


# ================================================================================== preflight
EM = {"terrain_levels_obst": ["x.py"], "promote_count": ["x.py"]}
GATE = {"reward_stand_support": (True, "g"), "penalty_stand_slide": (True, "g")}


@pytest.mark.parametrize("replay,ov,ref,rc", [
    (REPLAY, OV, (True, "ok"), 0),
    (None, OV, (True, "ok"), 3),
    ({"p2s": dict(REPLAY["p2s"], verdict="FAIL")}, OV, (True, "ok"), 3),
    ({"p2s": dict(REPLAY["p2s"], verdict="FAIL", overrule={"by": "user", "date": "2026-10-09"})}, OV, (True, "ok"), 0),
    ({"p2s": {"verdict": "PASS"}}, OV, (True, "ok"), 3),                                   # no magnitudes / weights
    (REPLAY, dict(OV, KRABBY_STAND_SLIDE_W="-1.0"), (True, "ok"), 3),                     # not the solved dose
    (REPLAY, dict(OV, KRABBY_OMNI_SLOTS="0.1:0.6:0.1:0.2"), (True, "ok"), 3),
    (REPLAY, OV, (False, "no record"), 3),
])
def test_p2s_preflight(live, replay, ov, ref, rc):
    assert live.judge_preflight("ST2", EM, replay, None, ov, GATE, ref)["rc"] == rc


def test_p2s_preflight_needs_the_construction(live):
    assert live.judge_preflight("ST2", EM, REPLAY, None, OV, {}, (True, "ok"))["rc"] == live.RC_FAIL
    assert live.judge_preflight("ST1", EM, REPLAY, None, OV, GATE, (True, "ok"))["rc"] == live.RC_FAIL   # no ST1 entry


def test_non_p2s_arms_are_untouched(live):
    assert live.judge_preflight("C0", EM, None, None)["rc"] == live.RC_PASS
    assert live.p2s_arm("P2__C0_s18") == "" and live.arm_of("P2__ST2_s18") == "ST2"
    assert live.p2s_arm("P2__ST2R1_s16") == "ST2" and live.arm_of("P2__ST2R1_s16") == ""


def _real_p2s():
    return json.loads((CAMP / "replay_gate.json").read_text()).get("p2s")


def test_the_real_replay_record_is_read(live):
    """scripts/p2s_replay.py's own record (replay_gate.json "p2s", when written): its episode_reward is declared
    sampled-calibrated (episode_reward_units) and its solved dose sits under "knobs"; both are what the reads use."""
    p2s = _real_p2s()
    if not p2s:
        pytest.skip("the p2s replay gate has not been written yet")
    rg = {"p2s": p2s}
    assert "SAMPLED" in p2s["episode_reward_units"].upper()
    for dose in ("ST1", "ST2"):
        for t in live.P2S_TERMS:
            v = live.p2s_replay_magnitude(rg, dose, t)
            assert v == p2s[dose]["terms"][t]["episode_reward"] and v != p2s[dose]["terms"][t]["episode_reward_det_ref"]
        assert set(live.p2s_replay_weights(rg, dose)) == set(live.P2S_WEIGHT_KNOBS.values())
    # the implied-sum relation p2_live uses reproduces the replay's own implied_sum_s_f at the sampled magnitude
    w = live.p2s_replay_weights(rg, "ST2")["KRABBY_STAND_SUPPORT_W"]
    er = live.p2s_replay_magnitude(rg, "ST2", "reward_stand_support")
    assert live.P2S_TRIPOD + er / (w * live.P2S_STAND_SHARE) == pytest.approx(p2s["ST2"]["implied_sum_s_f"], abs=1e-6)


def _preflight_cli(live, capsys, monkeypatch, cands: Path):
    monkeypatch.setattr(sys, "argv", ["p2_live.py", "--preflight", "--chain", "P2__ST2_s18", "--code-root", str(REPO),
                                      "--arms", str(ARMS), "--candidates", str(cands)])
    rc = live.main()
    return rc, json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_cli_preflight_on_the_real_tree(live, capsys, monkeypatch, tmp_path):
    """On the real candidates.yaml / replay_gate.json / code root / arms records: candidates v11 carries the replay
    gate's SOLVED doses, so ST2 passes every preflight check (construction, reference, magnitudes, verdict); the same
    chain at the old v10 PLANNING dose (1.12) is refused on the weights alone."""
    if not (ARMS / "P2__C0_s18.record.json").is_file():
        pytest.skip("phase-2 arms records not in this checkout")
    p2s = _real_p2s()
    rc, res = _preflight_cli(live, capsys, monkeypatch, CAMP / "candidates.yaml")
    if not p2s:
        assert rc == live.RC_FAIL and any("p2s replay verdict" in f for f in res["fails"])
        return
    if p2s.get("verdict") == "PASS":
        assert rc == live.RC_PASS, res["fails"]
        assert any("P2__C0_s18 run" in n for n in res["notes"])
    import yaml  # noqa: PLC0415
    d = yaml.safe_load((CAMP / "candidates.yaml").read_text())
    d["chains"]["P2__ST2_s18"]["stages"][0]["overlay"]["KRABBY_STAND_SUPPORT_W"] = "1.12"
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(d, sort_keys=False))
    rc, res = _preflight_cli(live, capsys, monkeypatch, tmp_path / "c.yaml")
    assert rc == live.RC_FAIL and any("KRABBY_STAND_SUPPORT_W" in f for f in res["fails"]), res["fails"]


def test_cli_at_p2s_on_a_non_p2s_chain_is_na(live, capsys, monkeypatch):
    rd = _run_dir("X12")
    if rd is None:
        pytest.skip("run X12 not in this checkout")
    monkeypatch.setattr(sys, "argv", ["p2_live.py", "--check", str(rd), "--chain", "P2__C0_s18", "--at", "p2s",
                                      "--arms", str(ARMS)])
    assert live.main() == live.RC_PASS
    assert "not a p2s arm" in capsys.readouterr().out
