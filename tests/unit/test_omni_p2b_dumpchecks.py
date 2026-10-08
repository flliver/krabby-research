"""Omni phase-2 stage p2b dump checks (P2B_PLAN step 1, 2026-10-08): the ``run_candidates.py``
parsers and ``OVERLAY_DUMP_CHECKS`` the CL (clearance, terrain-relative lift) and TG (tilt guard)
arms need before they can launch.

- CL: ``KRABBY_FOOT_CLEAR_W`` / ``KRABBY_SWING_MIN_CLEAR_W`` (weights), ``KRABBY_FOOT_CLEAR_MIN``
  (the income floor), ``KRABBY_FOOT_CLEAR_FLAT`` (``parkour_name: null`` on both clearance terms) and
  the new ``KRABBY_FOOT_CLEAR_TERRAIN`` (both clearance funcs swapped to the terrain-relative ones);
- TG: ``KRABBY_ORIENT_W`` (weight) and the new ``KRABBY_ORIENT_ALL_TILES`` (``reward_orientation``
  -> ``reward_orientation_upright``).

Every check is run in BOTH directions on synthetic dumps (written like IsaacLab's ``dump_yaml``) and
must FAIL on a dump that lacks its change -- no vacuous pass. On the REAL S1 p2a dump (all clearance
weights 0.0, ``reward_orientation`` stock at -0.7) the OFF values pass and the ON values fail. The
two new flags' parser / whitelist entries come from ``motion_knobs`` (the code half of step 1); their
tests skip until it lands. Pure Python: nothing imports Isaac or launches anything.
"""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
OMNI = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
RUN_CANDIDATES = OMNI / "experiments/tools/run_candidates.py"
#: The S1 s18 p2a head's run (the CL / TG fork source): the real dump the indents are checked against.
S1_DUMP = (REPO / "parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB"
           / "params/env.yaml")

#: Every key p2b overlays (env side).
P2B_KEYS = ("KRABBY_FOOT_CLEAR_W", "KRABBY_SWING_MIN_CLEAR_W", "KRABBY_FOOT_CLEAR_MIN", "KRABBY_FOOT_CLEAR_FLAT",
            "KRABBY_FOOT_CLEAR_TERRAIN", "KRABBY_ORIENT_W", "KRABBY_ORIENT_ALL_TILES")
NEW_FLAGS = ("KRABBY_FOOT_CLEAR_TERRAIN", "KRABBY_ORIENT_ALL_TILES")
#: The plan's arms (2b's CL values before any re-dose; TG at a weight inside the replay band).
CL_OVERLAY = {"KRABBY_FOOT_CLEAR_W": "1.0", "KRABBY_FOOT_CLEAR_MIN": "0.03", "KRABBY_FOOT_CLEAR_FLAT": "1",
              "KRABBY_SWING_MIN_CLEAR_W": "-0.4", "KRABBY_FOOT_CLEAR_TERRAIN": "1"}
TG_OVERLAY = {"KRABBY_ORIENT_ALL_TILES": "1", "KRABBY_ORIENT_W": "-1.5"}
#: (key, the value the S1 p2a dump shows / the knob's OFF spelling, the arm's ON value).
OFF_ON = [("KRABBY_FOOT_CLEAR_W", "0", "1.0"), ("KRABBY_SWING_MIN_CLEAR_W", "0", "-0.4"),
          ("KRABBY_FOOT_CLEAR_MIN", "0.07", "0.03"), ("KRABBY_FOOT_CLEAR_FLAT", "0", "1"),
          ("KRABBY_FOOT_CLEAR_TERRAIN", "0", "1"), ("KRABBY_ORIENT_W", "-0.7", "-1.5"),
          ("KRABBY_ORIENT_ALL_TILES", "0", "1")]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rc():
    return _load(RUN_CANDIDATES, "run_candidates_omni_p2b_test")


def _landed(rc) -> bool:
    """The code half of step 1 (the two flags in ``motion_knobs``) is in the tree."""
    mk = rc.knobs()._mk
    return all(hasattr(mk, a) for a in ("ORIENT_ALL_TILES_VAR", "FOOT_CLEAR_TERRAIN_VAR"))


# ============================================================ a params/env.yaml, written like the trainer
def _entity(body_names=None) -> dict:
    return {"name": "robot", "joint_names": None, "joint_ids": slice(None), "body_names": body_names,
            "body_ids": slice(None), "preserve_order": False}


def _clearance(func: str, **kw) -> dict:
    params = {"asset_cfg": _entity(".*_Footpad"), "sensor_cfg": {**_entity(".*_Footpad"), "name": "contact_forces"},
              "command_name": "base_velocity", "contact_force_threshold": 0.1}
    params.update(kw)
    params.update(min_forward_speed_cmd=0.12, ground_offset_from_root_m=-1.05, parkour_name="base_parkour")
    return {"func": f"parkour_isaaclab.envs.mdp.rewards:{func}", "params": params, "weight": 0.0}


#: The three p2b terms as the S1 p2a dump has them (stock funcs, weights 0.0 / 0.0 / -0.7).
STOCK = {
    "reward_foot_clearance": _clearance("reward_foot_clearance", min_clearance_m=0.07, max_clearance_m=0.26),
    "penalty_swing_min_clearance": _clearance("penalty_swing_min_clearance", min_clearance_m=0.04),
    "reward_orientation": {"func": "parkour_isaaclab.envs.mdp.rewards:reward_orientation",
                           "params": {"asset_cfg": _entity(), "parkour_name": "base_parkour"}, "weight": -0.7},
}


def env_dump(rewards: dict | None = None) -> str:
    """A ``params/env.yaml`` text: the stock terms, each replaced (or dropped, ``None``) by ``rewards``.

    A neighbour term sits on each side so a check that read the nearest ``weight:`` / ``func:``
    instead of its own term's would read a different number.
    """
    rew = {"reward_action_rate": {"func": "parkour_isaaclab.envs.mdp.rewards:reward_action_rate",
                                  "params": {"asset_cfg": _entity()}, "weight": -0.05}}
    rew.update(copy.deepcopy(STOCK))
    rew["track_lin_vel_xy_exp"] = {"func": "parkour_isaaclab.envs.mdp.rewards:track_lin_vel_xy_exp",
                                   "params": {"command_name": "base_velocity", "std": 0.5}, "weight": 1.5}
    for name, term in (rewards or {}).items():
        if term is None:
            rew.pop(name, None)
        else:
            rew[name] = term
    doc = {"seed": 18, "scene": {"num_envs": 256}, "episode_length_s": 40.0, "rewards": rew,
           "curriculum": None, "parkours": {"base_parkour": {"asset_name": "robot"}}}
    return yaml.dump(doc, default_flow_style=False, sort_keys=False)


def _with(term: str, **changes) -> dict:
    """``{term: STOCK[term]}`` with top-level ``func`` / ``weight`` and ``params`` keys changed."""
    out = copy.deepcopy(STOCK[term])
    for key, value in changes.items():
        if key in ("func", "weight"):
            out[key] = value
        else:
            out["params"][key] = value
    return {term: out}


def _swap(term: str, name: str, module: str = "parkour_tasks.crab_hex_omni_task.mdp.omni_clearance") -> dict:
    return _with(term, func=f"{module}:{name}")


#: The CL arm's composed dump (all five CL knobs on) and the TG arm's (both TG knobs on).
CL_DUMP = env_dump({**_with("reward_foot_clearance", weight=1.0, min_clearance_m=0.03, parkour_name=None,
                            func="parkour_tasks.crab_hex_omni_task.mdp.omni_clearance:reward_foot_clearance_terrain"),
                    **_with("penalty_swing_min_clearance", weight=-0.4, parkour_name=None,
                            func="parkour_tasks.crab_hex_omni_task.mdp.omni_clearance:"
                                 "penalty_swing_min_clearance_terrain")})
TG_DUMP = env_dump(_with("reward_orientation", weight=-1.5,
                         func="parkour_isaaclab.envs.mdp.rewards:reward_orientation_upright"))


def _check(rc, key):
    return rc.OVERLAY_DUMP_CHECKS[key]


# ============================================================ registration / parsers
def test_every_p2b_key_has_an_env_side_dump_check(rc):
    for key in P2B_KEYS:
        assert key in rc.OVERLAY_DUMP_CHECKS, key
        assert key not in rc.AGENT_DUMP_KEYS, f"{key} is env side (params/env.yaml)"


def test_the_reused_forward_knobs_are_whitelisted_and_parsed(rc):
    whitelist, parsers = rc.knob_whitelist(), rc.knob_parsers()
    for key in set(P2B_KEYS) - set(NEW_FLAGS):
        assert key in whitelist, key            # named by the forward mdp cfg
        assert key in parsers, key


def test_the_new_flags_come_from_motion_knobs(rc):
    if not _landed(rc):
        pytest.skip("motion_knobs has no ORIENT_ALL_TILES_VAR / FOOT_CLEAR_TERRAIN_VAR yet (code half of step 1)")
    mk = rc.knobs()._mk
    assert mk.ORIENT_ALL_TILES_VAR == rc.ORIENT_ALL_TILES_KEY == "KRABBY_ORIENT_ALL_TILES"
    assert mk.FOOT_CLEAR_TERRAIN_VAR == rc.FOOT_CLEAR_TERRAIN_KEY == "KRABBY_FOOT_CLEAR_TERRAIN"
    whitelist, parsers = rc.knob_whitelist(), rc.knob_parsers()
    for key in NEW_FLAGS:
        assert key in whitelist, key
        assert parsers[key] is mk.MOTION_KNOB_PARSERS[key], key


@pytest.mark.parametrize("key, good, bad", [
    ("KRABBY_FOOT_CLEAR_W", ["1.0", "0", "0.25"], ["", "-0.1", "nan", "x"]),
    ("KRABBY_SWING_MIN_CLEAR_W", ["-0.4", "0", "-1"], ["", "0.4", "inf", "x"]),
    ("KRABBY_ORIENT_W", ["-0.7", "-1.5", "0"], ["", "1.3", "-inf", "x"]),
    ("KRABBY_FOOT_CLEAR_MIN", ["0.03", "0.07", "0"], ["", "-0.01", "nan", "x"]),
    ("KRABBY_FOOT_CLEAR_FLAT", ["1", "0", ""], ["2", "true", "yes"]),
])
def test_p2b_forward_knob_round_trips(rc, key, good, bad):
    whitelist, parsers = rc.knob_whitelist(), rc.knob_parsers()
    for value in good:
        reasons = []
        rc._check_knob(key, value, "x", whitelist, parsers, reasons, set())
        assert reasons == [], (value, reasons)
    for value in bad:
        reasons = []
        rc._check_knob(key, value, "x", whitelist, parsers, reasons, set())
        assert any("does not round-trip" in r for r in reasons), (value, reasons)


def test_p2b_new_flag_round_trips(rc):
    if not _landed(rc):
        pytest.skip("motion_knobs has no ORIENT_ALL_TILES_VAR / FOOT_CLEAR_TERRAIN_VAR yet (code half of step 1)")
    whitelist, parsers = rc.knob_whitelist(), rc.knob_parsers()
    for key in NEW_FLAGS:
        for value in ("1", "0", ""):
            reasons = []
            rc._check_knob(key, value, "x", whitelist, parsers, reasons, set())
            assert reasons == [], (key, value, reasons)
        for value in ("2", "true", "on"):
            reasons = []
            rc._check_knob(key, value, "x", whitelist, parsers, reasons, set())
            assert any("does not round-trip" in r for r in reasons), (key, value, reasons)


# ============================================================ each check, both directions
@pytest.mark.parametrize("key, off, on", OFF_ON)
def test_off_passes_on_stock_and_fails_on_the_arm_dump(rc, key, off, on):
    """The stock dump (S1 p2a layout) passes the OFF value and FAILS the ON value; the arm's composed
    dump passes the ON value and FAILS the OFF value. No check passes both ways."""
    ok = _check(rc, key)
    arm = CL_DUMP if key in CL_OVERLAY else TG_DUMP
    stock = env_dump()
    assert ok(stock, off) and not ok(stock, on), (key, "stock")
    assert ok(arm, on) and not ok(arm, off), (key, "arm")


@pytest.mark.parametrize("key, off, on", OFF_ON)
def test_a_missing_term_fails_every_value(rc, key, off, on):
    """No vacuous pass: with the audited term(s) gone from the dump, neither the off nor the on value
    passes (an absent term is not 'weight 0' or 'stock func')."""
    terms = (["reward_orientation"] if key in TG_OVERLAY
             else ["reward_foot_clearance", "penalty_swing_min_clearance"])
    for term in terms:
        if key in ("KRABBY_FOOT_CLEAR_W", "KRABBY_FOOT_CLEAR_MIN") and term != "reward_foot_clearance":
            continue
        if key == "KRABBY_SWING_MIN_CLEAR_W" and term != "penalty_swing_min_clearance":
            continue
        dump = env_dump({term: None})
        assert not _check(rc, key)(dump, off) and not _check(rc, key)(dump, on), (key, term)


def test_weights_read_their_own_term(rc):
    """The neighbours' weights (-0.05 above, 1.5 below; the two clearance terms each other's) are
    never read for a weight check."""
    dump = env_dump(_with("penalty_swing_min_clearance", weight=-0.4))
    assert _check(rc, "KRABBY_SWING_MIN_CLEAR_W")(dump, "-0.4")
    assert not _check(rc, "KRABBY_FOOT_CLEAR_W")(dump, "-0.4") and _check(rc, "KRABBY_FOOT_CLEAR_W")(dump, "0")
    assert not _check(rc, "KRABBY_ORIENT_W")(env_dump(), "1.5")
    assert not _check(rc, "KRABBY_ORIENT_W")(env_dump(), "-0.05")


def test_foot_clear_min_reads_the_income_term_not_the_swing_floor(rc):
    ok = _check(rc, "KRABBY_FOOT_CLEAR_MIN")
    assert not ok(env_dump(), "0.04"), "0.04 is the swing-min penalty's floor, not the income's"
    moved_swing = env_dump(_with("penalty_swing_min_clearance", min_clearance_m=0.03))
    assert not ok(moved_swing, "0.03") and ok(moved_swing, "0.07")


def test_foot_clear_flat_needs_both_terms_unmasked(rc):
    ok = _check(rc, "KRABBY_FOOT_CLEAR_FLAT")
    for term in ("reward_foot_clearance", "penalty_swing_min_clearance"):
        half = env_dump(_with(term, parkour_name=None))
        assert not ok(half, "1") and not ok(half, "0"), term
    both = env_dump({**_with("reward_foot_clearance", parkour_name=None),
                     **_with("penalty_swing_min_clearance", parkour_name=None)})
    assert ok(both, "1") and not ok(both, "0") and not ok(both, "")
    # the KEY missing is not a lifted mask (nor a stock one)
    gone = copy.deepcopy(STOCK["reward_foot_clearance"])
    del gone["params"]["parkour_name"]
    nokey = env_dump({"reward_foot_clearance": gone,
                      **_with("penalty_swing_min_clearance", parkour_name=None)})
    assert not ok(nokey, "1") and not ok(nokey, "0")


def test_foot_clear_terrain_needs_both_funcs_swapped(rc):
    ok = _check(rc, "KRABBY_FOOT_CLEAR_TERRAIN")
    for term, name in rc.CLEARANCE_TERRAIN_FUNCS.items():
        half = env_dump(_swap(term, name))
        assert not ok(half, "1") and not ok(half, "0"), term
    both = env_dump({**_swap("reward_foot_clearance", "reward_foot_clearance_terrain"),
                     **_swap("penalty_swing_min_clearance", "penalty_swing_min_clearance_terrain")})
    assert ok(both, "1") and not ok(both, "0")
    # crossed funcs (each term carrying the OTHER term's terrain variant) pass neither way
    crossed = env_dump({**_swap("reward_foot_clearance", "penalty_swing_min_clearance_terrain"),
                        **_swap("penalty_swing_min_clearance", "reward_foot_clearance_terrain")})
    assert not ok(crossed, "1") and not ok(crossed, "0")


def test_orient_all_tiles_is_matched_exactly(rc):
    ok = _check(rc, "KRABBY_ORIENT_ALL_TILES")
    other = env_dump(_with("reward_orientation", func="parkour_isaaclab.envs.mdp.rewards:reward_orientation_v2"))
    assert not ok(other, "1") and not ok(other, "0")
    # the swap moves nothing else: the stock params / weight still read
    assert _check(rc, "KRABBY_ORIENT_W")(TG_DUMP, "-1.5")


@pytest.mark.parametrize("key, value", [("KRABBY_FOOT_CLEAR_FLAT", "true"), ("KRABBY_FOOT_CLEAR_TERRAIN", "on"),
                                        ("KRABBY_ORIENT_ALL_TILES", "2"), ("KRABBY_FOOT_CLEAR_MIN", "")])
def test_a_malformed_value_raises_instead_of_picking_a_side(rc, key, value):
    with pytest.raises(ValueError):
        _check(rc, key)(env_dump(), value)


# ============================================================ overlay_checks end to end
def _overlay(rc, tmp_path, text, env):
    run_dir = tmp_path / "run" / "params"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "env.yaml").write_text(text)
    return rc.overlay_checks({"ckpt": str(tmp_path / "run" / "model_24994.pt")}, env)


@pytest.mark.parametrize("overlay, arm_dump", [(CL_OVERLAY, CL_DUMP), (TG_OVERLAY, TG_DUMP)],
                         ids=["CL", "TG"])
def test_overlay_checks_pass_the_arm_and_fail_the_continuation(rc, tmp_path, overlay, arm_dump):
    good = _overlay(rc, tmp_path, arm_dump, overlay)
    assert len(good) == len(overlay) and all(ok for _n, ok in good), good
    bad = _overlay(rc, tmp_path, env_dump(), overlay)
    assert len(bad) == len(overlay) and not any(ok for _n, ok in bad), bad


# ============================================================ the real S1 p2a dump
@pytest.mark.skipif(not S1_DUMP.exists(), reason="the S1 s18 p2a run directory is not on this machine")
def test_the_real_s1_dump_passes_off_and_fails_on(rc):
    """Indent / layout check against what the trainer really wrote (S1 s18 p2a: both clearance terms
    at weight 0.0 with min 0.07 / 0.04 and parkour_name base_parkour; reward_orientation stock at -0.7)."""
    txt = S1_DUMP.read_text()
    for key, off, on in OFF_ON:
        assert _check(rc, key)(txt, off), (key, off)
        assert not _check(rc, key)(txt, on), (key, on)
    for key in ("KRABBY_FOOT_CLEAR_FLAT", "KRABBY_FOOT_CLEAR_TERRAIN", "KRABBY_ORIENT_ALL_TILES"):
        assert _check(rc, key)(txt, ""), key            # blank = off


@pytest.mark.skipif(not S1_DUMP.exists(), reason="the S1 s18 p2a run directory is not on this machine")
def test_the_real_s1_dump_fails_both_arm_overlays(rc, tmp_path):
    for overlay in (CL_OVERLAY, TG_OVERLAY):
        rows = _overlay(rc, tmp_path, S1_DUMP.read_text(), overlay)
        assert len(rows) == len(overlay) and not any(ok for _n, ok in rows), rows
