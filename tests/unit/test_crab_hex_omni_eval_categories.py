"""The omni suite's command categories (``crab_hex_omni_task/eval_categories.py``, plan 2026-09-21 B1).

Pure Python. The categories module, the omni presets, the ladder driver and its ``candidates.yaml``
are loaded BY PATH; nothing here imports ``parkour_tasks`` or Isaac (the module's own purity is
checked in a fresh interpreter). What is pinned:

* which categories a stage's COMPOSED env trains (decisive / present / watch), on normalised shares,
  with the command cfg's class defaults for absent keys (pinned against the class body's source);
* ``expand_rule`` -- the v2 rule restricted to the decisive set in today's flat
  ``blocks`` / ``guards`` / ``watch`` shape, in each unit system and as the union of both;
* ``composite_guards`` -- ``cat_tripod_min`` over the decisive translation categories only.
"""
import ast
import copy
import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
OMNI = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
MODULE = OMNI / "eval_categories.py"
PHASES_PY = OMNI / "config/crab_hex/crab_hex_omni_phases.py"
DRIVER = OMNI / "experiments/tools/run_candidates.py"
LADDER = OMNI / "experiments/2026-09-19_facing_tracking_ladder"
CANDS_YAML = LADDER / "candidates.yaml"
MANIFEST = OMNI / "experiments/eval/scenarios_omni.yaml"
SUITE_ROW = "suite__omni"

ALL = ("stand", "fwd", "side", "back", "diag", "turn", "move_turn")
POOL = "KRABBY_OMNI_CMD_POOL"
SLOTS = "KRABBY_OMNI_SLOTS"
BAND = "KRABBY_OMNI_DIR_BAND"
FACING = "KRABBY_OMNI_FACING"
ENCODING = "KRABBY_OMNI_FACING_OBS_ENCODING"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ec():
    return _load(MODULE, "eval_categories_under_test")


@pytest.fixture(scope="module")
def ph():
    return _load(PHASES_PY, "crab_hex_omni_phases_eval_categories_test")


@pytest.fixture(scope="module")
def rc():
    return _load(DRIVER, "run_candidates_eval_categories_test")


@pytest.fixture(scope="module")
def cands():
    return yaml.safe_load(CANDS_YAML.read_text())


def _chain(cands, name):
    return {str(s["stage"]): s for s in cands["chains"][name]["stages"]}


def _env(ph, stage: dict, **extra) -> dict:
    """The COMPOSED training env: the preset's env with the stage overlay on top (D4)."""
    env = dict(ph.phase_env(str(stage["base"])))
    env.update({str(k): str(v) for k, v in (stage.get("overlay") or {}).items()})
    env.update(extra)
    return env


def _cand(cands, cid):
    return next(c for c in cands["candidates"] if c["id"] == cid)


def _edited(rc, pool_text: str, edit: dict) -> str:
    """A pool string with a candidate's ``pool_edit`` applied the way ``compose_recipe`` does."""
    return rc.render_pool(rc.apply_pool_edit(rc.parse_pool(pool_text), edit, require_match=False))


def _base_env(ph, **extra) -> dict:
    """A 1a-based env with no band -- the curriculum stages' base, for hand-written pools."""
    env = dict(ph.phase_env("1a"))
    env[BAND] = ""
    env.update(extra)
    return env


def _sets(tc):
    return (sorted(tc.decisive), sorted(tc.present), dict(tc.watch))


# ================================================================== categories and holds
def test_categories_holds_and_hands(ec):
    assert ec.CATEGORIES == ALL
    holds = [h for c in ec.CATEGORIES for h in ec.CATEGORY_HOLDS[c]]
    assert sorted(holds) == sorted(ec.SUITE_HOLDS) and len(holds) == len(set(holds)) == 10
    for c in ec.PAIRED_CATEGORIES:
        left, right = ec.CATEGORY_HOLDS[c]
        assert (ec.HAND_OF_HOLD[left], ec.HAND_OF_HOLD[right]) == ("l", "r")
    for c in set(ec.CATEGORIES) - set(ec.PAIRED_CATEGORIES):
        assert len(ec.CATEGORY_HOLDS[c]) == 1 and ec.CATEGORY_HOLDS[c][0] not in ec.HAND_OF_HOLD
    for h in ec.SUITE_HOLDS:
        assert h in ec.CATEGORY_HOLDS[ec.category_of_hold(h)]
    assert ec.category_of_hold("settle") is None
    with pytest.raises(KeyError):
        ec.category_of_hold("curriculum_back")
    # the legacy label vocabulary is kept where the command is the same
    assert ec.category_of_hold("move_turn") == "move_turn" and ec.category_of_hold("lat_l") == "side"
    assert set(ec.TRANSLATION_CATEGORIES) == {"fwd", "side", "back", "diag"}


def test_category_holds_cover_the_suite_rows_labels_exactly(ec):
    """Every label the ``suite__omni`` row carries is one category's hold, and the settle hold is none.

    Needs the manifest row A5 adds (execution step 1, in parallel with this step); until it exists
    the test SKIPS and runs at the (1 || 2) join.
    """
    doc = yaml.safe_load(MANIFEST.read_text())
    row = next((s for s in doc.get("scenarios") or [] if s.get("id") == SUITE_ROW), None)
    if row is None:
        pytest.skip(f"{SUITE_ROW} is not in {MANIFEST.name} yet (added by execution step 1, A5); "
                    "this test runs at the (1 || 2) join")
    settle = row.get("settle") or {}
    settle_label = settle.get("label", "settle") if isinstance(settle, dict) else "settle"
    labels, categories = [], {}

    def walk(node):
        if isinstance(node, dict):
            if "label" in node and node is not settle:
                labels.append(node["label"])
                if "category" in node:
                    categories[node["label"]] = node["category"]
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk({k: v for k, v in row.items() if k != "settle"})
    own = [lab for lab in labels if lab != settle_label]
    assert sorted(set(own)) == sorted(ec.SUITE_HOLDS)
    assert len(own) == len(set(own)), f"a hold label repeats on the suite row: {own}"
    for label, cat in categories.items():
        assert ec.category_of_hold(label) == cat
    # the row's own category table (A5's shape: categories: [{name, holds: [{label, hand?}]}])
    table = row.get("categories")
    if isinstance(table, list) and all(isinstance(t, dict) and "name" in t for t in table):
        assert {t["name"]: tuple(h["label"] for h in t["holds"]) for t in table} == ec.CATEGORY_HOLDS
        assert [h["label"] for t in table for h in t["holds"]] == list(ec.SUITE_HOLDS)  # wave order
        for t in table:
            for h in t["holds"]:
                assert h.get("hand") == ec.HAND_OF_HOLD.get(h["label"]), h


# ================================================================== class defaults (pinned)
def _class_body_defaults(path: Path, cls: str) -> dict:
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls)
    out = {}
    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            try:
                out[stmt.target.id] = ast.literal_eval(stmt.value)
            except ValueError:
                continue
    return out


def test_absent_keys_take_the_command_cfg_class_defaults_pinned_against_the_class_body(ec):
    body = _class_body_defaults(ec.CMD_CFG_PY, ec.CMD_CFG_CLASS)
    for field, value in ec.CMD_CFG_DEFAULTS.items():
        assert field in body, f"{ec.CMD_CFG_CLASS} no longer declares {field} with a literal default"
        assert body[field] == value, f"{field}: class body {body[field]!r} != eval_categories {value!r}"
    # the facing-observation encoding default: the obs term's class body AND the env cfg's params
    obs = _class_body_defaults(OMNI / "mdp/crab_hex_omni_observations.py", "CrabHexOmniParkourObservations")
    assert obs["facing_obs_encoding"] == ec.FACING_OBS_ENCODING_DEFAULT
    env_src = ast.parse(ec.ENV_CFG_PY.read_text())
    params = [ast.literal_eval(v) for n in ast.walk(env_src) if isinstance(n, ast.Dict)
              for k, v in zip(n.keys, n.values)
              if isinstance(k, ast.Constant) and k.value == "facing_obs_encoding"]
    assert params and set(params) == {ec.FACING_OBS_ENCODING_DEFAULT}


def test_an_absent_key_reads_exactly_as_its_class_default(ec, ph):
    env = dict(ph.phase_env("1c"))
    assert BAND not in env and ENCODING not in env  # the 1c preset carries neither
    spelled = dict(env, **{BAND: "none", ENCODING: "raw"})
    assert ec.trained_categories(env) == ec.trained_categories(spelled)
    no_slots = {k: v for k, v in env.items() if k not in (SLOTS, FACING)}
    assert ec.trained_categories(no_slots) == ec.trained_categories(
        dict(no_slots, **{SLOTS: "0.15:0.55:0.10:0.20", FACING: "0.2:0.6"}))
    assert ec.trained_categories(no_slots).facing_band_hi == 0.6


# ================================================================== trained categories: shares
def test_decisive_present_watch_at_the_floor_share_on_normalised_shares(ec, ph):
    at = ec.trained_categories(_base_env(ph, **{POOL: "stand:0.2,fwd:0.775,back:0.025"}))
    below = ec.trained_categories(_base_env(ph, **{POOL: "stand:0.2,fwd:0.78,back:0.02"}))
    assert "back" in at.decisive and "back" not in at.watch
    assert "back" in below.present and "back" not in below.decisive
    assert below.watch == {"back": ec.NOTE_BELOW_FLOOR}
    # raw weights x10 and a raw weight below the floor whose NORMALISED share clears it
    scaled = ec.trained_categories(_base_env(ph, **{POOL: "stand:2,fwd:7.75,back:0.25"}))
    assert _sets(scaled) == _sets(at)
    small = ec.trained_categories(_base_env(ph, **{POOL: "stand:0.02,fwd:0.16,back:0.02"}))
    assert set(small.decisive) == {"stand", "fwd", "back"}
    # a type at weight 0 is neither present nor decisive
    zero = ec.trained_categories(_base_env(ph, **{POOL: "stand:0.2,fwd:0.8,back:0.0"}))
    assert "back" not in zero.present


def test_an_unrenormalised_set_share_edit_gives_the_same_sets(ec, ph, rc, cands):
    c6 = _chain(cands, "curriculum")["c6"]
    pool = c6["overlay"][POOL]
    plain = _edited(rc, pool, {"set_share": {"stand": 0.05}})
    renorm = _edited(rc, pool, {"set_share": {"stand": 0.05}, "renormalise": True})
    assert plain != renorm
    a = ec.trained_categories(_env(ph, c6, **{POOL: plain}))
    b = ec.trained_categories(_env(ph, c6, **{POOL: renorm}))
    assert _sets(a) == _sets(b) and sorted(a.decisive) == sorted(ALL)


def test_omni_makes_every_bearing_decisive_with_the_note(ec, ph):
    tc = ec.trained_categories(_base_env(ph, **{POOL: "stand:0.2,omni:0.8"}))
    assert sorted(tc.decisive) == sorted({"stand", "fwd", "side", "back", "diag"})
    assert tc.hands["side"] == ("l", "r")
    assert any(n.startswith("omni") and "uniform-bearing" in n for n in tc.notes)
    assert "turn" not in tc.present and "move_turn" not in tc.present


def test_one_handed_side_is_watch(ec, ph):
    tc = ec.trained_categories(_base_env(ph, **{POOL: "stand:0.2,fwd:0.5,side_l:0.3"}))
    assert "side" in tc.present and "side" not in tc.decisive
    assert tc.watch == {"side": ec.NOTE_ONE_HAND} and tc.hands["side"] == ("l",)
    both = ec.trained_categories(_base_env(ph, **{POOL: "stand:0.2,fwd:0.5,side_l:0.15,side_r:0.15"}))
    assert "side" in both.decisive and both.hands["side"] == ("l", "r")
    # the second hand below the floor share is still one hand
    thin = ec.trained_categories(_base_env(ph, **{POOL: "stand:0.2,fwd:0.49,side_l:0.3,side_r:0.01"}))
    assert thin.watch == {"side": ec.NOTE_ONE_HAND} and thin.hands["side"] == ("l", "r")
    # omni supplies both hands
    omni = ec.trained_categories(_base_env(ph, **{POOL: "stand:0.2,side_l:0.3,omni:0.5"}))
    assert "side" in omni.decisive


def test_c1b_and_c7_pool_edits_keep_turn_decisive(ec, ph, rc, cands):
    """``{"*_turn": 0.3}`` touches ``move_turn`` only -- ``fnmatch("turn", "*_turn")`` is False."""
    chain = _chain(cands, "curriculum")
    for cid, stage in (("C1b", "c4"), ("C1b", "c6"), ("C7", "c6")):
        edit = _cand(cands, cid)["pool_edit"]
        assert edit == {"ang_scale": {"*_turn": 0.3}}
        st = chain[stage]
        tc = ec.trained_categories(_env(ph, st, **{POOL: _edited(rc, st["overlay"][POOL], edit)}))
        assert "turn" in tc.decisive, (cid, stage)
        assert tc.watch == {"move_turn": ec.NOTE_REDUCED}, (cid, stage)
        assert tc.demand["move_turn"] == pytest.approx(0.3) and tc.demand["turn"] == 1.0
        unedited = ec.trained_categories(_env(ph, st))
        assert set(tc.decisive) == set(unedited.decisive) - {"move_turn"}
        assert set(tc.present) == set(unedited.present)


def test_an_ang_scale_on_the_turn_type_demotes_turn(ec, ph):
    tc = ec.trained_categories(_base_env(ph, **{
        POOL: "stand:0.15,fwd:0.35,side_l:0.1,side_r:0.1,turn:0.2:0.5,fwd_turn:0.1"}))
    assert tc.watch == {"turn": ec.NOTE_REDUCED}
    assert "move_turn" in tc.decisive and tc.demand["turn"] == 0.5
    # a turn type at full demand beside a reduced one keeps its category decisive
    mixed = ec.trained_categories(_base_env(ph, **{
        POOL: "stand:0.15,fwd:0.5,fwd_turn:0.2:0.3,omni_turn:0.15"}))
    assert "move_turn" in mixed.decisive


def test_a_speed_scale_demotes_only_that_types_translation_categories(ec, ph):
    tc = ec.trained_categories(_base_env(ph, **{
        POOL: "stand:0.2,fwd:0.3:1.0:0.9,side_l:0.15,side_r:0.15,back:0.2"}))
    assert tc.watch == {"fwd": ec.NOTE_REDUCED}
    assert {"side", "back"} <= tc.decisive
    omni = ec.trained_categories(_base_env(ph, **{POOL: "stand:0.2,fwd:0.3,omni:0.5:1.0:0.9"}))
    assert "fwd" in omni.decisive  # the full-demand fwd type still trains it
    assert omni.watch == {c: ec.NOTE_REDUCED for c in ("side", "back", "diag")}


def test_sincos_keeps_turn_decisive_and_records_the_encoding_note(ec, ph):
    raw = ec.trained_categories(ph.phase_env("1c"))
    sincos = ec.trained_categories(dict(ph.phase_env("1c"), **{ENCODING: "sincos"}))
    assert _sets(raw) == _sets(sincos) and "turn" in sincos.decisive
    assert (raw.encoding, sincos.encoding) == ("raw", "sincos")
    assert ec.turn_annotation(sincos) in sincos.notes
    assert sincos.notes[-1] == ("turn: demand 1.571 rad > FACING band 0.600; "
                                "encoding sincos sees the magnitude")
    assert raw.notes[-1] == "turn: demand 1.571 rad > FACING band 0.600; encoding raw clips at 0.6"
    # a band wider than the suite's demand still only annotates
    wide = ec.trained_categories(ph.phase_env("2b"))
    assert "turn" in wide.decisive and "<= FACING band 2.400" in wide.notes[-1]


def test_move_turn_is_decisive_at_full_demand_and_watch_under_a_reduced_one(ec, ph, rc, cands):
    b0 = _chain(cands, cands["baseline"]["chain"])["1c"]
    assert "move_turn" in ec.trained_categories(_env(ph, b0)).decisive
    c4 = _chain(cands, "curriculum")["c4"]
    assert "move_turn" in ec.trained_categories(_env(ph, c4)).decisive
    c6 = _chain(cands, "curriculum")["c6"]
    for cid, st in (("C1b", c4), ("C7", c6)):
        pool = _edited(rc, st["overlay"][POOL], _cand(cands, cid)["pool_edit"])
        tc = ec.trained_categories(_env(ph, st, **{POOL: pool}))
        assert "move_turn" in tc.present and tc.watch["move_turn"] == ec.NOTE_REDUCED


def test_slots_mode_reproduces_the_legacy_presets(ec, ph):
    expect = {
        "1a": ({"stand", "fwd"}, {}),
        "1b": ({"stand"}, {"side": ec.NOTE_ONE_HAND}),
        "1c": (set(ALL), {}), "2a": (set(ALL), {}), "2b": (set(ALL), {}), "2c": (set(ALL), {}),
        "2c_clockheld": (set(ALL), {}),
    }
    for name, (dec, watch) in expect.items():
        tc = ec.trained_categories(ph.phase_env(name))
        assert tc.source == "slots"
        assert set(tc.decisive) == dec and tc.watch == watch, name
    assert ec.trained_categories(ph.phase_env("1b")).hands["side"] == ("l",)
    # the 1c pool twin names the same sets as the 1c slots
    twin = dict(ph.phase_env("1c"), **{POOL: "stand:0.15,omni:0.55,turn:0.10,omni_turn:0.20"})
    tc = ec.trained_categories(twin)
    assert tc.source == "pool" and _sets(tc) == _sets(ec.trained_categories(ph.phase_env("1c")))
    # a band that holds two body axes supports both; one that holds none supports nothing
    two = ec.trained_categories(dict(ph.phase_env("1a"), **{BAND: "0:1.5708"}))
    assert {"fwd"} <= two.decisive and two.watch == {"side": ec.NOTE_ONE_HAND}
    none = ec.trained_categories(dict(ph.phase_env("1a"), **{BAND: "0.3:0.5"}))
    assert set(none.present) == {"stand"}


def test_curriculum_stages_read_from_candidates_yaml(ec, ph, cands):
    expect = {
        "c1": {"stand", "fwd"},
        "c2": {"stand", "fwd", "side", "back"},
        "c3": {"stand", "fwd", "side", "back", "turn"},
        "c4": {"stand", "fwd", "side", "back", "turn", "move_turn"},
        "c5": set(ALL),
        "c6": set(ALL),
    }
    stages = cands["chains"]["curriculum"]["stages"]
    assert [str(s["stage"]) for s in stages] == list(expect)
    envs = []
    for st in stages:
        env = _env(ph, st)
        tc = ec.trained_categories(env)
        assert set(tc.decisive) == expect[str(st["stage"])], st["stage"]
        assert set(tc.present) == set(tc.decisive) and not tc.watch and tc.source == "pool"
        # D6: previous_categories = U decisive(stages[:i]) -- envs, tuples and dict forms alike
        assert ec.previous_categories(envs) == frozenset().union(
            *(expect[str(s["stage"])] for s in stages[:len(envs)]))
        assert ec.previous_categories([ec.trained_categories(e).to_dict() for e in envs]) == \
            ec.previous_categories(envs)
        envs.append(env)
    scratch = ec.trained_categories(_env(ph, _chain(cands, "scratch_c6")["c6"]))
    assert set(scratch.decisive) == set(ALL)


def test_zero_stand_share_is_an_eval_mismatch(ec, ph):
    with pytest.raises(ec.EvalMismatch, match="stand"):
        ec.trained_categories(_base_env(ph, **{POOL: "stand:0.0,fwd:1.0"}))
    with pytest.raises(ec.EvalMismatch):
        ec.trained_categories(dict(ph.phase_env("1c"), **{SLOTS: "0:0.6:0.2:0.2"}))
    assert issubclass(ec.EvalMismatch, ValueError)


def test_trained_categories_round_trip_through_their_dict_form(ec, ph):
    import json

    for name in ("1a", "1b", "1c"):
        tc = ec.trained_categories(ph.phase_env(name))
        d = json.loads(json.dumps(tc.to_dict()))
        assert d["decisive"] == [c for c in ALL if c in tc.decisive]
        back = ec.TrainedCategories.from_dict(d)
        assert back == tc
        assert ec.decisive_set(d) == ec.decisive_set(tc) == tc.decisive


# ================================================================== expand_rule
def _rule(ec, **extra):
    return ec.default_rule_v2(text="per-category rule", bootstrap={"B": 2000, "rng_seed": 1},
                              control_head="1c-004", **extra)


def _scored(r):
    return set(r["guards"]["metrics"]).union(*(set(b["metrics"]) for b in r["blocks"].values()))


def test_expand_rule_at_b0_is_todays_flat_shape(ec):
    rule = _rule(ec)
    before = copy.deepcopy(rule)
    r = ec.expand_rule(rule, ALL, ALL, absolute_units=False)
    assert rule == before  # never mutated
    assert list(r["blocks"]) == ["F", "F_mt", "T"] and r["na_blocks"] == []
    assert r["blocks"]["F"] == {"level": 0.95, "metrics": ["cat.turn.facing_closed", "cat.turn.yaw_progress"],
                                "categories": ["turn"]}
    assert r["blocks"]["F_mt"]["metrics"] == ["cat.move_turn.facing_closed", "cat.move_turn.yaw_progress"]
    assert r["blocks"]["T"]["metrics"] == ["cat.fwd.ratio", "cat.side.ratio", "cat.back.ratio", "cat.diag.ratio"]
    assert r["guards"] == {"level": 0.99, "metrics": [f"cat.{c}.survival" for c in ALL]
                           + ["cat.move_turn.ratio", "cat_tripod_min"]}
    assert r["composite"] == {"cat_tripod_min": ["fwd", "side", "back", "diag"]}
    # the nominal null of the ratio system at B0: three blocks, nine guards
    assert len(r["blocks"]) == 3 and len(r["guards"]["metrics"]) == 9
    # carried through; the templates are not
    assert (r["text"], r["control_head"], r["bootstrap"]) == ("per-category rule", "1c-004", rule["bootstrap"])
    assert "per_category" not in r and r["rule_version"] == 2 and r["units"] == "ratio"
    # WATCH: a dict of id -> note; iterating it gives the ids (what today's readers do)
    w = r["watch"]
    assert isinstance(w, dict) and list(w) == list(w.keys())
    assert w["cat.move_turn.achieved_speed"] is None and w["cat.move_turn.tripod"] is None
    assert w["cat.stand.drift_speed"] is None and w["cat.turn.settle_s"] is None
    assert w["cat.side.by_hand.r.ratio"] is None and w["cat.move_turn.by_hand.l.facing_closed"] is None
    assert w["turn_survival_l90"] == "N/A (suite)" and w["completion"] == "legacy"
    assert not (set(w) & _scored(r))
    for mid in list(w) + sorted(_scored(r) - {"cat_tripod_min"}):
        if mid.startswith("cat."):
            assert ec.parse_cat_id(mid) is not None, mid
    # the ids are dotted paths into summarize()['cat']
    assert ec.cat_id("side", "ratio") == "cat.side.ratio"
    assert ec.parse_cat_id("cat.side.by_hand.l.ratio") == ("side", "l", "ratio")
    assert ec.parse_cat_id("cat.nope.ratio") is None and ec.parse_cat_id("ratio_by_hold.fwd") is None


def test_expand_rule_keeps_the_survival_guard_of_a_watch_category_and_watches_the_rest(ec):
    """C1b-style: move_turn trained at 30 % of the suite's turn demand."""
    decisive = set(ALL) - {"move_turn"}
    notes = {"move_turn": ec.NOTE_REDUCED}
    r = ec.expand_rule(_rule(ec), decisive, ALL, absolute_units=False, watch_notes=notes)
    assert "cat.move_turn.survival" in r["guards"]["metrics"]
    assert "cat.move_turn.ratio" not in r["guards"]["metrics"]
    assert r["blocks"]["F_mt"]["metrics"] == [] and r["na_blocks"] == ["F_mt"]
    assert r["blocks"]["F"]["metrics"] and r["blocks"]["T"]["metrics"]
    for mid in ("cat.move_turn.facing_closed", "cat.move_turn.yaw_progress", "cat.move_turn.ratio",
                "cat.move_turn.tripod", "cat.move_turn.settle_s", "cat.move_turn.achieved_speed",
                "cat.move_turn.by_hand.r.facing_closed"):
        assert r["watch"][mid] == "watch (reduced demand)", mid
    # the same through the TrainedCategories convenience
    tc = ec.TrainedCategories(frozenset(decisive), frozenset(ALL), notes, {}, {}, "pool", 0.6, "raw", ())
    assert ec.expand_for(_rule(ec), tc, absolute_units=False) == r
    assert ec.expand_for(_rule(ec), tc.to_dict(), absolute_units=False) == r
    # without a note a watch category reads plain "watch"
    bare = ec.expand_rule(_rule(ec), decisive, ALL, absolute_units=False)
    assert bare["watch"]["cat.move_turn.ratio"] == "watch"


def test_expand_rule_one_handed_side(ec):
    """1b-style: stand decisive, side trained on one hand only, nothing else."""
    r = ec.expand_rule(_rule(ec), {"stand"}, {"stand", "side"}, absolute_units=False,
                       watch_notes={"side": ec.NOTE_ONE_HAND})
    assert r["guards"]["metrics"] == ["cat.stand.survival", "cat.side.survival"]
    assert r["na_blocks"] == ["F", "F_mt", "T"] and r["composite"] == {}
    assert "cat_tripod_min" not in r["guards"]["metrics"]
    assert r["watch"]["cat.side.ratio"] == "watch (one hand)"
    assert r["watch"]["cat.side.tripod"] == "watch (one hand)"
    assert r["watch"]["cat.side.direction_error"] == "watch (one hand)"
    for c in ("fwd", "back", "diag", "turn", "move_turn"):
        assert r["watch"][f"cat.{c}.survival"] == "untrained"
    assert r["watch"]["cat.fwd.ratio"] == "untrained" and r["watch"]["cat.turn.facing_closed"] == "untrained"
    assert r["watch"]["cat.move_turn.ratio"] == "untrained"
    # D3's WATCH column is for PRESENT categories only
    assert "cat.fwd.direction_error" not in r["watch"] and "cat.turn.settle_s" not in r["watch"]
    assert r["watch"]["cat.stand.drift_speed"] is None


def test_a_block_with_no_decisive_member_is_na(ec):
    """Curriculum c2: no turning category is trained, so F and F_mt are N/A (never TIE)."""
    dec = {"stand", "fwd", "side", "back"}
    r = ec.expand_rule(_rule(ec), dec, dec, absolute_units=False)
    assert r["na_blocks"] == ["F", "F_mt"]
    assert r["blocks"]["F"] == {"level": 0.95, "metrics": [], "categories": []}
    assert r["blocks"]["T"]["metrics"] == ["cat.fwd.ratio", "cat.side.ratio", "cat.back.ratio"]
    assert r["composite"] == {"cat_tripod_min": ["fwd", "side", "back"]}
    assert r["watch"]["cat.diag.ratio"] == "untrained" and r["watch"]["cat.diag.tripod"] == "untrained"
    ra = ec.expand_rule(_rule(ec), dec, dec, absolute_units=True)
    assert ra["na_blocks"] == ["F_abs", "F_mt_abs"]


def test_expand_rule_absolute_units_swaps_the_move_turn_guard(ec):
    r = ec.expand_rule(_rule(ec), ALL, ALL, absolute_units=True)
    assert list(r["blocks"]) == ["F_abs", "F_mt_abs", "T_abs"] and r["units"] == "absolute"
    assert r["blocks"]["F_abs"]["metrics"] == ["cat.turn.facing_closed", "cat.turn.net_yaw"]
    assert r["blocks"]["F_mt_abs"]["metrics"] == ["cat.move_turn.facing_closed", "cat.move_turn.net_yaw"]
    assert r["blocks"]["T_abs"]["metrics"] == [f"cat.{c}.achieved_speed" for c in ("fwd", "side", "back", "diag")]
    g = r["guards"]["metrics"]
    assert "cat.move_turn.achieved_speed" in g and "cat.move_turn.ratio" not in g
    assert g == [f"cat.{c}.survival" for c in ALL] + ["cat.move_turn.achieved_speed", "cat_tripod_min"]
    assert r["watch"]["cat.move_turn.ratio"] is None  # the other system's guard is watched
    assert "cat.move_turn.achieved_speed" not in r["watch"]
    # watch category under absolute units: survival stays, the swapped guard is watched with its note
    rw = ec.expand_rule(_rule(ec), set(ALL) - {"move_turn"}, ALL, absolute_units=True,
                        watch_notes={"move_turn": ec.NOTE_REDUCED})
    assert "cat.move_turn.survival" in rw["guards"]["metrics"]
    assert rw["watch"]["cat.move_turn.achieved_speed"] == ec.NOTE_REDUCED
    assert rw["na_blocks"] == ["F_mt_abs"]


def test_expand_rule_union_covers_both_unit_systems(ec):
    u = ec.expand_rule_union(_rule(ec), ALL, ALL)
    assert list(u["blocks"]) == ["F", "F_mt", "T", "F_abs", "F_mt_abs", "T_abs"]
    assert u["units"] == "union" and u["na_blocks"] == []
    assert u["guards"]["metrics"] == ([f"cat.{c}.survival" for c in ALL]
                                      + ["cat.move_turn.ratio", "cat_tripod_min", "cat.move_turn.achieved_speed"])
    ratio = ec.expand_rule(_rule(ec), ALL, ALL, absolute_units=False)
    absu = ec.expand_rule(_rule(ec), ALL, ALL, absolute_units=True)
    for name, body in {**ratio["blocks"], **absu["blocks"]}.items():
        assert u["blocks"][name] == body
    assert not (set(u["watch"]) & _scored(u))
    assert "cat.move_turn.ratio" not in u["watch"] and "cat.move_turn.achieved_speed" not in u["watch"]
    assert set(u["watch"]) == (set(ratio["watch"]) | set(absu["watch"])) - _scored(u)
    # at a C1b-style set: both F_mt twins are N/A, move_turn survival is guarded, both translation guards watched
    uw = ec.expand_rule_union(_rule(ec), set(ALL) - {"move_turn"}, ALL,
                              watch_notes={"move_turn": ec.NOTE_REDUCED})
    assert uw["na_blocks"] == ["F_mt", "F_mt_abs"]
    assert "cat.move_turn.survival" in uw["guards"]["metrics"]
    assert uw["watch"]["cat.move_turn.ratio"] == uw["watch"]["cat.move_turn.achieved_speed"] == ec.NOTE_REDUCED


def test_expand_rule_refuses_a_v1_rule_and_unknown_names(ec):
    v1 = yaml.safe_load(CANDS_YAML.read_text())["decision_rule"]
    if int(v1.get("rule_version") or 1) == 1:
        with pytest.raises(ValueError, match="rule_version"):
            ec.expand_rule(v1, ALL, ALL, absolute_units=False)
    with pytest.raises(ValueError, match="per_category"):
        ec.expand_rule({"rule_version": 2}, ALL, ALL, absolute_units=False)
    with pytest.raises(ValueError, match="unknown categories"):
        ec.expand_rule(_rule(ec), {"sideways"}, ALL, absolute_units=False)
    bad = _rule(ec)
    bad["per_category"]["blocks"]["T"]["members"]["fwd"] = ["speed"]
    with pytest.raises(ValueError, match="keys"):
        ec.expand_rule(bad, ALL, ALL, absolute_units=False)
    # decisive is folded into present (a decisive category keeps its survival guard)
    r = ec.expand_rule(_rule(ec), {"stand", "fwd"}, {"stand"}, absolute_units=False)
    assert r["present"] == ["stand", "fwd"] and "cat.fwd.survival" in r["guards"]["metrics"]


def test_the_rule_template_is_yaml_round_trippable(ec):
    text = yaml.safe_dump({"decision_rule": ec.default_rule_v2()})
    assert yaml.safe_load(text)["decision_rule"]["per_category"] == ec.RULE_V2_PER_CATEGORY


# ================================================================== composite guards
def _screen(**tripods):
    return {"cat": {c: {"tripod": v, "survival": 0.9} for c, v in tripods.items()}, "completion": 0.5}


def test_composite_guards_min_over_the_decisive_translation_categories_only(ec):
    s = _screen(fwd=0.52, side=0.47, back=0.44, diag=0.50, move_turn=0.10, turn=0.05, stand=0.0)
    assert ec.composite_guards(s, ALL) == {"cat_tripod_min": 0.44}
    # an untrained (non-decisive) translation category never enters
    assert ec.composite_guards(s, {"stand", "fwd", "side", "turn", "move_turn"}) == {"cat_tripod_min": 0.47}
    # move_turn / turn never enter, even when they are the lowest
    assert ec.composite_guards(s, {"stand", "fwd", "move_turn", "turn"}) == {"cat_tripod_min": 0.52}
    # no decisive translation category -> no composite at all
    assert ec.composite_guards(s, {"stand", "turn"}) == {}
    # a missing or non-finite member is a MISSING value, never a narrower min
    assert ec.composite_guards(_screen(fwd=0.5, side=None), {"fwd", "side"}) == {"cat_tripod_min": None}
    assert ec.composite_guards(_screen(fwd=0.5), {"fwd", "back"}) == {"cat_tripod_min": None}
    assert ec.composite_guards(_screen(fwd=0.5, back=math.nan), {"fwd", "back"}) == {"cat_tripod_min": None}
    assert ec.composite_guards({}, {"fwd"}) == {"cat_tripod_min": None}
    # a TrainedCategories / its dict form stands in for the decisive set
    tc = ec.TrainedCategories(frozenset({"stand", "fwd", "back"}), frozenset({"stand", "fwd", "back"}),
                              {}, {}, {}, "pool", 0.6, "raw", ())
    assert ec.composite_guards(s, tc) == ec.composite_guards(s, tc.to_dict()) == {"cat_tripod_min": 0.44}


def test_with_composites_is_a_shallow_copy_and_matches_the_expanded_guard(ec):
    s = _screen(fwd=0.52, side=0.47, back=0.44, diag=0.50)
    out = ec.with_composites(s, ALL)
    assert out is not s and "cat_tripod_min" not in s and out["cat_tripod_min"] == 0.44
    assert out["cat"] is s["cat"]
    assert ec.with_composites(None, ALL) is None
    # every guard id of the expanded rule resolves on the injected screen
    r = ec.expand_rule(_rule(ec), ALL, ALL, absolute_units=False)
    for c in ALL:
        out["cat"].setdefault(c, {"survival": 0.9})
    out["cat"]["move_turn"]["ratio"] = 0.4
    for mid in r["guards"]["metrics"]:
        cur = out
        for part in mid.split("."):
            cur = cur[part]
        assert cur is not None, mid


# ================================================================== purity and hashing
def test_eval_categories_imports_without_isaac():
    """Loading the module by path and deriving a preset's sets must import no Isaac / parkour package.

    A FRESH interpreter: other unit tests in one pytest process legitimately stub ``isaaclab.*``.
    """
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('ec_pure', {str(MODULE)!r})\n"
        "ec = importlib.util.module_from_spec(spec); sys.modules['ec_pure'] = ec\n"
        "spec.loader.exec_module(ec)\n"
        f"spec = importlib.util.spec_from_file_location('ph_pure', {str(PHASES_PY)!r})\n"
        "ph = importlib.util.module_from_spec(spec); sys.modules['ph_pure'] = ph\n"
        "spec.loader.exec_module(ph)\n"
        "tc = ec.trained_categories(ph.phase_env('1c'))\n"
        "assert len(tc.decisive) == 7\n"
        "r = ec.expand_rule_union(ec.default_rule_v2(), tc.decisive, tc.present)\n"
        "assert ec.composite_guards({}, tc.decisive) == {'cat_tripod_min': None}\n"
        "bad = ('parkour_tasks', 'parkour_isaaclab', 'isaaclab', 'isaacsim', 'omni.', 'pxr', 'carb')\n"
        "print(sorted(m for m in sys.modules if m.startswith(bad) or m in ('omni', 'isaacsim')))\n"
    )
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.strip() == "[]", out.stdout


def test_eval_categories_is_outside_code_sha(rc):
    rel = MODULE.relative_to(REPO).as_posix()
    assert not any(rel == spec or rel.startswith(spec.rstrip("/") + "/") for spec in rc.CODE_PATHSPECS)
    assert not any(row.split(" ", 1)[1] == rel for row in rc.code_rows())
