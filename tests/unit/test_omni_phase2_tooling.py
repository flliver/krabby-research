"""Omni phase-2 driver tooling (plan foamy-wondering-ullman step 4, 2026-10-06): the
``run_candidates.py`` edits the phase-2 arms need before any of them can launch.

1. a BLANK ``KRABBY_PHASEOUT`` (the 2a preset's three clock/satellite ramps removed): the knob round
   trip returns None, the dump audit pins ``curriculum: null`` with no ``phaseout_`` block, and clock
   rule (ii) still refuses the un-blanked preset apex 1.0 under ``KRABBY_OMNI_CLOCK_HIDE=1``;
2. ``OVERLAY_DUMP_CHECKS`` for the field geometry, the promotion pair, the reused stride / airtime /
   power weights and the F1 / E1 terms -- each PINNED to the dumped value (the vacuity tests below
   run every field check on a RING dump and expect a FAIL);
3. ``_filtered_phase_checks``' geometry-aware generator check (a field record must not carry
   run_phases' ring check as a false FAIL; a ring record reads exactly as before);
4. the whitelist / parser table see every phase-2 key, so the campaign's own ``candidates.yaml``
   validates with only its named placeholders refused.

The dumps are built the way the trainer writes ``params/env.yaml`` (IsaacLab ``dump_yaml``:
``yaml.dump(..., default_flow_style=False, sort_keys=False)`` on the cfg dict -- tuples as
``!!python/tuple``, indentless lists), and the ring checks are also run on a REAL c6x dump when the
run directory is present. Pure Python: nothing imports Isaac or launches anything.
"""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
OMNI = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
TOOLS = OMNI / "experiments/tools"
RUN_CANDIDATES = TOOLS / "run_candidates.py"
P2_CANDIDATES = OMNI / "experiments/2026-10-07_omni_phase2/candidates.yaml"
TERRAINS = REPO / "parkour/parkour_isaaclab/terrains/omni"
#: The B head's own c6x run (the real ring dump the indents are checked against, when present).
C6X_DUMP = (REPO / "parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-10-06_02-24-21_seed18_X60_c6x_A15pB"
            / "params/env.yaml")

ACTUATED = [".*_Body_CamShaft_RevoluteJoint", ".*_Hip_Femur_RevoluteJoint", ".*_Femur_Tibia_RevoluteJoint"]
PRESET_2A_PHASEOUT = ("reward_clock_swing_apex:1.0:0.5:0:24000,reward_feet_air_time_positive:0.8:0.4:0:24000,"
                      "reward_stride_length:0.5:0.25:0:24000")
#: The phase-2 additions every field arm overlays (CHARTER 3), env side only.
P2_FIELD = {"KRABBY_PHASEOUT": "", "KRABBY_FLAT_TERRAIN_GEOM": "field", "KRABBY_FIELD_PAD_R": "1.7",
            "KRABBY_FIELD_BLOCKS": "14:28", "KRABBY_FIELD_KINDS": "0.4:0.4:0.2", "KRABBY_FIELD_EDGES": "ramp",
            "KRABBY_TERRAIN_PROMOTE": "0.45:0.25"}
#: Every key step 4 must make auditable.
P2_KEYS = ("KRABBY_PHASEOUT", "KRABBY_FLAT_TERRAIN_GEOM", "KRABBY_FIELD_PAD_R", "KRABBY_FIELD_BLOCKS",
           "KRABBY_FIELD_KINDS", "KRABBY_FIELD_EDGES", "KRABBY_FIELD_MIN_GAP", "KRABBY_FIELD_SIZE_SCALE",
           "KRABBY_TERRAIN_PROMOTE", "KRABBY_STRIDE_W", "KRABBY_AIRTIME_W", "KRABBY_POWER_W",
           "KRABBY_POWER_JOINTS", "KRABBY_STAND_STANCE_W")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rc():
    return _load(RUN_CANDIDATES, "run_candidates_omni_phase2_test")


# ============================================================ a params/env.yaml, written like the trainer
def _entity(joint_names=None, body_names=None) -> dict:
    return {"name": "robot", "joint_names": joint_names, "joint_ids": slice(None),
            "body_names": body_names, "body_ids": slice(None), "preserve_order": False}


def _sub(kind: str, **kw) -> dict:
    """One omni sub-terrain cfg as the generator dumps it (the field cfg's own defaults)."""
    base = {"function": f"parkour_isaaclab.terrains.omni.omni_terrains:omni_{kind}_terrain",
            "proportion": 0.25, "size": (10, 10), "horizontal_scale": 0.08, "vertical_scale": 0.005,
            "slope_threshold": 1.5, "noise_range": (0.02, 0.02)}
    if kind == "field":
        base.update(clear_radius=1.7, ring_outer_radius=4.8, field_margin=0.16, n_blocks=14,
                    bar_fraction=0.4, pit_fraction=0.2, min_gap=0.0, max_tries=64,
                    bar_length_range=(1.0, 2.0), bar_thickness_range=(0.16, 0.32), slab_side_range=(0.6, 1.2),
                    hurdle_height_range="0.02 + 0.05 * difficulty, 0.03 + 0.06 * difficulty",
                    step_height="0.02 + 0.06 * difficulty", pit_depth="0.02 + 0.06 * difficulty",
                    vertical_edges=False)
    elif kind == "ring":
        base.update(clear_radius=2.0, ring_outer_radius=4.5, n_blocks=8, bar_fraction=0.5)
    else:
        base.update(clear_radius=1.7, ring_outer_radius=4.8)
    base.update(kw)
    return base


def env_dump(geom: str = "field", *, subs: dict | None = None, parkour: dict | None = None,
             rewards: dict | None = None, curriculum=None) -> str:
    """A ``params/env.yaml`` text: scene/terrain generator, rewards, curriculum, parkours."""
    if subs is None:
        if geom == "field":
            subs = {"parkour_flat": _sub("flat", proportion=0.5),
                    "omni_field_sparse": _sub("field", n_blocks=14), "omni_field_dense": _sub("field", n_blocks=28)}
        else:
            subs = {"parkour_flat": _sub("flat", proportion=0.5, clear_radius=2.0, ring_outer_radius=4.5),
                    "omni_ring_sparse": _sub("ring", n_blocks=8), "omni_ring_dense": _sub("ring", n_blocks=16)}
    radii = next(s for s in subs.values())
    pk = {"class_type": "parkour_tasks.crab_hex_omni_task.mdp.omni_parkour_event:OmniParkourEvent",
          "asset_name": "robot", "freeze_terrain_levels": False, "move_up_frac": 0.45, "move_down_frac": 0.25,
          "clear_radius_m": radii["clear_radius"], "ring_outer_radius_m": radii["ring_outer_radius"],
          "n_sectors": 8}
    pk.update(parkour or {})
    rew = {
        "reward_feet_air_time_positive": {"func": "parkour_isaaclab.envs.mdp.rewards:reward_feet_air_time_positive",
                                          "params": {"sensor_cfg": _entity(body_names=".*_Footpad"),
                                                     "threshold": 0.38}, "weight": 0.8},
        "reward_stride_length": {"func": "parkour_isaaclab.envs.mdp.rewards:RewardStrideLength",
                                 "params": {"asset_cfg": _entity(), "speed_ref": 2.8}, "weight": 0.5},
        "penalty_mechanical_power": {"func": "parkour_isaaclab.envs.mdp.rewards:penalty_mechanical_power",
                                     "params": {"asset_cfg": _entity()}, "weight": 0.0},
        "reward_clock_schedule": {"func": "parkour_isaaclab.envs.mdp.rewards:reward_clock_schedule",
                                  "params": {"combine": "product"}, "weight": 0.001},
    }
    for name, term in (rewards or {}).items():
        if term is None:
            rew.pop(name, None)
        else:
            rew[name] = term
    doc = {
        "seed": 18,
        "scene": {"num_envs": 256,
                  "robot": {"class_type": "isaaclab.assets.articulation.articulation:Articulation"},
                  "terrain": {"class_type": "parkour_isaaclab.terrains.parkour_terrain_importer:ParkourTerrainImporter",
                              "prim_path": "/World/ground", "num_envs": 256, "terrain_type": "generator",
                              "terrain_generator": {"class_type": "isaaclab.terrains.terrain_generator:TerrainGenerator",
                                                    "seed": None, "curriculum": True, "size": (10, 10),
                                                    "border_width": 20, "slope_threshold": 1.5,
                                                    "sub_terrains": subs, "difficulty_range": (0.2, 0.7),
                                                    "num_goals": 8},
                              "usd_path": None},
                  "contact_forces": {"class_type": "x:ParkourHexContactSensor"}},
        "episode_length_s": 40.0,
        "rewards": rew,
        "curriculum": curriculum,
        "parkours": {"base_parkour": pk},
        "omni_command": True,
    }
    return yaml.dump(doc, default_flow_style=False, sort_keys=False)


def phaseout_curriculum(spec: str) -> dict:
    out = {}
    for chunk in spec.split(","):
        term, w0, w1, t0, t1 = chunk.split(":")
        out[f"phaseout_{term}"] = {"func": "parkour_tasks.crab_hex_forward_task.mdp.curriculums:ramp_reward_weight",
                                   "params": {"term_name": term, "w0": float(w0), "w1": float(w1),
                                              "t0": int(t0), "t1": int(t1)}}
    return out


def _check(rc, key):
    return rc.OVERLAY_DUMP_CHECKS[key]


# ============================================================ (4) whitelist / parsers / audit coverage
def test_every_phase2_key_is_whitelisted_parsed_and_audited(rc):
    whitelist, parsers = rc.knob_whitelist(), rc.knob_parsers()
    for key in P2_KEYS:
        assert key in whitelist, key
        assert key in parsers, key
        assert key in rc.OVERLAY_DUMP_CHECKS, key
        assert key not in rc.AGENT_DUMP_KEYS, f"{key} is env side (params/env.yaml)"
    # the field parsers ARE the knob module's (no driver copy to drift)
    tk = rc.knobs()._tk
    for var, parse in tk.FIELD_KNOB_PARSERS.items():
        assert parsers[var] is parse, var


@pytest.mark.parametrize("key, good, bad", [
    ("KRABBY_FLAT_TERRAIN_GEOM", ["ring", "field", " Field "], ["", "corridor", "rings"]),
    ("KRABBY_TERRAIN_PROMOTE", ["0.45:0.25", "0.75:0.4"], ["", "0.45", "0.25:0.45", "0.4:0.4", "nan:0.1",
                                                           "-0.1:-0.2", "a:b", "0.45:0.25:0.1"]),
    ("KRABBY_FIELD_PAD_R", ["1.7", "1.6"], ["1.5", "", "3.8", "x"]),
    ("KRABBY_FIELD_BLOCKS", ["14:28", "0:60"], ["14", "14.5:28", "14:61"]),
    ("KRABBY_FIELD_KINDS", ["0.4:0.4:0.2", "1:0:0"], ["0.4:0.4", "0.5:0.5:0.5"]),
    ("KRABBY_FIELD_EDGES", ["ramp", "wall"], ["vertical", ""]),
    ("KRABBY_STRIDE_W", ["0.5", "0.001"], ["", "x"]),
    ("KRABBY_STAND_STANCE_W", ["0.7", "0", ""], ["-0.5"]),
    ("KRABBY_POWER_JOINTS", ["actuated", "all", ""], ["rods"]),
])
def test_phase2_knob_round_trips(rc, key, good, bad):
    whitelist, parsers = rc.knob_whitelist(), rc.knob_parsers()
    for value in good:
        reasons = []
        rc._check_knob(key, value, "x", whitelist, parsers, reasons, set())
        assert reasons == [], (value, reasons)
    for value in bad:
        reasons = []
        rc._check_knob(key, value, "x", whitelist, parsers, reasons, set())
        assert any("does not round-trip" in r for r in reasons), (value, reasons)


@pytest.mark.skipif(not P2_CANDIDATES.exists(), reason="the phase-2 campaign scaffold is not in this tree")
def test_the_phase2_campaign_file_validates_except_its_named_placeholders(rc):
    """The real campaign spec (plan step 1): every refusal left is a PLACEHOLDER (the promotion pair and
    the replay-gate weights, filled later) -- no blank-PHASEOUT round trip, no missing dump check, no
    geometry composition refusal."""
    verdicts = rc.validate_candidates(rc.load_candidates(P2_CANDIDATES), strict=False, check_replay=False)
    reasons = [r for v in verdicts for r in v["reasons"]]
    assert reasons, "the scaffold still carries placeholders; this test expects to see them refused"
    assert all("is a PLACEHOLDER" in r for r in reasons), [r for r in reasons if "PLACEHOLDER" not in r]


def test_field_knobs_under_ring_are_refused_at_dry_run(rc):
    env = {"KRABBY_FLAT_TERRAIN_GEOM": "ring", "KRABBY_FIELD_PAD_R": "1.7"}
    reasons = []
    rc._check_field_geom_composition(env, "x", reasons)
    assert len(reasons) == 1 and "KRABBY_FIELD_PAD_R" in reasons[0]
    reasons = []
    rc._check_field_geom_composition({"KRABBY_FLAT_TERRAIN_GEOM": "field", "KRABBY_RING_BLOCKS": "8:16"},
                                     "x", reasons)
    assert len(reasons) == 1 and "KRABBY_RING_BLOCKS" in reasons[0]
    for ok_env in ({"KRABBY_FLAT_TERRAIN_GEOM": "field", **{k: v for k, v in P2_FIELD.items()}},
                   {"KRABBY_FLAT_TERRAIN_GEOM": "ring", "KRABBY_RING_BLOCKS": "8:16"},
                   {"KRABBY_FLAT_TERRAIN_GEOM": "ring", "KRABBY_FIELD_PAD_R": ""}):
        reasons = []
        rc._check_field_geom_composition(ok_env, "x", reasons)
        assert reasons == [], (ok_env, reasons)


# ============================================================ (1) blank KRABBY_PHASEOUT
def test_blank_phaseout_round_trips_to_none_and_a_spec_still_parses(rc):
    parse = rc.knob_parsers()["KRABBY_PHASEOUT"]
    assert parse("") is None
    assert [e["term_name"] for e in parse(PRESET_2A_PHASEOUT)] == [
        "reward_clock_swing_apex", "reward_feet_air_time_positive", "reward_stride_length"]
    for bad in (" ", "\t", "reward_clock_schedule:1.0:0.001", "reward_clock_schedule:1:0.001:10:5"):
        with pytest.raises(ValueError):
            parse(bad)


def test_blank_phaseout_dump_check_pins_curriculum_null_and_no_phaseout_block(rc):
    ok = _check(rc, "KRABBY_PHASEOUT")
    assert ok(env_dump(curriculum=None), "")
    # the un-blanked 2a preset: its three ramps reached the cfg
    preset = env_dump(curriculum=phaseout_curriculum(PRESET_2A_PHASEOUT))
    assert not ok(preset, "")
    # any non-null curriculum (another manager term) is not "never built"
    other = env_dump(curriculum={"terrain_levels": {"func": "x:terrain_levels", "params": {}}})
    assert not ok(other, "")
    # ... and the non-blank check is unchanged on the same dumps
    assert ok(preset, PRESET_2A_PHASEOUT)
    assert not ok(env_dump(curriculum=None), PRESET_2A_PHASEOUT)


def _stage(rc, overlay: dict, base: str = "2a") -> list:
    return [("p2a", 5000, rc._effective_stage_cfg({}, {"base": base}, overlay))]


def test_clock_rule_ii_passes_a_blank_phaseout_and_refuses_the_unblanked_preset_apex(rc):
    hidden = {"KRABBY_CLOCK_W": "0.001", "KRABBY_APEX_W": "0.001", "KRABBY_OMNI_CLOCK_HIDE": "1"}
    reasons = []
    rc._check_clock_chain(_stage(rc, {**hidden, "KRABBY_PHASEOUT": ""}), "x", reasons)
    assert reasons == []
    st = rc.clock_stage_state(_stage(rc, {**hidden, "KRABBY_PHASEOUT": ""})[0][2], 5000)
    assert st["phaseout_terms"] == [] and st["max_w"] == {"reward_clock_schedule": 0.001,
                                                           "reward_clock_swing_apex": 0.001}
    # a forgotten blank: the 2a preset's PHASEOUT (apex 1.0 -> 0.5) survives the overlay
    reasons = []
    rc._check_clock_chain(_stage(rc, hidden), "x", reasons)
    assert any("(ii)" in r and "reward_clock_swing_apex" in r for r in reasons), reasons


def test_a_blank_phaseout_stage_is_not_a_ramp_stage(rc):
    assert not rc.ramp_stage({"KRABBY_PHASEOUT": ""})
    assert rc.ramp_stage({"KRABBY_PHASEOUT": PRESET_2A_PHASEOUT})


# ============================================================ (2) the dump checks, pinned
def test_geometry_check_reads_the_sub_terrain_functions_and_the_radius_sync(rc):
    ok = _check(rc, "KRABBY_FLAT_TERRAIN_GEOM")
    field, ring = env_dump("field"), env_dump("ring")
    assert ok(field, "field") and not ok(field, "ring")
    assert ok(ring, "ring") and not ok(ring, "field")
    assert rc.dump_terrain_geom(field) == "field" and rc.dump_terrain_geom(ring) == "ring"
    # mixed / unknown / no generator: neither
    mixed = env_dump(subs={"parkour_flat": _sub("flat"), "omni_ring_sparse": _sub("ring", clear_radius=1.7,
                                                                                    ring_outer_radius=4.8),
                           "omni_field_dense": _sub("field")})
    assert rc.dump_terrain_geom(mixed) is None and not ok(mixed, "field") and not ok(mixed, "ring")
    odd = env_dump(subs={"parkour_flat": _sub("flat"), "stairs": {**_sub("field"), "function": "x:stairs_terrain"}})
    assert not ok(odd, "field")
    assert rc.dump_terrain_geom("rewards: {}\n") is None
    # the parkour term was not synced to the terrain (reach_edge would be measured on the ring's disc)
    assert not ok(env_dump("field", parkour={"clear_radius_m": 2.0}), "field")
    assert not ok(env_dump("field", parkour={"ring_outer_radius_m": 4.5}), "field")
    # a flat tile left on the ring's radii (sector anchors would differ between tile types)
    stale_flat = env_dump(subs={"parkour_flat": _sub("flat", clear_radius=2.0),
                                "omni_field_sparse": _sub("field"), "omni_field_dense": _sub("field")})
    assert not ok(stale_flat, "field")


def test_pad_radius_check(rc):
    ok = _check(rc, "KRABBY_FIELD_PAD_R")
    assert ok(env_dump("field"), "1.7")
    assert not ok(env_dump("field"), "1.8")
    pad18 = {"parkour_flat": _sub("flat", clear_radius=1.8), "omni_field_sparse": _sub("field", clear_radius=1.8),
             "omni_field_dense": _sub("field", clear_radius=1.8)}
    assert ok(env_dump(subs=pad18), "1.8")
    assert not ok(env_dump(subs=pad18, parkour={"clear_radius_m": 1.7}), "1.8"), "term not synced"
    one_left = copy.deepcopy(pad18)
    one_left["parkour_flat"]["clear_radius"] = 1.7
    assert not ok(env_dump(subs=one_left, parkour={"clear_radius_m": 1.8}), "1.8"), "flat tile left behind"
    assert not ok(env_dump("ring"), "1.7"), "a ring dump never passes a field check"
    # (2.0 is below the parser's bound only because 2.0 >= 1.6; the ring's own pad still fails)
    assert not ok(env_dump("ring"), "2.0"), "a ring dump never passes a field check"


def test_blocks_kinds_edges_min_gap_size_scale_checks(rc):
    field, ring = env_dump("field"), env_dump("ring")
    blocks, kinds, edges = (_check(rc, k) for k in ("KRABBY_FIELD_BLOCKS", "KRABBY_FIELD_KINDS",
                                                     "KRABBY_FIELD_EDGES"))
    gap, scale = _check(rc, "KRABBY_FIELD_MIN_GAP"), _check(rc, "KRABBY_FIELD_SIZE_SCALE")
    assert blocks(field, "14:28") and not blocks(field, "28:14") and not blocks(field, "14:27")
    assert not blocks(ring, "8:16"), "ring sub-terrains carry n_blocks too: still not a field"
    assert kinds(field, "0.4:0.4:0.2") and not kinds(field, "0.5:0.3:0.2") and not kinds(field, "0.4:0.5:0.1")
    assert not kinds(ring, "0.5:0.5:0.0")
    assert edges(field, "ramp") and not edges(field, "wall") and not edges(ring, "ramp")
    wall = env_dump(subs={"parkour_flat": _sub("flat"), "omni_field_sparse": _sub("field", vertical_edges=True),
                          "omni_field_dense": _sub("field", vertical_edges=True)})
    assert edges(wall, "wall") and not edges(wall, "ramp")
    half = env_dump(subs={"parkour_flat": _sub("flat"), "omni_field_sparse": _sub("field", vertical_edges=True),
                          "omni_field_dense": _sub("field")})
    assert not edges(half, "wall") and not edges(half, "ramp"), "EVERY field sub-terrain"
    assert gap(field, "0") and not gap(field, "0.2") and not gap(ring, "0")
    scaled = {k: (lo * 0.5, hi * 0.5) for k, (lo, hi) in rc.FIELD_SIZE_RANGES.items()}
    small = env_dump(subs={"parkour_flat": _sub("flat"), "omni_field_sparse": _sub("field", **scaled),
                           "omni_field_dense": _sub("field", **scaled)})
    # the two sub-terrains SHARE the scaled tuples, so the dump carries YAML anchors / aliases (as real
    # omni teacher dumps do for shared objects): the checks must resolve them, not read "*id001"
    assert "*id0" in small
    assert scale(small, "0.5") and not scale(field, "0.5") and scale(field, "1.0") and not scale(ring, "1.0")
    # three field sub-terrains: the sparse:dense pair has no meaning
    three = env_dump(subs={"parkour_flat": _sub("flat"), "a": _sub("field"), "b": _sub("field"), "c": _sub("field")})
    assert not blocks(three, "14:14")


def test_promotion_pair_check(rc):
    ok = _check(rc, "KRABBY_TERRAIN_PROMOTE")
    assert ok(env_dump(), "0.45:0.25") and not ok(env_dump(), "0.45:0.2") and not ok(env_dump(), "0.5:0.25")
    cal = env_dump(parkour={"move_up_frac": 0.75, "move_down_frac": 0.4})
    assert ok(cal, "0.75:0.4") and not ok(cal, "0.45:0.25")
    assert not ok(env_dump(parkour={"move_up_frac": None}), "0.45:0.25")


def test_weight_checks_stride_airtime_power(rc):
    dump = env_dump(rewards={"penalty_mechanical_power": {
        "func": "parkour_isaaclab.envs.mdp.rewards:penalty_mechanical_power",
        "params": {"asset_cfg": _entity()}, "weight": -0.0002}})
    assert _check(rc, "KRABBY_STRIDE_W")(dump, "0.5") and not _check(rc, "KRABBY_STRIDE_W")(dump, "0.25")
    assert _check(rc, "KRABBY_AIRTIME_W")(dump, "0.8") and not _check(rc, "KRABBY_AIRTIME_W")(dump, "0.4")
    assert _check(rc, "KRABBY_POWER_W")(dump, "-0.0002") and not _check(rc, "KRABBY_POWER_W")(dump, "0")
    # the stride check reads ITS term's weight, not the nearest one (airtime is 0.8 above it)
    assert not _check(rc, "KRABBY_STRIDE_W")(dump, "0.8")


STAND_TERM = {"func": "parkour_isaaclab.envs.mdp.rewards:reward_stand_stance_quality",
              "params": {"asset_cfg": _entity(body_names=[".*_Footpad"]), "command_name": "base_velocity",
                         "vel_ref": 0.05, "min_upright_gz": 0.9}, "weight": 0.7}


def test_stand_stance_w_check(rc):
    ok = _check(rc, "KRABBY_STAND_STANCE_W")
    absent, present = env_dump(), env_dump(rewards={"reward_stand_stance_quality": STAND_TERM})
    for off in ("0", "", "0.0"):
        assert ok(absent, off) and not ok(present, off), off
    assert ok(present, "0.7") and not ok(present, "0.5") and not ok(absent, "0.7")
    wrong_func = env_dump(rewards={"reward_stand_stance_quality": {**STAND_TERM,
                                                                   "func": "x:reward_stand_stance_quality_v2"}})
    assert not ok(wrong_func, "0.7")


def _power(joint_names, weight=-0.0002) -> dict:
    return {"penalty_mechanical_power": {"func": "parkour_isaaclab.envs.mdp.rewards:penalty_mechanical_power",
                                         "params": {"asset_cfg": _entity(joint_names=joint_names)},
                                         "weight": weight}}


def test_power_joints_check(rc):
    ok = _check(rc, "KRABBY_POWER_JOINTS")
    assert rc.actuated_joint_names() == ACTUATED, "the forward cfg's actuated list moved"
    every, actuated = env_dump(rewards=_power(None)), env_dump(rewards=_power(list(ACTUATED)))
    for off in ("all", "", "ALL"):
        assert ok(every, off) and not ok(actuated, off), off
    assert ok(actuated, "actuated") and not ok(every, "actuated")
    assert not ok(env_dump(rewards=_power(ACTUATED[1:])), "actuated"), "a joint missing"
    assert not ok(env_dump(rewards=_power(ACTUATED + [".*_Body_Hip_RevoluteJoint"])), "actuated")
    assert not ok(env_dump(rewards=_power(list(ACTUATED), weight=0.0)), "actuated"), "prices nothing"
    assert not ok(env_dump(rewards={"penalty_mechanical_power": None}), "all"), "no term at all"


def test_overlay_checks_end_to_end_on_a_field_dump_and_a_ring_dump(rc, tmp_path):
    def run(text):
        run_dir = tmp_path / "run" / "params"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "env.yaml").write_text(text)
        return rc.overlay_checks({"ckpt": str(tmp_path / "run" / "model_19995.pt")}, P2_FIELD)

    good = run(env_dump("field", curriculum=None))
    assert len(good) == len(P2_FIELD) and all(ok for _n, ok in good), good
    bad = dict(run(env_dump("ring", curriculum=phaseout_curriculum(PRESET_2A_PHASEOUT))))
    failed = {name.split("=")[0].removeprefix("overlay ") for name, ok in bad.items() if not ok}
    # the promotion pair is the same on both dumps; every geometry / field key and the blank fail
    assert failed == set(P2_FIELD) - {"KRABBY_TERRAIN_PROMOTE"}, failed


@pytest.mark.skipif(not C6X_DUMP.exists(), reason="the B head's c6x run directory is not on this machine")
def test_the_ring_checks_read_a_real_c6x_dump(rc):
    """Indent / layout check against what the trainer really wrote (B s18 c6x: ring, 0.45:0.25,
    airtime 0.8 / stride 0.5, all-joint power, no stand term, no curriculum)."""
    txt = C6X_DUMP.read_text()
    assert rc.dump_terrain_geom(txt) == "ring"
    assert _check(rc, "KRABBY_FLAT_TERRAIN_GEOM")(txt, "ring")
    assert not _check(rc, "KRABBY_FLAT_TERRAIN_GEOM")(txt, "field")
    assert _check(rc, "KRABBY_TERRAIN_PROMOTE")(txt, "0.45:0.25")
    assert _check(rc, "KRABBY_STRIDE_W")(txt, "0.5") and _check(rc, "KRABBY_AIRTIME_W")(txt, "0.8")
    assert _check(rc, "KRABBY_POWER_JOINTS")(txt, "all") and not _check(rc, "KRABBY_POWER_JOINTS")(txt, "actuated")
    assert _check(rc, "KRABBY_STAND_STANCE_W")(txt, "0")
    assert _check(rc, "KRABBY_PHASEOUT")(txt, "")
    for key, value in (("KRABBY_FIELD_PAD_R", "1.7"), ("KRABBY_FIELD_BLOCKS", "14:28"),
                       ("KRABBY_FIELD_KINDS", "0.4:0.4:0.2"), ("KRABBY_FIELD_EDGES", "ramp")):
        assert not _check(rc, key)(txt, value), key


# ============================================================ (3) the geometry-aware phase check
RING_CHECK = ("ring generator installed (omni_ring_sparse/dense sub-terrain) in params/env.yaml", False)


@pytest.fixture
def phase_ctx(rc, monkeypatch):
    """``_filtered_phase_checks`` with ``run_phases.phase_checks`` stubbed to its ring check (as
    run_phases computes it on the record's dump) plus the two stage-keyed checks it drops."""
    def fake_phase_checks(rec, spec, c):
        txt = (Path(rec["ckpt"]).parent / "params" / "env.yaml").read_text()
        return [("plant is A15+B (params/env.yaml usd path)", True),
                (RING_CHECK[0], "omni_ring_sparse" in txt or "omni_ring_dense" in txt),
                ("dir_band matches stage 2a (full range)", False),
                ("RSI source matches stage 2a (all)", False)]
    monkeypatch.setattr(rc.rp, "phase_checks", fake_phase_checks)
    return SimpleNamespace(c=None)


def _rec(tmp_path, dump: str, overlay: dict) -> dict:
    params = tmp_path / "run" / "params"
    params.mkdir(parents=True, exist_ok=True)
    (params / "env.yaml").write_text(dump)
    return {"ckpt": str(tmp_path / "run" / "model_19995.pt"), "overlay": overlay}


def test_a_field_record_gets_the_field_check_not_a_false_ring_fail(rc, phase_ctx, tmp_path):
    spec = SimpleNamespace(name="2a")
    out = rc._filtered_phase_checks(phase_ctx, _rec(tmp_path, env_dump("field"), dict(P2_FIELD)), spec)
    names = [n for n, _ok in out]
    assert not any(n.startswith("ring generator installed") for n in names)
    assert not any(n.startswith(("dir_band matches", "RSI source matches")) for n in names)
    field = [(n, ok) for n, ok in out if n.startswith("field generator installed")]
    assert len(field) == 1 and field[0][1] is True, out
    assert all(ok for _n, ok in out), out


def test_a_field_record_whose_dump_shows_a_ring_fails(rc, phase_ctx, tmp_path):
    """The geometry comes from the LAUNCH record, never from the dump the check reads."""
    out = dict(rc._filtered_phase_checks(phase_ctx, _rec(tmp_path, env_dump("ring"), dict(P2_FIELD)),
                                         SimpleNamespace(name="2a")))
    assert [ok for n, ok in out.items() if n.startswith("field generator installed")] == [False]


def test_a_field_dump_missing_one_field_key_fails(rc, phase_ctx, tmp_path):
    one = env_dump(subs={"parkour_flat": _sub("flat"), "omni_field_sparse": _sub("field")})
    out = dict(rc._filtered_phase_checks(phase_ctx, _rec(tmp_path, one, dict(P2_FIELD)), SimpleNamespace(name="2a")))
    assert [ok for n, ok in out.items() if n.startswith("field generator installed")] == [False]


@pytest.mark.parametrize("overlay", [{"KRABBY_FLAT_TERRAIN_GEOM": "ring"}, {}])
def test_a_ring_record_keeps_the_run_phases_check_verbatim(rc, phase_ctx, tmp_path, overlay):
    """A ring overlay -- or none at all (the 2a preset's OMNI_COMMON ring) -- reads exactly as before."""
    for dump, want in ((env_dump("ring"), True), (env_dump("field"), False)):
        out = rc._filtered_phase_checks(phase_ctx, _rec(tmp_path, dump, overlay), SimpleNamespace(name="2a"))
        assert (RING_CHECK[0], want) in out, out
        assert not any(n.startswith("field generator installed") for n, _ok in out)


# ============================================================ constants pinned to their sources
def test_driver_constants_match_the_generator_sources(rc):
    field_src = (TERRAINS / "config/omni_field.py").read_text()
    ring_src = (TERRAINS / "config/omni_ring.py").read_text()
    for key in rc.FIELD_SUB_TERRAIN_KEYS:
        assert f'"{key}": OmniFieldTerrainCfg(' in field_src, key
    assert field_src.index(rc.FIELD_SUB_TERRAIN_KEYS[0]) < field_src.index(rc.FIELD_SUB_TERRAIN_KEYS[1])
    for key in rc.RING_SUB_TERRAIN_KEYS:
        assert f'"{key}": OmniRingTerrainCfg(' in ring_src, key
    terrains_src = (TERRAINS / "omni_terrains.py").read_text()
    for func in rc.OMNI_SUB_TERRAIN_KINDS:
        assert f"def {func}(" in terrains_src, func
    cfg_src = (TERRAINS / "omni_terrains_cfg.py").read_text()
    field_cls = cfg_src[cfg_src.index("class OmniFieldTerrainCfg"):]
    for attr, (lo, hi) in rc.FIELD_SIZE_RANGES.items():
        assert f"{attr}: tuple[float, float] = ({lo}, {hi})" in field_cls, attr
    assert rc.TERRAIN_GEOM_VAR == rc.knobs()._tk.GEOM_VAR


def test_run_candidates_is_outside_both_sha_pathspecs(rc):
    """The driver is a TOOL file: editing it moves neither code_sha nor eval_sha (plan ground rule)."""
    rel = str(RUN_CANDIDATES.relative_to(REPO))
    for spec in tuple(rc.CODE_PATHSPECS) + tuple(rc.EVAL_PATHSPECS):
        assert not (rel == spec or rel.startswith(spec.rstrip("/") + "/")), spec
