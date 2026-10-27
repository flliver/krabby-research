"""Ring-terrain knobs and the ring-generator guard (plan D8 / spec B7), pure, no Isaac.

Two silent failures this guards, both of which produce a run that LOOKS fine:

* a forward terrain generator installed under an omni env cfg -- the omni command term,
  the ring spawn and the radial exposure would all run against a corridor, and every
  corridor geometry helper in ``apply_flat_walk_knobs`` is keyed on corridor sub-terrain
  names, so nothing complains;
* a leaked ``KRABBY_FLAT_TERRAIN_GEOM`` from a forward/sideways preset -- the omni presets
  set it to ``ring`` in every window, so any other value means the process inherited a
  campaign it was not asked to run.

The module keeps its cfg classes as parameters precisely so this file can pass stand-ins
and still exercise the shipped code path, ``apply_omni_ring_knobs`` included.
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
OMNI_MDP = REPO / "parkour" / "parkour_tasks" / "parkour_tasks" / "crab_hex_omni_task" / "mdp"
if str(OMNI_MDP) not in sys.path:
    sys.path.insert(0, str(OMNI_MDP))

import omni_terrain_knobs as tk  # noqa: E402


# --------------------------------------------------------------------------- stand-ins


class RingCfg:
    def __init__(self, n_blocks=8, clear_radius=2.0, ring_outer_radius=4.5):
        self.n_blocks = n_blocks
        self.clear_radius = clear_radius
        self.ring_outer_radius = ring_outer_radius


class FlatCfg:
    def __init__(self, clear_radius=2.0, ring_outer_radius=4.5):
        self.clear_radius = clear_radius
        self.ring_outer_radius = ring_outer_radius


class CorridorCfg:
    """Stands in for ExtremeParkourHurdleTerrainCfg & friends."""


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _ring_generator():
    return _Obj(sub_terrains={
        "parkour_flat": FlatCfg(),
        "omni_ring_sparse": RingCfg(n_blocks=8),
        "omni_ring_dense": RingCfg(n_blocks=16),
    })


def _forward_generator():
    return _Obj(sub_terrains={"parkour_flat": CorridorCfg(), "parkour_hurdle": CorridorCfg()})


def _cfg(generator):
    return _Obj(
        scene=_Obj(terrain=_Obj(terrain_generator=generator)),
        parkours=_Obj(base_parkour=_Obj(clear_radius_m=2.0, ring_outer_radius_m=4.5)),
    )


RING_ENV = {tk.GEOM_VAR: "ring"}


# ----------------------------------------------------------------------------- parsers


@pytest.mark.parametrize("text,want", [("2.0", 2.0), (" 2.5 ", 2.5), ("1", 1.0), ("0.5", 0.5)])
def test_parse_clear_radius(text, want):
    assert tk.parse_clear_radius(text) == pytest.approx(want)


@pytest.mark.parametrize("text", ["", "abc", "0", "-1", "nan", "inf", "2:3"])
def test_parse_clear_radius_rejects(text):
    with pytest.raises(ValueError):
        tk.parse_clear_radius(text)


@pytest.mark.parametrize("text,want", [("8:16", (8, 16)), (" 4 : 32 ", (4, 32)), ("0:0", (0, 0))])
def test_parse_ring_blocks(text, want):
    assert tk.parse_ring_blocks(text) == want


@pytest.mark.parametrize("text", ["8", "8:16:24", "8.5:16", "-1:16", "a:b", ""])
def test_parse_ring_blocks_rejects(text):
    with pytest.raises(ValueError):
        tk.parse_ring_blocks(text)


def test_parse_errors_name_the_variable_and_the_value():
    with pytest.raises(ValueError, match=tk.BLOCKS_VAR):
        tk.parse_ring_blocks("8")
    with pytest.raises(ValueError, match=tk.CLEAR_R_VAR):
        tk.parse_clear_radius("nope")


# ------------------------------------------------------------------------- geom variable


def test_require_ring_geom_accepts_ring():
    assert tk.require_ring_geom({tk.GEOM_VAR: "Ring "}) == "ring"


def test_require_ring_geom_names_the_leaked_value():
    with pytest.raises(RuntimeError, match="recal2b2w"):
        tk.require_ring_geom({tk.GEOM_VAR: "recal2b2w"})
    with pytest.raises(RuntimeError, match="<unset>"):
        tk.require_ring_geom({})


# -------------------------------------------------------------------- generator identity


def test_classify_splits_by_class_not_by_name():
    ring, flat, foreign = tk.classify_sub_terrains(_ring_generator().sub_terrains, RingCfg, FlatCfg)
    assert ring == ["omni_ring_sparse", "omni_ring_dense"]
    assert flat == ["parkour_flat"]  # the load-bearing shared key, kept
    assert foreign == []


def test_assert_ring_generator_accepts_the_ring():
    assert tk.assert_ring_generator(_ring_generator(), RingCfg, FlatCfg) == (
        ["omni_ring_sparse", "omni_ring_dense"], ["parkour_flat"]
    )


def test_assert_ring_generator_rejects_a_forward_generator():
    with pytest.raises(RuntimeError, match="parkour_hurdle=CorridorCfg"):
        tk.assert_ring_generator(_forward_generator(), RingCfg, FlatCfg)


def test_assert_ring_generator_rejects_none_and_empty():
    with pytest.raises(RuntimeError, match="OMNI_RING_TERRAINS_CFG"):
        tk.assert_ring_generator(None, RingCfg, FlatCfg)
    with pytest.raises(RuntimeError, match="no sub_terrains"):
        tk.assert_ring_generator(_Obj(sub_terrains={}), RingCfg, FlatCfg)


def test_assert_ring_generator_rejects_an_all_flat_generator():
    gen = _Obj(sub_terrains={"parkour_flat": FlatCfg()})
    with pytest.raises(RuntimeError, match="at least one RingCfg"):
        tk.assert_ring_generator(gen, RingCfg, FlatCfg)


# ------------------------------------------------------------------------- application


def test_apply_ring_geometry_writes_blocks_in_generator_order():
    subs = _ring_generator().sub_terrains
    tk.apply_ring_geometry(subs, RingCfg, FlatCfg, blocks=(3, 30))
    assert subs["omni_ring_sparse"].n_blocks == 3
    assert subs["omni_ring_dense"].n_blocks == 30


def test_apply_ring_geometry_writes_clear_radius_to_flat_tiles_too():
    """The flat tiles place the same sector anchors at (clear + outer)/2."""
    subs = _ring_generator().sub_terrains
    tk.apply_ring_geometry(subs, RingCfg, FlatCfg, clear_radius=2.75)
    assert all(s.clear_radius == pytest.approx(2.75) for s in subs.values())


def test_apply_ring_geometry_rejects_a_disc_that_swallows_the_annulus():
    subs = _ring_generator().sub_terrains
    with pytest.raises(ValueError, match="ring_outer_radius"):
        tk.apply_ring_geometry(subs, RingCfg, FlatCfg, clear_radius=4.5)


def test_apply_ring_geometry_needs_exactly_two_ring_sub_terrains_for_a_pair():
    subs = {"parkour_flat": FlatCfg(), "omni_ring_sparse": RingCfg()}
    with pytest.raises(RuntimeError, match="exactly two ring"):
        tk.apply_ring_geometry(subs, RingCfg, FlatCfg, blocks=(8, 16))


def test_apply_ring_geometry_is_a_no_op_without_knobs():
    subs = _ring_generator().sub_terrains
    tk.apply_ring_geometry(subs, RingCfg, FlatCfg)
    assert subs["omni_ring_sparse"].n_blocks == 8 and subs["omni_ring_dense"].n_blocks == 16
    assert subs["parkour_flat"].clear_radius == pytest.approx(2.0)


# ------------------------------------------------------------------ apply_omni_ring_knobs


def test_apply_omni_ring_knobs_end_to_end():
    cfg = _cfg(_ring_generator())
    env = dict(RING_ENV, **{tk.CLEAR_R_VAR: "2.5", tk.BLOCKS_VAR: "6:20"})
    tk.apply_omni_ring_knobs(cfg, env, classes=(RingCfg, FlatCfg))
    subs = cfg.scene.terrain.terrain_generator.sub_terrains
    assert subs["omni_ring_sparse"].n_blocks == 6 and subs["omni_ring_dense"].n_blocks == 20
    assert all(s.clear_radius == pytest.approx(2.5) for s in subs.values())
    # the parkour term's copy of the disc follows the terrain
    assert cfg.parkours.base_parkour.clear_radius_m == pytest.approx(2.5)
    assert cfg.parkours.base_parkour.ring_outer_radius_m == pytest.approx(4.5)


def test_apply_omni_ring_knobs_syncs_a_mismatched_parkour_term(capsys):
    cfg = _cfg(_ring_generator())
    cfg.parkours.base_parkour.clear_radius_m = 1.0  # would silently redefine reach_edge
    tk.apply_omni_ring_knobs(cfg, dict(RING_ENV), classes=(RingCfg, FlatCfg))
    assert cfg.parkours.base_parkour.clear_radius_m == pytest.approx(2.0)
    assert "did not match the terrain" in capsys.readouterr().out


def test_apply_omni_ring_knobs_is_quiet_when_everything_agrees(capsys):
    tk.apply_omni_ring_knobs(_cfg(_ring_generator()), dict(RING_ENV), classes=(RingCfg, FlatCfg))
    assert capsys.readouterr().out == ""


def test_apply_omni_ring_knobs_checks_the_geometry_variable_first():
    """A forward preset must produce the preset message, not a class assertion."""
    cfg = _cfg(_forward_generator())
    with pytest.raises(RuntimeError, match=tk.GEOM_VAR):
        tk.apply_omni_ring_knobs(cfg, {tk.GEOM_VAR: "shallow"}, classes=(RingCfg, FlatCfg))


def test_apply_omni_ring_knobs_rejects_a_forward_generator():
    cfg = _cfg(_forward_generator())
    with pytest.raises(RuntimeError, match="CorridorCfg"):
        tk.apply_omni_ring_knobs(cfg, dict(RING_ENV), classes=(RingCfg, FlatCfg))


def test_apply_omni_ring_knobs_tolerates_a_cfg_without_a_parkour_term():
    cfg = _Obj(scene=_Obj(terrain=_Obj(terrain_generator=_ring_generator())), parkours=None)
    tk.apply_omni_ring_knobs(cfg, dict(RING_ENV), classes=(RingCfg, FlatCfg))


def test_apply_omni_ring_knobs_rejects_a_bad_knob_value():
    cfg = _cfg(_ring_generator())
    with pytest.raises(ValueError):
        tk.apply_omni_ring_knobs(
            cfg, dict(RING_ENV, **{tk.BLOCKS_VAR: "8"}), classes=(RingCfg, FlatCfg)
        )


def test_real_cfg_classes_satisfy_the_guard():
    """The shipped OMNI_RING_TERRAINS_CFG must pass its own generator assertion.

    Skipped outside a Kit process (the cfg classes are ``@configclass``, whose import chain
    reaches ``pxr``); under Isaac this is the check that the terrain package and this guard
    agree about which classes exist. The Isaac delta test (Part E) covers it there.
    """
    try:
        from parkour_isaaclab.terrains.omni.config.omni_ring import OMNI_RING_TERRAINS_CFG
        from parkour_isaaclab.terrains.omni.omni_terrains_cfg import (
            OmniFlatTerrainCfg,
            OmniRingTerrainCfg,
        )
    except ModuleNotFoundError as exc:  # no pxr / no Kit
        pytest.skip(f"isaaclab terrain cfgs need a Kit process: {exc}")

    ring_keys, flat_keys = tk.assert_ring_generator(
        OMNI_RING_TERRAINS_CFG, OmniRingTerrainCfg, OmniFlatTerrainCfg
    )
    assert ring_keys == ["omni_ring_sparse", "omni_ring_dense"]
    assert flat_keys == ["parkour_flat"]
