"""Omni phase-2 plan, step 2 (2026-10-06): the field-geometry knobs, the install hook and the
ring/field guard in ``crab_hex_omni_task/mdp/omni_terrain_knobs.py`` -- pure, no Isaac.

Silent failures this file guards (each produces a run that LOOKS fine):

* ``KRABBY_FLAT_TERRAIN_GEOM=field`` with the ring still installed (the install hook did not run,
  or ran after the knob block) -- the run would train on the ring while every record says field;
* a ``KRABBY_RING_*`` knob under ``field``, or a ``KRABBY_FIELD_*`` knob under ``ring`` -- each knob
  family only knows its own sub-terrain class, so the knob would apply nothing;
* the parkour term left on the ring's 2.0 m disc under the 1.7 m pad -- ``reach_edge`` / ``field_frac``
  would quietly mean something else;
* the field generator shared (not deep-copied) between env cfgs -- one cfg's knobs leaking into the
  next;
* the knob whitelist (``omni_knobs`` ``*_VAR`` names, which the campaign driver reads) missing a
  field knob, so a valid overlay is refused -- or a typo'd one accepted.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
OMNI = REPO / "parkour" / "parkour_tasks" / "parkour_tasks" / "crab_hex_omni_task"
OMNI_MDP = OMNI / "mdp"
if str(OMNI_MDP) not in sys.path:
    sys.path.insert(0, str(OMNI_MDP))

import omni_terrain_knobs as tk  # noqa: E402


def _load(path: Path, name: str):
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- stand-ins


class RingCfg:
    def __init__(self, n_blocks=8, clear_radius=2.0, ring_outer_radius=4.5):
        self.n_blocks = n_blocks
        self.clear_radius = clear_radius
        self.ring_outer_radius = ring_outer_radius


class FieldCfg:
    def __init__(self, n_blocks=14, clear_radius=1.7, ring_outer_radius=4.80):
        self.n_blocks = n_blocks
        self.clear_radius = clear_radius
        self.ring_outer_radius = ring_outer_radius
        self.bar_fraction = 0.4
        self.pit_fraction = 0.2
        self.vertical_edges = False
        self.min_gap = 0.0
        self.bar_length_range = (1.0, 2.0)
        self.bar_thickness_range = (0.16, 0.32)
        self.slab_side_range = (0.6, 1.2)


class FlatCfg:
    def __init__(self, clear_radius=2.0, ring_outer_radius=4.5):
        self.clear_radius = clear_radius
        self.ring_outer_radius = ring_outer_radius


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _ring_generator():
    return _Obj(sub_terrains={
        "parkour_flat": FlatCfg(),
        "omni_ring_sparse": RingCfg(n_blocks=8),
        "omni_ring_dense": RingCfg(n_blocks=16),
    })


def _field_generator():
    return _Obj(sub_terrains={
        "parkour_flat": FlatCfg(clear_radius=1.7, ring_outer_radius=4.80),
        "omni_field_sparse": FieldCfg(n_blocks=14),
        "omni_field_dense": FieldCfg(n_blocks=28),
    })


def _cfg(generator):
    return _Obj(
        scene=_Obj(terrain=_Obj(terrain_generator=generator)),
        parkours=_Obj(base_parkour=_Obj(clear_radius_m=2.0, ring_outer_radius_m=4.5)),
    )


FIELD_ENV = {tk.GEOM_VAR: "field"}
RING_ENV = {tk.GEOM_VAR: "ring"}
FIELD_CLASSES = (FieldCfg, FlatCfg)
RING_CLASSES = (RingCfg, FlatCfg)


# ----------------------------------------------------------------------------- parsers


@pytest.mark.parametrize("text,want", [("1.7", 1.7), (" 1.6 ", 1.6), ("2.5", 2.5), ("3.79", 3.79)])
def test_parse_field_pad_r(text, want):
    assert tk.parse_field_pad_r(text) == pytest.approx(want)


@pytest.mark.parametrize("text", ["1.59", "1.0", "0", "-2", "3.8", "4.5", "nan", "inf", "", "abc"])
def test_parse_field_pad_r_rejects(text):
    with pytest.raises(ValueError, match=tk.FIELD_PAD_R_VAR):
        tk.parse_field_pad_r(text)


def test_parse_field_pad_r_bound_follows_the_half_width():
    assert tk.parse_field_pad_r("2.9", half_width=4.0) == pytest.approx(2.9)
    with pytest.raises(ValueError):
        tk.parse_field_pad_r("3.0", half_width=4.0)


@pytest.mark.parametrize("text,want", [("14:28", (14, 28)), (" 0 : 60 ", (0, 60)), ("7:7", (7, 7))])
def test_parse_field_blocks(text, want):
    assert tk.parse_field_blocks(text) == want


@pytest.mark.parametrize("text", ["14", "14:28:3", "14.5:28", "-1:28", "14:61", "a:b", ""])
def test_parse_field_blocks_rejects(text):
    with pytest.raises(ValueError, match=tk.FIELD_BLOCKS_VAR):
        tk.parse_field_blocks(text)


@pytest.mark.parametrize("text,want", [
    ("0.4:0.4:0.2", (0.4, 0.4, 0.2)), ("1:0:0", (1.0, 0.0, 0.0)), ("0.5 : 0.5 : 0", (0.5, 0.5, 0.0)),
])
def test_parse_field_kinds(text, want):
    assert tk.parse_field_kinds(text) == pytest.approx(want)


@pytest.mark.parametrize("text", ["0.4:0.4", "0.4:0.4:0.3", "0.6:0.6:-0.2", "a:b:c", "0.4:0.4:0.2:0"])
def test_parse_field_kinds_rejects(text):
    with pytest.raises(ValueError, match=tk.FIELD_KINDS_VAR):
        tk.parse_field_kinds(text)


@pytest.mark.parametrize("text,want", [("ramp", False), ("wall", True), (" WALL ", True), ("Ramp", False)])
def test_parse_field_edges(text, want):
    assert tk.parse_field_edges(text) is want


@pytest.mark.parametrize("text", ["walls", "1", "", "vertical"])
def test_parse_field_edges_rejects(text):
    with pytest.raises(ValueError, match=tk.FIELD_EDGES_VAR):
        tk.parse_field_edges(text)


def test_parse_min_gap_and_size_scale():
    assert tk.parse_field_min_gap("0") == 0.0 and tk.parse_field_min_gap("0.4") == pytest.approx(0.4)
    for bad in ("-0.1", "1.5", "x"):
        with pytest.raises(ValueError, match=tk.FIELD_MIN_GAP_VAR):
            tk.parse_field_min_gap(bad)
    assert tk.parse_field_size_scale("1.0") == 1.0 and tk.parse_field_size_scale("0.5") == 0.5
    for bad in ("0.25", "0", "1.6", "nan"):
        with pytest.raises(ValueError, match=tk.FIELD_SIZE_SCALE_VAR):
            tk.parse_field_size_scale(bad)


def test_parser_table_covers_every_field_var():
    assert set(tk.FIELD_KNOB_PARSERS) == set(tk.FIELD_VARS)
    assert all(v.startswith("KRABBY_FIELD_") for v in tk.FIELD_VARS)


# ------------------------------------------------------------------------------- guard


@pytest.mark.parametrize("value,want", [("ring", "ring"), ("field", "field"), (" Field ", "field")])
def test_require_omni_geom_accepts_ring_and_field(value, want):
    assert tk.require_omni_geom({tk.GEOM_VAR: value}) == want


@pytest.mark.parametrize("env", [{}, {tk.GEOM_VAR: "shallow"}, {tk.GEOM_VAR: "recal2b2"}, {tk.GEOM_VAR: ""}])
def test_require_omni_geom_refuses_leaked_presets(env):
    with pytest.raises(RuntimeError, match=tk.GEOM_VAR):
        tk.require_omni_geom(env)


@pytest.mark.parametrize("var", list(tk.FIELD_VARS))
def test_ring_refuses_field_knobs(var):
    with pytest.raises(RuntimeError, match=var):
        tk.require_omni_geom(dict(RING_ENV, **{var: "1"}))


@pytest.mark.parametrize("var", list(tk.RING_VARS))
def test_field_refuses_ring_knobs(var):
    with pytest.raises(RuntimeError, match=var):
        tk.require_omni_geom(dict(FIELD_ENV, **{var: "2.0"}))


def test_blank_foreign_knobs_count_as_unset():
    assert tk.require_omni_geom(dict(RING_ENV, **{tk.FIELD_PAD_R_VAR: "  "})) == "ring"
    assert tk.require_omni_geom(dict(FIELD_ENV, **{tk.CLEAR_R_VAR: ""})) == "field"


def test_require_ring_geom_is_unchanged():
    """The ring-only guard keeps refusing ``field`` (its own tests pin the rest)."""
    with pytest.raises(RuntimeError):
        tk.require_ring_geom(FIELD_ENV)
    assert tk.require_ring_geom(RING_ENV) == "ring"


# ----------------------------------------------------------------------------- install


def test_install_is_a_no_op_for_ring_and_unset():
    for env in (RING_ENV, {}, {tk.GEOM_VAR: "shallow"}):
        gen = _ring_generator()
        cfg = _cfg(gen)
        tk.install_omni_terrain_generator(cfg, env, field_generator=_field_generator())
        assert cfg.scene.terrain.terrain_generator is gen


def test_install_field_deep_copies_the_generator():
    template = _field_generator()
    a, b = _cfg(_ring_generator()), _cfg(_ring_generator())
    assert tk.install_omni_terrain_generator(a, FIELD_ENV, field_generator=template) == "field"
    tk.install_omni_terrain_generator(b, {tk.GEOM_VAR: " FIELD "}, field_generator=template)
    ga, gb = a.scene.terrain.terrain_generator, b.scene.terrain.terrain_generator
    assert ga is not template and gb is not ga
    assert ga.sub_terrains["omni_field_dense"].n_blocks == 28
    ga.sub_terrains["omni_field_dense"].n_blocks = 3  # a knob on one env cfg ...
    assert template.sub_terrains["omni_field_dense"].n_blocks == 28  # ... never reaches the template
    assert gb.sub_terrains["omni_field_dense"].n_blocks == 28  # ... or the next env cfg


def test_install_field_needs_a_terrain_importer():
    cfg = _Obj(scene=_Obj(terrain=None))
    with pytest.raises(RuntimeError, match="no terrain importer"):
        tk.install_omni_terrain_generator(cfg, FIELD_ENV, field_generator=_field_generator())


def test_env_cfg_installs_first_in_both_post_inits():
    """Source pin (the env cfg imports isaaclab): the hook is the first statement after the phase
    guard in the teacher and the first statement in the flat-walk cfg -- before
    ``super().__post_init__()``, so the forward knob block applies to the field."""
    import ast

    src = (OMNI / "config/crab_hex/crab_hex_omni_env_cfg.py").read_text()
    tree = ast.parse(src)
    want = {"CrabHexOmniTeacherEnvCfg": 1, "CrabHexOmniFlatWalkEnvCfg": 0}
    seen = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in want:
            post = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__"][0]
            calls = [ast.unparse(s.value.func) for s in post.body if isinstance(s, ast.Expr)
                     and isinstance(s.value, ast.Call)]
            seen[node.name] = calls
            idx = calls.index("install_omni_terrain_generator")
            assert idx == want[node.name], calls
            assert calls.index("super().__post_init__") == idx + 1, calls
    assert set(seen) == set(want)


# ------------------------------------------------------------------- field knobs / dispatcher


def test_field_knobs_end_to_end_through_the_dispatcher():
    cfg = _cfg(_field_generator())
    env = dict(FIELD_ENV, **{
        tk.FIELD_PAD_R_VAR: "1.8", tk.FIELD_BLOCKS_VAR: "10:40", tk.FIELD_KINDS_VAR: "0.5:0.3:0.2",
        tk.FIELD_EDGES_VAR: "wall", tk.FIELD_MIN_GAP_VAR: "0.2", tk.FIELD_SIZE_SCALE_VAR: "0.5",
    })
    tk.apply_omni_ring_knobs(cfg, env, classes=RING_CLASSES, field_classes=FIELD_CLASSES)
    subs = cfg.scene.terrain.terrain_generator.sub_terrains
    assert subs["omni_field_sparse"].n_blocks == 10 and subs["omni_field_dense"].n_blocks == 40
    assert all(s.clear_radius == pytest.approx(1.8) for s in subs.values())
    for key in ("omni_field_sparse", "omni_field_dense"):
        s = subs[key]
        assert (s.bar_fraction, s.pit_fraction) == pytest.approx((0.5, 0.2))
        assert s.vertical_edges is True and s.min_gap == pytest.approx(0.2)
        assert s.bar_length_range == pytest.approx((0.5, 1.0))
        assert s.slab_side_range == pytest.approx((0.3, 0.6))
    # the parkour term's radii follow the terrain: reach_edge = "left the pad"
    assert cfg.parkours.base_parkour.clear_radius_m == pytest.approx(1.8)
    assert cfg.parkours.base_parkour.ring_outer_radius_m == pytest.approx(4.80)


def test_field_defaults_write_nothing_but_the_radius_sync(capsys):
    gen = _field_generator()
    cfg = _cfg(gen)
    tk.apply_omni_ring_knobs(cfg, dict(FIELD_ENV), field_classes=FIELD_CLASSES)
    subs = gen.sub_terrains
    assert subs["omni_field_sparse"].n_blocks == 14 and subs["omni_field_dense"].n_blocks == 28
    assert subs["omni_field_sparse"].vertical_edges is False
    assert subs["omni_field_sparse"].bar_length_range == (1.0, 2.0)
    assert cfg.parkours.base_parkour.clear_radius_m == pytest.approx(1.7)
    assert "field geometry" in capsys.readouterr().out


def test_field_under_a_ring_generator_names_the_install_hook():
    cfg = _cfg(_ring_generator())
    with pytest.raises(RuntimeError, match="install_omni_terrain_generator"):
        tk.apply_omni_ring_knobs(cfg, dict(FIELD_ENV), field_classes=FIELD_CLASSES)


def test_ring_under_a_field_generator_is_refused():
    cfg = _cfg(_field_generator())
    with pytest.raises(RuntimeError, match="FieldCfg"):
        tk.apply_omni_ring_knobs(cfg, dict(RING_ENV), classes=RING_CLASSES)


def test_mixed_knobs_refused_through_the_dispatcher_both_ways():
    with pytest.raises(RuntimeError, match=tk.BLOCKS_VAR):
        tk.apply_omni_ring_knobs(
            _cfg(_field_generator()), dict(FIELD_ENV, **{tk.BLOCKS_VAR: "8:16"}), field_classes=FIELD_CLASSES
        )
    with pytest.raises(RuntimeError, match=tk.FIELD_BLOCKS_VAR):
        tk.apply_omni_ring_knobs(
            _cfg(_ring_generator()), dict(RING_ENV, **{tk.FIELD_BLOCKS_VAR: "14:28"}), classes=RING_CLASSES
        )


def test_ring_branch_unchanged_through_the_dispatcher(capsys):
    cfg = _cfg(_ring_generator())
    tk.apply_omni_ring_knobs(cfg, dict(RING_ENV, **{tk.BLOCKS_VAR: "6:20"}), classes=RING_CLASSES)
    subs = cfg.scene.terrain.terrain_generator.sub_terrains
    assert subs["omni_ring_sparse"].n_blocks == 6 and subs["omni_ring_dense"].n_blocks == 20
    assert cfg.parkours.base_parkour.clear_radius_m == pytest.approx(2.0)
    assert capsys.readouterr().out == ""


def test_field_pad_bound_checked_against_the_installed_half_width():
    gen = _field_generator()
    for s in gen.sub_terrains.values():
        s.ring_outer_radius = 3.5
    with pytest.raises(ValueError, match=tk.FIELD_PAD_R_VAR):
        tk.apply_omni_ring_knobs(_cfg(gen), dict(FIELD_ENV, **{tk.FIELD_PAD_R_VAR: "2.6"}),
                                 field_classes=FIELD_CLASSES)


def test_field_blocks_pair_needs_two_field_sub_terrains():
    gen = _field_generator()
    del gen.sub_terrains["omni_field_dense"]
    with pytest.raises(RuntimeError, match=tk.FIELD_BLOCKS_VAR):
        tk.apply_omni_ring_knobs(_cfg(gen), dict(FIELD_ENV, **{tk.FIELD_BLOCKS_VAR: "1:2"}),
                                 field_classes=FIELD_CLASSES)


def test_is_field_generator():
    assert tk.is_field_generator(_field_generator().sub_terrains, FieldCfg)
    assert not tk.is_field_generator(_ring_generator().sub_terrains, FieldCfg)
    assert not tk.is_field_generator(None, FieldCfg)


# ------------------------------------------------------------------------ knob whitelist


def test_omni_knobs_reexports_the_field_vars_and_parsers():
    """``run_candidates.knob_whitelist`` collects every ``*_VAR`` of ``omni_knobs``; the field knobs
    must be among them, and the parser table must be reachable for the round-trip check."""
    knobs = _load(OMNI / "config/crab_hex/omni_knobs.py", "omni_knobs__field_test")
    names = {getattr(knobs, a) for a in dir(knobs) if a.endswith("_VAR")}
    assert set(tk.FIELD_VARS) <= names
    assert set(knobs.FIELD_KNOB_PARSERS) == set(tk.FIELD_VARS)
    assert knobs.FIELD_KNOB_PARSERS[knobs.FIELD_BLOCKS_VAR]("14:28") == (14, 28)
    for name in ("FIELD_PAD_R_VAR", "FIELD_BLOCKS_VAR", "FIELD_KINDS_VAR", "FIELD_EDGES_VAR",
                 "FIELD_MIN_GAP_VAR", "FIELD_SIZE_SCALE_VAR", "FIELD_KNOB_PARSERS"):
        assert name in knobs.__all__


def test_real_field_cfg_classes_satisfy_the_guard():
    """Under Kit: the shipped OMNI_FIELD_TERRAINS_CFG passes its own assertion (skipped without pxr)."""
    try:
        from parkour_isaaclab.terrains.omni.config.omni_field import OMNI_FIELD_TERRAINS_CFG
        from parkour_isaaclab.terrains.omni.omni_terrains_cfg import (
            OmniFieldTerrainCfg,
            OmniFlatTerrainCfg,
        )
    except ModuleNotFoundError as exc:  # no pxr / no Kit
        pytest.skip(f"isaaclab terrain cfgs need a Kit process: {exc}")
    field_keys, flat_keys = tk.assert_field_sub_terrains(
        OMNI_FIELD_TERRAINS_CFG.sub_terrains, OmniFieldTerrainCfg, OmniFlatTerrainCfg
    )
    assert field_keys == ["omni_field_sparse", "omni_field_dense"] and flat_keys == ["parkour_flat"]
