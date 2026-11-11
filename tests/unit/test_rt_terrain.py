# SPDX-License-Identifier: BSD-3-Clause
"""Rough-terrain ladder (``KRABBY_SITE_LADDER=rt``; ``crab_hex_omni_task/experiments/2026-11-10_rough_terrain/PLAN.md``
step 2 and "Left for implementation" items 1, 2, 3, 15, 18): dose tests on the REALISED height fields of the COMPOSED
training cfg, plus the rt knob surface.

No Isaac Sim. The composed path is built from the real sources, not from hand-written numbers:

* the rt generator is read from ``config/omni_site.py`` (``OMNI_SITE_RT_TERRAINS_CFG``, by AST) and its sub-terrain
  cfgs get the class defaults of the real cfg classes (``omni_terrains_cfg.py`` and its bases, by AST) on stand-in
  classes with the same inheritance;
* the environment is the omni phase-2a preset (``crab_hex_omni_phases.phase_env("2a")``) under the campaign's own
  training env file ``envs/RT_train.env`` (the file wins, as ``activate_phase``'s setdefault does);
* the forward block's terrain step is the real ``_apply_crab_hex_easy_mixed_terrain`` (exec'd from
  ``crab_hex_env_cfg.py``: it forces ``noise_range = (0.02, 0.02)`` on every sub-terrain) plus the curriculum switch,
  then the real ``install_omni_terrain_generator`` / ``apply_omni_ring_knobs`` / ``apply_rt_friction_knobs``;
* the tiles come from the real decorated bodies through ``ParkourTerrainGenerator._generate_curriculum_terrains``'
  arithmetic (column = first ``index / num_cols + 0.001 < cumsum(p)``, row ``L`` at ``d = lo + (hi - lo) L / 9``,
  ``random_difficulty`` False as TDR's dump records), at levels 0, 4 and 9.

Realised = the 8-tuple's bordered int16 heights (roughness included) and its bordered ``layers`` (``surface0`` =
the pre-roughness surface). Gradients on the realised ``surface0`` use a least-squares plane fit over 5 x 5 pixels
(0.4 m, a foot-placement footprint): a 1-pixel difference reads the 5 mm vertical quantisation as a 3.6 deg facet,
which would swamp the 2 deg bottom rung.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("trimesh", reason="the terrain mesh utilities (terrains/utils.py) import trimesh")
pytest.importorskip("scipy", reason="the gravel roughness spline and the pit labelling use scipy")

from scipy import ndimage  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
LIB = REPO / "parkour/parkour_isaaclab"
OMNI_PKG = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
FWD_CFG_PY = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/config/crab_hex/crab_hex_env_cfg.py"
TERRAIN_KNOBS_PY = OMNI_PKG / "mdp/omni_terrain_knobs.py"
PHASES_PY = OMNI_PKG / "config/crab_hex/crab_hex_omni_phases.py"
OMNI_SITE_PY = LIB / "terrains/omni/config/omni_site.py"
CFG_SOURCES = (LIB / "terrains/parkour_terrain_generator_cfg.py",
               LIB / "terrains/extreme_parkour/extreme_parkour_terrains_cfg.py",
               LIB / "terrains/omni/omni_terrains_cfg.py")
CAMPAIGN = OMNI_PKG / "experiments/2026-11-10_rough_terrain"
TRAIN_ENV = CAMPAIGN / "envs/RT_train.env"
TDR_DUMP = Path("/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/"
                "2026-11-07_19-49-46_RSD_TDR_s19/params/env.yaml")
FIXTURES = Path(__file__).resolve().parent / "fixtures"
if str(FIXTURES) not in sys.path:
    sys.path.insert(0, str(FIXTURES))

import gen_omni_ring_golden as golden  # noqa: E402


def _load(path: Path, name: str) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


tk = _load(TERRAIN_KNOBS_PY, "crab_hex_omni_terrain_knobs")
ph = _load(PHASES_PY, "crab_hex_omni_phases__rt_terrain_test")
utils, fwd, omni = golden.load_terrain_modules()
st = golden._load_real_module("parkour_isaaclab.terrains.omni.site_terrains",
                              golden.LIB / "terrains" / "omni" / "site_terrains.py")
K = st._site_raster()

HS, VS = 0.08, 0.005
LEVELS = (0, 4, 9)
SPAWN_Z_M = 1.085                 # crab_hex_scene_cfg.py:167, the root above the local surface at spawn
TERMINATION_Z_M = -0.25
NUM_GOALS = 8
EVAL_TRACK = ((0.0, 0.0), (2.0, 0.0), (2.0, 1.76), (3.41, 3.17))
PIXEL_KIND = {"slopes": K.KIND_SLOPE, "hills": K.KIND_HILL, "ledge_up": K.KIND_LEDGE, "ledge_down": K.KIND_LEDGE,
              "pits": K.KIND_PIT, "gravel": K.KIND_GRAVEL, "rubble": K.KIND_RUBBLE}


# ================================================================================ the composed training cfg
def read_env_file(path: Path) -> dict[str, str]:
    env = {}
    for line in path.read_text().splitlines():
        if line.startswith("KRABBY_") and "=" in line:
            key, value = line.split("=", 1)
            env[key] = value
    return env


def _class_table() -> dict:
    """``{class name: (base names, literal field defaults, function (module alias, name) | None)}`` of the real cfg
    classes, by AST (the configclass decorator needs Isaac)."""
    table = {}
    for path in CFG_SOURCES:
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.ClassDef):
                continue
            fields, func = {}, None
            for stmt in node.body:
                target = value = None
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                    target, value = stmt.target.id, stmt.value
                elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    target, value = stmt.targets[0].id, stmt.value
                if target is None:
                    continue
                if target == "function" and isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
                    func = (value.value.id, value.attr)
                    continue
                try:
                    fields[target] = ast.literal_eval(value)
                except ValueError:
                    pass                                    # MISSING and other non-literals: no default
            table[node.name] = ([b.id for b in node.bases if isinstance(b, ast.Name)], fields, func)
    return table


TABLE = _class_table()
_MODULES = {"site_terrains": st, "omni_terrains": omni}
_CLASSES: dict[str, type] = {}


def standin_class(name: str) -> type:
    """A plain class named like the real cfg class, with the real inheritance among the parsed classes."""
    if name not in _CLASSES:
        bases = tuple(standin_class(b) for b in TABLE[name][0] if b in TABLE) or (object,)
        _CLASSES[name] = type(name, bases, {})
    return _CLASSES[name]


def instantiate(name: str, **kw):
    """A stand-in cfg: SubTerrainBaseCfg's defaults, then every parsed class default along the MRO, then ``kw``."""
    cls = standin_class(name)
    fields = {"proportion": 1.0, "size": (10.0, 10.0), "flat_patch_sampling": None}
    func = None
    for klass in reversed(cls.__mro__):
        if klass.__name__ in TABLE:
            _bases, defaults, f = TABLE[klass.__name__]
            fields.update(copy.deepcopy(defaults))
            func = f or func
    fields.update(kw)
    obj = cls.__new__(cls)
    obj.__dict__.update(fields)
    obj.function = getattr(_MODULES[func[0]], func[1])
    return obj


def rt_generator_from_source():
    """``OMNI_SITE_RT_TERRAINS_CFG`` as written in ``config/omni_site.py``, on stand-in cfgs."""
    tree = ast.parse(OMNI_SITE_PY.read_text())
    call = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", None) == "OMNI_SITE_RT_TERRAINS_CFG")
    gen = types.SimpleNamespace(num_goals=8, random_difficulty=False, seed=None)
    for kw in call.keywords:
        if kw.arg == "sub_terrains":
            subs = {}
            for key, val in zip(kw.value.keys, kw.value.values):
                subs[ast.literal_eval(key)] = instantiate(
                    val.func.id, **{k.arg: ast.literal_eval(k.value) for k in val.keywords})
            gen.sub_terrains = subs
        else:
            setattr(gen, kw.arg, ast.literal_eval(kw.value))
    return gen


def _forward_light_helper():
    """``_apply_crab_hex_easy_mixed_terrain`` exec'd from ``crab_hex_env_cfg.py`` (the module imports Isaac)."""
    tree = ast.parse(FWD_CFG_PY.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_apply_crab_hex_easy_mixed_terrain")
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(FWD_CFG_PY), "exec"), ns)  # noqa: S102
    return ns["_apply_crab_hex_easy_mixed_terrain"]


CLASSES = (standin_class("OmniFlatTerrainCfg"), standin_class("OmniSiteFieldTerrainCfg"),
           standin_class("OmniSiteTerrainCfg"))


def material(**kw):
    m = types.SimpleNamespace(static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
                              friction_combine_mode="average", restitution_combine_mode="average")
    m.__dict__.update(kw)
    return m


def env_cfg():
    """The omni env cfg's terrain / parkour / events / sim surface the knob block touches (TDR's values)."""
    return types.SimpleNamespace(
        scene=types.SimpleNamespace(terrain=types.SimpleNamespace(terrain_generator="ring", physics_material=material())),
        sim=types.SimpleNamespace(physics_material=material()),
        parkours=types.SimpleNamespace(base_parkour=types.SimpleNamespace(clear_radius_m=2.0, ring_outer_radius_m=4.5,
                                                                          freeze_terrain_levels=False)),
        events=types.SimpleNamespace(
            reset_root_state=types.SimpleNamespace(params={"offset": 0.0}),
            physics_material=types.SimpleNamespace(
                func=types.SimpleNamespace(__name__="randomize_rigid_body_material"), mode="startup",
                params={"asset_cfg": "robot(.*)", "friction_range": (0.6, 2.0), "num_buckets": 64})),
    )


def fake_contract(*_a, **_k):
    return None


fake_contract.__name__ = "rt_friction_contract"


def compose(env: dict, *, curriculum: bool):
    """The composed omni cfg for ``env`` (the knob block's terrain path, in ``__post_init__`` order)."""
    cfg = env_cfg()
    assert tk.install_omni_terrain_generator(cfg, env, rt_generator=rt_generator_from_source()) == "site"
    tg = cfg.scene.terrain.terrain_generator
    lo, hi = (float(x) for x in env.get("KRABBY_FLAT_TERRAIN_DIFF", "0.05:0.2").split(":"))
    _forward_light_helper()(tg, flat_proportion=float(env.get("KRABBY_FLAT_TERRAIN_FLAT_FRAC", "0.8")),
                            difficulty_range=(lo, hi))
    if curriculum:
        tg.curriculum = True
    tk.apply_omni_ring_knobs(cfg, env, site_classes=CLASSES)
    tk.apply_rt_friction_knobs(cfg, env, exclusive_with=("KRABBY_STRESS_MU",), contract_func=fake_contract)
    return cfg


def train_env() -> dict:
    env = dict(ph.phase_env("2a"))
    env.update(read_env_file(TRAIN_ENV))
    return env


def curriculum_columns(tg) -> list[str]:
    props = np.array([s.proportion for s in tg.sub_terrains.values()], dtype=float)
    props /= props.sum()
    keys = list(tg.sub_terrains)
    return [keys[int(np.min(np.where(i / tg.num_cols + 0.001 < np.cumsum(props))[0]))] for i in range(tg.num_cols)]


def tile_cfg(tg, key):
    """The generator's per-tile cfg: TerrainGenerator.__init__'s propagation, then _get_terrain_mesh's copy."""
    sub = copy.deepcopy(tg.sub_terrains[key])
    sub.size = tuple(tg.size)
    sub.horizontal_scale, sub.vertical_scale, sub.slope_threshold = tg.horizontal_scale, tg.vertical_scale, \
        tg.slope_threshold
    return sub


def generate_tile(tg, col: int, level: int, seed: int):
    key = curriculum_columns(tg)[col]
    lo, hi = tg.difficulty_range
    difficulty = lo + (hi - lo) * level / (tg.num_rows - 1)
    sub = tile_cfg(tg, key)
    sub.difficulty = float(difficulty)
    sub.tile_col = col                # what ParkourTerrainGenerator._get_terrain_mesh hands the rt cfg
    np.random.seed(seed)
    out = sub.function(difficulty, sub, NUM_GOALS)
    return key, difficulty, out


@pytest.fixture(scope="module")
def composed():
    return compose(train_env(), curriculum=True)


@pytest.fixture(scope="module")
def tiles(composed):
    """``{(kind, level, column): (difficulty, out)}`` for every rt hazard column at levels 0, 4 and 9."""
    tg = composed.scene.terrain.terrain_generator
    cols = curriculum_columns(tg)
    out = {}
    for col, key in enumerate(cols):
        sub = tg.sub_terrains[key]
        kind = getattr(sub, "kind", None)
        if kind not in st.RT_KINDS:
            continue
        for level in LEVELS:
            _key, d, res = generate_tile(tg, col, level, seed=1000 + 37 * col + level)
            out[(kind, level, col)] = (d, res)
    return out


# ------------------------------------------------------------------------------------- frame helpers
W = 126                                       # bordered tile pixels (int(10 / 0.08) + 1)
B = 1                                         # border pixels (border_width 0 -> int(0 / 0.08) + 1)
XS = (np.arange(W - 2 * B) + 0.5) * HS - 0.5 * (W - 2 * B) * HS
_DX, _DY = np.meshgrid(XS, XS, indexing="ij")
R = np.hypot(_DX, _DY)
CHEB = np.maximum(np.abs(_DX), np.abs(_DY))


def body(arr):
    """A bordered (126, 126) layer -> the body's (124, 124) pixel frame (tile-centre coords ``_DX`` / ``_DY``)."""
    return np.asarray(arr)[B:-B, B:-B]


def sample(field, x, y):
    """Nearest-pixel sample of a body-frame field at tile-centre coords."""
    i = np.clip(np.round((np.asarray(x) + 0.5 * XS.size * HS) / HS - 0.5).astype(int), 0, XS.size - 1)
    j = np.clip(np.round((np.asarray(y) + 0.5 * XS.size * HS) / HS - 0.5).astype(int), 0, XS.size - 1)
    return field[i, j]


def bilinear(field, x, y):
    """Bilinear sample of a body-frame field (pixel-centre grid) at tile-centre coords."""
    fi = (np.asarray(x, dtype=float) + 0.5 * XS.size * HS) / HS - 0.5
    fj = (np.asarray(y, dtype=float) + 0.5 * XS.size * HS) / HS - 0.5
    i0 = np.clip(np.floor(fi).astype(int), 0, XS.size - 2)
    j0 = np.clip(np.floor(fj).astype(int), 0, XS.size - 2)
    a, b = fi - i0, fj - j0
    return ((1 - a) * (1 - b) * field[i0, j0] + a * (1 - b) * field[i0 + 1, j0]
            + (1 - a) * b * field[i0, j0 + 1] + a * b * field[i0 + 1, j0 + 1])


def plane_grad(z, half=2):
    """Least-squares plane-fit gradient magnitude (m/m) over ``(2 half + 1)^2`` pixel windows (valid region)."""
    from numpy.lib.stride_tricks import sliding_window_view

    k = 2 * half + 1
    off = (np.arange(k) - half) * HS
    ox, oy = np.meshgrid(off, off, indexing="ij")
    sxx = float((ox * ox).sum())
    win = sliding_window_view(z, (k, k))
    return np.hypot((win * ox).sum(axis=(-1, -2)) / sxx, (win * oy).sum(axis=(-1, -2)) / sxx)


def ray_samples(r_to: float, n_rays: int = 720, step: float = 0.02):
    ang = np.arange(n_rays) * (2.0 * np.pi / n_rays)
    rr = np.arange(0.0, r_to + 1e-9, step)
    return ang, rr, rr[None, :] * np.cos(ang)[:, None], rr[None, :] * np.sin(ang)[:, None]


def d_site_of(level: int) -> float:
    return level / 9.0                        # u = (d - 0.2) / 0.8 with d = 0.2 + 0.8 L / 9, KRABBY_SITE_DIFF 0:1


def deg(m_per_m) -> float:
    return math.degrees(math.atan(float(m_per_m)))


def within(value, table, tol=0.10) -> bool:
    return abs(float(value) - float(table)) <= tol * abs(float(table)) + 1e-12


def _layers(res):
    return res[7]


def _heights(res):
    return body(res[5]).astype(np.int32)


def _surface0(res):
    return body(_layers(res)["surface0"]).astype(np.int32)


def _level(res):
    return body(_layers(res)["level"]).astype(np.int32)


def _kind(res):
    return body(_layers(res)["kind"])


# ================================================================================ the composed cfg itself
def test_the_training_env_file_is_tdrs_with_exactly_the_planned_changes():
    tdr_path = (Path("/home/nickmagus/krabby/krabby-research/.claude/worktrees/dr-stress-sweep/parkour/parkour_tasks/"
                     "parkour_tasks/crab_hex_omni_task/experiments/2026-11-07_rod_speed_dr/envs/RSD_TDR_s19.train.env"))
    if not tdr_path.is_file():
        tdr_path = OMNI_PKG / "experiments/2026-11-07_rod_speed_dr/envs/RSD_TDR_s19.train.env"
    if not tdr_path.is_file():
        pytest.skip("TDR's recorded training env file is not on this machine")
    tdr, rt = read_env_file(tdr_path), read_env_file(TRAIN_ENV)
    changed = {k for k in set(tdr) | set(rt) if tdr.get(k) != rt.get(k)}
    assert changed == {"KRABBY_FLAT_TERRAIN_GEOM", "KRABBY_SITE_LADDER", "KRABBY_SITE_MIX", "KRABBY_RT_FRICTION",
                       "KRABBY_DR_ROD_SPEED"}
    assert rt["KRABBY_FLAT_TERRAIN_GEOM"] == "site" and rt["KRABBY_SITE_LADDER"] == "rt"
    assert tk.parse_site_mix(rt["KRABBY_SITE_MIX"], "rt") == tk.parse_site_mix(tk.RT_TRAIN_MIX, "rt")
    assert rt["KRABBY_RT_FRICTION"] == "gravel:0.2:0.5" and rt["KRABBY_DR_ROD_SPEED"] == "0.6:1.35"


def test_composed_training_cfg_columns_functions_and_d_mapping(composed):
    tg = composed.scene.terrain.terrain_generator
    assert tuple(tg.sub_terrains) == tk.RT_SUBTERRAIN_KEYS
    assert tg.num_cols == 20 and tg.num_rows == 10 and tg.curriculum is True
    assert tuple(tg.difficulty_range) == (0.2, 1.0)
    cols = curriculum_columns(tg)
    kinds = [getattr(tg.sub_terrains[k], "kind", {"parkour_flat": "flat", "site_field_sparse": "field_sparse",
                                                  "site_field_dense": "field_dense"}.get(k)) for k in cols]
    assert kinds == (["flat"] * 3 + ["gravel"] * 2 + ["field_sparse"] * 2 + ["field_dense"] + ["slopes"] * 2
                     + ["hills"] * 2 + ["ledge_up"] * 2 + ["ledge_down"] * 2 + ["pits"] * 2 + ["rubble"] * 2)
    g = [i for i, k in enumerate(kinds) if k == "gravel"]
    assert {kinds[g[0] - 1], kinds[g[-1] + 1]} <= {"flat", "field_sparse", "field_dense"}   # gravel beside flat / field
    for key in tk.RT_SUBTERRAIN_KEYS:
        sub = tg.sub_terrains[key]
        assert tuple(sub.noise_range) == (0.02, 0.02)          # main's forced roughness reached every column
        assert sub.clear_radius == 1.7 and sub.ring_outer_radius == 4.8
        if getattr(sub, "kind", None) in st.RT_KINDS:
            assert sub.function is st.omni_site_terrain_rt and st.ladder_of(sub.function) == "rt"
            assert tuple(sub.site_d_range) == (0.0, 1.0) and tuple(sub.gen_d_range) == (0.2, 1.0)
    assert tg.sub_terrains["rt_gravel"].apply_roughness is False
    assert composed.events.reset_root_state.params["spawn_level_z"] is True
    assert composed.parkours.base_parkour.clear_radius_m == 1.7


def test_composed_forward_block_matches_tdrs_recorded_generator():
    """The emulated forward terrain step lands where TDR's own run did (curriculum, band, row count, roughness)."""
    if not TDR_DUMP.is_file():
        pytest.skip("TDR's params/env.yaml is not on this machine")
    yaml = pytest.importorskip("yaml")

    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_multi_constructor("tag:yaml.org,2002:python/", lambda loader, suffix, node: (
        loader.construct_sequence(node) if isinstance(node, yaml.SequenceNode) else None))
    doc = yaml.load(TDR_DUMP.read_text(), Loader=_Loader)
    tdr = doc["scene"]["terrain"]["terrain_generator"]
    cfg = compose(train_env(), curriculum=True)
    tg = cfg.scene.terrain.terrain_generator
    assert tdr["curriculum"] is True and tdr["random_difficulty"] is False
    assert list(tdr["difficulty_range"]) == list(tg.difficulty_range)
    assert tdr["num_rows"] == tg.num_rows and tdr["num_cols"] == tg.num_cols
    assert tdr["horizontal_scale"] == tg.horizontal_scale and tdr["vertical_scale"] == tg.vertical_scale
    assert list(tdr["sub_terrains"]["parkour_flat"]["noise_range"]) == list(tg.sub_terrains["parkour_flat"].noise_range)


# ================================================================================ realised doses (items 1, 2, 3, 15)
def test_every_rt_tile_reads_its_level_d_and_contract(tiles):
    for (kind, level, _col), (_d, res) in tiles.items():
        layers = _layers(res)
        K.check_tile_layers(layers, res[5].shape)
        assert layers["tile"]["d"] == pytest.approx(d_site_of(level), abs=1e-9), (kind, level)
        assert layers["tile"]["cross_r"] == st.RT_CROSS_R[kind] <= 3.2
        assert (_kind(res) == PIXEL_KIND[kind]).any(), (kind, level)
        assert not np.isin(_kind(res), K.BLOCKER_KINDS).any()


def test_spawn_level_is_the_realised_centre_and_the_lowest_level_keeps_the_termination_margin(tiles):
    centre = (R <= 0.12)
    for (kind, level, _col), (_d, res) in tiles.items():
        layers = _layers(res)
        c_level = layers["tile"]["centre_level_m"]
        assert np.abs(_surface0(res)[centre] * VS - c_level).max() <= 1e-9, (kind, level)
        assert np.abs(_level(res)[centre] * VS - c_level).max() <= 1e-9, (kind, level)
        # the realised (rough) surface at the spawn: the centre level within main's 0.02 m roughness
        assert np.abs(_heights(res)[centre] * VS - c_level).max() <= 0.02 + 0.5 * VS + 1e-9, (kind, level)
        walkable = ~np.isin(_kind(res), K.VOID_KINDS)
        lowest = min(_level(res).min(), _surface0(res)[walkable].min()) * VS
        assert lowest + SPAWN_Z_M >= TERMINATION_Z_M + 0.5, (kind, level, lowest)


def test_the_seam_ring_is_level_zero_on_every_tile(tiles):
    ring = CHEB >= tk.SITE_HALF_WIDTH - 0.40 + 1e-9                   # max(|x|, |y|) >= 4.4 m
    for (kind, level, _col), (_d, res) in tiles.items():
        assert (_level(res)[ring] == 0).all() and (_surface0(res)[ring] == 0).all(), (kind, level)
        assert np.abs(_heights(res)[ring] * VS).max() <= 0.05 + 1e-9, (kind, level)


def test_every_heading_meets_the_feature_within_3m(tiles):
    """Ray-cast from the spawn: every heading crosses a feature pixel (or, ledges, the rim) by r = 3 m. Pits: every
    angular slot holds its pit (item 15) -- asserted in the pit test."""
    ang, rr, x, y = ray_samples(3.0)
    for (kind, level, _col), (_d, res) in tiles.items():
        if kind == "pits":
            continue
        k = sample(_kind(res), x, y)
        if kind in ("ledge_up", "ledge_down"):
            met = (k == K.KIND_LEDGE).any(axis=1) & (k == K.KIND_GROUND).any(axis=1)
        else:
            met = (k == PIXEL_KIND[kind]).any(axis=1)
        assert met.all(), (kind, level, int((~met).sum()))


def test_slopes_rim_median_gradient_and_eval_track_gradient(tiles):
    seen_signs = set()
    for (kind, level, _col), (_d, res) in tiles.items():
        if kind != "slopes":
            continue
        grade = st.ladder_value("slopes", "grade_deg", d_site_of(level), "rt")
        g = math.tan(math.radians(grade))
        s0 = _surface0(res) * VS
        pad = _layers(res)["tile"]["centre_level_m"]
        seen_signs.add(np.sign(pad))
        _g, r_out, pad_abs = st.rt_slope_profile(grade)
        assert abs(abs(pad) - pad_abs) <= 0.5 * VS + 1e-9 and abs(pad) <= st.RT_SLOPE_PAD_MAX_M + 1e-9
        ang, rr, x, y = ray_samples(st.RT_CROSS_R["slopes"])
        prof = bilinear(s0, x, y)
        # rim distance: the first radius whose surface leaves the pad level (> 1 vertical unit) -- <= 2.0 m per heading
        left = np.abs(prof - pad) > VS + 1e-9
        rim = np.where(left.any(axis=1), rr[np.argmax(left, axis=1)], np.inf)
        assert rim.max() <= 2.0, (level, rim.max())
        # median gradient rim (1.7 m) -> crossing radius (3.0 m), over the 720 headings
        h_rim = bilinear(s0, 1.7 * np.cos(ang), 1.7 * np.sin(ang))
        h_x = bilinear(s0, 3.0 * np.cos(ang), 3.0 * np.sin(ang))
        med = float(np.median(np.abs(h_rim - h_x) / (3.0 - 1.7)))
        assert med >= 0.9 * g, (level, deg(med), grade)
        # the commanded eval track, sampled every 0.02 m, slope over a 0.24 m baseline
        pts = []
        for (x0, y0), (x1, y1) in zip(EVAL_TRACK[:-1], EVAL_TRACK[1:]):
            n = int(math.ceil(math.hypot(x1 - x0, y1 - y0) / 0.02))
            t = np.arange(n) / n
            pts.append(np.stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t], 1))
        pts = np.concatenate(pts + [np.array([EVAL_TRACK[-1]])])
        h = bilinear(s0, pts[:, 0], pts[:, 1])
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
        lag = 12
        track = np.abs(h[lag:] - h[:-lag]) / (s[lag:] - s[:-lag])
        assert track.max() >= 0.9 * g, (level, deg(track.max()), grade)
        assert (_layers(res)["kind"] != K.KIND_LEDGE).all()                   # no vertical rim: a pure incline
        assert res[4].sum() == 0                                             # no riser / slope-threshold edge
    assert seen_signs <= {-1.0, 1.0}


def layout_cfg(kind: str):
    """An ``OmniSiteRtTerrainCfg`` stand-in as the decorator hands it to the body (reduced 9.92 m tile)."""
    cfg = instantiate("OmniSiteRtTerrainCfg", kind=kind)
    cfg.size, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_threshold = (9.92, 9.92), HS, VS, 1.5
    return cfg


def test_slopes_draw_both_variants_and_the_riser_free_tiles_have_no_riser():
    up = down = 0
    for seed in range(40):
        lay = st.site_layout(np.random.RandomState(seed), 0.5, layout_cfg("slopes"), "rt")
        c = int(lay.level[62, 62])
        up += c < 0
        down += c > 0
        assert not lay.riser.any()
    assert up >= 10 and down >= 10
    for kind in ("hills", "gravel"):
        assert not st.site_layout(np.random.RandomState(1), 1.0, layout_cfg(kind), "rt").riser.any()
    assert st.site_layout(np.random.RandomState(1), 1.0, layout_cfg("ledge_up"), "rt").riser.any()   # rims stay vertical


def test_slopes_variant_alternates_by_column_parity_so_every_level_holds_both(tiles):
    """Plan item 10 / review: the up (sunk pad) / down (raised pad) variant is the column's parity, not a per-tile draw."""
    by_level: dict = {}
    for (kind, level, col), (_d, res) in tiles.items():
        if kind != "slopes":
            continue
        pad = _layers(res)["tile"]["centre_level_m"]
        assert (pad < 0) == (col % 2 == 0) and (pad > 0) == (col % 2 == 1), (level, col, pad)
        by_level.setdefault(level, set()).add(np.sign(pad))
    assert set(by_level) == set(LEVELS) and all(v == {-1.0, 1.0} for v in by_level.values()), by_level
    # deterministic in the column alone (the seed does not move it); the stream does not depend on the column
    for seed in range(5):
        for col in (0, 1, 2, 3):
            cfg = layout_cfg("slopes")
            cfg.tile_col = col
            lay = st.site_layout(np.random.RandomState(seed), 0.5, cfg, "rt")
            assert (int(lay.level[62, 62]) < 0) == (col % 2 == 0)
    a, b = np.random.RandomState(3), np.random.RandomState(3)
    cfg = layout_cfg("slopes")
    st.site_layout(a, 0.5, cfg, "rt")
    cfg.tile_col = 7
    st.site_layout(b, 0.5, cfg, "rt")
    assert a.uniform() == b.uniform()


def test_the_generator_hands_the_rt_cfg_its_column_and_leaves_other_cfgs_alone():
    src = (LIB / "terrains/parkour_terrain_generator.py").read_text()
    assert src.count("self._tile_col = sub_col") == 2                      # the random and the curriculum loop
    assert 'if hasattr(cfg, "tile_col"):' in src and 'cfg.tile_col = getattr(self, "_tile_col", None)' in src
    assert "tile_col" in TABLE["OmniSiteRtTerrainCfg"][1] and TABLE["OmniSiteRtTerrainCfg"][1]["tile_col"] is None
    assert not any("tile_col" in TABLE[c][1] for c in TABLE if c != "OmniSiteRtTerrainCfg")


def test_hills_realised_max_gradient_waves_and_pad(tiles):
    for (kind, level, _col), (_d, res) in tiles.items():
        if kind != "hills":
            continue
        grade = st.ladder_value("hills", "grade_deg", d_site_of(level), "rt")
        s0 = _surface0(res) * VS
        realised = deg(plane_grad(s0).max())
        assert within(realised, grade) and realised <= st.RAMP_MAX_DEG, (level, realised, grade)
        assert (s0[R <= st.PAD_R_M] == 0.0).all()                            # flat pad: no toe inside a rising wave
        assert res[4].sum() == 0                                             # riser-free
        # >= 2 waves along every heading: at least 2 crests between the pad edge and the seam ring
        ang, rr, x, y = ray_samples(tk.SITE_HALF_WIDTH - 0.40, n_rays=72, step=0.04)
        prof = bilinear(s0, x, y)
        crests = ((prof[:, 1:-1] > prof[:, :-2]) & (prof[:, 1:-1] >= prof[:, 2:]) & (prof[:, 1:-1] > 0.5 * VS)).sum(1)
        assert crests.min() >= 2, (level, crests.min())


def test_hills_design_gradient_is_exactly_the_table():
    """The design (pre-quantisation) maximum gradient is the ladder's grade; quantisation moves no pixel > 0.5 unit."""
    for d in (0.0, 4 / 9, 1.0):
        lay = st.site_layout(np.random.RandomState(3), d, layout_cfg("hills"), "rt")
        phi = np.random.RandomState(3).uniform(0.0, 2.0 * math.pi)              # the layout's one draw
        unit, grad = st._rt_hill_unit(st._Canvas(layout_cfg("hills")), st.PAD_R_M, phi)
        amp = math.tan(math.radians(st.ladder_value("hills", "grade_deg", d, "rt"))) / grad.max()
        assert deg(amp * grad.max()) == pytest.approx(st.ladder_value("hills", "grade_deg", d, "rt"), abs=1e-9)
        assert np.abs(lay.level * VS - amp * unit).max() <= 0.5 * VS + 1e-9
        assert (lay.level[st._Canvas(layout_cfg("hills")).r >= tk.SITE_HALF_WIDTH - 0.40] == 0).all()


def test_ledge_heights(tiles):
    for (kind, level, _col), (_d, res) in tiles.items():
        if kind not in ("ledge_up", "ledge_down"):
            continue
        table = st.ladder_value(kind, "height_m", d_site_of(level), "rt")
        cap = st.RISE_CAP_M if kind == "ledge_up" else st.DROP_CAP_M
        step = _layers(res)["tile"]["centre_level_m"]
        assert (step < 0) if kind == "ledge_up" else (step > 0)
        assert within(abs(step), table) and abs(step) <= cap + 1e-9, (kind, level, step, table)
        pad = _level(res)[R <= st.PAD_R_M - HS]
        assert np.abs(pad * VS - step).max() <= 1e-9 and (_level(res)[R > st.PAD_R_M + HS] == 0).all()


def test_pits_depth_side_band_and_cover_per_angular_slot(tiles):
    report = []
    band_area = math.pi * (st.RT_BAND_R_OUT_M ** 2 - st.RT_BAND_R_IN_M ** 2)
    for (kind, level, col), (_d, res) in tiles.items():
        if kind != "pits":
            continue
        d = d_site_of(level)
        side, n_px, n, cover = st.rt_pit_plan(d, HS)
        depth = st.ladder_value("pits", "depth_m", d, "rt")
        pit = _kind(res) == K.KIND_PIT
        s0 = _surface0(res)
        assert within(-s0[pit].min() * VS, depth) and np.abs(s0[pit] * VS + depth).max() <= 0.5 * VS + 1e-9
        assert (R[pit] >= st.RT_BAND_R_IN_M).all() and (R[pit] <= st.RT_BAND_R_OUT_M).all()
        lab, n_comp = ndimage.label(pit, structure=np.ones((3, 3)))
        sizes = np.bincount(lab.ravel())[1:]
        assert n_comp == n and (sizes == n_px).all(), (level, n_comp, n, set(sizes.tolist()))
        assert within(math.sqrt(sizes.mean()) * HS, side), (level, side)
        theta = np.arctan2(_DY, _DX) % (2.0 * np.pi)
        slot = np.floor(theta / (2.0 * np.pi / n)).astype(int)
        per_slot = np.bincount(slot[pit], minlength=n) * HS * HS / (band_area / n)
        assert per_slot.size == n and (per_slot >= 0.9 * cover).all() and (per_slot <= 1.1 * cover).all(), \
            (level, per_slot.min(), per_slot.max(), cover)
        report.append((level, col, side, n_px, n, cover, float(per_slot.min()), float(per_slot.max())))
    print("[rt pits] level col side n_px n_slots cover_target per_slot_cover_min max:", report)


def test_rubble_height_and_band(tiles):
    for (kind, level, _col), (_d, res) in tiles.items():
        if kind != "rubble":
            continue
        h = st.ladder_value("rubble", "height_max_m", d_site_of(level), "rt")
        feat = (_surface0(res) - _level(res)) * VS
        band = (R >= st.RT_BAND_R_IN_M) & (R <= st.RT_BAND_R_OUT_M)
        assert within(feat[band].max(), h) and feat[band].max() <= h + 0.5 * VS + 1e-9, (level, feat[band].max(), h)
        assert (feat[~band] == 0).all() and (_kind(res)[band] == K.KIND_RUBBLE).all()


def test_gravel_roughness_amplitude_is_built_by_the_layout_and_not_flattened(tiles, composed):
    sub = composed.scene.terrain.terrain_generator.sub_terrains["rt_gravel"]
    assert tuple(sub.noise_range) == (0.02, 0.02)              # main's knob loop wrote 0.02; gravel ignores it
    for (kind, level, _col), (_d, res) in tiles.items():
        if kind != "gravel":
            continue
        a = st.ladder_value("gravel", "rough_m", d_site_of(level), "rt")
        rough = (_heights(res) - _surface0(res)) * VS
        band = (R >= st.PAD_R_M + st.RT_BLEND_M) & (CHEB <= tk.SITE_HALF_WIDTH - 0.40 - st.RT_BLEND_M)
        amp = np.abs(rough[band]).max()
        assert within(amp, a) and amp <= 0.05 + 0.5 * VS + 1e-9, (level, amp, a)
        assert np.abs(rough[R <= st.PAD_R_M]).max() <= st.RT_GRAVEL_PAD_ROUGH_M + 0.5 * VS + 1e-9
        assert (_level(res) == 0).all() and (_surface0(res) == 0).all()      # flat; low mu is the contract's
        assert res[4].sum() == 0


def test_the_rt_report_line(tiles, capsys):
    """One summary line per kind x level (realised dose vs table), for the CHARTER."""
    rows = []
    for (kind, level, col), (_d, res) in sorted(tiles.items()):
        d = d_site_of(level)
        s0 = _surface0(res) * VS
        if kind in ("slopes", "hills"):
            val, tab = deg(plane_grad(s0).max()), st.ladder_value(kind, "grade_deg", d, "rt")
        elif kind in ("ledge_up", "ledge_down"):
            val, tab = abs(_layers(res)["tile"]["centre_level_m"]), st.ladder_value(kind, "height_m", d, "rt")
        elif kind == "pits":
            val, tab = -s0[_kind(res) == K.KIND_PIT].min(), st.ladder_value("pits", "depth_m", d, "rt")
        elif kind == "rubble":
            val, tab = ((_surface0(res) - _level(res)) * VS).max(), st.ladder_value("rubble", "height_max_m", d, "rt")
        else:
            val = np.abs((_heights(res) - _surface0(res)) * VS)[(R > 2.0) & (CHEB < 4.1)].max()
            tab = st.ladder_value("gravel", "rough_m", d, "rt")
        rows.append(f"{kind}/L{level}/c{col}: {val:.4f} vs {tab:.4f} ({val / tab:.3f})")
    print("[rt dose] " + "; ".join(rows))
    assert len(rows) == 3 * 14


# ================================================================================ eval kind rows
EVAL_BANDS = ("0.0:0.3", "0.3:0.6", "0.6:1.0")


def eval_env(kind: str, band: str) -> dict:
    env = {"KRABBY_FLAT_TERRAIN_MODE": "light", "KRABBY_FLAT_TERRAIN_FLAT_FRAC": "0.0",
           "KRABBY_FLAT_TERRAIN_GEOM": "site", "KRABBY_SITE_LADDER": "rt", "KRABBY_SITE_MIX": f"{kind}=20",
           "KRABBY_SITE_DIFF": band, "KRABBY_FLAT_TERRAIN_DIFF": "0.20:1.0"}
    if kind == "gravel":
        env["KRABBY_RT_FRICTION"] = "gravel:0.2:0.5"
    return env


@pytest.mark.parametrize("kind", st.RT_KINDS)
@pytest.mark.parametrize("band", EVAL_BANDS)
def test_eval_kind_rows_compose_one_kind_on_the_band(kind, band):
    cfg = compose(eval_env(kind, band), curriculum=False)
    tg = cfg.scene.terrain.terrain_generator
    assert tg.curriculum is False and tg.num_cols == 20
    props = {k: s.proportion for k, s in tg.sub_terrains.items()}
    key = tk.RT_SUBTERRAIN_KEYS[tk.RT_COLUMN_KINDS.index(kind)]
    assert props[key] == 1.0 and sum(props.values()) == 1.0
    lo, hi = (float(v) for v in band.split(":"))
    sub = tg.sub_terrains[key]
    assert tuple(sub.site_d_range) == (lo, hi) and sub.function is st.omni_site_terrain_rt
    # the eval generator's random mode: d ~ U(difficulty_range) -> d_site inside the band
    rng = np.random.default_rng(0)
    for d in rng.uniform(*tg.difficulty_range, size=5):
        assert lo - 1e-12 <= st.site_difficulty(d, sub) <= hi + 1e-12
    friction_armed = getattr(cfg, tk.RT_FRICTION_ATTR, None) is not None
    assert friction_armed == (kind == "gravel")
    if kind == "gravel":
        assert cfg.events.physics_material.params["gravel_range"] == (0.2, 0.5)


def test_one_eval_tile_per_kind_and_band_builds():
    for kind in st.RT_KINDS:
        for band in EVAL_BANDS:
            cfg = compose(eval_env(kind, band), curriculum=False)
            tg = cfg.scene.terrain.terrain_generator
            key = tk.RT_SUBTERRAIN_KEYS[tk.RT_COLUMN_KINDS.index(kind)]
            sub = tile_cfg(tg, key)
            np.random.seed(7)
            out = sub.function(0.6, sub, NUM_GOALS)
            lo, hi = (float(v) for v in band.split(":"))
            assert lo <= out[7]["tile"]["d"] <= hi and (body(out[7]["kind"]) == PIXEL_KIND[kind]).any()


# ================================================================================ knob surface and refusals
def test_rt_tables_restated_equal():
    assert tk.RT_LADDER == st.RT_LADDER == "rt"
    assert set(tk.RT_HAZARD_KINDS) == set(st.RT_KINDS)
    sr = K
    for kind, key in zip(tk.RT_COLUMN_KINDS, tk.RT_SUBTERRAIN_KEYS):
        assert sr.TILE_SUBTERRAIN_KEYS[sr.TILE_KINDS.index(kind)] == key
    assert sr.TILE_KINDS[:14] == sr.SITE_KINDS and sr.TILE_SUBTERRAIN_KEYS[:14] == sr.SITE_SUBTERRAIN_KEYS
    assert sr.PIXEL_KIND_NAMES[:16] == sr.KIND_NAMES and (sr.KIND_SLOPE, sr.KIND_HILL, sr.KIND_GRAVEL) == (16, 17, 18)
    assert tuple(rt_generator_from_source().sub_terrains) == tk.RT_SUBTERRAIN_KEYS
    assert tk.parse_site_mix(tk.RT_TRAIN_MIX, "rt") == {"flat": 3, "gravel": 2, "field_sparse": 2, "field_dense": 1,
                                                       "slopes": 2, "hills": 2, "ledge_up": 2, "ledge_down": 2,
                                                       "pits": 2, "rubble": 2}
    assert max(st.RT_CROSS_R.values()) <= 3.2


def test_parse_site_ladder_and_mix_on_rt():
    assert tk.parse_site_ladder(" RT ") == "rt"
    assert list(tk.parse_site_mix("slopes=20")) == list(tk.RT_COLUMN_KINDS)                  # auto: an rt-only kind
    assert list(tk.parse_site_mix("pits=20")) == list(K.SITE_KINDS)                    # a shared kind: the site table
    assert list(tk.parse_site_mix("pits=20", "rt")) == list(tk.RT_COLUMN_KINDS)
    for refused in tk.RT_REFUSED_KINDS:
        with pytest.raises(ValueError, match="refused under"):
            tk.parse_site_mix(f"slopes=1,{refused}=1")
        with pytest.raises(ValueError, match="refused under"):
            tk.parse_site_mix(f"{refused}=1", "rt")
    with pytest.raises(ValueError, match="exists only under"):
        tk.parse_site_mix("hills=2", None)
    with pytest.raises(ValueError, match="exists only under"):
        tk.parse_site_mix("gravel=2", "b4")


def test_rt_applier_refusals():
    base = dict(train_env())
    with pytest.raises(ValueError, match="required under"):
        compose(dict(base, KRABBY_SITE_MIX=""), curriculum=True)
    with pytest.raises(ValueError, match="refused under"):
        compose(dict(base, KRABBY_SITE_MIX="flat=10,walls=10"), curriculum=True)
    with pytest.raises(RuntimeError, match="gravel columns but KRABBY_RT_FRICTION is unset"):
        compose(dict(base, KRABBY_RT_FRICTION=""), curriculum=True)
    cfg = compose(dict(base, KRABBY_SITE_MIX="slopes=20", KRABBY_RT_FRICTION=""), curriculum=True)   # no gravel: ok
    assert getattr(cfg, tk.RT_FRICTION_ATTR, None) is None
    with pytest.raises(RuntimeError, match="rough-terrain generator"):           # a site generator under rt
        cfg = env_cfg()
        site_gen = types.SimpleNamespace(sub_terrains={"parkour_flat": instantiate("OmniFlatTerrainCfg")},
                                         difficulty_range=(0.2, 1.0), num_cols=20)
        cfg.scene.terrain.terrain_generator = site_gen
        tk.apply_omni_ring_knobs(cfg, base, site_classes=CLASSES)


def test_site_ladders_never_reach_an_rt_layout():
    """Today's ladder and b4 keep the site layouts (site band 2.0-3.6, site crossing radii, cfg roughness); the
    rt-only kinds do not exist there and the site-only kinds do not exist on rt."""
    for ladder in (None, "b4"):
        a = st.site_layout(np.random.RandomState(4), 0.7, layout_cfg("pits"), ladder)
        assert a.rough is None and a.cross_r == st.CROSS_R["pits"] == 4.1
        assert (st._Canvas(layout_cfg("pits")).r[a.kind == K.KIND_PIT] > st.RT_BAND_R_OUT_M).any()
        b = st.site_layout(np.random.RandomState(4), 0.7, layout_cfg("rubble"), ladder)
        assert b.cross_r == 4.1 and st.ladder_value("rubble", "height_max_m", 1.0, ladder) == 0.25
    for kind in ("slopes", "hills", "gravel"):
        with pytest.raises(ValueError, match="unknown site kind"):
            st.site_layout(np.random.RandomState(0), 0.5, layout_cfg(kind), None)
    for kind in tk.RT_REFUSED_KINDS:
        with pytest.raises(ValueError, match="unknown site kind"):
            st.site_layout(np.random.RandomState(0), 0.5, layout_cfg(kind), "rt")
    assert st.SITE_LADDER_BODIES == {"b4": st.omni_site_terrain_b4} and st.ladder_of(st.RT_BODY) == "rt"
    assert st.primary_metres("gravel", 1.0, "rt") == pytest.approx(0.05) and not st.has_primary("slopes", "rt")


# ================================================================================ per-kind telemetry on rt columns
def test_rt_columns_reach_the_per_kind_telemetry(monkeypatch):
    """The parkour term's site telemetry indexes ``site_raster.TILE_KINDS``: the rt-only columns (slopes / hills /
    gravel, indices after the site's 14) get their own ``site_level/<kind>`` keys, and ``site_metres/<kind>`` reads
    the rt ladder (gravel's roughness, pits' depth) and skips the degree-valued kinds. Runs the real class through
    ``test_site_curriculum.py``'s stand-in harness on a 4-column rt grid."""
    import importlib

    tsc = importlib.import_module("tests.unit.test_site_curriculum")
    sr, stc = tsc.sr, tsc._site_terrains()
    cols = ("parkour_flat", "site_pits", "rt_slopes", "rt_gravel")
    cross = {"site_pits": 3.2, "rt_slopes": 3.0, "rt_gravel": 3.0}

    def layers():
        frame = sr.TileFrame(size_xy=(10.0, 10.0), hs=0.08, vs=0.005, tile_px=(126, 126),
                             grid_rc=(tsc.N_LEVELS, len(cols)))
        shape = frame.global_shape
        tile_kind = np.array([[sr.TILE_SUBTERRAIN_KEYS.index(c) for c in cols]] * tsc.N_LEVELS, np.int8)
        cr = np.array([[cross.get(c, math.nan) for c in cols]] * tsc.N_LEVELS, np.float32)
        tile_d = np.array([[math.nan if c == "parkour_flat" else lv / 9 for c in cols] for lv in range(tsc.N_LEVELS)],
                          np.float32)
        return sr.make_test_layers(np.zeros(shape, np.int8), np.zeros(shape, np.float32),
                                   np.zeros(shape, np.float32), frame, tile_kind=tile_kind, tile_cross_r=cr,
                                   tile_d=tile_d)

    monkeypatch.setattr(tsc, "COLS", cols)
    monkeypatch.setattr(tsc, "_layers", layers)
    subs = {c: types.SimpleNamespace(function=(print if c == "parkour_flat" else stc.RT_BODY)) for c in cols}
    ev = tsc._term([0, 9, 4, 9], [0, 1, 2, 3], track=[0.9, 0.1, 0.9, 0.5], cmd=[1, 1, 1, 1],
                   r_max=[0.0, 3.5, 2.0, 3.1], subs=subs)
    assert ev._site_present == ("flat", "pits", "slopes", "gravel")
    assert set(ev._site_metres_kinds) == {"pits", "gravel"}
    ex = ev.reset(torch.arange(4))
    assert ex["site_level/slopes"] == 4.0 and ex["site_level/gravel"] == 9.0 and ex["site_level/pits"] == 9.0
    assert ex["site_metres/gravel"] == pytest.approx(stc.primary_metres("gravel", 1.0, "rt"))
    assert ex["site_metres/pits"] == pytest.approx(stc.primary_metres("pits", 1.0, "rt"))
    assert "site_metres/slopes" not in ex
