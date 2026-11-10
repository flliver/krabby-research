# SPDX-License-Identifier: BSD-3-Clause
"""``crab_hex_omni_task/mdp/site_raster.py`` -- the site kind raster (site teacher DESIGN §2, milestone M1).

CPU only, no Isaac: ``site_raster`` and its lazy siblings (``omni_exposure``, ``omni_clearance``, the
pure-torch ``exposure_stats``) are path-loaded under their real dotted names with stub parent
packages (the loader of ``test_omni_p2b_knobs.py``), and the env is a ``SimpleNamespace`` stand-in
carrying exactly what the module reads: ``scene.terrain`` (generator + cfg + origins), ``device`` and
the ``base_parkour`` term's tile frame.

Pinned here:

* the pixel / column kind constants (§2.1) and the emission-contract checker (§2.2);
* module purity (no isaaclab / omni / parkour package import at module scope);
* :func:`global_layer` = ``exposure_stats.global_height_field``; :func:`layer_under` =
  ``omni_exposure.tiled_pixel_index`` + ``global_pixel_index``; :func:`world_pixel` = the own-tile
  lookup on 10^4 random points (numpy and torch), incl. off-terrain;
* :func:`site_layers` assembly, dtypes, metres, tile arrays, the cache (built once, ``False`` on a
  non-site generator which also builds no height field), ``support_m is surface_m`` and
  ``surface_m is`` the ``omni_clearance`` cache;
* the env lookups (pit / ledge / seam / off-terrain / without site layers; ``(n,)`` and ``(n, k)``);
* the dump round trip (site and non-site) and :func:`make_test_layers`.
"""
from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
OMNI_MDP = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/mdp"
PARKOURS_DIR = REPO / "parkour/parkour_isaaclab/envs/mdp/parkours"
SITE_RASTER_PY = OMNI_MDP / "site_raster.py"


def _load_real_module(dotted: str, path: Path) -> types.ModuleType:
    """Get-or-create loader of ``test_omni_p2b_knobs.py``: the real source dirs as the parent
    packages' ``__path__``, their ``__init__`` never executed (so relative imports resolve)."""
    if dotted in sys.modules:
        return sys.modules[dotted]
    parts = dotted.split(".")
    directory = path.parent
    for i in range(len(parts) - 1, 0, -1):
        name = ".".join(parts[:i])
        pkg = sys.modules.get(name)
        if pkg is None:
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(directory)]
            sys.modules[name] = pkg
        elif str(directory) not in getattr(pkg, "__path__", []):
            pkg.__path__ = list(getattr(pkg, "__path__", [])) + [str(directory)]
        directory = directory.parent
    spec = importlib.util.spec_from_file_location(dotted, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    setattr(sys.modules[".".join(parts[:-1])], parts[-1], mod)
    return mod


_PKG = "parkour_tasks.crab_hex_omni_task.mdp"
xs = _load_real_module("parkour_isaaclab.envs.mdp.parkours.exposure_stats", PARKOURS_DIR / "exposure_stats.py")
ox = _load_real_module(f"{_PKG}.omni_exposure", OMNI_MDP / "omni_exposure.py")
oc = _load_real_module(f"{_PKG}.omni_clearance", OMNI_MDP / "omni_clearance.py")
sr = _load_real_module(f"{_PKG}.site_raster", SITE_RASTER_PY)

# The shipped geometry: 10 m tiles, 0.08 m pixels -> 126 px stored, placed 125 px apart; 5 mm units.
SIZE, HS, TILE_PX, GRID, VS = (10.0, 10.0), 0.08, (126, 126), (3, 4), 0.005
FRAME = sr.TileFrame(size_xy=SIZE, hs=HS, vs=VS, tile_px=TILE_PX, grid_rc=GRID)
PIT_Z, LEDGE_Z, SEAM_Z, ROUGH = -0.30, 0.20, 0.15, 3          # metres, metres, metres, units
NAMES = np.array([[["parkour_flat"], ["site_pits"], ["site_ledge_down"], ["site_walls"]],
                  [["site_field_sparse"], ["site_pits"], ["site_ledge_down"], ["site_walls"]],
                  [["site_ramps"], ["site_pits"], ["omni_field_dense"], ["site_walls"]]])


def _centre(row: int, col: int) -> torch.Tensor:
    """World xy of tile (row, col)'s centre, with the generator's whole-terrain centring."""
    return torch.tensor([(row + 0.5) * SIZE[0] - GRID[0] * SIZE[0] / 2,
                         (col + 0.5) * SIZE[1] - GRID[1] * SIZE[1] / 2])


def _origins() -> torch.Tensor:
    o = torch.zeros(GRID[0], GRID[1], 3)
    for r in range(GRID[0]):
        for c in range(GRID[1]):
            o[r, c, :2] = _centre(r, c)
            o[r, c, 2] = 0.01 * (r + c)
    return o


def _xy(row: int, col: int, px: float, py: float) -> tuple[float, float]:
    """Tile-local pixel (may lie past the tile) -> world xy, on tile (row, col)'s frame."""
    c = _centre(row, col)
    return float(c[0] - 0.5 * SIZE[0] + px * HS), float(c[1] - 0.5 * SIZE[1] + py * HS)


def _site_arrays() -> dict[str, np.ndarray]:
    """Per-tile site arrays as the generator stores them (DESIGN §2.2): a pit on (1, 1), a raised
    ledge pad on (1, 2), a wall just past (1, 1)'s +x seam (on (2, 1)), roughness-free otherwise."""
    shape = (*GRID, *TILE_PX)
    kind = np.zeros(shape, dtype=np.int8)
    level = np.zeros(shape, dtype=np.int16)
    surface0 = np.zeros(shape, dtype=np.int16)
    kind[1, 1, 30:41, 30:41] = sr.KIND_PIT
    surface0[1, 1, 30:41, 30:41] = round(PIT_Z / VS)              # level stays 0: a pit reads its rim
    kind[1, 2, 50:76, 50:76] = sr.KIND_LEDGE
    level[1, 2, 50:76, 50:76] = round(LEDGE_Z / VS)
    surface0[1, 2, 50:76, 50:76] = round(LEDGE_Z / VS)
    kind[2, 1, 0:9, 58:68] = sr.KIND_BLOCKER_WALL
    surface0[2, 1, 0:9, 58:68] = round(SEAM_Z / VS)
    cross = np.full(GRID, np.nan, dtype=np.float32)
    cross[:, 1] = 4.1
    cross[:, 2] = 3.2
    d = np.full(GRID, np.nan, dtype=np.float32)
    d[:, 1:] = np.array([0.0, 0.5, 1.0], dtype=np.float32)[:, None]
    centre = np.zeros(GRID, dtype=np.float32)
    centre[1, 2] = LEDGE_Z
    return {"kind": kind, "level": level, "surface0": surface0, "material": np.zeros(shape, dtype=np.int8),
            "cross_r": cross, "d": d, "centre_level_m": centre}


def _height_fields(arrays) -> np.ndarray:
    """The generator's height fields: surface0 plus a constant roughness offset everywhere."""
    return (arrays["surface0"] + ROUGH).astype(np.int16)


def _sub(key, prop):
    return types.SimpleNamespace(proportion=prop, site_d_range=(0.1, 0.9), gen_d_range=(0.0, 1.0)) \
        if key.startswith("site_") else types.SimpleNamespace(proportion=prop)


def _env(*, site=True, n=2, rows=(1, 0), cols=(1, 0), arrays=None):
    arrays = _site_arrays() if arrays is None else arrays
    gen = types.SimpleNamespace(height_fields=_height_fields(arrays), terrain_names=NAMES,
                                terrain_origins=_origins().numpy().astype(np.float64))
    for key, attr in sr.GENERATOR_ATTRS.items():
        setattr(gen, attr, arrays[key] if site else None)
    subs = {"parkour_flat": _sub("parkour_flat", 0.2), "site_field_sparse": _sub("site_field_sparse", 0.1),
            "site_pits": _sub("site_pits", 0.3), "site_ledge_down": _sub("site_ledge_down", 0.2),
            "site_walls": _sub("site_walls", 0.2), "site_ramps": _sub("site_ramps", 0.0)}
    tgc = types.SimpleNamespace(size=SIZE, horizontal_scale=HS, vertical_scale=VS, num_cols=10, sub_terrains=subs)
    levels, types_ = torch.tensor(rows[:n]), torch.tensor(cols[:n])
    terrain = types.SimpleNamespace(terrain_generator_class=gen, cfg=types.SimpleNamespace(terrain_generator=tgc),
                                    terrain_origins=_origins(), terrain_levels=levels, terrain_types=types_)
    env_origins = torch.stack([torch.cat([_centre(int(r), int(c)), torch.zeros(1)]) for r, c in zip(levels, types_)])
    term = types.SimpleNamespace(env_origins=env_origins, terrain=terrain, _tile_size=SIZE, _hscale=HS,
                                 _tile_px=TILE_PX, _grid_rc=GRID)
    return types.SimpleNamespace(num_envs=n, device="cpu", scene=types.SimpleNamespace(terrain=terrain),
                                 parkour_manager=types.SimpleNamespace(get_term=lambda name: term))


def _pts(*xy):
    return torch.tensor([p[0] for p in xy]), torch.tensor([p[1] for p in xy])


# ======================================================================= constants (§2.1)
def test_pixel_kind_constants():
    names = ["GROUND", "PIT", "TRENCH", "LEDGE", "TERRACE", "SNAG", "STRADDLE", "BLOCKER_CHEST", "BLOCKER_WALL",
             "RUBBLE", "RAMP", "FIELD_BLOCK", "FIELD_PIT", "RESERVED_SOFT", "RESERVED_LOWMU", "RESERVED_15"]
    assert [getattr(sr, f"KIND_{n}") for n in names] == list(range(16))
    assert sr.KIND_NAMES == ("ground", "pit", "trench", "ledge", "terrace", "snag", "straddle", "blocker_chest",
                             "blocker_wall", "rubble", "ramp", "field_block", "field_pit", "soft", "lowmu",
                             "reserved15")
    assert sr.VOID_KINDS == (1, 2) and sr.BLOCKER_KINDS == (7, 8)
    assert np.iinfo(np.int8).max >= len(sr.KIND_NAMES) - 1
    assert sr.KIND_FIELD_PIT not in sr.VOID_KINDS            # legacy field pits are never void


def test_column_kind_constants():
    assert len(sr.SITE_KINDS) == len(sr.SITE_SUBTERRAIN_KEYS) == len(sr.SITE_KIND_POLICY) == 14
    assert sr.SITE_SUBTERRAIN_KEYS[0] == "parkour_flat"
    for name, key in zip(sr.SITE_KINDS[3:], sr.SITE_SUBTERRAIN_KEYS[3:]):
        assert key == f"site_{name}"
    assert sr.SITE_SUBTERRAIN_KEYS[1:3] == ("site_field_sparse", "site_field_dense")
    pol = dict(zip(sr.SITE_KINDS, sr.SITE_KIND_POLICY))
    assert (sr.POLICY_LEGACY, sr.POLICY_CROSS, sr.POLICY_VOID, sr.POLICY_BLOCKER) == (0, 1, 2, 3)
    assert {k for k, v in pol.items() if v == sr.POLICY_LEGACY} == {"flat", "field_sparse", "field_dense"}
    assert {k for k, v in pol.items() if v == sr.POLICY_VOID} == {"pits", "trench"}
    assert {k for k, v in pol.items() if v == sr.POLICY_BLOCKER} == {"chests", "walls"}
    assert {k for k, v in pol.items() if v == sr.POLICY_CROSS} == {
        "ledge_down", "ledge_up", "terraces", "snags", "straddle", "rubble", "ramps"}
    assert sr.SITE_LAYERS_ATTR == "_krabby_site_layers"
    assert sr.TERRAIN_TERM_NAME == oc.TERRAIN_TERM_NAME


def test_module_is_pure_and_path_loads_standalone():
    """No isaaclab / omni / parkour package import at module scope: a bare path-load (no package) works
    and pulls in nothing but numpy / torch."""
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('site_raster_standalone', {str(SITE_RASTER_PY)!r})\n"
        "mod = importlib.util.module_from_spec(spec); sys.modules['site_raster_standalone'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('isaaclab', 'omni', 'pxr', 'parkour_tasks',"
        " 'parkour_isaaclab', 'isaacsim'))\n"
        "assert not bad, bad\n"
        "assert mod.KIND_PIT == 1 and mod.global_layer is not None\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr


# ======================================================================= emission contract (§2.2)
def _tile_layers(shape=(124, 124)):
    return {"kind": np.zeros(shape, np.int8), "level": np.zeros(shape, np.int16),
            "surface0": np.zeros(shape, np.int16), "material": np.zeros(shape, np.int8),
            "tile": {"cross_r": float("nan"), "d": 0.5, "centre_level_m": 0.0}}


def test_contract_constants():
    assert sr.LAYER_DTYPES == {"kind": np.int8, "level": np.int16, "surface0": np.int16, "material": np.int8}
    assert sr.LAYER_INIT == {"kind": sr.KIND_GROUND, "level": 0, "surface0": 0, "material": 0}
    assert set(sr.TILE_INIT) == {"cross_r", "d", "centre_level_m"}
    assert math.isnan(sr.TILE_INIT["cross_r"]) and math.isnan(sr.TILE_INIT["d"]) and sr.TILE_INIT["centre_level_m"] == 0
    assert sr.GENERATOR_ATTRS == {"kind": "site_kind", "level": "site_level", "surface0": "site_surface0",
                                  "material": "site_material", "cross_r": "site_cross_r", "d": "site_d",
                                  "centre_level_m": "site_centre_level_m"}


def test_check_tile_layers_accepts_the_contract_and_refuses_the_rest():
    sr.check_tile_layers(_tile_layers(), (124, 124))
    bad_dtype = _tile_layers()
    bad_dtype["level"] = bad_dtype["level"].astype(np.int32)
    bad_shape = _tile_layers()
    bad_shape["kind"] = np.zeros((126, 126), np.int8)
    missing = _tile_layers()
    del missing["material"]
    extra = dict(_tile_layers(), riser=np.zeros((124, 124), np.int16))
    bad_tile = _tile_layers()
    del bad_tile["tile"]["d"]
    nan_centre = _tile_layers()
    nan_centre["tile"]["centre_level_m"] = float("nan")
    for layers in (bad_dtype, bad_shape, missing, extra, bad_tile, nan_centre, "not a dict"):
        with pytest.raises(ValueError):
            sr.check_tile_layers(layers, (124, 124))


# ======================================================================= assembly (§2.3)
def test_global_layer_is_the_global_height_field_assembly():
    rng = np.random.default_rng(0)
    for dtype in (np.int8, np.int16, np.float32):
        per_tile = rng.integers(-100, 100, size=(*GRID, 7, 5)).astype(dtype)
        g = sr.global_layer(per_tile)
        assert g.dtype == torch.from_numpy(per_tile).dtype and tuple(g.shape) == (GRID[0] * 7, GRID[1] * 5)
        assert torch.equal(g, xs.global_height_field(torch.from_numpy(per_tile)))
        before = g.clone()
        per_tile += 1                                             # a copy, not a view of the generator array
        assert torch.equal(g, before)
    with pytest.raises(ValueError):
        sr.global_layer(np.zeros((3, 4, 5)))


def test_site_layers_built_once_with_the_contract_dtypes():
    arrays = _site_arrays()
    env = _env(arrays=arrays)
    layers = sr.site_layers(env)
    assert isinstance(layers, sr.SiteLayers) and getattr(env, sr.SITE_LAYERS_ATTR) is layers
    shape = (GRID[0] * 126, GRID[1] * 126)
    assert layers.kind.dtype == torch.int8 and layers.material.dtype == torch.int8
    for t in (layers.kind, layers.level_m, layers.surface0_m, layers.material, layers.surface_m):
        assert tuple(t.shape) == shape
    assert layers.level_m.dtype == layers.surface0_m.dtype == torch.float32
    assert torch.equal(layers.kind, xs.global_height_field(torch.from_numpy(arrays["kind"])))
    assert torch.equal(layers.level_m, xs.global_height_field(torch.from_numpy(arrays["level"])).float() * VS)
    assert torch.equal(layers.surface0_m, oc.global_height_field_m(arrays["surface0"], VS))
    # the nominal surface is the omni_clearance cache itself, and support IS surface (phase 1)
    assert layers.surface_m is getattr(env, oc.HEIGHT_FIELD_CACHE_ATTR)
    assert torch.equal(layers.surface_m, oc.global_height_field_m(_height_fields(arrays), VS))
    assert layers.support_m is layers.surface_m
    assert layers.frame == FRAME and layers.frame.global_shape == shape
    # tile arrays
    want_kind = [[0, 3, 5, 11], [1, 3, 5, 11], [13, 3, -1, 11]]
    assert layers.tile_kind.dtype == torch.int8 and layers.tile_kind.tolist() == want_kind
    assert torch.equal(layers.tile_cross_r.isnan(), torch.from_numpy(np.isnan(arrays["cross_r"])))
    assert float(layers.tile_cross_r[0, 1]) == pytest.approx(4.1) and layers.tile_d.dtype == torch.float32
    assert torch.equal(layers.tile_centre_level_m, torch.from_numpy(arrays["centre_level_m"]))
    # cached: a later change to the generator is not re-read, the same object comes back
    arrays["kind"][:] = sr.KIND_RUBBLE
    assert sr.site_layers(env) is layers and int(layers.kind.max()) == sr.KIND_BLOCKER_WALL


def test_site_layers_reads_the_importer_not_the_parkour_term():
    """Safe inside the parkour term's own __init__: no parkour manager needed to build."""
    env = _env()
    del env.parkour_manager
    assert sr.site_layers(env) is not None


def test_non_site_generator_caches_false_and_builds_nothing():
    env = _env(site=False)
    assert sr.site_layers(env) is None
    assert getattr(env, sr.SITE_LAYERS_ATTR) is False
    assert not hasattr(env, oc.HEIGHT_FIELD_CACHE_ATTR)       # no height field built either
    env.scene.terrain.terrain_generator_class.site_kind = np.zeros((*GRID, *TILE_PX), np.int8)
    assert sr.site_layers(env) is None                        # "looked, none" is final
    bare = types.SimpleNamespace(device="cpu")                # a stand-in without a terrain importer
    assert sr.site_layers(bare) is None and getattr(bare, sr.SITE_LAYERS_ATTR) is False


def test_site_layers_refuses_a_shape_mismatch():
    arrays = _site_arrays()
    arrays["level"] = arrays["level"][:, :, :-1]
    with pytest.raises(ValueError, match="site_level"):
        sr.site_layers(_env(arrays=arrays))


# ======================================================================= lookups (§2.3)
def test_layer_under_is_the_tiled_pixel_index_lookup():
    """10^4 random points around random own tiles (incl. off-terrain): layer_under reads exactly the
    tiled_pixel_index / global_pixel_index pixel, off_value where the tile does not exist."""
    g = torch.Generator().manual_seed(7)
    n, k = 2000, 5
    row = torch.randint(0, GRID[0], (n,), generator=g)
    col = torch.randint(0, GRID[1], (n,), generator=g)
    centre = torch.stack([_centre(int(r), int(c)) for r, c in zip(row, col)])
    x = centre[:, :1] + (torch.rand(n, k, generator=g) - 0.5) * 2.6 * SIZE[0]
    y = centre[:, 1:] + (torch.rand(n, k, generator=g) - 0.5) * 2.6 * SIZE[1]
    layer = torch.randint(-1000, 1000, FRAME.global_shape, generator=g, dtype=torch.int16)
    out = sr.layer_under(layer, x, y, centre, row, col, FRAME, -7)
    nr, nc, ix, iy, on, _ = ox.tiled_pixel_index(x, y, centre, row, col, SIZE, HS, TILE_PX, GRID)
    gx, gy = ox.global_pixel_index(nr, nc, ix, iy, TILE_PX)
    assert out.dtype == torch.int16 and out.shape == (n, k)
    assert torch.equal(out, torch.where(on, layer[gx, gy], torch.tensor(-7, dtype=torch.int16)))
    assert 0 < int((~on).sum()) < on.numel()                 # both branches exercised
    # a float layer through layer_under equals omni_clearance.terrain_height_under
    hf = oc.global_height_field_m(_height_fields(_site_arrays()), VS)
    assert torch.equal(sr.layer_under(hf, x, y, centre, row, col, FRAME, 0.0),
                       oc.terrain_height_under(x, y, centre, row, col, hf, SIZE, HS, TILE_PX, GRID))


def test_world_pixel_equals_the_own_tile_lookup_on_random_points():
    """10^4 random points over the terrain and past it: world_pixel (origins only) = tiled_pixel_index
    with the point's own tile, bit for bit; torch and numpy agree; any shape."""
    g = torch.Generator().manual_seed(11)
    o = _origins()
    n = 10_000
    half = torch.tensor([GRID[0] * SIZE[0] / 2, GRID[1] * SIZE[1] / 2])
    xy = (torch.rand(n, 2, generator=g) * 2 - 1) * (half + 3.0)              # 3 m past every outer edge
    x, y = xy[:, 0], xy[:, 1]
    # the point's own tile, independently: integer arithmetic on the generator's (uncentred) frame
    r_true = torch.floor((x.double() + half[0].double()) / SIZE[0]).long()
    c_true = torch.floor((y.double() + half[1].double()) / SIZE[1]).long()
    rc, cc = r_true.clamp(0, GRID[0] - 1), c_true.clamp(0, GRID[1] - 1)
    nr, nc, ix, iy, on, _ = ox.tiled_pixel_index(x, y, o[rc, cc, :2], rc, cc, SIZE, HS, TILE_PX, GRID)
    want_gx, want_gy = ox.global_pixel_index(nr, nc, ix, iy, TILE_PX)
    gx, gy, on_t = sr.world_pixel(x, y, FRAME, o)
    assert torch.equal(gx, want_gx) and torch.equal(gy, want_gy) and torch.equal(on_t, on)
    inside = (r_true >= 0) & (r_true < GRID[0]) & (c_true >= 0) & (c_true < GRID[1])
    assert torch.equal(on_t, inside) and 0 < int((~inside).sum()) < n
    # numpy in -> numpy out, same values; (n, k) shapes kept
    gxn, gyn, onn = sr.world_pixel(x.numpy(), y.numpy(), FRAME, o.numpy())
    assert isinstance(gxn, np.ndarray) and np.array_equal(gxn, gx.numpy()) and np.array_equal(gyn, gy.numpy())
    assert np.array_equal(onn, on.numpy())
    gx2, _, _ = sr.world_pixel(x.reshape(100, 100), y.reshape(100, 100), FRAME, o)
    assert torch.equal(gx2, gx.reshape(100, 100))
    with pytest.raises(ValueError):
        sr.world_pixel(x, y, FRAME, o[:2])


def test_world_pixel_matches_a_neighbour_own_tile_away_from_pixel_edges():
    """A point read from a NEIGHBOUR's frame (the seam-aware path) lands on the same global pixel,
    for points kept 0.1 px away from every rounding / seam boundary."""
    g = torch.Generator().manual_seed(3)
    o = _origins()
    n = 4000
    r = torch.randint(0, GRID[0], (n,), generator=g)
    c = torch.randint(0, GRID[1], (n,), generator=g)
    px = torch.randint(1, 125, (n, 2), generator=g).double() + (torch.rand(n, 2, generator=g, dtype=torch.float64)
                                                                 - 0.5) * 0.8
    x = (o[r, c, 0].double() - SIZE[0] / 2 + px[:, 0] * HS)
    y = (o[r, c, 1].double() - SIZE[1] / 2 + px[:, 1] * HS)
    own_r = (r + torch.randint(-1, 2, (n,), generator=g)).clamp(0, GRID[0] - 1)
    own_c = (c + torch.randint(-1, 2, (n,), generator=g)).clamp(0, GRID[1] - 1)
    nr, nc, ix, iy, on, _ = ox.tiled_pixel_index(x, y, o[own_r, own_c, :2].double(), own_r, own_c, SIZE, HS,
                                                  TILE_PX, GRID)
    want = ox.global_pixel_index(nr, nc, ix, iy, TILE_PX)
    gx, gy, on_t = sr.world_pixel(x, y, FRAME, o.double())
    assert bool(on.all()) and bool(on_t.all())
    assert torch.equal(gx, want[0]) and torch.equal(gy, want[1])


def test_a_standalone_load_reads_dumps_and_finds_the_siblings_by_absolute_name(tmp_path):
    """The W4 tools path-load this file standalone (no package): world_pixel needs no sibling, and
    the env lookups fall back to the task package's absolute name (a Kit process)."""
    spec = importlib.util.spec_from_file_location("site_raster__standalone_test", SITE_RASTER_PY)
    alone = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = alone                       # dataclasses resolve the module by name
    spec.loader.exec_module(alone)
    assert not alone.__package__
    path = tmp_path / "layers.npz"
    sr.dump_site_layers(_env(), path)
    d = alone.load_layers_dump(path)
    g = np.random.default_rng(2)
    x = g.uniform(-18.0, 18.0, 500)
    y = g.uniform(-23.0, 23.0, 500)
    got = alone.world_pixel(x, y, d["frame"], d["terrain_origins"])
    want = sr.world_pixel(x, y, FRAME, d["terrain_origins"])
    assert all(np.array_equal(a, b) for a, b in zip(got, want))
    assert alone._omni_exposure() is ox and alone._omni_clearance() is oc   # by absolute name
    assert sr._omni_exposure() is ox                                         # by package (relative)


def test_kind_and_level_under_pit_ledge_seam_and_off_terrain():
    env = _env(n=2, rows=(1, 0), cols=(1, 0))
    pit, bare, ledge = _xy(1, 1, 35, 35), _xy(1, 1, 62, 62), _xy(1, 2, 63, 63)   # the ledge pad is on (1, 2)
    seam = _xy(1, 1, 125 + 4, 63)                      # 0.32 m past (1, 1)'s +x seam: (2, 1)'s wall
    off = _xy(0, 0, -4, 62)                            # past the outermost tile
    x = torch.tensor([[pit[0], bare[0], ledge[0], seam[0]], [off[0], *_xy(0, 0, 62, 62)[:1], pit[0], seam[0]]])
    y = torch.tensor([[pit[1], bare[1], ledge[1], seam[1]], [off[1], *_xy(0, 0, 62, 62)[1:], pit[1], seam[1]]])
    kind = sr.kind_under(env, x, y)
    level = sr.level_under(env, x, y)
    assert kind.dtype == torch.int8 and kind.shape == (2, 4) and level.dtype == torch.float32
    assert kind.tolist() == [[sr.KIND_PIT, sr.KIND_GROUND, sr.KIND_LEDGE, sr.KIND_BLOCKER_WALL],
                             [sr.KIND_GROUND, sr.KIND_GROUND, sr.KIND_PIT, sr.KIND_BLOCKER_WALL]]
    assert torch.allclose(level, torch.tensor([[0.0, 0.0, LEDGE_Z, 0.0], [0.0, 0.0, 0.0, 0.0]]), atol=1e-7)
    # the surface (roughness included) and support: pit floor, ledge top, wall top, 0 off the terrain
    surf = sr.surface_under(env, x, y)
    r = ROUGH * VS
    want = torch.tensor([[PIT_Z + r, r, LEDGE_Z + r, SEAM_Z + r], [0.0, r, PIT_Z + r, SEAM_Z + r]])
    assert torch.allclose(surf, want, atol=1e-6)
    assert torch.equal(sr.support_under(env, x, y), surf)
    # (n,) inputs too
    k1 = sr.kind_under(env, x[:, 0], y[:, 0])
    assert k1.shape == (2,) and k1.tolist() == [sr.KIND_PIT, sr.KIND_GROUND]


def test_surface_under_equals_omni_clearance_with_and_without_site_layers():
    g = torch.Generator().manual_seed(5)
    x = _centre(1, 1)[0] + (torch.rand(2, 50, generator=g) - 0.5) * 24.0
    y = _centre(1, 1)[1] + (torch.rand(2, 50, generator=g) - 0.5) * 24.0
    site_env, plain_env = _env(), _env(site=False)
    term = plain_env.parkour_manager.get_term("base_parkour")
    hf = oc.global_height_field_m(_height_fields(_site_arrays()), VS)
    want = oc.terrain_height_under(x, y, term.env_origins[:, :2], term.terrain.terrain_levels,
                                   term.terrain.terrain_types, hf, SIZE, HS, TILE_PX, GRID)
    assert torch.equal(sr.surface_under(site_env, x, y), want)
    assert torch.equal(sr.surface_under(plain_env, x, y), want)
    assert torch.equal(sr.support_under(plain_env, x, y), want)
    assert sr.site_layers(plain_env) is None                     # the clearance path, not site layers


def test_without_site_layers_kind_is_ground_and_level_is_zero():
    env = _env(site=False)
    del env.parkour_manager                                      # not even consulted
    x, y = torch.rand(2, 6), torch.rand(2, 6)
    k, lv = sr.kind_under(env, x, y), sr.level_under(env, x, y)
    assert k.dtype == torch.int8 and k.shape == (2, 6) and not k.any()
    assert lv.dtype == torch.float32 and lv.shape == (2, 6) and not lv.any()


def test_lookups_refuse_a_parkour_term_without_the_tile_frame():
    env = _env()
    env.parkour_manager = types.SimpleNamespace(get_term=lambda name: types.SimpleNamespace())
    with pytest.raises(RuntimeError, match="base_parkour"):
        sr.kind_under(env, torch.zeros(2), torch.zeros(2))


# ======================================================================= dump (§2.5)
def test_dump_round_trip_on_a_site_generator(tmp_path):
    arrays = _site_arrays()
    env = _env(arrays=arrays)
    path = tmp_path / "layers_dump"                               # no .npz suffix: written at exactly this path
    sr.dump_site_layers(env, path)
    assert path.exists()
    d = sr.load_layers_dump(path)
    assert d["site"] is True and d["frame"] == FRAME
    assert d["vs"] == VS and d["hs"] == HS and d["size_xy"] == SIZE and d["tile_px"] == TILE_PX
    assert d["grid_rc"] == GRID
    for key, dtype in sr.LAYER_DTYPES.items():
        assert d[key].dtype == dtype
        assert np.array_equal(d[key], xs.global_height_field(torch.from_numpy(arrays[key])).numpy())
    assert d["height_field"].dtype == np.int16
    assert np.array_equal(d["height_field"], xs.global_height_field(torch.from_numpy(_height_fields(arrays))).numpy())
    assert d["tile_kind"].dtype == np.int8 and d["tile_kind"].tolist() == [[0, 3, 5, 11], [1, 3, 5, 11], [13, 3, -1, 11]]
    for key in ("cross_r", "d", "centre_level_m"):
        assert d[f"tile_{key}"].dtype == np.float32
        assert np.array_equal(d[f"tile_{key}"], arrays[key], equal_nan=True)
    assert d["terrain_names"].shape == GRID and d["terrain_names"][1, 1] == "site_pits"
    assert d["terrain_origins"].shape == (*GRID, 3) and np.allclose(d["terrain_origins"], _origins().numpy())
    assert d["mix"] == "flat=2,field_sparse=1,pits=3,ledge_down=2,walls=2"
    assert d["site_d_range"] == (0.1, 0.9) and d["gen_d_range"] == (0.0, 1.0)
    # the offline lookup: world_pixel on the dump reads what the live lookups read
    pit, ledge, seam = _xy(1, 1, 35, 35), _xy(1, 2, 63, 63), _xy(1, 1, 125 + 4, 63)
    xs_, ys_ = np.array([pit[0], ledge[0], seam[0]]), np.array([pit[1], ledge[1], seam[1]])
    gx, gy, on = sr.world_pixel(xs_, ys_, d["frame"], d["terrain_origins"])
    assert on.all() and d["kind"][gx, gy].tolist() == [sr.KIND_PIT, sr.KIND_LEDGE, sr.KIND_BLOCKER_WALL]
    assert np.allclose(d["level"][gx, gy] * d["vs"], [0.0, LEDGE_Z, 0.0])
    live = sr.kind_under(env, torch.from_numpy(np.stack([xs_, xs_])), torch.from_numpy(np.stack([ys_, ys_])))
    assert live[0].tolist() == d["kind"][gx, gy].tolist()


def test_dump_on_a_non_site_generator_writes_the_height_field_and_frame_only(tmp_path):
    env = _env(site=False)
    path = tmp_path / "ring.npz"
    sr.dump_site_layers(env, path)
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {"height_field", "vs", "hs", "size_xy", "tile_px", "grid_rc", "terrain_origins", "site"}
    d = sr.load_layers_dump(path)
    assert d["site"] is False and d["frame"] == FRAME and "mix" not in d
    assert not hasattr(env, sr.SITE_LAYERS_ATTR) and not hasattr(env, oc.HEIGHT_FIELD_CACHE_ATTR)


# ======================================================================= make_test_layers
def test_make_test_layers_global_and_per_tile_agree():
    arrays = _site_arrays()
    level_m = arrays["level"].astype(np.float32) * VS
    surface0_m = arrays["surface0"].astype(np.float32) * VS
    a = sr.make_test_layers(arrays["kind"], level_m, surface0_m, FRAME)
    b = sr.make_test_layers(sr.global_layer(arrays["kind"]), sr.global_layer(level_m), sr.global_layer(surface0_m),
                            FRAME, tile_kind=np.zeros(GRID, np.int8), tile_cross_r=arrays["cross_r"],
                            tile_centre_level_m=arrays["centre_level_m"])
    assert torch.equal(a.kind, b.kind) and torch.equal(a.level_m, b.level_m) and torch.equal(a.surface0_m, b.surface0_m)
    assert a.kind.dtype == torch.int8 and a.level_m.dtype == torch.float32 and a.material.dtype == torch.int8
    assert a.support_m is a.surface_m and b.support_m is b.surface_m
    assert torch.equal(a.surface_m, a.surface0_m) and a.surface_m is not a.surface0_m
    assert a.tile_kind.tolist() == [[-1] * 4] * 3 and bool(a.tile_cross_r.isnan().all()) and bool(a.tile_d.isnan().all())
    assert not a.tile_centre_level_m.any() and not a.material.any()
    assert float(b.tile_cross_r[0, 1]) == pytest.approx(4.1) and b.tile_kind.dtype == torch.int8
    with pytest.raises(ValueError):
        sr.make_test_layers(arrays["kind"][:, :, :-1], level_m, surface0_m, FRAME)
    with pytest.raises(ValueError):
        sr.make_test_layers(arrays["kind"], level_m, surface0_m, FRAME, tile_d=np.zeros((2, 2)))
    with pytest.raises(TypeError):
        sr.make_test_layers(arrays["kind"], level_m, surface0_m, FRAME, tile_names=None)


def test_make_test_layers_drives_the_lookups_on_a_stand_in_env():
    """The documented use: install on a stand-in that has only the parkour term's tile frame."""
    arrays = _site_arrays()
    layers = sr.make_test_layers(arrays["kind"], arrays["level"].astype(np.float32) * VS,
                                 arrays["surface0"].astype(np.float32) * VS, FRAME)
    env = _env()
    env.scene = None                                             # no importer: the installed layers are used
    setattr(env, sr.SITE_LAYERS_ATTR, layers)
    x, y = _pts(_xy(1, 1, 35, 35), _xy(0, 0, 62, 62))
    assert sr.kind_under(env, x, y).tolist() == [sr.KIND_PIT, sr.KIND_GROUND]
    assert torch.allclose(sr.surface_under(env, x, y), torch.tensor([PIT_Z, 0.0]))
    assert torch.allclose(sr.support_under(env, x, y), torch.tensor([PIT_Z, 0.0]))
