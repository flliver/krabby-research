# SPDX-License-Identifier: BSD-3-Clause
"""The construction-site KIND RASTER: per-pixel feature kinds and levels, and their world lookups
(site teacher, ``experiments/2026-10-28_site_teacher/DESIGN.md`` §2, milestone M1). DEFAULT OFF.

Nothing here runs unless the terrain generator carries site layers, which only the site
sub-terrains (``KRABBY_FLAT_TERRAIN_GEOM=site``) emit: on ring / field / flat generators
:func:`site_layers` returns None (cached as ``False`` on the env), :func:`kind_under` reads
``KIND_GROUND`` and :func:`level_under` reads 0 without building a tensor, and no cfg is touched.

Pixel kinds (:data:`KIND_NAMES`, int8)
--------------------------------------
A pixel's kind is the FEATURE stamped there, independent of roughness: void kinds (pits,
trench; :data:`VOID_KINDS`) are what the void penalty and STOP-0's "foot over void" read;
blocker kinds (chests / tables, walls; :data:`BLOCKER_KINDS`) are what the blocked-path slack,
the crash recorder and gate e read. Legacy field pits are ``KIND_FIELD_PIT``, never void (3-9 cm
at the trained band). ``KIND_RESERVED_*`` (and the ``material`` layer) exist for phase 2's soil and
friction classes and are never emitted in phase 1.

Column (tile) kinds (:data:`SITE_KINDS`) index the site generator's sub-terrains one to one
(:data:`SITE_SUBTERRAIN_KEYS`, same order; ``"parkour_flat"`` keeps its load-bearing name), and
:data:`SITE_KIND_POLICY` is each column's promotion class (DESIGN §7.2).

The emission contract (DESIGN §2.2)
-----------------------------------
A site body returns, besides its heights, a ``layers`` dict in its own pixel frame -- ``kind``
int8, ``level`` int16 (the walkable level: no features, no roughness), ``surface0`` int16 (the
pre-roughness surface: level + every feature, ramps included), ``material`` int8 (zeros in
phase 1) and ``tile = {"cross_r", "d", "centre_level_m"}`` -- which the decorator pads into the
bordered tile frame and the generator stores per tile as ``site_kind`` / ``site_level`` /
``site_surface0`` / ``site_material`` ``(R, C, W, L)`` and ``site_cross_r`` / ``site_d`` /
``site_centre_level_m`` ``(R, C)`` (:data:`GENERATOR_ATTRS`, initial values :data:`LAYER_INIT` /
:data:`TILE_INIT`). :func:`check_tile_layers` validates one tile's dict against that contract.

Global assembly, cache and frame
--------------------------------
:func:`site_layers` assembles the per-tile arrays into ``(R*W, C*L)`` global layers exactly like
``exposure_stats.global_height_field`` (:func:`global_layer`), once, on ``env.device``, and caches
the :class:`SiteLayers` on the env (:data:`SITE_LAYERS_ATTR`). It reads the terrain IMPORTER
(``env.scene.terrain``), never the parkour term, so the ``base_parkour`` term's own ``__init__``
may call it. Vertical layers become float32 metres (``int16 * vs``); kinds stay int8.

Lookups take world xy (``(n,)`` or ``(n, k)``) and return their shape. The own tile is the omni
``base_parkour`` term's (``env_origins[:, :2]``, ``terrain.terrain_levels`` /
``terrain.terrain_types``) and the pixel is :func:`omni_exposure.tiled_pixel_index` (seam-aware,
nearest pixel, no interpolation) -- exactly :func:`omni_clearance.terrain_height_under`'s lookup.
:func:`world_pixel` finds the same pixel from the terrain origins alone (no own tile), which is
what the offline tools use on a :func:`dump_site_layers` npz.

Support vs nominal (DESIGN §2.4, reserved for phase 2)
------------------------------------------------------
Phase 1 has one physical surface. ``surface_m`` is the ray-cast / nominal surface (what the scan,
the foot patches and any camera see: the mesh) and is the ``omni_clearance`` height-field cache
itself (``_krabby_omni_height_field_m``), never a second copy; ``support_m`` is where a loaded foot
comes to rest. In phase 1 ``support_m is surface_m`` (asserted). Phase 2's soil model lowers
``support_m`` on soft ``material`` cells without touching ``surface_m``, so every reader names the
one it means -- :func:`surface_under` for perception, :func:`support_under` for contact geometry --
and phase 2 changes no call site.

Purity
------
Module scope imports numpy and torch only, so this file path-loads standalone (constants,
:func:`global_layer`, :func:`make_test_layers`, :func:`load_layers_dump`, the contract checks).
:func:`world_pixel` is self-contained too, so offline tools read a dump from a standalone load.
The env lookups import ``omni_exposure`` / ``omni_clearance`` lazily: relative to the package this
module was loaded under (the task package, or the CPU tests' stub-package loader of
``tests/unit/test_omni_p2b_knobs.py``), else by the task package's absolute name (a standalone
path-load inside a Kit process). No isaaclab / omni import anywhere.
"""

from __future__ import annotations

import importlib
import math
import types
from dataclasses import dataclass

import numpy as np
import torch

__all__ = [
    # pixel kinds (§2.1)
    "KIND_GROUND", "KIND_PIT", "KIND_TRENCH", "KIND_LEDGE", "KIND_TERRACE", "KIND_SNAG", "KIND_STRADDLE",
    "KIND_BLOCKER_CHEST", "KIND_BLOCKER_WALL", "KIND_RUBBLE", "KIND_RAMP", "KIND_FIELD_BLOCK", "KIND_FIELD_PIT",
    "KIND_RESERVED_SOFT", "KIND_RESERVED_LOWMU", "KIND_RESERVED_15", "KIND_NAMES", "VOID_KINDS", "BLOCKER_KINDS",
    # column kinds
    "SITE_KINDS", "SITE_SUBTERRAIN_KEYS", "POLICY_LEGACY", "POLICY_CROSS", "POLICY_VOID", "POLICY_BLOCKER",
    "SITE_KIND_POLICY",
    # emission contract (§2.2)
    "LAYER_DTYPES", "LAYER_INIT", "TILE_INIT", "GENERATOR_ATTRS", "check_tile_layers",
    # assembly, cache, lookups (§2.3, §2.4)
    "SITE_LAYERS_ATTR", "TERRAIN_TERM_NAME", "TileFrame", "SiteLayers", "global_layer", "site_layers",
    "layer_under", "kind_under", "level_under", "surface_under", "support_under", "world_pixel",
    # CPU dump (§2.5) and the test constructor
    "dump_site_layers", "load_layers_dump", "make_test_layers",
]

# ============================================================================ pixel kinds (§2.1)
KIND_GROUND = 0
KIND_PIT = 1
KIND_TRENCH = 2
KIND_LEDGE = 3
KIND_TERRACE = 4
KIND_SNAG = 5
KIND_STRADDLE = 6
KIND_BLOCKER_CHEST = 7
KIND_BLOCKER_WALL = 8
KIND_RUBBLE = 9
KIND_RAMP = 10
KIND_FIELD_BLOCK = 11
KIND_FIELD_PIT = 12
KIND_RESERVED_SOFT = 13   # phase 2; never emitted in phase 1
KIND_RESERVED_LOWMU = 14  # phase 2; never emitted in phase 1
KIND_RESERVED_15 = 15     # phase 2; never emitted in phase 1
KIND_NAMES = ("ground", "pit", "trench", "ledge", "terrace", "snag", "straddle", "blocker_chest",
              "blocker_wall", "rubble", "ramp", "field_block", "field_pit", "soft", "lowmu", "reserved15")
#: The void penalty and STOP-0's "foot over void".
VOID_KINDS = (KIND_PIT, KIND_TRENCH)
#: The blocked-path slack, the crash recorder and gate e.
BLOCKER_KINDS = (KIND_BLOCKER_CHEST, KIND_BLOCKER_WALL)

# ============================================================================ column (tile) kinds
SITE_KINDS = ("flat", "field_sparse", "field_dense", "pits", "trench", "ledge_down", "ledge_up", "terraces",
              "snags", "straddle", "chests", "walls", "rubble", "ramps")
#: The site generator's sub-terrain keys, in :data:`SITE_KINDS` order ("parkour_flat" is load-bearing).
SITE_SUBTERRAIN_KEYS = ("parkour_flat", "site_field_sparse", "site_field_dense", "site_pits", "site_trench",
                        "site_ledge_down", "site_ledge_up", "site_terraces", "site_snags", "site_straddle",
                        "site_chests", "site_walls", "site_rubble", "site_ramps")
POLICY_LEGACY, POLICY_CROSS, POLICY_VOID, POLICY_BLOCKER = 0, 1, 2, 3
#: Promotion class per column kind (DESIGN §7.2), indexed like :data:`SITE_KINDS`.
SITE_KIND_POLICY = (0, 0, 0, 2, 2, 1, 1, 1, 1, 1, 3, 3, 1, 1)

# ============================================================================ emission contract (§2.2)
#: dtype of each per-pixel layer, in the body's dict and in the generator's ``(R, C, W, L)`` arrays.
LAYER_DTYPES = {"kind": np.int8, "level": np.int16, "surface0": np.int16, "material": np.int8}
#: Fill of every per-pixel layer where nothing was emitted: the decorator's border pixels and the
#: tiles the generator stored before its first site tile (flat tiles) -- ground at level 0.
LAYER_INIT = {"kind": KIND_GROUND, "level": 0, "surface0": 0, "material": 0}
#: Per-tile scalars (``layers["tile"]``) and the generator's initial value for tiles that emit none.
TILE_INIT = {"cross_r": math.nan, "d": math.nan, "centre_level_m": 0.0}
#: Generator attribute holding each layer / tile scalar (all None until the first site tile).
GENERATOR_ATTRS = {
    "kind": "site_kind", "level": "site_level", "surface0": "site_surface0", "material": "site_material",
    "cross_r": "site_cross_r", "d": "site_d", "centre_level_m": "site_centre_level_m",
}


def check_tile_layers(layers: dict, shape: tuple[int, int]) -> None:
    """Raise ``ValueError`` unless ``layers`` is one tile's emission dict of pixel ``shape``.

    Exactly the four :data:`LAYER_DTYPES` arrays (that dtype, that shape) plus ``tile``, a dict with
    exactly the :data:`TILE_INIT` keys as real numbers (``cross_r`` / ``d`` may be NaN). Used by the
    terrain tests on a body's raw dict and on the decorator's bordered one.
    """
    if not isinstance(layers, dict):
        raise ValueError(f"site layers must be a dict, got {type(layers).__name__}")
    want = set(LAYER_DTYPES) | {"tile"}
    if set(layers) != want:
        raise ValueError(f"site layers keys {sorted(layers)} != {sorted(want)}")
    for key, dtype in LAYER_DTYPES.items():
        arr = layers[key]
        if not isinstance(arr, np.ndarray) or arr.dtype != dtype or arr.shape != tuple(shape):
            got = (type(arr).__name__, getattr(arr, "dtype", None), getattr(arr, "shape", None))
            raise ValueError(f"site layer {key!r} must be {np.dtype(dtype)} {tuple(shape)}, got {got}")
    tile = layers["tile"]
    if not isinstance(tile, dict) or set(tile) != set(TILE_INIT):
        raise ValueError(f"site layers 'tile' must be a dict with keys {sorted(TILE_INIT)}, got {tile!r}")
    for key, value in tile.items():
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"site layers tile[{key!r}] must be a number, got {value!r}")
    if not math.isfinite(float(tile["centre_level_m"])):
        raise ValueError(f"site layers tile['centre_level_m'] must be finite, got {tile['centre_level_m']!r}")


# ============================================================================ frame and container (§2.3)
#: Env attribute the :class:`SiteLayers` are cached under; ``False`` = "looked, this generator has none".
SITE_LAYERS_ATTR = "_krabby_site_layers"
#: The omni parkour term whose tile frame the lookups index with (``omni_clearance.TERRAIN_TERM_NAME``).
TERRAIN_TERM_NAME = "base_parkour"


@dataclass(frozen=True)
class TileFrame:
    """The generator's pixel frame: tile size (m), metres per pixel, metres per vertical unit,
    pixels per tile ``(W, L)`` (stored ``size / hs + 1``, placed ``size`` apart) and the tile grid
    ``(R, C)``."""

    size_xy: tuple[float, float]
    hs: float
    vs: float
    tile_px: tuple[int, int]
    grid_rc: tuple[int, int]

    @property
    def global_shape(self) -> tuple[int, int]:
        """``(R*W, C*L)``: the shape of every global layer."""
        return self.grid_rc[0] * self.tile_px[0], self.grid_rc[1] * self.tile_px[1]


@dataclass
class SiteLayers:
    """The site generator's layers, assembled into global ``(R*W, C*L)`` fields on ``env.device``."""

    kind: torch.Tensor                 # (R*W, C*L) int8
    level_m: torch.Tensor              # (R*W, C*L) float32 metres: walkable level, no features / roughness
    surface0_m: torch.Tensor           # (R*W, C*L) float32 metres: pre-roughness surface (level + features)
    material: torch.Tensor             # (R*W, C*L) int8 (phase 1: zeros)
    surface_m: torch.Tensor            # the omni_clearance global height field (metres, roughness included)
    support_m: torch.Tensor            # phase 1: THE SAME OBJECT as surface_m; phase 2: soft cells lowered
    tile_kind: torch.Tensor            # (R, C) int8 index into SITE_KINDS, -1 if not a site key
    tile_cross_r: torch.Tensor         # (R, C) float32, NaN = no crossing
    tile_d: torch.Tensor               # (R, C) float32 site difficulty (NaN = the tile emitted none)
    tile_centre_level_m: torch.Tensor  # (R, C) float32
    frame: TileFrame


def global_layer(per_tile) -> torch.Tensor:
    """``(R, C, W, L)`` per-tile array -> ``(R*W, C*L)`` global tensor (a copy, same dtype), assembled
    exactly like ``exposure_stats.global_height_field`` so ``omni_exposure.global_pixel_index``
    indexes it."""
    t = per_tile.detach().clone() if torch.is_tensor(per_tile) else torch.from_numpy(np.array(per_tile))
    if t.dim() != 4:
        raise ValueError(f"global_layer expects a (rows, cols, W, L) array, got shape {tuple(t.shape)}")
    r, c, w, ell = t.shape
    return t.permute(0, 2, 1, 3).reshape(r * w, c * ell).contiguous()


def _metres(per_tile, vs: float) -> torch.Tensor:
    """Per-tile vertical units -> global float32 metres (``omni_clearance.global_height_field_m``'s
    arithmetic: cast to float32, times ``vs``, assemble)."""
    return global_layer(torch.from_numpy(np.asarray(per_tile, dtype=np.float32)) * float(vs))


def _tile_kind_from_names(terrain_names) -> np.ndarray:
    """The generator's ``terrain_names`` ``(R, C[, 1])`` -> ``(R, C)`` int8 index into
    :data:`SITE_KINDS` (by :data:`SITE_SUBTERRAIN_KEYS`), -1 for any other name."""
    names = np.asarray(terrain_names)
    if names.ndim == 3:
        names = names[..., 0]
    lut = {key: i for i, key in enumerate(SITE_SUBTERRAIN_KEYS)}
    out = np.full(names.shape, -1, dtype=np.int8)
    for idx, name in np.ndenumerate(names):
        out[idx] = lut.get(str(name), -1)
    return out


def _frame_of(terrain) -> TileFrame:
    """The importer's generator frame (the parkour term's ``_tile_size`` / ``_hscale`` / ``_tile_px``
    / ``_grid_rc`` come from the same two objects)."""
    gen = terrain.terrain_generator_class
    tgc = terrain.cfg.terrain_generator
    r, c, w, ell = (int(v) for v in np.shape(gen.height_fields))
    return TileFrame(size_xy=(float(tgc.size[0]), float(tgc.size[1])), hs=float(tgc.horizontal_scale),
                     vs=float(tgc.vertical_scale), tile_px=(w, ell), grid_rc=(r, c))


def _generator_site_arrays(gen) -> dict | None:
    """The generator's seven site arrays by contract key, or None when it stored none."""
    if getattr(gen, GENERATOR_ATTRS["kind"], None) is None:
        return None
    return {key: getattr(gen, attr) for key, attr in GENERATOR_ATTRS.items()}


# ============================================================================ lazy sibling imports
#: The package the env-facing siblings live in, for a standalone path-load (no ``__package__``).
_MDP_PACKAGE = "parkour_tasks.crab_hex_omni_task.mdp"


def _sibling(name: str):
    """``omni_exposure`` / ``omni_clearance``, imported at first use: relative to this module's
    package when it was loaded under one (the task package, or the CPU tests' stub packages), else
    by the package's absolute name (a tool that path-loaded this file standalone inside a Kit
    process, e.g. the site recorder)."""
    return importlib.import_module(f"{__package__ or _MDP_PACKAGE}.{name}")


def _omni_exposure():
    return _sibling("omni_exposure")


def _omni_clearance():
    return _sibling("omni_clearance")


# ============================================================================ the cache
def site_layers(env) -> SiteLayers | None:
    """The env's :class:`SiteLayers`, built once on ``env.device`` and cached on
    :data:`SITE_LAYERS_ATTR`; None (cached as ``False``) when the generator stored no site layers
    (ring / field / flat-only generators, or a stand-in env without a terrain importer).

    Reads ``env.scene.terrain`` only (not the parkour term), so it is safe inside the term's own
    ``__init__``. ``surface_m`` is the ``omni_clearance`` height-field cache, built through it if
    absent; on a non-site generator nothing is built.
    """
    cached = getattr(env, SITE_LAYERS_ATTR, None)
    if cached is False:
        return None
    if cached is not None:
        return cached
    terrain = getattr(getattr(env, "scene", None), "terrain", None)
    gen = getattr(terrain, "terrain_generator_class", None) if terrain is not None else None
    arrays = _generator_site_arrays(gen) if gen is not None else None
    if arrays is None:
        setattr(env, SITE_LAYERS_ATTR, False)
        return None

    frame = _frame_of(terrain)
    r, c = frame.grid_rc
    for key in LAYER_DTYPES:
        if tuple(np.shape(arrays[key])) != (r, c, *frame.tile_px):
            raise ValueError(f"generator.{GENERATOR_ATTRS[key]} shape {np.shape(arrays[key])} != height "
                             f"fields {(r, c, *frame.tile_px)}")
    dev = env.device
    # The omni_clearance cache IS the nominal surface (built here if no clearance term ran yet); the
    # holder carries the one attribute _height_field_m reads.
    surface_m = _omni_clearance()._height_field_m(env, types.SimpleNamespace(terrain=terrain))

    def _tile(key):
        return torch.as_tensor(np.asarray(arrays[key], dtype=np.float32), device=dev).clone()

    layers = SiteLayers(
        kind=global_layer(np.asarray(arrays["kind"], dtype=np.int8)).to(dev),
        level_m=_metres(arrays["level"], frame.vs).to(dev),
        surface0_m=_metres(arrays["surface0"], frame.vs).to(dev),
        material=global_layer(np.asarray(arrays["material"], dtype=np.int8)).to(dev),
        surface_m=surface_m,
        support_m=surface_m,
        tile_kind=torch.as_tensor(_tile_kind_from_names(gen.terrain_names), device=dev),
        tile_cross_r=_tile("cross_r"),
        tile_d=_tile("d"),
        tile_centre_level_m=_tile("centre_level_m"),
        frame=frame,
    )
    assert layers.support_m is layers.surface_m, "phase 1: support_m must be surface_m (DESIGN §2.4)"
    setattr(env, SITE_LAYERS_ATTR, layers)
    return layers


# ============================================================================ lookups
def layer_under(layer, x_w, y_w, tile_centre_w, row, col, frame: TileFrame, off_value) -> torch.Tensor:
    """``layer`` (a global ``(R*W, C*L)`` tensor) at the pixel under each ``(x_w, y_w)``.

    The pixel of the tile the point is ON (own or a neighbour; :func:`omni_exposure.tiled_pixel_index`
    from the own tile ``tile_centre_w`` ``(n, 2)``, ``row`` / ``col`` ``(n,)``); ``off_value`` where
    that tile does not exist (off the terrain). Output: ``x_w``'s shape, ``layer``'s dtype.
    """
    nr, nc, ix, iy, on_terrain, _off_tile = _omni_exposure().tiled_pixel_index(
        x_w, y_w, tile_centre_w, row, col, frame.size_xy, frame.hs, frame.tile_px, frame.grid_rc
    )
    w, ell = int(frame.tile_px[0]), int(frame.tile_px[1])
    v = layer[nr * w + ix, nc * ell + iy]
    return torch.where(on_terrain, v, torch.full_like(v, off_value))


def _own_tile(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(tile_centre_w (N, 2), row (N,), col (N,))`` of the omni ``base_parkour`` term."""
    term = env.parkour_manager.get_term(TERRAIN_TERM_NAME)
    terrain = getattr(term, "terrain", None)
    if not hasattr(term, "env_origins") or not hasattr(terrain, "terrain_levels"):
        raise RuntimeError(
            f"the site raster lookups need the omni {TERRAIN_TERM_NAME!r} parkour term's tile frame "
            f"(env_origins, terrain.terrain_levels / terrain_types); {type(term).__name__} has none"
        )
    return term.env_origins[:, :2], terrain.terrain_levels, terrain.terrain_types


def kind_under(env, x_w, y_w) -> torch.Tensor:
    """int8 pixel kind under each point, ``x_w``'s shape; ``KIND_GROUND`` off the terrain and
    everywhere when the generator has no site layers."""
    layers = site_layers(env)
    if layers is None:
        return torch.full_like(x_w, KIND_GROUND, dtype=torch.int8)
    centre, row, col = _own_tile(env)
    return layer_under(layers.kind, x_w, y_w, centre, row, col, layers.frame, KIND_GROUND)


def level_under(env, x_w, y_w) -> torch.Tensor:
    """float32 walkable level (m) under each point: features and roughness removed (a pit interior
    reads its rim, a pad its own level); 0.0 off the terrain and everywhere without site layers
    (the base plane of ring / field / flat generators)."""
    layers = site_layers(env)
    if layers is None:
        return torch.zeros_like(x_w, dtype=torch.float32)
    centre, row, col = _own_tile(env)
    return layer_under(layers.level_m, x_w, y_w, centre, row, col, layers.frame, 0.0)


def _surface_like(env, x_w, y_w, attr: str) -> torch.Tensor:
    layers = site_layers(env)
    if layers is not None:
        centre, row, col = _own_tile(env)
        return layer_under(getattr(layers, attr), x_w, y_w, centre, row, col, layers.frame, 0.0)
    # Any other omni generator: omni_clearance's own sample (the same cache, the same lookup).
    oc = _omni_clearance()
    term = oc._terrain_term(env)
    terrain = term.terrain
    return oc.terrain_height_under(
        x_w, y_w, term.env_origins[:, :2], terrain.terrain_levels, terrain.terrain_types,
        oc._height_field_m(env, term), term._tile_size, term._hscale, term._tile_px, term._grid_rc,
    )


def surface_under(env, x_w, y_w) -> torch.Tensor:
    """float32 NOMINAL surface z (m, roughness included) under each point -- what the scan, the
    foot patches and a camera see. Any omni generator (the ``omni_clearance`` cache); 0 off the
    terrain."""
    return _surface_like(env, x_w, y_w, "surface_m")


def support_under(env, x_w, y_w) -> torch.Tensor:
    """float32 SUPPORT surface z (m) under each point -- where a loaded foot rests (contact
    geometry). Phase 1: equal to :func:`surface_under` (``support_m is surface_m``)."""
    return _surface_like(env, x_w, y_w, "support_m")


def world_pixel(x_w, y_w, frame: TileFrame, terrain_origins):
    """World xy -> global pixel ``(gx, gy, on_terrain)`` from the terrain origins alone.

    ``terrain_origins`` is the importer's ``(R, C, 3)`` (tile centres, whole-terrain centred). The
    tile is ``floor((x - (o[0, 0, 0] - sx/2)) / sx)`` per axis (clamped into the grid); the pixel is
    then :func:`omni_exposure.tiled_pixel_index` with THAT tile as the own tile, so the result equals
    the own-tile lookup of a point on its own tile bit for bit, seams and clamping included.
    ``on_terrain`` is False past the outermost tiles (``gx`` / ``gy`` then index the nearest edge
    tile's clamped pixel, as the lookups do before masking). Numpy in -> numpy out; torch in ->
    torch out (on ``x_w``'s device). Any shape.

    Self-contained (``tiled_pixel_index``'s arithmetic restated op for op; the tests pin the two
    equal), so an offline tool that path-loads this file standalone can read a dump without the
    task package.
    """
    as_numpy = not torch.is_tensor(x_w)
    x = torch.as_tensor(x_w)
    y = torch.as_tensor(y_w, device=x.device)
    o = torch.as_tensor(terrain_origins, device=x.device)
    n_rows, n_cols = int(frame.grid_rc[0]), int(frame.grid_rc[1])
    if tuple(o.shape[:2]) != (n_rows, n_cols):
        raise ValueError(f"terrain_origins shape {tuple(o.shape)} does not match grid_rc {frame.grid_rc}")
    sx, sy = float(frame.size_xy[0]), float(frame.size_xy[1])
    hs = float(frame.hs)
    w, ell = int(frame.tile_px[0]), int(frame.tile_px[1])
    r = torch.floor((x - (o[0, 0, 0] - 0.5 * sx)) / sx).long().clamp(0, n_rows - 1)
    c = torch.floor((y - (o[0, 0, 1] - 0.5 * sy)) / sy).long().clamp(0, n_cols - 1)
    # omni_exposure.tiled_pixel_index with (r, c) as the own tile, op for op.
    lx = x - (o[r, c, 0] - 0.5 * sx)
    ly = y - (o[r, c, 1] - 0.5 * sy)
    ox = torch.floor(lx / sx).long()
    oy = torch.floor(ly / sy).long()
    nr, nc = r + ox, c + oy
    on_terrain = (nr >= 0) & (nr < n_rows) & (nc >= 0) & (nc < n_cols)
    ix = torch.clamp(((lx - ox.to(lx.dtype) * sx) / hs).round().long(), 0, w - 1)
    iy = torch.clamp(((ly - oy.to(ly.dtype) * sy) / hs).round().long(), 0, ell - 1)
    gx = nr.clamp(0, n_rows - 1) * w + ix
    gy = nc.clamp(0, n_cols - 1) * ell + iy
    if as_numpy:
        return gx.cpu().numpy(), gy.cpu().numpy(), on_terrain.cpu().numpy()
    return gx, gy, on_terrain


# ============================================================================ CPU dump (§2.5)
def _mix_string(tgc) -> str:
    """The realised column mix ``name=count,...`` (non-zero kinds, :data:`SITE_KINDS` order), from
    the generator cfg's proportions x ``num_cols`` (the site applier writes ``count / total``)."""
    subs = dict(getattr(tgc, "sub_terrains", {}) or {})
    props = {k: float(getattr(s, "proportion", 0.0)) for k, s in subs.items()}
    total = sum(props.values())
    if not total > 0.0:
        return ""
    n_cols = int(tgc.num_cols)
    parts = []
    for name, key in zip(SITE_KINDS, SITE_SUBTERRAIN_KEYS):
        count = int(round(props.get(key, 0.0) / total * n_cols))
        if count:
            parts.append(f"{name}={count}")
    return ",".join(parts)


def _d_ranges(tgc) -> tuple[np.ndarray, np.ndarray]:
    """``(site_d_range, gen_d_range)`` from the first site sub-terrain that carries them (the site
    applier writes the same pair into every one), NaN pairs if none does."""
    for sub in (getattr(tgc, "sub_terrains", {}) or {}).values():
        if hasattr(sub, "site_d_range") and hasattr(sub, "gen_d_range"):
            return (np.asarray(sub.site_d_range, dtype=np.float64), np.asarray(sub.gen_d_range, dtype=np.float64))
    nan = np.array([np.nan, np.nan])
    return nan, nan.copy()


def dump_site_layers(env, path) -> None:
    """Write the generator's layers and frame to a CPU npz at ``path`` (exactly that path).

    Keys: ``height_field`` (global int16), ``vs``, ``hs``, ``size_xy``, ``tile_px``, ``grid_rc``,
    ``terrain_origins`` ``(R, C, 3)``, ``site``; and on a site generator also ``kind``, ``level``,
    ``surface0``, ``material`` (global int arrays), ``tile_kind``, ``tile_cross_r``, ``tile_d``,
    ``tile_centre_level_m``, ``terrain_names`` ``(R, C)`` str, ``mix`` (str), ``site_d_range``,
    ``gen_d_range``. Read from the generator's CPU arrays (exact integer units), never the GPU
    cache. :func:`load_layers_dump` reads it back.
    """
    terrain = env.scene.terrain
    gen = terrain.terrain_generator_class
    tgc = terrain.cfg.terrain_generator
    frame = _frame_of(terrain)
    origins = getattr(terrain, "terrain_origins", None)
    if origins is None:
        origins = gen.terrain_origins
    origins = (origins.detach().cpu().numpy() if torch.is_tensor(origins) else np.asarray(origins))
    arrays = _generator_site_arrays(gen)
    out = {
        "height_field": global_layer(np.asarray(gen.height_fields, dtype=np.int16)).numpy(),
        "vs": np.float64(frame.vs),
        "hs": np.float64(frame.hs),
        "size_xy": np.asarray(frame.size_xy, dtype=np.float64),
        "tile_px": np.asarray(frame.tile_px, dtype=np.int64),
        "grid_rc": np.asarray(frame.grid_rc, dtype=np.int64),
        "terrain_origins": origins.astype(np.float32),
        "site": np.bool_(arrays is not None),
    }
    if arrays is not None:
        for key in LAYER_DTYPES:
            out[key] = global_layer(np.asarray(arrays[key], dtype=LAYER_DTYPES[key])).numpy()
        out["tile_kind"] = _tile_kind_from_names(gen.terrain_names)
        for key in TILE_INIT:
            out[f"tile_{key}"] = np.asarray(arrays[key], dtype=np.float32)
        names = np.asarray(gen.terrain_names)
        out["terrain_names"] = (names[..., 0] if names.ndim == 3 else names).astype(str)
        out["mix"] = np.str_(_mix_string(tgc))
        out["site_d_range"], out["gen_d_range"] = _d_ranges(tgc)
    with open(path, "wb") as fh:
        np.savez_compressed(fh, **out)


def load_layers_dump(path) -> dict:
    """Read a :func:`dump_site_layers` npz into a dict (no pickle).

    Arrays stay numpy at their stored dtypes; ``vs`` / ``hs`` become floats, ``size_xy`` /
    ``tile_px`` / ``grid_rc`` tuples, ``site`` a bool, ``mix`` a str and the two d ranges float
    tuples; ``frame`` is added as the :class:`TileFrame` (``world_pixel(x, y, d["frame"],
    d["terrain_origins"])``).
    """
    with np.load(path, allow_pickle=False) as z:
        d = {key: z[key] for key in z.files}
    d["vs"], d["hs"] = float(d["vs"]), float(d["hs"])
    d["size_xy"] = tuple(float(v) for v in d["size_xy"])
    d["tile_px"] = tuple(int(v) for v in d["tile_px"])
    d["grid_rc"] = tuple(int(v) for v in d["grid_rc"])
    d["site"] = bool(d["site"])
    if d["site"]:
        d["mix"] = str(d["mix"])
        d["site_d_range"] = tuple(float(v) for v in d["site_d_range"])
        d["gen_d_range"] = tuple(float(v) for v in d["gen_d_range"])
    d["frame"] = TileFrame(size_xy=d["size_xy"], hs=d["hs"], vs=d["vs"], tile_px=d["tile_px"], grid_rc=d["grid_rc"])
    return d


# ============================================================================ the test constructor
_TEST_TILE_KEYS = ("material", "surface_m", "tile_kind", "tile_cross_r", "tile_d", "tile_centre_level_m", "device")


def make_test_layers(kind, level_m, surface0_m, frame: TileFrame, **tile) -> SiteLayers:
    """A CPU :class:`SiteLayers` for stand-in envs (every worker's tests).

    ``kind`` / ``level_m`` / ``surface0_m`` (metres) are global ``(R*W, C*L)`` or per-tile
    ``(R, C, W, L)`` arrays or tensors (per-tile ones are assembled with :func:`global_layer`).
    Optional keywords: ``material`` (default zeros), ``surface_m`` (default a copy of ``surface0_m``),
    ``tile_kind`` (default -1), ``tile_cross_r`` / ``tile_d`` (default NaN), ``tile_centre_level_m``
    (default 0), ``device`` (default cpu). ``support_m is surface_m``. Install it on a stand-in env
    with ``setattr(env, SITE_LAYERS_ATTR, layers)``; the lookups then need only the env's
    ``parkour_manager.get_term("base_parkour")`` (``env_origins``, ``terrain.terrain_levels`` /
    ``terrain_types``).
    """
    unknown = set(tile) - set(_TEST_TILE_KEYS)
    if unknown:
        raise TypeError(f"make_test_layers got unexpected keywords {sorted(unknown)}; allowed {_TEST_TILE_KEYS}")
    dev = tile.get("device", "cpu")
    shape = frame.global_shape
    rc = tuple(frame.grid_rc)

    def _global(value, dtype):
        t = value if torch.is_tensor(value) else torch.as_tensor(np.asarray(value))
        if t.dim() == 4:
            t = global_layer(t)
        if tuple(t.shape) != shape:
            raise ValueError(f"layer shape {tuple(t.shape)} != frame global shape {shape}")
        return t.to(device=dev, dtype=dtype).clone()

    def _per_tile(key, fill):
        value = tile.get(key)
        t = torch.full(rc, fill, dtype=torch.float32) if value is None else torch.as_tensor(value, dtype=torch.float32)
        if tuple(t.shape) != rc:
            raise ValueError(f"{key} shape {tuple(t.shape)} != grid {rc}")
        return t.to(dev).clone()

    surface0 = _global(surface0_m, torch.float32)
    surface = surface0.clone() if tile.get("surface_m") is None else _global(tile["surface_m"], torch.float32)
    material = (torch.zeros(shape, dtype=torch.int8, device=dev) if tile.get("material") is None
                else _global(tile["material"], torch.int8))
    tile_kind = tile.get("tile_kind")
    tk = torch.full(rc, -1, dtype=torch.int8) if tile_kind is None else torch.as_tensor(tile_kind, dtype=torch.int8)
    if tuple(tk.shape) != rc:
        raise ValueError(f"tile_kind shape {tuple(tk.shape)} != grid {rc}")
    layers = SiteLayers(
        kind=_global(kind, torch.int8),
        level_m=_global(level_m, torch.float32),
        surface0_m=surface0,
        material=material,
        surface_m=surface,
        support_m=surface,
        tile_kind=tk.to(dev).clone(),
        tile_cross_r=_per_tile("tile_cross_r", math.nan),
        tile_d=_per_tile("tile_d", math.nan),
        tile_centre_level_m=_per_tile("tile_centre_level_m", 0.0),
        frame=frame,
    )
    assert layers.support_m is layers.surface_m
    return layers
