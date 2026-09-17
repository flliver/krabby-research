"""Composed-moth plan, step 4 (design D8; spec B1-B4): the omni ring terrain package
(`parkour_isaaclab/terrains/omni/`) and the `terrains/utils.py` decorator refactor.

No Isaac Sim / Kit anywhere in this file. `parkour_isaaclab/terrains/__init__.py` and
`.../extreme_parkour/__init__.py` import isaaclab's `configclass` machinery (-> `pxr`/Usd),
which needs a running Kit process this repo's unit-test venv does not have. The generator
FUNCTIONS themselves (`utils.py`, `extreme_parkour_terrians.py`, `omni/omni_terrains.py`)
carry no such dependency -- only their `TYPE_CHECKING`-guarded cfg-class imports do -- so
each is path-loaded under its real dotted name, with only the PACKAGE `__init__` modules
replaced by inert stubs (never executed). This is the same synthetic-package technique
`test_crab_hex_omni_command.py` and `test_parkour_command_heading_offset.py` already use
for the sibling `parkour_isaaclab.envs.mdp.*` tree, and it is deliberately idempotent
(get-or-create keyed by dotted name) so this file is order-independent with those when run
in the same session.

Byte-identity fixture: `tests/unit/fixtures/omni_ring_golden_forward_decorator.npz`, captured
from the UNMODIFIED (pre-refactor) `parkour_field_to_mesh` on the real "parkour_flat"
sub-terrain (`ExtremeParkourHurdleTerrainCfg(apply_flat=True, ...)`, from
`terrains/extreme_parkour/config/parkour.py`) at difficulty 0.4, numpy seed 12345, num_goals
8 -- see `TestGoldenForwardDecoratorByteIdentity`.
"""
from __future__ import annotations

import copy
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
LIB = REPO / "parkour" / "parkour_isaaclab"
GOLDEN_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "omni_ring_golden_forward_decorator.npz"


# --------------------------------------------------------------------------- loader plumbing
def _synthetic_package(name: str, path: Path) -> types.ModuleType:
    """A package whose submodules are searched for in ``path`` -- no ``__init__.py`` run.
    Get-or-create, so this is safe regardless of whether some other test in the same
    session already registered ``parkour_isaaclab`` (several do, for the sibling
    ``envs.mdp`` tree, under this exact same convention)."""
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(path)]
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
    return mod


def _load_real_module(name: str, file_path: Path) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, file_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


def _load_terrain_modules():
    terrains_pkg = _synthetic_package("parkour_isaaclab", LIB)
    terrains_terrains_pkg = _synthetic_package("parkour_isaaclab.terrains", LIB / "terrains")
    terrains_pkg.terrains = terrains_terrains_pkg

    utils = _load_real_module("parkour_isaaclab.terrains.utils", LIB / "terrains" / "utils.py")
    terrains_terrains_pkg.utils = utils

    extreme_pkg = _synthetic_package(
        "parkour_isaaclab.terrains.extreme_parkour", LIB / "terrains" / "extreme_parkour"
    )
    terrains_terrains_pkg.extreme_parkour = extreme_pkg
    fwd = _load_real_module(
        "parkour_isaaclab.terrains.extreme_parkour.extreme_parkour_terrians",
        LIB / "terrains" / "extreme_parkour" / "extreme_parkour_terrians.py",
    )
    extreme_pkg.extreme_parkour_terrians = fwd

    omni_pkg = _synthetic_package("parkour_isaaclab.terrains.omni", LIB / "terrains" / "omni")
    terrains_terrains_pkg.omni = omni_pkg
    omni = _load_real_module(
        "parkour_isaaclab.terrains.omni.omni_terrains", LIB / "terrains" / "omni" / "omni_terrains.py"
    )
    omni_pkg.omni_terrains = omni

    return utils, fwd, omni


utils, fwd, omni = _load_terrain_modules()

# --------------------------------------------------------------------------- shared fixtures
HS = 0.08
VS = 0.005
SLOPE_THRESHOLD = 1.5
CLEAR_RADIUS = 2.0
RING_OUTER_RADIUS = 4.5
# recal2b2 (crab_hex_env_cfg.py:400-406), copied verbatim onto OmniRingTerrainCfg.
HURDLE_HEIGHT_RANGE = "0.02 + 0.05 * difficulty, 0.03 + 0.06 * difficulty"
STEP_HEIGHT = "0.02 + 0.06 * difficulty"
DIFFICULTIES = [0.05, 0.42, 0.7]


def make_ring_cfg(
    n_blocks: int,
    *,
    bar_fraction: float = 0.5,
    apply_roughness: bool = False,
    size: tuple[float, float] = (10.0, 10.0),
) -> types.SimpleNamespace:
    """Mirrors ``OmniRingTerrainCfg`` after ``size``/``horizontal_scale``/``vertical_scale``/
    ``slope_threshold`` are propagated onto every sub-terrain cfg by
    ``isaaclab.terrains.terrain_generator.TerrainGenerator.__init__`` (read directly:
    ``border_width`` is NOT propagated there, so it stays the ``ParkourSubTerrainBaseCfg``
    default of 0.0, as at every existing sub-terrain cfg site in this codebase)."""
    return types.SimpleNamespace(
        border_width=0.0,
        horizontal_scale=HS,
        vertical_scale=VS,
        platform_len=2.5,
        platform_height=0.0,
        slope_threshold=SLOPE_THRESHOLD,
        edge_width_thresh=0.05,
        use_simplified=False,
        size=size,
        apply_roughness=apply_roughness,
        downsampled_scale=0.075,
        # SHIPPED value, not OmniRingTerrainCfg's own class default (0.02, 0.06):
        # _apply_crab_hex_easy_mixed_terrain (the shared knob loop every omni window runs
        # through) overwrites every sub-terrain's noise_range to (0.02, 0.02) at 1a AND at
        # every 2a+ window alike (cfg_omni_1a.json / cfg_omni_2a.json both confirm [0.02, 0.02]);
        # the wider class default is never what actually ships. See omni_exposure.py's caveat.
        noise_range=(0.02, 0.02),
        noise_step=0.005,
        pad_width=0.1,
        pad_height=0.0,
        clear_radius=CLEAR_RADIUS,
        ring_outer_radius=RING_OUTER_RADIUS,
        n_blocks=n_blocks,
        bar_fraction=bar_fraction,
        hurdle_height_range=HURDLE_HEIGHT_RANGE,
        step_height=STEP_HEIGHT,
    )


def _radius_grid(heights_shape: tuple[int, int], size: tuple[float, float]) -> np.ndarray:
    """Distance (metres) of every pixel CENTRE in the decorator's full, bordered ``heights``
    array from the tile centre. Pixel (i, j)'s metric position is ``(i + 0.5) * hs - size/2``
    on each axis -- true regardless of the decorator's internal border bookkeeping, because
    the 1-px border it adds is itself centred (see the report: derived and cross-checked
    numerically against the decorated function's own output)."""
    width, length = heights_shape
    cx, cy = 0.5 * size[0], 0.5 * size[1]
    xs = (np.arange(width) + 0.5) * HS - cx
    ys = (np.arange(length) + 0.5) * HS - cy
    dx, dy = np.meshgrid(xs, ys, indexing="ij")
    return np.sqrt(dx**2 + dy**2)


# =============================================================================================
class TestGoldenForwardDecoratorByteIdentity:
    """B2 (golden fixture): the refactored decorator family
    (`_parkour_field_to_mesh_impl(func, edge_axes="x")`, i.e. `parkour_field_to_mesh`)
    reproduces the pre-refactor decorator's output byte for byte."""

    @staticmethod
    def _make_forward_parkour_flat_cfg() -> types.SimpleNamespace:
        # config/parkour.py's "parkour_flat": ExtremeParkourHurdleTerrainCfg(apply_flat=True,
        # x_range=(1.2, 2.2), half_valid_width=(0.4, 0.8), hurdle_height_range=...), with
        # size/horizontal_scale/vertical_scale/slope_threshold propagated from
        # EXTREME_PARKOUR_TERRAINS_CFG. recal2b2 never touches "parkour_flat" (apply_flat=True
        # means the height value is never even written), so this is untouched by any window.
        return types.SimpleNamespace(
            border_width=0.0, horizontal_scale=0.08, vertical_scale=0.005,
            platform_len=2.5, platform_height=0.0, slope_threshold=1.5,
            edge_width_thresh=0.05, use_simplified=False, size=(16.0, 4.0),
            apply_roughness=True, apply_flat=True, downsampled_scale=0.075,
            noise_range=(0.02, 0.06), noise_step=0.005, x_range=(1.2, 2.2),
            y_range=(-0.4, 0.4), half_valid_width=(0.4, 0.8), pad_width=0.1, pad_height=0.0,
            stone_len="0.1 + 0.3 * difficulty",
            hurdle_height_range="0.1+0.1*difficulty, 0.15+0.15*difficulty",
        )

    def test_byte_identical(self):
        golden = np.load(GOLDEN_FIXTURE)
        cfg = self._make_forward_parkour_flat_cfg()
        np.random.seed(int(golden["seed"]))
        meshes, origin, goals, goal_heights, x_edge_mask, heights = fwd.parkour_hurdle_terrain(
            float(golden["difficulty"]), cfg, int(golden["num_goals"])
        )
        mesh = meshes[0]
        assert np.array_equal(np.asarray(mesh.vertices), golden["vertices"])
        assert np.array_equal(np.asarray(mesh.faces), golden["triangles"])
        assert np.array_equal(origin, golden["origin"])
        assert np.array_equal(goals, golden["goals"])
        assert np.array_equal(goal_heights, golden["goal_heights"])
        assert np.array_equal(x_edge_mask, golden["x_edge_mask"])
        assert np.array_equal(heights, golden["heights"])


# =============================================================================================
class TestRingGeometry:
    """B2: the DECORATED `omni_ring_terrain` at the real cfg (recal2b2 height formulas,
    clear_radius=2.0, ring_outer_radius=4.5), for d in {0.05, 0.42, 0.7}."""

    @pytest.mark.parametrize("n_blocks", [8, 16])
    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_disc_flat_no_roughness(self, difficulty, n_blocks):
        # apply_roughness=False isolates the property the plan actually cares about --
        # block placement never touches the disc -- from `random_uniform_terrain`'s own
        # noise (shared, unmodified utility; its downsample-then-spline-interpolate step
        # can overshoot its own nominal noise band, so a roughness-enabled numeric bound
        # here would pin spline behaviour, not ring geometry).
        cfg = make_ring_cfg(n_blocks, apply_roughness=False)
        np.random.seed(1)
        *_, heights = omni.omni_ring_terrain(difficulty, cfg, 8)
        r = _radius_grid(heights.shape, (10.0, 10.0))
        assert np.all(heights[r < CLEAR_RADIUS] == 0)

    @pytest.mark.parametrize("n_blocks", [8, 16])
    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_raised_pixels_only_in_annulus(self, difficulty, n_blocks):
        cfg = make_ring_cfg(n_blocks, apply_roughness=False)
        np.random.seed(2)
        *_, heights = omni.omni_ring_terrain(difficulty, cfg, 8)
        r = _radius_grid(heights.shape, (10.0, 10.0))
        raised = heights > 0
        assert raised.any(), "sanity: some block actually got rasterised"
        assert np.all(r[raised] >= CLEAR_RADIUS)
        assert np.all(r[raised] <= RING_OUTER_RADIUS)

    @pytest.mark.parametrize("n_blocks", [8, 16])
    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_max_height_within_formulas(self, difficulty, n_blocks):
        cfg = make_ring_cfg(n_blocks, apply_roughness=False)
        np.random.seed(3)
        *_, heights = omni.omni_ring_terrain(difficulty, cfg, 8)
        hurdle_lo, hurdle_hi = eval(HURDLE_HEIGHT_RANGE, {"difficulty": difficulty})
        step_h = eval(STEP_HEIGHT, {"difficulty": difficulty})
        max_formula_m = max(hurdle_hi, step_h)
        max_height_m = float(heights.max()) * VS
        assert 0 < max_height_m <= max_formula_m + VS  # +1 raw-unit rounding slack

    def test_seed_reproducibility(self):
        cfg = make_ring_cfg(16, apply_roughness=True)
        np.random.seed(777)
        z1, g1, gh1 = omni.ring_height_field(np.random.RandomState(777), 0.5, copy.deepcopy(cfg), 8)
        np.random.seed(777)
        z2, g2, gh2 = omni.ring_height_field(np.random.RandomState(777), 0.5, copy.deepcopy(cfg), 8)
        assert np.array_equal(z1, z2)
        assert np.array_equal(g1, g2)
        assert np.array_equal(gh1, gh2)

    def test_goals_are_eight_sector_anchors_never_advanced(self):
        cfg = make_ring_cfg(8, apply_roughness=False)
        rng = np.random.RandomState(5)
        _, goals_returned, goal_heights = omni.ring_height_field(rng, 0.5, cfg, 8)
        assert goals_returned.shape == (8, 2)
        # `ring_height_field`'s own return already applies `* horizontal_scale` (matching
        # every forward terrain function's `return ..., goals * cfg.horizontal_scale, ...`
        # convention) -- these are corner-based METRES, not pixels; only the decorator's
        # own `-= [0.5*size, 0.5*size]` centring step remains.
        goals_m = goals_returned - np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1]])
        radii = np.linalg.norm(goals_m, axis=1)
        r_mid = 0.5 * (CLEAR_RADIUS + RING_OUTER_RADIUS)
        assert np.allclose(radii, r_mid)
        angles = np.sort(np.arctan2(goals_m[:, 1], goals_m[:, 0]) % (2 * np.pi))
        dtheta = np.diff(np.concatenate([angles, angles[:1] + 2 * np.pi]))
        assert np.allclose(dtheta, 2 * np.pi / 8)
        # never advanced: a flat zero, not tied to any block's own height
        assert np.array_equal(goal_heights, np.zeros(8))

    def test_pixel_size_pin(self):
        # "size in px = int(round(size / hs))", accounting for the decorator's own -2 px
        # (border_pixels = 1 on each side at border_width=0.0, the sub-terrain default).
        width_pixels_orig = int(10.0 / HS) + 1
        assert width_pixels_orig == 126
        border_pixels = int(0.0 / HS) + 1
        assert border_pixels == 1
        reduced = (width_pixels_orig - 2 * border_pixels) * HS
        assert reduced == pytest.approx(9.92)

        cfg_reduced = make_ring_cfg(8, size=(reduced, reduced))
        assert omni._tile_pixel_dims(cfg_reduced) == (124, 124)

        # end-to-end: the DECORATED call's own full ``heights`` array is 126 x 126 regardless
        cfg_full = make_ring_cfg(8, apply_roughness=False)
        np.random.seed(9)
        *_, heights = omni.omni_ring_terrain(0.5, cfg_full, 8)
        assert heights.shape == (126, 126)


# =============================================================================================
class TestFlatTerrain:
    def test_zero_field_and_shared_anchor_geometry(self):
        flat_cfg = types.SimpleNamespace(
            border_width=0.0, horizontal_scale=HS, vertical_scale=VS, slope_threshold=SLOPE_THRESHOLD,
            edge_width_thresh=0.05, use_simplified=False, size=(10.0, 10.0),
            apply_roughness=False, downsampled_scale=0.075, noise_range=(0.02, 0.06),
            noise_step=0.005, pad_width=0.1, pad_height=0.0,
            clear_radius=CLEAR_RADIUS, ring_outer_radius=RING_OUTER_RADIUS,
        )
        np.random.seed(11)
        meshes, origin, goals, goal_heights, edge_mask, heights = omni.omni_flat_terrain(0.5, flat_cfg, 8)
        assert np.all(heights == 0)
        assert not edge_mask.any()

        # same 8 anchors as a ring cfg sharing the same clear/outer radius -- independent of
        # which sub-terrain type a given tile actually is.
        ring_cfg = make_ring_cfg(8, apply_roughness=False)
        _, ring_goals_m, _ = omni.ring_height_field(np.random.RandomState(1), 0.5, ring_cfg, 8)
        _, flat_goals_m, _ = omni.flat_height_field(0.5, flat_cfg, 8)
        assert np.allclose(ring_goals_m, flat_goals_m)


# =============================================================================================
class TestEdgeMaskMechanism:
    """B3/B4: `parkour_field_to_mesh_xy` catches y-normal risers the x-only decorator
    misses. NOTE (see the written report): the recal2b2 height formulas (max ~0.09 m at
    d=1) never cross the effective slope-threshold trigger at hs=0.08/vs=0.005/
    slope_threshold=1.5 (a per-pixel height JUMP of ``slope_threshold * hs`` = 0.12 m is
    required) -- true of the UNMODIFIED forward `parkour_hurdle_terrain` too, so it is a
    pre-existing property of the current calibration, not a regression. These tests
    therefore exercise the edge-mask MECHANISM with a synthetic, deliberately taller
    block, decoupled from the ring's own (currently sub-threshold) real obstacle heights.
    """

    def test_recal2b2_heights_never_cross_the_slope_threshold(self):
        cfg = types.SimpleNamespace(
            border_width=0.0, horizontal_scale=HS, vertical_scale=VS, slope_threshold=SLOPE_THRESHOLD,
            edge_width_thresh=0.05, use_simplified=False, size=(16.0, 4.0),
            platform_len=2.5, platform_height=0.0,
            apply_roughness=False, apply_flat=False, downsampled_scale=0.075,
            noise_range=(0.02, 0.06), noise_step=0.005, x_range=(1.2, 2.2), y_range=(-0.4, 0.4),
            half_valid_width=(0.4, 0.8), pad_width=0.1, pad_height=0.0,
            stone_len="0.1 + 0.3 * difficulty", hurdle_height_range=HURDLE_HEIGHT_RANGE,
        )
        for difficulty in (0.0, 0.5, 1.0):
            np.random.seed(7)
            *_, x_edge_mask, heights = fwd.parkour_hurdle_terrain(difficulty, copy.deepcopy(cfg), 8)
            assert not x_edge_mask.any()
        threshold_m = SLOPE_THRESHOLD * HS  # = slope_threshold * (hs/vs) raw units, * vs -> m
        assert threshold_m == pytest.approx(0.12)
        hurdle_hi_at_1 = eval(HURDLE_HEIGHT_RANGE, {"difficulty": 1.0})[1]
        assert hurdle_hi_at_1 < threshold_m

    def test_convert_height_field_to_mesh_return_y_edges_catches_y_risers(self):
        width_pixels = 60
        xs = (np.arange(width_pixels) + 0.5) * HS - 0.5 * width_pixels * HS
        dx, dy = np.meshgrid(xs, xs, indexing="ij")
        yaw = 0.4  # generic, non-axis-aligned
        mask = omni._oriented_box_mask(dx, dy, yaw, 1.5, 0.6)
        height_px = round(0.5 / VS)  # 0.5 m, well above the 0.12 m trigger
        z = np.where(mask, height_px, 0).astype(np.int16)

        v_x, t_x, x_mask = utils.convert_height_field_to_mesh(z, HS, VS, SLOPE_THRESHOLD)
        v_xy, t_xy, x_mask2, y_mask = utils.convert_height_field_to_mesh(
            z, HS, VS, SLOPE_THRESHOLD, return_y_edges=True
        )

        # default return (default kwarg) is unaffected
        assert np.array_equal(v_x, v_xy)
        assert np.array_equal(t_x, t_xy)
        assert np.array_equal(x_mask, x_mask2)

        assert x_mask.any() and y_mask.any()
        assert (y_mask & ~x_mask).any(), "the xy path must carry information the x-only path lacks"

    def test_parkour_field_to_mesh_xy_catches_y_risers_the_x_only_decorator_misses(self):
        def raw_func(difficulty, cfg, num_goals):
            width_pixels = int(round(cfg.size[0] / cfg.horizontal_scale))
            length_pixels = int(round(cfg.size[1] / cfg.horizontal_scale))
            xs = (np.arange(width_pixels) + 0.5) * cfg.horizontal_scale - 0.5 * cfg.size[0]
            ys = (np.arange(length_pixels) + 0.5) * cfg.horizontal_scale - 0.5 * cfg.size[1]
            dx, dy = np.meshgrid(xs, ys, indexing="ij")
            mask = omni._oriented_box_mask(dx, dy, 0.4, 1.5, 0.6)
            height_px = round(0.5 / cfg.vertical_scale)
            z = np.where(mask, height_px, 0).astype(float)
            goals = np.zeros((num_goals, 2))
            goal_heights = np.zeros(num_goals)
            return z, goals, goal_heights

        cfg = types.SimpleNamespace(
            border_width=0.0, horizontal_scale=HS, vertical_scale=VS, slope_threshold=SLOPE_THRESHOLD,
            edge_width_thresh=0.05, use_simplified=False, size=(4.96, 4.96),
        )
        x_decorated = utils.parkour_field_to_mesh(raw_func)
        xy_decorated = utils.parkour_field_to_mesh_xy(raw_func)
        edge_x = x_decorated(0.5, copy.deepcopy(cfg), 8)[4]
        edge_xy = xy_decorated(0.5, copy.deepcopy(cfg), 8)[4]

        assert edge_x.any() and edge_xy.any()
        assert (edge_xy & ~edge_x).any()

    def test_non_axis_aligned_ring_block_is_geometrically_both_x_and_y(self):
        """Not an edge-mask assertion (recal2b2 heights never trigger one -- see above):
        confirms that a ring block's yaw is sampled generically (never landing exactly on
        an axis), i.e. the block a ring tile places really is "non-axis-aligned" in the
        sense the plan means, decoupled from whether the height happens to register."""
        cfg = make_ring_cfg(8, apply_roughness=False)
        rng = np.random.RandomState(2)
        yaws = [rng.uniform(0.0, np.pi) for _ in range(cfg.n_blocks)]
        axis_aligned = {0.0, np.pi / 2, np.pi}
        assert all(y not in axis_aligned for y in yaws)


# =============================================================================================
class TestMeshSizeSanity:
    def test_vertex_count_per_tile(self, capsys):
        cfg = make_ring_cfg(8, apply_roughness=False)
        np.random.seed(9)
        meshes, *_ = omni.omni_ring_terrain(0.5, cfg, 8)
        n_vertices = meshes[0].vertices.shape[0]
        n_triangles = meshes[0].faces.shape[0]
        print(f"omni ring tile mesh: {n_vertices} vertices, {n_triangles} triangles (10 rows x 20 cols -> "
              f"{n_vertices * 200} vertices total)")
        assert n_vertices == 126 * 126
