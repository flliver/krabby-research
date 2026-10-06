"""Omni phase-2 plan, step 2 (2026-10-06): the omni FIELD geometry (``KRABBY_FLAT_TERRAIN_GEOM=field``)
in ``parkour_isaaclab/terrains/omni/omni_terrains.py``, the optional 4th body value / 7th decorator
value and the forced-wall ``riser_field`` path in ``terrains/utils.py`` -- and the proof that none of
it moved the ring or the forward corridors.

No Isaac Sim: the terrain functions are path-loaded under their real dotted names with inert package
stubs (``fixtures/gen_omni_ring_golden.py::load_terrain_modules``, the technique
``test_crab_hex_omni_ring_terrain.py`` documents), and the cfgs are ``SimpleNamespace`` stand-ins
mirroring ``OmniFieldTerrainCfg`` after ``TerrainGenerator.__init__`` propagates size / scales /
slope threshold.

What is pinned (the plan's Verification list for step 2):

* ring byte-identity -- ``fixtures/omni_ring_golden.npz`` was captured from the UNMODIFIED ring code
  (commit ec979a98) before any terrain edit; the forward decorator's own golden lives in
  ``test_crab_hex_omni_ring_terrain.py`` and must keep passing alongside this file;
* the field: pad clear, blocks inside the +-4.80 m square, coverage band, sector balance, heights
  per difficulty, pits <= 0 and never cutting raised blocks, determinism;
* the pit/raised gap (2026-10-06): every pit >= 2 clear pixels from every raised pixel, so the ramp-mode
  edge mask of the real decorated function is empty on all 10 shipped curriculum rows (40 seeds per
  row, roughness on), the kind mix is still fully placed up to d = 1.0, and bars / slabs draw the
  pre-gap RNG stream;
* walls: the edge mask equals the block boundary (the low side of every riser), the face and vertex
  counts are unchanged, and ``riser_field=None`` is the original mesh.
"""
from __future__ import annotations

import copy
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import binary_dilation, label

FIXTURES = Path(__file__).resolve().parent / "fixtures"
if str(FIXTURES) not in sys.path:
    sys.path.insert(0, str(FIXTURES))

import gen_omni_ring_golden as golden  # noqa: E402

utils, fwd, omni = golden.load_terrain_modules()

HS = 0.08
VS = 0.005
PAD_R = 1.7
HALF = 4.80
HURDLE = "0.02 + 0.05 * difficulty, 0.03 + 0.06 * difficulty"
STEP = "0.02 + 0.06 * difficulty"
PIT = "0.02 + 0.06 * difficulty"
REDUCED = (9.92, 9.92)  # the decorator-reduced 10 m tile the raw body sees


def make_field_cfg(n_blocks: int = 14, *, size=(10.0, 10.0), apply_roughness: bool = False, **kw):
    """``OmniFieldTerrainCfg`` defaults as the generator hands them to the function."""
    cfg = types.SimpleNamespace(
        border_width=0.0, horizontal_scale=HS, vertical_scale=VS, platform_len=2.5, platform_height=0.0,
        slope_threshold=1.5, edge_width_thresh=0.05, use_simplified=False, size=size,
        apply_roughness=apply_roughness, apply_flat=False, downsampled_scale=0.075,
        noise_range=(0.02, 0.02), noise_step=0.005, pad_width=0.1, pad_height=0.0,
        clear_radius=PAD_R, ring_outer_radius=HALF, field_margin=0.16, n_blocks=n_blocks,
        bar_fraction=0.4, pit_fraction=0.2, min_gap=0.0, max_tries=64,
        bar_length_range=(1.0, 2.0), bar_thickness_range=(0.16, 0.32), slab_side_range=(0.6, 1.2),
        hurdle_height_range=HURDLE, step_height=STEP, pit_depth=PIT, vertical_edges=False,
    )
    cfg.__dict__.update(kw)
    return cfg


def _raw_grid(shape=(124, 124), size=REDUCED):
    xs = (np.arange(shape[0]) + 0.5) * HS - 0.5 * size[0]
    ys = (np.arange(shape[1]) + 0.5) * HS - 0.5 * size[1]
    return np.meshgrid(xs, ys, indexing="ij")


def _layout(seed, d=0.5, n=14, **kw):
    return omni.field_layout(np.random.RandomState(seed), d, make_field_cfg(n, size=REDUCED, **kw))


class _ScriptedRng:
    """``.uniform(lo, hi)`` returning queued ABSOLUTE values (asserted inside ``[lo, hi]``)."""

    def __init__(self, values):
        self.values = list(values)

    def uniform(self, lo, hi):
        v = self.values.pop(0)
        assert lo - 1e-9 <= v <= hi + 1e-9, (lo, v, hi)
        return v


# ============================================================================== byte identity
class TestRingGoldenByteIdentity:
    """The ring course every phase-1 head trained on is unchanged, array for array."""

    @pytest.mark.parametrize("case", golden.CASES, ids=[c[0] for c in golden.CASES])
    def test_case(self, case):
        gold = np.load(golden.OUT)
        res = golden.run_case(omni, case)
        for key in golden.FIELDS:
            want = gold[f"{case[0]}__{key}"]
            assert res[key].dtype == want.dtype, key
            assert np.array_equal(res[key], want), key

    def test_ring_still_returns_six_values(self):
        np.random.seed(1)
        out = omni.omni_ring_terrain(0.5, golden.make_cfg(golden.CASES[0]), 8)
        assert len(out) == 6

    def test_riser_field_none_is_the_original_mesh(self):
        rng = np.random.RandomState(0)
        hf = (rng.rand(40, 50) * 60).astype(np.int16)  # jumps well above the 24-unit trigger
        a = utils.convert_height_field_to_mesh(hf, HS, VS, 1.5, return_y_edges=True)
        b = utils.convert_height_field_to_mesh(hf, HS, VS, 1.5, return_y_edges=True, riser_field=None)
        for x, y in zip(a, b):
            assert np.array_equal(x, y)

    def test_riser_field_of_zeros_reproduces_the_threshold_mesh(self):
        """The riser path with nothing stamped is the threshold rule alone (same moves)."""
        rng = np.random.RandomState(1)
        hf = (rng.rand(40, 50) * 60).astype(np.int16)
        a = utils.convert_height_field_to_mesh(hf, HS, VS, 1.5, return_y_edges=True)
        b = utils.convert_height_field_to_mesh(
            hf, HS, VS, 1.5, return_y_edges=True, riser_field=np.zeros_like(hf)
        )
        for x, y in zip(a, b):
            assert np.array_equal(x, y)


# ============================================================================== layout
class TestFieldLayout:
    @pytest.mark.parametrize("n", [14, 28])
    @pytest.mark.parametrize("seed", range(12))
    def test_pad_clear_and_square_bound(self, seed, n):
        raw, occ, n_placed = _layout(seed, n=n)
        dx, dy = _raw_grid()
        r = np.hypot(dx, dy)
        assert n_placed == n
        assert not occ[r < PAD_R + HS].any()
        assert np.all(raw[r < PAD_R + HS] == 0)
        assert not occ[(np.abs(dx) > HALF) | (np.abs(dy) > HALF)].any()
        # occupancy and the stamped heights describe the same pixels
        assert np.array_equal(occ, raw != 0)

    @pytest.mark.parametrize("n,lo,hi", [(14, 0.07, 0.10), (28, 0.14, 0.19)])
    def test_coverage_band(self, n, lo, hi):
        """~8.5 % / ~17 % of the tile (the ring's 0.157 / 0.313 per m2 density, plan step 2)."""
        cover = [_layout(s, n=n)[1].mean() for s in range(60)]
        assert lo <= float(np.mean(cover)) <= hi
        assert min(cover) > 0.5 * lo

    def test_sectors_balanced_per_area(self):
        """Blocks are uniform over the square: per-AREA cover is even across the 8 anchor-centred
        sectors (diagonal sectors hold more ground -- the corners -- and get proportionally more)."""
        dx, dy = _raw_grid()
        r = np.hypot(dx, dy)
        width = 2 * np.pi / 8
        sec = np.floor(np.remainder(np.arctan2(dy, dx) + 0.5 * width, 2 * np.pi) / width).astype(int)
        valid = (r >= PAD_R + HS) & (np.abs(dx) <= HALF) & (np.abs(dy) <= HALF)
        acc = np.zeros(8)
        for s in range(150):
            occ = _layout(s, n=28)[1]
            for k in range(8):
                acc[k] += occ[valid & (sec == k)].sum()
        area = np.array([(valid & (sec == k)).sum() for k in range(8)], dtype=float)
        density = acc / area
        assert np.all(np.abs(density / density.mean() - 1.0) < 0.15), density
        share = acc / acc.sum()
        assert share[1::2].sum() > share[0::2].sum()  # the corners are populated

    @pytest.mark.parametrize("d", [0.05, 0.2, 0.42, 0.7, 1.0])
    def test_heights_per_difficulty(self, d):
        hurdle_lo, hurdle_hi = eval(HURDLE, {"difficulty": d})
        step_px = round(eval(STEP, {"difficulty": d}) / VS)
        pit_px = round(eval(PIT, {"difficulty": d}) / VS)
        values = set()
        for s in range(8):
            raw, _occ, _n = _layout(s, d=d, n=28)
            values.update(np.unique(raw).tolist())
        raised = sorted(v for v in values if v > 0)
        sunk = sorted(v for v in values if v < 0)
        assert raised and sunk
        for v in raised:  # a bar height from the range, or the single slab height
            assert v == step_px or round(hurdle_lo / VS) <= v <= round(hurdle_hi / VS), (v, d)
        assert sunk == [-pit_px]  # one pit depth per difficulty: 0.02 + 0.06 d
        assert max(raised) * VS <= max(hurdle_hi, eval(STEP, {"difficulty": d})) + VS

    def test_pits_are_never_positive_and_never_cut_raised_blocks(self):
        """One slab then one pit (deterministic kind order), overlapping by 0.5 m along x: the
        overlap keeps the slab's height, the rest of the pit is sunk, nothing else moves. The
        stamping fallback, with the pit/raised gap switched off (``pit_gap_px=0``) so the overlap
        can be placed at all."""
        cfg = make_field_cfg(2, size=REDUCED, bar_fraction=0.0, pit_fraction=0.5, pit_gap_px=0)
        # slab: yaw, side, centre x, centre y; pit: the same
        rng = _ScriptedRng([0.0, 1.0, 3.0, 0.0, 0.0, 1.0, 3.5, 0.0])
        raw, occ, n = omni.field_layout(rng, 0.5, cfg)
        assert n == 2 and not rng.values
        dx, dy = _raw_grid()
        slab = omni._oriented_box_mask(dx - 3.0, dy, 0.0, 1.0, 1.0)
        pit = omni._oriented_box_mask(dx - 3.5, dy, 0.0, 1.0, 1.0)
        step_px, pit_px = round(0.05 / VS), round(0.05 / VS)
        assert np.all(raw[slab] == step_px)
        assert np.all(raw[pit & ~slab] == -pit_px)
        assert np.all(raw[~(slab | pit)] == 0)
        assert np.array_equal(occ, slab | pit)

    def test_pit_near_a_raised_block_is_redrawn_centre_only(self):
        """A pit centre whose box comes within 2 px of the slab is rejected; yaw / size / depth are kept
        and the next centre is stamped."""
        cfg = make_field_cfg(2, size=REDUCED, bar_fraction=0.0, pit_fraction=0.5)
        # slab: yaw 0, side 1.0, centre (3, 0); pit: yaw 0, side 1.0, centre (4.08, 0) -- edge-to-edge 0.08 m,
        # i.e. at most one clear pixel -- rejected, then (-3, 0) accepted
        assert omni.FIELD_PIT_GAP_PX == 2
        rng = _ScriptedRng([0.0, 1.0, 3.0, 0.0, 0.0, 1.0, 4.08, 0.0, -3.0, 0.0])
        raw, occ, n = omni.field_layout(rng, 0.5, cfg)
        assert n == 2 and not rng.values
        dx, dy = _raw_grid()
        slab = omni._oriented_box_mask(dx - 3.0, dy, 0.0, 1.0, 1.0)
        pit = omni._oriented_box_mask(dx + 3.0, dy, 0.0, 1.0, 1.0)
        assert np.array_equal(occ, slab | pit)
        assert np.all(raw[pit] < 0) and np.all(raw[slab] > 0)

    def test_the_gap_is_two_clear_pixels_not_more(self):
        """A pit exactly 2 clear pixels from a slab is accepted; 1 clear pixel is rejected (Chebyshev)."""
        dx, dy = _raw_grid()
        slab = omni._oriented_box_mask(dx - 3.0, dy, 0.0, 1.0, 1.0)
        slab_hi = int(np.where(slab.any(axis=1))[0].max())  # the slab's last pixel row along +x
        x_of = lambda i: (i + 0.5) * HS - 0.5 * REDUCED[0]  # noqa: E731 -- pixel-centre x
        for clear, accepted in ((2, True), (1, False)):
            first_pit_row = slab_hi + clear + 1
            # a 0.6 m pit whose first pixel row is first_pit_row: centre 0.3 m beyond that pixel's lower edge
            cx = x_of(first_pit_row) - 0.5 * HS + 0.3 + 1e-6
            pit = omni._oriented_box_mask(dx - cx, dy, 0.0, 0.6, 0.6)
            assert int(np.where(pit.any(axis=1))[0].min()) == first_pit_row
            cfg = make_field_cfg(2, size=REDUCED, bar_fraction=0.0, pit_fraction=0.5, max_tries=1)
            # with max_tries 1 a pit gets FIELD_PIT_TRIES_MULT tries: the rejected one is followed by tries
            # inside the pad (rejected for the pad, never placed)
            rng = _ScriptedRng([0.0, 1.0, 3.0, 0.0, 0.0, 0.6, cx, 0.0]
                               + [0.0, 0.0] * (omni.FIELD_PIT_TRIES_MULT - 1) * (not accepted))
            raw, occ, n = omni.field_layout(rng, 0.5, cfg)
            assert (n == 2) is accepted, clear
            assert np.array_equal(occ, slab | pit) if accepted else np.array_equal(occ, slab)

    def test_pad_rejection_redraws_the_centre_only(self):
        """A centre inside the pad is rejected and redrawn; yaw / size / height are kept."""
        cfg = make_field_cfg(1, size=REDUCED, bar_fraction=0.0, pit_fraction=0.0)
        rng = _ScriptedRng([0.3, 0.9, 0.0, 0.0, 3.0, -2.0])  # yaw, side, (0,0) rejected, (3,-2)
        raw, occ, n = omni.field_layout(rng, 0.5, cfg)
        assert n == 1 and not rng.values
        dx, dy = _raw_grid()
        assert np.array_equal(occ, omni._oriented_box_mask(dx - 3.0, dy + 2.0, 0.3, 0.9, 0.9))

    def test_unplaceable_block_is_counted_not_forced(self):
        # a pad wider than the square's corners (r <= 6.79 m): no pixel can ever be placed
        cfg = make_field_cfg(3, size=REDUCED, clear_radius=7.0, max_tries=5)
        raw, occ, n = omni.field_layout(np.random.RandomState(0), 0.5, cfg)
        assert n == 0 and not occ.any() and not raw.any()

    def test_min_gap_keeps_blocks_apart(self):
        for s in range(6):
            raw, occ, n = _layout(s, n=14, min_gap=0.4)
            # with a 0.4 m gap no two blocks touch: each block is its own connected component
            _lab, n_comp = label(occ)
            assert n == 14 and n_comp == n

    def test_kind_counts(self):
        assert omni.field_kind_counts(14, 0.4, 0.2) == (6, 5, 3)
        assert omni.field_kind_counts(28, 0.4, 0.2) == (11, 11, 6)
        assert omni.field_kind_counts(0, 0.4, 0.2) == (0, 0, 0)
        assert sum(omni.field_kind_counts(7, 0.5, 0.5)) == 7
        with pytest.raises(ValueError):
            omni.field_kind_counts(10, 0.7, 0.4)

    def test_determinism(self):
        for d in (0.1, 0.7):
            cfg = make_field_cfg(28, size=REDUCED, apply_roughness=True)
            np.random.seed(77)
            a = omni.field_height_field(np.random.RandomState(5), d, copy.deepcopy(cfg), 8)
            np.random.seed(77)
            b = omni.field_height_field(np.random.RandomState(5), d, copy.deepcopy(cfg), 8)
            for x, y in zip(a, b):
                assert np.array_equal(x, y)
            # and the production entry point (module-global RNG) is reproducible from the seed
            np.random.seed(9)
            c = omni.omni_field_terrain(d, make_field_cfg(28, apply_roughness=True), 8)
            np.random.seed(9)
            e = omni.omni_field_terrain(d, make_field_cfg(28, apply_roughness=True), 8)
            assert np.array_equal(c[0][0].vertices, e[0][0].vertices)
            assert np.array_equal(c[5], e[5]) and np.array_equal(c[6], e[6])

    def test_roughness_never_reaches_the_stamped_copy(self):
        cfg = make_field_cfg(14, size=REDUCED, apply_roughness=True)
        np.random.seed(3)
        hf, goals, gh, stamped = omni.field_height_field(np.random.RandomState(3), 0.5, cfg, 8)
        raw, occ, _ = _layout(3, d=0.5, n=14)
        assert stamped.dtype == np.int16
        assert np.array_equal(stamped, np.rint(raw).astype(np.int16))
        assert not np.array_equal(hf, stamped)  # roughness is in the field, not in the copy

    def test_sector_anchors_sit_between_pad_and_half_width(self):
        cfg = make_field_cfg(14, size=REDUCED)
        _hf, goals, gh, _st = omni.field_height_field(np.random.RandomState(0), 0.5, cfg, 8)
        radii = np.linalg.norm(goals - 0.5 * np.array(REDUCED), axis=1)
        assert np.allclose(radii, 0.5 * (PAD_R + HALF))
        assert np.array_equal(gh, np.zeros(8))


# ============================================================================== pit / raised gap
#: the generator's 10 curriculum rows at the shipped difficulty_range (0.20, 0.70) -- d = lo + (hi - lo) * row / 9
SHIPPED_ROWS = tuple(float(x) for x in np.linspace(0.20, 0.70, 10))
#: the same 10 rows stretched to d = 1.0 (beyond the shipped range)
FULL_ROWS = tuple(float(x) for x in np.linspace(0.20, 1.00, 10))
GAP_SEEDS = range(40)


def _stamped_bordered(seed, d, n, **kw):
    """The pre-roughness stamped heights of the tile ``omni_field_terrain`` builds after ``np.random.seed(seed)``,
    in the decorator's bordered (126 x 126) frame."""
    np.random.seed(seed)
    raw, occ, n_placed = omni.field_layout(np.random, d, make_field_cfg(n, size=REDUCED, **kw))
    rf = np.zeros((126, 126))
    rf[1:-1, 1:-1] = raw
    return rf, n_placed


class TestPitRaisedGap:
    """2026-10-06: ramp mode (``vertical_edges`` false, slope_threshold 1.5 -> a wall only above 0.12 m) must be
    ramp-only. A raised pixel next to a pit pixel stepped block + pit + roughness, over 0.12 m from d ~ 0.31 (0.13 m
    walls at d >= 0.42); every pit now keeps 2 clear pixels to every raised pixel."""

    @pytest.mark.parametrize("n", [14, 28])
    @pytest.mark.parametrize("d", SHIPPED_ROWS, ids=[f"d{d:.3f}" for d in SHIPPED_ROWS])
    def test_ramp_mode_edge_mask_is_empty_on_every_shipped_row(self, d, n):
        """The real decorated function, roughness on, 40 seeds per row: no edge pixel on any field tile."""
        cfg = make_field_cfg(n, apply_roughness=True)
        for seed in GAP_SEEDS:
            np.random.seed(seed)
            out = omni.omni_field_terrain(d, copy.deepcopy(cfg), 8)
            assert not np.asarray(out[4]).any(), (d, n, seed, int(np.asarray(out[4]).sum()))

    @pytest.mark.parametrize("n", [14, 28])
    @pytest.mark.parametrize("d", FULL_ROWS, ids=[f"d{d:.3f}" for d in FULL_ROWS])
    def test_gap_holds_and_pits_are_placed_up_to_d1(self, d, n):
        """Up to d = 1.0: every block placed (so the 0.4:0.4:0.2 mix is the drawn mix), every pit >= 2 clear px
        (Chebyshev) from every raised pixel, and no edge pixel touches a raised-to-pit junction. (Beyond the shipped
        0.70 a single block's own step plus roughness can cross 0.12 m -- a block-to-ground riser, pits or not --
        which is why the next test pins emptiness only where it holds.)"""
        cfg = make_field_cfg(n, apply_roughness=True)
        st = np.ones((5, 5), dtype=bool)
        n_bar, n_slab, n_pit = omni.field_kind_counts(n, 0.4, 0.2)
        for seed in GAP_SEEDS:
            rf, n_placed = _stamped_bordered(seed, d, n)
            assert n_placed == n, (d, n, seed)
            raised, pit = rf > 0, rf < 0
            assert pit.any() and raised.any()
            assert not (binary_dilation(raised, st) & pit).any(), (d, n, seed)
            np.random.seed(seed)
            edge = np.asarray(omni.omni_field_terrain(d, copy.deepcopy(cfg), 8)[4]).astype(bool)
            near_r, near_p = binary_dilation(raised, np.ones((3, 3), bool)), binary_dilation(pit, np.ones((3, 3), bool))
            assert not (edge & near_r & near_p).any(), (d, n, seed)
        assert (n_bar, n_slab, n_pit) == ((6, 5, 3) if n == 14 else (11, 11, 6))

    def test_beyond_the_shipped_range_only_block_to_ground_risers_remain(self):
        """d 0.8-1.0 with pits switched off shows edges of the same order as with them: the residual risers are the
        blocks' own steps + roughness, not the gap. Pinned so a later change to either side is seen."""
        def edges(d, **kw):
            tot = 0
            for seed in range(20):
                np.random.seed(seed)
                tot += int(np.asarray(omni.omni_field_terrain(d, make_field_cfg(28, apply_roughness=True, **kw), 8)[4]).sum())
            return tot
        assert edges(0.75) == 0
        with_pits, raised_only = edges(1.0), edges(1.0, pit_fraction=0.0)
        assert with_pits > 0 and raised_only > 0

    @pytest.mark.parametrize("d", [0.2, 0.42, 0.7, 1.0])
    def test_bars_and_slabs_draw_the_pre_gap_stream(self, d):
        """Raised blocks are drawn before any pit, so the gap never rejects one: the raised pixels are identical with
        the gap on and off (same seed); only pits may move."""
        for seed in range(12):
            for n in (14, 28):
                on, _ = _stamped_bordered(seed, d, n)
                off, _ = _stamped_bordered(seed, d, n, pit_gap_px=0)
                assert np.array_equal(np.where(on > 0, on, 0), np.where(off > 0, off, 0)), (d, seed, n)

    def test_pit_share_of_the_cover_is_kept(self):
        """Pits still take their share of the tile: sunken share of the occupied pixels ~ the pre-gap value."""
        def share(**kw):
            sunk = occ = 0
            for seed in range(60):
                raw, o, _n = _layout(seed, d=0.5, n=28, **kw)
                sunk += int((raw < 0).sum())
                occ += int(o.sum())
            return sunk / occ
        with_gap, without = share(), share(pit_gap_px=0)
        assert 0.2 < with_gap < 0.4
        assert abs(with_gap - without) < 0.03


# ============================================================================== decorator
class TestDecoratedField:
    def test_seven_values_and_exact_occupancy_in_the_bordered_frame(self):
        np.random.seed(4)
        out = omni.omni_field_terrain(0.5, make_field_cfg(28, apply_roughness=True), 8)
        assert len(out) == 7
        meshes, origin, goals, goal_heights, edge_mask, heights, block_mask = out
        assert heights.shape == block_mask.shape == (126, 126)
        assert block_mask.dtype == bool
        # the 1-px decorator border and the 2-px field margin stay block-free -> 0.4-0.5 m seams
        assert not block_mask[:3].any() and not block_mask[-3:].any()
        assert not block_mask[:, :3].any() and not block_mask[:, -3:].any()
        np.random.seed(4)
        raw, occ, _ = omni.field_layout(np.random, 0.5, make_field_cfg(28, size=REDUCED))
        assert np.array_equal(block_mask[1:-1, 1:-1], occ)
        # origin is the (flat) pad's height, as for the ring
        assert origin[2] <= 0.03

    def test_spawn_pad_flat_in_the_mesh(self):
        np.random.seed(8)
        *_, heights, block_mask = omni.omni_field_terrain(0.7, make_field_cfg(28), 8)
        xs = np.arange(126) * HS - 5.0  # pixel i sits at tile-local i * hs; the centre is 5.0 m
        dx, dy = np.meshgrid(xs, xs, indexing="ij")
        assert np.all(heights[np.hypot(dx, dy) < PAD_R] == 0)


# ============================================================================== walls
def _expected_riser_edges(rf: np.ndarray) -> np.ndarray:
    """The low side of every riser: a pixel with a strictly higher neighbour on EXACTLY one side
    along x, or along y (a pixel lower than both its x-neighbours keeps its x position -- the
    original rule's cancellation, kept)."""
    rf = rf.astype(np.int32)
    up_x = np.zeros(rf.shape, bool)
    dn_x = np.zeros(rf.shape, bool)
    up_x[:-1] = rf[1:] > rf[:-1]
    dn_x[1:] = rf[:-1] > rf[1:]
    up_y = np.zeros(rf.shape, bool)
    dn_y = np.zeros(rf.shape, bool)
    up_y[:, :-1] = rf[:, 1:] > rf[:, :-1]
    dn_y[:, 1:] = rf[:, :-1] > rf[:, 1:]
    return (up_x ^ dn_x) | (up_y ^ dn_y)


class TestWalls:
    @pytest.mark.parametrize("roughness", [False, True])
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_edge_mask_is_the_block_boundary(self, seed, roughness):
        cfg = make_field_cfg(28, apply_roughness=roughness, vertical_edges=True)
        np.random.seed(seed)
        out = omni.omni_field_terrain(0.42, cfg, 8)
        edge_mask, heights, block_mask = out[4], out[5], out[6]
        np.random.seed(seed)
        _hf, _g, _gh, stamped = omni.field_height_field(
            np.random, 0.42, make_field_cfg(28, size=REDUCED, apply_roughness=roughness), 8
        )
        rf = np.zeros((126, 126), dtype=np.int16)
        rf[1:-1, 1:-1] = stamped
        # half_edge_width = int(0.05 / 0.08) = 0 -> the "dilation" is the identity
        assert np.array_equal(edge_mask, _expected_riser_edges(rf))
        assert edge_mask.any()
        assert not (edge_mask & ~binary_dilation(block_mask)).any()

    def test_ramp_mode_has_no_low_block_risers(self):
        """The reason the wall option exists: at d = 0.42 every block is under the 0.12 m trigger."""
        np.random.seed(1)
        out = omni.omni_field_terrain(0.42, make_field_cfg(14, bar_fraction=0.5, pit_fraction=0.0), 8)
        assert not out[4].any()

    def test_faces_vertices_and_heights_unchanged_only_xy_moves(self):
        np.random.seed(5)
        ramp = omni.omni_field_terrain(0.7, make_field_cfg(28, apply_roughness=True), 8)
        np.random.seed(5)
        wall = omni.omni_field_terrain(0.7, make_field_cfg(28, apply_roughness=True, vertical_edges=True), 8)
        # same height field, same occupancy, same face count in the decorated mesh
        assert np.array_equal(ramp[5], wall[5]) and np.array_equal(ramp[6], wall[6])
        assert ramp[0][0].faces.shape == wall[0][0].faces.shape
        # at the conversion itself (before trimesh merges coincident vertices, which it also does
        # for any tall threshold step): identical vertex count, triangles and z; xy moves only
        heights = wall[5]
        np.random.seed(5)
        _hf, _g, _gh, stamped = omni.field_height_field(
            np.random, 0.7, make_field_cfg(28, size=REDUCED, apply_roughness=True), 8
        )
        rf = np.zeros_like(heights)
        rf[1:-1, 1:-1] = stamped
        vr, tr, *_ = utils.convert_height_field_to_mesh(heights, HS, VS, 1.5, return_y_edges=True)
        vw, tw, *_ = utils.convert_height_field_to_mesh(heights, HS, VS, 1.5, return_y_edges=True, riser_field=rf)
        assert vr.shape == vw.shape and np.array_equal(tr, tw)
        assert np.array_equal(vr[:, 2], vw[:, 2])
        assert not np.array_equal(vr[:, :2], vw[:, :2])
        # whole-pixel moves only (a riser and a threshold corner move can compose to two pixels)
        steps = np.abs(vw[:, :2] - vr[:, :2]) / HS
        assert np.allclose(steps, np.round(steps), atol=1e-4) and steps.max() <= 2.0 + 1e-4

    def test_a_riser_is_vertical(self):
        """A lone slab: the low-side vertex sits exactly under the high side's edge."""
        hf = np.zeros((20, 20), dtype=np.int16)
        hf[8:12, 8:12] = 10  # 0.05 m, far under the 24-unit trigger
        v_ramp, *_ = utils.convert_height_field_to_mesh(hf, HS, VS, 1.5)
        v_wall, *_ = utils.convert_height_field_to_mesh(hf, HS, VS, 1.5, riser_field=hf)
        assert np.array_equal(v_ramp, utils.convert_height_field_to_mesh(hf, HS, VS, 1.5)[0])
        grid_w = v_wall.reshape(20, 20, 3)
        # x-riser at i = 7 -> 8: vertex (7, 10) moved onto x of (8, 10)
        assert grid_w[7, 10, 0] == pytest.approx(grid_w[8, 10, 0])
        assert grid_w[12, 10, 0] == pytest.approx(grid_w[11, 10, 0])
        assert grid_w[10, 7, 1] == pytest.approx(grid_w[10, 8, 1])
        assert grid_w[10, 12, 1] == pytest.approx(grid_w[10, 11, 1])

    def test_riser_field_shape_mismatch_raises(self):
        hf = np.zeros((10, 10), dtype=np.int16)
        with pytest.raises(ValueError, match="riser_field"):
            utils.convert_height_field_to_mesh(hf, HS, VS, 1.5, riser_field=np.zeros((9, 10)))
