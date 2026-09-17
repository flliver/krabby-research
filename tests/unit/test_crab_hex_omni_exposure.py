"""Omni ring exposure, sectors, block mask and ratio promotion (plan D9 / spec B5), on CPU.

Everything asserted here is a QUIET failure mode in training: a promotion rule that never
fires leaves the terrain curriculum frozen at level 0 for 5000 iterations and reports
nothing; a sector index off by half a sector halves ``goals_passed``; a block mask indexed
with the wrong pixel frame reports obstacle contact from the wrong side of the tile. None of
those raise, and none of them show up in the reward.

The module is path-loaded (no ``parkour_tasks`` import, no Isaac Sim): ``omni_exposure``'s
only non-torch dependency is the forward ``exposure_stats``, which is itself pure torch and
is registered here under its real dotted name so the ledger under test is the REAL one --
the whole point of the omni ledger is that it keeps the forward key names.
"""
import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
PARKOURS_DIR = REPO / "parkour" / "parkour_isaaclab" / "envs" / "mdp" / "parkours"
OMNI_MDP = REPO / "parkour" / "parkour_tasks" / "parkour_tasks" / "crab_hex_omni_task" / "mdp"


def _load_real_module(dotted: str, path: Path) -> types.ModuleType:
    """Register the real file at ``path`` under ``dotted``, creating parent packages.

    The parent packages get the REAL source directories as their ``__path__`` rather than an
    empty one: other suites in the same pytest process path-load different modules out of
    the same dotted tree (``test_crab_hex_omni_command.py`` imports
    ``parkour_isaaclab.envs.mdp.travel_frame`` this way), and a stub package with no search
    path would make whichever suite ran second fail to import. Their real ``__init__.py``
    files are still never executed -- that is the point, they reach Isaac.
    """
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


xs = _load_real_module(
    "parkour_isaaclab.envs.mdp.parkours.exposure_stats", PARKOURS_DIR / "exposure_stats.py"
)

if str(OMNI_MDP) not in sys.path:
    sys.path.insert(0, str(OMNI_MDP))

import omni_exposure as ox  # noqa: E402

N_SECT = 8
TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- ledger keys
#
# These names are a contract with the campaign driver, which greps
# ``Metrics/base_parkour/<key>`` out of the training log
# (``crab_hex_sideways_task/experiments/tools/run_phases.py:89``). Renaming one is invisible
# until a campaign reports "n/a" for every exposure column.
DRIVER_BASE = ["reach_edge_frac", "reach_obst_frac", "field_frac_mean", "field_steps_mean",
               "goals_passed_mean"]


def test_emit_keys_keep_every_key_the_driver_greps():
    keys = set(ox.OmniExposureLedger.emit_keys(N_SECT))
    for base in DRIVER_BASE:
        assert base in keys and f"{base}_rsi" in keys
    for k in range(1, N_SECT + 1):
        assert f"obst_coverage_{k}" in keys and f"obst_coverage_{k}_rsi" in keys
    for extra in ("crab_failure_flat", "crab_failure_obst", "crab_failure_hazard_flat",
                  "crab_failure_hazard_obst", "ep_steps_flat", "ep_steps_obst",
                  "spread_frac_actual", "rsi_frac_actual"):
        assert extra in keys


def test_emit_keys_add_exactly_the_four_omni_keys():
    base = set(xs.ExposureLedger.emit_keys(N_SECT))
    omni = set(ox.OmniExposureLedger.emit_keys(N_SECT))
    assert omni - base == set(ox.OMNI_EXTRA_KEYS)
    assert ox.OMNI_EXTRA_KEYS == (
        "track_ratio_mean", "heading_err_rms_mean", "block_steps_mean", "r_max_mean"
    )


def test_coverage_key_count_follows_n_sectors():
    for n in (4, 8, 12):
        keys = ox.OmniExposureLedger.emit_keys(n)
        assert f"obst_coverage_{n}" in keys and f"obst_coverage_{n + 1}" not in keys


# ---------------------------------------------------------------------------- promotion


def test_promotion_mask_up_down_and_hold():
    track = torch.tensor([9.0, 1.0, 3.5, 9.0])  # ratios 0.90 / 0.10 / 0.35 (between) / --
    cmd = torch.tensor([10.0, 10.0, 10.0, 0.5])  # last: barely commanded -> HOLD
    up, down = ox.promotion_mask(track, cmd, 0.45, 0.25, min_cmd_dist=1.0)
    assert up.tolist() == [True, False, False, False]
    assert down.tolist() == [False, True, False, False]


def test_promotion_thresholds_are_strict_and_exclusive():
    # exactly at the thresholds -> neither promote nor demote
    track = torch.tensor([4.5, 2.5])
    cmd = torch.tensor([10.0, 10.0])
    up, down = ox.promotion_mask(track, cmd, 0.45, 0.25, min_cmd_dist=1.0)
    assert not up.any() and not down.any()


def test_promotion_is_a_ratio_not_a_distance():
    """Scaling the episode (a longer horizon, a faster resample) must not change the vote.

    This is the fix for the corridor rule, whose threshold was ``frac * cmd_vx *
    episode_length_s`` and therefore had to be recalibrated whenever the horizon moved.
    """
    track, cmd = torch.tensor([3.0]), torch.tensor([5.0])
    for scale in (0.5, 1.0, 4.0, 17.0):
        up, down = ox.promotion_mask(track * scale, cmd * scale, 0.45, 0.25, min_cmd_dist=1.0)
        assert up.tolist() == [True] and down.tolist() == [False]


def test_promotion_holds_a_standing_episode_instead_of_demoting_it():
    """A stand-only episode has cmd == 0: the ratio is 0/0, and voting it would make the
    curriculum a function of the slot mix."""
    up, down = ox.promotion_mask(torch.zeros(1), torch.zeros(1), 0.45, 0.25, min_cmd_dist=1.0)
    assert not up.any() and not down.any()


def test_promotion_demotes_a_blocked_robot():
    """Accepted behaviour (plan caveat 9): commanded 8 m, travelled 0.1 m -> demote."""
    up, down = ox.promotion_mask(torch.tensor([0.1]), torch.tensor([8.0]), 0.45, 0.25, 1.0)
    assert down.tolist() == [True] and not up.any()


def test_promotion_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        ox.promotion_mask(torch.zeros(3), torch.zeros(2), 0.45, 0.25, 1.0)


# ------------------------------------------------------------------------------ sectors


def test_sector_index_is_centred_on_the_ring_anchors():
    """Anchor k sits at bearing k*2pi/n; walking straight at it must stay in sector k."""
    for k in range(N_SECT):
        ang = torch.tensor([k * TWO_PI / N_SECT])
        for r in (2.5, 4.0):
            dx, dy = r * torch.cos(ang), r * torch.sin(ang)
            assert int(ox.sector_index(dx, dy, N_SECT)[0]) == k


def test_sector_index_sector_zero_contains_world_plus_x():
    half = math.pi / N_SECT
    inside = torch.tensor([-0.99 * half, 0.0, 0.99 * half])
    idx = ox.sector_index(torch.cos(inside), torch.sin(inside), N_SECT)
    assert idx.tolist() == [0, 0, 0]


def test_sector_index_steps_at_the_half_sector_boundary():
    half = math.pi / N_SECT
    just_past = torch.tensor([1.001 * half])
    assert int(ox.sector_index(torch.cos(just_past), torch.sin(just_past), N_SECT)[0]) == 1
    just_before = torch.tensor([-1.001 * half])
    assert int(ox.sector_index(torch.cos(just_before), torch.sin(just_before), N_SECT)[0]) == N_SECT - 1


def test_sector_index_partitions_the_circle_uniformly():
    ang = torch.linspace(-math.pi, math.pi, 8001)[:-1]
    idx = ox.sector_index(torch.cos(ang), torch.sin(ang), N_SECT)
    counts = torch.bincount(idx, minlength=N_SECT)
    assert int(idx.min()) == 0 and int(idx.max()) == N_SECT - 1
    assert int(counts.max() - counts.min()) <= 2


def test_sector_index_rejects_non_positive_n():
    with pytest.raises(ValueError):
        ox.sector_index(torch.zeros(1), torch.zeros(1), 0)


def test_in_clear_disc_boundary_belongs_to_the_disc():
    dx = torch.tensor([0.0, 1.999, 2.0, 2.001])
    dy = torch.zeros(4)
    assert ox.in_clear_disc(dx, dy, 2.0).tolist() == [True, True, True, False]
    assert torch.allclose(ox.radius_xy(torch.tensor([3.0]), torch.tensor([4.0])), torch.tensor([5.0]))


# --------------------------------------------------------------------------- block mask

W, L = 7, 5          # pixels per tile
ROWS, COLS = 2, 3    # tiles
HS, VS = 0.1, 0.005
SIZE = ((W - 1) * HS, (L - 1) * HS)


def _synthetic_field():
    """Global field with a per-tile pad offset, roughness, and one raised patch per tile."""
    hf = torch.zeros(ROWS, W, COLS, L, dtype=torch.int16)
    for r in range(ROWS):
        for c in range(COLS):
            pad = 10 * (r + 2 * c)                     # tiles sit at different heights
            tile = torch.full((W, L), pad, dtype=torch.int16)
            tile[0, 0] = pad + 2                       # +1 cm: roughness, below tolerance
            tile[3:5, 1:3] = pad + 12                  # +6 cm: a block
            hf[r, :, c, :] = tile
    return hf.reshape(ROWS * W, COLS * L)


def test_block_mask_finds_the_raised_patch_and_nothing_else():
    hf = _synthetic_field()
    tol_px = 0.015 / VS  # 3 raw units
    mask = ox.block_mask_from_height_field(hf, (W, L), tol_px)
    assert mask.shape == hf.shape
    for r in range(ROWS):
        for c in range(COLS):
            tile = mask[r * W:(r + 1) * W, c * L:(c + 1) * L]
            assert bool(tile[3:5, 1:3].all()), "the +6 cm patch must be a block"
            assert int(tile.sum()) == 4, "nothing else may be marked (per-tile pad, +1 cm noise below tol)"


def test_block_mask_pad_is_per_tile_not_global():
    """A global threshold would mark every pixel of the highest tile."""
    hf = _synthetic_field()
    mask = ox.block_mask_from_height_field(hf, (W, L), 0.015 / VS)
    top_tile = mask[1 * W:2 * W, 2 * L:3 * L]  # pad = 10*(1+4) = 50 raw = 25 cm
    assert int(top_tile.sum()) == 4


def test_block_mask_tolerance_moves_the_threshold():
    hf = _synthetic_field()
    loose = ox.block_mask_from_height_field(hf, (W, L), 0.005 / VS)  # 1 raw unit: +1cm counts
    tight = ox.block_mask_from_height_field(hf, (W, L), 0.10 / VS)   # 20 raw units: nothing
    assert int(loose[:W, :L].sum()) == 5
    assert int(tight.sum()) == 0


def test_block_mask_rejects_a_ragged_tiling():
    with pytest.raises(ValueError):
        ox.block_mask_from_height_field(torch.zeros(9, 5, dtype=torch.int16), (W, L), 1.0)


def test_tile_pixel_index_maps_the_centre_to_the_middle_pixel():
    centre = torch.tensor([[3.0, -7.0]])
    ix, iy = ox.tile_pixel_index(centre[:, 0], centre[:, 1], centre, SIZE, HS, (W, L))
    assert int(ix[0]) == (W - 1) // 2 and int(iy[0]) == (L - 1) // 2


def test_tile_pixel_index_clamps_outside_the_tile():
    centre = torch.tensor([[0.0, 0.0]])
    far = torch.tensor([100.0, -100.0])
    ix, iy = ox.tile_pixel_index(far, -far, centre.expand(2, 2), SIZE, HS, (W, L))
    assert ix.tolist() == [W - 1, 0] and iy.tolist() == [0, L - 1]


def test_block_lookup_round_trip_through_the_global_index():
    """The whole chain a foot goes through: world xy -> tile pixel -> global pixel -> mask.

    The tile pixel frame is exact per tile (``W = size/hs + 1`` pixels spanning ``size``
    metres); the shared global formula in ``exposure_stats.world_to_pixel`` drifts by one
    pixel per tile row, which is why the omni term does not use it.
    """
    hf = _synthetic_field()
    mask = ox.block_mask_from_height_field(hf, (W, L), 0.015 / VS)
    # tile (row=1, col=2), centre placed at an arbitrary world position
    centre = torch.tensor([[12.5, -4.25]])
    row = torch.tensor([1])
    col = torch.tensor([2])
    # block pixels are ix in {3, 4}, iy in {1, 2}; pixel i sits at tile-local i*hs
    for ipx, jpx, expect in ((3, 1, True), (4, 2, True), (0, 0, False), (6, 4, False)):
        x_w = centre[:, 0] - 0.5 * SIZE[0] + ipx * HS
        y_w = centre[:, 1] - 0.5 * SIZE[1] + jpx * HS
        ix, iy = ox.tile_pixel_index(x_w, y_w, centre, SIZE, HS, (W, L))
        assert (int(ix[0]), int(iy[0])) == (ipx, jpx)
        gx, gy = ox.global_pixel_index(row, col, ix, iy, (W, L))
        assert bool(mask[gx, gy][0]) is expect


# ------------------------------------------------------------------------ episode record


def _episode(n=4, n_sect=N_SECT):
    r_max = torch.tensor([0.5, 2.5, 4.0, 3.0])[:n]
    seen = torch.zeros(n, n_sect, dtype=torch.bool)
    seen[1, 0] = True
    seen[2, 0] = True
    seen[2, 3] = True
    seen[3, 7] = True
    block = torch.tensor([0.0, 0.0, 12.0, 3.0])[:n]
    field = torch.tensor([0.0, 40.0, 300.0, 100.0])[:n]
    steps = torch.tensor([400.0, 400.0, 400.0, 200.0])[:n]
    return r_max, seen, block, field, steps


def test_episode_exposure_omni_has_exactly_the_ledger_keys():
    per_ep = ox.episode_exposure_omni(*_episode(), clear_radius=2.0)
    expected = {"reach_edge", "reach_obst", "goals_passed", "field_steps", "field_frac"}
    expected |= {f"obst_coverage_{k}" for k in range(1, N_SECT + 1)}
    assert set(per_ep) == expected
    # the forward ledger consumes exactly these -- pushing must not KeyError
    ledger = ox.OmniExposureLedger(N_SECT, capacity=8)
    ledger.push_episodes(
        per_ep, torch.zeros(4, dtype=torch.long), torch.ones(4, dtype=torch.bool),
        torch.zeros(4), _episode()[4],
        track_ratio=torch.zeros(4), track_ratio_valid=torch.ones(4, dtype=torch.bool),
        heading_err_rms=torch.zeros(4), block_steps=torch.zeros(4), r_max=torch.zeros(4),
    )


def test_episode_exposure_omni_values():
    r_max, seen, block, field, steps = _episode()
    per_ep = ox.episode_exposure_omni(r_max, seen, block, field, steps, clear_radius=2.0)
    assert per_ep["reach_edge"].tolist() == [0.0, 1.0, 1.0, 1.0]
    assert per_ep["reach_obst"].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert per_ep["goals_passed"].tolist() == [0.0, 1.0, 2.0, 1.0]
    assert per_ep["field_frac"].tolist() == pytest.approx([0.0, 0.1, 0.75, 0.5])
    # goals_passed is the sum of the per-sector coverage keys, by construction
    cov = torch.stack([per_ep[f"obst_coverage_{k}"] for k in range(1, N_SECT + 1)], dim=1)
    assert torch.equal(cov.sum(dim=1), per_ep["goals_passed"])


def test_episode_exposure_omni_rejects_a_misaligned_sector_table():
    r_max, seen, block, field, steps = _episode()
    with pytest.raises(ValueError):
        ox.episode_exposure_omni(r_max, seen[:2], block, field, steps, clear_radius=2.0)


# ------------------------------------------------------------------------ ledger groups


def test_extra_keys_grouping():
    """track_ratio excludes under-commanded episodes; block_steps is obstacle-tiles only."""
    ledger = ox.OmniExposureLedger(N_SECT, capacity=64)
    per_ep = ox.episode_exposure_omni(*_episode(), clear_radius=2.0)
    is_obst = torch.tensor([True, True, False, False])
    ledger.push_episodes(
        per_ep,
        spawn_kind=torch.zeros(4, dtype=torch.long),
        is_obst=is_obst,
        failed=torch.zeros(4),
        ep_steps=_episode()[4],
        track_ratio=torch.tensor([0.9, 0.1, 100.0, 100.0]),
        track_ratio_valid=torch.tensor([True, True, False, False]),
        heading_err_rms=torch.tensor([0.1, 0.2, 0.3, 0.4]),
        block_steps=torch.tensor([2.0, 4.0, 1000.0, 1000.0]),
        r_max=torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )
    means = ledger.means()
    assert means["track_ratio_mean"] == pytest.approx(0.5)      # only the two valid rows
    assert means["block_steps_mean"] == pytest.approx(3.0)      # only the two obstacle tiles
    assert means["heading_err_rms_mean"] == pytest.approx(0.25)  # all four
    assert means["r_max_mean"] == pytest.approx(2.5)            # all four


def test_extra_keys_are_nan_before_any_episode():
    means = ox.OmniExposureLedger(N_SECT, capacity=8).means()
    for key in ox.OMNI_EXTRA_KEYS:
        assert means[key] != means[key], f"{key} should start as NaN (empty group)"


def test_means_still_carry_every_forward_key():
    means = ox.OmniExposureLedger(N_SECT, capacity=8).means()
    assert set(ox.OmniExposureLedger.emit_keys(N_SECT)) == set(means)


# -------------------------------------------------------------------- spawn reporting


class _Term:
    def __init__(self):
        self.calls = []

    def note_spawn_xy(self, env_ids, xy, kind):
        self.calls.append((env_ids, xy.clone(), kind))


class _OtherTerm:
    """A parkour term without note_spawn_xy (e.g. the forward term) must be skipped."""


def _env_with_terms(terms):
    pm = types.SimpleNamespace(active_terms=list(terms), get_term=lambda n: terms[n])
    return types.SimpleNamespace(parkour_manager=pm)


def test_note_spawn_xy_reaches_only_terms_that_implement_it():
    term = _Term()
    env = _env_with_terms({"base_parkour": term, "other": _OtherTerm()})
    xy = torch.tensor([[1.0, 2.0]])
    ox.note_spawn_xy_to_parkour(env, torch.tensor([0]), xy, ox.SPAWN_RSI)
    assert len(term.calls) == 1 and term.calls[0][2] == ox.SPAWN_RSI
    assert torch.equal(term.calls[0][1], xy)


def test_note_spawn_xy_tolerates_an_env_without_a_parkour_manager():
    ox.note_spawn_xy_to_parkour(types.SimpleNamespace(), torch.tensor([0]), torch.zeros(1, 2))


def test_default_spawn_kind_is_platform():
    term = _Term()
    ox.note_spawn_xy_to_parkour(_env_with_terms({"p": term}), torch.tensor([0]), torch.zeros(1, 2))
    assert term.calls[0][2] == ox.SPAWN_PLATFORM == xs.SPAWN_PLATFORM
