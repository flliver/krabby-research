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


def test_emit_keys_add_exactly_the_five_omni_keys():
    """FIVE since 2026-09-18: ``track_ratio_move_mean`` joins the four original omni keys.

    The slot-gated tracking ratio is WATCHED telemetry, so it is added next to
    ``track_ratio_mean`` rather than replacing it -- promotion and the driver's gate keep
    reading the original key, and the pair is what makes "did tracking really fall, or did the
    window merely enable turning?" answerable from one log.
    """
    base = set(xs.ExposureLedger.emit_keys(N_SECT))
    omni = set(ox.OmniExposureLedger.emit_keys(N_SECT))
    assert omni - base == set(ox.OMNI_EXTRA_KEYS)
    assert ox.OMNI_EXTRA_KEYS == (
        "track_ratio_mean", "track_ratio_move_mean", "heading_err_rms_mean",
        "block_steps_mean", "r_max_mean",
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


def test_move_ratio_args_are_optional_and_leave_the_mean_empty():
    """A caller that does not measure the slot-gated pair must not be forced to fake one.

    An omitted ``track_ratio_move`` leaves the ring EMPTY (NaN), which the parkour term emits
    as 0.0 -- distinguishable, in the code, from "measured and zero"; a fabricated zero would
    not be. The other four means must be unaffected by the omission.
    """
    ledger = ox.OmniExposureLedger(N_SECT, capacity=64)
    per_ep = ox.episode_exposure_omni(*_episode(), clear_radius=2.0)
    ledger.push_episodes(
        per_ep,
        spawn_kind=torch.zeros(4, dtype=torch.long),
        is_obst=torch.ones(4, dtype=torch.bool),
        failed=torch.zeros(4),
        ep_steps=_episode()[4],
        track_ratio=torch.tensor([0.4, 0.6, 0.4, 0.6]),
        track_ratio_valid=torch.ones(4, dtype=torch.bool),
        heading_err_rms=torch.zeros(4),
        block_steps=torch.zeros(4),
        r_max=torch.zeros(4),
    )
    means = ledger.means()
    assert means["track_ratio_move_mean"] != means["track_ratio_move_mean"], "empty group -> NaN"
    assert means["track_ratio_mean"] == pytest.approx(0.5)


def test_move_ratio_uses_its_own_validity_rule():
    """``track_ratio_move_valid`` selects rows INDEPENDENTLY of ``track_ratio_valid``.

    They are two different integrals with the same ``min_cmd_dist_m`` threshold: an episode
    that was commanded plenty of distance overall may have been asked to WALK for almost none
    of it (a turn-heavy episode), and counting its meaningless move-ratio would drag the
    watched number around for a reason that has nothing to do with walking.
    """
    ledger = ox.OmniExposureLedger(N_SECT, capacity=64)
    per_ep = ox.episode_exposure_omni(*_episode(), clear_radius=2.0)
    ledger.push_episodes(
        per_ep,
        spawn_kind=torch.zeros(4, dtype=torch.long),
        is_obst=torch.ones(4, dtype=torch.bool),
        failed=torch.zeros(4),
        ep_steps=_episode()[4],
        track_ratio=torch.tensor([0.1, 0.1, 0.1, 0.1]),
        track_ratio_valid=torch.ones(4, dtype=torch.bool),
        heading_err_rms=torch.zeros(4),
        block_steps=torch.zeros(4),
        r_max=torch.zeros(4),
        track_ratio_move=torch.tensor([0.8, 0.6, 500.0, 500.0]),
        track_ratio_move_valid=torch.tensor([True, True, False, False]),
    )
    means = ledger.means()
    assert means["track_ratio_move_mean"] == pytest.approx(0.7), "only the two valid rows"
    assert means["track_ratio_mean"] == pytest.approx(0.1), "the original ratio is untouched"
    # ... and with no mask at all, every pushed episode counts
    open_ledger = ox.OmniExposureLedger(N_SECT, capacity=64)
    open_ledger.push_episodes(
        per_ep,
        spawn_kind=torch.zeros(4, dtype=torch.long),
        is_obst=torch.ones(4, dtype=torch.bool),
        failed=torch.zeros(4),
        ep_steps=_episode()[4],
        track_ratio=torch.zeros(4),
        track_ratio_valid=torch.ones(4, dtype=torch.bool),
        heading_err_rms=torch.zeros(4),
        block_steps=torch.zeros(4),
        r_max=torch.zeros(4),
        track_ratio_move=torch.tensor([0.2, 0.4, 0.6, 0.8]),
    )
    assert open_ledger.means()["track_ratio_move_mean"] == pytest.approx(0.5)


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


# ========================================================== the term's slot-gated telemetry
#
# ``omni_parkour_event.py`` is the OTHER half of the watched tracking number: the ledger above
# only means anything if the term integrates the right rows into it. The module cannot be
# imported (its base class reaches Isaac through ``parkour_isaaclab.terrains``), so the ONE
# class statement under test is compiled out of the source with the names it closes over
# injected -- the ``_exec_module_level`` idea of ``test_crab_hex_omni_rewards.py``, widened to
# a ClassDef. Nothing else in the file is executed and nothing is registered in ``sys.modules``.
import ast  # noqa: E402
import textwrap  # noqa: E402

import crab_hex_omni_command_math as omni_math  # noqa: E402

OMNI_EVENT_PY = OMNI_MDP / "omni_parkour_event.py"


class _BaseParkourEvent:
    """Stand-in for ``ParkourEvent``: only the two hooks the overrides under test call up to."""

    def _begin_episodes(self, env_ids) -> None:
        self.base_begin_calls.append(env_ids)

    def _failure_flags(self, env_ids) -> torch.Tensor:
        return torch.zeros(len(env_ids))


def _load_event_class():
    src = OMNI_EVENT_PY.read_text()
    tree = ast.parse(src)
    node = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OmniParkourEvent"
    )
    ns = {
        "torch": torch,
        "np": None,  # only the debug-vis path uses numpy, and no test drives it
        "ParkourEvent": _BaseParkourEvent,
        "SLOT_MOVE": omni_math.SLOT_MOVE,
        "wrap_to_pi": omni_math.wrap_to_pi,
    }
    for name in ("SPAWN_PLATFORM", "OmniExposureLedger", "block_mask_from_height_field",
                 "episode_exposure_omni", "global_pixel_index", "promotion_mask",
                 "sector_index", "tile_pixel_index", "tracking_ratio"):
        ns[name] = getattr(ox, name)
    source = "from __future__ import annotations\n" + textwrap.dedent(
        " " * node.col_offset + ast.get_source_segment(src, node)
    )
    exec(compile(source, str(OMNI_EVENT_PY), "exec"), ns)  # noqa: S102
    return ns["OmniParkourEvent"]


OmniParkourEvent = _load_event_class()

EV_N = 4
EV_DT = 0.1


def _event(slot_ids, *, cmd_xy=(1.0, 0.0), speed_b=(1.0, 0.0), min_cmd_dist=1.0):
    """A bare term with only the attributes the three methods under test read.

    ``object.__new__``: the real ``__init__`` builds a block mask off the terrain. Every env
    gets the SAME kinematics and the same command magnitude, so the only thing that can
    separate the rows in what follows is the slot.
    """
    ev = object.__new__(OmniParkourEvent)
    n = len(slot_ids)
    ev.num_envs = n
    ev.device = "cpu"
    ev.simulation_time = EV_DT
    ev.clear_radius = 2.0
    ev.n_sectors = N_SECT
    ev.min_cmd_dist = min_cmd_dist
    ev._foot_ids = []
    ev._env_arange = torch.arange(n)
    ev.env_origins = torch.zeros(n, 3)
    ev.base_begin_calls = []
    ev.robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            root_pos_w=torch.zeros(n, 3),
            root_lin_vel_b=torch.tensor([list(speed_b) for _ in range(n)]),
            heading_w=torch.zeros(n),
        )
    )
    ev._cmd_term_cache = types.SimpleNamespace(
        vel_command_b=torch.tensor([list(cmd_xy) + [0.0] for _ in range(n)]),
        slot_id=torch.tensor(list(slot_ids), dtype=torch.long),
        heading_target=torch.zeros(n),
        travel_dir_w=torch.zeros(n),
        speed_cmd=torch.zeros(n),
    )
    for name in ("ep_r_max", "ep_field_steps", "ep_steps", "ep_block_steps", "ep_track_dist",
                 "ep_cmd_dist", "ep_track_dist_move", "ep_cmd_dist_move", "ep_head_err_sq"):
        setattr(ev, name, torch.zeros(n))
    ev.ep_sector_seen = torch.zeros(n, N_SECT, dtype=torch.bool)
    ev.spawn_kind = torch.zeros(n, dtype=torch.long)
    ev.is_obst_tile = torch.ones(n, dtype=torch.bool)
    ev.exposure = ox.OmniExposureLedger(N_SECT, capacity=64)
    return ev


def test_move_distances_accumulate_on_SLOT_MOVE_rows_only():
    """The gate is ``slot_id == SLOT_MOVE`` -- moving and NOT turning.

    ``SLOT_MOVE_TURN`` is the row the watched number exists to exclude: its body command
    swings as the commanded turn executes, so it keeps adding to the denominator while the
    robot is correctly doing something else. A gate of ``slot_is_moving`` instead of
    ``== SLOT_MOVE`` would silently let it back in, and the number would fall for a reason
    that has nothing to do with walking.
    """
    ev = _event([omni_math.SLOT_STAND, omni_math.SLOT_MOVE,
                 omni_math.SLOT_TURN, omni_math.SLOT_MOVE_TURN])
    ev._accumulate_exposure()
    # the unrestricted pair integrates over EVERY row, as it always has
    assert ev.ep_cmd_dist.tolist() == pytest.approx([EV_DT] * 4)
    assert ev.ep_track_dist.tolist() == pytest.approx([EV_DT] * 4)
    # the slot-gated twin counts the one walking row and nothing else
    assert ev.ep_cmd_dist_move.tolist() == pytest.approx([0.0, EV_DT, 0.0, 0.0])
    assert ev.ep_track_dist_move.tolist() == pytest.approx([0.0, EV_DT, 0.0, 0.0])


def test_move_distances_are_the_same_integrands_as_the_unrestricted_pair():
    """On an all-``SLOT_MOVE`` population the two pairs must be BYTE-identical.

    That is what makes ``track_ratio_move_mean`` comparable with ``track_ratio_mean``: the
    only difference between them is which rows are in the sum, never how a row is measured
    (a swapped numerator/denominator, or a missing ``dt``, would show up here)."""
    ev = _event([omni_math.SLOT_MOVE] * EV_N, cmd_xy=(0.3, -0.4), speed_b=(0.2, 0.1))
    for _ in range(5):
        ev._accumulate_exposure()
    assert torch.equal(ev.ep_track_dist, ev.ep_track_dist_move)
    assert torch.equal(ev.ep_cmd_dist, ev.ep_cmd_dist_move)
    # ... and they are the projected / commanded distances, not something else
    assert ev.ep_cmd_dist.tolist() == pytest.approx([5 * EV_DT * 0.5] * EV_N)   # |(0.3,-0.4)|
    # v . c_hat = (0.2, 0.1) . (0.6, -0.8) = 0.04 (the 1e-6 guard in c_hat costs ~2e-6 rel)
    assert ev.ep_track_dist.tolist() == pytest.approx([5 * EV_DT * 0.04] * EV_N, rel=1e-4)


def test_move_distances_reset_with_the_episode():
    """Both buffers clear in ``_begin_episodes``, exactly like their two siblings.

    A missing reset leaks the previous episode's distance into the next one -- silent, and
    it biases the watched mean upward for exactly the envs that reset most often."""
    ev = _event([omni_math.SLOT_MOVE] * EV_N)
    ev._accumulate_exposure()
    assert float(ev.ep_cmd_dist_move.sum()) > 0.0
    ids = torch.tensor([0, 2])
    ev._begin_episodes(ids)
    assert ev.base_begin_calls == [ids], "the base bookkeeping still runs first"
    for name in ("ep_track_dist", "ep_cmd_dist", "ep_track_dist_move", "ep_cmd_dist_move"):
        buf = getattr(ev, name)
        assert buf[0] == 0.0 and buf[2] == 0.0, name
        assert buf[1] > 0.0 and buf[3] > 0.0, f"{name}: only the reset rows"


def test_finalize_pushes_the_move_ratio_with_its_own_validity_rule():
    """``track_ratio_move_mean`` reaches the ledger from the term, gated on the WALKING rows'
    commanded distance -- the same ``min_cmd_dist_m`` threshold the existing ratio uses.

    Row 0 walks the whole episode; row 1 is commanded the same total distance but almost all
    of it while turning, so its move-ratio is meaningless and must be left OUT of the mean
    rather than counted as a near-zero."""
    ev = _event([omni_math.SLOT_MOVE] * EV_N, min_cmd_dist=1.0)
    ev.ep_steps = torch.full((EV_N,), 10.0)
    ev.ep_cmd_dist = torch.tensor([4.0, 4.0, 4.0, 4.0])
    ev.ep_track_dist = torch.tensor([2.0, 2.0, 2.0, 2.0])
    ev.ep_cmd_dist_move = torch.tensor([4.0, 0.5, 2.0, 0.0])
    ev.ep_track_dist_move = torch.tensor([3.0, 0.4, 1.0, 0.0])
    ev._finalize_episodes(torch.arange(EV_N))
    means = ev.exposure.means()
    # rows 0 and 2 are valid (>= 1.0 m commanded while walking): 0.75 and 0.5
    assert means["track_ratio_move_mean"] == pytest.approx(0.625)
    # the unrestricted ratio is untouched and keeps its own (all-valid) population
    assert means["track_ratio_mean"] == pytest.approx(0.5)
    assert ev._exposure_means["track_ratio_move_mean"] == pytest.approx(0.625)


def test_the_command_term_probe_requires_the_slot_the_gate_reads():
    """``_cmd_term`` fails with a message on a command term without ``slot_id``.

    The gate reads it every step; without the probe a wrong command class would surface as an
    ``AttributeError`` in the middle of a step instead of at the first lookup."""
    ev = _event([omni_math.SLOT_MOVE])
    ev._cmd_term_cache = None
    ev._command_name = "base_velocity"
    full = types.SimpleNamespace(
        travel_dir_w=None, speed_cmd=None, heading_target=None,
        vel_command_b=None, slot_id=None,
    )
    for missing in ("slot_id", "travel_dir_w"):
        term = types.SimpleNamespace(**{k: v for k, v in vars(full).items() if k != missing})
        ev.env = types.SimpleNamespace(
            command_manager=types.SimpleNamespace(get_term=lambda name, t=term: t)
        )
        with pytest.raises(TypeError, match=missing):
            _ = ev._cmd_term
    ev.env = types.SimpleNamespace(
        command_manager=types.SimpleNamespace(get_term=lambda name: full)
    )
    assert ev._cmd_term is full
