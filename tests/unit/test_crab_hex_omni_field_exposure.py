"""Omni phase-2 plan, step 2 (2026-10-06): the FIELD-mode exposure telemetry, on CPU.

Under ``KRABBY_FLAT_TERRAIN_GEOM=field`` the blocks run to 0.16 m from every tile edge, so a foot
across a seam stands on the NEIGHBOUR's field. The ring's lookup (``tile_pixel_index``) clamps it
into the own tile and reads the own tile's empty border -- a silent under-count of exactly the
cross-seam exposure the field exists to add. Pinned here:

* ``tiled_pixel_index`` across all four seams (and the diagonal corner), off the terrain, and equal
  to ``tile_pixel_index`` on the own tile;
* the exact occupancy channel's global assembly (same indexing as the height field);
* ``feet_block_flags_tiled`` (over a block / over a neighbour's block / off the own tile);
* the ledger: the two ``OMNI_FIELD_KEYS`` exist ONLY in field mode -- ring-mode key lists and means
  are unchanged;
* the parkour term's field branch (compiled out of ``omni_parkour_event.py`` with its free names
  injected, the technique ``test_crab_hex_omni_exposure.py`` documents): accumulation, reset and
  finalize, and the ring path untouched when ``field_mode`` is False.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import textwrap
import types
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
PARKOURS_DIR = REPO / "parkour" / "parkour_isaaclab" / "envs" / "mdp" / "parkours"
OMNI_MDP = REPO / "parkour" / "parkour_tasks" / "parkour_tasks" / "crab_hex_omni_task" / "mdp"


def _load_real_module(dotted: str, path: Path) -> types.ModuleType:
    """Same get-or-create loader as ``test_crab_hex_omni_exposure.py`` (real source dirs as the
    parent packages' ``__path__``, their ``__init__`` never executed)."""
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


xs = _load_real_module("parkour_isaaclab.envs.mdp.parkours.exposure_stats", PARKOURS_DIR / "exposure_stats.py")

if str(OMNI_MDP) not in sys.path:
    sys.path.insert(0, str(OMNI_MDP))

import crab_hex_omni_command_math as omni_math  # noqa: E402
import omni_exposure as ox  # noqa: E402
import omni_terrain_knobs as tk  # noqa: E402

# the shipped geometry: 10 m tiles, 0.08 m pixels -> 126 px stored, placed 125 px apart
SIZE = (10.0, 10.0)
HS = 0.08
TILE_PX = (126, 126)
GRID = (3, 4)  # rows x cols
N_SECT = 8


def _centre(row: int, col: int) -> torch.Tensor:
    """World xy of tile (row, col)'s centre, with the generator's whole-terrain centring."""
    return torch.tensor([(row + 0.5) * SIZE[0] - GRID[0] * SIZE[0] / 2,
                         (col + 0.5) * SIZE[1] - GRID[1] * SIZE[1] / 2])


def _tiled(x, y, row=1, col=1):
    c = _centre(row, col).unsqueeze(0)
    return ox.tiled_pixel_index(
        torch.tensor([float(x)]), torch.tensor([float(y)]), c, torch.tensor([row]), torch.tensor([col]),
        SIZE, HS, TILE_PX, GRID,
    )


# ======================================================================== tiled_pixel_index
def test_equals_tile_pixel_index_on_the_own_tile():
    gen = torch.Generator().manual_seed(0)
    n = 4000
    row = torch.randint(0, GRID[0], (n,), generator=gen)
    col = torch.randint(0, GRID[1], (n,), generator=gen)
    centre = torch.stack([_centre(int(r), int(c)) for r, c in zip(row, col)])
    local = (torch.rand(n, 2, generator=gen) - 0.5) * 9.999  # strictly inside the own tile
    x, y = centre[:, 0] + local[:, 0], centre[:, 1] + local[:, 1]
    nr, nc, ix, iy, on, off = ox.tiled_pixel_index(x, y, centre, row, col, SIZE, HS, TILE_PX, GRID)
    jx, jy = ox.tile_pixel_index(x, y, centre, SIZE, HS, TILE_PX)
    assert torch.equal(ix, jx) and torch.equal(iy, jy)
    assert torch.equal(nr, row) and torch.equal(nc, col)
    assert bool(on.all()) and not bool(off.any())


@pytest.mark.parametrize("dx,dy,drow,dcol", [
    (+5.30, 0.0, +1, 0), (-5.30, 0.0, -1, 0), (0.0, +5.30, 0, +1), (0.0, -5.30, 0, -1), (+5.3, -5.3, +1, -1),
])
def test_each_seam_lands_on_the_neighbour(dx, dy, drow, dcol):
    """0.30 m past the +x / -x / +y / -y edge (and a corner): the NEIGHBOUR tile, at its own
    pixel -- tile-local 0.30 m from its near edge is pixel round(0.30 / 0.08) = 4 (or 125 - 4 from
    the far one) -- where the own-tile lookup would clamp to the border pixel 0 / 125."""
    c = _centre(1, 1)
    nr, nc, ix, iy, on, off = _tiled(c[0] + dx, c[1] + dy)
    assert (int(nr), int(nc)) == (1 + drow, 1 + dcol)
    assert bool(on) and bool(off)

    def want(d):
        if d > 5.0:
            return round((d - 5.0) / HS)
        if d < -5.0:
            return round((10.0 + (d + 5.0)) / HS)
        return round((d + 5.0) / HS)

    assert (int(ix), int(iy)) == (want(dx), want(dy))
    jx, jy = ox.tile_pixel_index(torch.tensor([c[0] + dx]), torch.tensor([c[1] + dy]),
                                 c.unsqueeze(0), SIZE, HS, TILE_PX)
    assert (int(jx), int(jy)) != (int(ix), int(iy))  # the clamp would have read the wrong pixel


def test_the_seam_point_belongs_to_the_neighbour_at_its_pixel_zero():
    """Stored 126 px, placed 125 px apart: own pixel 125 and the neighbour's pixel 0 are the same
    world point; the floor gives it to the neighbour (both read the same stamped height)."""
    c = _centre(1, 1)
    nr, nc, ix, iy, on, off = _tiled(c[0] + 5.0, c[1])
    assert (int(nr), int(ix)) == (2, 0) and bool(off)
    nr, nc, ix, iy, on, off = _tiled(c[0] + 4.999, c[1])
    assert (int(nr), int(ix)) == (1, 125) and not bool(off)


@pytest.mark.parametrize("row,col,dx,dy", [(0, 1, -5.5, 0.0), (2, 1, +5.5, 0.0), (1, 0, 0.0, -5.5),
                                           (1, 3, 0.0, +5.5)])
def test_off_the_terrain_is_off_tile_and_not_on_terrain(row, col, dx, dy):
    c = _centre(row, col)
    nr, nc, ix, iy, on, off = _tiled(c[0] + dx, c[1] + dy, row=row, col=col)
    assert not bool(on) and bool(off)
    assert 0 <= int(nr) < GRID[0] and 0 <= int(nc) < GRID[1]  # still safe to index


def test_accepts_per_foot_shapes():
    n, f = 5, 6
    row, col = torch.ones(n, dtype=torch.long), torch.ones(n, dtype=torch.long)
    centre = _centre(1, 1).expand(n, 2)
    x = centre[:, :1] + torch.linspace(-6, 6, f).expand(n, f)
    y = centre[:, 1:].expand(n, f)
    out = ox.tiled_pixel_index(x, y, centre, row, col, SIZE, HS, TILE_PX, GRID)
    assert all(t.shape == (n, f) for t in out)


# ===================================================================== occupancy + feet flags
def _masks():
    bm = torch.zeros(GRID[0], GRID[1], *TILE_PX, dtype=torch.bool)
    bm[2, 1, 2:6, 60:66] = True  # a block just past the +x seam of tile (1, 1)
    bm[1, 1, 20:24, 20:24] = True  # a block on the own tile, away from the centre (pixel 62)
    return bm


def test_global_block_mask_indexes_like_the_height_field():
    bm = _masks()
    g = ox.global_block_mask(bm.numpy(), tuple(bm.shape))
    hf = xs.global_height_field(bm.to(torch.int16))
    assert torch.equal(g, hf.bool())
    gx, gy = ox.global_pixel_index(torch.tensor([2]), torch.tensor([1]), torch.tensor([3]), torch.tensor([62]), TILE_PX)
    assert bool(g[gx, gy])


def test_global_block_mask_none_is_all_false_and_shape_checked():
    g = ox.global_block_mask(None, (GRID[0], GRID[1], *TILE_PX))
    assert g.shape == (GRID[0] * 126, GRID[1] * 126) and not bool(g.any())
    with pytest.raises(ValueError):
        ox.global_block_mask(torch.zeros(2, 2, 126, 126, dtype=torch.bool), (GRID[0], GRID[1], *TILE_PX))


def test_feet_block_flags_tiled():
    g = ox.global_block_mask(_masks(), (GRID[0], GRID[1], *TILE_PX))
    c = _centre(1, 1)
    local = lambda px: px * HS - 5.0  # noqa: E731 -- tile-local pixel -> centre-relative metres
    # env 0: one foot on the neighbour's block across the +x seam; env 1: a foot on the own block;
    # env 2: a foot across the seam on bare ground; env 3: every foot at the centre on bare ground
    feet = torch.tensor([
        [[c[0] + 5.0 + 4 * HS, c[1] + local(62)], [c[0], c[1]]],
        [[c[0] + local(21), c[1] + local(21)], [c[0], c[1]]],
        [[c[0] + 5.0 + 40 * HS, c[1] + local(62)], [c[0], c[1]]],
        [[c[0], c[1]], [c[0] + 0.1, c[1]]],
    ])
    n = feet.shape[0]
    over, over_off, off = ox.feet_block_flags_tiled(
        feet[..., 0], feet[..., 1], c.expand(n, 2), torch.ones(n, dtype=torch.long),
        torch.ones(n, dtype=torch.long), g, SIZE, HS, TILE_PX, GRID,
    )
    assert over.tolist() == [True, True, False, False]
    assert over_off.tolist() == [True, False, False, False]
    assert off.tolist() == [True, False, True, False]
    # the ring lookup reads the own tile's border for env 0 -- the under-count the field fixes
    ix, iy = ox.tile_pixel_index(feet[0:1, 0, 0], feet[0:1, 0, 1], c.unsqueeze(0), SIZE, HS, TILE_PX)
    gx, gy = ox.global_pixel_index(torch.tensor([1]), torch.tensor([1]), ix, iy, TILE_PX)
    assert not bool(g[gx, gy])


def test_a_foot_off_the_terrain_is_over_no_block():
    bm = torch.ones(GRID[0], GRID[1], *TILE_PX, dtype=torch.bool)  # everything is a block ...
    g = ox.global_block_mask(bm, tuple(bm.shape))
    c = _centre(2, 1)
    over, over_off, off = ox.feet_block_flags_tiled(
        torch.tensor([[float(c[0]) + 6.0]]), torch.tensor([[float(c[1])]]), c.unsqueeze(0),
        torch.tensor([2]), torch.tensor([1]), g, SIZE, HS, TILE_PX, GRID,
    )
    assert over.tolist() == [False] and off.tolist() == [True]  # ... except past the last tile


# ============================================================================== ledger
def test_field_keys_only_in_field_mode():
    ring = ox.OmniExposureLedger.emit_keys(N_SECT)
    field = ox.OmniExposureLedger.emit_keys(N_SECT, field_mode=True)
    assert field == ring + list(ox.OMNI_FIELD_KEYS)
    assert not set(ox.OMNI_FIELD_KEYS) & set(ring)
    assert set(ox.OmniExposureLedger(N_SECT).means()) == set(ring)
    assert set(ox.OmniExposureLedger(N_SECT, field_mode=True).means()) == set(field)


def _push(ledger, **field_kw):
    n = 4
    per_ep = ox.episode_exposure_omni(
        torch.tensor([1.0, 2.5, 3.0, 0.5]), torch.zeros(n, N_SECT, dtype=torch.bool),
        torch.tensor([0.0, 3.0, 0.0, 1.0]), torch.tensor([0.0, 5.0, 2.0, 0.0]), torch.full((n,), 10.0), 1.7,
    )
    ledger.push_episodes(
        per_ep, torch.zeros(n, dtype=torch.long), torch.tensor([False, True, True, False]),
        torch.zeros(n), torch.full((n,), 10.0), track_ratio=torch.full((n,), 0.5),
        track_ratio_valid=torch.ones(n, dtype=torch.bool), heading_err_rms=torch.zeros(n),
        block_steps=torch.tensor([0.0, 3.0, 0.0, 1.0]), r_max=torch.tensor([1.0, 2.5, 3.0, 0.5]), **field_kw,
    )


def test_field_keys_grouped_over_all_episodes():
    led = ox.OmniExposureLedger(N_SECT, field_mode=True)
    _push(led, off_tile_frac=torch.tensor([0.0, 0.2, 0.4, 0.2]),
          off_tile_block_steps=torch.tensor([0.0, 1.0, 0.0, 3.0]))
    m = led.means()
    assert m["off_tile_frac_mean"] == pytest.approx(0.2)
    assert m["off_tile_block_steps_mean"] == pytest.approx(1.0)  # flat-tile episode 3 counts too
    # reach_edge against the pad (1.7 m): episodes 1 and 2 left it
    assert m["reach_edge_frac"] == pytest.approx(1.0)  # obstacle-tile group: episodes 1, 2


def test_ring_ledger_ignores_field_kwargs_and_matches_a_plain_push():
    a, b = ox.OmniExposureLedger(N_SECT), ox.OmniExposureLedger(N_SECT)
    _push(a)
    _push(b, off_tile_frac=torch.ones(4), off_tile_block_steps=torch.ones(4))
    ma, mb = a.means(), b.means()
    assert set(ma) == set(mb)
    for k in ma:
        assert (ma[k] == mb[k]) or (ma[k] != ma[k] and mb[k] != mb[k]), k


# ======================================================================= the term's branch
OMNI_EVENT_PY = OMNI_MDP / "omni_parkour_event.py"


class _BaseParkourEvent:
    def _begin_episodes(self, env_ids) -> None:
        self.base_begin_calls.append(env_ids)

    def _failure_flags(self, env_ids) -> torch.Tensor:
        return torch.zeros(len(env_ids))


def _load_event_class():
    src = OMNI_EVENT_PY.read_text()
    tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OmniParkourEvent")
    ns = {
        "torch": torch, "np": None, "ParkourEvent": _BaseParkourEvent,
        "SLOT_MOVE": omni_math.SLOT_MOVE, "wrap_to_pi": omni_math.wrap_to_pi,
        "is_field_generator": tk.is_field_generator,
    }
    for name in ("OMNI_SPAWN_SPLIT_KEYS", "SPAWN_PLATFORM", "OmniExposureLedger", "block_mask_from_height_field",
                 "episode_exposure_omni", "feet_block_flags_tiled", "global_block_mask", "global_pixel_index",
                 "promotion_mask", "sector_index", "tile_pixel_index", "tracking_ratio"):
        ns[name] = getattr(ox, name)
    source = "from __future__ import annotations\n" + textwrap.dedent(
        " " * node.col_offset + ast.get_source_segment(src, node)
    )
    exec(compile(source, str(OMNI_EVENT_PY), "exec"), ns)  # noqa: S102
    return ns["OmniParkourEvent"]


OmniParkourEvent = _load_event_class()


def test_ring_is_the_class_default():
    assert OmniParkourEvent.field_mode is False


def _event(field_mode: bool, foot_xy):
    """Two envs on tile (1, 1), each with one foot at ``foot_xy[i]`` (world), the body at the centre."""
    ev = object.__new__(OmniParkourEvent)
    n = len(foot_xy)
    ev.num_envs, ev.device, ev.simulation_time = n, "cpu", 0.1
    ev.clear_radius, ev.n_sectors, ev.min_cmd_dist = 1.7, N_SECT, 1.0
    ev.field_mode = field_mode
    ev._foot_ids = [0]
    ev._env_arange = torch.arange(n)
    c = _centre(1, 1)
    ev.env_origins = torch.cat([c.expand(n, 2), torch.zeros(n, 1)], dim=1)
    ev.base_begin_calls = []
    ev.robot = types.SimpleNamespace(data=types.SimpleNamespace(
        root_pos_w=ev.env_origins.clone(), root_lin_vel_b=torch.zeros(n, 3), heading_w=torch.zeros(n),
        body_pos_w=torch.tensor([[[x, y, 0.0]] for x, y in foot_xy]),
    ))
    ev._cmd_term_cache = types.SimpleNamespace(
        vel_command_b=torch.zeros(n, 3), slot_id=torch.full((n,), omni_math.SLOT_MOVE, dtype=torch.long),
        heading_target=torch.zeros(n), travel_dir_w=torch.zeros(n), speed_cmd=torch.zeros(n),
    )
    ev.terrain = types.SimpleNamespace(terrain_levels=torch.ones(n, dtype=torch.long),
                                       terrain_types=torch.ones(n, dtype=torch.long))
    ev._tile_size, ev._hscale, ev._tile_px, ev._grid_rc = SIZE, HS, TILE_PX, GRID
    ev._block_mask = ox.global_block_mask(_masks(), (GRID[0], GRID[1], *TILE_PX))
    for name in ("ep_r_max", "ep_field_steps", "ep_steps", "ep_block_steps", "ep_track_dist", "ep_cmd_dist",
                 "ep_track_dist_move", "ep_cmd_dist_move", "ep_head_err_sq", "ep_off_tile_steps",
                 "ep_off_tile_block_steps"):
        setattr(ev, name, torch.zeros(n))
    ev.ep_sector_seen = torch.zeros(n, N_SECT, dtype=torch.bool)
    ev.spawn_kind = torch.zeros(n, dtype=torch.long)
    ev.is_obst_tile = torch.ones(n, dtype=torch.bool)
    ev.exposure = ox.OmniExposureLedger(N_SECT, capacity=64, field_mode=field_mode)
    ev.metrics = {}
    return ev


def _feet_across_and_home():
    c = _centre(1, 1)
    return [(float(c[0]) + 5.0 + 4 * HS, float(c[1]) + 62 * HS - 5.0), (float(c[0]), float(c[1]))]


def test_field_branch_counts_the_neighbours_block_and_the_seam():
    ev = _event(True, _feet_across_and_home())
    for _ in range(3):
        ev._accumulate_exposure()
    assert ev.ep_block_steps.tolist() == [3.0, 0.0]
    assert ev.ep_off_tile_block_steps.tolist() == [3.0, 0.0]
    assert ev.ep_off_tile_steps.tolist() == [3.0, 0.0]
    ev._finalize_episodes(torch.arange(2))
    m = ev.exposure.means()
    assert m["off_tile_frac_mean"] == pytest.approx(0.5)  # env 0: 3/3, env 1: 0/3
    assert m["off_tile_block_steps_mean"] == pytest.approx(1.5)
    assert m["block_steps_mean"] == pytest.approx(1.5)


def test_field_buffers_reset_with_the_episode():
    ev = _event(True, _feet_across_and_home())
    ev._accumulate_exposure()
    ev._begin_episodes(torch.tensor([0]))
    assert ev.ep_off_tile_steps[0] == 0.0 and ev.ep_off_tile_block_steps[0] == 0.0


def test_ring_branch_is_untouched_by_the_field_buffers():
    """``field_mode`` False: the ring lookup (own tile, clamped) runs -- the cross-seam block is
    NOT seen, the field buffers never move, and no field key reaches the ledger."""
    ev = _event(False, _feet_across_and_home())
    ev._accumulate_exposure()
    assert ev.ep_block_steps.tolist() == [0.0, 0.0]
    assert ev.ep_off_tile_steps.tolist() == [0.0, 0.0]
    ev._finalize_episodes(torch.arange(2))
    assert not set(ox.OMNI_FIELD_KEYS) & set(ev.exposure.means())
