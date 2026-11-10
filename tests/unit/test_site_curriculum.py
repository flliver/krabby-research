# SPDX-License-Identifier: BSD-3-Clause
"""Site-mode per-kind TELEMETRY of the omni parkour term (rough-terrain port, ``experiments/2026-11-10_rough_terrain/
PLAN.md`` step 1, 2026-11-10; the subset of the closed site teacher's ``tests/unit/test_site_curriculum.py`` at
``worktree-sen0581-cameras`` 2a3e3c51 that covers what was ported). CPU only, no Isaac.

``omni_exposure`` / ``site_raster`` are path-loaded under their real dotted names with stub parent packages; the
``OmniParkourEvent`` class statement is compiled out of ``omni_parkour_event.py`` with the names it closes over
injected (the ``test_crab_hex_omni_obst_telemetry.py`` technique) and driven on a bare stand-in (``object.__new__``;
the real ``__init__`` reaches the terrain importer). For the neutrality pins, main's class (``git show
3f944c4b:...``, origin/main before the port) is compiled the same way and both are run on identical stand-ins.

Pinned here:

* :class:`omni_exposure.SiteKindLedger` (lazy per-kind windows, the p50 rule, crossed only for crossing kinds),
  :func:`site_level_extras`, :func:`site_count_extras`;
* the parkour term in site mode: ``site_level/<kind>`` read before tiles move, ``site_promote`` / ``site_demote``
  per kind of the tile the episode ENDED on, the lazy ledger keys reaching ``self.metrics``, freeze, loop-back;
* the promotion rule in site mode IS main's ratio rule (``promotion_mask``, ``KRABBY_TERRAIN_PROMOTE``): the same
  levels and the same values of every one of main's reset-log keys as main's class, falls and crossings never vote;
  the per-kind counts (``site_cnt_*``) count its APPLIED masks;
* the metres tag (on the column's own ladder, ``b4`` honoured); counts equal episode ends under a stale-dict runner
  replay (the zero fill), every other key logged as without it;
* off site mode the class is main's on the ring / field: same reset log, same levels, same metric keys;
* the unported site attributes (``site_promote`` / ``site_crash_n`` / rule ``loc`` / ``site_slack_geom``) are refused
  by ``__init__`` (source pin: the real ``__init__`` needs the importer);
* the hazard radii = ``site_terrains``'.

Dropped from the site teacher's file (features not ported): the fall-aware site promotion table, rule ``loc``, the
crash flag, ``site_impact`` as a failure, the T3 blocker per-level counts, the blocker step counters, the
``site_task_knobs`` knobs (``KRABBY_SITE_PROMOTE_RULE``, ``KRABBY_MAX_INIT_TERRAIN_LEVEL``) and the
``replay_site_rewards.py`` tests.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import math
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
OMNI = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
OMNI_MDP = OMNI / "mdp"
PARKOURS_DIR = REPO / "parkour/parkour_isaaclab/envs/mdp/parkours"
OMNI_EVENT_PY = OMNI_MDP / "omni_parkour_event.py"
#: origin/main before the port (the rod-speed DR merge); ``omni_parkour_event.py`` there is unchanged since ab390e6f.
HEAD = "3f944c4b"


def _load_real_module(dotted: str, path: Path) -> types.ModuleType:
    """Get-or-create loader of ``test_omni_p2b_knobs.py`` (real source dirs as stub packages' ``__path__``)."""
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
_load_real_module("parkour_isaaclab.envs.mdp.parkours.exposure_stats", PARKOURS_DIR / "exposure_stats.py")
ox = _load_real_module(f"{_PKG}.omni_exposure", OMNI_MDP / "omni_exposure.py")
sr = _load_real_module(f"{_PKG}.site_raster", OMNI_MDP / "site_raster.py")
omni_math = _load_real_module(f"{_PKG}.crab_hex_omni_command_math", OMNI_MDP / "crab_hex_omni_command_math.py")
tk = _load_real_module(f"{_PKG}.omni_terrain_knobs", OMNI_MDP / "omni_terrain_knobs.py")
N_SECT = 8
K = {name: i for i, name in enumerate(sr.SITE_KINDS)}


# ============================================================================ the pure helpers
def test_site_ledger_is_lazy_and_per_kind():
    led = ox.SiteKindLedger(sr.SITE_KINDS, sr.SITE_KIND_POLICY, capacity=4)
    assert led.means() == {}
    keys = set(led.keys())
    assert "site_crossed_frac/pits" in keys and "site_crossed_frac/ledge_down" in keys
    assert "site_crossed_frac/chests" not in keys and "site_crossed_frac/flat" not in keys
    assert not any(k.startswith("site_crash_frac/") for k in keys)          # no crash flag in the port
    assert {k.split("/")[0] for k in keys} == set(ox.SITE_LEDGER_STEMS)
    tkind = torch.tensor([K["pits"], K["pits"], K["walls"], -1])
    led.push(tkind, crossed=torch.tensor([True, False, False, True]), failed=torch.tensor([False, True, False, True]),
             ratio=torch.tensor([0.9, 0.2, 0.0, 0.5]), ratio_valid=torch.tensor([True, True, True, True]))
    m = led.means()
    assert m["site_crossed_frac/pits"] == 0.5 and m["site_fail_frac/pits"] == 0.5
    assert m["site_track_p50/pits"] == pytest.approx(0.55)
    assert m["site_fail_frac/walls"] == 0.0
    assert not any(k.endswith("/flat") for k in m)                    # no flat episode yet; -1 skipped
    # the p50 rule: invalid and non-finite ratios are left out; the window is per kind
    led.push(torch.tensor([K["pits"]] * 3), torch.zeros(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool),
             torch.tensor([float("nan"), 0.1, 0.3]), torch.tensor([True, False, True]))
    m = led.means()
    assert m["site_track_p50/pits"] == pytest.approx(0.3)             # window {0.9, 0.2, 0.3} after the nan drop
    assert m["site_crossed_frac/pits"] == 0.0                         # window of 4 over T F | F F F: the T rolled out


def test_site_level_and_count_extras():
    levels = torch.tensor([0, 2, 4, 9])
    tile_kind = torch.tensor([K["pits"], K["pits"], K["walls"], -1])
    ex = ox.site_level_extras(levels, tile_kind, sr.SITE_KINDS)
    assert ex == {"site_level/pits": 1.0, "site_level/walls": 4.0}
    c = ox.site_count_extras({"pits": 2}, {"walls": 1}, ("pits", "walls"))
    assert c == {"site_promote/pits": 2.0, "site_demote/pits": 0.0, "site_promote/walls": 0.0,
                 "site_demote/walls": 1.0}


def test_exposure_all_is_complete():
    assert set(ox.__all__) <= set(dir(ox))


def test_tally_counts_and_clear():
    tally = ox.SiteCurriculumTally(tuple(sr.SITE_KINDS), ("pits", "flat"))
    assert tally.present == ("flat", "pits")                          # kind order
    assert tally.keys() == [f"{s}/{k}" for k in ("flat", "pits") for s in ox.SITE_COUNT_STEMS]
    f = torch.tensor
    tally.add(f([K["pits"], K["pits"], K["flat"], -1, K["walls"]]), crossed=f([True, False, False, True, True]),
              failed=f([False, True, False, False, False]), up=f([True, False, False, True, True]),
              down=f([False, True, False, False, False]))
    ex = tally.extras()
    assert [ex[f"{s}/pits"] for s in ox.SITE_COUNT_STEMS] == [2.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    assert [ex[f"{s}/flat"] for s in ox.SITE_COUNT_STEMS] == [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    assert set(ex) == set(tally.keys())                               # walls not in the grid, -1 skipped
    tally.clear()
    assert set(tally.extras().values()) == {0.0}


# ============================================================================ the parkour term
class _BaseParkourEvent:
    """``ParkourEvent`` / ``ParkourTerm`` stand-in: the hooks the overrides call up to, written as the real bases."""

    def _begin_episodes(self, env_ids) -> None:
        self.ep_steps[env_ids] = 0.0
        self.ep_field_steps[env_ids] = 0.0

    def _failure_flags(self, env_ids) -> torch.Tensor:   # ParkourEvent._failure_flags
        try:
            return self.env.termination_manager.get_term("crab_failure")[env_ids].float()
        except (AttributeError, KeyError, ValueError):
            return torch.zeros(len(env_ids), device=self.device)

    def _gather_cur_goals(self, future=0):
        return torch.zeros(self.num_envs, 3)

    def _obstacle_tile_mask(self) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.env_per_terrain_name).reshape(-1) != "parkour_flat")

    def _update_metrics(self):
        for key, val in self._exposure_means.items():
            self.metrics[key][:] = 0.0 if val != val else val

    def reset(self, env_ids=None):  # ParkourTerm.reset
        if env_ids is None:
            env_ids = slice(None)
        extras = {}
        for metric_name, metric_value in self.metrics.items():
            extras[metric_name] = torch.mean(metric_value).item()
            metric_value[env_ids] = 0.0
        if len(env_ids) != 0:
            self._resample_command(env_ids)
        return extras


class _Terms:
    """``termination_manager`` stand-in."""

    def __init__(self, terms: dict):
        self.terms = terms
        self.active_terms = list(terms)

    def get_term(self, name):
        return self.terms[name]


def _compile_class(src: str, label: str, extra_ns: dict):
    tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OmniParkourEvent")
    ns = {"torch": torch, "np": np, "ParkourEvent": _BaseParkourEvent, "Sequence": object,
          "SLOT_MOVE": omni_math.SLOT_MOVE, "wrap_to_pi": omni_math.wrap_to_pi,
          "is_field_generator": tk.is_field_generator}
    for name in (*ox.__all__, "tracking_ratio"):
        ns[name] = getattr(ox, name)
    ns.update(extra_ns)
    source = "from __future__ import annotations\n" + textwrap.dedent(
        " " * node.col_offset + ast.get_source_segment(src, node))
    exec(compile(source, label, "exec"), ns)  # noqa: S102
    return ns["OmniParkourEvent"]


def _site_terrains():
    """``site_terrains`` through the golden loader's inert package stubs (``test_site_terrain.py``'s loader)."""
    pytest.importorskip("trimesh", reason="the terrain mesh utilities (terrains/utils.py) import trimesh")
    fixtures = str(REPO / "tests/unit/fixtures")
    if fixtures not in sys.path:
        sys.path.insert(0, fixtures)
    import gen_omni_ring_golden as golden  # noqa: PLC0415

    golden.load_terrain_modules()
    return golden._load_real_module("parkour_isaaclab.terrains.omni.site_terrains",
                                    golden.LIB / "terrains" / "omni" / "site_terrains.py")


SITE_NS = {"_site_raster": sr, "_load_site_terrains": _site_terrains, "SITE_UNPORTED_CFG_ATTRS": ()}
OmniParkourEvent = _compile_class(OMNI_EVENT_PY.read_text(), str(OMNI_EVENT_PY), SITE_NS)

#: 10 levels x 4 columns: flat, pits, ledge_down, walls (the site generator's own sub-terrain names)
COLS = ("parkour_flat", "site_pits", "site_ledge_down", "site_walls")
N_LEVELS = 10
CROSS_R = {"site_pits": 4.1, "site_ledge_down": 3.2}


def _layers():
    frame = sr.TileFrame(size_xy=(10.0, 10.0), hs=0.08, vs=0.005, tile_px=(126, 126), grid_rc=(N_LEVELS, len(COLS)))
    shape = frame.global_shape
    tile_kind = np.array([[sr.SITE_SUBTERRAIN_KEYS.index(c) for c in COLS]] * N_LEVELS, np.int8)
    cross = np.array([[CROSS_R.get(c, math.nan) for c in COLS]] * N_LEVELS, np.float32)
    # d_site = level / 9 on the hazard columns (curriculum mode, full band); NaN on flat
    tile_d = np.array([[math.nan if c == "parkour_flat" else lv / 9 for c in COLS] for lv in range(N_LEVELS)],
                      np.float32)
    return sr.make_test_layers(np.zeros(shape, np.int8), np.zeros(shape, np.float32), np.zeros(shape, np.float32),
                               frame, tile_kind=tile_kind, tile_cross_r=cross, tile_d=tile_d)


def _subs(ladder=None):
    """The generator's ``sub_terrains`` as the metres telemetry reads them (``function`` per column key)."""
    st = _site_terrains()
    body = st.omni_site_terrain if ladder is None else st.SITE_LADDER_BODIES[ladder]
    return {c: types.SimpleNamespace(function=(print if c == "parkour_flat" else body)) for c in COLS}


def _term(levels, types_, *, track, cmd, r_max=None, cls=OmniParkourEvent, site=True, crab_failure=None,
          freeze=False, max_level=N_LEVELS, up=0.7, down=0.4, root_xy=None, subs=None, ep_steps=None):
    n = len(levels)
    ev = object.__new__(cls)
    ev.num_envs, ev.device, ev.simulation_time = n, "cpu", 0.1
    ev.clear_radius, ev.n_sectors, ev.min_cmd_dist = 1.7, N_SECT, 1.0
    ev.field_mode = True
    ev.cfg = types.SimpleNamespace(move_up_frac=up, move_down_frac=down, freeze_terrain_levels=freeze,
                                   num_future_goal_obs=2)
    ev.debug_vis = False
    names = np.array([[[c] for c in COLS]] * N_LEVELS)
    if site and subs is None:
        subs = _subs()
    ev.terrain = types.SimpleNamespace(
        terrain_levels=torch.tensor(levels, dtype=torch.long), terrain_types=torch.tensor(types_, dtype=torch.long),
        max_terrain_level=max_level, terrain_origins=torch.zeros(N_LEVELS, len(COLS), 3),
        cfg=types.SimpleNamespace(terrain_generator=types.SimpleNamespace(sub_terrains=subs)),
    )
    ev.terrain_class = torch.zeros(N_LEVELS, len(COLS))
    ev.terrain_goals = torch.zeros(N_LEVELS, len(COLS), 3, 3)
    ev.total_terrain_names = names
    ev.env_per_terrain_name = names[np.asarray(levels), np.asarray(types_)]
    ev.is_obst_tile = ev._obstacle_tile_mask()
    ev.env_origins = torch.zeros(n, 3)
    ev.env_class = torch.zeros(n)
    ev.env_goals = torch.zeros(n, 5, 3)
    ev.spawn_xy_w = torch.zeros(n, 2)
    ev.dis_to_start_pos = torch.zeros(n)
    ev.reach_goal_timer = torch.zeros(n)
    ev.cur_goal_idx = torch.zeros(n, dtype=torch.long)
    root = torch.zeros(n, 3)
    if root_xy is not None:
        root[:, :2] = torch.tensor(root_xy, dtype=torch.float32)
    ev.robot = types.SimpleNamespace(data=types.SimpleNamespace(root_pos_w=root))
    ev._cmd_term_cache = types.SimpleNamespace(travel_dir_w=torch.zeros(n), heading_target=torch.zeros(n))
    for name in ("ep_field_steps", "ep_block_steps", "ep_track_dist_move", "ep_cmd_dist_move",
                 "ep_head_err_sq", "ep_off_tile_steps", "ep_off_tile_block_steps"):
        setattr(ev, name, torch.zeros(n))
    ev.ep_r_max = torch.zeros(n) if r_max is None else torch.tensor(r_max, dtype=torch.float32)
    ev.ep_steps = torch.full((n,), 10.0) if ep_steps is None else torch.tensor(ep_steps, dtype=torch.float32)
    ev.ep_track_dist = torch.tensor(track, dtype=torch.float32)
    ev.ep_cmd_dist = torch.tensor(cmd, dtype=torch.float32)
    ev.ep_sector_seen = torch.zeros(n, N_SECT, dtype=torch.bool)
    ev.spawn_kind = torch.zeros(n, dtype=torch.long)
    ev._env_arange = torch.arange(n)
    ev.exposure = ox.OmniExposureLedger(N_SECT, capacity=64, field_mode=True)
    ev._exposure_means = {}
    ev.metrics = {k: torch.zeros(n) for k in ("terrain_levels",) + tuple(
        ox.OmniExposureLedger.emit_keys(N_SECT, field_mode=True))}
    terms = {"crab_failure": torch.tensor(crab_failure if crab_failure is not None else [False] * n)}
    ev.env = types.SimpleNamespace(termination_manager=_Terms(terms), extras={})
    if site:
        setattr(ev.env, sr.SITE_LAYERS_ATTR, _layers())
        ev.site_mode = True
        ev._init_site_mode(capacity=64)
    return ev


F, P, L, W = 0, 1, 2, 3     # column index of flat, pits, ledge_down, walls
PREFIX = f"Metrics/{sr.TERRAIN_TERM_NAME}/"


def test_init_site_mode_state():
    ev = _term([0, 0], [P, W], track=[0, 0], cmd=[1, 1])
    assert ev._site_present == ("flat", "pits", "ledge_down", "walls")
    assert ev._site_tally is not None and ev._site_tally.present == ev._site_present
    assert ev._site_log_keys == frozenset(PREFIX + k for k in ev._site_tally.keys())
    assert set(ev._site_metres_kinds) == {"pits", "ledge_down"}
    for gone in ("ep_crash", "_site_promote", "_site_crash_n", "_site_promote_rule", "_site_blk_steps",
                 "_site_policy_lut", "_site_rew"):
        assert not hasattr(ev, gone), gone


def test_reset_logs_site_keys_before_tiles_move():
    # main's ratio rule (0.7 / 0.4): env 0 pits L3 tracks -> up; env 1 walls L5 stopped (ratio 0) -> down;
    # env 2 flat L0 ratio 0.1 -> down (floor); env 3 pits L1 ratio 0.5 -> hold
    ev = _term([3, 5, 0, 1], [P, W, F, P], track=[0.9, 0.0, 0.1, 0.5], cmd=[1, 1, 1, 1], r_max=[5.0, 0, 0, 0])
    ex = ev.reset(torch.arange(4))
    assert ex["site_level/pits"] == 2.0 and ex["site_level/walls"] == 5.0 and ex["site_level/flat"] == 0.0
    assert "site_level/ledge_down" not in ex                           # no env on that kind
    assert ex["site_promote/pits"] == 1.0 and ex["site_demote/walls"] == 1.0 and ex["site_promote/walls"] == 0.0
    assert ex["site_demote/flat"] == 1.0 and ex["site_promote/ledge_down"] == 0.0
    assert ev.terrain.terrain_levels.tolist() == [4, 4, 0, 1]
    # the window counts are per reset call
    ev.ep_steps[:] = 10.0
    ex2 = ev.reset(torch.arange(4))
    assert set(k for k in ex2 if k.startswith("site_promote/")) == {f"site_promote/{k}" for k in ev._site_present}


def test_ledger_keys_reach_the_metrics_lazily():
    ev = _term([3, 5], [P, W], track=[0.9, 0.0], cmd=[1, 1], r_max=[5.0, 0])
    assert not any(k.startswith("site_") for k in ev.metrics)
    ev.reset(torch.arange(2))
    assert "site_crossed_frac/pits" in ev.metrics and "site_fail_frac/walls" in ev.metrics
    assert "site_track_p50/pits" in ev.metrics and "site_crash_frac/walls" not in ev.metrics
    assert float(ev.metrics["site_crossed_frac/pits"][0]) == 1.0
    ev._update_metrics()
    ev.ep_steps[:] = 0.0
    ex = ev.reset(torch.arange(2))
    assert ex["site_track_p50/pits"] == pytest.approx(0.9)


def test_site_mode_promotes_by_mains_ratio_rule_only():
    """Falls and crossings never vote in the port: the levels are ``promotion_mask``'s (ratio > up promotes even after
    a fall, ratio < down demotes even after a crossing); the per-kind counts record exactly those applied masks."""
    # 0 pits fell before crossing, tracking -> up; 1 pits crossed, ratio 0.2 -> down; 2 ledge crossed, ratio 0.5 ->
    # hold; 3 walls fell, ratio 0.9 -> up; 4 flat inactive (cmd 0.5 < 1 m floor) -> hold
    ev = _term([4] * 5, [P, P, L, W, F], track=[0.9, 0.2, 0.5, 0.9, 0.0], cmd=[1, 1, 1, 1, 0.5],
               r_max=[3.0, 4.5, 3.5, 0, 0], crab_failure=[True, False, False, True, False])
    up, down = ox.promotion_mask(ev.ep_track_dist, ev.ep_cmd_dist, 0.7, 0.4, 1.0)
    ex = ev.reset(torch.arange(5))
    assert ev.terrain.terrain_levels.tolist() == [5, 3, 4, 5, 4]
    assert ev.terrain.terrain_levels.tolist() == (4 + up.long() - down.long()).tolist()
    assert [ex[f"{s}/pits"] for s in ox.SITE_COUNT_STEMS] == [2.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    assert [ex[f"{s}/ledge_down"] for s in ox.SITE_COUNT_STEMS] == [1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert [ex[f"{s}/walls"] for s in ox.SITE_COUNT_STEMS] == [1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    assert [ex[f"{s}/flat"] for s in ox.SITE_COUNT_STEMS] == [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    assert ex["site_promote/pits"] == 1.0 and ex["site_demote/pits"] == 1.0 and ex["site_promote/walls"] == 1.0


def test_freeze_and_loop_back():
    ev = _term([4, 4], [P, W], track=[0.9, 0.0], cmd=[1, 1], r_max=[5.0, 0], freeze=True)
    ex = ev.reset(torch.arange(2))
    assert ev.terrain.terrain_levels.tolist() == [4, 4]
    assert all(ex[f"site_promote/{k}"] == 0.0 and ex[f"site_demote/{k}"] == 0.0 for k in ev._site_present)
    assert ex["site_cnt_up/pits"] == 0.0 and ex["site_cnt_held/pits"] == 1.0
    torch.manual_seed(0)
    ev = _term([9] * 6, [P] * 6, track=[0.9] * 6, cmd=[1] * 6)
    ex = ev.reset(torch.arange(6))
    assert ex["site_promote/pits"] == 6.0
    assert all(0 <= lv < N_LEVELS for lv in ev.terrain.terrain_levels.tolist())


def test_init_refuses_the_unported_site_attributes():
    """``__init__`` (not runnable on a stand-in: it reaches the importer) refuses every unported cfg attribute; the
    module constant lists the site teacher's five."""
    src = OMNI_EVENT_PY.read_text()
    tree = ast.parse(src)
    const = next(n for n in tree.body if isinstance(n, ast.Assign)
                 and any(getattr(t, "id", None) == "SITE_UNPORTED_CFG_ATTRS" for t in n.targets))
    assert ast.literal_eval(const.value) == ("site_promote", "site_crash_n", "site_promote_rule", "site_slack_geom",
                                             "site_slack_halfw")
    init = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    seg = ast.get_source_segment(src, init)
    assert "unported = [a for a in SITE_UNPORTED_CFG_ATTRS if getattr(cfg, a, None) not in (None, False)]" in seg
    assert seg.index("if unported:") < seg.index("self.site_mode = _site_raster.site_layers(env) is not None")
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {
        n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    for gone in ("promotion_mask_site", "promotion_mask_site_rule", "_site_promotion", "site_rewards", "crash_now",
                 "_site_blocker_step", "SITE_IMPACT_TERM", "SiteBlockerSteps", "ep_crash"):
        assert gone not in names, gone


def test_counts_metres_on_the_columns_ladder():
    st = _site_terrains()
    ev = _term([4, 4, 2, 2, 7, 4], [P, P, L, W, W, F], track=[0.9, 0.9, 0.1, 0.0, 0.0, 0.1], cmd=[1] * 6,
               r_max=[4.5, 2.2, 1.0, 0, 0, 0], crab_failure=[False, True, False, False, True, False],
               subs=_subs("b4"))
    assert set(ev._site_metres_kinds) == {"pits", "ledge_down"}
    ex = ev.reset(torch.arange(6))
    assert [ex[f"{s}/pits"] for s in ox.SITE_COUNT_STEMS] == [2.0, 1.0, 1.0, 2.0, 0.0, 0.0]
    assert [ex[f"{s}/ledge_down"] for s in ox.SITE_COUNT_STEMS] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert [ex[f"{s}/walls"] for s in ox.SITE_COUNT_STEMS] == [2.0, 0.0, 1.0, 0.0, 2.0, 0.0]
    assert [ex[f"{s}/flat"] for s in ox.SITE_COUNT_STEMS] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert not any("_l" in k.split("/")[0][-3:] for k in ex if k.startswith("site_cnt"))   # no per-level stems
    # metres: each env's own tile (read before tiles move), on the ladder its column stamps (b4 here)
    d4, d2 = float(np.float32(4 / 9)), float(np.float32(2 / 9))
    assert ex["site_metres/pits"] == pytest.approx(st.primary_metres("pits", d4, "b4"), rel=1e-6)
    assert ex["site_metres/ledge_down"] == pytest.approx(st.primary_metres("ledge_down", d2, "b4"), rel=1e-6)
    assert st.primary_metres("pits", d4, "b4") != st.primary_metres("pits", d4, None)
    assert "site_metres/walls" not in ex and "site_metres/flat" not in ex
    # the next reset call starts from 0 (counts, not running totals), with the full stable tag set
    ev.ep_steps[:] = 10.0
    ex2 = ev.reset(torch.tensor([5]))
    assert ex2["site_cnt_end/flat"] == 1.0 and ex2["site_cnt_end/pits"] == 0.0
    assert set(k for k in ex2 if k.startswith("site_cnt")) == set(k for k in ex if k.startswith("site_cnt"))


def test_metres_on_todays_ladder():
    st = _site_terrains()
    ev = _term([3], [P], track=[0.0], cmd=[1.0])
    ex = ev.reset(torch.arange(1))
    assert ex["site_metres/pits"] == pytest.approx(st.primary_metres("pits", float(np.float32(3 / 9)), None),
                                                  rel=1e-6)


def test_counts_frozen_and_unrun_episodes():
    ev = _term([4, 4, 4], [P, P, W], track=[0.9, 0.9, 0.0], cmd=[1] * 3, r_max=[4.5, 4.5, 0], freeze=True,
               ep_steps=[10, 0, 10])
    ex = ev.reset(torch.arange(3))
    assert ev.terrain.terrain_levels.tolist() == [4, 4, 4]
    assert [ex[f"{s}/pits"] for s in ox.SITE_COUNT_STEMS] == [1.0, 1.0, 0.0, 0.0, 0.0, 1.0]   # the 0-step one: out
    assert [ex[f"{s}/walls"] for s in ox.SITE_COUNT_STEMS] == [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def _runner_means(ep_infos: list) -> dict:
    """``OnPolicyRunnerWithExtractor.log``'s ep_infos reduction, statement for statement (keys of the first dict,
    in-place tensor conversion, ``nanmean`` over every append)."""
    out = {}
    for key in ep_infos[0]:
        infotensor = torch.tensor([])
        for ep_info in ep_infos:
            if key not in ep_info:
                continue
            if not isinstance(ep_info[key], torch.Tensor):
                ep_info[key] = torch.Tensor([ep_info[key]])
            if len(ep_info[key].shape) == 0:
                ep_info[key] = ep_info[key].unsqueeze(0)
            infotensor = torch.cat((infotensor, ep_info[key]))
        out[key] = torch.nanmean(infotensor) if infotensor.numel() > 0 else torch.tensor(float("nan"))
    return out


def test_counts_equal_episode_ends_under_a_stale_dict_replay():
    """The runner appends ``extras["log"]`` every step and the env replaces it only on a reset step. With the zero fill
    the iteration's logged count x ``num_steps_per_env`` is the exact number of ends (applied up / down too); every
    other key logs exactly what it logs without the fill; without it the stale repeats over-count."""
    steps_per_it, n_it = 24, 4
    kinds_of = [P, L, W, F] * 3
    n = len(kinds_of)
    ev = _term([1] * n, kinds_of, track=[0.5] * n, cmd=[1] * n, root_xy=[[2.0, 0.0]] * n)
    gen = torch.Generator().manual_seed(7)
    seq, truth = [], []          # per step: the reset dict (or None); per iteration: the true counts
    log = {f"{PREFIX}{k}": v for k, v in ev.reset(torch.arange(n)).items()}   # the env's construction reset
    for _ in range(n_it):
        tally = {}
        for step in range(steps_per_it):
            ids = torch.nonzero(torch.rand(n, generator=gen) < 0.15).reshape(-1)
            if step % 6 == 5 or ids.numel() == 0:
                seq.append(None)
                continue
            ev.ep_steps[ids] = 10.0
            ev.ep_track_dist[ids] = torch.rand(ids.numel(), generator=gen)
            ev.env.termination_manager.terms["crab_failure"][ids] = torch.rand(ids.numel(), generator=gen) < 0.4
            torch.manual_seed(step)
            ex = ev.reset(ids)
            for i in ids.tolist():
                name = sr.SITE_KINDS[sr.SITE_SUBTERRAIN_KEYS.index(COLS[kinds_of[i]])]
                tally[name] = tally.get(name, 0) + 1
            d = {f"{PREFIX}{k}": v for k, v in ex.items()}
            d["Episode_Reward/other_manager"] = float(step)
            seq.append(d)
        truth.append(tally)

    def replay(fill: bool) -> list[dict]:
        env = types.SimpleNamespace(extras={"log": copy.deepcopy(log)})
        filler = types.SimpleNamespace(env=env, _site_log_keys=ev._site_log_keys, _site_log_zeroed=None)
        means, k = [], 0
        for _ in range(n_it):
            ep_infos = []
            for _s in range(steps_per_it):
                if fill:
                    OmniParkourEvent._site_log_zero_fill(filler)        # _update_command's first statement
                if seq[k] is not None:
                    env.extras["log"] = copy.deepcopy(seq[k])          # _reset_idx: a fresh dict
                ep_infos.append(env.extras["log"])
                k += 1
            means.append(_runner_means(ep_infos))
        return means

    filled, stale = replay(True), replay(False)
    over = 0
    for it in range(n_it):
        for name in ("pits", "ledge_down", "walls", "flat"):
            key = f"{PREFIX}site_cnt_end/{name}"
            got = ox.site_counts_from_logged(float(filled[it][key]), steps_per_it)
            assert got == pytest.approx(truth[it].get(name, 0), abs=1e-4), (it, name)
            parts = sum(ox.site_counts_from_logged(float(filled[it][f"{PREFIX}{s}/{name}"]), steps_per_it)
                        for s in ("site_cnt_up", "site_cnt_down", "site_cnt_held"))
            assert parts == pytest.approx(got, abs=1e-4)
            over += abs(float(stale[it][key]) * steps_per_it - truth[it].get(name, 0)) > 0.5
        for key, val in filled[it].items():
            if key not in ev._site_log_keys:
                assert torch.equal(val, stale[it][key]) or (val.isnan() and stale[it][key].isnan()), key
    assert over > 0                                                    # the stale re-count the fill removes


def test_zero_fill_runs_only_in_site_mode():
    src = OMNI_EVENT_PY.read_text()
    assert "if self._site_log_keys:\n            self._site_log_zero_fill()" in src
    ev = _term([1], [P], track=[0.0], cmd=[1.0], site=False)
    assert ev._site_log_keys == frozenset() and ev._site_tally is None


# ---------------------------------------------------------------------------- constants
def test_hazard_radii_are_site_terrains():
    st = _site_terrains()
    stm = ox._site_task_math()
    assert stm._HAZARD_CENTRE_R_M == {"trench": st.TRENCH_R_M, "straddle": st.STRADDLE_R_M}
    assert stm.HAZARD_INNER_R_M["pits"] == stm.HAZARD_INNER_R_M["rubble"] == stm.HAZARD_INNER_R_M["ramps"] == (
        st.BAND_R_IN_M)
    assert stm.HAZARD_INNER_R_M["ledge_down"] == stm.HAZARD_INNER_R_M["ledge_up"] == st.PAD_R_M
    assert stm.HAZARD_INNER_R_M["terraces"] == st.TERRACE_RISER_R_M[0]
    assert stm.HAZARD_INNER_R_M["snags"] == pytest.approx(st.SNAG_R_M[0] - 0.5 * st.SNAG_WIDTH_M, abs=1e-12)
    # R_cross = centre + w / 2 + 0.5 (site_terrains' trench / straddle layouts) -> inner radius centre - w / 2
    for kind, centre in (("trench", st.TRENCH_R_M), ("straddle", st.STRADDLE_R_M)):
        for w in (0.05, 0.15, 0.33, 0.45):
            assert stm.hazard_inner_radius(kind, centre + 0.5 * w + 0.5) == pytest.approx(centre - 0.5 * w,
                                                                                         abs=1e-12)
    # every CROSS / VOID kind has one; no other kind does
    for kind, pc in zip(sr.SITE_KINDS, sr.SITE_KIND_POLICY):
        has = math.isfinite(stm.hazard_inner_radius(kind, 3.0))
        assert has == (pc in (sr.POLICY_CROSS, sr.POLICY_VOID)), kind
    assert set(st.PRIMARY_PARAM) <= set(stm.HAZARD_INNER_R_M) | set(stm._HAZARD_CENTRE_R_M)


# ============================================================================ neutrality against main's class
def _head_source() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "show", f"{HEAD}:{OMNI_EVENT_PY.relative_to(REPO)}"],
                             capture_output=True, text=True, timeout=30, check=True)
    except Exception:  # noqa: BLE001 -- no git / shallow clone: skipped below
        return None
    return out.stdout


def _head_class():
    src = _head_source()
    if src is None:
        pytest.skip(f"git show {HEAD} unavailable")
    return _compile_class(src, f"{HEAD}:omni_parkour_event.py", {})


@pytest.mark.parametrize("freeze", [False, True])
def test_off_site_mode_is_heads_class(freeze):
    HeadEvent = _head_class()
    kw = dict(track=[0.9, 0.1, 0.5, 0.95, 0.0], cmd=[1, 1, 1, 1, 0.5], r_max=[5, 0, 0, 3, 0], freeze=freeze,
              crab_failure=[False, True, False, False, False], site=False)
    levels, types_ = [0, 3, 9, 2, 1], [P, W, F, L, P]
    head = _term(levels, types_, cls=HeadEvent, **kw)
    new = _term(levels, types_, **kw)
    assert new.site_mode is False and not hasattr(new, "_site_ledger")
    for _ in range(3):
        torch.manual_seed(1)
        a = head.reset(torch.arange(5))
        torch.manual_seed(1)
        b = new.reset(torch.arange(5))
        assert a == b
        assert torch.equal(head.terrain.terrain_levels, new.terrain.terrain_levels)
        assert set(head.metrics) == set(new.metrics)
        assert torch.equal(head._failure_flags(torch.arange(5)), new._failure_flags(torch.arange(5)))
        for ev in (head, new):
            ev.ep_steps[:] = 10.0
            ev.ep_track_dist[:] = torch.tensor([0.9, 0.1, 0.5, 0.95, 0.0])


@pytest.mark.parametrize("freeze", [False, True])
def test_site_mode_levels_and_mains_keys_equal_heads_class(freeze):
    """Site mode adds keys only: on the same episodes main's class (no site layers) and the site-mode class move the
    same levels (the loop-back draw included) and log the same value for every one of main's keys."""
    HeadEvent = _head_class()
    kw = dict(track=[0.9, 0.1, 0.5, 0.95, 0.0, 0.9], cmd=[1, 1, 1, 1, 0.5, 1], r_max=[5, 0, 0, 3, 0, 4.5],
              freeze=freeze, crab_failure=[False, True, False, False, False, True])
    levels, types_ = [0, 3, 9, 2, 1, 9], [P, W, F, L, P, P]
    head = _term(levels, types_, cls=HeadEvent, site=False, **kw)
    new = _term(levels, types_, **kw)
    assert new.site_mode is True
    for _ in range(3):
        torch.manual_seed(1)
        a = head.reset(torch.arange(6))
        torch.manual_seed(1)
        b = new.reset(torch.arange(6))
        assert set(a) <= set(b) and {k: b[k] for k in a} == a
        assert all(k.startswith("site_") for k in set(b) - set(a))
        assert torch.equal(head.terrain.terrain_levels, new.terrain.terrain_levels)
        assert set(head.metrics) <= set(new.metrics)
        for ev in (head, new):
            ev.ep_steps[:] = 10.0
            ev.ep_track_dist[:] = torch.tensor([0.9, 0.1, 0.5, 0.95, 0.0, 0.9])
