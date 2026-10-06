"""Omni phase 2 (2026-10-06): the obstacle-tile curriculum telemetry the CHARTER's gates read, on CPU.

The CHARTER 8 "terrain at +1000" kill and the CHARTER 5 calibration read four keys, logged as
``Metrics/base_parkour/<key>`` on ring AND field runs:

* ``terrain_levels_obst`` -- mean terrain level over the envs on obstacle tiles (``name != parkour_flat``);
* ``promote_count`` / ``demote_count`` -- applied promotion / demotion decisions on episodes that ENDED
  on an obstacle tile, in that reset call;
* ``track_ratio_obst_p50`` -- the median per-episode ``ep_track_dist / ep_cmd_dist`` (the ratio
  ``promotion_mask`` gates on, its ``min_cmd_dist`` exclusion applied) over the ledger window's
  obstacle-tile episodes; absent until the window holds one, never NaN.

Pinned here: the pure helpers, the ledger key (median, grouping, exclusions, NaN safety, absence),
the parkour term's reset / resample wiring (compiled out of ``omni_parkour_event.py`` with its free
names injected, the technique ``test_crab_hex_omni_exposure.py`` documents), that the ring's config
dump is unchanged (no cfg field added), and that the campaign readers use these exact tag names.
"""
from __future__ import annotations

import ast
import importlib.util
import math
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
PARKOURS_DIR = REPO / "parkour" / "parkour_isaaclab" / "envs" / "mdp" / "parkours"
OMNI = REPO / "parkour" / "parkour_tasks" / "parkour_tasks" / "crab_hex_omni_task"
OMNI_MDP = OMNI / "mdp"
P2_SCRIPTS = OMNI / "experiments" / "2026-10-07_omni_phase2" / "scripts"
OMNI_EVENT_PY = OMNI_MDP / "omni_parkour_event.py"
#: the B head's own c6x run: a REAL ring config dump written before this telemetry existed
C6X_DUMP = (REPO / "parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-10-06_02-24-21_seed18_X60_c6x_A15pB"
            / "params/env.yaml")


def _load_real_module(dotted: str, path: Path) -> types.ModuleType:
    """Same get-or-create loader as ``test_crab_hex_omni_exposure.py``."""
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

N_SECT = 8
P50 = ox.OMNI_OBST_RATIO_KEY


# ================================================================== pure helpers
def test_obstacle_level_mean_is_over_obstacle_envs_only():
    levels = torch.tensor([0, 2, 4, 9])
    obst = torch.tensor([False, True, True, False])
    assert ox.obstacle_level_mean(levels, obst) == pytest.approx(3.0)


def test_obstacle_level_mean_is_none_without_an_obstacle_env():
    assert ox.obstacle_level_mean(torch.tensor([1, 2]), torch.tensor([False, False])) is None


def test_promotion_counts_ignore_flat_tiles():
    up = torch.tensor([True, True, False, False, True])
    down = torch.tensor([False, False, True, True, False])
    obst = torch.tensor([True, False, True, False, True])
    assert ox.obstacle_promotion_counts(up, down, obst) == (2, 1)


def test_promotion_counts_reject_misaligned_masks():
    with pytest.raises(ValueError, match="shape mismatch"):
        ox.obstacle_promotion_counts(torch.zeros(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool),
                                     torch.zeros(2, dtype=torch.bool))


def test_window_extras_keys_and_skips():
    full = ox.terrain_window_extras(2.5, 3, 1)
    assert full == {"terrain_levels_obst": 2.5, "promote_count": 3.0, "demote_count": 1.0}
    assert set(full) == set(ox.OMNI_TERRAIN_WINDOW_KEYS)
    for bad in (None, float("nan"), float("inf")):
        out = ox.terrain_window_extras(bad, 0, 0)
        assert "terrain_levels_obst" not in out and out == {"promote_count": 0.0, "demote_count": 0.0}


# ================================================================== the ledger key
def _push(led, ratio, valid, obst):
    n = len(ratio)
    per_ep = ox.episode_exposure_omni(
        torch.zeros(n), torch.zeros(n, N_SECT, dtype=torch.bool), torch.zeros(n), torch.zeros(n),
        torch.full((n,), 10.0), 1.7,
    )
    led.push_episodes(
        per_ep, torch.zeros(n, dtype=torch.long), torch.tensor(obst), torch.zeros(n), torch.full((n,), 10.0),
        track_ratio=torch.tensor(ratio, dtype=torch.float32), track_ratio_valid=torch.tensor(valid),
        heading_err_rms=torch.zeros(n), block_steps=torch.zeros(n), r_max=torch.zeros(n),
    )


@pytest.mark.parametrize("field_mode", [False, True], ids=["ring", "field"])
def test_p50_is_the_median_of_valid_obstacle_tile_episodes(field_mode):
    led = ox.OmniExposureLedger(N_SECT, capacity=64, field_mode=field_mode)
    #           obst valid -> 0.2 0.9 0.5 0.4 ; flat 0.0 (excluded) ; invalid obst 0.01 (excluded)
    _push(led, [0.2, 0.9, 0.5, 0.4, 0.0, 0.01],
          [True, True, True, True, True, False],
          [True, True, True, True, False, True])
    m = led.means()
    assert m[P50] == pytest.approx(0.45)  # interpolated median of {0.2, 0.4, 0.5, 0.9}
    # the all-tile mean is NOT the key: it includes the flat episode
    assert m["track_ratio_mean"] == pytest.approx((0.2 + 0.9 + 0.5 + 0.4 + 0.0) / 5)


def test_p50_uses_the_promotion_rules_exclusion():
    """The validity the term passes is ``ep_cmd_dist >= min_cmd_dist`` -- the rows promotion_mask HOLDS
    are exactly the rows left out (an excluded near-zero would drag the median down)."""
    ev_track = torch.tensor([0.05, 4.0, 1.0])
    ev_cmd = torch.tensor([0.5, 5.0, 2.0])  # row 0 below min_cmd_dist 1.0
    up, down = ox.promotion_mask(ev_track, ev_cmd, 0.6, 0.3, 1.0)
    assert not up[0] and not down[0]
    led = ox.OmniExposureLedger(N_SECT, capacity=8)
    _push(led, ox.tracking_ratio(ev_track, ev_cmd).tolist(), (ev_cmd >= 1.0).tolist(), [True] * 3)
    assert led.means()[P50] == pytest.approx(0.65)  # median of 0.8 and 0.5


def test_p50_is_absent_until_an_obstacle_episode_and_never_nan():
    led = ox.OmniExposureLedger(N_SECT, capacity=8)
    assert P50 not in led.means()
    _push(led, [0.7, 0.7], [True, True], [False, False])  # flat only
    assert P50 not in led.means()
    _push(led, [0.7], [False], [True])  # obstacle but held (invalid)
    assert P50 not in led.means()
    _push(led, [float("nan"), float("inf")], [True, True], [True, True])  # a blown-up sim
    assert P50 not in led.means()
    _push(led, [float("nan"), 0.3], [True, True], [True, True])
    assert led.means()[P50] == pytest.approx(0.3)


def test_p50_rolls_over_the_ledger_window():
    led = ox.OmniExposureLedger(N_SECT, capacity=4)
    _push(led, [0.1] * 4, [True] * 4, [True] * 4)
    _push(led, [0.9] * 3, [True] * 3, [True] * 3)
    assert led.means()[P50] == pytest.approx(0.9)  # window = {0.1, 0.9, 0.9, 0.9}


def test_p50_is_lazy_not_an_emit_key():
    for fm in (False, True):
        assert P50 not in ox.OmniExposureLedger.emit_keys(N_SECT, field_mode=fm)
    assert P50 in ox.OmniExposureLedger.lazy_keys
    assert set(ox.OMNI_SPAWN_SPLIT_KEYS) <= set(ox.OmniExposureLedger.lazy_keys)


# ================================================================== the term
class _BaseParkourEvent:
    """Stand-in for ``ParkourEvent`` / ``ParkourTerm``: the hooks the overrides under test call up to,
    with ``reset`` and ``_update_metrics`` written exactly as the real bases do them."""

    def _begin_episodes(self, env_ids) -> None:
        self.ep_steps[env_ids] = 0.0
        self.ep_field_steps[env_ids] = 0.0

    def _failure_flags(self, env_ids) -> torch.Tensor:
        return torch.zeros(len(env_ids))

    def _gather_cur_goals(self, future=0):
        return torch.zeros(self.num_envs, 3)

    def _obstacle_tile_mask(self) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.env_per_terrain_name).reshape(-1) != "parkour_flat")

    def _update_metrics(self):  # ParkourEvent._update_metrics, exposure part
        for key, val in self._exposure_means.items():
            self.metrics[key][:] = 0.0 if val != val else val

    def reset(self, env_ids=None):  # ParkourTerm.reset, verbatim logic
        if env_ids is None:
            env_ids = slice(None)
        extras = {}
        for metric_name, metric_value in self.metrics.items():
            extras[metric_name] = torch.mean(metric_value).item()
            metric_value[env_ids] = 0.0
        if len(env_ids) != 0:
            self._resample_command(env_ids)
        return extras


def _load_event_class():
    src = OMNI_EVENT_PY.read_text()
    tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OmniParkourEvent")
    ns = {"torch": torch, "np": np, "ParkourEvent": _BaseParkourEvent, "Sequence": object,
          "SLOT_MOVE": omni_math.SLOT_MOVE, "wrap_to_pi": omni_math.wrap_to_pi,
          "is_field_generator": tk.is_field_generator}
    for name in (*ox.__all__, "tracking_ratio"):  # tracking_ratio is public but not in __all__
        ns[name] = getattr(ox, name)
    source = "from __future__ import annotations\n" + textwrap.dedent(
        " " * node.col_offset + ast.get_source_segment(src, node)
    )
    exec(compile(source, str(OMNI_EVENT_PY), "exec"), ns)  # noqa: S102
    return ns["OmniParkourEvent"]


OmniParkourEvent = _load_event_class()

#: 3 levels x 2 types; type 0 is flat, type 1 an obstacle course -- except level 1 of type 1, flat, so a
#: promotion can move an env OFF an obstacle tile (the count must follow the tile the episode ended on)
NAMES = np.array([["parkour_flat", "omni_field"], ["parkour_flat", "parkour_flat"], ["parkour_flat", "omni_field"]])


def _term(levels, types_, *, track, cmd, freeze=False, field_mode=False, up=0.6, down=0.3, max_level=3):
    n = len(levels)
    ev = object.__new__(OmniParkourEvent)
    ev.num_envs, ev.device, ev.simulation_time = n, "cpu", 0.1
    ev.clear_radius, ev.n_sectors, ev.min_cmd_dist = 1.7, N_SECT, 1.0
    ev.field_mode = field_mode
    ev.cfg = types.SimpleNamespace(move_up_frac=up, move_down_frac=down, freeze_terrain_levels=freeze,
                                   num_future_goal_obs=2)
    ev.debug_vis = False
    ev.terrain = types.SimpleNamespace(
        terrain_levels=torch.tensor(levels, dtype=torch.long), terrain_types=torch.tensor(types_, dtype=torch.long),
        max_terrain_level=max_level, terrain_origins=torch.zeros(3, 2, 3),
    )
    ev.terrain_class = torch.zeros(3, 2)
    ev.terrain_goals = torch.zeros(3, 2, 3, 3)
    ev.total_terrain_names = NAMES
    ev.env_per_terrain_name = NAMES[np.asarray(levels), np.asarray(types_)]
    ev.is_obst_tile = ev._obstacle_tile_mask()
    ev.env_origins = torch.zeros(n, 3)
    ev.env_class = torch.zeros(n)
    ev.env_goals = torch.zeros(n, 5, 3)
    ev.spawn_xy_w = torch.zeros(n, 2)
    ev.dis_to_start_pos = torch.zeros(n)
    ev.reach_goal_timer = torch.zeros(n)
    ev.cur_goal_idx = torch.zeros(n, dtype=torch.long)
    ev.robot = types.SimpleNamespace(data=types.SimpleNamespace(root_pos_w=torch.zeros(n, 3)))
    ev._cmd_term_cache = types.SimpleNamespace(travel_dir_w=torch.zeros(n), heading_target=torch.zeros(n))
    for name in ("ep_r_max", "ep_field_steps", "ep_block_steps", "ep_track_dist_move", "ep_cmd_dist_move",
                 "ep_head_err_sq", "ep_off_tile_steps", "ep_off_tile_block_steps"):
        setattr(ev, name, torch.zeros(n))
    ev.ep_steps = torch.full((n,), 10.0)
    ev.ep_track_dist = torch.tensor(track, dtype=torch.float32)
    ev.ep_cmd_dist = torch.tensor(cmd, dtype=torch.float32)
    ev.ep_sector_seen = torch.zeros(n, N_SECT, dtype=torch.bool)
    ev.spawn_kind = torch.zeros(n, dtype=torch.long)
    ev._env_arange = torch.arange(n)
    ev.exposure = ox.OmniExposureLedger(N_SECT, capacity=64, field_mode=field_mode)
    ev._exposure_means = {}
    ev.metrics = {k: torch.zeros(n) for k in ("terrain_levels",) + tuple(
        ox.OmniExposureLedger.emit_keys(N_SECT, field_mode=field_mode))}
    return ev


@pytest.mark.parametrize("field_mode", [False, True], ids=["ring", "field"])
def test_reset_logs_the_obstacle_keys_of_that_reset_call(field_mode):
    # env 0 obstacle promotes (0.9), env 1 obstacle demotes (0.1), env 2 flat promotes, env 3 flat demotes,
    # env 4 obstacle held (cmd below min_cmd_dist)
    ev = _term([0, 1, 0, 1, 0], [1, 1, 0, 0, 1], track=[0.9, 0.1, 0.9, 0.1, 0.0], cmd=[1, 1, 1, 1, 0.5],
               field_mode=field_mode)
    # env 1 sits on (1, 1), which NAMES makes flat: obstacle envs are 0 and 4, both at level 0
    assert ev.is_obst_tile.tolist() == [True, False, False, False, True]
    ex = ev.reset(torch.arange(5))
    assert ex["terrain_levels_obst"] == pytest.approx(0.0)  # read BEFORE any tile moved
    assert ex["promote_count"] == 1.0  # env 0 only; env 2 promoted on a flat tile
    assert ex["demote_count"] == 0.0  # env 1 / env 3 demoted on flat tiles; env 4 held
    # the counts belong to THIS call and are cleared: a quiet second reset reports zero
    ev.ep_track_dist[:] = 0.5
    ev.ep_cmd_dist[:] = 1.0
    ev.ep_steps[:] = 10.0
    ex2 = ev.reset(torch.arange(5))
    assert (ex2["promote_count"], ex2["demote_count"]) == (0.0, 0.0)


def test_counts_follow_the_tile_the_episode_ended_on():
    """Env 0 ends on (0, 1) -- an obstacle tile -- and is promoted to (1, 1), which is flat: counted."""
    ev = _term([0], [1], track=[0.9], cmd=[1.0])
    ex = ev.reset(torch.tensor([0]))
    assert ex["promote_count"] == 1.0
    assert ev.terrain.terrain_levels.tolist() == [1] and ev.is_obst_tile.tolist() == [False]
    # ... and the next reset's level key is skipped, not 0.0: no env is on an obstacle tile any more
    ev.ep_steps[:] = 10.0
    assert "terrain_levels_obst" not in ev.reset(torch.tensor([0]))


def test_frozen_levels_report_no_promotions():
    ev = _term([0, 0], [1, 1], track=[0.9, 0.1], cmd=[1.0, 1.0], freeze=True)
    ex = ev.reset(torch.arange(2))
    assert (ex["promote_count"], ex["demote_count"]) == (0.0, 0.0)
    assert ev.terrain.terrain_levels.tolist() == [0, 0]


def test_level_mean_averages_obstacle_envs_only():
    """Obstacle envs at levels 0 and 2 -> 1.0; the all-tile mean (with the flat env at 0) would be 0.667."""
    ev = _term([0, 0, 2], [0, 1, 1], track=[0.5] * 3, cmd=[1.0] * 3)
    assert ev.reset(torch.arange(3))["terrain_levels_obst"] == pytest.approx(1.0)


def test_p50_reaches_the_metrics_path_lazily():
    """The median rides the existing ledger -> ``self.metrics`` -> reset path: no metric before an
    obstacle-tile episode ended, then the broadcast value one reset log later (like every ledger key)."""
    ev = _term([0, 0, 1], [1, 1, 0], track=[0.2, 0.8, 0.9], cmd=[1.0, 1.0, 1.0])
    ex = ev.reset(torch.arange(3))
    assert P50 not in ex  # finalised during this call's resample, after the log
    assert P50 in ev.metrics
    ev._update_metrics()
    ev.ep_steps[:] = 0.0  # nothing ends in the next reset
    ex2 = ev.reset(torch.arange(3))
    assert ex2[P50] == pytest.approx(0.5)  # median of the two obstacle-tile episodes; the flat 0.9 is out


# ================================================================== ring config dump unchanged
OMNI_CFG_FIELDS = {  # OmniParkourEventsCfg's own fields and defaults, as the ring dumps carry them
    "clear_radius_m": 2.0, "ring_outer_radius_m": 4.5, "n_sectors": 8, "min_cmd_dist_m": 1.0,
    "block_height_tol_m": 0.03, "foot_body_names": (".*_Footpad",), "command_name": "base_velocity",
}


def _cfg_fields(path: Path, cls: str) -> dict:
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    out = {}
    for st in node.body:
        if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
            try:
                out[st.target.id] = ast.literal_eval(st.value) if st.value is not None else None
            except ValueError:
                out[st.target.id] = "<expr>"
    return out


def test_the_cfg_gains_no_field():
    """Telemetry only: the term's cfg -- the only part of it a config dump records -- is unchanged."""
    fields = _cfg_fields(OMNI_EVENT_PY, "OmniParkourEventsCfg")
    assert set(fields) == set(OMNI_CFG_FIELDS) | {"class_type"}
    for k, v in OMNI_CFG_FIELDS.items():
        assert fields[k] == v, k


@pytest.mark.skipif(not C6X_DUMP.is_file(), reason="the B head's c6x run dump is not on this machine")
def test_a_real_ring_dump_is_reproduced_key_for_key():
    import yaml

    class _L(yaml.SafeLoader):
        pass

    _L.add_constructor("tag:yaml.org,2002:python/tuple", lambda ld, n: tuple(ld.construct_sequence(n)))
    _L.add_multi_constructor("", lambda ld, s, n: None)
    bp = yaml.load(C6X_DUMP.read_text(), Loader=_L)["parkours"]["base_parkour"]  # noqa: S506
    base = set(_cfg_fields(PARKOURS_DIR / "parkour_events_cfg.py", "ParkourEventsCfg")) | set(
        _cfg_fields(REPO / "parkour/parkour_isaaclab/managers/parkour_manager_term_cfg.py", "ParkourTermCfg"))
    own = set(_cfg_fields(OMNI_EVENT_PY, "OmniParkourEventsCfg"))
    assert set(bp) == base | own
    for k, v in OMNI_CFG_FIELDS.items():
        assert bp[k] == v, k


# ================================================================== the campaign readers
def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def p2_live():
    return _load(P2_SCRIPTS / "p2_live.py", "_t_p2_live")


@pytest.fixture(scope="module")
def p2_read():
    return _load(P2_SCRIPTS / "p2_read.py", "_t_p2_read")


def test_live_reads_use_the_emitted_names(p2_live):
    pre = p2_live.METRIC_PREFIX
    assert pre == "Metrics/base_parkour/"
    assert p2_live.OBST_LEVEL_TAGS == (pre + "terrain_levels_obst",)
    assert p2_live.PROMO_TAGS == (pre + "promote_count",)
    for tag in p2_live.OBST_LEVEL_TAGS + p2_live.PROMO_TAGS:
        assert tag[len(pre):] in ox.OMNI_TERRAIN_WINDOW_KEYS


def test_preflight_finds_both_keys_in_the_tree(p2_live):
    em = p2_live.emitted_metric_keys(REPO, ["terrain_levels_obst", "promote_count"])
    assert all(em.values()), em
    assert p2_live.judge_preflight("A1", em, None, None)["rc"] == p2_live.RC_PASS


def test_calibrate_reads_the_emitted_median(p2_read):
    assert p2_read.CAL_TAG == "Metrics/base_parkour/" + ox.OMNI_OBST_RATIO_KEY
    assert p2_read.CAL_TAIL == 100


def test_calibrate_run_on_a_real_tfevents_file(p2_read, tmp_path):
    tb = pytest.importorskip("torch.utils.tensorboard")
    w = tb.SummaryWriter(str(tmp_path))
    for i in range(300):  # 200 at 0.6, then the last 100 at 0.8, plus an unrelated tag
        w.add_scalar(p2_read.CAL_TAG, 0.6 if i < 200 else 0.8, 14996 + i)
        w.add_scalar("Metrics/base_parkour/track_ratio_mean", 0.1, 14996 + i)
    w.close()
    r = p2_read.calibrate_run(tmp_path)
    assert r["ok"] and r["promote"] == "0.7:0.4" and math.isclose(r["r_bar"], 0.8, rel_tol=1e-6)
    assert (r["first_step"], r["last_step"]) == (15196, 15295)
