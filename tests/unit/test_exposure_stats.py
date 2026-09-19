"""PLAN H B0/B3: exposure ledger, goal-index convention, spawn-spread helpers (pure torch).

Goal convention pinned against the generator: parkour_gap/hurdle put goal 0 at
``platform_len - 1`` px, parkour_step at ``platform_len - 1 m`` (1.5 m tile-local, BEFORE the
platform edge), goals 1..6 at the obstacles, goal 7 at the final marker; the wrapper subtracts
0.5*size to make them centre-relative. So the platform edge comes from the config
(platform_edge_rel), a goal-INDEX test for "reached the first obstacle" uses index 1, and
goals_passed counts obstacle goals (index - 1) so passing marker 0 is never credited.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

MOD_DIR = Path(__file__).resolve().parents[2] / "parkour" / "parkour_isaaclab" / "envs" / "mdp" / "parkours"
if str(MOD_DIR) not in sys.path:
    sys.path.insert(0, str(MOD_DIR))

import exposure_stats as xs  # noqa: E402

HSCALE, VSCALE = 0.05, 0.005
SIZE = (16.0, 4.0)
PLATFORM_LEN = 2.5


def _generator_goal_x(n_obst=6, spacing_m=1.5):
    """Origin-relative goal x the way the generator builds it (metres, centre-relative)."""
    platform_px = round(PLATFORM_LEN / HSCALE)
    goals_px = [platform_px - 1]  # goal 0: platform edge marker (gap/hurdle convention)
    dis = platform_px
    for _ in range(n_obst):
        dis += round(spacing_m / HSCALE)
        goals_px.append(dis - round(spacing_m / HSCALE) // 2)
    goals_px.append(dis + round(spacing_m / HSCALE))
    goals_m = np.array(goals_px, dtype=float) * HSCALE
    # wrapper: goals -= 0.5 * cfg.size (size here = usable size; 1-px border -> 15.92)
    usable_x = (int(SIZE[0] / HSCALE) + 1 - 2) * HSCALE
    return torch.tensor(goals_m - 0.5 * usable_x, dtype=torch.float32)


class TestGoalConvention:
    def test_goal0_marker_and_config_edge(self):
        gx = _generator_goal_x().unsqueeze(0)
        usable_x = (int(SIZE[0] / HSCALE) + 1 - 2) * HSCALE
        local = gx[0] + 0.5 * usable_x
        assert local[0].item() == pytest.approx(PLATFORM_LEN - HSCALE, abs=1e-6)  # gap/hurdle marker 2.45 m
        assert xs.platform_edge_x(gx)[0].item() + 0.5 * usable_x == pytest.approx(2.50, abs=1e-6)
        # config-derived edge (used by the telemetry): platform_len from the field start + border px
        edge = xs.platform_edge_rel(PLATFORM_LEN, SIZE[0], HSCALE)
        assert edge + 0.5 * SIZE[0] == pytest.approx(PLATFORM_LEN + HSCALE, abs=1e-6)
        assert abs(edge - xs.platform_edge_x(gx)[0].item()) < 0.1  # the two frames agree within 10 cm
        assert local[1].item() > local[0].item()
        assert xs.num_obstacles(gx) == 6

    def test_step_terrain_marker_sits_before_the_edge(self):
        # parkour_step: goals[0] = platform_len - 1 m -> 1.5 m tile-local, 0.5 m after the spawn.
        # An edge taken from goal 0 would credit reach_edge on the platform; the config edge does not.
        step_marker_local = PLATFORM_LEN - 1.0
        edge_local = xs.platform_edge_rel(PLATFORM_LEN, SIZE[0], HSCALE) + 0.5 * SIZE[0]
        assert step_marker_local < 2.0 < edge_local

    def test_obstacle_goals_passed_ignores_marker_zero(self):
        assert xs.obstacle_goals_passed(torch.tensor([0, 1, 2, 7])).tolist() == [0, 0, 1, 6]

    def test_reach_obst_uses_index_one_not_zero(self):
        gx = _generator_goal_x().unsqueeze(0)
        edge = xs.platform_edge_rel(PLATFORM_LEN, SIZE[0], HSCALE)
        just_past_edge = torch.tensor([edge + 0.10])
        per = xs.episode_exposure(just_past_edge, gx, edge, torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long),
                                  torch.tensor([3.0]), torch.tensor([10.0]))
        assert per["reach_edge"].item() == 1.0
        assert per["reach_obst"].item() == 0.0
        at_first = gx[:, 1]
        per = xs.episode_exposure(at_first, gx, edge, torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long),
                                  torch.tensor([3.0]), torch.tensor([10.0]))
        assert per["reach_obst"].item() == 1.0
        assert per["obst_coverage_1"].item() == 1.0 and per["obst_coverage_2"].item() == 0.0

    def test_next_goal_index_for_spread_spawn(self):
        gx = _generator_goal_x().unsqueeze(0).repeat(3, 1)
        x = torch.stack([gx[0, 0] - 1.0, gx[0, 3] + 0.1, gx[0, -1] + 5.0])
        idx = xs.next_goal_index(gx, x)
        assert idx.tolist() == [0, 4, gx.shape[1] - 1]
        # margin pushes a spawn sitting just before a goal past it
        idx = xs.next_goal_index(gx, gx[:, 2] - 0.1, margin=0.3)
        assert idx[0].item() == 3


class TestEpisodeExposure:
    def test_hand_computed_fractions(self):
        gx = _generator_goal_x().unsqueeze(0).repeat(4, 1)
        edge = xs.platform_edge_rel(PLATFORM_LEN, SIZE[0], HSCALE)
        max_x = torch.stack([gx[0, 0] - 0.5, torch.tensor(edge + 0.2), gx[0, 3], gx[0, 6] + 0.2])
        spawn = torch.tensor([0, 0, 0, 0])
        end = torch.tensor([0, 1, 4, 7])  # marker 0 passed in ep 1; obstacles 1-3 in ep 2; 1-6 in ep 3
        fs = torch.tensor([0.0, 5.0, 50.0, 90.0])
        es = torch.tensor([100.0, 100.0, 100.0, 100.0])
        per = xs.episode_exposure(max_x, gx, edge, spawn, end, fs, es)
        assert per["reach_edge"].tolist() == [0.0, 1.0, 1.0, 1.0]
        assert per["reach_obst"].tolist() == [0.0, 0.0, 1.0, 1.0]
        assert per["goals_passed"].tolist() == [0.0, 0.0, 3.0, 6.0]
        assert per["field_frac"].tolist() == pytest.approx([0.0, 0.05, 0.5, 0.9])
        assert per["obst_coverage_3"].tolist() == [0.0, 0.0, 1.0, 1.0]
        assert per["obst_coverage_6"].tolist() == [0.0, 0.0, 0.0, 1.0]

    def test_goals_passed_credited_from_spawn_goal(self):
        gx = _generator_goal_x().unsqueeze(0)
        edge = xs.platform_edge_rel(PLATFORM_LEN, SIZE[0], HSCALE)
        per = xs.episode_exposure(gx[:, 5], gx, edge, torch.tensor([3]), torch.tensor([5]), torch.tensor([10.0]), torch.tensor([20.0]))
        assert per["goals_passed"].item() == 2.0

    def test_field_frac_is_horizon_independent(self):
        gx = _generator_goal_x().unsqueeze(0).repeat(2, 1)
        edge = xs.platform_edge_rel(PLATFORM_LEN, SIZE[0], HSCALE)
        per = xs.episode_exposure(gx[:, 2], gx, edge, torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.long),
                                  torch.tensor([50.0, 150.0]), torch.tensor([100.0, 300.0]))
        assert per["field_frac"][0].item() == per["field_frac"][1].item() == 0.5


class TestRingMean:
    def test_wraparound_keeps_last_capacity(self):
        r = xs.RingMean(4)
        assert math.isnan(r.mean())
        r.push(torch.tensor([1.0, 2.0, 3.0]))
        assert r.mean() == pytest.approx(2.0)
        r.push(torch.tensor([4.0, 5.0]))  # keeps 2,3,4,5
        assert r.mean() == pytest.approx(3.5)
        r.push(torch.arange(10, 20).float())  # more than capacity -> last 4
        assert r.mean() == pytest.approx(17.5)
        r.push(torch.empty(0))
        assert r.mean() == pytest.approx(17.5)


class TestLedgerGroupRules:
    def _batch(self):
        gx = _generator_goal_x().unsqueeze(0).repeat(6, 1)
        #            plat-obst  plat-obst  spread-obst  rsi-obst  plat-flat  rsi-flat
        kind = torch.tensor([0, 0, 1, 2, 0, 2])
        is_obst = torch.tensor([1, 1, 1, 1, 0, 0]).bool()
        max_x = torch.stack([gx[0, 0] - 0.5, gx[0, 2], gx[0, 6], gx[0, 4], gx[0, 6], gx[0, 6]])
        spawn = torch.tensor([0, 0, 5, 0, 0, 0])   # spread episode spawned with goal 5 ahead (obstacles 1-4 behind)
        end = torch.tensor([0, 3, 7, 5, 7, 7])      # obstacle goals passed: 0, 2, 6-4=2, 4, 6, 6
        fs = torch.tensor([0.0, 40.0, 100.0, 100.0, 90.0, 90.0])
        es = torch.tensor([100.0, 100.0, 100.0, 100.0, 100.0, 50.0])
        failed = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        edge = xs.platform_edge_rel(PLATFORM_LEN, SIZE[0], HSCALE)
        per = xs.episode_exposure(max_x, gx, edge, spawn, end, fs, es)
        return per, kind, is_obst, failed, es

    def test_reach_metrics_only_platform_spawned_obstacle_episodes(self):
        m = xs.summarise_exposure(*self._batch(), n_obst=6)
        assert m["reach_edge_frac"] == pytest.approx(0.5)   # episodes 0 (no), 1 (yes)
        assert m["reach_obst_frac"] == pytest.approx(0.5)
        assert m["reach_edge_frac_rsi"] == pytest.approx(1.0)  # episode 3 only

    def test_coverage_unconditional_including_spread(self):
        m = xs.summarise_exposure(*self._batch(), n_obst=6)
        # non-RSI obstacle episodes: 0, 1, 2 -> coverage_6 hit only by the spread episode
        assert m["obst_coverage_6"] == pytest.approx(1 / 3)
        assert m["obst_coverage_2"] == pytest.approx(2 / 3)
        assert m["obst_coverage_6_rsi"] == pytest.approx(0.0)

    def test_goals_passed_and_field_frac_over_non_rsi_obstacle_episodes(self):
        m = xs.summarise_exposure(*self._batch(), n_obst=6)
        assert m["goals_passed_mean"] == pytest.approx((0 + 2 + 2) / 3)  # spread credited from its spawn goal
        assert m["field_frac_mean"] == pytest.approx((0.0 + 0.4 + 1.0) / 3)
        assert m["field_steps_mean"] == pytest.approx((0 + 40 + 100) / 3)

    def test_failure_split_and_hazard(self):
        m = xs.summarise_exposure(*self._batch(), n_obst=6)
        assert m["crab_failure_obst"] == pytest.approx(3 / 4)
        assert m["crab_failure_flat"] == pytest.approx(1 / 2)
        assert m["crab_failure_obst_rsi"] == pytest.approx(1.0)
        assert m["crab_failure_obst_spread"] == pytest.approx(1.0)  # the one spread obstacle episode failed
        assert m["ep_steps_obst_spread"] == pytest.approx(100.0)
        # hazard = failures per 1000 steps: flat -> 0.5 / 75 steps * 1000
        assert m["crab_failure_hazard_flat"] == pytest.approx(0.5 / 75.0 * 1000.0)
        assert m["crab_failure_hazard_obst"] == pytest.approx(0.75 / 100.0 * 1000.0)

    def test_spawn_kind_shares(self):
        m = xs.summarise_exposure(*self._batch(), n_obst=6)
        assert m["rsi_frac_actual"] == pytest.approx(2 / 6)
        assert m["spread_frac_actual"] == pytest.approx(1 / 4)  # among non-RSI resets

    def test_empty_group_is_nan_and_emit_keys_complete(self):
        per, kind, is_obst, failed, es = self._batch()
        m = xs.summarise_exposure(per, kind, torch.zeros_like(is_obst), failed, es, n_obst=6)
        assert math.isnan(m["reach_obst_frac"])
        assert set(xs.ExposureLedger.emit_keys(6)) == set(m.keys())


class TestMotionProfile:
    def test_bins(self):
        t = torch.tensor([1.0, 2.0, 7.0, 8.0, 13.0, 19.0, 20.0])
        w = torch.tensor([1, 0, 1, 1, 0, 1, 1]).float()
        prof = xs.motion_profile(t, w, horizon_s=20.0, bin_s=6.0)
        assert len(prof) == 4
        assert prof == pytest.approx([0.5, 1.0, 0.0, 1.0])
        with pytest.raises(ValueError):
            xs.motion_profile(t, w, 0.0)


class TestSpawnHelpers:
    def test_world_to_pixel_matches_feet_edge_reward_formula(self):
        rows_offset, cols_offset = SIZE[0] * 10 / 2, SIZE[1] * 20 / 2
        x = torch.tensor([-80.0, 0.0, 12.34])
        y = torch.tensor([-40.0, 0.0, 3.21])
        ix, iy = xs.world_to_pixel(x, y, rows_offset, cols_offset, HSCALE, (3210, 1620))
        ref_x = ((x + rows_offset) / HSCALE).round().long().clamp(0, 3209)
        ref_y = ((y + cols_offset) / HSCALE).round().long().clamp(0, 1619)
        assert torch.equal(ix, ref_x) and torch.equal(iy, ref_y)

    def test_local_to_world_roundtrip_platform_spawn(self):
        origin_x = torch.tensor([100.0])
        # reset_root_state: x_w = origin - (size_y + offset) = origin - 7 -> tile-local 1.0
        assert xs.local_to_world_x(torch.tensor([1.0]), origin_x, SIZE[0]).item() == pytest.approx(93.0)

    def test_sample_spread_x_bounds_and_per_env_clamp(self):
        u = torch.tensor([0.0, 1.0, 0.5, 1.0])
        hi = torch.tensor([11.0, 11.0, 11.0, 0.5])  # last env clamped below lo
        x = xs.sample_spread_x(u, 1.0, hi)
        assert x.tolist() == pytest.approx([1.0, 11.0, 6.0, 1.0])

    def test_patch_is_flat_rejects_obstacle_faces_and_borders(self):
        hf = torch.zeros(200, 80, dtype=torch.int16)
        hf[100:110, :] = 12  # a 0.06 m step/hurdle band across the tile
        hf[:, :10] = -20  # side trench
        ix = torch.tensor([50, 95, 105, 3, 150])
        iy = torch.tensor([40, 40, 40, 40, 20])
        ok = xs.patch_is_flat(hf, ix, iy, rx_px=12, ry_px=24, tol=10.0)
        assert ok.tolist() == [True, False, False, False, False]
        # noise inside tolerance passes
        hf2 = torch.randint(-4, 5, (200, 80), dtype=torch.int16)
        assert xs.patch_is_flat(hf2, torch.tensor([100]), torch.tensor([40]), 12, 24, tol=10.0).item()

    def test_global_height_field_assembly(self):
        rows, cols, w, l = 2, 3, 4, 5
        hf = torch.arange(rows * cols * w * l).reshape(rows, cols, w, l)
        g = xs.global_height_field(hf)
        assert g.shape == (rows * w, cols * l)
        assert g[w + 1, 2 * l + 3].item() == hf[1, 2, 1, 3].item()
