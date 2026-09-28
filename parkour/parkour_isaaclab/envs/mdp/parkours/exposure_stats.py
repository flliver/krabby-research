# SPDX-License-Identifier: BSD-3-Clause
"""Pure-torch exposure bookkeeping for the obstacle-exposure campaign (PLAN H B0, 2026-09-03).

No Isaac imports -- everything here is unit-tested on CPU. ``ParkourEvent`` (training-time
telemetry) and ``training_timeline_probe.py`` (offline summariser) both go through
:class:`ExposureLedger`, so the two report identical numbers for identical episodes.

Conventions
-----------
* x is **origin-relative** (world x minus the env's tile-origin x). Tile-local x, the
  frame the plan quotes (spawn 1.0 m, platform edge 2.48 m), is ``x + 0.5 * size_x``.
* ``goal_x[:, 0]`` is the start marker, which is NOT the platform edge on every terrain
  (gap/hurdle: ``platform_len - 1`` px; step: ``platform_len - 1 m`` = 1.5 m tile-local,
  half a metre after the spawn; stones: ``platform_len - stone_len // 2``). The platform
  edge is therefore taken from the config -- :func:`platform_edge_rel` -- and passed in
  explicitly. ``goal_x[:, 1 .. G-2]`` are the obstacle goals (gap/hurdle place goal k at
  the midpoint *before* obstacle k, step and stones on top of it) and ``goal_x[:, G-1]``
  is the final marker. ``reach_obst`` (max x >= goal_x[:, 1]) reads "committed to the first
  obstacle", never "through it".
* ``goals_passed`` counts OBSTACLE goals only: ``cur_goal_idx = k`` means markers 0..k-1
  were passed, and marker 0 is never an obstacle, so obstacle goals passed =
  max(k - 1, 0), credited relative to the same count at spawn.
* ``spawn_kind``: 0 = platform (fixed offset), 1 = spread (``KRABBY_SPAWN_SPREAD``),
  2 = RSI. Episodes are always grouped by the tile class they ran on (obstacle vs flat).

Group rules (the plan's definitions, applied identically in training and in the probe):
* ``reach_edge_frac`` / ``reach_obst_frac``: obstacle-tile, **platform-spawned** episodes
  (spread spawns would satisfy both trivially; RSI is split out).
* ``field_frac_mean``, ``field_steps_mean``, ``goals_passed_mean``, ``obst_coverage_k``:
  obstacle-tile **non-RSI** episodes, spread spawns included (coverage is unconditional).
* ``*_rsi``: the same quantities over obstacle-tile RSI episodes.
* ``crab_failure_flat`` / ``crab_failure_obst``: failure share per tile class over all
  spawn kinds; ``crab_failure_hazard_*`` = failures per 1000 env steps (episode-length
  neutral, for arms that change the horizon).
"""
from __future__ import annotations

import math

import torch

SPAWN_PLATFORM = 0
SPAWN_SPREAD = 1
SPAWN_RSI = 2

EDGE_MARGIN_M = 0.05
"""Platform edge = goal-0 marker + one height-field pixel (goal 0 sits one px inside)."""

HAZARD_PER_STEPS = 1000.0


def platform_edge_rel(platform_len: float, size_x: float, horizontal_scale: float) -> float:
    """Origin-relative x of the platform edge from the terrain config.

    Terrain functions lay ``platform_len`` metres of platform from the inner field's first
    pixel; the field is padded by one border pixel and the mesh is centred on ``size_x``.
    """
    return float(platform_len) - 0.5 * float(size_x) + float(horizontal_scale)


def platform_edge_x(goal_x: torch.Tensor) -> torch.Tensor:
    """Fallback edge estimate from the goal table (valid for gap/hurdle only, where goal 0
    sits one pixel inside the edge). Prefer :func:`platform_edge_rel`."""
    return goal_x[:, 0] + EDGE_MARGIN_M


def obstacle_goals_passed(goal_idx: torch.Tensor) -> torch.Tensor:
    """Obstacle goals passed at goal index k: markers 0..k-1 passed, marker 0 is not an obstacle."""
    return (goal_idx - 1).clamp(min=0)


def num_obstacles(goal_x: torch.Tensor) -> int:
    """Obstacle goals per tile: G minus the edge marker and the final marker."""
    return int(goal_x.shape[1]) - 2


def next_goal_index(goal_x: torch.Tensor, x_rel: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """First goal index whose x lies ahead of ``x_rel`` (+margin), clamped to the last goal.

    Used for spread spawns: the goal-tracking machinery must not target a marker behind the
    robot, and ``goals_passed`` is credited from this index.
    """
    ahead = goal_x > (x_rel + margin).unsqueeze(1)
    idx = torch.where(ahead.any(dim=1), ahead.float().argmax(dim=1), torch.full_like(x_rel, goal_x.shape[1] - 1, dtype=torch.long))
    return idx.long()


def episode_exposure(
    max_x: torch.Tensor,
    goal_x: torch.Tensor,
    edge_x: torch.Tensor | float,
    spawn_goal_idx: torch.Tensor,
    end_goal_idx: torch.Tensor,
    field_steps: torch.Tensor,
    ep_steps: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Per-episode exposure quantities (each value shape (n,), float).

    Args:
        max_x: running max origin-relative x over the episode.
        goal_x: (n, G) origin-relative goal x table for the tile each episode ran on.
        edge_x: origin-relative platform edge (scalar or (n,)), see platform_edge_rel.
        spawn_goal_idx / end_goal_idx: goal index at spawn and at episode end.
        field_steps: env steps spent past the platform edge.
        ep_steps: episode length in env steps.
    """
    if goal_x.dim() != 2 or goal_x.shape[0] != max_x.shape[0]:
        raise ValueError("goal_x must be (n, G) aligned with max_x")
    edge = torch.as_tensor(edge_x, dtype=max_x.dtype, device=max_x.device)
    out = {
        "reach_edge": (max_x > edge).float(),
        "reach_obst": (max_x >= goal_x[:, 1]).float(),
        "goals_passed": (obstacle_goals_passed(end_goal_idx) - obstacle_goals_passed(spawn_goal_idx)).clamp(min=0).float(),
        "field_steps": field_steps.float(),
        "field_frac": field_steps.float() / ep_steps.clamp(min=1).float(),
    }
    for k in range(1, num_obstacles(goal_x) + 1):
        out[f"obst_coverage_{k}"] = (max_x >= goal_x[:, k]).float()
    return out


class RingMean:
    """Mean over the last ``capacity`` pushed scalars (episode-level ring buffer)."""

    def __init__(self, capacity: int, device: str | torch.device = "cpu"):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.buf = torch.zeros(self.capacity, device=device)
        self.n = 0
        self.head = 0

    def push(self, values: torch.Tensor) -> None:
        values = values.reshape(-1).to(self.buf.device, dtype=self.buf.dtype)
        m = values.numel()
        if m == 0:
            return
        if m >= self.capacity:
            self.buf[:] = values[-self.capacity:]
            self.n = self.capacity
            self.head = 0
            return
        idx = (self.head + torch.arange(m, device=self.buf.device)) % self.capacity
        self.buf[idx] = values
        self.head = int((self.head + m) % self.capacity)
        self.n = min(self.capacity, self.n + m)

    def mean(self) -> float:
        if self.n == 0:
            return math.nan
        return float(self.buf[: self.n].mean())


class ExposureLedger:
    """Episode-level ring statistics implementing the plan's group rules."""

    def __init__(self, n_obst: int, capacity: int = 1024, device: str | torch.device = "cpu"):
        self.n_obst = int(n_obst)
        self.device = device
        self._rings: dict[str, RingMean] = {}
        for key in self.ring_keys(self.n_obst):
            self._rings[key] = RingMean(capacity, device)

    @staticmethod
    def ring_keys(n_obst: int) -> list[str]:
        base = ["reach_edge_frac", "reach_obst_frac", "field_frac_mean", "field_steps_mean", "goals_passed_mean"]
        base += [f"obst_coverage_{k}" for k in range(1, n_obst + 1)]
        keys = list(base) + [f"{k}_rsi" for k in base]
        keys += [
            "crab_failure_flat", "crab_failure_obst", "crab_failure_obst_rsi", "crab_failure_obst_spread",
            "ep_steps_flat", "ep_steps_obst", "ep_steps_obst_spread",
            "spread_frac_actual", "rsi_frac_actual",
        ]
        return keys

    @staticmethod
    def emit_keys(n_obst: int) -> list[str]:
        """Keys reported by :meth:`means` (ring keys + hazard-normalised failure)."""
        return ExposureLedger.ring_keys(n_obst) + ["crab_failure_hazard_flat", "crab_failure_hazard_obst"]

    def push_episodes(
        self,
        per_ep: dict[str, torch.Tensor],
        spawn_kind: torch.Tensor,
        is_obst: torch.Tensor,
        failed: torch.Tensor,
        ep_steps: torch.Tensor,
    ) -> None:
        is_obst = is_obst.bool()
        rsi = spawn_kind == SPAWN_RSI
        plat = spawn_kind == SPAWN_PLATFORM
        obst_nonrsi = is_obst & ~rsi
        obst_plat = is_obst & plat
        obst_rsi = is_obst & rsi
        flat = ~is_obst
        failed = failed.float()
        ep_steps = ep_steps.float()

        def push(key: str, values: torch.Tensor, mask: torch.Tensor) -> None:
            self._rings[key].push(values[mask])

        for suffix, g_cond, g_all in (("", obst_plat, obst_nonrsi), ("_rsi", obst_rsi, obst_rsi)):
            push(f"reach_edge_frac{suffix}", per_ep["reach_edge"], g_cond)
            push(f"reach_obst_frac{suffix}", per_ep["reach_obst"], g_cond)
            push(f"field_frac_mean{suffix}", per_ep["field_frac"], g_all)
            push(f"field_steps_mean{suffix}", per_ep["field_steps"], g_all)
            push(f"goals_passed_mean{suffix}", per_ep["goals_passed"], g_all)
            for k in range(1, self.n_obst + 1):
                push(f"obst_coverage_{k}{suffix}", per_ep[f"obst_coverage_{k}"], g_all)
        push("crab_failure_flat", failed, flat)
        push("crab_failure_obst", failed, is_obst)
        push("crab_failure_obst_rsi", failed, obst_rsi)
        obst_spread = is_obst & (spawn_kind == SPAWN_SPREAD)
        push("crab_failure_obst_spread", failed, obst_spread)   # z-lookup sanity: spread spawns must not fall on arrival
        push("ep_steps_flat", ep_steps, flat)
        push("ep_steps_obst", ep_steps, is_obst)
        push("ep_steps_obst_spread", ep_steps, obst_spread)
        push("spread_frac_actual", (spawn_kind == SPAWN_SPREAD).float(), ~rsi)
        push("rsi_frac_actual", rsi.float(), torch.ones_like(rsi))

    def means(self) -> dict[str, float]:
        out = {k: r.mean() for k, r in self._rings.items()}
        for cls in ("flat", "obst"):
            f, s = out[f"crab_failure_{cls}"], out[f"ep_steps_{cls}"]
            out[f"crab_failure_hazard_{cls}"] = (f / s * HAZARD_PER_STEPS) if (s and s > 0 and not math.isnan(f)) else math.nan
        return out


def summarise_exposure(
    per_ep: dict[str, torch.Tensor],
    spawn_kind: torch.Tensor,
    is_obst: torch.Tensor,
    failed: torch.Tensor,
    ep_steps: torch.Tensor,
    n_obst: int,
) -> dict[str, float]:
    """One-shot summary over a batch of episodes (probe path; same rules as training)."""
    ledger = ExposureLedger(n_obst, capacity=max(1, int(spawn_kind.numel())))
    ledger.push_episodes(per_ep, spawn_kind, is_obst, failed, ep_steps)
    return ledger.means()


def motion_profile(t_s: torch.Tensor, walking: torch.Tensor, horizon_s: float, bin_s: float = 6.0) -> list[float]:
    """Walking-time fraction per episode-time bin [0, bin), [bin, 2bin), ..., [.., horizon]."""
    if horizon_s <= 0 or bin_s <= 0:
        raise ValueError("horizon_s and bin_s must be positive")
    edges = [i * bin_s for i in range(int(math.ceil(horizon_s / bin_s)) + 1)]
    edges[-1] = max(edges[-1], horizon_s)
    walking = walking.float()
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (t_s >= lo) & (t_s < hi) if hi < edges[-1] else (t_s >= lo) & (t_s <= hi)
        out.append(float(walking[m].mean()) if m.any() else math.nan)
    return out


# ---------------------------------------------------------------------------
# Height-field / spawn helpers (B3 spawn spread)
# ---------------------------------------------------------------------------

def global_height_field(height_fields: torch.Tensor) -> torch.Tensor:
    """(rows, cols, W, L) per-tile fields -> (rows*W, cols*L) global field.

    Same assembly as the feet-edge reward's ``x_edge_masks_tensor`` so the two index with
    the same world-to-pixel math.
    """
    r, c, w, l = height_fields.shape
    return height_fields.permute(0, 2, 1, 3).reshape(r * w, c * l)


def world_to_pixel(
    x_w: torch.Tensor, y_w: torch.Tensor, rows_offset: float, cols_offset: float, hscale: float, shape: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """World xy -> global height-field pixel (rewards.py feet-edge convention)."""
    ix = ((x_w + rows_offset) / hscale).round().long().clamp(0, shape[0] - 1)
    iy = ((y_w + cols_offset) / hscale).round().long().clamp(0, shape[1] - 1)
    return ix, iy


def local_to_world_x(x_local: torch.Tensor, origin_x: torch.Tensor, size_x: float) -> torch.Tensor:
    """Tile-local x (0 at the tile's -x border) -> world x for the given tile origins."""
    return origin_x + x_local - 0.5 * size_x


def sample_spread_x(u: torch.Tensor, lo: float, hi: torch.Tensor) -> torch.Tensor:
    """Uniform tile-local x in [lo, hi] with a per-env upper bound (clamped to >= lo)."""
    span = (hi - lo).clamp(min=0.0)
    return lo + u * span


def patch_is_flat(hf: torch.Tensor, ix: torch.Tensor, iy: torch.Tensor, rx_px: int, ry_px: int, tol: float) -> torch.Tensor:
    """True where the (2rx+1) x (2ry+1) window around each pixel has max-min <= tol.

    Windows that run off the field count as not flat (spawn rejected).
    """
    W, L = hf.shape
    out = torch.zeros(ix.shape[0], dtype=torch.bool, device=hf.device)
    for i in range(ix.shape[0]):
        x0, x1 = int(ix[i]) - rx_px, int(ix[i]) + rx_px + 1
        y0, y1 = int(iy[i]) - ry_px, int(iy[i]) + ry_px + 1
        if x0 < 0 or y0 < 0 or x1 > W or y1 > L:
            continue
        win = hf[x0:x1, y0:y1].float()
        out[i] = bool((win.max() - win.min()) <= tol)
    return out
