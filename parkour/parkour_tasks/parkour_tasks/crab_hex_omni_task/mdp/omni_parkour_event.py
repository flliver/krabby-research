# SPDX-License-Identifier: BSD-3-Clause
"""The omni parkour term: ring spawn, ring exposure, ratio promotion (plan D9, spec B5).

``ParkourEvent`` (``$LIB/envs/mdp/parkours/parkour_event.py``) is built around a CORRIDOR:
the robot spawns on a platform at ``origin - (size_y + offset, 0)``, walks ``+x`` through a
line of goals, and the term (a) re-targets the current goal every step and advances
``cur_goal_idx`` when the robot reaches one, (b) measures exposure along ``x``, and
(c) promotes the tile's terrain level when the distance from the platform start exceeds a
fraction of ``cmd_vx * episode_length_s``.

None of the three survives an omnidirectional task, and each fails QUIETLY:

* the goals are ring sector anchors that exist only to feed the terrain machinery; a robot
  commanded to walk west would "reach" the east anchors by accident and the goal index would
  march on (and, with the stock ``terminate_episode``, end the episode);
* ``max x`` is meaningless when travel is uniform over ``+-pi`` -- half the population would
  report zero exposure while walking hard;
* ``cmd_vx * T`` is <= 0 for every lateral and backward command, so the promotion threshold
  is <= 0 (promote always) and the demotion threshold <= 0 (demote never), i.e. the terrain
  curriculum would run open-loop.

:class:`OmniParkourEvent` overrides exactly the methods those three concerns live in and
inherits everything else (the goal tables, the level/origin bookkeeping shape, the metrics
plumbing). The debug vis is a fourth, PLAY-only override with the same shape of reason --
two of the corridor's four visualisers assert a waypoint order the ring does not have -- and
is documented marker by marker at the bottom of the class. What it keeps identical on
purpose: the ledger KEY names the campaign
driver greps, the reset-time ordering (finalize the old episode -> move the tile -> begin the
new one), and the ``reset_root_state.params['offset']`` read in ``ParkourEvent.__init__``
(so the omni reset event must keep an ``offset`` param).

Frames, stated once because mixing them up is silent:

* ``env_origins[:, :2]`` is the TILE CENTRE in world xy (``parkour_field_to_mesh`` returns
  ``origin = (0.5*size_x, 0.5*size_y, z)`` and the generator translates the tile so that
  lands at ``((row+0.5)*size_x, (col+0.5)*size_y)`` before re-centring). The clear disc, the
  ring annulus and the sector anchors are all concentric with it, so every radial quantity
  here is measured from ``env_origins``.
* ``spawn_xy_w`` is where the robot ACTUALLY started (tile centre plus the reset event's
  radial jitter). ``dis_to_start_pos`` -- the promotion-era telemetry -- is measured from
  it, exactly as the corridor measures from its platform start.

FIELD MODE (omni phase-2 plan step 2, 2026-10-06): when the installed generator carries
``OmniFieldTerrainCfg`` sub-terrains (``KRABBY_FLAT_TERRAIN_GEOM=field``) the term switches its
"foot over a block" lookup to the generator's exact occupancy channel indexed across tile seams,
and adds the field-only keys of ``omni_exposure.OMNI_FIELD_KEYS``. Detected from the generator's
classes -- no new cfg field -- so a ring run's config dump and metric list are unchanged. See
``omni_exposure``'s FIELD MODE section for what the shared keys mean there.

OBSTACLE-TILE CURRICULUM TELEMETRY (omni phase-2 CHARTER 8 / 5, 2026-10-06), ring AND field: the
``reset`` override adds ``terrain_levels_obst`` / ``promote_count`` / ``demote_count`` to each reset
call's log, and the ledger reports ``track_ratio_obst_p50`` (the median promotion ratio of its window's
obstacle-tile episodes). Telemetry only -- no cfg field, so no config dump changes; both geometries
gain the same four ``Metrics/base_parkour/*`` keys. Definitions: ``omni_exposure``'s
OBSTACLE-TILE CURRICULUM TELEMETRY section.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.utils import configclass

from parkour_isaaclab.envs.mdp.parkours.parkour_event import ParkourEvent
from parkour_isaaclab.envs.mdp.parkours.parkour_events_cfg import ParkourEventsCfg

from .crab_hex_omni_command_math import SLOT_MOVE, wrap_to_pi
from .omni_exposure import (
    SPAWN_PLATFORM,
    OmniExposureLedger,
    block_mask_from_height_field,
    episode_exposure_omni,
    feet_block_flags_tiled,
    global_block_mask,
    global_pixel_index,
    obstacle_level_mean,
    obstacle_promotion_counts,
    promotion_mask,
    sector_index,
    terrain_window_extras,
    tile_pixel_index,
    tracking_ratio,
)
from .omni_terrain_knobs import is_field_generator

if TYPE_CHECKING:
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv


class OmniParkourEvent(ParkourEvent):
    """Ring-shaped parkour term: no goal advance, radial exposure, ratio promotion."""

    cfg: OmniParkourEventsCfg

    #: Field mode (omni phase-2 plan step 2): set in ``__init__`` from the generator's sub-terrain
    #: classes. The class-level False is the ring, so every ring code path below reads exactly as
    #: it did before the field existed.
    field_mode: bool = False

    #: Obstacle-tile promotions / demotions applied since the last ``reset`` log (CHARTER 8 telemetry);
    #: class-level zeros so a bare test double starts clean, instance attributes once incremented.
    _win_promote: int = 0
    _win_demote: int = 0

    def __init__(self, cfg: OmniParkourEventsCfg, env: ParkourManagerBasedRLEnv):
        # NOTE: the base __init__ calls self._platform_start_xy(slice(None)) and
        # self._obstacle_tile_mask(); both overrides below only use attributes the base has
        # already set at that point (env_origins, env_per_terrain_name).
        super().__init__(cfg, env)
        self.clear_radius = float(cfg.clear_radius_m)
        self.ring_outer_radius = float(cfg.ring_outer_radius_m)
        self.n_sectors = int(cfg.n_sectors)
        self.min_cmd_dist = float(cfg.min_cmd_dist_m)
        self._command_name = str(cfg.command_name)
        if not (0.0 <= self.clear_radius < self.ring_outer_radius):
            raise ValueError(
                "OmniParkourEventsCfg requires 0 <= clear_radius_m < ring_outer_radius_m, got "
                f"({cfg.clear_radius_m}, {cfg.ring_outer_radius_m})"
            )
        if self.n_sectors <= 0:
            raise ValueError(f"n_sectors must be positive, got {cfg.n_sectors}")
        # Field mode (omni phase-2 step 2): by the generator's sub-terrain CLASSES, no cfg field.
        from parkour_isaaclab.terrains.omni.omni_terrains_cfg import OmniFieldTerrainCfg

        self.field_mode = is_field_generator(
            getattr(self.terrain.cfg.terrain_generator, "sub_terrains", None), OmniFieldTerrainCfg
        )

        n, dev = self.num_envs, self.device
        # -- per-episode accumulators (reset by _begin_episodes, consumed by _finalize_episodes)
        self.ep_r_max = torch.zeros(n, device=dev)
        self.ep_sector_seen = torch.zeros(n, self.n_sectors, dtype=torch.bool, device=dev)
        self.ep_block_steps = torch.zeros(n, device=dev)
        self.ep_track_dist = torch.zeros(n, device=dev)
        self.ep_cmd_dist = torch.zeros(n, device=dev)
        # The same two integrals restricted to the rows that are actually being asked to WALK
        # (``SLOT_MOVE``): see _accumulate_exposure for why the unrestricted pair cannot
        # answer "is it tracking".
        self.ep_track_dist_move = torch.zeros(n, device=dev)
        self.ep_cmd_dist_move = torch.zeros(n, device=dev)
        self.ep_head_err_sq = torch.zeros(n, device=dev)
        # Field mode only (read only when ``self.field_mode``): steps with a foot off the own tile,
        # and steps with a foot over a NEIGHBOURING tile's block.
        self.ep_off_tile_steps = torch.zeros(n, device=dev)
        self.ep_off_tile_block_steps = torch.zeros(n, device=dev)
        # The base sets this inside _update_command; ours never advances goals, so define it
        # once so any consumer that reads it sees a well-shaped all-false mask.
        self.reached_goal_ids = torch.zeros(n, dtype=torch.bool, device=dev)
        self._env_arange = torch.arange(n, device=dev)

        # -- ledger over SECTORS (the base sized it over corridor obstacle goals)
        _stale = set(OmniExposureLedger.emit_keys(self.n_obst)) - set(
            OmniExposureLedger.emit_keys(self.n_sectors, field_mode=self.field_mode)
        )
        self.n_obst = self.n_sectors
        self.exposure = OmniExposureLedger(
            self.n_sectors, capacity=int(getattr(cfg, "exposure_window", 1024)), device="cpu",
            field_mode=self.field_mode,
        )
        for key in _stale:
            self.metrics.pop(key, None)
        for key in OmniExposureLedger.emit_keys(self.n_sectors, field_mode=self.field_mode):
            if key not in self.metrics:
                self.metrics[key] = torch.zeros(n, device="cpu")

        self._build_block_mask()
        self._resolve_feet(cfg)
        self._cmd_term_cache = None
        # First targets: the base __init__ derived them from the corridor goals.
        self._write_targets()

    # ------------------------------------------------------------------ construction

    def _build_block_mask(self) -> None:
        """Global bool mask of raised block pixels + the tile pixel frame to index it.

        The height field itself comes from ``events._height_field_cpu`` so the omni term
        shares the env-level cache with the spawn machinery instead of keeping a second
        copy of a multi-megabyte array.
        """
        from parkour_isaaclab.envs.mdp.events import _height_field_cpu

        gen = self.terrain.terrain_generator_class
        tgc = self.terrain.cfg.terrain_generator
        hf = _height_field_cpu(self.env)
        self._tile_px = (int(gen.height_fields.shape[2]), int(gen.height_fields.shape[3]))
        self._tile_size = (float(tgc.size[0]), float(tgc.size[1]))
        self._hscale = float(tgc.horizontal_scale)
        self._grid_rc = (int(gen.height_fields.shape[0]), int(gen.height_fields.shape[1]))
        if self.field_mode:
            # The generator's EXACT occupancy (blocks and pits), not the median + tol rule: field
            # pits sit BELOW the median and would never pass it. ``block_masks`` is None when no
            # tile returned one (an all-flat field run) -> all False.
            self._block_mask = global_block_mask(
                getattr(gen, "block_masks", None), tuple(gen.height_fields.shape)
            ).to(self.device)
            return
        tol_px = float(self.cfg.block_height_tol_m) / float(tgc.vertical_scale)
        self._block_mask = block_mask_from_height_field(hf, self._tile_px, tol_px).to(self.device)

    def _resolve_feet(self, cfg: OmniParkourEventsCfg) -> None:
        """Body indices of the feet, for the "foot over a block pixel" telemetry.

        Telemetry only: an unresolvable pattern warns and disables ``reach_obst`` /
        ``block_steps_mean`` rather than failing a run (the term must stay usable on a
        robot whose feet are named differently).
        """
        self._foot_ids: list[int] = []
        names = [str(p) for p in cfg.foot_body_names]
        try:
            ids, resolved = self.robot.find_bodies(names, preserve_order=False)
            self._foot_ids = list(ids)
            self._foot_names = list(resolved)
        except Exception as exc:  # noqa: BLE001 -- any resolution failure is non-fatal
            self._foot_names = []
            print(
                f"[OmniParkourEvent] foot_body_names {names} did not resolve ({exc}); "
                "reach_obst / block_steps_mean will stay at zero.",
                flush=True,
            )

    # ------------------------------------------------------------------ command access

    @property
    def _cmd_term(self):
        """The omni command term (``travel_dir_w``, ``speed_cmd``, ``heading_target``, ``slot_id``).

        Resolved lazily and cached: the command manager is built before the parkour manager
        (``ParkourManagerBasedRLEnv.load_managers``), so an eager lookup would work today,
        but a lazy one also survives a manager-order change and keeps unit-test doubles
        simple. The attribute check makes a wrong command class fail HERE, with a message,
        instead of as an AttributeError in the middle of a step.
        """
        if self._cmd_term_cache is None:
            term = self.env.command_manager.get_term(self._command_name)
            for attr in ("travel_dir_w", "speed_cmd", "heading_target", "vel_command_b", "slot_id"):
                if not hasattr(term, attr):
                    raise TypeError(
                        f"OmniParkourEvent needs an omni command term at "
                        f"'{self._command_name}' (missing '{attr}' on {type(term).__name__})"
                    )
            self._cmd_term_cache = term
        return self._cmd_term_cache

    # ------------------------------------------------------------------ spawn / targets

    def _platform_start_xy(self, env_ids) -> torch.Tensor:
        """World xy where the ring reset spawns env_ids: the TILE CENTRE.

        ``.clone()`` is load-bearing: the base assigns the return value straight into
        ``self.spawn_xy_w`` in ``__init__``, and a view of ``env_origins`` would alias the
        spawn record to the live tile origins (which are rewritten on every promotion).
        """
        return self.env_origins[env_ids, :2].clone()

    def note_spawn_xy(self, env_ids, xy: torch.Tensor, kind=SPAWN_PLATFORM) -> None:
        """Record where ``env_ids`` were actually placed (called by the reset events).

        Runs AFTER ``_begin_episodes`` (parkour manager resets before the event manager in
        ``_reset_idx``), so it overwrites the tile-centre default with the true spawn.
        ``kind`` drives the ledger's non-RSI / RSI split.
        """
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        if ids.numel() == 0:
            return
        kind_t = torch.as_tensor(kind, device=self.device, dtype=torch.long)
        if kind_t.dim() == 0:
            kind_t = kind_t.expand(ids.numel())
        self.spawn_xy_w[ids] = xy.to(self.device).reshape(-1, 2)
        self.spawn_kind[ids] = kind_t

    def note_spawn(self, env_ids, x_w: torch.Tensor, kind) -> None:
        """Refuse the corridor spawn callback -- it can only come from a mis-wired env.

        ``note_spawn`` carries a spawn's ``x`` alone, because the corridor only ever needed
        downrange distance. Its only callers are the corridor reset events
        (``events.reset_root_state`` and the forward ``crab_hex_rsi``), so reaching this
        method means one of those is installed in an omni env: the robot would be spawned
        ``size_y + offset`` metres off the tile centre -- outside the ring entirely -- while
        the exposure ledger kept reporting a disc it never stood in. Loud beats quiet.
        """
        raise RuntimeError(
            "OmniParkourEvent received note_spawn() (the corridor, x-only spawn callback). "
            "The omni env must install the ring reset events (omni_events.reset_root_state_ring "
            "and crab_hex_omni_rsi.reset_from_reference_states_omni), which report through "
            "note_spawn_xy(); a corridor reset event is wired into events.reset_root_state or "
            "events.rsi_reference_reset."
        )

    def _write_targets(self) -> None:
        """Point the goal-facing attributes at the COMMAND instead of at a course goal.

        ``target_pos_rel`` becomes the unit vector of the commanded world travel direction
        (every consumer normalises it anyway -- ``rewards.py:734``), and ``target_yaw``
        becomes the commanded world facing, so ``wrap(target_yaw - yaw)`` is exactly the
        facing error. ``next_*`` mirror them: with no goal sequence there is no "next".

        ``target_yaw`` is NOT wrapped, matching ``heading_target``'s house convention
        (``crab_hex_omni_command_math``: consumers wrap the difference). The forward
        ``reward_tracking_yaw``, which subtracts unwrapped, is replaced by the omni facing
        reward; the omni observation wraps.
        """
        cmd = self._cmd_term
        theta = cmd.travel_dir_w
        self.target_pos_rel = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)
        self.next_target_pos_rel = self.target_pos_rel.clone()
        self.target_yaw = cmd.heading_target.clone()
        self.next_target_yaw = self.target_yaw.clone()

    # ------------------------------------------------------------------ per-step update

    def _update_command(self):
        """Every step: re-point the targets at the command and accumulate exposure.

        Deliberately does NOT touch ``cur_goal_idx`` / ``reach_goal_timer``: the ring's
        goals are geometry anchors, not waypoints, so the goal index stays 0 for the whole
        episode (which is also what keeps ``terminate_episode``'s goal cutoff inert -- see
        the omni terminations, which drop it outright).
        """
        self._write_targets()
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)
        self.dis_to_start_pos = torch.norm(
            self.spawn_xy_w - self.robot.data.root_pos_w[:, :2], dim=1
        )
        self._accumulate_exposure()

    def _accumulate_exposure(self) -> None:
        """Radial / sector / block / tracking / facing accumulation for the live episode."""
        cmd = self._cmd_term
        dt = self.simulation_time
        rel = self.robot.data.root_pos_w[:, :2] - self.env_origins[:, :2]
        r = torch.norm(rel, dim=1)
        outside = r > self.clear_radius
        self.ep_r_max = torch.maximum(self.ep_r_max, r)
        self.ep_field_steps += outside.float()
        self.ep_steps += 1.0
        sec = sector_index(rel[:, 0], rel[:, 1], self.n_sectors)
        # A sector counts only from OUTSIDE the disc: pivoting on the spawn sweeps every
        # bearing without covering any ground.
        self.ep_sector_seen[self._env_arange, sec] |= outside
        if self.field_mode:
            over, over_off, off = self._feet_field_flags()
            self.ep_block_steps += over.float()
            self.ep_off_tile_block_steps += over_off.float()
            self.ep_off_tile_steps += off.float()
        else:
            self.ep_block_steps += self._feet_over_block().float()

        cmd_xy = cmd.vel_command_b[:, :2]
        cmd_norm = torch.norm(cmd_xy, dim=1)
        c_hat = cmd_xy / (cmd_norm.unsqueeze(1) + 1e-6)
        v_along = (self.robot.data.root_lin_vel_b[:, :2] * c_hat).sum(dim=1).clamp(min=0.0)
        self.ep_track_dist += v_along * dt
        self.ep_cmd_dist += cmd_norm * dt
        # -- the SLOT-GATED twin (WATCH telemetry, 2026-09-18). The pair above integrates over
        # every row, and the gate they feed (``track_ratio_mean``, the promotion rule) is a
        # ratio of two integrals -- so a ``move_turn`` row whose body command swings from
        # forward to lateral as the commanded turn executes ("clamp-only re-cap") keeps adding
        # to the DENOMINATOR while the robot is, correctly, doing something else. The result is
        # a tracking number that falls as soon as a window enables turning, for a reason that
        # has nothing to do with walking worse. Restricting to ``SLOT_MOVE`` -- moving, not
        # turning -- gives the number that IS comparable across windows: "when it is simply
        # told to walk somewhere, how much of that does it cover". Watched only; the existing
        # ratio stays exactly as it is and keeps driving promotion.
        move = (cmd.slot_id == SLOT_MOVE).to(v_along.dtype)
        self.ep_track_dist_move += v_along * move * dt
        self.ep_cmd_dist_move += cmd_norm * move * dt
        err = wrap_to_pi(cmd.heading_target - self.robot.data.heading_w)
        self.ep_head_err_sq += err * err

    def _feet_over_block(self) -> torch.Tensor:
        """``(N,)`` bool: at least one foot standing over a raised block pixel."""
        if not self._foot_ids:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        feet = self.robot.data.body_pos_w[:, self._foot_ids, :2]  # (N, F, 2)
        ix, iy = tile_pixel_index(
            feet[..., 0], feet[..., 1], self.env_origins[:, :2],
            self._tile_size, self._hscale, self._tile_px,
        )
        gx, gy = global_pixel_index(
            self.terrain.terrain_levels, self.terrain.terrain_types, ix, iy, self._tile_px
        )
        return self._block_mask[gx, gy].any(dim=1)

    def _feet_field_flags(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Field mode: ``(over_block, over_block_off_tile, off_tile)``, ``(N,)`` bool each, with
        feet on neighbouring tiles looked up on THOSE tiles (``omni_exposure.feet_block_flags_tiled``)."""
        if not self._foot_ids:
            z = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return z, z, z
        feet = self.robot.data.body_pos_w[:, self._foot_ids, :2]  # (N, F, 2)
        return feet_block_flags_tiled(
            feet[..., 0], feet[..., 1], self.env_origins[:, :2],
            self.terrain.terrain_levels, self.terrain.terrain_types, self._block_mask,
            self._tile_size, self._hscale, self._tile_px, self._grid_rc,
        )

    # ------------------------------------------------------------------ episode borders

    def _begin_episodes(self, env_ids) -> None:
        """Base bookkeeping (spawn defaults) plus the omni accumulators."""
        super()._begin_episodes(env_ids)
        self.ep_r_max[env_ids] = 0.0
        self.ep_sector_seen[env_ids] = False
        self.ep_block_steps[env_ids] = 0.0
        self.ep_track_dist[env_ids] = 0.0
        self.ep_cmd_dist[env_ids] = 0.0
        self.ep_track_dist_move[env_ids] = 0.0
        self.ep_cmd_dist_move[env_ids] = 0.0
        self.ep_head_err_sq[env_ids] = 0.0
        if self.field_mode:
            self.ep_off_tile_steps[env_ids] = 0.0
            self.ep_off_tile_block_steps[env_ids] = 0.0

    def _finalize_episodes(self, env_ids) -> None:
        """Push the finished episodes' exposure records into the omni ledger."""
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        ids = ids[self.ep_steps[ids] > 0]
        if ids.numel() == 0:
            return
        steps = self.ep_steps[ids]
        field_kw = {}
        # The LEDGER is the authority on which keys exist (it was built with field_mode in
        # __init__); a ring ledger has no field rings, so nothing field-only is computed for it.
        if self.exposure.field_mode:
            field_kw = {
                "off_tile_frac": (self.ep_off_tile_steps[ids] / steps.clamp(min=1)).cpu(),
                "off_tile_block_steps": self.ep_off_tile_block_steps[ids].cpu(),
            }
        per_ep = episode_exposure_omni(
            self.ep_r_max[ids], self.ep_sector_seen[ids], self.ep_block_steps[ids],
            self.ep_field_steps[ids], steps, self.clear_radius,
        )
        self.exposure.push_episodes(
            {k: v.cpu() for k, v in per_ep.items()},
            self.spawn_kind[ids].cpu(),
            self.is_obst_tile[ids].cpu(),
            self._failure_flags(ids).cpu(),
            steps.cpu(),
            track_ratio=tracking_ratio(self.ep_track_dist[ids], self.ep_cmd_dist[ids]).cpu(),
            track_ratio_valid=(self.ep_cmd_dist[ids] >= self.min_cmd_dist).cpu(),
            # Same rule, same threshold, restricted to the walking rows: an episode that was
            # barely asked to WALK is left out of the mean rather than counted as a zero.
            track_ratio_move=tracking_ratio(
                self.ep_track_dist_move[ids], self.ep_cmd_dist_move[ids]
            ).cpu(),
            track_ratio_move_valid=(self.ep_cmd_dist_move[ids] >= self.min_cmd_dist).cpu(),
            heading_err_rms=torch.sqrt(self.ep_head_err_sq[ids] / steps.clamp(min=1)).cpu(),
            block_steps=self.ep_block_steps[ids].cpu(),
            r_max=self.ep_r_max[ids].cpu(),
            **field_kw,
        )
        self._exposure_means = self.exposure.means()
        # The lazy keys (omni_exposure.OMNI_LAZY_KEYS: the spawn-kind split and the obstacle-tile
        # median ratio) appear in the means only once their group is non-empty; the metric is
        # created THEN, seeded with its value, so an empty group is never logged (rather than
        # logged as 0.0 like the emit_keys rings).
        for key in self.exposure.lazy_keys:
            if key in self._exposure_means and key not in self.metrics:
                self.metrics[key] = torch.full((self.num_envs,), float(self._exposure_means[key]), device="cpu")

    # ------------------------------------------------------------------ reset / promote

    def _resample_command(self, env_ids: Sequence[int]):
        """Close the old episode, promote/demote the tile, open the new one.

        The level/origin/goal-table bookkeeping is the forward term's, line for line; only
        the promotion CRITERION changes (see :func:`omni_exposure.promotion_mask`). The
        ordering the forward term relies on is preserved: exposure is finalised BEFORE the
        tile moves, and ``_begin_episodes`` runs last so the reset events (which run after
        the parkour manager) can overwrite the spawn record.
        """
        self._finalize_episodes(env_ids)

        # Telemetry ("how_far_from_start_point"): distance from the ACTUAL spawn. Written
        # in place -- the forward term rebinds the (N,) buffer to a (len(env_ids),) tensor
        # here, which is only safe because _update_command rebinds it back on the next step.
        self.dis_to_start_pos[env_ids] = torch.norm(
            self.spawn_xy_w[env_ids] - self.robot.data.root_pos_w[env_ids, :2], dim=1
        )
        move_up, move_down = promotion_mask(
            self.ep_track_dist[env_ids], self.ep_cmd_dist[env_ids],
            self.cfg.move_up_frac, self.cfg.move_down_frac, self.min_cmd_dist,
        )

        if not self.cfg.freeze_terrain_levels:
            # CHARTER 8 telemetry: the APPLIED decisions on episodes that ended on an obstacle tile
            # (``is_obst_tile`` still describes the old tile here; it is refreshed below).
            n_up, n_down = obstacle_promotion_counts(move_up, move_down, self.is_obst_tile[env_ids])
            self._win_promote += n_up
            self._win_demote += n_down
            self.terrain.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
            # Robots that solve the last level are sent to a random one
            self.terrain.terrain_levels[env_ids] = torch.where(
                self.terrain.terrain_levels[env_ids] >= self.terrain.max_terrain_level,
                torch.randint_like(self.terrain.terrain_levels[env_ids], self.terrain.max_terrain_level),
                torch.clip(self.terrain.terrain_levels[env_ids], 0),
            )
        self.env_origins[env_ids] = self.terrain.terrain_origins[
            self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids]
        ]
        self.env_class[env_ids] = self.terrain_class[
            self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids]
        ]

        temp = self.terrain_goals[self.terrain.terrain_levels, self.terrain.terrain_types]
        last_col = temp[:, -1].unsqueeze(1)
        self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.num_future_goal_obs, 1)), dim=1)[:]
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)
        self._write_targets()

        numpy_terrain_levels = self.terrain.terrain_levels.detach().cpu().numpy()
        numpy_terrain_types = self.terrain.terrain_types.detach().cpu().numpy()
        self.env_per_terrain_name = self.total_terrain_names[numpy_terrain_levels, numpy_terrain_types]

        self.reach_goal_timer[env_ids] = 0
        self.cur_goal_idx[env_ids] = 0
        self.is_obst_tile = self._obstacle_tile_mask()
        self._begin_episodes(env_ids)

        if self.debug_vis:
            self.future_goal_idx[env_ids, 0] = False
            self.future_goal_idx[env_ids, 1:] = True
            self.env_per_heights = self.total_heights[
                self.terrain.terrain_levels, self.terrain.terrain_types
            ]

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """The base reset log plus the obstacle-tile curriculum keys of THIS reset call.

        ``ParkourTerm.reset`` logs ``self.metrics`` and only then resamples, so a count written into
        ``self.metrics`` during the resample would surface one reset call late. Here the obstacle
        level is read BEFORE the base call (tiles not yet moved: the instant the all-tile
        ``terrain_levels`` metric describes) and the promotion counts AFTER it (the resample has
        just applied them), and both go into the same returned dict the command manager prefixes
        with ``Metrics/<term>/`` -- the path every other key of this term takes.
        """
        level_obst = obstacle_level_mean(self.terrain.terrain_levels, self.is_obst_tile)
        extras = super().reset(env_ids)
        extras.update(terrain_window_extras(level_obst, self._win_promote, self._win_demote))
        self._win_promote = 0
        self._win_demote = 0
        return extras

    # ------------------------------------------------------------- debug visualisation
    #
    # PLAY only (``_apply_play`` sets ``parkours.base_parkour.debug_vis``). The corridor
    # term's four visualisers say things the ring does not mean, and two of them now
    # collide with the command term's own arrows (see ``CrabHexOmniCommand``'s colour
    # table). What each one becomes here -- nothing is dropped silently:
    #
    #   base marker                  corridor meaning          omni treatment
    #   ---------------------------  ------------------------  ------------------------
    #   current_goal_pose (1 GREEN   "the waypoint you are     HIDDEN. The ring's goals
    #   sphere, r 0.165)             walking to"               are geometry anchors, not
    #                                                          waypoints (``cur_goal_idx``
    #                                                          never advances), so calling
    #                                                          anchor 0 "current" invents
    #                                                          an ordering -- and a green
    #                                                          sphere now reads as
    #                                                          "commanded travel".
    #   future_goal_poses (RED       "the waypoints after it"  KEPT, and widened to draw
    #   spheres, r 0.165)                                      ALL ``n_sectors`` anchors
    #                                                          uniformly: they mark where
    #                                                          the block ring's sectors
    #                                                          are, which is real and
    #                                                          useful next to the command.
    #   current_arrow (8 small       "bead trail towards the   KEPT, meaning changed with
    #   GREEN spheres, 0.3..1.0 m)   current waypoint"         ``_write_targets``: it is
    #                                                          the COMMANDED TRAVEL
    #                                                          DIRECTION, on the ground --
    #                                                          the ground track under the
    #                                                          command term's green arrow,
    #                                                          and gated on the commanded
    #                                                          speed so the two appear and
    #                                                          vanish together.
    #   future_arrow (8 small RED    "bead trail towards the   HIDDEN. ``_write_targets``
    #   spheres, 0.6..2.0 m)         next waypoint"            sets ``next_target_pos_rel
    #                                                          == target_pos_rel``, so this
    #                                                          is the green trail again at
    #                                                          twice the spacing -- pure
    #                                                          duplication, and the red
    #                                                          beads then run through the
    #                                                          red sector anchors.

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create the base's four visualisers, then hide the two the ring makes meaningless."""
        super()._set_debug_vis_impl(debug_vis)
        if not debug_vis:
            return
        self.current_goal_pose_visualizer.set_visibility(False)
        self.future_arrow_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Ring sector anchors + the ground bead trail of the commanded travel direction.

        The expressions are the base callback's, with the goal split dropped (all anchors
        are drawn by the one visualiser) and the duplicated "future" trail dropped.
        """
        if not self.robot.is_initialized:
            return
        # -- all ``num_goals`` ring sector anchors, in world xy at their own goal height.
        env_per_goals = self.terrain_goals[self.terrain.terrain_levels, self.terrain.terrain_types]
        anchors_xy = env_per_goals[:, :, :2].reshape(self.num_envs, -1, 2) + self.env_origins[:, :2].unsqueeze(1)
        anchors_z = self.env_per_heights.unsqueeze(-1) * self.terrain.cfg.terrain_generator.vertical_scale
        self.future_goal_poses_visualizer.visualize(
            translations=torch.concat([anchors_xy, anchors_z], dim=-1).reshape(-1, 3),
        )
        # -- bead trail along the commanded world travel direction, at the base's own z.
        # GATED on the commanded speed. ``travel_dir_w`` is drawn for every row whatever the
        # slot (``_resample_command`` keeps the RNG stream slot-independent), so on a stand
        # or turn slot ``target_pos_rel`` is a direction nobody is being asked to walk in --
        # and a trail pointing east while the robot is commanded to turn on the spot reads
        # as a contradiction of the command term's (correctly absent) green arrow. Collapsed
        # onto the base rather than hidden, so the visualiser keeps a constant instance
        # count; the beads stack invisibly under the chassis.
        moving = (self._cmd_term.speed_cmd > 0.0).float().unsqueeze(-1)
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5) * moving
        root_pos_w = self.robot.data.root_pos_w
        beads = [
            torch.concat(
                [root_pos_w[:, :2] + 0.1 * (i + 3) * target_vec_norm[:, :2], root_pos_w[:, 2][:, None]],
                dim=1,
            )
            for i in range(self.arrow_num)
        ]
        self.current_arrow_visualizer.visualize(translations=torch.concat(beads, dim=0))


@configclass
class OmniParkourEventsCfg(ParkourEventsCfg):
    """Cfg for :class:`OmniParkourEvent` (plan D9/B5 defaults).

    ``move_up_frac`` / ``move_down_frac`` are INHERITED (0.8 / 0.4) rather than re-declared:
    every omni window sets ``KRABBY_TERRAIN_PROMOTE=0.45:0.25`` through the shared knob
    block, and leaving the inherited values in place keeps the omni-vs-forward config delta
    to the genuinely new fields. Note the units changed even though the names did not --
    they are now a dimensionless TRACKING RATIO, not a fraction of commanded distance -- so
    the inherited 0.8/0.4 is a strict, conservative default (a plant tracking ~0.5 never
    promotes) rather than a silently wrong one.
    """

    class_type: type = OmniParkourEvent

    clear_radius_m: float = 2.0
    """Radius of the block-free disc the robot spawns in, metres. Must match the terrain's
    ``OmniRingTerrainCfg.clear_radius``; ``reach_edge`` and ``field_frac`` are defined
    against it. Under the field geometry the knob layer syncs it to the PAD radius
    (``OmniFieldTerrainCfg.clear_radius``, 1.7 m), so ``reach_edge`` = "left the pad"."""

    ring_outer_radius_m: float = 4.5
    """Outer radius of the block annulus, metres (terrain ``ring_outer_radius``). Used for
    validation and for reading ``r_max_mean``; the exposure rules only need the disc. Under the
    field geometry it is synced to the field's square HALF-WIDTH (4.80 m)."""

    n_sectors: int = 8
    """Number of bearing sectors, centred on the ring's goal anchors. Sets the number of
    ``obst_coverage_k`` ledger keys and the range of ``goals_passed``."""

    min_cmd_dist_m: float = 1.0
    """Commanded distance below which an episode neither promotes nor demotes (and is left
    out of ``track_ratio_mean``)."""

    block_height_tol_m: float = 0.03
    """Height above the tile's median at which a height-field pixel counts as a block.

    RAISED from the plan's original 0.015 (decision recorded in D9). At the SHIPPED preset
    (``noise_range = (0.02, 0.02)`` at every difficulty -- see the caveat in :mod:`omni_exposure`;
    the class default ``(0.02, 0.06)`` is never what actually ships) the ring's own roughness
    measures -0.030..+0.025 m about the tile median at EVERY difficulty, so 0.015 would have sat
    inside that band and produced false-positive block pixels. 0.03 clears the roughness band
    everywhere: zero pixels rise above median + 0.03 m from roughness alone (measured on the
    real decorated ring function at the real cfg, d in {0.05, 0.2, 0.7}). The blocks themselves
    are 0.0225-0.042 m at 1a and 0.055-0.072 m at the 2a+ default, so the mask is conservative
    rather than saturated: the lowest 1a blocks (2.25-3.3 cm at d 0.05-0.2) sit AT the tolerance
    and can be missed -- an under-reporting bias, never an over-reporting one from roughness.
    ``reach_obst`` and ``block_steps_mean`` stay TELEMETRY, never a gate, at every difficulty."""

    foot_body_names: tuple[str, ...] = (".*_Footpad",)
    """Body-name patterns whose world xy is tested against the block mask. The crab's six
    feet are ``{FL,FR,ML,MR,RL,RR}_Footpad``."""

    command_name: str = "base_velocity"
    """Command-manager term the travel direction / facing target are read from."""
