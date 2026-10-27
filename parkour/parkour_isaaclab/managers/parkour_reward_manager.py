
from __future__ import annotations

from collections.abc import Sequence

import torch
from typing import TYPE_CHECKING
import omni.kit.app
from isaaclab.managers import RewardManager

if TYPE_CHECKING:
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv

class ParkourRewardManager(RewardManager):
    _env: ParkourManagerBasedRLEnv

    def __init__(self, cfg: object, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)

    def compute(self, dt: float) -> torch.Tensor:
        """
        Same to Legged Gym
        """
        # reset computation
        self._reward_buf[:] = 0.0
        # OPT-IN per-category telemetry (stride/posture plan, step 4). Off by default and for
        # every shipped cfg -- ``getattr`` so a cfg that never sets the attribute pays nothing
        # here. It only READS ``_reward_buf``/``_episode_sums``/``_step_reward`` below, never
        # writes them, so this branch cannot change what OFF computes.
        split = bool(getattr(getattr(self._env, "cfg", None), "reward_category_split", False))
        step_values: list[torch.Tensor] | None = [None] * len(self._term_names) if split else None
        # iterate over all the reward terms
        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            # skip if weight is zero (kind of a micro-optimization)
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                if split:
                    step_values[term_idx] = torch.zeros_like(self._reward_buf)
                continue
            # compute term's value
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            # update total reward
            self._reward_buf += value
            # update episodic sum
            self._episode_sums[name] += value
            # Update current reward for this step.
            self._step_reward[:, term_idx] = value / dt
            if split:
                step_values[term_idx] = value
        # ``_reward_buf`` right here IS the pre-clip total -- clone it before the clip mutates
        # ``_reward_buf`` in place, so OFF and ON compute byte-identical post-clip rewards.
        pre_clip_total = self._reward_buf.clone() if split else None
        self._reward_buf[:] = torch.clip(self._reward_buf[:], min=0.)
        if split:
            self._accumulate_category_split(dt, pre_clip_total, self._reward_buf, step_values)
        return self._reward_buf

    def _accumulate_category_split(
        self,
        dt: float,
        pre_clip_total: torch.Tensor,
        post_clip_total: torch.Tensor,
        step_values: list[torch.Tensor],
    ) -> None:
        """The ``reward_category_split`` bookkeeping for one ``compute()`` call.

        Everything here is READ-ONLY with respect to the base manager's own buffers -- it only
        accumulates into the ``_cat_*`` buffers below, keyed by this step's
        :meth:`telemetry_category`, all in per-second units (divided by ``dt`` as it lands, so
        :meth:`reset` just averages by dwell-step count -- see its docstring for the formula).
        """
        env = self._env
        command_term_name = getattr(env.cfg, "reward_category_command", "base_velocity")
        term = env.command_manager.get_term(command_term_name)
        category = term.telemetry_category()
        # Duck-typed, not imported: this module (``parkour_isaaclab``) must not depend on any
        # task package, so the category NAMES come off whatever term ``telemetry_category()``
        # was read from (``CrabHexOmniCommand.TELEMETRY_CATEGORIES``), not off an import.
        self._reward_cat_names = getattr(term, "TELEMETRY_CATEGORIES", None)

        n_envs = category.shape[0]
        n_cats = int(category.max().item()) + 1 if n_envs > 0 else 0
        self._grow_category_buffers(n_cats)
        env_idx = torch.arange(n_envs, device=category.device)

        self._cat_dwell[env_idx, category] += 1.0
        self._cat_pre_clip_sum[env_idx, category] += pre_clip_total / dt
        self._cat_post_clip_sum[env_idx, category] += post_clip_total / dt
        self._cat_clip_bind_count[env_idx, category] += (pre_clip_total < 0.0).to(self._cat_clip_bind_count.dtype)

        event_term_names = tuple(getattr(env.cfg, "reward_category_event_terms", ("reward_stride_length",)))
        pre_clip_floor = torch.clamp(pre_clip_total, min=0.0)
        event_pay_total = torch.zeros_like(pre_clip_total)
        absorbed_total = torch.zeros_like(pre_clip_total)
        for event_name in event_term_names:
            if event_name not in self._term_names:
                continue
            v_e = step_values[self._term_names.index(event_name)]
            # absorbed = v_e - min(v_e, max(0, pre_clip_total)): 0 with headroom to spare,
            # v_e in full once the total floor clips it away entirely.
            absorbed_total = absorbed_total + (v_e - torch.minimum(v_e, pre_clip_floor))
            event_pay_total = event_pay_total + v_e
        self._cat_event_pay_sum[env_idx, category] += event_pay_total
        self._cat_absorbed_sum[env_idx, category] += absorbed_total

        # One index_put for all terms (not one per term): [N, T] per-second values.
        self._cat_term_sum[env_idx, category] += torch.stack(step_values, dim=1) / dt

    def _grow_category_buffers(self, n_cats: int) -> None:
        """Lazily allocate, or zero-pad, the ``_cat_*`` accumulators to ``n_cats`` columns.

        Sized off the LARGEST category index seen so far rather than a fixed constant --
        this manager has no import on :data:`TELEMETRY_CATEGORIES` (see
        :meth:`_accumulate_category_split`) and so no compile-time count to allocate against.
        Existing accumulated totals are preserved; only the new columns start at zero.
        """
        n_envs = self._reward_buf.shape[0]
        n_terms = len(self._term_names)
        device = self._reward_buf.device
        dtype = self._reward_buf.dtype
        if not hasattr(self, "_cat_dwell"):
            self._cat_dwell = torch.zeros(n_envs, n_cats, dtype=dtype, device=device)
            self._cat_term_sum = torch.zeros(n_envs, n_cats, n_terms, dtype=dtype, device=device)
            self._cat_pre_clip_sum = torch.zeros(n_envs, n_cats, dtype=dtype, device=device)
            self._cat_post_clip_sum = torch.zeros(n_envs, n_cats, dtype=dtype, device=device)
            self._cat_clip_bind_count = torch.zeros(n_envs, n_cats, dtype=dtype, device=device)
            self._cat_absorbed_sum = torch.zeros(n_envs, n_cats, dtype=dtype, device=device)
            self._cat_event_pay_sum = torch.zeros(n_envs, n_cats, dtype=dtype, device=device)
            return
        current = self._cat_dwell.shape[1]
        if n_cats <= current:
            return
        pad = n_cats - current

        def _zeros(*extra_dims: int) -> torch.Tensor:
            return torch.zeros(n_envs, pad, *extra_dims, dtype=dtype, device=device)

        self._cat_dwell = torch.cat([self._cat_dwell, _zeros()], dim=1)
        self._cat_term_sum = torch.cat([self._cat_term_sum, _zeros(n_terms)], dim=1)
        self._cat_pre_clip_sum = torch.cat([self._cat_pre_clip_sum, _zeros()], dim=1)
        self._cat_post_clip_sum = torch.cat([self._cat_post_clip_sum, _zeros()], dim=1)
        self._cat_clip_bind_count = torch.cat([self._cat_clip_bind_count, _zeros()], dim=1)
        self._cat_absorbed_sum = torch.cat([self._cat_absorbed_sum, _zeros()], dim=1)
        self._cat_event_pay_sum = torch.cat([self._cat_event_pay_sum, _zeros()], dim=1)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """The base per-term ``Episode_Reward/*`` extras, plus -- only when this env ever ran
        a ``reward_category_split`` compute -- a per-:data:`TELEMETRY_CATEGORIES` breakdown
        over ``env_ids``, then those envs' ``_cat_*`` accumulators are zeroed.

        Every known category is logged on every reset, so its keys are present in the first
        logged step of each iteration (the runner logs only those keys). A category with zero
        dwell over ``env_ids`` logs NaN rates, which the runner's ``nanmean`` ignores, and
        ``dwell_s`` 0. Per category ``c``:

        * ``Episode_RewardCat/<c>/<term>``, ``/pre_clip_total``, ``/post_clip_total`` -- the
          per-second rate, i.e. ``sum(value) / (sum(dwell_steps) * dt)``. Because
          :meth:`_accumulate_category_split` already divides by ``dt`` as each step lands,
          this is exactly the accumulated sum divided by the dwell-step count.
        * ``/clip_bind_frac`` -- the fraction of dwell steps whose pre-clip total was negative.
        * ``/clip_absorbed_event`` -- the event terms' (``reward_category_event_terms``)
          absorbed pay over their total pay, 0 when they paid nothing over this dwell.
        * ``/dwell_s`` -- the category's wall time over ``env_ids``, ``dwell_steps * dt``.
        """
        extras = super().reset(env_ids)
        if not hasattr(self, "_cat_dwell"):
            return extras
        device = self._cat_dwell.device
        if env_ids is None:
            env_ids_t = torch.arange(self._cat_dwell.shape[0], device=device)
        elif torch.is_tensor(env_ids):
            env_ids_t = env_ids.to(device=device)
        else:
            env_ids_t = torch.as_tensor(list(env_ids), dtype=torch.long, device=device)
        if env_ids_t.numel() == 0:
            return extras

        dt = float(self._env.step_dt)
        names = getattr(self, "_reward_cat_names", None)
        # Vectorised on device: ONE host sync per reset (the zero-dwell skip list), and every
        # logged value is a 0-dim view into ``table`` (no per-key ``.item()`` sync or H2D copy
        # -- a per-key loop is ~7 x (n_terms + 6) syncs on nearly every GPU step, and every
        # screen arm, B included, carries the split flag).
        dwell = self._cat_dwell[env_ids_t].sum(dim=0)  # [C] dwell steps
        safe_dwell = dwell.clamp(min=1.0)
        event_pay = self._cat_event_pay_sum[env_ids_t].sum(dim=0)
        absorbed = self._cat_absorbed_sum[env_ids_t].sum(dim=0)
        nonzero_pay = event_pay != 0.0
        clip_absorbed_event = torch.where(
            nonzero_pay,
            absorbed / torch.where(nonzero_pay, event_pay, torch.ones_like(event_pay)),
            torch.zeros_like(event_pay),
        )
        rates = torch.cat(
            [
                self._cat_term_sum[env_ids_t].sum(dim=0) / safe_dwell[:, None],  # [C, T]
                torch.stack(
                    [
                        self._cat_pre_clip_sum[env_ids_t].sum(dim=0) / safe_dwell,
                        self._cat_post_clip_sum[env_ids_t].sum(dim=0) / safe_dwell,
                        self._cat_clip_bind_count[env_ids_t].sum(dim=0) / safe_dwell,
                        clip_absorbed_event,
                    ],
                    dim=1,
                ),  # [C, 4]
            ],
            dim=1,
        )
        rates = torch.where((dwell > 0.0)[:, None], rates, torch.full_like(rates, float("nan")))
        table = torch.cat([rates, (dwell * dt)[:, None]], dim=1)  # [C, T + 5]; all on device, no host sync
        columns = list(self._term_names) + [
            "pre_clip_total", "post_clip_total", "clip_bind_frac", "clip_absorbed_event", "dwell_s",
        ]
        n_known = len(names) if names is not None else table.shape[0]
        for c in range(n_known):
            cat_name = names[c] if names is not None else str(c)
            prefix = f"Episode_RewardCat/{cat_name}"
            for col, col_name in enumerate(columns):
                if c < table.shape[0]:
                    extras[f"{prefix}/{col_name}"] = table[c, col]
                else:  # a known category the lazily-grown buffers have not reached yet: never dwelt
                    extras[f"{prefix}/{col_name}"] = torch.tensor(0.0 if col_name == "dwell_s" else float("nan"), device=device)

        self._cat_dwell[env_ids_t] = 0.0
        self._cat_term_sum[env_ids_t] = 0.0
        self._cat_pre_clip_sum[env_ids_t] = 0.0
        self._cat_post_clip_sum[env_ids_t] = 0.0
        self._cat_clip_bind_count[env_ids_t] = 0.0
        self._cat_absorbed_sum[env_ids_t] = 0.0
        self._cat_event_pay_sum[env_ids_t] = 0.0
        return extras
