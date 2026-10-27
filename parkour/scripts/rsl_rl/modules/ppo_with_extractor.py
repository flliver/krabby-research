
from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any

from tensordict import TensorDict

from rsl_rl.extensions import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_optimizer

from parkour_isaaclab.utils.nonfinite_logging import warn_if_nonfinite

from .actor_critic_with_encoder import ActorCriticRMA
from .crab_actor_critic_with_encoder import ACTION_STD_MAX, ACTION_STD_MIN, soft_std

# ------------------------------------------------------------------ PPO construction (2026-10-01)
# PPO-construction plan step 2 (``~/.claude/plans/foamy-wondering-ullman.md``): always-on telemetry
# under ``PPO/`` plus DEFAULT-OFF construction knobs (action-bound loss, step-size schedules, KL early
# stop, history-encoder LR, per-hand diagnostics). Every knob's constructor default reproduces the
# pre-change update bit for bit; the telemetry reads storage / runs extra no-grad forwards only and
# draws no random numbers, so it cannot move a trained value (pinned by
# ``tests/unit/test_ppo_construction_knobs.py``).

#: The step-size schedules. ``adaptive`` is the rsl_rl per-minibatch KL rule (every shipped cfg);
#: ``fixed`` keeps the configured LR (optionally with :attr:`PPOWithExtractor.kl_stop`'s early stop);
#: ``epoch_kl`` applies the adaptive /1.5 or *1.5 rule ONCE per epoch on the epoch-mean KL.
SCHEDULES = ("adaptive", "fixed", "epoch_kl")
#: The adaptive rule's LR floor and cap (also used by ``epoch_kl``).
LR_FLOOR = 1.0e-5
LR_CAP = 1.0e-2
#: The raw-action clip every crab runner applies (``clip_actions = 1.0``): what is EXECUTED.
EXEC_ACTION_CLIP = 1.0
#: Rows of the stored batch (flattened ``[T * N]``, first rows) the executed-change and per-hand
#: diagnostics read: a FIXED, non-random slice, so neither draws a random number.
TELEMETRY_SLICE_ROWS = 4096
#: The crab action layout (``run_meta.action_joint_names``): 6 cam-shaft VELOCITY channels, then
#: the 12 rod channels (6 hip-femur, 6 femur-tibia). Saturation is reported per group for an
#: 18-dim action; any other width (Go2) is reported as one ``_all`` group.
CRAB_ACTION_DIM = 18
CRAB_CAM_DIMS = 6
#: Per-hand diagnostics: every ``HAND_DIAG_EVERY`` training ITERATIONS (``iteration % 25 == 0``; the
#: runner sets :attr:`PPOWithExtractor.iteration` before each update -- :attr:`counter` would drift, since
#: ``update_dagger`` advances it too), rows split by the sign of the observed lateral command beyond a
#: dead zone. Without an iteration (a bare algorithm, as in the unit tests) the counter keys it.
HAND_DIAG_EVERY = 25
HAND_DIAG_DEADZONE = 0.05
#: p2r statue distillation (``P2R_STATUE_PLAN.md`` step 1, 2026-10-09): the gradient-norm split
#: (``Loss/grad_norm_statue`` / ``_anchor`` / ``_rest``) runs on the first minibatch of every
#: ``STATUE_GRAD_SPLIT_EVERY``-th iteration (same iteration/counter keying as the hand diagnostics).
STATUE_GRAD_SPLIT_EVERY = 50
#: A row is a STAND row when its mask column exceeds this (the bit is exactly 0.0 or 1.0).
STAND_BIT_THRESHOLD = 0.5
#: NOTE(p2r Z2, 2026-10-10): the statue prior's non-zero targets (``stand_prior_target``). ``restance`` = the
#: scripted re-stance controller R(obs) of ``crab_hex_omni_restance.py`` (``P2R_Z2_PLAN.md``), loaded BY PATH
#: (pure torch; the task package's ``__init__`` would pull in Isaac).
STAND_PRIOR_TARGETS = ("restance",)
RESTANCE_PY = (Path(__file__).resolve().parents[3]
               / "parkour_tasks/parkour_tasks/crab_hex_omni_task/mdp/crab_hex_omni_restance.py")
#: A bit-1 row's R target counts as non-zero (``Loss/restance_nonzero``) when any component exceeds this.
RESTANCE_NONZERO_EPS = 1e-6
#: NOTE(p2r Z2c, 2026-10-10): the clean-state cloning (BC) bank (``P2R_Z2C_PLAN.md``): the recorded policy
#: observation width every bank must carry (the omni teacher's, ``p2r_stand_bank.npz`` ``obs``). Only the actor's
#: training-path columns (``num_prop + num_scan + num_priv_explicit + num_priv_latent`` = 0:257 on the omni actor)
#: are kept on the device; R is computed once, at load, on the full-width rows (it reads columns 3, 4, 18..34).
BC_BANK_OBS_WIDTH = 1149
_RESTANCE_MOD = None


def restance_module():
    """``crab_hex_omni_restance`` (R, :class:`RestanceParams`, the branch names), loaded by path once."""
    global _RESTANCE_MOD
    if _RESTANCE_MOD is None:
        spec = importlib.util.spec_from_file_location("_krabby_crab_hex_omni_restance__ppo", RESTANCE_PY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _RESTANCE_MOD = mod
    return _RESTANCE_MOD


def anchor_sigma_from_state(state: dict) -> torch.Tensor:
    """The fixed ``sigma_A`` of a checkpoint's ``model_state_dict``: the std its policy samples with.

    ``std_raw`` (the ``soft`` type, every head of record since c6x) -> :func:`soft_std`; ``std`` (scalar)
    -> clamped to the stock [0.05, 2.0] band; ``log_std`` -> ``exp`` clamped the same way.
    """
    if "std_raw" in state:
        sigma = soft_std(state["std_raw"].float())
    elif "std" in state:
        sigma = state["std"].float().clamp(ACTION_STD_MIN, ACTION_STD_MAX)
    elif "log_std" in state:
        sigma = torch.exp(state["log_std"].float()).clamp(ACTION_STD_MIN, ACTION_STD_MAX)
    else:
        raise ValueError("anchor checkpoint: model_state_dict carries no std_raw / std / log_std")
    return sigma.detach().reshape(-1).clone()


def statue_row_losses(mu: torch.Tensor, sigma_a: torch.Tensor, mu_a: torch.Tensor | None = None) -> torch.Tensor:
    """Per row ``0.5 * sum_d ((mu - mu_a) / sigma_a,d)^2`` (``mu_a`` None -> the statue target 0)."""
    diff = mu if mu_a is None else mu - mu_a
    return 0.5 * (diff / sigma_a.reshape(1, -1)).square().sum(dim=-1)


def _detach_output(_module, _inputs, output):
    """A forward hook that returns its module's output detached (BC's priv-encoder latent)."""
    return output.detach()


def _safe_advantage_normalize(adv: torch.Tensor) -> torch.Tensor:
    """Finite normalization when global advantage std is 0 or non-finite."""
    warn_if_nonfinite("advantage_normalize.input", adv)
    adv = torch.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)
    std = adv.std()
    if torch.isfinite(std) and std > 1e-8:
        return (adv - adv.mean()) / (std + 1e-8)
    return adv - adv.mean()


def action_bound_loss(mu: torch.Tensor, bound: float) -> torch.Tensor:
    """The rl_games ``bound_loss`` on the raw Gaussian mean (M1b spec, 2026-09-23).

    ``(relu(mu - b)^2 + relu(-b - mu)^2)`` summed over action dims, averaged over rows: exactly zero
    inside ``[-b, b]``, quadratic outside, gradient ``2 (|mu| - b) sign(mu)`` there.
    """
    return (torch.relu(mu - bound).square() + torch.relu(-bound - mu).square()).sum(dim=-1).mean()


def _gaussian_kl(mu, sigma, old_mu, old_sigma) -> torch.Tensor:
    """Mean over rows of KL(old || new) for diagonal Gaussians -- the rsl_rl adaptive-schedule
    expression, verbatim (the adaptive path must stay bitwise what it was)."""
    kl = torch.sum(
        torch.log(sigma / old_sigma + 1.0e-5)
        + (torch.square(old_sigma) + torch.square(old_mu - mu)) / (2.0 * torch.square(sigma))
        - 0.5,
        axis=-1,
    )
    return torch.mean(kl)


def _explained_variance(returns: torch.Tensor, residual: torch.Tensor) -> float:
    """``1 - Var(returns - values) / Var(returns)`` (``residual`` = the raw advantages)."""
    var_r = torch.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0).var()
    if not bool(torch.isfinite(var_r)) or float(var_r) <= 1e-12:
        return float("nan")
    return float(1.0 - residual.var() / var_r)


def updates_through_iteration(iteration: int, dagger_update_freq: int) -> int:
    """``PPOWithExtractor.counter`` after iterations ``0..iteration`` of one uninterrupted run.

    ``learn_rl`` runs ``update`` every iteration and ``update_dagger`` on iterations with
    ``it % dagger_update_freq == 0``; both advance the counter. Used to rebuild the counter
    for checkpoints saved before it was stored in them.
    """
    if iteration < 0:
        return 0
    return (iteration + 1) + (iteration // dagger_update_freq + 1)


class PPOWithExtractor:
    """PPO with privileged estimator and RMA policy (ActorCriticRMA).

    This class is **not** a subclass of :class:`rsl_rl.algorithms.PPO`. Isaac Lab 5.x / rsl_rl 3.x
    changed ``PPO`` to require separate ``MLPModel`` actor and critic plus ``TensorDict`` rollouts.
    This implementation keeps the legacy ActorCritic + estimator stack and uses ``RolloutStorage``
    with observation keys ``policy`` and ``critic``.
    """

    policy: ActorCriticRMA

    #: Always-on ``PPO/`` telemetry. A class attribute, not a cfg field: the unit tests turn it off
    #: on one of two identically seeded runs to prove it moves no trained value and no RNG state.
    telemetry_enabled: bool = True

    def __init__(
        self,
        policy: ActorCriticRMA,
        estimator: nn.Module,
        estimator_paras: dict[str, Any],
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float | None = 0.01,
        device: str = "cpu",
        normalize_advantage_per_mini_batch: bool = False,
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        priv_reg_coef_schedual: list[float] | tuple[float, ...] = (0.0, 0.1, 2000.0, 3000.0),
        multi_gpu_cfg: dict | None = None,
        optimizer: str = "adam",
        action_bound_loss_coef: float = 0.0,
        action_bound: float = 1.1,
        action_bound_loss_ramp: list[float] | tuple[float, ...] | None = None,
        kl_stop: float | None = None,
        hist_encoder_learning_rate: float | None = None,
        hand_diag_vy_obs_col: int | None = None,
        anchor_coef: float = 0.0,
        anchor_ckpt: str | None = None,
        stand_prior_coef: float = 0.0,
        stand_mask_col: int | None = None,
        stand_prior_target: str | None = None,
        restance: dict | None = None,
        bc_coef: float = 0.0,
        bc_bank: list[str] | tuple[str, ...] | str | None = None,
        bc_batch_size: int = 2048,
        bc_seed: int = 0,
        bc_stand_mask: bool = False,
    ) -> None:
        """The six keywords after ``optimizer`` are the PPO-construction knobs (2026-10-01), all DEFAULT OFF:

        ``action_bound_loss_coef`` / ``action_bound`` / ``action_bound_loss_ramp``
            :func:`action_bound_loss` on the minibatch mean, weighted by :meth:`action_bound_coef`
            (``KRABBY_ACTION_BOUND_LOSS`` / ``_BOUND`` / ``_BOUND_RAMP``). Added to the loss only
            while the effective coefficient is > 0, so at 0 the update is bitwise the old one.
        ``kl_stop``
            ``schedule="fixed"`` only (``KRABBY_PPO_KL_STOP``): a minibatch KL above ``2 * kl_stop``
            stops the rest of the update at once (that minibatch takes no step); an epoch whose
            mean KL exceeds ``kl_stop`` skips the remaining epochs.
        ``hist_encoder_learning_rate``
            the history encoder's own Adam LR (``KRABBY_PPO_HIST_LR``); ``None`` = the configured
            ``learning_rate`` (the old behaviour). When set it is also re-written after a load.
        ``hand_diag_vy_obs_col``
            per-hand diagnostics (``KRABBY_PPO_HAND_DIAG=1``): the POLICY observation column of the
            body-frame lateral command (9 on the omni term); ``None`` = off.

        The last four are the p2r statue distillation (``P2R_STATUE_PLAN.md`` step 1, 2026-10-09), all
        DEFAULT OFF (``KRABBY_ANCHOR_W`` / ``KRABBY_ANCHOR_CKPT`` / ``KRABBY_STAND_PRIOR_W``):

        ``anchor_coef`` / ``stand_prior_coef``
            weights of two masked losses on the minibatch's action MEAN (never the std), split by the
            stand bit ``obs[:B, stand_mask_col] > 0.5``: the walking anchor on bit-0 rows,
            ``0.5 * sum_d ((mu - mu_A) / sigma_A,d)^2`` with ``mu_A`` a frozen copy of the actor loaded
            from ``anchor_ckpt`` (``hist_encoding=False``, forwarded on the same minibatch tensor); the
            statue prior on bit-1 rows, ``0.5 * sum_d (mu / sigma_A,d)^2``. Each is the SUM over its
            rows divided by the minibatch size B. ``sigma_A`` is the checkpoint's own std, fixed.
            Both 0 -> nothing is built or computed and the update is bitwise the old one.
        ``anchor_ckpt``
            the checkpoint (``model_state_dict``: ``actor.*`` and the std parameter); required when
            either weight is > 0.
        ``stand_mask_col``
            the POLICY observation column carrying the stand bit (15: ``KRABBY_OMNI_STAND_OBS=clock``).

        The last two are the p2r Z2 re-stance target (``P2R_Z2_PLAN.md`` step 1, 2026-10-10), DEFAULT OFF
        (``KRABBY_STAND_PRIOR_TARGET`` / ``KRABBY_RESTANCE``):

        ``stand_prior_target``
            ``"restance"``: the statue prior on bit-1 rows becomes ``0.5 * sum_d ((mu - R) / sigma_A,d)^2`` with
            ``R = restance_target(obs[:B], RestanceParams.from_dict(restance))[0]`` under ``no_grad`` -- the same
            mask, the same sum / B, the same ``sigma_A``. Needs ``stand_prior_coef > 0``. ``None`` = the target 0,
            Z1's path bit for bit.
        ``restance``
            R's constants (``RestanceParams.to_dict()``); given exactly when ``stand_prior_target`` is.

        The last four are the p2r Z2c clean-state cloning (``P2R_Z2C_PLAN.md`` step 1, 2026-10-10), DEFAULT OFF
        (``KRABBY_BC_W`` / ``KRABBY_BC_BANK`` / ``KRABBY_BC_BATCH``):

        ``bc_coef``
            weight of ``L_bc = mean_rows 0.5 * sum_d ((mu - R) / sigma_A,d)^2`` on ``bc_batch_size`` rows drawn (with
            replacement, from a PRIVATE ``torch.Generator`` seeded ``bc_seed``, so the global RNG is never touched)
            per minibatch from the bank. ``mu`` = ``policy.actor(rows, hist_encoding=False)`` -- the training-path
            forward PPO uses -- with the PRIV-ENCODER LATENT DETACHED (BC's gradient reaches the scan encoder and the
            backbone, never the latent the eval-path history encoder must match). ``R`` = ``restance_target`` of the
            full-width bank rows with this run's constants, computed once at load. Added after the statue terms,
            inside the global clip and the non-finite skip. Needs ``stand_prior_target="restance"`` (R's constants)
            and ``anchor_ckpt`` (``sigma_A``). 0 -> nothing is loaded or computed and the update is bitwise the old
            one.
        ``bc_bank``
            the bank ``.npz`` paths (each ``obs`` (N, 1149), stand bit ``obs[:, stand_mask_col] == 1`` on every
            row); the actor's training-path columns are kept on the device as plain attributes (no ``state_dict``).
        ``bc_batch_size`` / ``bc_seed``
            the rows per minibatch and the private generator's seed (its state rides in :meth:`resume_state`).

        The last one is the p2r Z2d stand-row input mask (``P2R_Z2D_PLAN.md`` step 1, 2026-10-11), DEFAULT OFF (set
        by ``KRABBY_OMNI_STAND_MASK=1`` beside ``KRABBY_BC_W``):

        ``bc_stand_mask``
            True applies ``crab_hex_omni_restance.stand_mask_obs`` -- the env's own stand-row mask (last action 0,
            proxy -0.5, current step + the real history frames) -- to every FULL-width bank row after R is computed
            and before the training-path trim, so the cloned rows look like what the masked env feeds the policy. R
            reads columns 3, 4, 18..34 only, so the targets are unchanged. Needs ``bc_coef > 0``.
        """
        if schedule not in SCHEDULES:
            raise ValueError(f"PPOWithExtractor: schedule must be one of {SCHEDULES}, got {schedule!r}")
        if schedule == "epoch_kl" and desired_kl is None:
            raise ValueError("PPOWithExtractor: schedule 'epoch_kl' needs a desired_kl")
        if kl_stop is not None:
            if schedule != "fixed":
                raise ValueError(f"PPOWithExtractor: kl_stop is the 'fixed' schedule's early stop, got "
                                 f"schedule {schedule!r}")
            if not (math.isfinite(float(kl_stop)) and float(kl_stop) > 0.0):
                raise ValueError(f"PPOWithExtractor: kl_stop must be > 0, got {kl_stop!r}")
        action_bound_loss_coef = float(action_bound_loss_coef)
        if not (math.isfinite(action_bound_loss_coef) and action_bound_loss_coef >= 0.0):
            raise ValueError(f"PPOWithExtractor: action_bound_loss_coef must be >= 0, got {action_bound_loss_coef!r}")
        if not (math.isfinite(float(action_bound)) and float(action_bound) > 0.0):
            raise ValueError(f"PPOWithExtractor: action_bound must be > 0, got {action_bound!r}")
        if action_bound_loss_ramp is not None:
            ramp = list(action_bound_loss_ramp)
            if len(ramp) != 3:
                raise ValueError(f"PPOWithExtractor: action_bound_loss_ramp is [c0, c1, n], got {ramp!r}")
            c0, c1, n = float(ramp[0]), float(ramp[1]), float(ramp[2])
            if not (c0 >= 0.0 and c1 > 0.0 and n >= 1.0 and n == int(n)):
                raise ValueError(f"PPOWithExtractor: action_bound_loss_ramp needs c0 >= 0, c1 > 0 and an "
                                 f"integer n >= 1, got {ramp!r}")
            if c1 != action_bound_loss_coef:
                raise ValueError(f"PPOWithExtractor: action_bound_loss_ramp ends at {c1}, but "
                                 f"action_bound_loss_coef is {action_bound_loss_coef}: the ramp's c1 IS the "
                                 "final coefficient")
            action_bound_loss_ramp = (c0, c1, int(n))
        if hist_encoder_learning_rate is not None and not float(hist_encoder_learning_rate) > 0.0:
            raise ValueError(f"PPOWithExtractor: hist_encoder_learning_rate must be > 0, got "
                             f"{hist_encoder_learning_rate!r}")
        anchor_coef, stand_prior_coef = float(anchor_coef), float(stand_prior_coef)
        for name, coef in (("anchor_coef", anchor_coef), ("stand_prior_coef", stand_prior_coef)):
            if not (math.isfinite(coef) and coef >= 0.0):
                raise ValueError(f"PPOWithExtractor: {name} must be >= 0, got {coef!r}")
        statue_armed = anchor_coef > 0.0 or stand_prior_coef > 0.0
        if statue_armed:
            if not anchor_ckpt:
                raise ValueError("PPOWithExtractor: anchor_coef / stand_prior_coef need anchor_ckpt (sigma_A and "
                                 "mu_A come from it)")
            if stand_mask_col is None or int(stand_mask_col) < 0:
                raise ValueError(f"PPOWithExtractor: anchor_coef / stand_prior_coef need a stand_mask_col >= 0, "
                                 f"got {stand_mask_col!r}")
        elif anchor_ckpt is not None:
            raise ValueError("PPOWithExtractor: anchor_ckpt is set but anchor_coef and stand_prior_coef are both 0")
        restance_params = None
        if stand_prior_target is not None:
            if stand_prior_target not in STAND_PRIOR_TARGETS:
                raise ValueError(f"PPOWithExtractor: stand_prior_target must be one of {STAND_PRIOR_TARGETS} "
                                 f"(or None), got {stand_prior_target!r}")
            if not stand_prior_coef > 0.0:
                raise ValueError(f"PPOWithExtractor: stand_prior_target={stand_prior_target!r} needs "
                                 f"stand_prior_coef > 0 (got {stand_prior_coef}): it would train nothing")
            if restance is None:
                raise ValueError(f"PPOWithExtractor: stand_prior_target={stand_prior_target!r} needs restance "
                                 "(R's constants)")
            restance_params = restance_module().RestanceParams.from_dict(dict(restance))
        elif restance is not None:
            raise ValueError("PPOWithExtractor: restance is set but stand_prior_target is not: the constants "
                             "would distil nothing")
        # NOTE(p2r Z2c, 2026-10-10): clean-state cloning (default off; see the docstring)
        bc_coef = float(bc_coef)
        if not (math.isfinite(bc_coef) and bc_coef >= 0.0):
            raise ValueError(f"PPOWithExtractor: bc_coef must be >= 0, got {bc_coef!r}")
        if isinstance(bc_bank, (str, Path)):
            bc_bank = [bc_bank]
        bc_bank = None if bc_bank is None else [str(b) for b in bc_bank]
        if bc_coef > 0.0:
            if stand_prior_target != "restance" or restance_params is None:
                raise ValueError("PPOWithExtractor: bc_coef > 0 needs stand_prior_target='restance' and restance "
                                 "(the bank's targets are R with this run's constants)")
            if not anchor_ckpt:
                raise ValueError("PPOWithExtractor: bc_coef > 0 needs anchor_ckpt (sigma_A comes from it)")
            if not bc_bank:
                raise ValueError("PPOWithExtractor: bc_coef > 0 needs bc_bank (one or more .npz paths)")
            if isinstance(bc_batch_size, bool) or int(bc_batch_size) != bc_batch_size or int(bc_batch_size) < 1:
                raise ValueError(f"PPOWithExtractor: bc_batch_size must be a positive integer, got {bc_batch_size!r}")
            if isinstance(bc_seed, bool) or int(bc_seed) != bc_seed:
                raise ValueError(f"PPOWithExtractor: bc_seed must be an integer, got {bc_seed!r}")
        elif bc_bank:
            raise ValueError("PPOWithExtractor: bc_bank is set but bc_coef is 0: the bank would train nothing")
        # NOTE(p2r Z2d, 2026-10-11): the bank rows' stand-row mask (default off)
        if not isinstance(bc_stand_mask, bool):
            raise ValueError(f"PPOWithExtractor: bc_stand_mask must be a bool, got {bc_stand_mask!r}")
        if bc_stand_mask and not bc_coef > 0.0:
            raise ValueError("PPOWithExtractor: bc_stand_mask is set but bc_coef is 0: there is no bank to mask")
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        if rnd_cfg:
            rnd_lr = rnd_cfg.pop("learning_rate", 1e-3)
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            self.rnd_optimizer = optim.Adam(self.rnd.predictor.parameters(), lr=rnd_lr)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        if symmetry_cfg is not None:
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            symmetry_cfg["data_augmentation_func"] = resolve_callable(symmetry_cfg["data_augmentation_func"])
            if not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    f"Symmetry configuration exists but the function is not callable: "
                    f"{symmetry_cfg['data_augmentation_func']}"
                )
            if policy.is_recurrent:
                raise ValueError("Symmetry augmentation is not supported for recurrent policies.")
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        self.policy = policy.to(self.device)
        self.estimator = estimator.to(self.device)
        self.optimizer = resolve_optimizer(optimizer)(self.policy.parameters(), lr=learning_rate)

        self.priv_states_dim = estimator_paras["num_priv_explicit"]
        self.num_prop = estimator_paras["num_prop"]
        self.num_scan = estimator_paras["num_scan"]
        self.estimator_optimizer = optim.Adam(
            self.estimator.parameters(), lr=estimator_paras["learning_rate"]
        )
        self.train_with_estimated_states = estimator_paras["train_with_estimated_states"]
        self.hist_encoder_learning_rate = (
            None if hist_encoder_learning_rate is None else float(hist_encoder_learning_rate)
        )
        self.hist_encoder_optimizer = optim.Adam(
            self.policy.actor.history_encoder.parameters(),
            lr=learning_rate if self.hist_encoder_learning_rate is None else self.hist_encoder_learning_rate,
        )
        self.priv_reg_coef_schedual = list(priv_reg_coef_schedual)
        self.counter = 0

        self.storage: RolloutStorage | None = None
        self.transition = RolloutStorage.Transition()

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.intrinsic_rewards: torch.Tensor | None = None

        # -- PPO-construction knobs (all default off; see the constructor docstring)
        self.action_bound_loss_coef = action_bound_loss_coef
        self.action_bound = float(action_bound)
        self.action_bound_loss_ramp = action_bound_loss_ramp
        self.kl_stop = None if kl_stop is None else float(kl_stop)
        self.hand_diag_vy_obs_col = None if hand_diag_vy_obs_col is None else int(hand_diag_vy_obs_col)
        # -- p2r statue distillation (default off; see the constructor docstring)
        self.anchor_coef = anchor_coef
        self.stand_prior_coef = stand_prior_coef
        self.anchor_ckpt = None if anchor_ckpt is None else str(anchor_ckpt)
        self.stand_mask_col = None if stand_mask_col is None else int(stand_mask_col)
        #: The frozen anchor actor (``anchor_coef > 0`` only) and the fixed ``sigma_A`` (either weight > 0).
        self.anchor_actor: nn.Module | None = None
        self.anchor_sigma: torch.Tensor | None = None
        #: NOTE(p2r Z2): the statue prior's target (None = 0) and R's validated constants.
        self.stand_prior_target = stand_prior_target
        self.restance_params = restance_params
        if statue_armed:
            self._build_anchor()
        #: NOTE(p2r Z2c): clean-state cloning. Armed only with ``bc_coef > 0``; the bank rows / targets are PLAIN
        #: attributes (no buffer, no parameter: in no ``state_dict``) and the generator is private.
        self.bc_coef = bc_coef
        self.bc_bank = bc_bank if bc_coef > 0.0 else None
        self.bc_batch_size = int(bc_batch_size) if bc_coef > 0.0 else None
        self.bc_seed = int(bc_seed) if bc_coef > 0.0 else None
        #: NOTE(p2r Z2d): the bank rows carry the env's stand-row mask (applied on load).
        self.bc_stand_mask = bc_stand_mask
        self.bc_obs: torch.Tensor | None = None
        self.bc_target: torch.Tensor | None = None
        self.bc_generator: torch.Generator | None = None
        if bc_coef > 0.0:
            self._load_bc_bank()
        #: The runner's training iteration of the coming :meth:`update` (set by the runner; read only by
        #: the per-hand diagnostic cadence). ``None`` = not driven by a runner.
        self.iteration: int | None = None
        #: ``PPO/`` telemetry of the last :meth:`update` (the runner logs it); empty before the first.
        self.telemetry: dict[str, float] = {}
        #: Explained variance of the last :meth:`compute_returns`, before advantage normalisation.
        self._explained_var: float | None = None
        if self.schedule != "adaptive":
            self.apply_configured_learning_rates()
        armed = {
            k: v for k, v in (
                ("schedule", schedule if schedule != "adaptive" else None),
                ("action_bound_loss_coef", action_bound_loss_coef or None),
                ("action_bound", self.action_bound if action_bound_loss_coef > 0.0 else None),
                ("action_bound_loss_ramp", action_bound_loss_ramp),
                ("kl_stop", self.kl_stop),
                ("hist_encoder_learning_rate", self.hist_encoder_learning_rate),
                ("hand_diag_vy_obs_col", self.hand_diag_vy_obs_col),
                ("anchor_coef", anchor_coef or None),
                ("stand_prior_coef", stand_prior_coef or None),
                ("anchor_ckpt", self.anchor_ckpt),
                ("stand_mask_col", self.stand_mask_col if statue_armed else None),
                ("stand_prior_target", self.stand_prior_target),
                ("restance", None if restance_params is None
                 else restance_module().format_restance(restance_params)),
                ("bc_coef", bc_coef or None),
                ("bc_bank_rows", None if self.bc_obs is None else int(self.bc_obs.shape[0])),
                ("bc_batch_size", self.bc_batch_size),
                ("bc_seed", self.bc_seed),
                ("bc_stand_mask", True if bc_stand_mask else None),
            ) if v is not None
        }
        if armed:
            # Only when a construction knob is armed (knob-off stdout is unchanged): the line that
            # proves the knobs reached the ALGORITHM object, not merely the cfg dump.
            print("[PPO] construction: " + ", ".join(f"{k}={v}" for k, v in armed.items())
                  + f", learning_rate={self.learning_rate}, desired_kl={self.desired_kl}", flush=True)

    # ------------------------------------------------------------------ p2r statue distillation
    def _build_anchor(self) -> None:
        """Load ``anchor_ckpt``: ``sigma_A`` always; with ``anchor_coef > 0`` also the frozen anchor actor.

        The anchor is a DEEP COPY of this policy's whole actor module (priv / scan / history encoders
        included) with the checkpoint's ``actor.*`` weights loaded strictly, every parameter
        ``requires_grad=False``, in ``eval()``. It is not a submodule of the policy, so it is in no
        optimizer, no ``state_dict`` and no checkpoint this run writes.
        """
        loaded = torch.load(self.anchor_ckpt, map_location=self.device, weights_only=False)
        state = loaded["model_state_dict"] if isinstance(loaded, dict) and "model_state_dict" in loaded else loaded
        self.anchor_sigma = anchor_sigma_from_state(state).to(self.device)
        n_act = int(self.policy.actor.num_actions) if hasattr(self.policy.actor, "num_actions") else None
        if n_act is not None and self.anchor_sigma.numel() != n_act:
            raise ValueError(f"anchor checkpoint: sigma_A has {self.anchor_sigma.numel()} dims, the policy {n_act}")
        if self.anchor_coef > 0.0:
            actor_state = {k[len("actor."):]: v for k, v in state.items() if k.startswith("actor.")}
            if not actor_state:
                raise ValueError(f"anchor checkpoint {self.anchor_ckpt}: no actor.* weights")
            anchor = copy.deepcopy(self.policy.actor)
            anchor.load_state_dict(actor_state, strict=True)
            for p in anchor.parameters():
                p.requires_grad_(False)
            self.anchor_actor = anchor.to(self.device).eval()

    def _statue_losses(self, obs_batch: torch.Tensor, mu_batch: torch.Tensor, batch_size: int) -> dict:
        """The two masked losses on one minibatch (call only when a statue weight is > 0).

        ``obs_batch`` is the FULL tensor the policy forward ran on (augmented rows included);
        ``mu_batch`` its first ``batch_size`` rows' means. The anchor forwards on that same full
        tensor and is sliced afterwards, so at the fork it reproduces ``mu_batch`` bit for bit.
        Returns ``{"bit", "anchor", "prior"}`` (a loss is None while its weight is 0); with the re-stance
        target (p2r Z2) also ``"restance_target"`` ``(B, 18)`` and ``"restance_branch"`` ``(B, 6)``, R on every
        original row (detached; only the bit-1 rows enter the prior).
        """
        col = self.stand_mask_col
        if col >= obs_batch.shape[-1]:
            raise ValueError(f"stand_mask_col {col} is outside the policy obs width {obs_batch.shape[-1]}")
        bit = obs_batch[:batch_size, col] > STAND_BIT_THRESHOLD
        out = {"bit": bit, "anchor": None, "prior": None}
        zero = mu_batch.new_zeros(())
        if self.anchor_coef > 0.0:
            with torch.no_grad():
                mu_a = self.anchor_actor(obs_batch, False)[:batch_size]
            k_anchor = statue_row_losses(mu_batch, self.anchor_sigma, mu_a.detach())
            out["anchor"] = torch.where(~bit, k_anchor, zero).sum() / batch_size
        if self.stand_prior_coef > 0.0:
            if self.stand_prior_target is None:
                k_prior = statue_row_losses(mu_batch, self.anchor_sigma)
            else:
                # NOTE(p2r Z2, 2026-10-10): the target R(obs) of the ORIGINAL rows (an augmented copy's bit is
                # never a stand row), under no_grad -- a fixed, observation-only target like the anchor's mu_A.
                with torch.no_grad():
                    target, branch = restance_module().restance_target(obs_batch[:batch_size], self.restance_params)
                out["restance_target"], out["restance_branch"] = target.detach(), branch
                k_prior = statue_row_losses(mu_batch, self.anchor_sigma, target.detach())
            out["prior"] = torch.where(bit, k_prior, zero).sum() / batch_size
        return out

    # ------------------------------------------------------------------ p2r Z2c clean-state cloning
    def _load_bc_bank(self) -> None:
        """Load every ``bc_bank`` file's ``obs``, validate it, compute R once on the full-width rows (this run's
        constants), keep the actor's training-path columns + the targets on the device, seed the private generator.

        Refused: a file without ``obs``, a width other than :data:`BC_BANK_OBS_WIDTH`, no rows, a non-finite value,
        or a row whose stand bit (``obs[:, stand_mask_col]``) is not exactly 1 (a bank of stand rows only).
        With ``bc_stand_mask`` (p2r Z2d) every full-width row gets the env's stand-row mask after R, before the trim.
        """
        actor = self.policy.actor
        try:
            cols = int(actor.num_prop) + int(actor.num_scan) + int(actor.num_priv_explicit) + int(actor.num_priv_latent)
        except AttributeError as exc:
            raise ValueError("PPOWithExtractor: bc_coef > 0 needs an Actor with num_prop / num_scan / "
                             "num_priv_explicit / num_priv_latent (the training-path slice)") from exc
        col = self.stand_mask_col
        rst = restance_module()
        obs_parts, tgt_parts = [], []
        for path in self.bc_bank:
            with np.load(path) as data:
                if "obs" not in data.files:
                    raise ValueError(f"BC bank {path}: no 'obs' array (has {sorted(data.files)})")
                obs_np = np.asarray(data["obs"])
            if obs_np.ndim != 2 or obs_np.shape[1] != BC_BANK_OBS_WIDTH:
                raise ValueError(f"BC bank {path}: obs must be (N, {BC_BANK_OBS_WIDTH}), got {tuple(obs_np.shape)}")
            if obs_np.shape[0] == 0:
                raise ValueError(f"BC bank {path}: no rows")
            if not (0 <= col < BC_BANK_OBS_WIDTH and cols <= BC_BANK_OBS_WIDTH):
                raise ValueError(f"BC bank {path}: stand_mask_col {col} / training-path width {cols} outside the "
                                 f"bank width {BC_BANK_OBS_WIDTH}")
            obs = torch.from_numpy(obs_np.astype(np.float32, copy=False))
            if not bool(torch.isfinite(obs).all()):
                raise ValueError(f"BC bank {path}: obs carries non-finite values")
            bad = int((obs[:, col] != 1.0).sum())
            if bad:
                raise ValueError(f"BC bank {path}: {bad} / {obs.shape[0]} rows have the stand bit "
                                 f"obs[:, {col}] != 1 (a BC bank holds stand rows only)")
            with torch.no_grad():
                target, _branch = rst.restance_target(obs, self.restance_params)
            if self.bc_stand_mask:
                # NOTE(p2r Z2d, 2026-10-11): the env's stand-row mask on every (stand) row, full width, after R
                # (which reads none of the masked columns) and before the trim
                rst.stand_mask_obs(obs, None, inplace=True)
            obs_parts.append(obs[:, :cols].clone())
            tgt_parts.append(target.float())
            del obs, obs_np
        self.bc_obs = torch.cat(obs_parts).to(self.device)
        self.bc_target = torch.cat(tgt_parts).to(self.device)
        n_act = int(self.anchor_sigma.numel())
        if self.bc_target.shape[1] != n_act:
            raise ValueError(f"BC: R has {self.bc_target.shape[1]} action dims, sigma_A {n_act}")
        self.bc_generator = torch.Generator(device=self.device)
        self.bc_generator.manual_seed(self.bc_seed)

    def _bc_policy_mean(self, obs: torch.Tensor) -> torch.Tensor:
        """``policy.actor(obs, hist_encoding=False)`` -- the training-path forward PPO runs, through the actor's own
        ``forward`` -- with the priv encoder's OUTPUT detached by a forward hook held only for this call (the PPO
        forward is never touched), so no BC gradient reaches the priv encoder."""
        actor = self.policy.actor
        handle = actor.priv_encoder.register_forward_hook(_detach_output)
        try:
            return actor(obs, False)
        finally:
            handle.remove()

    def _bc_loss(self) -> torch.Tensor:
        """One minibatch's ``L_bc`` (``bc_batch_size`` bank rows drawn from the PRIVATE generator)."""
        idx = torch.randint(self.bc_obs.shape[0], (self.bc_batch_size,), generator=self.bc_generator,
                            device=self.bc_obs.device)
        mu = self._bc_policy_mean(self.bc_obs[idx])
        return statue_row_losses(mu, self.anchor_sigma, self.bc_target[idx]).mean()

    def _actor_grad_norm(self, term: torch.Tensor) -> float:
        """``||d term / d actor params||`` (logging only: ``autograd.grad``, graph retained, no ``.grad``)."""
        params = [p for p in self.policy.actor.parameters() if p.requires_grad]
        grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        sq = sum(float(g.detach().square().sum()) for g in grads if g is not None)
        return math.sqrt(sq)

    # ------------------------------------------------------------------ construction knobs
    def action_bound_coef(self) -> float:
        """The bound loss's effective coefficient at this update.

        Without a ramp: ``action_bound_loss_coef``. With ``[c0, c1, n]``: linear from ``c0`` to ``c1``
        over the first ``n`` updates, keyed on :attr:`counter` (which ``update`` AND ``update_dagger``
        advance, and which a resume restores) -- so a resumed stage continues the ramp, never restarts it.
        """
        if self.action_bound_loss_ramp is None:
            return self.action_bound_loss_coef
        c0, c1, n = self.action_bound_loss_ramp
        frac = min(max(self.counter, 0) / n, 1.0)
        return c0 + (c1 - c0) * frac

    def _write_policy_lr(self) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

    def apply_configured_learning_rates(self) -> None:
        """Write the schedule's LR into the optimizers after construction or a checkpoint load.

        ``fixed`` / ``epoch_kl``: :attr:`learning_rate` into EVERY policy param group (a loaded
        optimizer state carries the SAVED run's LR -- an adaptive head's floor, say -- which neither
        schedule would ever overwrite on its own). ``adaptive`` is left alone: its first minibatch
        writes the LR anyway, exactly as before. An explicit ``hist_encoder_learning_rate`` is
        re-written into the history-encoder optimizer for the same reason; unset, its loaded LR is
        kept (the old behaviour).
        """
        if self.schedule != "adaptive":
            self._write_policy_lr()
        if self.hist_encoder_learning_rate is not None:
            for param_group in self.hist_encoder_optimizer.param_groups:
                param_group["lr"] = self.hist_encoder_learning_rate

    def _epoch_kl_step(self, epoch_kl: float) -> None:
        """``epoch_kl``: the adaptive /1.5 or *1.5 rule, once, on one epoch's mean KL."""
        if self.gpu_global_rank == 0:
            if epoch_kl > self.desired_kl * 2.0:
                self.learning_rate = max(LR_FLOOR, self.learning_rate / 1.5)
            elif 0.0 < epoch_kl < self.desired_kl / 2.0:
                self.learning_rate = min(LR_CAP, self.learning_rate * 1.5)
        if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()
        self._write_policy_lr()

    # ------------------------------------------------------------------ telemetry (RNG-free)
    def _stored_rows(self, tensor: torch.Tensor) -> torch.Tensor:
        """The fixed telemetry slice: the first ``min(4096, T * N)`` rows of a ``[T, N, ...]`` buffer."""
        flat = tensor.reshape(-1, *tensor.shape[2:])
        return flat[: min(TELEMETRY_SLICE_ROWS, flat.shape[0])]

    def _slice_obs(self) -> torch.Tensor:
        obs = self._stored_rows(self.storage.observations["policy"])
        return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _policy_mean(self, obs: torch.Tensor) -> torch.Tensor:
        """The update's own forward (``act(obs, hist_encoding=False)``'s distribution) minus the
        sample; the policy's ``distribution`` attribute is restored afterwards."""
        saved = self.policy.distribution
        try:
            self.policy.update_distribution(obs, False)
            return self.policy.action_mean
        finally:
            self.policy.distribution = saved

    def _executed_mean(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._policy_mean(obs).clamp(-EXEC_ACTION_CLIP, EXEC_ACTION_CLIP)

    @staticmethod
    def _saturation(mu: torch.Tensor, sigma: torch.Tensor, suffix: str, out: dict) -> None:
        a = mu.abs()
        out[f"sat1_{suffix}"] = (a > 1.0).float().mean().item()
        out[f"sat15_{suffix}"] = (a > 1.5).float().mean().item()
        out[f"mu_med_{suffix}"] = a.median().item()
        out[f"lock_{suffix}"] = (a > 1.0 + 2.0 * sigma).float().mean().item()

    def _rollout_telemetry(self, out: dict) -> None:
        """Saturation of the ROLLOUT means, executed action rate, effective std, explained variance."""
        st = self.storage
        with torch.no_grad():
            if st.distribution_params is not None:
                mu = st.distribution_params[0].reshape(-1, st.distribution_params[0].shape[-1])
                sigma = st.distribution_params[1].reshape(-1, st.distribution_params[1].shape[-1])
                if mu.shape[-1] == CRAB_ACTION_DIM:
                    self._saturation(mu[:, :CRAB_CAM_DIMS], sigma[:, :CRAB_CAM_DIMS], "cam", out)
                    self._saturation(mu[:, CRAB_CAM_DIMS:], sigma[:, CRAB_CAM_DIMS:], "rod", out)
                else:
                    self._saturation(mu, sigma, "all", out)
                out["effective_std"] = sigma.mean().item()
            if st.num_transitions_per_env > 1:
                executed = st.actions.clamp(-EXEC_ACTION_CLIP, EXEC_ACTION_CLIP)
                step = (executed[1:] - executed[:-1]).norm(dim=-1)              # [T-1, N]
                same_episode = st.dones[:-1].reshape(step.shape) == 0          # a_t is not a reset's first action
                if bool(same_episode.any()):
                    out["exec_action_rate"] = step[same_episode].mean().item()
        if self._explained_var is not None:
            out["explained_var"] = self._explained_var

    def _hand_surrogate_grad(self, obs, actions, old_logp, adv, params) -> torch.Tensor:
        """Flat gradient of the clipped PPO surrogate on one subset w.r.t. ``params``
        (``torch.autograd.grad``: no ``.grad`` buffer, no optimizer state, no sampling)."""
        saved = self.policy.distribution
        try:
            with torch.enable_grad():
                self.policy.update_distribution(obs, False)
                logp = self.policy.get_actions_log_prob(actions)
                ratio = torch.exp(torch.clamp(logp - torch.squeeze(old_logp, -1), min=-20.0, max=20.0))
                a = torch.squeeze(adv, -1)
                surrogate = torch.max(
                    -a * ratio, -a * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                ).mean()
                grads = torch.autograd.grad(surrogate, params, allow_unused=True)
        finally:
            self.policy.distribution = saved
        return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                          for g, p in zip(grads, params)])

    def _hand_diag(self, out: dict) -> None:
        """Per-hand surrogate-gradient cosine on the shared actor backbone (``KRABBY_PPO_HAND_DIAG``).

        On the fixed telemetry slice, at the rollout parameters (before any step of this update):
        rows whose observed lateral command is above ``+0.05`` (left hand, +y) vs below ``-0.05``
        (right hand). Leaves parameters, ``.grad`` buffers, optimizer state and RNG state untouched.
        """
        col = self.hand_diag_vy_obs_col
        obs = self._slice_obs()
        if col >= obs.shape[-1]:
            raise ValueError(f"hand diagnostics: obs column {col} is outside the policy obs width {obs.shape[-1]}")
        st = self.storage
        actions = self._stored_rows(st.actions)
        old_logp = self._stored_rows(st.actions_log_prob)
        adv = self._stored_rows(st.advantages)
        vy = obs[:, col]
        left, right = vy > HAND_DIAG_DEADZONE, vy < -HAND_DIAG_DEADZONE
        out["hand_frac_l"] = left.float().mean().item()
        out["hand_frac_r"] = right.float().mean().item()
        if bool(left.any()):
            out["hand_adv_mean_l"] = adv[left].mean().item()
        if bool(right.any()):
            out["hand_adv_mean_r"] = adv[right].mean().item()
        if not (bool(left.any()) and bool(right.any())):
            return
        backbone = getattr(self.policy.actor, "actor_backbone", self.policy.actor)
        params = [p for p in backbone.parameters() if p.requires_grad]
        g_l = self._hand_surrogate_grad(obs[left], actions[left], old_logp[left], adv[left], params)
        g_r = self._hand_surrogate_grad(obs[right], actions[right], old_logp[right], adv[right], params)
        n_l, n_r = g_l.norm(), g_r.norm()
        if float(n_l) > 0.0 and float(n_r) > 0.0:
            out["hand_grad_cos"] = float(torch.dot(g_l, g_r) / (n_l * n_r))

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs_shapes: list[int],
        critic_obs_shapes: list[int],
        actions_shape: list[int],
    ) -> None:
        num_obs = obs_shapes[0]
        num_critic_obs = critic_obs_shapes[0]
        obs_template = TensorDict(
            {
                "policy": torch.zeros(num_envs, num_obs, device=self.device),
                "critic": torch.zeros(num_envs, num_critic_obs, device=self.device),
            },
            batch_size=[num_envs],
            device=self.device,
        )
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs_template,
            actions_shape,
            self.device,
        )

    def train_mode(self) -> None:
        self.policy.train()
        self.estimator.train()
        if self.rnd:
            self.rnd.train()

    def eval_mode(self) -> None:
        self.policy.eval()
        self.estimator.eval()
        if self.rnd:
            self.rnd.eval()

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor, hist_encoding: bool = False) -> torch.Tensor:
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        if self.train_with_estimated_states:
            obs_est = obs.clone()
            priv_states_estimated = self.estimator(obs_est[:, : self.num_prop])
            obs_est[
                :,
                self.num_prop
                + self.num_scan : self.num_prop
                + self.num_scan
                + self.priv_states_dim,
            ] = priv_states_estimated
            self.transition.actions = self.policy.act(obs_est, hist_encoding).detach()
        else:
            self.transition.actions = self.policy.act(obs, hist_encoding).detach()

        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.distribution_params = (
            self.policy.action_mean.detach(),
            self.policy.action_std.detach(),
        )
        self.transition.observations = TensorDict(
            {"policy": obs, "critic": critic_obs},
            batch_size=[obs.shape[0]],
            device=self.device,
        )
        return self.transition.actions

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: dict,
    ) -> None:
        if self.storage is None:
            raise RuntimeError("Rollout storage is not initialized; call init_storage first.")

        if self.rnd:
            self.rnd.update_normalization(obs)

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            self.transition.rewards = self.transition.rewards + self.intrinsic_rewards
        else:
            self.intrinsic_rewards = None

        if "time_outs" in infos:
            self.transition.rewards = self.transition.rewards + self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs: torch.Tensor) -> None:
        if self.storage is None:
            raise RuntimeError("Rollout storage is not initialized; call init_storage first.")
        st = self.storage
        last_values_raw = self.policy.evaluate(last_critic_obs).detach()
        warn_if_nonfinite("compute_returns.last_values", last_values_raw)
        last_values = torch.nan_to_num(
            last_values_raw,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            warn_if_nonfinite("compute_returns.delta", delta)
            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        advantages_raw = st.returns - st.values
        warn_if_nonfinite("compute_returns.advantages_raw", advantages_raw)
        st.advantages = torch.nan_to_num(advantages_raw, nan=0.0, posinf=0.0, neginf=0.0)
        if self.telemetry_enabled:
            # PPO/explained_var: the critic's fit to THIS rollout's returns, before normalisation.
            self._explained_var = _explained_variance(st.returns, st.advantages)
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = _safe_advantage_normalize(st.advantages)

    def update(self) -> dict[str, float]:  # noqa: C901
        if self.storage is None:
            raise RuntimeError("Rollout storage is not initialized; call init_storage first.")

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_priv_reg_loss = 0.0
        mean_entropy = 0.0
        mean_estimator_loss = 0.0
        if self.rnd:
            mean_rnd_loss = 0.0
        else:
            mean_rnd_loss = None
        if self.symmetry:
            mean_symmetry_loss = 0.0
        else:
            mean_symmetry_loss = None

        # -- PPO-construction state (telemetry + knobs). Nothing below draws a random number, and
        #    every knob at its default leaves the loss, the step and the LR exactly as they were.
        tel = self.telemetry_enabled
        self.telemetry = {}
        telemetry = self.telemetry
        bound_armed = self.action_bound_loss_coef > 0.0
        bound_coef = self.action_bound_coef() if bound_armed else 0.0
        mean_bound_loss = 0.0
        kl_sum, kl_n, kl_mb0 = 0.0, 0, None
        epoch_kl_sum, epoch_kl_n, epochs_done = 0.0, 0, 0
        clip_frac_sum, n_ratio = 0.0, 0
        lr_sum, grad_norm_sum, n_grad_clipped, n_steps, n_skipped = 0.0, 0.0, 0, 0, 0
        n_completed = 0
        kl_stopped = False
        exec_obs = exec_before = None
        if tel:
            self._rollout_telemetry(telemetry)
            exec_obs = self._slice_obs()
            exec_before = self._executed_mean(exec_obs)
        diag_step = self.counter if self.iteration is None else self.iteration
        if self.hand_diag_vy_obs_col is not None and diag_step % HAND_DIAG_EVERY == 0:
            self._hand_diag(telemetry)
        # -- p2r statue distillation (default off: nothing below runs and the loss is untouched)
        statue_armed = self.anchor_coef > 0.0 or self.stand_prior_coef > 0.0
        statue_split = statue_armed and diag_step % STATUE_GRAD_SPLIT_EVERY == 0
        st_anchor_sum = st_prior_sum = st_anchor_rows = st_prior_rows = 0.0
        st_n = 0
        st_anchor_mb0 = None
        st_split: dict[str, float] = {}
        # p2r Z2: pooled counts over the minibatches that took a step -- bit-1 rows, those with a non-zero R
        # target, and bit-1 leg-rows per R branch
        rst_rows = rst_nonzero = rst_legs = 0
        rst_branch = [0] * len(restance_module().BRANCH_NAMES) if self.stand_prior_target is not None else []
        # p2r Z2c clean-state cloning (default off: nothing below runs, no generator is drawn)
        bc_armed = self.bc_coef > 0.0
        bc_sum, bc_n, bc_mb0 = 0.0, 0, None
        st_prior_mb0 = None               # (logged only beside BC: the prior's own pre-step value, P2R_Z2C_PLAN)

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        priv_reg_coef = 0.0

        for mb_idx, batch in enumerate(generator):
            obs_batch = batch.observations["policy"]
            critic_obs_batch = batch.observations["critic"]
            actions_batch = batch.actions
            target_values_batch = batch.values
            advantages_batch = batch.advantages
            returns_batch = batch.returns
            old_actions_log_prob_batch = batch.old_actions_log_prob
            old_mu_batch = batch.old_distribution_params[0]
            old_sigma_batch = batch.old_distribution_params[1]
            hid_states_batch = batch.hidden_states
            masks_batch = batch.masks

            original_batch_size = batch.observations.batch_size[0]
            num_aug = 1

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = _safe_advantage_normalize(advantages_batch)

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                )
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            warn_if_nonfinite("update.obs_policy", obs_batch)
            warn_if_nonfinite("update.obs_critic", critic_obs_batch)
            obs_batch = torch.nan_to_num(obs_batch, nan=0.0, posinf=0.0, neginf=0.0)
            critic_obs_batch = torch.nan_to_num(
                critic_obs_batch, nan=0.0, posinf=0.0, neginf=0.0
            )

            self.policy.act(obs_batch, hist_encoding=False, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            priv_latent_batch = self.policy.actor.infer_priv_latent(obs_batch)
            with torch.inference_mode():
                hist_latent_batch = self.policy.actor.infer_hist_latent(obs_batch)
            priv_reg_loss = (priv_latent_batch - hist_latent_batch.detach()).norm(p=2, dim=1).mean()
            priv_reg_coef = self.priv_reg_coef()

            priv_states_predicted = self.estimator(obs_batch[:, : self.num_prop])
            estimator_loss = (
                priv_states_predicted
                - obs_batch[
                    :,
                    self.num_prop
                    + self.num_scan : self.num_prop
                    + self.num_scan
                    + self.priv_states_dim,
                ]
            ).pow(2).mean()
            self.estimator_optimizer.zero_grad()
            if torch.isfinite(estimator_loss).all():
                estimator_loss.backward()
                nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
                self.estimator_optimizer.step()
            else:
                self.estimator_optimizer.zero_grad(set_to_none=True)

            # -- this minibatch's Gaussian KL (rollout -> current), under EVERY schedule: the adaptive
            #    rule, the fixed schedule's early stop, epoch_kl and PPO/kl_* all read it. Same
            #    expression and the same inference-mode tensor the adaptive rule always compared.
            with torch.inference_mode():
                kl_mean = _gaussian_kl(mu_batch, sigma_batch, old_mu_batch, old_sigma_batch)

                if self.is_multi_gpu:
                    torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                    kl_mean /= self.gpu_world_size

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            kl_value = float(kl_mean)
            kl_sum += kl_value
            kl_n += 1
            if kl_mb0 is None:
                kl_mb0 = kl_value                   # minibatch 0: before any step of this update
            epoch_kl_sum += kl_value
            epoch_kl_n += 1
            if self.kl_stop is not None and kl_value > 2.0 * self.kl_stop:
                # fixed schedule, hard stop: this minibatch takes no step and the update ends here
                kl_stopped = True
                break

            log_ratio = actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
            ratio = torch.exp(log_ratio)
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            if tel:
                with torch.no_grad():
                    clip_frac_sum += ((ratio - 1.0).abs() > self.clip_param).float().mean().item()
                n_ratio += 1

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + priv_reg_coef * priv_reg_loss
            )

            # -- action-bound loss (KRABBY_ACTION_BOUND_LOSS): on the ORIGINAL rows' raw mean, and
            #    added only while its effective coefficient is > 0 (off -> the loss above, bitwise).
            bound_loss = None
            if bound_armed:
                bound_loss = action_bound_loss(mu_batch, self.action_bound)
                if bound_coef > 0.0:
                    loss = loss + bound_coef * bound_loss

            # -- p2r statue distillation (KRABBY_ANCHOR_W / KRABBY_STAND_PRIOR_W): after the bound loss
            #    and BEFORE the symmetry block reassigns ``obs_batch``, on the full minibatch, then masked
            #    by the stand bit. Each term is added only while its weight is > 0.
            statue = None
            statue_part = None
            if statue_armed:
                statue = self._statue_losses(obs_batch, mu_batch, original_batch_size)
                for coef, term in ((self.anchor_coef, statue["anchor"]), (self.stand_prior_coef, statue["prior"])):
                    if term is not None and coef > 0.0:
                        statue_part = coef * term if statue_part is None else statue_part + coef * term
                if statue_part is not None:
                    loss = loss + statue_part
                if statue["anchor"] is not None and st_anchor_mb0 is None:
                    st_anchor_mb0 = float(statue["anchor"])   # the iteration's first minibatch (fork: 0)

            # -- p2r Z2c clean-state cloning (KRABBY_BC_W): after the statue terms, inside the global clip and the
            #    non-finite skip below. Its rows come from the bank (private generator), not from this minibatch.
            bc_loss = None
            bc_part = None
            if bc_armed:
                bc_loss = self._bc_loss()
                bc_part = self.bc_coef * bc_loss
                loss = loss + bc_part
                if bc_mb0 is None:
                    bc_mb0 = float(bc_loss)                   # the first minibatch, before any step of this update
                    if statue is not None and statue["prior"] is not None:
                        st_prior_mb0 = float(statue["prior"])  # the prior on that minibatch, before BC's first step

            symmetry_loss = None
            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                    )
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                )

                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:],
                    actions_mean_symm_batch.detach()[original_batch_size:],
                )
                if self.symmetry["use_mirror_loss"]:
                    loss = loss + self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            rnd_loss = None
            if self.rnd:
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                rnd_loss = torch.nn.functional.mse_loss(predicted_embedding, target_embedding)

            if statue_split and not st_split and statue_part is not None and torch.isfinite(loss).all():
                # logging only (autograd.grad, graph retained): the WEIGHTED terms' actor-gradient norms,
                # on the iteration's first minibatch -- the dominance read of P2R_STATUE_PLAN step 4
                if statue["prior"] is not None:
                    st_split["grad_norm_statue"] = self._actor_grad_norm(self.stand_prior_coef * statue["prior"])
                if statue["anchor"] is not None:
                    st_split["grad_norm_anchor"] = self._actor_grad_norm(self.anchor_coef * statue["anchor"])
                if bc_part is not None:
                    st_split["grad_norm_bc"] = self._actor_grad_norm(bc_part)
                st_split["grad_norm_rest"] = self._actor_grad_norm(
                    loss - statue_part if bc_part is None else loss - statue_part - bc_part)

            self.optimizer.zero_grad()
            if torch.isfinite(loss).all():
                loss.backward()

                if self.rnd and rnd_loss is not None:
                    self.rnd_optimizer.zero_grad()
                    rnd_loss.backward()

                if self.is_multi_gpu:
                    self.reduce_parameters()

                grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                if tel:
                    grad_norm_value = float(grad_norm)          # the PRE-clip total norm
                    lr_sum += float(self.optimizer.param_groups[0]["lr"])
                    grad_norm_sum += grad_norm_value
                    n_grad_clipped += int(grad_norm_value > self.max_grad_norm)
                    n_steps += 1
                self.optimizer.step()

                if self.rnd_optimizer:
                    self.rnd_optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_entropy += entropy_batch.mean().item()
                mean_priv_reg_loss += priv_reg_loss.mean().item()
                if bound_loss is not None:
                    mean_bound_loss += bound_loss.item()
                if statue is not None:
                    n_rows = float(statue["bit"].numel())
                    n_stand = float(statue["bit"].sum())
                    st_prior_rows += n_stand / n_rows
                    st_anchor_rows += (n_rows - n_stand) / n_rows
                    if statue["anchor"] is not None:
                        st_anchor_sum += statue["anchor"].item()
                    if statue["prior"] is not None:
                        st_prior_sum += statue["prior"].item()
                    if "restance_target" in statue:
                        with torch.no_grad():
                            bit = statue["bit"]
                            rst_rows += int(bit.sum())
                            big = statue["restance_target"].abs() > RESTANCE_NONZERO_EPS
                            rst_nonzero += int((big.any(dim=-1) & bit).sum())
                            br = statue["restance_branch"][bit]
                            rst_legs += br.numel()
                            counts = torch.bincount(br.reshape(-1), minlength=len(rst_branch)).tolist()
                            rst_branch = [a + b for a, b in zip(rst_branch, counts)]
                    st_n += 1
                if bc_loss is not None:
                    bc_sum += bc_loss.item()
                    bc_n += 1
            else:
                self.optimizer.zero_grad(set_to_none=True)
                n_skipped += 1

            mean_estimator_loss += estimator_loss.item() if torch.isfinite(estimator_loss).all() else 0.0

            if mean_rnd_loss is not None and rnd_loss is not None and torch.isfinite(rnd_loss).all():
                mean_rnd_loss += rnd_loss.item()
            if (
                mean_symmetry_loss is not None
                and symmetry_loss is not None
                and torch.isfinite(symmetry_loss).all()
            ):
                mean_symmetry_loss += symmetry_loss.item()

            # -- end of an epoch: epoch_kl's one LR decision, the fixed schedule's per-epoch stop
            n_completed += 1
            if (mb_idx + 1) % self.num_mini_batches == 0:
                epochs_done += 1
                epoch_kl = epoch_kl_sum / max(epoch_kl_n, 1)
                epoch_kl_sum, epoch_kl_n = 0.0, 0
                if self.schedule == "epoch_kl":
                    self._epoch_kl_step(epoch_kl)
                if (self.kl_stop is not None and epoch_kl > self.kl_stop
                        and epochs_done < self.num_learning_epochs):
                    kl_stopped = True
                    break

        num_updates = self.num_learning_epochs * self.num_mini_batches
        if kl_stopped:
            num_updates = max(n_completed, 1)       # the minibatches that actually ran
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_priv_reg_loss /= num_updates
        mean_entropy /= num_updates
        mean_estimator_loss /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        if tel:
            exec_after = self._executed_mean(exec_obs)
            telemetry["exec_change"] = (exec_after - exec_before).abs().mean().item()
            if kl_n:
                telemetry["kl_mean"] = kl_sum / kl_n
                telemetry["kl_mb0"] = kl_mb0
            if n_ratio:
                telemetry["clip_frac"] = clip_frac_sum / n_ratio
            if n_steps:
                telemetry["lr_applied"] = lr_sum / n_steps
                telemetry["grad_norm"] = grad_norm_sum / n_steps
                telemetry["grad_clip_share"] = n_grad_clipped / n_steps
            telemetry["skipped_updates"] = float(n_skipped)
            if self.hist_encoder_learning_rate is not None:
                # the history-encoder optimizer's OWN LR (the 4a wiring read: pinned, never the policy's)
                telemetry["hist_lr"] = float(self.hist_encoder_optimizer.param_groups[0]["lr"])
            if self.kl_stop is not None:
                telemetry["kl_stop"] = float(kl_stopped)
                telemetry["epochs_done"] = float(epochs_done)

        self.storage.clear()
        self.update_counter()
        loss_dict: dict[str, float] = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "priv_reg": mean_priv_reg_loss,
            "entropy": mean_entropy,
            "estimator": mean_estimator_loss,
            "priv_reg_coef": float(priv_reg_coef),
        }
        if self.rnd and mean_rnd_loss is not None:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry and mean_symmetry_loss is not None:
            loss_dict["symmetry"] = mean_symmetry_loss
        if bound_armed:
            loss_dict["action_bound"] = mean_bound_loss / num_updates
            loss_dict["action_bound_coef"] = float(bound_coef)
        if statue_armed:
            # ``Loss/<key>`` in the runner's log: means over the minibatches that took a step
            n = max(st_n, 1)
            if self.anchor_coef > 0.0:
                loss_dict["anchor"] = st_anchor_sum / n
                if st_anchor_mb0 is not None:
                    loss_dict["anchor_mb0"] = st_anchor_mb0
            if self.stand_prior_coef > 0.0:
                loss_dict["stand_prior"] = st_prior_sum / n
            loss_dict["anchor_rows"] = st_anchor_rows / n
            loss_dict["prior_rows"] = st_prior_rows / n
            if self.stand_prior_target is not None:
                # ``Loss/restance_nonzero`` / ``Loss/restance_<branch>`` (P2R_Z2_PLAN step 4's +200 wiring); 0.0 when
                # no bit-1 row was seen
                names = restance_module().BRANCH_NAMES
                loss_dict["restance_nonzero"] = rst_nonzero / rst_rows if rst_rows else 0.0
                for k, name in enumerate(names):
                    loss_dict[f"restance_{name}"] = rst_branch[k] / rst_legs if rst_legs else 0.0
            if bc_armed:
                # ``Loss/bc`` (mean over the minibatches that took a step) and ``Loss/bc_mb0`` (the update's first
                # minibatch, before any step: P2R_Z2C_PLAN step 4's +200 dose read)
                loss_dict["bc"] = bc_sum / max(bc_n, 1)
                if bc_mb0 is not None:
                    loss_dict["bc_mb0"] = bc_mb0
                if st_prior_mb0 is not None:
                    # ``Loss/stand_prior_mb0``: the statue prior's first-minibatch value (``Loss/stand_prior`` is a
                    # mean over the update, which BC's pre-fit inside the first update moves)
                    loss_dict["stand_prior_mb0"] = st_prior_mb0
            loss_dict.update(st_split)
        return loss_dict

    def priv_reg_coef(self) -> float:
        """``priv_reg_coef_schedual`` = [start, end, begin, length], in updates (``counter``)."""
        sched = self.priv_reg_coef_schedual
        if len(sched) >= 4:
            priv_reg_stage = min(
                max((self.counter - sched[2]), 0) / sched[3],
                1.0,
            )
            return priv_reg_stage * (sched[1] - sched[0]) + sched[0]
        return sched[0] if sched else 0.0

    def update_counter(self) -> None:
        self.counter += 1

    def resume_state(self) -> dict[str, Any]:
        """Training state a resumed run needs besides the networks and the policy optimizer.

        ``counter`` drives the ``priv_reg`` coefficient schedule (and the action-bound ramp);
        ``learning_rate`` is the adaptive / epoch_kl value (adaptive overwrites the policy
        optimizer's param groups with it on the first minibatch after a resume; epoch_kl's is
        written by :meth:`apply_configured_learning_rates`, which the runner calls after a load).
        """
        state = {
            "counter": self.counter,
            "learning_rate": self.learning_rate,
            "estimator_optimizer": self.estimator_optimizer.state_dict(),
            "hist_encoder_optimizer": self.hist_encoder_optimizer.state_dict(),
        }
        if self.bc_generator is not None:
            # NOTE(p2r Z2c): the private BC generator (armed only), so a resume continues its draw sequence
            state["bc_generator"] = self.bc_generator.get_state()
        return state

    def load_resume_state(self, state: dict[str, Any]) -> None:
        self.counter = int(state["counter"])
        if self.schedule in ("adaptive", "epoch_kl"):
            self.learning_rate = float(state["learning_rate"])
        if "estimator_optimizer" in state:
            self.estimator_optimizer.load_state_dict(state["estimator_optimizer"])
        if "hist_encoder_optimizer" in state:
            self.hist_encoder_optimizer.load_state_dict(state["hist_encoder_optimizer"])
        if self.bc_generator is not None and state.get("bc_generator") is not None:
            # (a fork from a checkpoint without BC keeps the freshly seeded generator)
            self.bc_generator.set_state(state["bc_generator"].cpu())

    def update_dagger(self) -> float:
        if self.storage is None:
            raise RuntimeError("Rollout storage is not initialized; call init_storage first.")

        mean_hist_latent_loss = 0.0
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            obs_batch = batch.observations["policy"]
            hid_states_batch = batch.hidden_states
            masks_batch = batch.masks
            with torch.inference_mode():
                self.policy.act(
                    obs_batch,
                    hist_encoding=True,
                    masks=masks_batch,
                    hidden_states=hid_states_batch[0],
                )

            with torch.inference_mode():
                priv_latent_batch = self.policy.actor.infer_priv_latent(obs_batch)
            hist_latent_batch = self.policy.actor.infer_hist_latent(obs_batch)
            hist_latent_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(p=2, dim=1).mean()
            self.hist_encoder_optimizer.zero_grad()
            hist_latent_loss.backward()
            nn.utils.clip_grad_norm_(
                self.policy.actor.history_encoder.parameters(), self.max_grad_norm
            )
            self.hist_encoder_optimizer.step()
            mean_hist_latent_loss += hist_latent_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_hist_latent_loss /= num_updates
        self.storage.clear()
        self.update_counter()
        return mean_hist_latent_loss

    def broadcast_parameters(self) -> None:
        model_params = [self.policy.state_dict(), self.estimator.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])
        self.estimator.load_state_dict(model_params[1])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[2])

    def reduce_parameters(self) -> None:
        all_params = list(self.policy.parameters()) + list(self.estimator.parameters())
        if self.rnd:
            all_params.extend(list(self.rnd.parameters()))
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
