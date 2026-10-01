
from __future__ import annotations

import math

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
    ) -> None:
        """The six trailing keywords are the PPO-construction knobs (2026-10-01), all DEFAULT OFF:

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
            ) if v is not None
        }
        if armed:
            # Only when a construction knob is armed (knob-off stdout is unchanged): the line that
            # proves the knobs reached the ALGORITHM object, not merely the cfg dump.
            print("[PPO] construction: " + ", ".join(f"{k}={v}" for k, v in armed.items())
                  + f", learning_rate={self.learning_rate}, desired_kl={self.desired_kl}", flush=True)

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
        return {
            "counter": self.counter,
            "learning_rate": self.learning_rate,
            "estimator_optimizer": self.estimator_optimizer.state_dict(),
            "hist_encoder_optimizer": self.hist_encoder_optimizer.state_dict(),
        }

    def load_resume_state(self, state: dict[str, Any]) -> None:
        self.counter = int(state["counter"])
        if self.schedule in ("adaptive", "epoch_kl"):
            self.learning_rate = float(state["learning_rate"])
        if "estimator_optimizer" in state:
            self.estimator_optimizer.load_state_dict(state["estimator_optimizer"])
        if "hist_encoder_optimizer" in state:
            self.hist_encoder_optimizer.load_state_dict(state["hist_encoder_optimizer"])

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
