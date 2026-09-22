
from __future__ import annotations

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


def _safe_advantage_normalize(adv: torch.Tensor) -> torch.Tensor:
    """Finite normalization when global advantage std is 0 or non-finite."""
    warn_if_nonfinite("advantage_normalize.input", adv)
    adv = torch.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)
    std = adv.std()
    if torch.isfinite(std) and std > 1e-8:
        return (adv - adv.mean()) / (std + 1e-8)
    return adv - adv.mean()


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
    ) -> None:
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
        self.hist_encoder_optimizer = optim.Adam(
            self.policy.actor.history_encoder.parameters(), lr=learning_rate
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

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        priv_reg_coef = 0.0

        for batch in generator:
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

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

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

            log_ratio = actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
            ratio = torch.exp(log_ratio)
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

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

                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                if self.rnd_optimizer:
                    self.rnd_optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_entropy += entropy_batch.mean().item()
                mean_priv_reg_loss += priv_reg_loss.mean().item()
            else:
                self.optimizer.zero_grad(set_to_none=True)

            mean_estimator_loss += estimator_loss.item() if torch.isfinite(estimator_loss).all() else 0.0

            if mean_rnd_loss is not None and rnd_loss is not None and torch.isfinite(rnd_loss).all():
                mean_rnd_loss += rnd_loss.item()
            if (
                mean_symmetry_loss is not None
                and symmetry_loss is not None
                and torch.isfinite(symmetry_loss).all()
            ):
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_priv_reg_loss /= num_updates
        mean_entropy /= num_updates
        mean_estimator_loss /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

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

        ``counter`` drives the ``priv_reg`` coefficient schedule; ``learning_rate`` is the
        adaptive-schedule value (the policy optimizer's param groups are overwritten with it on
        the first minibatch after a resume).
        """
        return {
            "counter": self.counter,
            "learning_rate": self.learning_rate,
            "estimator_optimizer": self.estimator_optimizer.state_dict(),
            "hist_encoder_optimizer": self.hist_encoder_optimizer.state_dict(),
        }

    def load_resume_state(self, state: dict[str, Any]) -> None:
        self.counter = int(state["counter"])
        if self.schedule == "adaptive":
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
