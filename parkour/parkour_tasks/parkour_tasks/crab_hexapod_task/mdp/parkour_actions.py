"""Crab-hex action terms (clip-before-history for last_action alignment; Go2 keeps shared behavior)."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from parkour_isaaclab.envs.mdp.parkour_actions import DelayedJointPositionActionCfg
from parkour_isaaclab.envs.mdp.parkour_actions.joint_actions import DelayedJointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class CrabHexDelayedJointPositionAction(DelayedJointPositionAction):
    """Clip policy actions before the delay buffer so ``last_action`` obs matches applied commands."""

    def _clip_raw_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.cfg.clip is None:
            return actions
        return torch.clamp(actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1])

    def process_actions(self, actions: torch.Tensor):
        if self.env.common_step_counter % self._delay_update_global_steps == 0:
            if len(self._action_delay_steps) != 0:
                self.delay = torch.tensor(
                    self._action_delay_steps.pop(0), device=self.device, dtype=torch.float
                )
        clipped_actions = self._clip_raw_actions(actions)
        self._action_history_buf = torch.cat(
            [self._action_history_buf[:, 1:].clone(), clipped_actions[:, None, :].clone()], dim=1
        )
        indices = -1 - self.delay
        if self._use_delay:
            self._raw_actions[:] = self._action_history_buf[:, indices.long()]
        else:
            self._raw_actions[:] = clipped_actions
        self._processed_actions = self._raw_actions * self._scale + self._offset


@configclass
class CrabHexDelayedJointPositionActionCfg(DelayedJointPositionActionCfg):
    class_type: type[ActionTerm] = CrabHexDelayedJointPositionAction
