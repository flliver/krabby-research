"""IsaacSim firmware-backed MCU SDK for the fullsim HAL command path.

Skeleton: threshold and joint ordering match ``IsaacSimMCUSDK``; firmware, pins,
and krabby-SDK wiring arrive in later Task 5 checklist items.
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np
import torch

from hal.client.data_structures.hardware import JointCommand
from hal.server.isaac.isaacsim_mcusdk import IsaacSimMCUSDK
from hal.server.robot_definition import RobotDefinition

logger = logging.getLogger(__name__)

__all__ = ["IsaacSimFirmwareMCUSDK"]


class IsaacSimFirmwareMCUSDK:
    """Firmware-in-the-loop command SDK for IsaacSim (skeleton).

    Accepts ``JointCommand`` like ``IsaacSimMCUSDK`` and returns a torch action
    tensor in the same shape ``IsaacSimHalServer.apply_command`` produces today
    ``(num_envs, action_dim)`` for ``env.step()``.
    """

    def __init__(
        self,
        joint_names: tuple[str, ...],
        action_dim: int,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if action_dim > len(joint_names):
            raise ValueError(
                f"action_dim ({action_dim}) exceeds joint_names length ({len(joint_names)})"
            )
        self._joint_names = joint_names
        self._action_dim = action_dim
        self._device = torch.device(device) if isinstance(device, str) else device
        self._direct_sdk = IsaacSimMCUSDK()
        logger.info(
            "IsaacSimFirmwareMCUSDK initialized (skeleton passthrough, action_dim=%d)",
            action_dim,
        )

    @classmethod
    def from_robot_definition(
        cls,
        robot_definition: RobotDefinition,
        device: Union[torch.device, str] = "cpu",
    ) -> IsaacSimFirmwareMCUSDK:
        """Build SDK from HAL robot topology (joint order + action width)."""
        return cls(
            joint_names=robot_definition.get_joint_names(),
            action_dim=robot_definition.get_total_joint_count(),
            device=device,
        )

    def apply_command(
        self,
        command: JointCommand,
        num_envs: int = 1,
    ) -> torch.Tensor:
        """Apply joint command and return env action tensor.

        Args:
            command: JointCommand with positions keyed by joint names (radians / normalized PWM).
            num_envs: Batch size for ``env.step()`` (expanded when > 1).

        Returns:
            Float tensor of shape ``(num_envs, action_dim)``.

        Raises:
            ValueError: If ``num_envs`` is invalid or command processing fails.
        """
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")

        # TODO(Task 5): radians -> firmware command, tick control loop, read pin drive.
        action_dict = self._direct_sdk.apply_command(command, num_envs=num_envs)
        return self._action_dict_to_tensor(action_dict, num_envs=num_envs)

    def _action_dict_to_tensor(
        self,
        action_dict: dict[str, float],
        *,
        num_envs: int,
    ) -> torch.Tensor:
        """Convert ordered joint dict to batched action tensor (matches hal_server.apply_command)."""
        action_list = [
            action_dict[name] for name in self._joint_names[: self._action_dim]
        ]
        action_np = np.array(action_list, dtype=np.float32)
        action = torch.from_numpy(action_np).to(device=self._device, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.shape[0] == 1 and num_envs > 1:
            action = action.expand(num_envs, -1)
        return action
