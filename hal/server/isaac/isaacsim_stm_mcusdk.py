"""IsaacSim MCU SDK for fullsim STM CAN transport (duty feedback like drive pins)."""

from __future__ import annotations

import logging
import time
from typing import Optional, Union

import numpy as np
import torch

from hal.client.data_structures.hardware import JointCommand
from hal.server.isaac.firmware_can_loop import CanStmCommandLoop
from hal.server.isaac.isaacsim_mcusdk import IsaacSimMCUSDK
from hal.server.robot_definition import RobotDefinition

logger = logging.getLogger(__name__)

__all__ = ["IsaacSimStmMCUSDK"]


class IsaacSimStmMCUSDK:
    """Duty-based fullsim SDK: SET_DUTY over CAN, normalized duty action for ``env.step``."""

    def __init__(
        self,
        joint_names: tuple[str, ...],
        action_dim: int,
        device: Union[torch.device, str] = "cpu",
        *,
        mcu_joint_names: tuple[str, ...] = (),
        firmware_loop: Optional[CanStmCommandLoop] = None,
    ) -> None:
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        self._joint_names = joint_names
        self._action_dim = action_dim
        self._mcu_joint_names = mcu_joint_names
        self._firmware_loop = firmware_loop
        self._device = torch.device(device) if isinstance(device, str) else device
        self._direct_sdk = IsaacSimMCUSDK()
        if firmware_loop is not None and len(mcu_joint_names) == 0:
            raise ValueError("mcu_joint_names required when firmware_loop is set")
        logger.info(
            "IsaacSimStmMCUSDK initialized (action_dim=%d, can_loop=%s)",
            action_dim,
            "enabled" if firmware_loop is not None else "passthrough",
        )

    @classmethod
    def from_robot_definition(
        cls,
        robot_definition: RobotDefinition,
        device: Union[torch.device, str] = "cpu",
        *,
        firmware_loop: Optional[CanStmCommandLoop] = None,
    ) -> IsaacSimStmMCUSDK:
        return cls(
            joint_names=robot_definition.get_joint_names(),
            action_dim=robot_definition.get_total_joint_count(),
            device=device,
            mcu_joint_names=robot_definition.get_mcu_joints(),
            firmware_loop=firmware_loop,
        )

    def apply_command(
        self,
        command: JointCommand,
        num_envs: int = 1,
    ) -> torch.Tensor:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")

        if self._firmware_loop is None:
            action_dict = self._direct_sdk.apply_command(command, num_envs=num_envs)
            return self._action_dict_to_tensor(action_dict, num_envs=num_envs)

        t0 = time.perf_counter()
        self._firmware_loop.apply_hal_command(command)
        t_cmd = (time.perf_counter() - t0) * 1000.0
        self._firmware_loop.tick_firmware()
        tick_ms = (time.perf_counter() - t0) * 1000.0 - t_cmd
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "IsaacSimStmMCUSDK: stm loop apply_hal_command=%.1f ms tick_firmware=%.1f ms",
                t_cmd,
                tick_ms,
            )
        elif tick_ms + t_cmd >= 500.0:
            logger.warning(
                "IsaacSimStmMCUSDK: slow stm loop (apply=%.1f ms tick=%.1f ms)",
                t_cmd,
                tick_ms,
            )

        cmd_dict = command.to_positions_dict()
        action_dict: dict[str, float] = {}
        for name in self._joint_names[: self._action_dim]:
            if name in self._mcu_joint_names:
                action_dict[name] = self._firmware_loop.duty_for_hal_joint(name)
            else:
                action_dict[name] = float(cmd_dict.get(name, 0.0))
        return self._action_dict_to_tensor(action_dict, num_envs=num_envs)

    def _action_dict_to_tensor(
        self,
        action_dict: dict[str, float],
        *,
        num_envs: int,
    ) -> torch.Tensor:
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