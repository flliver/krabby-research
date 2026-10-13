"""IsaacSim firmware-backed MCU SDK for the fullsim HAL command path."""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import torch

from hal.client.data_structures.hardware import JointCommand
from hal.server.isaac.firmware_command_mapping import (
    hal_joint_to_firmware_id,
    joint_radians_to_jog_pwm,
    signed_pwm_to_normalized_action,
)
from hal.server.isaac.firmware_loop_adapter import FirmwareCommandLoop
from hal.server.isaac.isaacsim_mcusdk import IsaacSimMCUSDK
from hal.server.robot_definition import RobotDefinition

logger = logging.getLogger(__name__)

__all__ = ["IsaacSimFirmwareMCUSDK"]


class IsaacSimFirmwareMCUSDK:
    """Firmware-in-the-loop command SDK for IsaacSim.

    Accepts ``JointCommand`` like ``IsaacSimMCUSDK`` and returns a torch action
    tensor in the same shape ``IsaacSimHalServer.apply_command`` produces today
    ``(num_envs, action_dim)`` for ``env.step()``.

    With ``firmware_loop`` set, radians are jogged through the host firmware,
    one control tick runs, and drive pins are read back as normalized actions.
    Without ``firmware_loop``, falls back to direct threshold passthrough (tests).
    """

    def __init__(
        self,
        joint_names: tuple[str, ...],
        action_dim: int,
        device: Union[torch.device, str] = "cpu",
        *,
        mcu_joint_names: tuple[str, ...] = (),
        firmware_loop: Optional[FirmwareCommandLoop] = None,
    ) -> None:
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if action_dim > len(joint_names):
            raise ValueError(
                f"action_dim ({action_dim}) exceeds joint_names length ({len(joint_names)})"
            )
        self._joint_names = joint_names
        self._action_dim = action_dim
        self._mcu_joint_names = mcu_joint_names
        self._firmware_loop = firmware_loop
        self._device = torch.device(device) if isinstance(device, str) else device
        self._direct_sdk = IsaacSimMCUSDK()
        if firmware_loop is not None and len(mcu_joint_names) == 0:
            raise ValueError("mcu_joint_names required when firmware_loop is set")
        logger.info(
            "IsaacSimFirmwareMCUSDK initialized (action_dim=%d, firmware_loop=%s)",
            action_dim,
            "enabled" if firmware_loop is not None else "passthrough",
        )

    @classmethod
    def from_robot_definition(
        cls,
        robot_definition: RobotDefinition,
        device: Union[torch.device, str] = "cpu",
        *,
        firmware_loop: Optional[FirmwareCommandLoop] = None,
    ) -> IsaacSimFirmwareMCUSDK:
        """Build SDK from HAL robot topology (joint order + action width)."""
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

        if self._firmware_loop is None:
            action_dict = self._direct_sdk.apply_command(command, num_envs=num_envs)
            return self._action_dict_to_tensor(action_dict, num_envs=num_envs)

        cmd_dict = command.to_positions_dict()
        if not cmd_dict:
            raise ValueError("command must be non-empty")

        jog_map = joint_radians_to_jog_pwm(cmd_dict, self._mcu_joint_names)
        apply_hal = getattr(self._firmware_loop, "apply_hal_command", None)
        if callable(apply_hal):
            apply_hal(command)
        else:
            for firmware_id, pwm in jog_map.items():
                self._firmware_loop.jog_actuator(firmware_id, pwm)
            self._firmware_loop.tick_firmware()

        action_dict = self._action_dict_from_firmware(jog_map.keys())
        return self._action_dict_to_tensor(action_dict, num_envs=num_envs)

    def _action_dict_from_firmware(self, firmware_ids) -> dict[str, float]:
        """Ordered HAL joint values from simulated drive pins / manager PWM."""
        fw_pwm = {
            fw_id: self._firmware_loop.signed_command_pwm(fw_id)  # type: ignore[union-attr]
            for fw_id in firmware_ids
        }
        action_dict: dict[str, float] = {}
        for name in self._joint_names[: self._action_dim]:
            fw_id = hal_joint_to_firmware_id(name)
            pwm = fw_pwm.get(fw_id, 0)
            action_dict[name] = signed_pwm_to_normalized_action(pwm)
        return action_dict

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
