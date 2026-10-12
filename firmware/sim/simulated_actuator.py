"""Map recorded Mega drive pins to Isaac joint velocity targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from firmware.sim.pin_state import PinStateModel
from firmware.sim.simulated_robot_config import SimulatedSlotConfig

PWM_COMMAND_FULL_SCALE = 255


@runtime_checkable
class JointVelocityTarget(Protocol):
    """Subset of Isaac Lab Articulation used for open-loop drive."""

    def set_joint_velocity_target(
        self,
        velocities,
        joint_ids=None,
        env_ids=None,
    ) -> None: ...


@dataclass
class SimulatedActuator:
    """One config row: read drive pins from pin state, drive one Isaac joint."""

    slot: SimulatedSlotConfig
    joint_index: int
    max_joint_speed: float = 1.0

    def signed_pwm_command(self, pin_state: PinStateModel) -> int:
        pwm_r, pwm_l, pin_en = self.slot.drive_pins
        return pin_state.signed_pwm_command(pwm_r, pwm_l, pin_en)

    def drive_scalar(self, pin_state: PinStateModel) -> float:
        """Linear map: cmd in [-255, 255] -> [-max_joint_speed, max_joint_speed]."""
        cmd = self.signed_pwm_command(pin_state)
        return (cmd / PWM_COMMAND_FULL_SCALE) * self.max_joint_speed

    def apply_velocity_target(
        self,
        pin_state: PinStateModel,
        articulation: JointVelocityTarget,
        *,
        env_ids=None,
    ) -> float:
        speed = self.drive_scalar(pin_state)
        articulation.set_joint_velocity_target(
            speed,
            joint_ids=self.joint_index,
            env_ids=env_ids,
        )
        return speed


class RecordedArticulation:
    """Test double: records set_joint_velocity_target calls."""

    def __init__(self) -> None:
        self.velocity_by_joint: dict[int, float] = {}

    def set_joint_velocity_target(
        self,
        velocities,
        joint_ids=None,
        env_ids=None,
    ) -> None:
        del env_ids
        joint_id = joint_ids if isinstance(joint_ids, int) else int(joint_ids[0])
        if hasattr(velocities, "__getitem__"):
            value = float(velocities[0])
        else:
            value = float(velocities)
        self.velocity_by_joint[joint_id] = value

    def velocity_for(self, joint_id: int) -> float | None:
        return self.velocity_by_joint.get(joint_id)
