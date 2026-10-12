"""Drive-only loop: firmware step -> pin state -> config rows drive Isaac -> advance sim."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from firmware.sim.host_bridge import HostActuatorBridge, load_host_bridge
from firmware.sim.isaac_scene import advance_sim, physics_step_ms
from firmware.sim.pot_readback import sync_pot_analog_reads
from firmware.sim.pin_state import PinStateModel
from firmware.sim.simulated_actuator import JointVelocityTarget, SimulatedActuator
from firmware.sim.simulated_robot_config import SimulatedRobotConfig

FIRMWARE_STEP_MS = 10  # arduino.ino main-loop cadence on hardware (sim clock uses physics dt)


@dataclass
class RobotCore:
    """Config rows own every pin number; this loop only sequences firmware and Isaac."""

    bridge: HostActuatorBridge
    actuators: tuple[SimulatedActuator, ...]
    articulation: JointVelocityTarget
    env: Any
    advance: Callable[[Any], Any] | None = None
    pin_state: PinStateModel = field(default_factory=PinStateModel)

    @classmethod
    def front_board(
        cls,
        env: Any,
        articulation: JointVelocityTarget,
        joint_index_for_name: Mapping[str, int] | Callable[[str], int],
        *,
        bridge: HostActuatorBridge | None = None,
        config: SimulatedRobotConfig | None = None,
        max_joint_speed: float = 1.0,
        **kwargs: Any,
    ) -> RobotCore:
        cfg = config or SimulatedRobotConfig.front_board_rev3()
        return cls(
            bridge=bridge if bridge is not None else load_host_bridge(),
            actuators=cfg.actuators(joint_index_for_name, max_joint_speed=max_joint_speed),
            articulation=articulation,
            env=env,
            **kwargs,
        )

    def reset(self) -> None:
        self.bridge.reset()
        self.pin_state.reset()

    def jog(self, actuator_id: str, pwm: int) -> None:
        """Open-loop jog; the drive pins reach Isaac on the next step."""
        self.bridge.handle_jog(actuator_id, pwm)

    def set_target(self, actuator_id: str, position: float) -> None:
        """Closed-loop target in 0..1 (T command); ramps use ``millis()`` from physics time."""
        self.bridge.set_target(actuator_id, position)

    def millis(self) -> int:
        return self.bridge.millis()

    def step(self) -> Any:
        sync_pot_analog_reads(self.bridge, self.actuators, self.articulation)
        self.bridge.advance_millis(physics_step_ms(self.env))
        self.bridge.update_all()
        for actuator in self.actuators:
            pwm_r, pwm_l, pin_en = actuator.slot.drive_pins
            self.pin_state.analog_write(pwm_r, self.bridge.pin_pwm(pwm_r))
            self.pin_state.analog_write(pwm_l, self.bridge.pin_pwm(pwm_l))
            self.pin_state.digital_write(pin_en, self.bridge.pin_digital(pin_en))
            actuator.apply_velocity_target(self.pin_state, self.articulation)
        # Not env.step(): the task's RL action term rewrites joint position targets and
        # clears velocity targets every step, so a firmware-driven joint would never keep
        # the command written above.
        advance = self.advance or advance_sim
        return advance(self.env)
