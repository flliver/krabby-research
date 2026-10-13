"""Firmware-in-Isaac loop: sensor readback, firmware step, drive pins, advance physics."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from firmware.sim.host_bridge import HostActuatorBridge, load_host_bridge
from firmware.sim.isaac_scene import advance_sim, physics_step_ms
from firmware.sim.pot_readback import sync_sensor_reads
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
    config: SimulatedRobotConfig | None = None
    advance: Callable[[Any], Any] | None = None
    pin_state: PinStateModel = field(default_factory=PinStateModel)
    _wrench_body_indices: dict[str, int] | None = field(default=None, init=False, repr=False)

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
            config=cfg,
            **kwargs,
        )

    def reset(self) -> None:
        self.bridge.reset()
        self.pin_state.reset()
        self._wrench_body_indices = None
        if self.config is not None:
            self.bridge.set_hall_edges_per_full_travel(
                self.config.hall_edges_per_full_travel
            )
        for actuator in self.actuators:
            band = actuator.pot_band
            if band is None:
                continue
            slot_id = actuator.slot.firmware_actuator_id
            self.bridge.set_actuator_stops(slot_id, band.min_adc, band.max_adc)

    def jog(self, actuator_id: str, pwm: int) -> None:
        """Open-loop jog; the drive pins reach Isaac on the next step."""
        self.bridge.handle_jog(actuator_id, pwm)

    def set_target(self, actuator_id: str, position: float) -> None:
        """Closed-loop target in 0..1 (T command); ramps use ``millis()`` from physics time."""
        self.bridge.set_target(actuator_id, position)

    def millis(self) -> int:
        return self.bridge.millis()

    def _wrench_body_indices_cached(self) -> dict[str, int] | None:
        if self.config is None:
            return None
        if self._wrench_body_indices is not None:
            return self._wrench_body_indices
        from firmware.sim.wrench_readback import resolve_wrench_body_indices

        articulation = self.articulation
        if not hasattr(articulation, "find_bodies"):
            return None
        try:
            self._wrench_body_indices = resolve_wrench_body_indices(articulation, self.config)
        except (AttributeError, KeyError, TypeError):
            return None
        return self._wrench_body_indices

    def record_drive_pins_to_pin_state(self) -> None:
        """Snapshot host shim pin outputs into ``pin_state`` (for HAL action readback)."""
        for actuator in self.actuators:
            pwm_r, pwm_l, pin_en = actuator.slot.drive_pins
            self.pin_state.analog_write(pwm_r, self.bridge.pin_pwm(pwm_r))
            self.pin_state.analog_write(pwm_l, self.bridge.pin_pwm(pwm_l))
            self.pin_state.digital_write(pin_en, self.bridge.pin_digital(pin_en))

    def tick_firmware_without_physics(self, *, process_usb_serial: bool = False) -> None:
        """Run one firmware control tick without advancing Isaac physics."""
        sync_sensor_reads(
            self.bridge,
            self.actuators,
            self.articulation,
            wrench_body_index_by_actuator=self._wrench_body_indices_cached(),
        )
        self.bridge.advance_millis(physics_step_ms(self.env))
        if process_usb_serial:
            self.bridge.firmware_loop_once()
        else:
            self.bridge.update_all()
        self.record_drive_pins_to_pin_state()

    def step(self, *, process_usb_serial: bool = False) -> Any:
        # Plant state is from the previous step's physics; refresh pots before firmware reads.
        self.tick_firmware_without_physics(process_usb_serial=process_usb_serial)
        for actuator in self.actuators:
            actuator.apply_velocity_target(self.pin_state, self.articulation)
        # Not env.step(): the task's RL action term rewrites joint position targets and
        # clears velocity targets every step, so a firmware-driven joint would never keep
        # the command written above.
        advance = self.advance or advance_sim
        return advance(self.env)
