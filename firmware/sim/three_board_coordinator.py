"""Coordinate three host firmware boards against 18 Isaac joints in one env.step."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from firmware.sim.host_bridge import HostActuatorBridge, load_host_bridge
from firmware.sim.isaac_scene import advance_sim, physics_step_ms, resolve_eighteen_joint_indices
from firmware.sim.pot_readback import sync_sensor_reads
from firmware.sim.simulated_actuator import JointVelocityTarget, SimulatedActuator
from firmware.sim.simulated_robot_config import ThreeBoardSimConfigs
from firmware.sim.sim_mode import single_leader_eighteen_enabled
from firmware.sim.three_board_boot import boot_three_board_emulators


@dataclass
class ThreeBoardRobotCore:
    """One Isaac scene, three firmware roles, 18 disjoint joint bindings."""

    bridge: HostActuatorBridge
    actuators: tuple[SimulatedActuator, ...]
    articulation: JointVelocityTarget
    env: Any
    configs: ThreeBoardSimConfigs
    advance: Any = None
    _wrench_body_indices: dict[str, int] | None = field(default=None, init=False, repr=False)

    @classmethod
    def eighteen_joints(
        cls,
        env: Any,
        articulation: JointVelocityTarget,
        joint_index_for_name: dict[str, int],
        *,
        bridge: HostActuatorBridge | None = None,
        configs: ThreeBoardSimConfigs | None = None,
        max_joint_speed: float = 1.0,
        **kwargs: Any,
    ) -> ThreeBoardRobotCore:
        cfg = configs or ThreeBoardSimConfigs.rev3()
        cfg.validate_eighteen_distinct_joints()
        host = bridge if bridge is not None else load_host_bridge()
        return cls(
            bridge=host,
            actuators=cfg.actuators(
                joint_index_for_name,
                max_joint_speed=max_joint_speed,
            ),
            articulation=articulation,
            env=env,
            configs=cfg,
            **kwargs,
        )

    def reset(self) -> None:
        if single_leader_eighteen_enabled():
            self.bridge.enable_single_leader_eighteen_joint(True)
        else:
            boot_three_board_emulators(self.bridge)
            self.bridge.enable_three_board_firmware_loop(True)
        self.bridge.firmware_setup()
        self._wrench_body_indices = None
        for board_cfg in (self.configs.front, self.configs.left, self.configs.right):
            self.bridge.set_hall_edges_per_full_travel(
                board_cfg.hall_edges_per_full_travel
            )
        for actuator in self.actuators:
            band = actuator.pot_band
            if band is None:
                continue
            slot_id = actuator.slot.firmware_actuator_id
            self.bridge.set_actuator_stops(slot_id, band.min_adc, band.max_adc)

    def jog(self, actuator_id: str, pwm: int) -> None:
        self.bridge.serial_usb_rx_write(f"J{actuator_id} {int(pwm)}\n".encode("ascii"))

    def _wrench_body_indices_cached(self) -> dict[str, int] | None:
        if self._wrench_body_indices is not None:
            return self._wrench_body_indices
        from firmware.sim.wrench_readback import resolve_wrench_body_indices

        articulation = self.articulation
        if not hasattr(articulation, "find_bodies"):
            return None
        try:
            self._wrench_body_indices = resolve_wrench_body_indices(
                articulation,
                self.configs.front,
            )
        except (AttributeError, KeyError, TypeError):
            return None
        return self._wrench_body_indices

    def step(self, *, process_usb_serial: bool = False) -> Any:
        sync_sensor_reads(
            self.bridge,
            self.actuators,
            self.articulation,
            wrench_body_index_by_actuator=self._wrench_body_indices_cached(),
        )
        self.bridge.advance_millis(physics_step_ms(self.env))
        self.bridge.firmware_loop_once()

        for actuator in self.actuators:
            name = actuator.slot.firmware_actuator_id
            command_pwm = self.bridge.three_board_actuator_command_pwm(name)
            actuator.apply_velocity_from_command_pwm(command_pwm, self.articulation)

        advance = self.advance or advance_sim
        return advance(self.env)
