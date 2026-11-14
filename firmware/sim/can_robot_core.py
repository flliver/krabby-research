"""Six STM leg backends on one CAN bus in a single IsaacSim env."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from firmware.sim.cam_hip_slave import apply_cam_hip_slave, build_cam_hip_slave_binding
from firmware.sim.hall_encoder import resolve_stm_counts_per_full_travel
from firmware.sim.can_board_bridge import Leg, StmBoardBridge, load_stm_board_bridge
from firmware.sim.isaac_scene import advance_sim, physics_step_ms
from firmware.sim.plant_readback import PlantReadbackView
from firmware.sim.pot_position import joint_angle_to_travel_fraction
from firmware.sim.simulated_actuator import JointVelocityTarget, _joint_velocity_target_tensor
from firmware.sim.simulated_robot_config import (
    KRABBY_HEX_FIRMWARE_ACTUATOR_IDS,
    isaac_joint_name_for_firmware_id,
)
from firmware.stm import ids
from firmware.stm.can_transport import send_fd_frame

_STM_POT_MAX_ADC = 3590


def leg_motor_from_actuator_id(actuator_id: str) -> tuple[int, int]:
    leg_codes = {"FL": 1, "FR": 2, "ML": 3, "MR": 4, "RL": 5, "RR": 6}
    joint_codes = {"HY": 1, "HL": 2, "KL": 3}
    leg = leg_codes[actuator_id[:2]]
    joint = joint_codes[actuator_id[2:]]
    return leg, joint - 1


def route_rx_to_legs(
    can_id: int,
    data: bytes,
    legs: tuple[Leg, ...],
    inject: Callable[[Leg, int, bytes], None],
) -> None:
    """Fan-in host CAN frames to per-board RX queues."""
    if can_id == ids.set_duty_can_id():
        for leg in legs:
            inject(leg, can_id, data)
        return
    base = ids.CAN_ID_CALIBRATE_BASE
    if base < can_id <= base + len(legs):
        leg_id = can_id - base
        inject(legs[leg_id - 1], can_id, data)


@dataclass(frozen=True, slots=True)
class StmSimJoint:
    leg_id: int
    motor_index: int
    joint_index: int
    actuator_id: str


@dataclass
class CanRobotCore:
    bridge: StmBoardBridge
    legs: tuple[Leg, ...]
    joints: tuple[StmSimJoint, ...]
    articulation: JointVelocityTarget
    env: Any
    bus: Any | None = None
    max_joint_speed: float = 1.0
    advance: Callable[[Any], Any] | None = None
    _sim_ms: int = field(default=0, init=False)
    _step_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _cam_hip_slave: Any = field(default=None, init=False, repr=False)
    _cam_hip_slave_resolved: bool = field(default=False, init=False, repr=False)

    @classmethod
    def from_isaac(
        cls,
        env: Any,
        articulation: JointVelocityTarget,
        joint_index_for_name: Mapping[str, int] | Callable[[str], int],
        *,
        bridge: StmBoardBridge | None = None,
        bus: Any | None = None,
        max_joint_speed: float = 1.0,
    ) -> CanRobotCore:
        br = bridge if bridge is not None else load_stm_board_bridge()
        legs = tuple(br.create_leg(node_id) for node_id in ids.LEG_IDS)
        joints: list[StmSimJoint] = []
        for actuator_id in KRABBY_HEX_FIRMWARE_ACTUATOR_IDS:
            leg_id, motor_index = leg_motor_from_actuator_id(actuator_id)
            isaac_name = isaac_joint_name_for_firmware_id(actuator_id)
            if callable(joint_index_for_name):
                joint_index = int(joint_index_for_name(isaac_name))
            else:
                joint_index = int(joint_index_for_name[isaac_name])
            joints.append(
                StmSimJoint(
                    leg_id=leg_id,
                    motor_index=motor_index,
                    joint_index=joint_index,
                    actuator_id=actuator_id,
                )
            )
        return cls(
            bridge=br,
            legs=legs,
            joints=tuple(joints),
            articulation=articulation,
            env=env,
            bus=bus,
            max_joint_speed=max_joint_speed,
        )

    def reset(self) -> None:
        self._sim_ms = 0
        self.bridge.set_tick_ms(0)
        self._cam_hip_slave = None
        self._cam_hip_slave_resolved = False

    def _poll_bus(self) -> None:
        if self.bus is None:
            return
        while True:
            msg = self.bus.recv(timeout=0.0)
            if msg is None:
                break
            route_rx_to_legs(
                int(msg.arbitration_id),
                bytes(msg.data),
                self.legs,
                self.bridge.inject_rx,
            )

    def _sync_sensors(self, plant: PlantReadbackView | None) -> None:
        if plant is None:
            return
        counts_scale = resolve_stm_counts_per_full_travel()
        for joint in self.joints:
            travel = plant.joint_travel(joint.joint_index)
            if travel is None:
                continue
            pos_rad, lower_rad, upper_rad = travel
            fraction = joint_angle_to_travel_fraction(pos_rad, lower_rad, upper_rad)
            adc = int(round(fraction * _STM_POT_MAX_ADC))
            if adc < 0:
                adc = 0
            elif adc > _STM_POT_MAX_ADC:
                adc = _STM_POT_MAX_ADC
            hall = int(round(fraction * counts_scale))
            leg = self.legs[joint.leg_id - 1]
            self.bridge.set_pot(leg, joint.motor_index, adc)
            self.bridge.set_hall(leg, joint.motor_index, hall)

    def _tick_boards(self) -> None:
        self.bridge.set_tick_ms(self._sim_ms)
        for leg in self.legs:
            self.bridge.app_tick(leg, self._sim_ms)
            if self.bus is None:
                continue
            tx = self.bridge.last_tx(leg)
            if tx is None:
                continue
            frame_id, payload = tx
            send_fd_frame(self.bus, frame_id, payload)

    def _apply_duties(self) -> None:
        for joint in self.joints:
            leg = self.legs[joint.leg_id - 1]
            duty = self.bridge.get_duty(leg, joint.motor_index)
            speed = duty * self.max_joint_speed
            self.articulation.set_joint_velocity_target(
                _joint_velocity_target_tensor(self.articulation, speed, joint.joint_index),
                joint_ids=joint.joint_index,
            )

    def _slave_body_hip_after_physics(self) -> None:
        if not self._cam_hip_slave_resolved:
            self._cam_hip_slave = build_cam_hip_slave_binding(self.articulation)
            self._cam_hip_slave_resolved = True
        if self._cam_hip_slave is None:
            return
        apply_cam_hip_slave(self._cam_hip_slave, self.articulation)

    def step(self) -> Any:
        with self._step_lock:
            self._poll_bus()
            plant = PlantReadbackView.from_articulation(self.articulation)
            self._sync_sensors(plant)
            self._tick_boards()
            dt_ms = int(physics_step_ms(self.env))
            self._sim_ms += dt_ms

            with torch.inference_mode():
                self._apply_duties()
                advance = self.advance or advance_sim
                result = advance(self.env)
                self._slave_body_hip_after_physics()
            return result