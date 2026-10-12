"""Declarative pin-to-joint binding for firmware-in-Isaac simulation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Sequence

from firmware.sim.hall_encoder import DEFAULT_HALL_EDGES_PER_FULL_TRAVEL
from firmware.sim.pot_calibration import (
    DEFAULT_POT_CALIBRATION_SEED,
    PotCalibrationBand,
    draw_pot_calibration_bands,
)

# REV 3 drive + analog pins - keep in sync with firmware/arduino/board_pins.h (KRABBY_PIN_REV=3)
# and ACT_LIST_FRONT wiring in firmware/arduino/arduino.ino.


@dataclass(frozen=True, slots=True)
class SimulatedSlotConfig:
    """One H-bridge slot: drive pins, pot/IS analog pins, firmware name, Isaac joint."""

    slot_index: int  # S0 .. S5 on the Mega
    pin_pwm_r: int
    pin_pwm_l: int
    pin_en: int
    pin_pot: int  # potentiometer (analogRead)
    pin_is: int  # current sense (analogRead)
    firmware_actuator_id: str  # e.g. FLHY - matches LinearActuator name / telemetry
    isaac_joint_name: str  # USD PhysicsRevoluteJoint name in assets/crab_simple.usda

    @property
    def drive_pins(self) -> tuple[int, int, int]:
        return (self.pin_pwm_r, self.pin_pwm_l, self.pin_en)


# FRONT board, REV 3 - mirrors ACT_LIST_FRONT order in arduino.ino
_FRONT_BOARD_REV3_SLOTS: tuple[SimulatedSlotConfig, ...] = (
    SimulatedSlotConfig(0, 2, 3, 22, 0, 6, "FLHY", "FL_Body_Hip_RevoluteJoint"),
    SimulatedSlotConfig(1, 4, 5, 24, 1, 7, "FLHL", "FL_Hip_Femur_RevoluteJoint"),
    SimulatedSlotConfig(2, 6, 7, 26, 2, 8, "FLKL", "FL_Femur_Tibia_RevoluteJoint"),
    SimulatedSlotConfig(3, 8, 9, 23, 3, 9, "FRHY", "FR_Body_Hip_RevoluteJoint"),
    SimulatedSlotConfig(4, 10, 11, 25, 4, 10, "FRHL", "FR_Hip_Femur_RevoluteJoint"),
    SimulatedSlotConfig(5, 12, 13, 27, 5, 11, "FRKL", "FR_Femur_Tibia_RevoluteJoint"),
)

PIN_REVISION_LABEL = "PINS_REV3_UNO_V02"  # boardPinRevisionLabel() when KRABBY_PIN_REV=3
KRABBY_PIN_REV = 3

# Per-leg Isaac find_joints-style patterns (exclude CamShaft by using exact names from slots).
ISAAC_LEG_JOINT_PATTERN = "{leg}_.*_RevoluteJoint"


@dataclass(frozen=True, slots=True)
class SimulatedRobotConfig:
    """Six-slot FRONT-board robot (open-loop drive path; sensor readback wired separately)."""

    pin_revision_label: str
    board_role: str
    slots: tuple[SimulatedSlotConfig, ...]
    pot_bands: tuple[PotCalibrationBand, ...]
    hall_edges_per_full_travel: int = DEFAULT_HALL_EDGES_PER_FULL_TRAVEL

    @classmethod
    def front_board_rev3(
        cls,
        *,
        pot_calibration_seed: int = DEFAULT_POT_CALIBRATION_SEED,
        hall_edges_per_full_travel: int = DEFAULT_HALL_EDGES_PER_FULL_TRAVEL,
    ) -> SimulatedRobotConfig:
        slots = _FRONT_BOARD_REV3_SLOTS
        bands = draw_pot_calibration_bands(len(slots), pot_calibration_seed)
        return cls(
            pin_revision_label=PIN_REVISION_LABEL,
            board_role="FRONT",
            slots=slots,
            pot_bands=bands,
            hall_edges_per_full_travel=hall_edges_per_full_travel,
        )

    def pot_band_for_slot(self, slot_index: int) -> PotCalibrationBand:
        for slot, band in zip(self.slots, self.pot_bands, strict=True):
            if slot.slot_index == slot_index:
                return band
        raise KeyError(f"unknown slot index: {slot_index}")

    def isaac_joint_names(self) -> tuple[str, ...]:
        return tuple(s.isaac_joint_name for s in self.slots)

    def slot_by_firmware_id(self, actuator_id: str) -> SimulatedSlotConfig:
        for slot in self.slots:
            if slot.firmware_actuator_id == actuator_id:
                return slot
        raise KeyError(f"unknown firmware actuator id: {actuator_id!r}")

    def slot_by_index(self, index: int) -> SimulatedSlotConfig:
        for slot in self.slots:
            if slot.slot_index == index:
                return slot
        raise KeyError(f"unknown slot index: {index}")

    def validate_distinct_joints(self) -> None:
        names = self.isaac_joint_names()
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate Isaac joint names in config: {names!r}")

    def validate_distinct_drive_pins(self) -> None:
        drive_sets = [slot.drive_pins for slot in self.slots]
        if len(set(drive_sets)) != len(drive_sets):
            raise ValueError(f"duplicate drive pin sets in config: {drive_sets!r}")

    @staticmethod
    def _resolve_joint_index(
        isaac_joint_name: str,
        joint_index_for_name: Mapping[str, int] | Callable[[str], int],
    ) -> int:
        if callable(joint_index_for_name):
            return int(joint_index_for_name(isaac_joint_name))
        try:
            return int(joint_index_for_name[isaac_joint_name])
        except KeyError as exc:
            raise KeyError(f"no Isaac joint index for {isaac_joint_name!r}") from exc

    def actuators(
        self,
        joint_index_for_name: Mapping[str, int] | Callable[[str], int],
        *,
        max_joint_speed: float = 1.0,
    ) -> tuple["SimulatedActuator", ...]:
        from firmware.sim.simulated_actuator import SimulatedActuator

        return tuple(
            SimulatedActuator(
                slot=slot,
                joint_index=self._resolve_joint_index(slot.isaac_joint_name, joint_index_for_name),
                max_joint_speed=max_joint_speed,
                pot_band=band,
            )
            for slot, band in zip(self.slots, self.pot_bands, strict=True)
        )

    def actuators_by_firmware_id(
        self,
        joint_index_for_name: Mapping[str, int] | Callable[[str], int],
        *,
        max_joint_speed: float = 1.0,
    ) -> dict[str, "SimulatedActuator"]:
        return {
            actuator.slot.firmware_actuator_id: actuator
            for actuator in self.actuators(
                joint_index_for_name,
                max_joint_speed=max_joint_speed,
            )
        }

    def actuator_by_firmware_id(
        self,
        actuator_id: str,
        joint_index_for_name: Mapping[str, int] | Callable[[str], int],
        *,
        max_joint_speed: float = 1.0,
    ) -> "SimulatedActuator":
        return self.actuators_by_firmware_id(
            joint_index_for_name,
            max_joint_speed=max_joint_speed,
        )[actuator_id]

    def isaac_joint_pattern(self, leg: str) -> str:
        return ISAAC_LEG_JOINT_PATTERN.format(leg=leg)


def front_board_rev3_slots() -> Sequence[SimulatedSlotConfig]:
    """Convenience accessor for tests and SimulatedActuator wiring."""
    return SimulatedRobotConfig.front_board_rev3().slots
