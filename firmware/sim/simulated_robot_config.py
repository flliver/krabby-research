"""Declarative pin-to-joint binding for firmware-in-Isaac simulation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Sequence

from firmware.sim.hall_encoder import DEFAULT_HALL_EDGES_PER_FULL_TRAVEL
from firmware.sim.pot_calibration import (
    DEFAULT_POT_CALIBRATION_SEED,
    PotCalibrationBand,
    draw_pot_calibration_bands,
)

# REV 3 drive + analog pins - keep in sync with firmware/arduino/board_pins.h (KRABBY_PIN_REV=3)
# and ACT_LIST_FRONT wiring in firmware/arduino/arduino.ino.


# Position sensor fitted to a slot's actuator. A hall actuator's pot pin carries
# its hall B line (see firmware/sim/pot_autocal_plant.py).
SENSOR_POT = "pot"
SENSOR_HALL = "hall"
SLOT_SENSORS = (SENSOR_POT, SENSOR_HALL)

# Hall edges from one stroke end to the other in the calibration plant. Knees:
# FLKL's measured full stroke with its hip curled (2026-11-23), inside the
# firmware's provisional CAL_HALL_MIN_STROKE_EDGES..CAL_HALL_MAX_STROKE_EDGES
# (450..1500). This is not hall_encoder_scale.json's edges_per_full_travel, which
# maps an Isaac joint's radians to edges.
DEFAULT_HALL_EDGES_PER_STROKE = 776
# Hall hips in the plant. FLHL counted 941-1003 edges from its curl end before its
# curled tibia pressed against the bottom of the robot, which is not the actuator's
# own end; that end was not measured. The plant has no body, so this placeholder is
# simply longer than that reach. SWEEP stops a hall hip after CAL_HIP_SWEEP_EDGES
# (700) anyway, and saves [0, 700].
DEFAULT_HALL_HIP_EDGES_PER_STROKE = 1100

# Board ids of the host sim (krabby_host_board_* functions): FRONT, LEFT, RIGHT.
BOARD_ID_BY_ROLE = {"FRONT": 0, "LEFT": 1, "RIGHT": 2}


@dataclass(frozen=True, slots=True)
class SimulatedSlotConfig:
    """One H-bridge slot: drive pins, pot/IS analog pins, firmware name, Isaac joint."""

    slot_index: int  # S0 .. S5 on the Mega
    pin_pwm_r: int
    pin_pwm_l: int
    pin_en: int
    pin_pot: int  # potentiometer (analogRead); hall B on a hall actuator
    pin_is: int  # current sense (analogRead)
    firmware_actuator_id: str  # e.g. FLHY - matches LinearActuator name / telemetry
    isaac_joint_name: str  # USD PhysicsRevoluteJoint name on the loaded hex plant
    sensor: str = SENSOR_POT  # SENSOR_POT or SENSOR_HALL
    hall_edges_per_stroke: int = DEFAULT_HALL_EDGES_PER_STROKE

    def __post_init__(self) -> None:
        if self.sensor not in SLOT_SENSORS:
            raise ValueError(f"unknown slot sensor: {self.sensor!r}")
        if self.hall_edges_per_stroke <= 0:
            raise ValueError(
                f"hall_edges_per_stroke must be > 0: {self.hall_edges_per_stroke}"
            )

    @property
    def drive_pins(self) -> tuple[int, int, int]:
        return (self.pin_pwm_r, self.pin_pwm_l, self.pin_en)

    @property
    def is_hall(self) -> bool:
        return self.sensor == SENSOR_HALL


def isaac_joint_name_for_firmware_id(firmware_actuator_id: str) -> str:
    """Map FLHY-style ids to USD revolute joint names (all six legs)."""
    leg = firmware_actuator_id[:2]
    segment = firmware_actuator_id[2:]
    segment_map = {
        "HY": "Body_CamShaft",
        "HL": "Hip_Femur",
        "KL": "Femur_Tibia",
    }
    return f"{leg}_{segment_map[segment]}_RevoluteJoint"


def isaac_body_hip_joint_name_for_firmware_id(firmware_actuator_id: str) -> str:
    """Passive ``Body_Hip`` DOF on the plant USD (paired with each ``*HY`` cam shaft)."""
    leg = firmware_actuator_id[:2]
    if firmware_actuator_id[2:] != "HY":
        raise ValueError(f"not a yaw actuator id: {firmware_actuator_id!r}")
    return f"{leg}_Body_Hip_RevoluteJoint"


# ``assets/crab_hex_ref.usd`` (HAL / joystick task) — URDF-style prefixes, 18 actuated DOFs.
_CRAB_HEX_REF_LEG_PREFIX = {
    "FL": "lf",
    "FR": "rf",
    "ML": "lm",
    "MR": "rm",
    "RL": "lr",
    "RR": "rr",
}
_CRAB_HEX_REF_SEGMENT_SUFFIX = {
    "HY": "hip_yaw",
    "HL": "hip_pitch",
    "KL": "knee_actuator_joint",
}


def isaac_joint_name_for_firmware_id_crab_hex_ref(firmware_actuator_id: str) -> str:
    """Map FLHY-style ids to ``crab_hex_ref.usd`` joint names (``lf_hip_yaw``, …)."""
    leg = firmware_actuator_id[:2]
    segment = firmware_actuator_id[2:]
    prefix = _CRAB_HEX_REF_LEG_PREFIX[leg]
    suffix = _CRAB_HEX_REF_SEGMENT_SUFFIX[segment]
    return f"{prefix}_{suffix}"


def _rev3_slots_for_firmware_ids(
    firmware_ids: Sequence[str],
) -> tuple[SimulatedSlotConfig, ...]:
    drive = (
        (2, 3, 22),
        (4, 5, 24),
        (6, 7, 26),
        (8, 9, 23),
        (10, 11, 25),
        (12, 13, 27),
    )
    pots = (0, 1, 2, 3, 4, 5)
    iss = (6, 7, 8, 9, 10, 11)
    return tuple(
        SimulatedSlotConfig(
            slot_index=index,
            pin_pwm_r=drive[index][0],
            pin_pwm_l=drive[index][1],
            pin_en=drive[index][2],
            pin_pot=pots[index],
            pin_is=iss[index],
            firmware_actuator_id=firmware_id,
            isaac_joint_name=isaac_joint_name_for_firmware_id(firmware_id),
        )
        for index, firmware_id in enumerate(firmware_ids)
    )


# FRONT board, REV 3 - mirrors ACT_LIST_FRONT order in arduino.ino
_FRONT_BOARD_REV3_SLOTS: tuple[SimulatedSlotConfig, ...] = (
    SimulatedSlotConfig(0, 2, 3, 22, 0, 6, "FLHY", "FL_Body_CamShaft_RevoluteJoint"),
    SimulatedSlotConfig(1, 4, 5, 24, 1, 7, "FLHL", "FL_Hip_Femur_RevoluteJoint"),
    SimulatedSlotConfig(2, 6, 7, 26, 2, 8, "FLKL", "FL_Femur_Tibia_RevoluteJoint"),
    SimulatedSlotConfig(3, 8, 9, 23, 3, 9, "FRHY", "FR_Body_CamShaft_RevoluteJoint"),
    SimulatedSlotConfig(4, 10, 11, 25, 4, 10, "FRHL", "FR_Hip_Femur_RevoluteJoint"),
    SimulatedSlotConfig(5, 12, 13, 27, 5, 11, "FRKL", "FR_Femur_Tibia_RevoluteJoint"),
)

_FRONT_BOARD_FIRMWARE_IDS = tuple(s.firmware_actuator_id for s in _FRONT_BOARD_REV3_SLOTS)

# LEFT / RIGHT - ACT_LIST_LEFT / ACT_LIST_RIGHT in arduino.ino
_LEFT_BOARD_FIRMWARE_IDS = ("RLHY", "RLHL", "RLKL", "MLHY", "MLHL", "MLKL")
_RIGHT_BOARD_FIRMWARE_IDS = ("RRHY", "RRHL", "RRKL", "MRHY", "MRHL", "MRKL")
_LEFT_BOARD_REV3_SLOTS = _rev3_slots_for_firmware_ids(_LEFT_BOARD_FIRMWARE_IDS)
_RIGHT_BOARD_REV3_SLOTS = _rev3_slots_for_firmware_ids(_RIGHT_BOARD_FIRMWARE_IDS)
_SLOTS_BY_ROLE: dict[str, tuple[SimulatedSlotConfig, ...]] = {
    "FRONT": _FRONT_BOARD_REV3_SLOTS,
    "LEFT": _LEFT_BOARD_REV3_SLOTS,
    "RIGHT": _RIGHT_BOARD_REV3_SLOTS,
}


def board_id_for_firmware_id(firmware_actuator_id: str) -> int:
    """Host sim board (0 FRONT, 1 LEFT, 2 RIGHT) whose firmware drives this actuator."""
    for role, slots in _SLOTS_BY_ROLE.items():
        if any(slot.firmware_actuator_id == firmware_actuator_id for slot in slots):
            return BOARD_ID_BY_ROLE[role]
    raise KeyError(f"unknown firmware actuator id: {firmware_actuator_id!r}")


def default_hall_edges_per_stroke(firmware_actuator_id: str) -> int:
    """Plant stroke of a hall slot: hips DEFAULT_HALL_HIP_EDGES_PER_STROKE, else the knee's."""
    if firmware_actuator_id[2:] == "HL":
        return DEFAULT_HALL_HIP_EDGES_PER_STROKE
    return DEFAULT_HALL_EDGES_PER_STROKE


def _with_sensors(
    slots: Sequence[SimulatedSlotConfig],
    sensor_for_id: Callable[[str], str],
    hall_edges_per_stroke: int | None,
) -> tuple[SimulatedSlotConfig, ...]:
    """``hall_edges_per_stroke`` None gives each slot its default (hip or knee)."""
    return tuple(
        replace(
            slot,
            sensor=sensor_for_id(slot.firmware_actuator_id),
            hall_edges_per_stroke=(
                default_hall_edges_per_stroke(slot.firmware_actuator_id)
                if hall_edges_per_stroke is None
                else hall_edges_per_stroke
            ),
        )
        for slot in slots
    )

# Leg-major order aligned with hal/server/robot_definition_krabby_hex.py mcu_joints naming.
KRABBY_HEX_FIRMWARE_ACTUATOR_IDS: tuple[str, ...] = (
    "FLHY",
    "FLHL",
    "FLKL",
    "FRHY",
    "FRHL",
    "FRKL",
    "MLHY",
    "MLHL",
    "MLKL",
    "MRHY",
    "MRHL",
    "MRKL",
    "RLHY",
    "RLHL",
    "RLKL",
    "RRHY",
    "RRHL",
    "RRKL",
)

PIN_REVISION_LABEL = "PINS_REV3_UNO_V02"  # boardPinRevisionLabel() when KRABBY_PIN_REV=3
KRABBY_PIN_REV = 3

# Per-leg Isaac find_joints-style patterns (exclude CamShaft by using exact names from slots).
ISAAC_LEG_JOINT_PATTERN = "{leg}_.*_RevoluteJoint"


@dataclass(frozen=True, slots=True)
class SimulatedRobotConfig:
    """Six-slot FRONT-board robot (open-loop drive path; sensor readback wired separately).

    ``quadrature`` is what ``hallHwHasQuadrature()`` reports on this board in the
    host sim. It is off by default, so the pot tests and goldens keep their meaning.
    """

    pin_revision_label: str
    board_role: str
    slots: tuple[SimulatedSlotConfig, ...]
    pot_bands: tuple[PotCalibrationBand, ...]
    hall_edges_per_full_travel: int = DEFAULT_HALL_EDGES_PER_FULL_TRAVEL
    quadrature: bool = False

    @classmethod
    def _board_rev3(
        cls,
        *,
        board_role: str,
        slots: tuple[SimulatedSlotConfig, ...],
        pot_calibration_seed: int = DEFAULT_POT_CALIBRATION_SEED,
        hall_edges_per_full_travel: int = DEFAULT_HALL_EDGES_PER_FULL_TRAVEL,
    ) -> SimulatedRobotConfig:
        bands = draw_pot_calibration_bands(len(slots), pot_calibration_seed)
        return cls(
            pin_revision_label=PIN_REVISION_LABEL,
            board_role=board_role,
            slots=slots,
            pot_bands=bands,
            hall_edges_per_full_travel=hall_edges_per_full_travel,
        )

    @classmethod
    def front_board_rev3(
        cls,
        *,
        pot_calibration_seed: int = DEFAULT_POT_CALIBRATION_SEED,
        hall_edges_per_full_travel: int = DEFAULT_HALL_EDGES_PER_FULL_TRAVEL,
    ) -> SimulatedRobotConfig:
        return cls._board_rev3(
            board_role="FRONT",
            slots=_FRONT_BOARD_REV3_SLOTS,
            pot_calibration_seed=pot_calibration_seed,
            hall_edges_per_full_travel=hall_edges_per_full_travel,
        )

    @classmethod
    def left_board_rev3(
        cls,
        *,
        pot_calibration_seed: int = DEFAULT_POT_CALIBRATION_SEED,
        hall_edges_per_full_travel: int = DEFAULT_HALL_EDGES_PER_FULL_TRAVEL,
    ) -> SimulatedRobotConfig:
        return cls._board_rev3(
            board_role="LEFT",
            slots=_LEFT_BOARD_REV3_SLOTS,
            pot_calibration_seed=pot_calibration_seed,
            hall_edges_per_full_travel=hall_edges_per_full_travel,
        )

    @classmethod
    def right_board_rev3(
        cls,
        *,
        pot_calibration_seed: int = DEFAULT_POT_CALIBRATION_SEED,
        hall_edges_per_full_travel: int = DEFAULT_HALL_EDGES_PER_FULL_TRAVEL,
    ) -> SimulatedRobotConfig:
        return cls._board_rev3(
            board_role="RIGHT",
            slots=_RIGHT_BOARD_REV3_SLOTS,
            pot_calibration_seed=pot_calibration_seed,
            hall_edges_per_full_travel=hall_edges_per_full_travel,
        )

    @classmethod
    def rev2_pot(
        cls,
        *,
        board_role: str = "FRONT",
        pot_calibration_seed: int = DEFAULT_POT_CALIBRATION_SEED,
    ) -> SimulatedRobotConfig:
        """Every slot a pot, no quadrature halls (Rev 2 behaviour on the Rev 3 pin map)."""
        return cls._board_rev3(
            board_role=board_role,
            slots=_SLOTS_BY_ROLE[board_role],
            pot_calibration_seed=pot_calibration_seed,
        )

    @classmethod
    def rev3_hall(
        cls,
        *,
        board_role: str = "FRONT",
        pot_calibration_seed: int = DEFAULT_POT_CALIBRATION_SEED,
        hall_edges_per_stroke: int | None = None,
    ) -> SimulatedRobotConfig:
        """Every slot a hall encoder (yaws too) on a quadrature board, as on the robot.

        ``hall_edges_per_stroke`` sets every slot's plant stroke; None gives hips
        DEFAULT_HALL_HIP_EDGES_PER_STROKE and the other slots DEFAULT_HALL_EDGES_PER_STROKE.
        """
        slots = _with_sensors(
            _SLOTS_BY_ROLE[board_role],
            lambda _aid: SENSOR_HALL,
            hall_edges_per_stroke,
        )
        return replace(
            cls._board_rev3(
                board_role=board_role,
                slots=slots,
                pot_calibration_seed=pot_calibration_seed,
            ),
            quadrature=True,
        )

    @classmethod
    def mixed(
        cls,
        *,
        board_role: str = "FRONT",
        pot_calibration_seed: int = DEFAULT_POT_CALIBRATION_SEED,
        hall_edges_per_stroke: int | None = None,
    ) -> SimulatedRobotConfig:
        """Quadrature board with one leg of each kind: first leg hall, second leg pot."""
        slots = _SLOTS_BY_ROLE[board_role]
        hall_leg = slots[0].firmware_actuator_id[:2]
        mixed_slots = _with_sensors(
            slots,
            lambda aid: SENSOR_HALL if aid[:2] == hall_leg else SENSOR_POT,
            hall_edges_per_stroke,
        )
        return replace(
            cls._board_rev3(
                board_role=board_role,
                slots=mixed_slots,
                pot_calibration_seed=pot_calibration_seed,
            ),
            quadrature=True,
        )

    @property
    def board_id(self) -> int:
        """Host sim board id (0 FRONT, 1 LEFT, 2 RIGHT)."""
        return BOARD_ID_BY_ROLE[self.board_role]

    def hall_slots(self) -> tuple[SimulatedSlotConfig, ...]:
        return tuple(slot for slot in self.slots if slot.is_hall)

    def pot_slots(self) -> tuple[SimulatedSlotConfig, ...]:
        return tuple(slot for slot in self.slots if not slot.is_hall)

    def pot_band_for_slot(self, slot_index: int) -> PotCalibrationBand:
        for slot, band in zip(self.slots, self.pot_bands, strict=True):
            if slot.slot_index == slot_index:
                return band
        raise KeyError(f"unknown slot index: {slot_index}")

    def isaac_joint_names(self) -> tuple[str, ...]:
        return tuple(s.isaac_joint_name for s in self.slots)

    def firmware_actuator_ids(self) -> tuple[str, ...]:
        return tuple(s.firmware_actuator_id for s in self.slots)

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


@dataclass(frozen=True, slots=True)
class ThreeBoardSimConfigs:
    """FRONT + LEFT + RIGHT REV3 configs (18 joints, disjoint Isaac DOFs)."""

    front: SimulatedRobotConfig
    left: SimulatedRobotConfig
    right: SimulatedRobotConfig

    @classmethod
    def rev3(
        cls,
        *,
        pot_calibration_seed: int = DEFAULT_POT_CALIBRATION_SEED,
        hall_edges_per_full_travel: int = DEFAULT_HALL_EDGES_PER_FULL_TRAVEL,
    ) -> ThreeBoardSimConfigs:
        kwargs = {
            "pot_calibration_seed": pot_calibration_seed,
            "hall_edges_per_full_travel": hall_edges_per_full_travel,
        }
        return cls(
            front=SimulatedRobotConfig.front_board_rev3(**kwargs),
            left=SimulatedRobotConfig.left_board_rev3(**kwargs),
            right=SimulatedRobotConfig.right_board_rev3(**kwargs),
        )

    def all_slots(self) -> tuple[SimulatedSlotConfig, ...]:
        return self.front.slots + self.left.slots + self.right.slots

    def firmware_actuator_ids(self) -> tuple[str, ...]:
        return tuple(slot.firmware_actuator_id for slot in self.all_slots())

    def validate_eighteen_distinct_joints(self) -> None:
        names = self.firmware_actuator_ids()
        joints = tuple(slot.isaac_joint_name for slot in self.all_slots())
        if len(set(names)) != 18:
            raise ValueError(f"expected 18 firmware ids, got {names!r}")
        if len(set(joints)) != 18:
            raise ValueError(f"expected 18 distinct Isaac joints, got {joints!r}")
        if set(names) != set(KRABBY_HEX_FIRMWARE_ACTUATOR_IDS):
            raise ValueError("firmware ids do not cover KRABBY_HEX set")

    def actuators(
        self,
        joint_index_for_name: Mapping[str, int] | Callable[[str], int],
        *,
        max_joint_speed: float = 1.0,
    ) -> tuple["SimulatedActuator", ...]:
        out: list[SimulatedActuator] = []
        for board_cfg in (self.front, self.left, self.right):
            out.extend(
                board_cfg.actuators(
                    joint_index_for_name,
                    max_joint_speed=max_joint_speed,
                )
            )
        return tuple(out)

    def slot_by_firmware_id(self, firmware_actuator_id: str) -> SimulatedSlotConfig:
        for board_cfg in (self.front, self.left, self.right):
            try:
                return board_cfg.slot_by_firmware_id(firmware_actuator_id)
            except KeyError:
                continue
        raise KeyError(f"unknown firmware actuator id: {firmware_actuator_id!r}")
