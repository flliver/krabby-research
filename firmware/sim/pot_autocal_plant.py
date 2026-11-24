"""Travel-fraction calibration plant for USB auto-cal (pot and hall actuators).

Each actuator has a travel fraction: 0 fully retracted, 1 fully extended. +PWM at
or above ``AUTOCAL_PWM_DRIVE_THRESHOLD`` extends it by one step per firmware loop,
-PWM retracts it, and it stops at both stroke ends.

- A pot actuator's pot pin reads its band ADC at that travel.
- A hall actuator counts signed edges on its board (retract counts up, extend counts
  down, as on the robot). No edge is counted once a stroke end stops the travel, and
  the count saturates at the int32 range. Its pot pin carries hall B: one B edge
  falls between two counted A edges, so the level flips with every edge and rests
  near ``HALL_B_REST_LOW_ADC`` or ``HALL_B_REST_HIGH_ADC``.
- Every actuator's current-sense pin (``pin_is``) reads what the 2026-11-24 end
  current runs measured, which the firmware's calibration end detector needs: 0
  undriven, ``CAL_PLANT_IS_RUN_ADC`` while driven, ``CAL_PLANT_IS_SWITCH_OPEN_ADC``
  driven back into the retracted end (the actuator's limit switch opens) and
  ``CAL_PLANT_IS_STALL_ADC`` pressed into an extended end that stops it (the
  stroke end, an ``extend_stops`` obstacle, a stored knee's curl end).

Two optional models make ``CalibrationTravelPlant`` behave like the robot where the
calibration depends on it (both off by default, so the goldens keep their meaning):

- ``StoredKneeModel``: a stored hall knee's compliant curl end, measured on FLKL
  (2026-11-23): it needs a while to break free, springs out after a nudge stops,
  and a PWM 200 curl parks it deeper than a PWM 80 curl reaches.
- ``extend_stops``: an obstacle (or the curled tibia meeting the body) that stops
  a joint's extension at a travel fraction.

``tests/unit/firmware/test_host_auto_calibration_plant.py`` runs the host firmware
against this plant; ``RobotCore`` uses it while a calibration runs in Isaac.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Protocol, Sequence

from firmware.sim.hall_encoder import HALL_EXTEND_COUNT_SIGN
from firmware.sim.pin_state import PinStateModel
from firmware.sim.pot_calibration import PotCalibrationBand
from firmware.sim.simulated_actuator import SimulatedActuator
from firmware.sim.simulated_robot_config import SimulatedRobotConfig, SimulatedSlotConfig

# Pot plant step per 10 ms firmware loop: a full stroke in about 0.17 s. The pot
# goldens on boards without quadrature halls are built on this value.
AUTOCAL_TRAVEL_STEP = 0.06
# Step on a quadrature board: a full stroke in about 6.7 s, a half stroke in 3.3 s.
# An unclassified pot's first move there is windowed on hall edges and stops after
# about 2.7 s; it must still be short of its curl end then, or the repeat with pot
# windows sees no motion: a knee is then nudged (the shipped default; a strict run,
# options 0, fails it NoMotion).
AUTOCAL_QUADRATURE_TRAVEL_STEP = 0.0015
# Below the slowest calibration speed (CAL_PWM_MIN = 40 in calibration_constants.h),
# so ``C 40`` moves the plant too; the firmware ramps |PWM| up from 0 in steps of 5.
AUTOCAL_PWM_DRIVE_THRESHOLD = 20

# Hall B rest levels seen on the robot (CAL_HALLB_REST_LOW_MAX / HIGH_MIN are 19 / 1013).
HALL_B_REST_LOW_ADC = 19
HALL_B_REST_HIGH_ADC = 1016

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1

# One host firmware loop (the plant advances once per loop).
PLANT_TICK_MS = 10

# Current-sense ADC (avgIS units) of a driven actuator: running, driven into its
# retracted end (limit switch open: 7-11 on the robot) and pressed into an
# extended end (13-24 at a knee's leg stop), 2026-11-24.
CAL_PLANT_IS_RUN_ADC = 12
CAL_PLANT_IS_SWITCH_OPEN_ADC = 8
CAL_PLANT_IS_STALL_ADC = 18


@dataclass(frozen=True, slots=True)
class StoredKneeModel:
    """Curl end of a stored hall knee, from the FLKL runs of 2026-11-23 (PWM 80, ramped).

    Distances are hall edges out from the deepest park (travel 1.0, reached only by a
    strong curl such as the GUI's PWM 200 jog).
    """

    # A curl below ``strong_curl_pwm`` stops this far out: a PWM 200 curl parks the
    # knee about 17 edges deeper than PWM 80 reaches.
    weak_curl_end_edges: int = 17
    strong_curl_pwm: int = 150
    # Drive time (at or above the plant's drive threshold) before a knee resting at
    # its curl end starts to move out: the first count came 0.36-0.77 s after the
    # drive started, ramp included.
    breakaway_ms: int = 700
    # After an outward drive stops within ``spring_zone_edges`` of the curl end, the
    # knee springs out this many more edges (5-9 measured; a 12-edge nudge ended
    # 18-21 edges out).
    spring_edges: int = 9
    spring_zone_edges: int = 60


# The knee measured on the robot (FLKL).
ROBOT_STORED_KNEE = StoredKneeModel()


class HallCountBridge(Protocol):
    def board_hall_edge_count(self, board_id: int, hall_slot: int) -> int: ...

    def board_set_hall_edge_count(self, board_id: int, hall_slot: int, count: int) -> None: ...


def travel_step_for_config(config: SimulatedRobotConfig | None) -> float:
    """Plant step per firmware loop: slower on quadrature boards (see the constants)."""
    if config is not None and config.quadrature:
        return AUTOCAL_QUADRATURE_TRAVEL_STEP
    return AUTOCAL_TRAVEL_STEP


def advance_travel(frac: float, signed_pwm: int, step: float) -> float:
    """One plant step: extend on +PWM, retract on -PWM, stop at the stroke ends."""
    if signed_pwm >= AUTOCAL_PWM_DRIVE_THRESHOLD:
        return min(1.0, frac + step)
    if signed_pwm <= -AUTOCAL_PWM_DRIVE_THRESHOLD:
        return max(0.0, frac - step)
    return frac


def stroke_edges(travel: float, edges_per_stroke: int) -> int:
    """Edges from the retracted end at ``travel`` (clamped to the stroke)."""
    return int(round(min(1.0, max(0.0, travel)) * edges_per_stroke))


def hall_count_delta(old_travel: float, new_travel: float, edges_per_stroke: int) -> int:
    """Signed count change for a travel change: extending counts down."""
    edges = stroke_edges(new_travel, edges_per_stroke) - stroke_edges(old_travel, edges_per_stroke)
    return HALL_EXTEND_COUNT_SIGN * edges


def saturating_count(count: int, delta: int) -> int:
    return max(INT32_MIN, min(INT32_MAX, int(count) + int(delta)))


def hall_b_level(count: int) -> int:
    """Hall B ADC level after ``count`` edges: it flips with every counted edge."""
    return HALL_B_REST_HIGH_ADC if int(count) & 1 else HALL_B_REST_LOW_ADC


def add_hall_edges(bridge: HallCountBridge, board_id: int, hall_slot: int, delta: int) -> int:
    """Add ``delta`` to a board's hall count (saturating); returns the new count.

    Read-modify-write, so a firmware ``hallHwInit()`` that zeroed the board stays in force.
    """
    count = bridge.board_hall_edge_count(board_id, hall_slot)
    if delta == 0:
        return count
    count = saturating_count(count, delta)
    bridge.board_set_hall_edge_count(board_id, hall_slot, count)
    return count


def calibration_is_adc(signed_pwm: int, old_travel: float, new_travel: float) -> int:
    """Current-sense ADC after one plant step of a drive of ``signed_pwm``."""
    if signed_pwm == 0:
        return 0
    if signed_pwm < 0 and new_travel <= 0.0:
        return CAL_PLANT_IS_SWITCH_OPEN_ADC
    if signed_pwm >= AUTOCAL_PWM_DRIVE_THRESHOLD and new_travel <= old_travel:
        return CAL_PLANT_IS_STALL_ADC
    return CAL_PLANT_IS_RUN_ADC


def default_travel_fractions(actuators: Sequence[SimulatedActuator]) -> dict[str, float]:
    return {a.slot.firmware_actuator_id: 0.5 for a in actuators}


def pot_pin_adc(
    slot: SimulatedSlotConfig,
    band: PotCalibrationBand,
    frac: float,
    hall_count: int = 0,
) -> int:
    """Pot pin level: the band ADC for a pot, the hall B level for a hall."""
    if slot.is_hall:
        return hall_b_level(hall_count)
    return int(round(band.min_adc + frac * (band.max_adc - band.min_adc)))


def sync_pot_reads_from_travel(
    bridge,
    actuators: Sequence[SimulatedActuator],
    travel: Mapping[str, float],
    *,
    board_id: int = 0,
    currents: Mapping[str, int] | None = None,
) -> None:
    """Pot pins from the travel; with ``currents``, also each ``pin_is`` (0 if absent)."""
    for actuator in actuators:
        slot = actuator.slot
        if currents is not None:
            bridge.set_analog_read(slot.pin_is, currents.get(slot.firmware_actuator_id, 0))
        if slot.is_hall:
            count = bridge.board_hall_edge_count(board_id, slot.slot_index)
            bridge.set_analog_read(slot.pin_pot, hall_b_level(count))
            continue
        band = actuator.pot_band
        if band is None:
            continue
        frac = travel.get(slot.firmware_actuator_id, 0.5)
        bridge.set_analog_read(slot.pin_pot, pot_pin_adc(slot, band, frac))


def advance_travel_from_pwm(
    actuators: Sequence[SimulatedActuator],
    pin_state: PinStateModel,
    travel: MutableMapping[str, float],
    *,
    step: float = AUTOCAL_TRAVEL_STEP,
    bridge: HallCountBridge | None = None,
    board_id: int = 0,
    currents: MutableMapping[str, int] | None = None,
) -> None:
    """Move each actuator one step; with ``bridge``, hall actuators also count edges.

    With ``currents``, each actuator's current-sense ADC after the step is stored
    there (``calibration_is_adc``) for the next ``sync_pot_reads_from_travel``.
    """
    for actuator in actuators:
        cmd = actuator.signed_pwm_command(pin_state)
        slot = actuator.slot
        aid = slot.firmware_actuator_id
        frac = travel.get(aid, 0.5)
        moved = advance_travel(frac, cmd, step)
        travel[aid] = moved
        if currents is not None:
            currents[aid] = calibration_is_adc(cmd, frac, moved)
        if bridge is not None and slot.is_hall:
            delta = hall_count_delta(frac, moved, slot.hall_edges_per_stroke)
            add_hall_edges(bridge, board_id, slot.slot_index, delta)


class CalibrationTravelPlant:
    """One board's actuators driven by the host firmware's drive pins.

    ``sync_inputs`` writes the pot pins before a firmware loop; ``advance`` reads the
    drive pins after it, moves the travel and counts hall edges on the board.

    ``knee_model`` applies to the hall knees (``*KL``) only. ``extend_stops`` maps an
    actuator id to the travel fraction past which it cannot extend.
    """

    def __init__(
        self,
        config: SimulatedRobotConfig,
        *,
        travel: Mapping[str, float] | None = None,
        step: float | None = None,
        knee_model: StoredKneeModel | None = None,
        extend_stops: Mapping[str, float] | None = None,
    ) -> None:
        self.config = config
        self.board_id = config.board_id
        self.step = travel_step_for_config(config) if step is None else step
        self.travel = {slot.firmware_actuator_id: 0.5 for slot in config.slots}
        if travel is not None:
            self.travel.update(travel)
        self.knee_model = knee_model
        self.extend_stops = dict(extend_stops or {})
        self._driving_out = {slot.firmware_actuator_id: False for slot in config.slots}
        self._stuck_ms = {slot.firmware_actuator_id: 0 for slot in config.slots}
        # Current-sense ADC per actuator after the last step (0: undriven).
        self.current_adc = {slot.firmware_actuator_id: 0 for slot in config.slots}

    def slot_pwm(self, bridge, slot: SimulatedSlotConfig) -> int:
        pwm_r, pwm_l, pin_en = slot.drive_pins
        if bridge.pin_digital(pin_en) == 0:
            return 0
        return bridge.pin_pwm(pwm_r) - bridge.pin_pwm(pwm_l)

    def hall_count(self, bridge: HallCountBridge, actuator_id: str) -> int:
        slot = self.config.slot_by_firmware_id(actuator_id)
        return bridge.board_hall_edge_count(self.board_id, slot.slot_index)

    def pot_adc(self, bridge, slot: SimulatedSlotConfig) -> int:
        band = self.config.pot_band_for_slot(slot.slot_index)
        count = self.hall_count(bridge, slot.firmware_actuator_id) if slot.is_hall else 0
        return pot_pin_adc(slot, band, self.travel[slot.firmware_actuator_id], count)

    def sync_inputs(self, bridge) -> None:
        for slot in self.config.slots:
            bridge.set_analog_read(slot.pin_pot, self.pot_adc(bridge, slot))
            bridge.set_analog_read(slot.pin_is, self.current_adc[slot.firmware_actuator_id])

    def _is_modelled_knee(self, slot: SimulatedSlotConfig) -> bool:
        return (
            self.knee_model is not None
            and slot.is_hall
            and slot.firmware_actuator_id[2:] == "KL"
        )

    def _knee_move(self, slot: SimulatedSlotConfig, frac: float, cmd: int) -> float:
        """Travel after one step of a modelled knee (it curls by extending, toward 1.0)."""
        model = self.knee_model
        aid = slot.firmware_actuator_id
        stroke = slot.hall_edges_per_stroke
        weak_end = 1.0 - model.weak_curl_end_edges / stroke
        moved = advance_travel(frac, cmd, self.step)
        if cmd >= AUTOCAL_PWM_DRIVE_THRESHOLD:
            if cmd < model.strong_curl_pwm:
                # A weak curl stops at its end; a knee parked deeper stays put.
                moved = min(moved, max(frac, weak_end))
            self._stuck_ms[aid] = 0
            self._driving_out[aid] = False
        elif cmd <= -AUTOCAL_PWM_DRIVE_THRESHOLD:
            self._driving_out[aid] = True
            if frac >= weak_end and self._stuck_ms[aid] < model.breakaway_ms:
                # Resting at its curl end: it has to break free first.
                self._stuck_ms[aid] += PLANT_TICK_MS
                moved = frac
        else:
            self._stuck_ms[aid] = 0
            out_edges = (1.0 - frac) * stroke
            if self._driving_out[aid] and out_edges <= model.spring_zone_edges:
                # An outward drive just stopped near the curl end: the knee springs out.
                moved = max(0.0, frac - model.spring_edges / stroke)
            self._driving_out[aid] = False
        return moved

    def advance(self, bridge) -> dict[str, int]:
        """One plant step from the current drive pins; returns the signed PWM per actuator."""
        pwm: dict[str, int] = {}
        for slot in self.config.slots:
            aid = slot.firmware_actuator_id
            cmd = self.slot_pwm(bridge, slot)
            pwm[aid] = cmd
            frac = self.travel[aid]
            if self._is_modelled_knee(slot):
                moved = self._knee_move(slot, frac, cmd)
            else:
                moved = advance_travel(frac, cmd, self.step)
            stop = self.extend_stops.get(aid)
            if stop is not None and moved > frac:
                moved = max(frac, min(moved, stop))
            self.travel[aid] = moved
            self.current_adc[aid] = calibration_is_adc(cmd, frac, moved)
            if slot.is_hall:
                delta = hall_count_delta(frac, moved, slot.hall_edges_per_stroke)
                add_hall_edges(bridge, self.board_id, slot.slot_index, delta)
        return pwm
