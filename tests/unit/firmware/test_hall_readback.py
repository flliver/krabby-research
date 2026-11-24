"""Hall edge sync from articulation joint state (per board, signed in the host sim)."""

from unittest.mock import MagicMock

from firmware.sim.hall_readback import (
    retarget_hall_baselines_when_idle,
    sync_hall_b_levels,
    sync_hall_edge_counts,
)
from firmware.sim.pin_state import PinStateModel
from firmware.sim.plant_readback import PlantReadbackView
from firmware.sim.pot_autocal_plant import HALL_B_REST_HIGH_ADC, HALL_B_REST_LOW_ADC
from firmware.sim.simulated_robot_config import SimulatedRobotConfig
from tests.unit.firmware.plant_readback_fixtures import articulation_with_joint_travel


def _actuators(config):
    indices = {name: i for i, name in enumerate(config.isaac_joint_names())}
    return config.actuators(indices)


def test_sync_hall_edge_counts_calls_host_per_slot():
    bridge = MagicMock()
    config = SimulatedRobotConfig.front_board_rev3()
    actuators = _actuators(config)
    flhl = actuators[1]
    articulation = articulation_with_joint_travel(
        len(actuators),
        flhl.joint_index,
        position=0.25,
        lower=-0.5,
        upper=0.75,
    )
    plant = PlantReadbackView.from_articulation(articulation)

    sync_hall_edge_counts(bridge, actuators, plant)

    bridge.board_sync_hall_joint.assert_any_call(0, 1, 0.25, -0.5, 0.75)
    bridge.sync_hall_joint.assert_not_called()


def test_left_and_right_boards_sync_their_own_counts():
    bridge = MagicMock()
    for config, board_id in (
        (SimulatedRobotConfig.left_board_rev3(), 1),
        (SimulatedRobotConfig.right_board_rev3(), 2),
    ):
        actuators = _actuators(config)
        knee = actuators[5]
        articulation = articulation_with_joint_travel(
            len(actuators),
            knee.joint_index,
            position=0.1,
            lower=0.0,
            upper=1.0,
        )
        plant = PlantReadbackView.from_articulation(articulation)
        sync_hall_edge_counts(bridge, actuators, plant)
        bridge.board_sync_hall_joint.assert_any_call(board_id, 5, 0.1, 0.0, 1.0)


def test_retarget_uses_the_board_and_skips_driven_actuators():
    bridge = MagicMock()
    config = SimulatedRobotConfig.left_board_rev3()
    actuators = _actuators(config)
    driven = actuators[1]
    articulation = articulation_with_joint_travel(
        len(actuators),
        driven.joint_index,
        position=0.3,
        lower=0.0,
        upper=1.0,
    )
    plant = PlantReadbackView.from_articulation(articulation)
    pin_state = PinStateModel()
    pwm_r, _pwm_l, pin_en = driven.slot.drive_pins
    pin_state.digital_write(pin_en, 1)
    pin_state.analog_write(pwm_r, 80)

    retarget_hall_baselines_when_idle(bridge, actuators, plant, pin_state)

    slots = [call.args[1] for call in bridge.board_retarget_hall_joint.call_args_list]
    assert driven.slot.slot_index not in slots
    assert all(call.args[0] == 1 for call in bridge.board_retarget_hall_joint.call_args_list)


def test_hall_b_levels_follow_the_count_on_hall_slots_only():
    config = SimulatedRobotConfig.mixed()
    actuators = _actuators(config)
    counts = {0: 0, 1: 7, 2: -4, 3: 99, 4: 5, 5: 6}

    class Bridge:
        def __init__(self) -> None:
            self.pots: dict[int, int] = {}

        def board_hall_edge_count(self, board_id: int, slot: int) -> int:
            assert board_id == 0
            return counts[slot]

        def set_analog_read(self, pin: int, value: int) -> None:
            self.pots[pin] = value

    bridge = Bridge()
    sync_hall_b_levels(bridge, actuators)

    by_id = {a.slot.firmware_actuator_id: a.slot.pin_pot for a in actuators}
    assert bridge.pots == {
        by_id["FLHY"]: HALL_B_REST_LOW_ADC,
        by_id["FLHL"]: HALL_B_REST_HIGH_ADC,
        by_id["FLKL"]: HALL_B_REST_LOW_ADC,
    }
