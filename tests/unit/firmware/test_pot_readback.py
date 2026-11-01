"""Pot analogRead sync from articulation joint state."""

from unittest.mock import MagicMock

from firmware.sim.plant_readback import PlantReadbackView
from firmware.sim.pot_position import joint_angle_to_pot_adc
from firmware.sim.pot_readback import sync_pot_analog_reads
from firmware.sim.simulated_robot_config import SimulatedRobotConfig
from tests.unit.firmware.plant_readback_fixtures import articulation_with_joint_travel


def test_sync_pot_analog_reads_writes_expected_adc():
    bridge = MagicMock()
    config = SimulatedRobotConfig.front_board_rev3()
    indices = {name: i for i, name in enumerate(config.isaac_joint_names())}
    actuators = config.actuators(indices)
    flhy = actuators[0]
    articulation = articulation_with_joint_travel(
        len(indices),
        flhy.joint_index,
        position=0.0,
        lower=-1.0,
        upper=1.0,
    )
    plant = PlantReadbackView.from_articulation(articulation)

    sync_pot_analog_reads(bridge, actuators, plant)

    expected = joint_angle_to_pot_adc(0.0, -1.0, 1.0, band=flhy.pot_band)
    bridge.set_analog_read.assert_any_call(flhy.slot.pin_pot, expected)
