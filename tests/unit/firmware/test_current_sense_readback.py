"""Current-sense analogRead sync from wrench + EN + joint velocity."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from firmware.sim.current_sense_readback import sync_current_sense_analog_reads
from firmware.sim.plant_readback import PlantReadbackView
from firmware.sim.simulated_robot_config import SimulatedRobotConfig
from tests.unit.firmware.plant_readback_fixtures import TensorCell, articulation_with_joint_travel


def test_sync_current_sense_sets_pin_is_from_wrench_and_en():
    bridge = MagicMock()
    bridge.pin_digital.return_value = 1
    config = SimulatedRobotConfig.front_board_rev3()
    indices = {name: i for i, name in enumerate(config.isaac_joint_names())}
    actuators = config.actuators(indices)
    flhl = actuators[1]
    row = [TensorCell(0.0)] * 6
    row[2] = TensorCell(150.0)
    torque_row = [TensorCell(0.0) for _ in range(len(indices))]
    articulation = articulation_with_joint_travel(
        len(indices),
        flhl.joint_index,
        position=0.0,
        lower=-1.0,
        upper=1.0,
        joint_velocity=0.0,
        extra_data=SimpleNamespace(
            body_incoming_joint_wrench_b=[[row]],
            applied_torque=[torque_row],
        ),
    )
    plant = PlantReadbackView.from_articulation(articulation)
    assert plant is not None

    sync_current_sense_analog_reads(
        bridge,
        actuators,
        plant,
        {"FLHL": 0},
    )
    bridge.set_analog_read.assert_any_call(flhl.slot.pin_is, 30)
