"""Hall edge sync from articulation joint state."""

from unittest.mock import MagicMock

from firmware.sim.hall_readback import sync_hall_edge_counts
from firmware.sim.plant_readback import PlantReadbackView
from firmware.sim.simulated_robot_config import SimulatedRobotConfig
from tests.unit.firmware.plant_readback_fixtures import articulation_with_joint_travel


def test_sync_hall_edge_counts_calls_host_per_slot():
    bridge = MagicMock()
    config = SimulatedRobotConfig.front_board_rev3()
    indices = {name: i for i, name in enumerate(config.isaac_joint_names())}
    actuators = config.actuators(indices)
    flhl = actuators[1]
    articulation = articulation_with_joint_travel(
        len(indices),
        flhl.joint_index,
        position=0.25,
        lower=-0.5,
        upper=0.75,
    )
    plant = PlantReadbackView.from_articulation(articulation)

    sync_hall_edge_counts(bridge, actuators, plant)

    bridge.sync_hall_joint.assert_any_call(1, 0.25, -0.5, 0.75)
