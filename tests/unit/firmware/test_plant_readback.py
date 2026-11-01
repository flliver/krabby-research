"""``PlantReadbackView`` factory and joint sampling."""

from firmware.sim.plant_readback import PlantReadbackView
from tests.unit.firmware.plant_readback_fixtures import articulation_with_joint_travel


def test_from_articulation_returns_none_without_data():
    assert PlantReadbackView.from_articulation(object()) is None


def test_joint_travel_normalizes_limits():
    articulation = articulation_with_joint_travel(
        4,
        2,
        position=0.25,
        lower=-0.5,
        upper=0.75,
    )
    plant = PlantReadbackView.from_articulation(articulation)
    assert plant is not None
    assert plant.joint_travel(2) == (0.25, -0.5, 0.75)


def test_joint_velocity_rad_s():
    articulation = articulation_with_joint_travel(
        4,
        1,
        position=0.0,
        lower=-1.0,
        upper=1.0,
        joint_velocity=1.5,
    )
    plant = PlantReadbackView.from_articulation(articulation)
    assert plant is not None
    assert plant.joint_velocity_rad_s(1) == 1.5
