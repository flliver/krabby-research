"""Minimal articulation doubles for ``PlantReadbackView`` unit tests."""

from __future__ import annotations

from types import SimpleNamespace


class TensorCell:
    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


def articulation_with_joint_travel(
    num_joints: int,
    joint_index: int,
    *,
    position: float,
    lower: float,
    upper: float,
    joint_velocity: float | None = None,
    extra_data: SimpleNamespace | None = None,
) -> SimpleNamespace:
    joint_pos_row = [TensorCell(0.0) for _ in range(num_joints)]
    joint_pos_row[joint_index] = TensorCell(position)
    limits_row = [[TensorCell(-1.0), TensorCell(1.0)] for _ in range(num_joints)]
    limits_row[joint_index] = [TensorCell(lower), TensorCell(upper)]
    joint_vel_row = [TensorCell(0.0) for _ in range(num_joints)]
    if joint_velocity is not None:
        joint_vel_row[joint_index] = TensorCell(joint_velocity)
    data = SimpleNamespace(
        joint_pos=[joint_pos_row],
        soft_joint_pos_limits=[limits_row],
        joint_vel=[joint_vel_row],
    )
    if extra_data is not None:
        for key, value in vars(extra_data).items():
            setattr(data, key, value)
    return SimpleNamespace(data=data)
