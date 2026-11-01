"""Shared Isaac articulation snapshot for pot, hall, and current-sense readback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from firmware.sim.pot_position import normalize_joint_travel_limits


@dataclass(frozen=True, slots=True)
class PlantReadbackView:
    """One env row of joint buffers; avoids repeated duck-typing on the plant."""

    env_index: int
    joint_pos: Any
    joint_limits: Any
    joint_vel: Any | None
    articulation: Any

    @classmethod
    def from_articulation(
        cls,
        articulation: Any,
        *,
        env_index: int = 0,
    ) -> PlantReadbackView | None:
        data = getattr(articulation, "data", None)
        if data is None:
            return None
        joint_pos = getattr(data, "joint_pos", None)
        limits = getattr(data, "soft_joint_pos_limits", None)
        if joint_pos is None or limits is None:
            return None
        joint_vel = getattr(data, "joint_vel", None)
        return cls(
            env_index=env_index,
            joint_pos=joint_pos,
            joint_limits=limits,
            joint_vel=joint_vel,
            articulation=articulation,
        )

    def joint_travel(self, joint_index: int) -> tuple[float, float, float] | None:
        """``(position_rad, lower_rad, upper_rad)`` with normalized travel limits."""
        try:
            position = float(self.joint_pos[self.env_index][joint_index].item())
            lower = float(self.joint_limits[self.env_index][joint_index][0].item())
            upper = float(self.joint_limits[self.env_index][joint_index][1].item())
        except (AttributeError, IndexError, TypeError):
            return None
        lower, upper = normalize_joint_travel_limits(lower, upper)
        return position, lower, upper

    def joint_velocity_rad_s(self, joint_index: int) -> float | None:
        if self.joint_vel is None:
            return None
        try:
            return float(self.joint_vel[self.env_index][joint_index].item())
        except (AttributeError, IndexError, TypeError):
            return None
