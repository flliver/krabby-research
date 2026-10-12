"""Read ``body_incoming_joint_wrench_b`` at each FRONT actuator's child link."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from firmware.sim.simulated_robot_config import SimulatedRobotConfig, SimulatedSlotConfig


@dataclass(frozen=True, slots=True)
class WrenchSample:
    """Spatial wrench on a link inbound joint in the parent body frame (Isaac Lab)."""

    fx: float
    fy: float
    fz: float
    tx: float
    ty: float
    tz: float

    def force_magnitude(self) -> float:
        return math.sqrt(self.fx * self.fx + self.fy * self.fy + self.fz * self.fz)

    def torque_magnitude(self) -> float:
        return math.sqrt(self.tx * self.tx + self.ty * self.ty + self.tz * self.tz)


def wrench_body_name_for_joint(isaac_joint_name: str) -> str:
    """Child link body for the joint that the firmware slot drives (``find_bodies`` name)."""
    leg = isaac_joint_name.split("_", 1)[0]
    if "_Body_Hip_RevoluteJoint" in isaac_joint_name:
        return f"{leg}_Hip"
    if "_Hip_Femur_RevoluteJoint" in isaac_joint_name:
        return f"{leg}_Femur"
    if "_Femur_Tibia_RevoluteJoint" in isaac_joint_name:
        return f"{leg}_Tibia"
    raise ValueError(f"unsupported Isaac joint for wrench readback: {isaac_joint_name!r}")


def wrench_body_name_for_slot(slot: SimulatedSlotConfig) -> str:
    return wrench_body_name_for_joint(slot.isaac_joint_name)


def resolve_wrench_body_indices(
    robot: Any,
    config: SimulatedRobotConfig,
) -> dict[str, int]:
    """Map firmware actuator id -> ``body_incoming_joint_wrench_b`` body index."""
    names = [wrench_body_name_for_slot(slot) for slot in config.slots]
    body_ids, resolved = robot.find_bodies(names, preserve_order=True)
    if len(body_ids) != len(names):
        raise KeyError(
            f"find_bodies expected {len(names)} bodies, got {len(body_ids)} "
            f"for {names!r} (resolved {list(resolved)!r})"
        )
    out: dict[str, int] = {}
    for slot, body_id in zip(config.slots, body_ids, strict=True):
        idx = int(body_id.item()) if hasattr(body_id, "item") else int(body_id)
        out[slot.firmware_actuator_id] = idx
    return out


def read_body_incoming_wrench(
    robot: Any,
    body_index: int,
    *,
    env_index: int = 0,
) -> WrenchSample:
    data = getattr(robot, "data", None)
    if data is None:
        raise AttributeError("articulation has no data")
    wrench = getattr(data, "body_incoming_joint_wrench_b", None)
    if wrench is None:
        raise AttributeError("articulation.data.body_incoming_joint_wrench_b missing")
    try:
        row = wrench[env_index, body_index]
    except TypeError:
        row = wrench[env_index][body_index]
    return WrenchSample(
        fx=float(row[0].item()),
        fy=float(row[1].item()),
        fz=float(row[2].item()),
        tx=float(row[3].item()),
        ty=float(row[4].item()),
        tz=float(row[5].item()),
    )


def sample_actuator_wrenches(
    robot: Any,
    body_index_by_actuator: Mapping[str, int],
    *,
    env_index: int = 0,
) -> dict[str, WrenchSample]:
    return {
        actuator_id: read_body_incoming_wrench(robot, body_index, env_index=env_index)
        for actuator_id, body_index in body_index_by_actuator.items()
    }


def max_actuator_wrench_force(
    samples: Mapping[str, WrenchSample],
    actuator_ids: Sequence[str] | None = None,
) -> float:
    ids = actuator_ids if actuator_ids is not None else list(samples.keys())
    return max((samples[aid].force_magnitude() for aid in ids if aid in samples), default=0.0)
