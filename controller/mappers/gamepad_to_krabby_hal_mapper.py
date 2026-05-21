"""Mapper from gamepad controller state to HAL command format (for Krabby real hardware).

Maps ControllerState plus a HAL :class:`~hal.server.robot_definition.RobotDefinition`
(e.g. Krabby hex 18-DOF or Unitree Go2 12-DOF) to :class:`~hal.client.data_structures.hardware.JointCommand`.
"""

import logging
import time
from typing import Dict, Optional, Set

import numpy as np

from controller.input.state import ControllerState, LegIdentifier
from hal.client.data_structures.hardware import JointCommand, JointCommandSource
from hal.server.robot_definition import RobotDefinition
from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION

logger = logging.getLogger(__name__)


def _leg_joint_indices_for(robot_definition: RobotDefinition) -> Dict[LegIdentifier, tuple[int, int, int]]:
    """Global joint indices (hip_yaw, hip_pitch, knee) per controller leg enum, only if leg exists."""
    jt_len = len(robot_definition.joint_types)
    if jt_len != 3:
        raise ValueError(
            f"GamepadToKrabbyHALMapper expects 3 joint_types per leg, got {jt_len}"
        )
    out: dict[LegIdentifier, tuple[int, int, int]] = {}
    for leg in LegIdentifier:
        abbr = leg.value
        if abbr not in robot_definition.legs:
            continue
        leg_idx = robot_definition.legs.index(abbr)
        o = leg_idx * jt_len
        out[leg] = (o, o + 1, o + 2)
    return out


# Hex topology index map (backward compat / tests): FL…RR × (hip_yaw, hip_pitch, knee)
LEG_TO_JOINT_INDICES: Dict[LegIdentifier, tuple[int, int, int]] = _leg_joint_indices_for(
    KRABBY_HEX_DEFINITION
)

DEFAULT_HIP_UP_DOWN_SCALE = 0.3  # radians per unit (max ~0.3 rad = ~17 degrees)
DEFAULT_KNEE_OUT_IN_SCALE = 0.3
DEFAULT_HIP_YAW_SCALE = 0.2


class GamepadToKrabbyHALMapper:
    """Maps gamepad state to JointCommand sized for the given robot topology."""

    def __init__(
        self,
        hip_up_down_scale: float = DEFAULT_HIP_UP_DOWN_SCALE,
        knee_out_in_scale: float = DEFAULT_KNEE_OUT_IN_SCALE,
        hip_yaw_scale: float = DEFAULT_HIP_YAW_SCALE,
        *,
        robot_definition: Optional[RobotDefinition] = None,
    ):
        self.hip_up_down_scale = hip_up_down_scale
        self.knee_out_in_scale = knee_out_in_scale
        self.hip_yaw_scale = hip_yaw_scale
        self._robot_definition = robot_definition or KRABBY_HEX_DEFINITION
        self._leg_joint_indices = _leg_joint_indices_for(self._robot_definition)

    def _select_legs(self, state: ControllerState) -> Set[LegIdentifier]:
        select_FL = state.LT and not state.LB
        select_RL = state.LB and not state.LT
        select_ML = state.LS
        select_MR = state.RS
        select_FR = state.RT and not state.RB
        select_RR = state.RB and not state.RT

        combo_left = state.LT and state.LB
        combo_right = state.RT and state.RB

        legs = set()
        if combo_left:
            legs |= {
                LegIdentifier.FRONT_LEFT,
                LegIdentifier.REAR_LEFT,
                LegIdentifier.MIDDLE_RIGHT,
            }
        if combo_right:
            legs |= {
                LegIdentifier.FRONT_RIGHT,
                LegIdentifier.REAR_RIGHT,
                LegIdentifier.MIDDLE_LEFT,
            }
        if not combo_left and not combo_right:
            if select_FL:
                legs.add(LegIdentifier.FRONT_LEFT)
            if select_RL:
                legs.add(LegIdentifier.REAR_LEFT)
            if select_ML:
                legs.add(LegIdentifier.MIDDLE_LEFT)
            if select_MR:
                legs.add(LegIdentifier.MIDDLE_RIGHT)
            if select_FR:
                legs.add(LegIdentifier.FRONT_RIGHT)
            if select_RR:
                legs.add(LegIdentifier.REAR_RIGHT)

        return legs

    def _map_axes(self, state: ControllerState) -> tuple[float, float, float]:
        hip_up_down = -state.LY
        knee_out_in = state.LX
        hip_yaw = state.RY
        return hip_up_down, knee_out_in, hip_yaw

    def map(
        self,
        state: ControllerState,
        observation_timestamp_ns: Optional[int] = None,
    ) -> JointCommand:
        selected_legs = self._select_legs(state)
        hip_up_down, knee_out_in, hip_yaw = self._map_axes(state)

        dof = self._robot_definition.get_total_joint_count()
        joint_positions = np.zeros(dof, dtype=np.float32)

        if selected_legs:
            for leg in selected_legs:
                ij = self._leg_joint_indices.get(leg)
                if ij is None:
                    # Leg missing on this topology (e.g. ML/MR on quad tripod combos).
                    continue
                hip_yaw_idx, hip_pitch_idx, knee_idx = ij

                joint_positions[hip_pitch_idx] = hip_up_down * self.hip_up_down_scale
                joint_positions[knee_idx] = knee_out_in * self.knee_out_in_scale
                joint_positions[hip_yaw_idx] = hip_yaw * self.hip_yaw_scale

                logger.debug(
                    f"Leg {leg.value}: hip_yaw_idx={hip_yaw_idx} pos={joint_positions[hip_yaw_idx]:.3f}, "
                    f"hip_pitch_idx={hip_pitch_idx} pos={joint_positions[hip_pitch_idx]:.3f}, "
                    f"knee_idx={knee_idx} pos={joint_positions[knee_idx]:.3f}"
                )

        else:
            logger.debug("No legs selected, all joints at neutral position")

        current_timestamp_ns = time.time_ns()
        if observation_timestamp_ns is None:
            observation_timestamp_ns = current_timestamp_ns

        joint_names = self._robot_definition.get_joint_names()
        joint_cmd = JointCommand(
            _joint_positions=joint_positions,
            timestamp_ns=current_timestamp_ns,
            observation_timestamp_ns=observation_timestamp_ns,
            joint_names=joint_names,
            source=JointCommandSource.OPERATOR,
        )

        logger.debug(
            f"Mapped gamepad state: {len(selected_legs)} legs selected (robot={self._robot_definition.name}), "
            f"joint_positions range=[{joint_positions.min():.3f}, {joint_positions.max():.3f}]"
        )

        return joint_cmd
