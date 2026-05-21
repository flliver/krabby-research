"""Mapper from gamepad controller state to HAL command format (for IsaacSim).

This mapper encapsulates the complete mapping from:
- Input: Gamepad controller state (ControllerState)
- Robot embodiment: Quad (4 legs, 12 joints) or hexapod (6 legs, 18 joints), via robot_definition
- Input control type: Gamepad
- Output: HAL JointCommand format (used with IsaacSim HAL server)

Works for both quad and hexapod: pass robot_definition with legs (e.g. FL, FR, RL, RR for quad
or FL, FR, ML, MR, RL, RR for hex). Leg selection buttons (LT/LB/RT/RB/LS/RS) only affect legs
that exist on the robot; middle legs (ML/MR) are ignored for quad.

The mapper handles leg selection, axis mapping, and joint position calculation,
outputting absolute joint positions in JointCommand format.

Scaling (radians per unit stick) is supplied by the caller. The mapper is only
used via the control loop currently, which gets scale values from ControlLoopConfig (set
by the CLI, e.g. krabby-uno-sim). 
"""

import logging
import time
from typing import Optional, Set

import numpy as np

from controller.input.state import ControllerState, LegIdentifier
from hal.client.data_structures.hardware import JointCommand, JointCommandSource
from hal.server.robot_definition import RobotDefinition

logger = logging.getLogger(__name__)

LEG_NAME_TO_ID = {e.value: e for e in LegIdentifier}

LEG_TO_JOINT_INDICES = {
    LegIdentifier.FRONT_LEFT: (0, 1, 2),
    LegIdentifier.FRONT_RIGHT: (3, 4, 5),
    LegIdentifier.MIDDLE_LEFT: (6, 7, 8),
    LegIdentifier.MIDDLE_RIGHT: (9, 10, 11),
    LegIdentifier.REAR_LEFT: (12, 13, 14),
    LegIdentifier.REAR_RIGHT: (15, 16, 17),
}


class GamepadToIsaacSimHALMapper:
    """Maps gamepad controller state to IsaacSim HAL joint commands.
    
    This mapper encapsulates the complete transformation from raw gamepad input
    to joint commands for a robot in IsaacSim (quad 12 joints or hex 18 joints
    via robot_definition):
    
    - Input control type: Gamepad
    - Output environment: IsaacSim
    
    The mapper handles:
    1. Leg selection based on button combinations
    2. Axis mapping (sticks → joint control axes)
    3. Joint position calculation (absolute positions)
    
    Zero-copy guarantees:
    - Joint positions array is always newly created (not a view)
    - Timestamp is always copied (scalar)
    """
    
    def __init__(
        self,
        hip_up_down_scale: float,
        knee_out_in_scale: float,
        hip_yaw_scale: float,
        *,
        robot_definition: RobotDefinition,
    ):
        self.hip_up_down_scale = hip_up_down_scale
        self.knee_out_in_scale = knee_out_in_scale
        self.hip_yaw_scale = hip_yaw_scale
        self._robot = robot_definition
        jt = len(self._robot.joint_types)
        if jt != 3:
            raise ValueError(
                f"GamepadToIsaacSimHALMapper expects 3 joint_types per leg, got {jt}"
            )
        self._n_joints = self._robot.get_total_joint_count()
        self._leg_to_indices: dict[LegIdentifier, tuple[int, int, int]] = {}
        for i, leg_name in enumerate(self._robot.legs):
            if leg_name in LEG_NAME_TO_ID:
                o = i * jt
                self._leg_to_indices[LEG_NAME_TO_ID[leg_name]] = (o, o + 1, o + 2)
        self._allowed_legs = set(self._leg_to_indices.keys())
    
    def _select_legs(self, state: ControllerState) -> Set[LegIdentifier]:
        """Select legs based on controller button state.
        
        Leg selection rules:
        - LT (without LB): Select Front Left (FL)
        - LB (without LT): Select Rear Left (RL)
        - LS: Select Left Middle (ML)
        - RS: Select Right Middle (MR)
        - RT (without RB): Select Front Right (FR)
        - RB (without RT): Select Rear Right (RR)
        - LT + LB: Select FL, RL, MR (tripod combo left)
        - RT + RB: Select FR, RR, ML (tripod combo right)
        
        Args:
            state: Controller state with button presses.
            
        Returns:
            Set of selected leg identifiers.
        """
        # Determine leg selections
        select_FL = state.LT and not state.LB
        select_RL = state.LB and not state.LT
        select_ML = state.LS
        select_MR = state.RS
        select_FR = state.RT and not state.RB
        select_RR = state.RB and not state.RT
        
        # Combo triggers
        combo_left = state.LT and state.LB  # FL/RL/MR
        combo_right = state.RT and state.RB  # FR/RR/ML
        
        # Build selected legs set
        legs = set()
        if combo_left:
            legs |= {LegIdentifier.FRONT_LEFT, LegIdentifier.REAR_LEFT, LegIdentifier.MIDDLE_RIGHT}
        if combo_right:
            legs |= {LegIdentifier.FRONT_RIGHT, LegIdentifier.REAR_RIGHT, LegIdentifier.MIDDLE_LEFT}
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
        return legs & self._allowed_legs
    
    def _map_axes(self, state: ControllerState) -> tuple[float, float, float]:
        """Map controller stick axes to control axes.
        
        Axis mappings:
        - Left stick Y: Hip up/down (inverted: -LY, so up = positive)
        - Left stick X: Knee out/in (direct: LX)
        - Right stick Y: Hip yaw forward/back (direct: RY)
        
        Args:
            state: Controller state with stick values.
            
        Returns:
            Tuple of (hip_up_down, knee_out_in, hip_yaw) axis values.
        """
        hip_up_down = -state.LY  # Invert Y axis (up = positive)
        knee_out_in = state.LX
        hip_yaw = state.RY
        
        return hip_up_down, knee_out_in, hip_yaw
    
    def map(
        self,
        state: ControllerState,
        observation_timestamp_ns: Optional[int] = None,
    ) -> JointCommand:
        """Map gamepad controller state to joint command.
        
        This method encapsulates the complete mapping from raw gamepad input
        to absolute joint positions for a hexapod robot in IsaacSim.
        
        Args:
            state: ControllerState with button and stick values.
            observation_timestamp_ns: Optional timestamp of the observation this
                command responds to. If None, uses current time.
                
        Returns:
            JointCommand with absolute joint positions.
            
        Raises:
            ValueError: If state is invalid.
        """
        if not isinstance(state, ControllerState):
            raise ValueError(f"state must be ControllerState, got {type(state)}")
        
        # Select legs based on button state
        selected_legs = self._select_legs(state)
        
        # Map stick axes to control axes
        hip_up_down, knee_out_in, hip_yaw = self._map_axes(state)

        # Start from neutral position (all joints at 0.0)
        # TODO: In the future, starting joint positions will be available from the robot state/observations   
        joint_positions = np.zeros(self._n_joints, dtype=np.float32)
        if selected_legs:
            for leg in selected_legs:
                hip_yaw_idx, hip_pitch_idx, knee_idx = self._leg_to_indices[leg]
                joint_positions[hip_pitch_idx] = hip_up_down * self.hip_up_down_scale
                joint_positions[knee_idx] = knee_out_in * self.knee_out_in_scale
                joint_positions[hip_yaw_idx] = hip_yaw * self.hip_yaw_scale
                logger.debug(
                    "Leg %s: hip_yaw=%.3f hip_pitch=%.3f knee=%.3f",
                    leg.value, joint_positions[hip_yaw_idx], joint_positions[hip_pitch_idx], joint_positions[knee_idx],
                )

        current_timestamp_ns = time.time_ns()
        if observation_timestamp_ns is None:
            observation_timestamp_ns = current_timestamp_ns

        joint_cmd = JointCommand(
            _joint_positions=joint_positions,
            timestamp_ns=current_timestamp_ns,
            observation_timestamp_ns=observation_timestamp_ns,
            joint_names=self._robot.get_joint_names(),
            source=JointCommandSource.OPERATOR,
        )
        
        logger.debug(
            f"Mapped gamepad state: {len(selected_legs)} legs selected, "
            f"joint_positions range=[{joint_positions.min():.3f}, {joint_positions.max():.3f}]"
        )
        
        return joint_cmd
