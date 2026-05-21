"""Unit tests for GamepadToKrabbyHALMapper.

How to Run the Tests:
====================

From the krabby-research directory, run:

    # Run all tests in this file
    python -m pytest tests/unit/controller/mappers/test_gamepad_to_krabby_hal_mapper.py -v

    # Run a specific test class
    python -m pytest tests/unit/controller/mappers/test_gamepad_to_krabby_hal_mapper.py::TestMapperInitialization -v

    # Run a specific test
    python -m pytest tests/unit/controller/mappers/test_gamepad_to_krabby_hal_mapper.py::TestMapperInitialization::test_default_initialization -v

Prerequisites:
    - pytest: pip install pytest
"""

import numpy as np
import pytest

from controller.input.state import ControllerState, LegIdentifier
from controller.mappers.gamepad_to_krabby_hal_mapper import (
    DEFAULT_HIP_UP_DOWN_SCALE,
    DEFAULT_HIP_YAW_SCALE,
    DEFAULT_KNEE_OUT_IN_SCALE,
    GamepadToKrabbyHALMapper,
    LEG_TO_JOINT_INDICES,
)
from hal.client.data_structures.hardware import JointCommand
from hal.server.robot_definition_unitree_go2 import UNITREE_GO2_DEFINITION





class TestMapperSingleLegMapping:
    """Test mapping single leg control data."""

    def setup_method(self):
        """Create a fresh mapper for each test."""
        self.mapper = GamepadToKrabbyHALMapper()

    def test_map_front_left_leg(self):
        """Test mapping Front Left leg with axis values."""
        # LT without LB selects FL
        state = ControllerState(LT=True, LB=False, LY=-0.5, LX=-0.3, RY=0.2)
        
        joint_cmd = self.mapper.map(state)
        
        assert isinstance(joint_cmd, JointCommand)
        d = joint_cmd.to_positions_dict()
        assert len(d) == 18
        
        # FL leg indices: 0 (hip_yaw), 1 (hip_pitch), 2 (knee)
        hip_yaw_idx, hip_pitch_idx, knee_idx = LEG_TO_JOINT_INDICES[LegIdentifier.FRONT_LEFT]
        
        # LY=-0.5 -> hip_up_down=0.5 (inverted), LX=-0.3 -> knee_out_in=-0.3, RY=0.2 -> hip_yaw=0.2
        assert d[joint_cmd.joint_names[hip_yaw_idx]] == pytest.approx(0.2 * DEFAULT_HIP_YAW_SCALE)
        assert d[joint_cmd.joint_names[hip_pitch_idx]] == pytest.approx(0.5 * DEFAULT_HIP_UP_DOWN_SCALE)
        assert d[joint_cmd.joint_names[knee_idx]] == pytest.approx(-0.3 * DEFAULT_KNEE_OUT_IN_SCALE)
        
        # Other joints should remain at zero
        for i in range(18):
            if i not in (hip_yaw_idx, hip_pitch_idx, knee_idx):
                assert d[joint_cmd.joint_names[i]] == pytest.approx(0.0)

    def test_map_front_right_leg(self):
        """Test mapping Front Right leg."""
        # RT without RB selects FR
        state = ControllerState(RT=True, RB=False, LY=-0.4, LX=0.6, RY=-0.1)
        
        joint_cmd = self.mapper.map(state)
        
        # FR leg indices: 3 (hip_yaw), 4 (hip_pitch), 5 (knee)
        hip_yaw_idx, hip_pitch_idx, knee_idx = LEG_TO_JOINT_INDICES[LegIdentifier.FRONT_RIGHT]
        d = joint_cmd.to_positions_dict()
        # LY=-0.4 -> hip_up_down=0.4, LX=0.6 -> knee_out_in=0.6, RY=-0.1 -> hip_yaw=-0.1
        assert d[joint_cmd.joint_names[hip_yaw_idx]] == pytest.approx(-0.1 * DEFAULT_HIP_YAW_SCALE)
        assert d[joint_cmd.joint_names[hip_pitch_idx]] == pytest.approx(0.4 * DEFAULT_HIP_UP_DOWN_SCALE)
        assert d[joint_cmd.joint_names[knee_idx]] == pytest.approx(0.6 * DEFAULT_KNEE_OUT_IN_SCALE)


class TestMapperMultipleLegsMapping:
    """Test mapping multiple legs simultaneously."""

    def setup_method(self):
        """Create a fresh mapper for each test."""
        self.mapper = GamepadToKrabbyHALMapper()

    def test_map_two_legs(self):
        """Test mapping two legs at once."""
        # LT and RT both pressed (without LB/RB) selects FL and FR
        state = ControllerState(LT=True, LB=False, RT=True, RB=False, LY=-0.5, LX=-0.3, RY=0.2)
        
        joint_cmd = self.mapper.map(state)
        
        d = joint_cmd.to_positions_dict()
        # Check FL leg
        fl_hip_yaw, fl_hip_pitch, fl_knee = LEG_TO_JOINT_INDICES[LegIdentifier.FRONT_LEFT]
        assert d[joint_cmd.joint_names[fl_hip_yaw]] == pytest.approx(0.2 * DEFAULT_HIP_YAW_SCALE)
        assert d[joint_cmd.joint_names[fl_hip_pitch]] == pytest.approx(0.5 * DEFAULT_HIP_UP_DOWN_SCALE)
        assert d[joint_cmd.joint_names[fl_knee]] == pytest.approx(-0.3 * DEFAULT_KNEE_OUT_IN_SCALE)
        
        # Check FR leg
        fr_hip_yaw, fr_hip_pitch, fr_knee = LEG_TO_JOINT_INDICES[LegIdentifier.FRONT_RIGHT]
        assert d[joint_cmd.joint_names[fr_hip_yaw]] == pytest.approx(0.2 * DEFAULT_HIP_YAW_SCALE)
        assert d[joint_cmd.joint_names[fr_hip_pitch]] == pytest.approx(0.5 * DEFAULT_HIP_UP_DOWN_SCALE)
        assert d[joint_cmd.joint_names[fr_knee]] == pytest.approx(-0.3 * DEFAULT_KNEE_OUT_IN_SCALE)


class TestMapperUnitreeGo2Topology:
    """12-DOF quad: joint commands match HalServer when --robot go2."""

    def test_front_left_twelve_dof(self):
        mapper = GamepadToKrabbyHALMapper(robot_definition=UNITREE_GO2_DEFINITION)
        state = ControllerState(LT=True, LB=False, LY=-0.5, LX=-0.3, RY=0.2)
        joint_cmd = mapper.map(state)
        d = joint_cmd.to_positions_dict()
        assert len(d) == 12
        # FL first leg × 3: indices 0,1,2
        names = joint_cmd.joint_names
        hip_yaw_idx, hip_pitch_idx, knee_idx = (0, 1, 2)
        assert d[names[hip_yaw_idx]] == pytest.approx(0.2 * DEFAULT_HIP_YAW_SCALE)
        assert d[names[hip_pitch_idx]] == pytest.approx(0.5 * DEFAULT_HIP_UP_DOWN_SCALE)
        assert d[names[knee_idx]] == pytest.approx(-0.3 * DEFAULT_KNEE_OUT_IN_SCALE)


class TestMapperNoLegsSelected:
    """Test behavior when no legs are selected."""

    def setup_method(self):
        """Create a fresh mapper for each test."""
        self.mapper = GamepadToKrabbyHALMapper()

    def test_no_legs_selected(self):
        """Test that no legs selected results in all zeros."""
        state = ControllerState()
        
        joint_cmd = self.mapper.map(state)
        
        # All positions should be zero
        assert all(v == 0.0 for v in joint_cmd.to_positions_dict().values())


