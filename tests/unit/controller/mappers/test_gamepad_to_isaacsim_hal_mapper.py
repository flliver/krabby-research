"""Unit tests for GamepadToIsaacSimHALMapper.

How to Run the Tests:
====================

From the krabby-research directory, run:

    # Run all tests in this file
    python -m pytest tests/unit/controller/mappers/test_gamepad_to_isaacsim_hal_mapper.py -v

    # Run a specific test class
    python -m pytest tests/unit/controller/mappers/test_gamepad_to_isaacsim_hal_mapper.py::TestMapperInitialization -v

    # Run a specific test
    python -m pytest tests/unit/controller/mappers/test_gamepad_to_isaacsim_hal_mapper.py::TestMapperInitialization::test_initialization_with_scale_args -v

    # Run with coverage
    python -m pytest tests/unit/controller/mappers/test_gamepad_to_isaacsim_hal_mapper.py --cov=controller.mappers --cov-report=html

    # Run with verbose output and show print statements
    python -m pytest tests/unit/controller/mappers/test_gamepad_to_isaacsim_hal_mapper.py -v -s


Prerequisites:
    - pytest: pip install pytest
    - pytest-cov (optional, for coverage): pip install pytest-cov

"""

import time

import numpy as np
import pytest

from controller.input.state import ControllerState, LegIdentifier
from controller.mappers.gamepad_to_isaacsim_hal_mapper import (
    GamepadToIsaacSimHALMapper,
    LEG_TO_JOINT_INDICES,
)
from hal.client.data_structures.hardware import JointCommand
from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION

# Scale values used when constructing the mapper in tests (match CLI default / Jetson).
HIP_SCALE = 0.3
KNEE_SCALE = 0.3
HIP_YAW_SCALE = 0.2


def _make_mapper(hip=HIP_SCALE, knee=KNEE_SCALE, yaw=HIP_YAW_SCALE):
    """Helper: mapper with standard or custom scaling."""
    return GamepadToIsaacSimHALMapper(
        hip_up_down_scale=hip,
        knee_out_in_scale=knee,
        hip_yaw_scale=yaw,
        robot_definition=KRABBY_HEX_DEFINITION,
    )


class TestMapperInitialization:
    """Test mapper initialization."""

    def test_initialization_with_scale_args(self):
        """Test mapper stores scaling factors passed by caller (e.g. CLI via config)."""
        mapper = _make_mapper()
        assert mapper.hip_up_down_scale == HIP_SCALE
        assert mapper.knee_out_in_scale == KNEE_SCALE
        assert mapper.hip_yaw_scale == HIP_YAW_SCALE

    def test_custom_initialization(self):
        """Test mapper initialization with custom scaling factors."""
        mapper = GamepadToIsaacSimHALMapper(
            hip_up_down_scale=0.5,
            knee_out_in_scale=0.4,
            hip_yaw_scale=0.3,
            robot_definition=KRABBY_HEX_DEFINITION,
        )
        
        assert mapper.hip_up_down_scale == 0.5
        assert mapper.knee_out_in_scale == 0.4
        assert mapper.hip_yaw_scale == 0.3


class TestMapperSingleLegMapping:
    """Test mapping single leg control data."""

    def setup_method(self):
        """Create a fresh mapper for each test."""
        self.mapper = _make_mapper()

    def test_map_front_left_leg(self):
        """Test mapping Front Left leg with axis values."""
        # LT without LB selects FL
        state = ControllerState(LT=True, LB=False, LY=-0.5, LX=-0.3, RY=0.2)
        
        joint_cmd = self.mapper.map(state)
        
        assert isinstance(joint_cmd, JointCommand)
        positions = joint_cmd.to_positions_dict()
        assert len(positions) == 18
        
        # FL leg indices: 0 (hip_yaw), 1 (hip_pitch), 2 (knee)
        hip_yaw_idx, hip_pitch_idx, knee_idx = LEG_TO_JOINT_INDICES[LegIdentifier.FRONT_LEFT]
        
        # LY=-0.5 -> hip_up_down=0.5 (inverted), LX=-0.3 -> knee_out_in=-0.3, RY=0.2 -> hip_yaw=0.2
        assert positions[joint_cmd.joint_names[hip_yaw_idx]] == pytest.approx(0.2 * HIP_YAW_SCALE)
        assert positions[joint_cmd.joint_names[hip_pitch_idx]] == pytest.approx(0.5 * HIP_SCALE)
        assert positions[joint_cmd.joint_names[knee_idx]] == pytest.approx(-0.3 * KNEE_SCALE)
        
        # Other joints should remain at zero
        for i in range(18):
            if i not in (hip_yaw_idx, hip_pitch_idx, knee_idx):
                assert positions[joint_cmd.joint_names[i]] == pytest.approx(0.0)

    def test_map_front_right_leg(self):
        """Test mapping Front Right leg."""
        # RT without RB selects FR
        state = ControllerState(RT=True, RB=False, LY=-0.4, LX=0.6, RY=-0.1)
        
        joint_cmd = self.mapper.map(state)
        
        # FR leg indices: 3 (hip_yaw), 4 (hip_pitch), 5 (knee)
        hip_yaw_idx, hip_pitch_idx, knee_idx = LEG_TO_JOINT_INDICES[LegIdentifier.FRONT_RIGHT]
        positions = joint_cmd.to_positions_dict()
        # LY=-0.4 -> hip_up_down=0.4, LX=0.6 -> knee_out_in=0.6, RY=-0.1 -> hip_yaw=-0.1
        assert positions[joint_cmd.joint_names[hip_yaw_idx]] == pytest.approx(-0.1 * HIP_YAW_SCALE)
        assert positions[joint_cmd.joint_names[hip_pitch_idx]] == pytest.approx(0.4 * HIP_SCALE)
        assert positions[joint_cmd.joint_names[knee_idx]] == pytest.approx(0.6 * KNEE_SCALE)

    def test_map_all_single_legs(self):
        """Test mapping each leg individually."""
        legs = [
            (LegIdentifier.FRONT_LEFT, ControllerState(LT=True, LB=False)),
            (LegIdentifier.FRONT_RIGHT, ControllerState(RT=True, RB=False)),
            (LegIdentifier.MIDDLE_LEFT, ControllerState(LS=True)),
            (LegIdentifier.MIDDLE_RIGHT, ControllerState(RS=True)),
            (LegIdentifier.REAR_LEFT, ControllerState(LB=True, LT=False)),
            (LegIdentifier.REAR_RIGHT, ControllerState(RB=True, RT=False)),
        ]
        
        for leg, base_state in legs:
            # Add axis values
            state = ControllerState(
                LT=base_state.LT,
                LB=base_state.LB,
                RT=base_state.RT,
                RB=base_state.RB,
                LS=base_state.LS,
                RS=base_state.RS,
                LY=-0.5,  # hip_up_down = 0.5
                LX=0.5,   # knee_out_in = 0.5
                RY=0.5,   # hip_yaw = 0.5
            )
            
            joint_cmd = self.mapper.map(state)
            hip_yaw_idx, hip_pitch_idx, knee_idx = LEG_TO_JOINT_INDICES[leg]
            
            # All should have the same scaled values
            expected = 0.5 * HIP_SCALE
            positions = joint_cmd.to_positions_dict()
            assert positions[joint_cmd.joint_names[hip_pitch_idx]] == pytest.approx(expected)
            assert positions[joint_cmd.joint_names[knee_idx]] == pytest.approx(0.5 * KNEE_SCALE)
            assert positions[joint_cmd.joint_names[hip_yaw_idx]] == pytest.approx(0.5 * HIP_YAW_SCALE)


class TestMapperMultipleLegsMapping:
    """Test mapping multiple legs simultaneously."""

    def setup_method(self):
        """Create a fresh mapper for each test."""
        self.mapper = _make_mapper()

    def test_map_two_legs(self):
        """Test mapping two legs at once."""
        # LT and RT both pressed (without LB/RB) selects FL and FR
        state = ControllerState(LT=True, LB=False, RT=True, RB=False, LY=-0.5, LX=-0.3, RY=0.2)
        
        joint_cmd = self.mapper.map(state)
        
        # Check FL leg
        fl_hip_yaw, fl_hip_pitch, fl_knee = LEG_TO_JOINT_INDICES[LegIdentifier.FRONT_LEFT]
        positions = joint_cmd.to_positions_dict()
        assert positions[joint_cmd.joint_names[fl_hip_yaw]] == pytest.approx(0.2 * HIP_YAW_SCALE)
        assert positions[joint_cmd.joint_names[fl_hip_pitch]] == pytest.approx(0.5 * HIP_SCALE)
        assert positions[joint_cmd.joint_names[fl_knee]] == pytest.approx(-0.3 * KNEE_SCALE)
        
        # Check FR leg
        fr_hip_yaw, fr_hip_pitch, fr_knee = LEG_TO_JOINT_INDICES[LegIdentifier.FRONT_RIGHT]
        assert positions[joint_cmd.joint_names[fr_hip_yaw]] == pytest.approx(0.2 * HIP_YAW_SCALE)
        assert positions[joint_cmd.joint_names[fr_hip_pitch]] == pytest.approx(0.5 * HIP_SCALE)
        assert positions[joint_cmd.joint_names[fr_knee]] == pytest.approx(-0.3 * KNEE_SCALE)

    def test_map_tripod_left(self):
        """Test mapping left tripod (FL, RL, MR)."""
        # LT + LB combo selects FL, RL, MR
        state = ControllerState(LT=True, LB=True, LY=-0.3, LX=0.4, RY=-0.2)
        
        joint_cmd = self.mapper.map(state)
        
        # Check all three legs have the same values
        for leg in [LegIdentifier.FRONT_LEFT, LegIdentifier.REAR_LEFT, LegIdentifier.MIDDLE_RIGHT]:
            hip_yaw_idx, hip_pitch_idx, knee_idx = LEG_TO_JOINT_INDICES[leg]
            positions = joint_cmd.to_positions_dict()
            assert positions[joint_cmd.joint_names[hip_yaw_idx]] == pytest.approx(-0.2 * HIP_YAW_SCALE)
            assert positions[joint_cmd.joint_names[hip_pitch_idx]] == pytest.approx(0.3 * HIP_SCALE)
            assert positions[joint_cmd.joint_names[knee_idx]] == pytest.approx(0.4 * KNEE_SCALE)

    def test_map_all_legs(self):
        """Test mapping all six legs simultaneously."""
        # Select all legs individually
        state = ControllerState(
            LT=True, LB=True, RT=True, RB=True, LS=True, RS=True,
            LY=-0.1, LX=0.2, RY=0.3
        )
        
        joint_cmd = self.mapper.map(state)
        positions = joint_cmd.to_positions_dict()

        # All legs should have the same values
        for leg in LegIdentifier:
            hip_yaw_idx, hip_pitch_idx, knee_idx = LEG_TO_JOINT_INDICES[leg]
            assert positions[joint_cmd.joint_names[hip_yaw_idx]] == pytest.approx(0.3 * HIP_YAW_SCALE)
            assert positions[joint_cmd.joint_names[hip_pitch_idx]] == pytest.approx(0.1 * HIP_SCALE)
            assert positions[joint_cmd.joint_names[knee_idx]] == pytest.approx(0.2 * KNEE_SCALE)


class TestMapperNoLegsSelected:
    """Test behavior when no legs are selected."""

    def setup_method(self):
        """Create a fresh mapper for each test."""
        self.mapper = _make_mapper()

    def test_no_legs_selected(self):
        """Test that no legs selected results in all zeros."""
        state = ControllerState()
        
        joint_cmd = self.mapper.map(state)
        
        # All positions should be zero
        assert all(v == 0.0 for v in joint_cmd.to_positions_dict().values())


class TestMapperTimestamps:
    """Test timestamp handling."""

    def setup_method(self):
        """Create a fresh mapper for each test."""
        self.mapper = _make_mapper()

    def test_timestamp_generation(self):
        """Test that timestamps are generated correctly."""
        state = ControllerState(LT=True, LB=False)
        
        before_ns = time.time_ns()
        joint_cmd = self.mapper.map(state)
        after_ns = time.time_ns()
        
        # Timestamp should be within the time window
        assert before_ns <= joint_cmd.timestamp_ns <= after_ns
        assert joint_cmd.timestamp_ns > 0
        assert joint_cmd.observation_timestamp_ns > 0

    def test_observation_timestamp_provided(self):
        """Test that provided observation timestamp is used."""
        state = ControllerState(LT=True, LB=False)
        
        observation_ts = 1234567890123456789
        joint_cmd = self.mapper.map(state, observation_timestamp_ns=observation_ts)
        
        assert joint_cmd.observation_timestamp_ns == observation_ts
        assert joint_cmd.timestamp_ns != observation_ts  # Should be different (current time)

    def test_observation_timestamp_none(self):
        """Test that None observation timestamp uses current time."""
        state = ControllerState(LT=True, LB=False)
        
        before_ns = time.time_ns()
        joint_cmd = self.mapper.map(state, observation_timestamp_ns=None)
        after_ns = time.time_ns()
        
        # Both timestamps should be within the time window
        assert before_ns <= joint_cmd.timestamp_ns <= after_ns
        assert before_ns <= joint_cmd.observation_timestamp_ns <= after_ns


class TestMapperErrorHandling:
    """Test error handling."""

    def setup_method(self):
        """Create a fresh mapper for each test."""
        self.mapper = _make_mapper()

    def test_invalid_state_type(self):
        """Test that invalid state type raises ValueError."""
        with pytest.raises(ValueError, match="state must be ControllerState"):
            self.mapper.map("invalid")  # type: ignore

    def test_invalid_state_none(self):
        """Test that None state raises ValueError."""
        with pytest.raises(ValueError, match="state must be ControllerState"):
            self.mapper.map(None)  # type: ignore


class TestMapperCustomScaling:
    """Test mapper with custom scaling factors."""

    def test_custom_scaling_factors(self):
        """Test that custom scaling factors (passed by caller) are applied correctly."""
        mapper = _make_mapper(hip=0.5, knee=0.4, yaw=0.3)
        
        state = ControllerState(LT=True, LB=False, LY=-1.0, LX=1.0, RY=1.0)
        
        joint_cmd = mapper.map(state)
        
        fl_hip_yaw, fl_hip_pitch, fl_knee = LEG_TO_JOINT_INDICES[LegIdentifier.FRONT_LEFT]
        # LY=-1.0 -> hip_up_down=1.0, LX=1.0 -> knee_out_in=1.0, RY=1.0 -> hip_yaw=1.0
        positions = joint_cmd.to_positions_dict()
        assert positions[joint_cmd.joint_names[fl_hip_pitch]] == pytest.approx(0.5)
        assert positions[joint_cmd.joint_names[fl_knee]] == pytest.approx(0.4)
        assert positions[joint_cmd.joint_names[fl_hip_yaw]] == pytest.approx(0.3)
