"""STM duty path for ParkourLocomotionToHWMapper."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from compute.parkour.mappers.model_to_hardware import ParkourLocomotionToHWMapper
from compute.parkour.parkour_types import InferenceResponse
from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION


def _response(action: np.ndarray | torch.Tensor) -> InferenceResponse:
    if isinstance(action, np.ndarray):
        action = torch.from_numpy(action)
    return InferenceResponse.create_success(action=action, timing_breakdown=[])


class TestModelToHardwareStm:
    def test_stm_clamps_actions_to_unit_duty(self):
        raw = np.array([2.0, -3.0, 0.5] + [0.0] * 15, dtype=np.float32)
        mapper = ParkourLocomotionToHWMapper(KRABBY_HEX_DEFINITION, stm=True)
        cmd = mapper.map(_response(raw), observation_timestamp_ns=0)
        d = cmd.to_positions_dict()
        names = KRABBY_HEX_DEFINITION.get_joint_names()
        assert d[names[0]] == pytest.approx(1.0)
        assert d[names[1]] == pytest.approx(-1.0)
        assert d[names[2]] == pytest.approx(0.5)

    def test_non_stm_passes_through_unclamped(self):
        raw = np.array([2.0, -3.0, 0.5] + [0.0] * 15, dtype=np.float32)
        mapper = ParkourLocomotionToHWMapper(KRABBY_HEX_DEFINITION, stm=False)
        cmd = mapper.map(_response(raw), observation_timestamp_ns=0)
        d = cmd.to_positions_dict()
        names = KRABBY_HEX_DEFINITION.get_joint_names()
        assert d[names[0]] == pytest.approx(2.0)
        assert d[names[1]] == pytest.approx(-3.0)
        assert d[names[2]] == pytest.approx(0.5)

    def test_stm_default_flag_is_false(self):
        mapper = ParkourLocomotionToHWMapper(KRABBY_HEX_DEFINITION)
        raw = np.array([5.0] + [0.0] * 17, dtype=np.float32)
        cmd = mapper.map(_response(raw), observation_timestamp_ns=0)
        first = KRABBY_HEX_DEFINITION.get_joint_names()[0]
        assert cmd.to_positions_dict()[first] == pytest.approx(5.0)