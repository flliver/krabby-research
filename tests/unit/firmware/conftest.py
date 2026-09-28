"""Shared fixtures for firmware unit tests."""
import pytest
from unittest.mock import Mock

from firmware.krabby_mcu import KrabbyMCUSDK


@pytest.fixture
def bare_sdk():
    """KrabbyMCUSDK with a mock serial and no reader thread.

    Bypasses __init__ (no port detection, no connect); tests deliver replies by
    setting the mailbox attrs (_last_ver_line / _last_get_line) directly, standing
    in for the reader thread.
    """
    sdk = object.__new__(KrabbyMCUSDK)
    sdk._last_ver_line = None
    sdk._last_get_line = None
    sdk._last_cal_line = None
    sdk.ser = Mock()
    sdk.ser.is_open = True
    return sdk
