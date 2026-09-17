"""OLED simulator test setup."""
import sys
from pathlib import Path

import pytest

_SIM = Path(__file__).resolve().parents[3] / "firmware" / "oled_sim"
if str(_SIM) not in sys.path:
    sys.path.insert(0, str(_SIM))


def pytest_collection_modifyitems(config, items):
    """Skip render tests when SparkFun font headers are unavailable."""
    import ssd1306

    if ssd1306.fonts_available():
        return
    skip = pytest.mark.skip(
        reason="SparkFun OLED font headers not found; run "
        "firmware/scripts/fetch_arduino_libs.py or set QWIIC_OLED_LIB_DIR"
    )
    for item in items:
        item.add_marker(skip)
