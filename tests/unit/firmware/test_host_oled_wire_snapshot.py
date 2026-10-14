"""Host OLED path uses SSD1306 data writes on sim Wire (integration smoke)."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_host_oled_sources_wired_in_cmake():
    cmake = (REPO / "firmware/sim/host/CMakeLists.txt").read_text(encoding="utf-8")
    assert "sim_ssd1306.cpp" in cmake
    assert "host_display.cpp" in cmake
    assert "host_oled_canvas.cpp" in cmake


def test_native_oled_wire_test_registered():
    cmake = (REPO / "tests/native/firmware/CMakeLists.txt").read_text(encoding="utf-8")
    assert "arduino/test_host_oled_wire" in cmake
