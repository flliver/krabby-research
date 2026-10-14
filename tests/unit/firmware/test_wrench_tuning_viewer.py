"""Status line formatting for the wrench tuning viewer."""

from firmware.sim.wrench_tuning_viewer import _format_status_line


def test_format_status_line_includes_load_and_adc():
    line = _format_status_line("FLKL", 42.5, 30, 28, 0.12, True)
    assert "FLKL" in line
    assert "42.50N" in line
    assert "map_adc=  30" in line
    assert "shim=28" in line
    assert "EN" in line
