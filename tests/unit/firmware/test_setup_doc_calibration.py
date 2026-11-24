"""firmware/SETUP.md keeps the calibration procedure checks the code and design rely on."""

from pathlib import Path

import pytest

SETUP = Path(__file__).resolve().parents[3] / "firmware" / "SETUP.md"


@pytest.fixture(scope="module")
def setup_text():
    if not SETUP.is_file():
        pytest.skip("firmware/SETUP.md is not in this checkout")
    return SETUP.read_text(encoding="utf-8")


def _line_starting(text, prefix):
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith(prefix)]
    assert len(lines) == 1, prefix
    return lines[0]


def test_acceptance_step_1_checks_the_hall_b_rest_band(setup_text):
    # R1-25 (design WP-R step 1): hall B must rest outside the pot band 32..990.
    step = _line_starting(setup_text, "1. Boot: `SENS` shows")
    assert "32..990" in step
    assert "below 32 or above 990 on all 12 hips and knees" in step
    assert "Pot" in step


def test_measurement_asks_only_for_what_the_gui_jog_can_produce(setup_text):
    # R1-23: the GUI jog runs at PWM 200 only.
    step = _line_starting(setup_text, "- **Measurement**")
    assert "if possible, PWM 40" not in step
    assert "hold one joint by hand at PWM 40" not in step
    assert "PWM 200 only" in step
    assert "10 edges/s" in step


def test_timeout_failure_does_not_advise_a_faster_run(setup_text):
    row = _line_starting(setup_text, "| `<j> <curl\\|uncurl>: no end of travel in 120 s`")
    assert "Raise the speed" not in row
    assert "do not raise the calibration speed" in row


def test_menu_key_9_is_described_as_the_full_run_on_the_stand(setup_text):
    # R1-21: key 9 is SEEK + SWEEP and saves; it is not the power-up homing step.
    entry = _line_starting(setup_text, "- Keyboard menu (`python -m firmware`): press `9`")
    assert "key `9` also homes the encoders" not in entry
    assert "on its stand" in entry
    assert "Curl only in the GUI" in entry


def test_banner_description_covers_unknown_joints(setup_text):
    # R1-19 / R1-20.
    assert "Sensor not yet detected: <names>" in setup_text
    assert "when any listed joint is `s` or `?`" in setup_text


def test_fixed_hip_sweep_and_its_pot_limitation_are_documented(setup_text):
    step = _line_starting(setup_text, "- **Fixed hip sweep (hall hips).**")
    assert "700 drive edges (`CAL_HIP_SWEEP_EDGES`" in step
    assert "never runs to a stall" in step
    assert "stopped <n> edges short of its 700-edge sweep: obstacle?" in step
    assert "`[0, 700]`" in step
    assert "a pot hip" in step and "still sweeps to its end" in step


def test_measurements_of_2026_11_23_are_recorded(setup_text):
    for fact in (
        "full stroke **776 edges**",
        "**curled tibia pressing up against the bottom of the robot**",
        "941 edges",
        "1003 edges",
        "about 17 edges deeper than PWM 80 reaches",
        "sprang out 5-9 more edges",
        "24 edges reached at 1.31 s",
        "8-14, the same as while moving (9-12)",
        "Longest move at PWM 80: 36 s",
    ):
        assert fact in setup_text, fact


def test_end_detector_and_default_speed_are_documented(setup_text):
    assert "SEEK pass, then SWEEP pass, at the default speed (PWM 128)" in setup_text
    assert "**Calibrate at the default PWM 128.**" in setup_text
    assert "calibrate at PWM 80" not in setup_text
    step = _line_starting(setup_text, "- **End detection (hall joints on Rev 3 boards).**")
    assert "**limit switch**" in step and "**mechanical stop**" in step
    assert "by=switch cur=" in step and "by=stop cur=" in step
    row = _line_starting(setup_text, "| `<j> stuck mid-stroke after <n> edges at PWM <p>: raise PWM or check load`")
    assert "never taken as an end" in row
    # Below arming speed, and for a curl that starts close to its end, ends are
    # found by stall alone; a crawl stall above running is not a switch.
    assert "below arming speed an end is found by stall alone" in step
    assert "or in 1 s once it has made 12 edges" in step
    assert "ends as a plain stall (no `by=`) instead of failing stuck" in step
    assert "at least 10% below its running level" in step
    assert "8 edges inside the retracted end" in setup_text
    assert "25 edges inside the extended end" in setup_text
    assert "End-current measurement** (2026-11-24" in setup_text


def test_acceptance_needs_a_new_full_run_and_watches_the_tibia(setup_text):
    entry = _line_starting(setup_text, "- **Acceptance**")
    assert "Calibration joint <hip> invalid: defaults" in entry
    step = _line_starting(setup_text, "4. Full calibration on the stand:")
    assert "each hip goes out about 700 edges and back" in step
    assert "never touches the body" in step
    assert "`min=0 max=700 edges (fixed sweep)`" in step


def test_knee_nudge_spring_and_no_early_stop(setup_text):
    assert "350 ms" not in setup_text
    # The blind knee bound at the PWM 128 nudge cap, measured 2026-11-24; the
    # nudge runs before its hip is proven curled.
    assert "up to about 95 edges out" in setup_text
    assert "up to about 58 edges out" not in setup_text
    assert "its hip stays curled" not in setup_text
    assert "up to about 41 edges out with its hip curled" not in setup_text
    assert "a 12-edge nudge ends about 18-21 edges out" in setup_text
    assert "no SENS: flash all three boards with the same firmware, or wait a few seconds after connecting" in setup_text
    assert "outside 300-20000" not in setup_text
    assert "outside 450-1500" in setup_text
