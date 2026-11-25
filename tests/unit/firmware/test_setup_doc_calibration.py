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


def test_table_safe_sweep_and_its_pot_limitation_are_documented(setup_text):
    step = _line_starting(setup_text, "- **Table-safe hip sweep (hall hips).**")
    assert "by count, to `CAL_HIP_SWEEP_EDGES` = 840" in step
    assert "only with its knee parked" in step
    assert "never sought on a solid stand" in step
    assert "no knee setting keeps the leg clear of the belly, the table and the floor" in step
    order = _line_starting(setup_text, "- **SWEEP pass**")
    assert "table-safe order" in order and "eight steps per leg" in order
    assert "seven steps" not in setup_text
    seq = setup_text[setup_text.index("- **SWEEP pass**"):setup_text.index("- **Table-safe hip sweep")]
    for part in (
        "knee curl;",
        "hip curl (home 0);",
        "knee curl again, as in SEEK: with its hip curled the knee's home is proven before the park counts from it, also when SWEEP runs on its own",
        "knee uncurl to its limit switch (about 785 edges out on FL);",
        "**264 edges out of its curl stop** (firmware position -264)",
        "inside the band 244-280 edges out (`-280..-244`)",
        "**by count to 840 edges**",
        "within **10 edges of 0**",
        "knee curl (this completes the knee, as before).",
    ):
        assert part in seq, part
    # A pot hip keeps the old order.
    assert "keeps the old order: knee curl, hip curl, hip uncurl to its end" in seq
    assert "<hip> and its knee must both be hall encoders" in seq
    # Nothing of the old 700-edge sweep stays as a claim.
    assert "stopped <n> edges short of its 700-edge sweep" not in setup_text
    assert "`CAL_HIP_SWEEP_EDGES` = 700" not in setup_text
    assert "fixed 700-edge" not in setup_text and "about 70%" not in setup_text
    # `[0, 700]` survives only as an invalid old record.
    for line in setup_text.splitlines():
        if "`[0, 700]`" in line:
            assert "invalid" in line or "rejected" in line, line[:120]


def test_seek_is_knee_hip_knee_again(setup_text):
    step = _line_starting(setup_text, "- **SEEK pass**")
    assert "**knee curl, hip curl, knee curl again.**" in step
    assert "may stop on the body" in step
    assert "`WARN <knee> home moved <n> edges`" in step


def test_derived_extend_limit_and_all_legs_use_fl_values(setup_text):
    step = _line_starting(setup_text, "- **Derived extend limit.**")
    assert "`[0, 1069]`" in step and "`CAL_HIP_STROKE_EDGES` = 1069" in step
    assert "3 of 3" in step and "203.2 mm" in step and "5.26 counts/mm" in step
    assert "**derived, not switch-verified**" in step
    assert "Calibration joint <hip> extend derived (1069 edges): not switch-verified" in step
    assert "`T` stays 10 edges short of it" in step and "8..1059" in step
    assert "Old `[0, 700]` hip records are invalid: run `C`" in step
    legs = _line_starting(setup_text, "- **All six legs use FL's values**")
    assert "knee park 264, hip sweep 840, stroke 1069" in legs
    assert "Verify each leg" in legs


def test_hollow_stand_sweep_is_optional_and_never_on_a_solid_table(setup_text):
    step = _line_starting(setup_text, "- **Optional hollow-stand sweep**")
    assert "off by default" in step and "-DKRABBY_CAL_HOLLOW_STAND=1" in step
    assert "192 edges out (band 157-232)" in step
    assert "**1069 +- 20**" in step and "switch-verified" in step
    assert "**Never on a solid table**" in step and "940-985" in step


def test_obstacle_handling_and_failure_reasons(setup_text):
    step = _line_starting(setup_text, "- **Obstacles.**")
    assert "`WARN <hip> stopped at <x> edges: retracting`" in step
    assert "no automatic retry" in step
    assert "`WARN <j> left out at <x> edges: jog it to its curl end`" in setup_text
    for reason in (
        "| `<j> stopped <n> edges short of its 840-edge sweep: obstacle? check its home` |",
        "| `<j> lost counts: switch at <x> edges, expected 1069` (or `expected 0`) |",
        "| `<j> at <x> edges, outside its park band -280..-244` |",
        "| `<j> stopped <n> edges short of its park at -264: obstacle?` |",
        "| `<j> not back at its curl switch (hall=<x>): jog it home` |",
        "| `<j> and its knee must both be hall encoders` |",
    ):
        assert reason in setup_text, reason


def test_speeds_pwm_110_to_150_and_80_is_not_accepted(setup_text):
    assert "| `C <pwm>` | Same, at PWM magnitude `<pwm>` (integer, 110-150) |" in setup_text
    assert "pwm 110-150" in setup_text
    assert "40-150" not in setup_text
    step = _line_starting(setup_text, "- Default speed is PWM 128;")
    assert "**`C` accepts PWM 110-150**" in step
    assert "PWM 80 is no longer accepted (owner decision F4)" in step
    assert "no current signature" in step
    assert "229 edges short of its curl switch" in step


def test_nudges_caps_and_recovery(setup_text):
    nudge = _line_starting(setup_text, "- **Hip nudge.**")
    assert "capped at PWM 128 whatever the `C` speed" in nudge
    assert "stops after 1 s if it counted no edge" in nudge
    assert "about 140 edges in 2 s at 128" in nudge
    assert "owner kept the cap at 128 (decision F3, 2026-11-24)" in setup_text
    assert "Whether to keep the cap at 128 or lower it" not in setup_text
    rec = _line_starting(setup_text, "- **Recovery after a hip `stuck mid-stroke`.**")
    assert "Never rerun at 128: jog the hip to its curl end, or rerun at 150" in rec
    assert "more than 40 edges short of it fails `stopped <n> edges short of its curl end`" in rec
    assert "Reset the board" in rec and "only if the joint was really moved by hand" in rec
    kept = _line_starting(setup_text, "- **Kept home.**")
    assert "raw home count is kept" in kept


def test_knee_never_moves_while_its_hip_is_known_out(setup_text):
    step = _line_starting(setup_text, "- **A knee never moves while its hall hip is known to be out past 560 edges.**")
    assert "homed this boot, kept after a failure (the kept home), or a home that a guard cleared" in step
    assert "`<hip> out at <x> edges: jog it to its curl end first`" in step
    assert "nothing moves and the hip keeps its home" in step
    assert "not checked, so operator rule 6 still applies" in step
    row = _line_starting(setup_text, "| `<j> out at <x> edges: jog it to its curl end first` |")
    assert "HipOut" in row and "nothing moved" in row


def test_no_motion_curl_press_is_not_failed_stuck(setup_text):
    step = _line_starting(setup_text, "- **A curl press that shows no motion at all**")
    assert "no switch-open current is handled as a curl with no motion" in step
    assert "else the nudge, whose return needs the switch-open signature" in step
    assert "reads a count high is not failed `stuck`" in step


def test_switch_end_timing_is_the_six_flat_samples(setup_text):
    assert "150 ms after the last edge" not in setup_text
    assert "0.11-0.22 s" not in setup_text
    assert setup_text.count("about 0.3-0.5 s after the count stop") == 2
    assert "six samples flat at the floor; harmless, the switch has already cut the motor" in setup_text


def test_neutral_and_t_keep_hips_at_or_below_560_on_a_stand(setup_text):
    neutral = _line_starting(setup_text, "- Robot moves the hips and knees that have a position to center (0.5).")
    assert "about 534 edges" in neutral and "about 392 edges out" in neutral
    assert "at or below 560 edges" in neutral
    assert "open decision" in neutral and "8..1059" in neutral
    assert "941-1003" in neutral
    step = _line_starting(setup_text, "6. Then Neutral for the whole robot.")
    assert "(it was 350)" in step and "at or below 560 edges" in step
    assert "Neutral's own 534 is inside that limit" in step


def test_calibrate_at_128_and_110_to_127_is_not_validated(setup_text):
    step = _line_starting(setup_text, "- Default speed is PWM 128;")
    assert "110-127 is not validated on the robot" in step
    assert "the stall evidence is at 80 and 128" in step


def test_rules_before_any_c_jog_hips_and_knees(setup_text):
    rule = _line_starting(setup_text, "6. **Before any `C` on the robot")
    assert "jog every hip inside 0-560 edges and every knee to its curl end" in rule
    assert "`by=switch` or `open=` END line" in rule
    assert "never rerun at 128" in _line_starting(setup_text, "7. **After a hip stops")


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
    _line_starting(setup_text, "- **End detection (hall joints on Rev 3 boards).**")
    switch = _line_starting(setup_text, "- **Into a limit switch**")
    assert "every hip curl, every knee uncurl, and the hip's extended end" in switch
    assert "**limit switch**" in switch
    assert "**only with the switch-open signature**" in switch
    assert "flat (within 1 count) at the channel's own offset" in switch
    assert "measures that level itself in its first samples" in switch
    assert "FL 7-8 counts, FRHL 11" in switch
    assert "at least 10% below its running level" in switch
    assert "**Any stall without the signature fails `stuck mid-stroke`, however short the drive**" in switch
    assert "a press of a joint at its end, a nudge return, a start next to the stall point" in switch
    knee = _line_starting(setup_text, "- **A knee curl**")
    assert "**mechanical stop**" in knee
    assert "an obstacle on a knee curl looks the same" in knee.lower()
    assert "second curl (SEEK) and the SWEEP re-curl catch it" in knee
    assert "treats any stall as an obstacle" in _line_starting(setup_text, "- **A hip sweep**")
    names = _line_starting(setup_text, "- The END line names the kind:")
    assert "by=switch cur=<a>/<b> floor=<f>" in names and "by=stop cur=<a>/<b>" in names
    assert "` open=<a>/<f>`" in names
    row = _line_starting(setup_text, "| `<j> stuck mid-stroke after <n> edges at PWM <p>: raise PWM or check load`")
    assert "without the switch-open signature" in row
    assert "however short the drive" in row
    assert "never taken as an end" not in setup_text
    assert "stays stuck" not in setup_text
    # Below arming speed, and for a curl that starts close to its end, ends are
    # found by stall alone; a crawl stall above running is not a switch.
    stall = _line_starting(setup_text, "- A stall without a running phase")
    assert "below arming speed an end is found by stall alone" in stall
    assert "or in 1 s once it has made 12 edges" in stall
    assert "ends as a plain stall (no `by=`) instead of failing stuck" in stall
    assert "for a drive into a limit switch only with the switch-open signature (else stuck)" in stall
    assert "8 edges inside the retracted end" in setup_text
    assert "25 edges inside the knee's extended end" in setup_text
    assert "10 edges inside the hip's extended end" in setup_text
    assert "End-current measurement** (2026-11-24" in setup_text


def test_acceptance_needs_a_new_full_run_and_watches_the_tibia(setup_text):
    entry = _line_starting(setup_text, "- **Acceptance**")
    assert "Calibration joint <hip> invalid: defaults" in entry
    step = _line_starting(setup_text, "4. Full calibration on the stand:")
    assert "jog every hip inside 0-560 and every knee to its curl end" in step
    assert "the knee parks 264 edges out" in step
    assert "each hip goes out 840 edges and back with its knee parked" in step
    assert "never touches the table or the body" in step
    assert "`min=0 max=1069 edges (extend derived from FL, not switch-verified)`" in step
    assert "700" not in step


def test_knee_nudge_spring_and_no_early_stop(setup_text):
    assert "350 ms" not in setup_text
    # The blind knee bound at the PWM 128 nudge cap, measured 2026-11-24; the
    # nudge runs before its hip is proven curled.
    assert "up to about 95 edges out" in setup_text
    assert "no jogging is needed for the nudge" in setup_text.lower()
    assert "up to about 58 edges out" not in setup_text
    assert "its hip stays curled" not in setup_text
    assert "up to about 41 edges out with its hip curled" not in setup_text
    assert "a 12-edge nudge ends about 18-21 edges out" in setup_text
    assert "no SENS: flash all three boards with the same firmware, or wait a few seconds after connecting" in setup_text
    assert "outside 300-20000" not in setup_text
    assert "outside 450-1500" in setup_text
