import os

REPO = r'C:\Users\nickm\krabby\krabby-research'


def edit(rel, pairs):
    p = os.path.join(REPO, rel)
    s = open(p, newline='', encoding='utf-8').read()
    crlf = '\r\n' in s
    s = s.replace('\r\n', '\n')
    for old, new in pairs:
        assert s.count(old) == 1, (rel, old[:80], s.count(old))
        s = s.replace(old, new)
    if crlf:
        s = s.replace('\n', '\r\n')
    open(p, 'w', newline='', encoding='utf-8').write(s)
    print('edited', rel)


edit(r'tests\unit\firmware\test_setup_doc_calibration.py', [(
    """def test_knee_nudge_spring_and_no_early_stop(setup_text):
    assert "350 ms" not in setup_text
""", """def test_knee_nudge_spring_and_no_early_stop(setup_text):
    assert "350 ms" not in setup_text
    # The blind knee bound at the PWM 128 nudge cap, measured 2026-11-24; the
    # nudge runs before its hip is proven curled.
    assert "up to about 95 edges out" in setup_text
    assert "up to about 58 edges out" not in setup_text
    assert "its hip stays curled" not in setup_text
    assert "up to about 41 edges out with its hip curled" not in setup_text
"""), (
    """    row = _line_starting(setup_text, "| `<j> stuck mid-stroke after <n> edges at PWM <p>: raise PWM or check load`")
    assert "never taken as an end" in row
""", """    row = _line_starting(setup_text, "| `<j> stuck mid-stroke after <n> edges at PWM <p>: raise PWM or check load`")
    assert "never taken as an end" in row
    # Below arming speed, and for a curl that starts close to its end, ends are
    # found by stall alone; a crawl stall above running is not a switch.
    assert "below arming speed an end is found by stall alone" in step
    assert "or in 1 s once it has made 12 edges" in step
    assert "ends as a plain stall (no `by=`) instead of failing stuck" in step
    assert "at least 10% below its running level" in step
""")])

edit(r'tests\unit\firmware\test_gui_calibration.py', [(
    """        assert "stops it as stuck mid-stroke" in text
""", """        assert "stops it as stuck mid-stroke" in text
        assert "its ends are found by stall alone" in text
""")])

edit(r'tests\unit\firmware\test_krabby_mcu_calibration.py', [(
    """    def test_bad_pwm(self):
        with pytest.raises(ValueError):
            build_calibrate_command(39)
""", """    def test_docstring_examples_use_the_default_speed(self):
        doc = build_calibrate_command.__doc__
        assert "``C 128``" in doc and "``C SEEK 128``" in doc and "``C SWEEP 128``" in doc
        assert " 80``" not in doc

    def test_bad_pwm(self):
        with pytest.raises(ValueError):
            build_calibrate_command(39)
""")])
