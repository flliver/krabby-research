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


# F4: GUI help.
edit(r'firmware\gui\app.py', [(
    """    f"Use the default speed (PWM {CAL_DEFAULT_PWM}): ends are found from motor current "
    "and motion; below about PWM 110 a joint may stall short of its end and the run "
    "stops it as stuck mid-stroke.\\n\"""",
    """    f"Use the default speed (PWM {CAL_DEFAULT_PWM}): ends are found from motor current "
    "and motion; below about PWM 110 a joint may stall short of its end and the run "
    "stops it as stuck mid-stroke. A joint slower than about 20 counts per second "
    "(PWM 40-50) never shows a running phase: its ends are found by stall alone, as "
    "before, and a long pause can then pass for its end.\\n\"""")])

# F8: krabby_mcu docstring examples.
edit(r'firmware\krabby_mcu.py', [(
    """    ``C`` / ``C 80`` run SEEK then SWEEP; ``C SEEK 80`` / ``C SWEEP 80`` run one pass.""",
    """    ``C`` / ``C 128`` run SEEK then SWEEP; ``C SEEK 128`` / ``C SWEEP 128`` run one pass.""")])

# F4 / F2: glossary.
edit(r'docs\GLOSSARY.md', [(
    """It is never taken as an end: the run fails `<j> stuck mid-stroke after <n> edges at PWM <p>: raise PWM or check load` before the partner joint moves. Causes: an obstacle, too low a speed (below about PWM 110), a hall dropout. |""",
    """It is never taken as an end: the run fails `<j> stuck mid-stroke after <n> edges at PWM <p>: raise PWM or check load` before the partner joint moves. Causes: an obstacle, too low a speed (below about PWM 110), a hall dropout. Two kinds of drive have no running phase to judge a signature against, and their ends are found by stall alone, as before the end detector: one slower than about 20 counts/s (PWM 40-50), and a curl that started so close to its end (about 50-130 counts) that it was already slowing into it; there a stall is taken as the end (a curl retracting the rod only if its current did not rise above running). |""")])

# SETUP.md: F1/F2/F4 in the end-detection step, F6 and F3 in the nudge text.
edit(r'firmware\SETUP.md', [(
    """so the end is called 150 ms after the last edge once the current is at least 12% below its running median (a joint that arrives crawling needs 30% and a new low).""",
    """so the end is called 150 ms after the last edge once the current is at least 12% below its running median (a joint that arrives crawling needs 30%, a new low, and at least 10% below its running level: FRHL curling at 128 from the body stalled 878 edges short of its switch with its current wobbling above running, 2026-11-24, and that is stuck, not a switch)."""), (
    """A drive that stalls with neither signature is **stuck mid-stroke** (`by=stuck`): it is never accepted as an end, the pass fails `<j> stuck mid-stroke after <n> edges at PWM <p>: raise PWM or check load` and nothing else moves. A drive that never ran (a press of a joint already at its end, a short curl back) ends as before on its count alone: a curl after at least 1 s of drive (the curl back after a nudge at least as long as the nudge drove) with fewer than 3 edges in 0.5 s, so a joint already at its end finishes in about 1 s (was about 3 s); an uncurl gets the old 1.5 s + 1 s break-free allowance after the ramp.""",
    """A drive that stalls with neither signature is **stuck mid-stroke** (`by=stuck`): it is never accepted as an end, the pass fails `<j> stuck mid-stroke after <n> edges at PWM <p>: raise PWM or check load` and nothing else moves. A drive that never ran (a press of a joint already at its end, a short curl back, a joint slower than 20 edges/s such as a hip at PWM 40-50) ends as before on its count alone: a curl after at least 1 s of drive (the curl back after a nudge at least as long as the nudge drove, and 1 s longer when the nudge saw no edges: joints curl back slower than they uncurl) with fewer than 3 edges in 0.5 s, or in 1 s once it has made 12 edges (a slow joint's pause is not its end), so a joint already at its end finishes in about 1 s (was about 3 s); an uncurl gets the old 1.5 s + 1 s break-free allowance after the ramp. So below arming speed an end is found by stall alone, and a slow joint that pauses for over a second can still pass for being at its end. A curl that starts about 50-130 edges from its end arms while already slowing into it, so it has no running phase either: when it stalls with fewer than 12 normal-speed samples since arming and its last edge within 2 s of arming, it ends as a plain stall (no `by=`) instead of failing stuck, unless it is retracting and its current rose above running (a hip stalled mid-stroke stays stuck)."""), (
    """a knee with no hall feedback (hall A dead, cable off) runs blind to the 1 s cap at no more than PWM 128, so it can end **up to about 58 edges out** (about 48 edges of drive if it broke free early, estimated from the PWM 80 rate scaled to 128 and not yet measured, plus its spring; at PWM 80 this was about 41; its hip stays curled, so this is harmless), is curled back and fails `no feedback` before its hip moves.""",
    """a knee with no hall feedback (hall A dead, cable off) runs blind to the 1 s cap at no more than PWM 128, so it can end **up to about 95 edges out** if it breaks free at once (measured 2026-11-24: FLKL driven out of its curl end at 128 had moved 72 edges 1.0 s after its drive started, with a ramp slower than the firmware's; the FR knees run about 120-130 edges/s at 128 and coast 3-9 edges after a stop; at PWM 80 this was about 41), is curled back and fails `no feedback` before its hip moves. Its hip is not proven curled at that point (the nudge is the one move allowed before that), so the knee nudge's PWM cap trades this blind reach against breaking a stored knee free: FRKL could not break free at 80 or 100. Whether to keep the cap at 128 or lower it to about 110 is the owner's call."""), (
    """(a knee with hall A dead runs blind to its 1 s cap, up to about 41 edges out with its hip curled, and fails `no feedback`)""",
    """(a knee with hall A dead runs blind to its 1 s cap, up to about 95 edges out at PWM 128 before its hip is proven curled, and fails `no feedback`)""")])
