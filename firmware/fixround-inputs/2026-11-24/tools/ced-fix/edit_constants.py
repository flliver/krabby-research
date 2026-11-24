import sys

p = sys.argv[1]
s = open(p, newline='').read()
assert '\r\n' in s
s = s.replace('\r\n', '\n')


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old, s.count(old))
    s = s.replace(old, new)


# F3: the blind knee nudge bound at the PWM 128 cap, measured.
rep("""// A knee with no hall feedback (hall A dead, cable off) is stopped only by the
// 1 s cap: a stored knee needs 0.36-0.77 s of drive to break free of its curl
// end (FLKL, 2026-11-23), so an earlier no-edge stop would also stop healthy
// knees, and a deadline at 0.9 s or later would save under 0.1 s of blind drive.
// Such a blind knee that breaks free early (0.36 s) drives about 0.64 s: at PWM
// 80 (about 50 edges/s) about 30 edges, then its spring-out, so up to about 41
// edges out; at the PWM 128 nudge cap an estimated 77 edges/s (not measured)
// gives about 48 edges, up to about 58 out (still with its hip curled, far
// inside the 776-edge knee stroke that clears the body) before it is curled
// back and fails NoFeedback.
""", """// A knee with no hall feedback (hall A dead, cable off) is stopped only by the
// 1 s cap: a stored knee needs 0.36-0.77 s of drive to break free of its curl
// end (FLKL, 2026-11-23), so an earlier no-edge stop would also stop healthy
// knees, and a deadline at 0.9 s or later would save under 0.1 s of blind drive.
// At PWM 80 such a blind knee ended up to about 41 edges out. At the PWM 128
// nudge cap it is about twice that, measured 2026-11-24: FLKL driven out of its
// curl end at 128 had moved 72 edges 1.0 s after its drive started (with a
// 0.64 s ramp; the firmware ramps in 0.52 s), the FR knees ran about 120-130
// edges/s at 128, and a knee coasts 3-9 edges after its stop, so a blind knee
// that breaks free at once ends up to about 95 edges out (inside the 776-edge
// knee stroke) before it is curled back and fails NoFeedback. Its hip is not
// proven curled yet: the nudge is the one move allowed before that.
""")
rep("""// The knee nudge never ramps past this PWM, whatever the C pwm (less overshoot
// and coast past its 12 edges). Tied to the default: pinned at 80 it could not
// break FRKL free (2026-11-24). Its overshoot at 128 is not measured yet.
""", """// The knee nudge never ramps past this PWM, whatever the C pwm (less overshoot
// and coast past its 12 edges). Tied to the default: pinned at 80 it could not
// break FRKL free, nor at 100 (2026-11-24). The cost is the blind knee bound
// above (about 95 edges out at 128 against about 41 at 80); a lower cap (about
// 110) would shorten it, but a knee that cannot break free then fails
// NoFeedback. The owner's choice.
""")
rep("""// end (where the count stopped, current dropped: "by=switch"), an extending rod's""", """// end (where the count stopped, current dropped: "by=switch"), an extending rod's""", 0) if False else None
# Detector: slow-arrival switch and short curl drive.
rep("""// Switch end (retract): the count still for 3 samples (150 ms); reference =
// median (and min) of the 20 samples (1 s) before the last forward edge, frozen
// per stop. A fast arrival (speed at that edge >= 50% of vRef) ends when the
// highest of the last 3 currents is <= 88% of it (hips drop 20-40%); a slow one
// (a knee crawling into its switch with its current raised) at <= 70% and below
// the reference minimum (crawl stalls without a switch dip only to 82-91%).
static constexpr uint8_t CAL_END_SWITCH_STILL_SAMPLES = 3;
static constexpr uint8_t CAL_END_SWITCH_REF_SAMPLES = 20;
static constexpr int CAL_END_ARRIVE_FAST_PCT = 50;
static constexpr int CAL_END_SWITCH_FAST_PCT = 88;
static constexpr int CAL_END_SWITCH_SLOW_PCT = 70;
""", """// Switch end (retract): the count still for 3 samples (150 ms); reference =
// median (and min) of the 20 samples (1 s) before the last forward edge, frozen
// per stop. A fast arrival (speed at that edge >= 50% of vRef) ends when the
// highest of the last 3 currents is <= 88% of it (hips drop 20-40%); a slow one
// (a joint crawling into its switch with its current raised: knee uncurls, and
// hips curling back from deep, at 0.50-0.62 of the reference) at <= 70%, below
// the reference minimum and <= 90% of iRef. A switch opens below the running
// level; a motor stalled mid-stroke draws more: FRHL curling at 128 from the
// body stalled 878 edges short of its switch (2026-11-24) with its current
// wobbling 17-31 counts over a running level of 16, and three samples at 17
// were 68% of the reference and below its minimum, but above 90% of iRef.
static constexpr uint8_t CAL_END_SWITCH_STILL_SAMPLES = 3;
static constexpr uint8_t CAL_END_SWITCH_REF_SAMPLES = 20;
static constexpr int CAL_END_ARRIVE_FAST_PCT = 50;
static constexpr int CAL_END_SWITCH_FAST_PCT = 88;
static constexpr int CAL_END_SWITCH_SLOW_PCT = 70;
static constexpr int CAL_END_SWITCH_SLOW_IREF_PCT = 90;
""")
rep("""// Stuck mid-stroke (armed, no signature): fewer than CAL_HALL_STALL_MIN_EDGES
// edges over CAL_STALL_WINDOW_MS and no forward edge for this long (a knee
// crawling into its switch at 120 made an edge at least every ~0.3 s).
static constexpr uint32_t CAL_END_STUCK_STILL_MS = 400;
""", """// Stuck mid-stroke (armed, no signature): fewer than CAL_HALL_STALL_MIN_EDGES
// edges over CAL_STALL_WINDOW_MS and no forward edge for this long (a knee
// crawling into its switch at 120 made an edge at least every ~0.3 s).
static constexpr uint32_t CAL_END_STUCK_STILL_MS = 400;
// Short curl drive: a curl that starts 50-130 edges from its end arms in the
// deceleration into it, so iRef and vRef are learned at the end and no later
// sample is normal running: its end never shows a signature against them. Such
// a stall ends Still (the plain stall) when the curl had fewer than this many
// normal-speed samples since arming (the 2026-11-24 curl ends replayed from a
// later start had at most 10; a full curl ending stuck had 27, a slowly
// stalling one 7, see the next) ...
static constexpr uint8_t CAL_END_SHORT_NORMAL_SAMPLES = 15;
// ... and its last forward edge came fewer than this many samples after arming
// (at most 25 on those replays; the FRHL stall above ran 71). A retract whose
// median current is above iRef (a motor stalled mid-stroke) stays Stuck.
static constexpr uint8_t CAL_END_SHORT_RUN_SAMPLES = 30;
""")
rep("""// Unarmed curl (a press at its end, a short return): Still after fewer than
// CAL_HALL_STALL_MIN_EDGES edges in this window, not before the minimum drive
// time (a Return drives at least as long as its nudge did). Pressing at an end
// thus takes about 1 s instead of about 3 s.
""", """// Unarmed curl (a press at its end, a short return, a joint slower than
// CAL_END_RUN_MIN_EDGES per 200 ms): Still after fewer than
// CAL_HALL_STALL_MIN_EDGES edges in this window, not before the minimum drive
// time (a Return drives at least as long as its nudge did, and a Return whose
// nudge proved no motion, a blind one, CAL_STALL_WINDOW_MS longer). Pressing
// at an end thus takes about 1 s instead of about 3 s. Once the drive has made
// CAL_HALL_CURL_EVIDENCE_EDGES edges the window is CAL_STALL_WINDOW_MS: a slow
// joint (PWM 40-50 hips run 10-20 edges/s) pausing 0.5 s mid-stroke is not at
// its curl end, as it was not before the detector.
""")
rep("""static_assert(CAL_END_STUCK_STILL_MS < CAL_STALL_WINDOW_MS,
              "stuck needs less stillness than the stall window");
""", """static_assert(CAL_END_STUCK_STILL_MS < CAL_STALL_WINDOW_MS,
              "stuck needs less stillness than the stall window");
static_assert(CAL_END_SWITCH_SLOW_IREF_PCT < 100 && CAL_END_SHORT_NORMAL_SAMPLES > 0 &&
                  CAL_END_SHORT_RUN_SAMPLES > 0,
              "a slow switch opens below the running level; the short-curl bounds must be positive");
static_assert(CAL_HIP_NUDGE_MAX_MS + CAL_STALL_WINDOW_MS < 0xFFFFUL,
              "a blind Return's minimum drive time must fit uint16");
""")
open(p, 'w', newline='').write(s.replace('\n', '\r\n'))
print('ok')
