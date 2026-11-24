import sys

p = sys.argv[1]
s = open(p, newline='').read()
crlf = '\r\n' in s
s = s.replace('\r\n', '\n')


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old, s.count(old))
    s = s.replace(old, new)


rep("""// Unarmed (no running phase yet: a press of a joint already at its end, a
// short return, a slow joint), only stall rules apply:
//   curl drive:   Still once the ramp is done, k * 50 >= minStillMs, k >= 10,
//                 and fewer than CAL_HALL_STALL_MIN_EDGES edges in the last
//                 CAL_END_STILL_WINDOW_MS.
""", """// Unarmed (no running phase yet: a press of a joint already at its end, a
// short return, a joint slower than arming speed), only stall rules apply, so
// such a drive's end is found by stall alone, as before the detector:
//   curl drive:   Still once the ramp is done, k * 50 >= minStillMs, and fewer
//                 than CAL_HALL_STALL_MIN_EDGES edges in the last
//                 CAL_END_STILL_WINDOW_MS (k >= 10), or in the last
//                 CAL_STALL_WINDOW_MS (k >= 20) once the drive has made
//                 CAL_HALL_CURL_EVIDENCE_EDGES edges (a slow joint pausing
//                 0.5 s mid-stroke is not at its end).
""")
rep("""// Armed, the PWM sign picks the end rule:
//   retract (-PWM, the rod's limit switch): Switch once still >= 3 and the
//     highest of the last 3 currents is <= 88% of the reference (the median of
//     up to 20 samples before the last forward edge), or after a slow arrival
//     (speed at the last edge < 50% of vRef: a crawl into the switch) <= 70%
//     and below every reference sample.
//   extend (+PWM, a mechanical stop): Stop after 3 samples in a row with
//     v <= 25% of vRef and the median of the last 4 currents >= 125% of iRef.
//   either: Stuck once k >= 20, fewer than CAL_HALL_STALL_MIN_EDGES edges in
//     the last 1000 ms and no forward edge for CAL_END_STUCK_STILL_MS. A stall
//     with neither signature is never an end.
""", """// Armed, the PWM sign picks the end rule:
//   retract (-PWM, the rod's limit switch): Switch once still >= 3 and the
//     highest of the last 3 currents is <= 88% of the reference (the median of
//     up to 20 samples before the last forward edge), or after a slow arrival
//     (speed at the last edge < 50% of vRef: a crawl into the switch) <= 70%,
//     below every reference sample and <= 90% of iRef (a switch opens below
//     the running level; a stalled motor's current wobbles above it).
//   extend (+PWM, a mechanical stop): Stop after 3 samples in a row with
//     v <= 25% of vRef and the median of the last 4 currents >= 125% of iRef.
//   either: Stuck once k >= 20, fewer than CAL_HALL_STALL_MIN_EDGES edges in
//     the last 1000 ms and no forward edge for CAL_END_STUCK_STILL_MS. A stall
//     with neither signature after a running phase is never an end.
//   Short curl drive: a curl that armed in its own end approach (the joint
//     started 50-130 edges from its end, so the six arming samples fall in the
//     deceleration into it and iRef and vRef are already the end's) has no
//     running phase to judge a signature against. Such a stall (fewer than
//     CAL_END_SHORT_NORMAL_SAMPLES normal-speed samples since arming, its last
//     forward edge fewer than CAL_END_SHORT_RUN_SAMPLES samples after arming)
//     ends Still, the plain stall of an unarmed drive, unless it is a retract
//     whose median current is above iRef (a stalled motor, never a switch).
""")
rep("""    Stuck,   // armed: no progress and no end signature
    Still,   // unarmed: no progress (stall-only rule, as before the detector)
""", """    Stuck,   // armed: no progress and no end signature
    Still,   // unarmed or a short curl: no progress (stall-only rule, as before the detector)
""")
rep("""        runStreak_ = 0;
        stopStreak_ = 0;
""", """        runStreak_ = 0;
        stopStreak_ = 0;
        normalSamples_ = 0;
""")
rep("""            iRef_ = static_cast<uint16_t>(iRef_ + step);
            if (v > vRef_)
                vRef_ = v;
        }""", """            iRef_ = static_cast<uint16_t>(iRef_ + step);
            if (v > vRef_)
                vRef_ = v;
            if (normalSamples_ < 0xFFU)
                ++normalSamples_;
        }""")
rep("""                if (static_cast<uint32_t>(worst) * 100UL <= pct * stopRef_ &&
                    (stopFast_ || worst < stopMin_))
                {""", """                if (static_cast<uint32_t>(worst) * 100UL <= pct * stopRef_ &&
                    (stopFast_ ||
                     (worst < stopMin_ &&
                      static_cast<uint32_t>(worst) * 100UL <=
                          static_cast<uint32_t>(CAL_END_SWITCH_SLOW_IREF_PCT) * iRef_)))
                {""")
rep("""        const uint16_t still = static_cast<uint16_t>(k - (hasForward_ ? lastForward_ : 0));
        if (k >= kStallSamples && progressOver(kStallSamples) < CAL_HALL_STALL_MIN_EDGES &&
            static_cast<uint32_t>(still) * CAL_END_SAMPLE_MS >= CAL_END_STUCK_STILL_MS)
        {
            return decide(CalibrationEnd::Stuck, med4, iRef_);
        }
        return CalibrationEnd::None;""", """        const uint16_t still = static_cast<uint16_t>(k - (hasForward_ ? lastForward_ : 0));
        if (k >= kStallSamples && progressOver(kStallSamples) < CAL_HALL_STALL_MIN_EDGES &&
            static_cast<uint32_t>(still) * CAL_END_SAMPLE_MS >= CAL_END_STUCK_STILL_MS)
        {
            const uint16_t ran = static_cast<uint16_t>(
                (hasForward_ && lastForward_ > kArm_ ? lastForward_ : kArm_) - kArm_);
            if (!leaving_ && normalSamples_ < CAL_END_SHORT_NORMAL_SAMPLES &&
                ran < CAL_END_SHORT_RUN_SAMPLES && (!retract_ || med4 <= iRef_))
            {
                return decide(CalibrationEnd::Still, med4, iRef_);
            }
            return decide(CalibrationEnd::Stuck, med4, iRef_);
        }
        return CalibrationEnd::None;""")
rep("""    // Samples taken since begin().
    uint16_t samples() const { return count_; }""", """    // Samples taken since begin().
    uint16_t samples() const { return count_; }
    // Armed samples at normal speed (they moved the references), saturating.
    uint8_t normalSamples() const { return normalSamples_; }""")
rep("""    // The decision's current and the reference it was compared with (tenths):
    // Switch: the highest of the last 3 samples / the frozen stop reference;
    // Stop and Stuck: the median of the last 4 samples / iRef.""", """    // The decision's current and the reference it was compared with (tenths):
    // Switch: the highest of the last 3 samples / the frozen stop reference;
    // Stop, Stuck and a short curl's Still: the median of the last 4 samples /
    // iRef; an unarmed Still: 0 / 0.""")
rep("""        if (!leaving_)
        {
            if (k >= minStillSamples_ && k >= kStillWindowSamples &&
                progressOver(kStillWindowSamples) < CAL_HALL_STALL_MIN_EDGES)
            {
                return decide(CalibrationEnd::Still, 0, 0);
            }
        }""", """        if (!leaving_)
        {
            const uint16_t window = e >= CAL_HALL_CURL_EVIDENCE_EDGES ? kStallSamples : kStillWindowSamples;
            if (k >= minStillSamples_ && k >= window && progressOver(window) < CAL_HALL_STALL_MIN_EDGES)
                return decide(CalibrationEnd::Still, 0, 0);
        }""")
rep("""    uint8_t runStreak_;
    uint8_t stopStreak_;
""", """    uint8_t runStreak_;
    uint8_t stopStreak_;
    uint8_t normalSamples_;
""")
open(p, 'w', newline='').write(s.replace('\n', '\r\n'))
print('ok', 'was crlf' if crlf else 'converted to crlf')
