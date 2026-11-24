"""Scratch-only candidate: a retract end (Switch, or a curl's Still) needs a flat current (motor cut)."""
import sys
src = open(sys.argv[1], newline="").read().replace("\r\n", "\n")
def rep(a, b, count=1):
    global src
    assert src.count(a) == count, (a, src.count(a))
    src = src.replace(a, b)
# 1. Switch: also flat over the last 8 samples, all of them after the last forward edge
rep("""                    return decide(CalibrationEnd::Switch, worst, stopRef_);""",
    """                    if (k - last >= kFlatSamples && flatBack())
                        return decide(CalibrationEnd::Switch, worst, stopRef_);""")
# 2. short-curl fallback on a retract: flat, else stuck
rep("""ran < CAL_END_SHORT_RUN_SAMPLES && (!retract_ || med4 <= iRef_))""",
    """ran < CAL_END_SHORT_RUN_SAMPLES && (!retract_ || (med4 <= iRef_ && flatBack())))""")
# 3. unarmed retract curl: a stall whose current is not flat is a stalled motor
rep("""            if (k >= minStillSamples_ && k >= window && progressOver(window) < CAL_HALL_STALL_MIN_EDGES)
                return decide(CalibrationEnd::Still, 0, 0);""",
    """            if (k >= minStillSamples_ && k >= window && progressOver(window) < CAL_HALL_STALL_MIN_EDGES)
                return decide(retract_ && !flatBack() ? CalibrationEnd::Stuck : CalibrationEnd::Still, 0, 0);""")
rep("""    // Median of the 6 speeds ending at the newest sample.""",
    """    static constexpr uint16_t kFlatSamples = 8;
    // The last 8 currents lie within 1.5 counts: the motor is cut (a switch is open).
    bool flatBack() const
    {
        uint16_t lo = 0xFFFFU, hi = 0;
        for (uint8_t back = 0; back < kFlatSamples; ++back)
        {
            const uint16_t value = current_[slot(back)];
            if (value < lo) lo = value;
            if (value > hi) hi = value;
        }
        return hi - lo <= 15;
    }

    // Median of the 6 speeds ending at the newest sample.""")
open(sys.argv[2], "w", newline="\n").write(src)
