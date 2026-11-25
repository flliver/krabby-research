#pragma once

#include <stdint.h>

#include "calibration_constants.h"

// End-of-travel detector for one hall calibration drive (LegCalibration): it
// tells an actuator's internal limit switch, a mechanical stop and a joint
// stuck mid-stroke apart from the drive edges and the motor current, so the
// drive stops as soon as an end shows instead of pressing into it for seconds.
//
// Samples: every CAL_END_SAMPLE_MS from the drive start (update(), once per
// loop) the detector closes one sample: e = the drive edges at that moment
// (clamped to int16) and i = the boxcar mean of the current tenths passed over
// the sample period (10 x avgIS, rounded). sample() is the same step with the
// sample given directly (the robot-trace replays feed it).
//
// v = e_k - e_(k-4): edges per 200 ms. Forward progress is a new maximum of e
// (a bounce back at a stop is not progress); still = samples since the last one.
//
// The switch-open signature (2026-11-24, every hip and knee limit switch on the
// FL and FR legs): once an actuator's internal limit switch opens, the motor no
// longer conducts and the current sits flat at the IS channel's own offset
// (FL 7-8 counts, FRHL 11), the level it also reads at the start of a drive
// while the motor draws almost nothing. A motor stalled mid-stroke still
// conducts: its current sits above that level and wobbles (FL at PWM 80: 9-14;
// FRHL at 128: 17-31). So the detector keeps the drive's floor, the lowest
// current of samples CAL_END_FLOOR_FIRST_SAMPLE.. through the ramp (the PWM
// still low, the IS settled after EN went high), and a switch end must show
// the last CAL_END_OPEN_SAMPLES currents at most CAL_END_OPEN_MARGIN_TENTHS
// above that floor and within CAL_END_OPEN_FLAT_TENTHS of each other
// (switchOpen). It is measured per drive, so it needs no per-joint table.
//
// Unarmed (no running phase yet: a press of a joint already at its end, a
// short return, a joint slower than arming speed), only stall rules apply:
//   curl drive:   a stall once the ramp is done, k * 50 >= minStillMs, and
//                 fewer than CAL_HALL_STALL_MIN_EDGES edges in the last
//                 CAL_END_STILL_WINDOW_MS (k >= 10), or in the last
//                 CAL_STALL_WINDOW_MS (k >= 20) once the drive has made
//                 CAL_HALL_CURL_EVIDENCE_EDGES edges (a slow joint pausing
//                 0.5 s mid-stroke is not at its end).
//   uncurl drive: a stall CAL_START_GRACE_MS + CAL_STALL_WINDOW_MS after the
//                 ramp with fewer than CAL_HALL_STALL_MIN_EDGES edges in the
//                 last CAL_STALL_WINDOW_MS (the time a joint has to break free).
//   A stall of an extend drive ends Still (by stall alone, as before the
//   detector: a knee's leg stop or an obstacle). A stall of a drive into a
//   limit switch (a retract, or a hip extend) waits until the count has been
//   still CAL_END_OPEN_SETTLE_SAMPLES samples, then ends Still only with the
//   switch-open signature and Stuck without it (judged again for up to
//   CAL_END_OPEN_WAIT_SAMPLES while its current is at the floor or falling): a
//   hip curl that never reached running speed (a press mid-stroke, a nudge
//   return, a start close to a stall point) is never taken as its home on a
//   stall alone. (The sequencer sends a Stuck press that counted no edge at
//   all to its verification nudge instead of failing it.)
// Arming: from CAL_END_LEARN_DELAY_MS after the ramp, CAL_END_ARM_SAMPLES
// samples in a row with v >= CAL_END_RUN_MIN_EDGES and at least
// CAL_HALL_CURL_EVIDENCE_EDGES drive edges. Then iRef = median of those six
// currents and vRef = median of their speeds; afterwards every sample at normal
// speed (v >= CAL_END_RUN_MIN_EDGES and >= CAL_END_NORMAL_SPEED_PCT of vRef)
// moves iRef by at most CAL_END_IREF_STEP toward its current and raises vRef to
// its speed (peak hold).
// Armed, the drive's end kinds pick the rules:
//   switch end (a retract into the rod's limit switch, and a hip extend: its
//     extended end is a limit switch too): Switch once still >= 3, the
//     highest of the last 3 currents is <= 88% of the reference (the median of
//     up to 20 samples before the last forward edge), or after a slow arrival
//     (the fastest of the speeds at the last forward edge and the two samples
//     before it < 50% of vRef: a crawl into the switch) also below every
//     reference sample and <= 90% of iRef, and the current shows switchOpen.
//   extend (+PWM: a knee's mechanical stop, or for a hip an obstacle): Stop
//     after 3 samples in a row with v <= 25% of vRef and the median of the last
//     4 currents >= 125% of iRef. On a hip extend (its end a switch) the Stop
//     also needs the count still CAL_END_SWITCH_STOP_STILL_SAMPLES samples,
//     unless the streak reaches CAL_END_SWITCH_STOP_LONG_SAMPLES (a creeping
//     obstacle): the current rises 25-50% in the last edges before the hip's
//     extend switch opens (2026-11-24, FLHL 3 of 3), and the switch must get
//     the chance to show its drop first.
//   either: Stuck once k >= 20, fewer than CAL_HALL_STALL_MIN_EDGES edges in
//     the last 1000 ms and no forward edge for CAL_END_STUCK_STILL_MS
//     (CAL_END_SWITCH_STUCK_STILL_MS on a switch-end drive, which also waits
//     while the current is still falling, a switch drop in progress, or
//     already sits at the floor, up to CAL_END_STUCK_STILL_MAX_MS). A stall with neither signature after a
//     running phase is never an end.
//   Short curl drive: a curl that armed in its own end approach (the joint
//     started 50-130 edges from its end, so the six arming samples fall in the
//     deceleration into it and iRef and vRef are already the end's) has no
//     running phase to judge a signature against. Such a stall (fewer than
//     CAL_END_SHORT_NORMAL_SAMPLES normal-speed samples since arming, its last
//     forward edge fewer than CAL_END_SHORT_RUN_SAMPLES samples after arming)
//     ends Still, the plain stall, on an extend drive (a knee's leg stop: an
//     obstacle on a knee curl looks the same), and on a switch-end drive only
//     with switchOpen, else Stuck.
// So a stall is taken as an end without a signature only on a knee curl (an
// obstacle there looks like the leg stop). A hip extend's end is a switch: a
// stall there without a signature is Stuck, and the sequencer treats any end
// of a hip sweep before its target, of any kind, as an obstacle. The
// robot traces this was checked against (2026-11-23/24) are named in the
// native test.

// The detector's current from a smoothed IS reading (avgIS): round(10 * avgIS),
// clamped to [0, 65535].
inline uint16_t calibrationCurrentTenths(float avgIS)
{
    const float tenths = avgIS * 10.0f + 0.5f;
    if (!(tenths > 0.0f))
        return 0;
    if (tenths >= 65535.0f)
        return 0xFFFFU;
    return static_cast<uint16_t>(tenths);
}

enum class CalibrationEnd : uint8_t
{
    None,
    Switch,  // armed switch end: the count stopped, the current dropped to switch-open
    Stop,    // armed extend: the speed collapsed and the current rose
    Stuck,   // no progress and no end signature (a switch-end stall without switch-open)
    Still,   // a stall without a running phase to judge (extend), or one at switch-open
};

class CalibrationEndDetector
{
public:
    CalibrationEndDetector() { begin(0, false, false, CAL_END_STILL_MIN_DRIVE_MS); }

    // A new drive starting at nowMs. retract: -PWM (limit-switch end), else the
    // extend rules. leaving: an uncurl (it must break free of the curl end).
    // minStillMs: the least drive time before an unarmed curl may end Still.
    // extendSwitch: an extend drive whose end is also a limit switch (a hip:
    // the switch rule and switchOpen apply as on a retract, besides the stop
    // rule, which then means an obstacle).
    void begin(uint32_t nowMs, bool retract, bool leaving, uint32_t minStillMs,
               bool extendSwitch = false)
    {
        retract_ = retract;
        leaving_ = leaving;
        switchEnd_ = retract || extendSwitch;
        const uint32_t samples = minStillMs / CAL_END_SAMPLE_MS + (minStillMs % CAL_END_SAMPLE_MS != 0 ? 1U : 0U);
        minStillSamples_ = samples > 0xFFFFUL ? 0xFFFFU : static_cast<uint16_t>(samples);
        nextSampleMs_ = nowMs + CAL_END_SAMPLE_MS;
        sum_ = 0;
        loops_ = 0;
        count_ = 0;
        head_ = 0;
        rampSeen_ = false;
        kRamp_ = 0;
        hasForward_ = false;
        lastForward_ = 0;
        eMax_ = 0;
        armed_ = false;
        kArm_ = 0;
        runStreak_ = 0;
        stopStreak_ = 0;
        normalSamples_ = 0;
        iRef_ = 0;
        vRef_ = 0;
        stopRefValid_ = false;
        stopRef_ = 0;
        stopMin_ = 0;
        stopFast_ = false;
        floor_ = 0xFFFFU;
        openWait_ = 0;
        endCurrent_ = 0;
        endReference_ = 0;
        for (uint8_t n = 0; n < CAL_END_RING; ++n)
        {
            edges_[n] = 0;
            current_[n] = 0;
        }
    }

    // Once per loop of the drive (every phase but Settle). rampDone: the drive
    // has reached its move PWM. Returns the end decided on this call, if any.
    CalibrationEnd update(uint32_t nowMs, int32_t driveEdges, uint16_t currentTenths, bool rampDone)
    {
        if (loops_ < 0xFFFFU)
        {
            sum_ += currentTenths;
            ++loops_;
        }
        if (static_cast<int32_t>(nowMs - nextSampleMs_) < 0)
            return CalibrationEnd::None;
        const uint16_t mean = static_cast<uint16_t>((sum_ + loops_ / 2U) / loops_);
        sum_ = 0;
        loops_ = 0;
        nextSampleMs_ += CAL_END_SAMPLE_MS;
        // After a long loop: one sample, then resynchronise (never a burst).
        if (static_cast<int32_t>(nowMs - nextSampleMs_) >= 0)
            nextSampleMs_ = nowMs + CAL_END_SAMPLE_MS;
        return sample(driveEdges, mean, rampDone);
    }

    // One closed sample (k = the samples taken before it).
    CalibrationEnd sample(int32_t driveEdges, uint16_t currentTenths, bool rampDone)
    {
        const int16_t e = driveEdges > 32767 ? static_cast<int16_t>(32767)
            : driveEdges < -32768 ? static_cast<int16_t>(-32768)
            : static_cast<int16_t>(driveEdges);
        const uint16_t k = count_;
        if (count_ < 0xFFFFU)
            ++count_;
        head_ = k == 0 ? 0 : static_cast<uint8_t>(head_ + 1U == CAL_END_RING ? 0 : head_ + 1U);
        edges_[head_] = e;
        current_[head_] = currentTenths;

        if (k == 0)
        {
            eMax_ = e;
        }
        else if (e > eMax_)
        {
            eMax_ = e;
            hasForward_ = true;
            lastForward_ = k;
            stopRefValid_ = false;
        }
        // The floor: from CAL_END_FLOOR_FIRST_SAMPLE through the sample that
        // completes the ramp (at least through CAL_END_FLOOR_LAST_MIN_SAMPLE).
        if (k >= CAL_END_FLOOR_FIRST_SAMPLE &&
            (!rampSeen_ || k <= CAL_END_FLOOR_LAST_MIN_SAMPLE) && currentTenths < floor_)
        {
            floor_ = currentTenths;
        }
        if (!rampSeen_)
        {
            if (!rampDone)
                return CalibrationEnd::None;
            rampSeen_ = true;
            kRamp_ = k;
        }
        const uint16_t sinceRamp = static_cast<uint16_t>(k - kRamp_);
        const int32_t v = speedAt(k, 0);

        if (!armed_)
            return unarmed(k, sinceRamp, v, e);

        if (v >= CAL_END_RUN_MIN_EDGES &&
            v * 100 >= static_cast<int32_t>(CAL_END_NORMAL_SPEED_PCT) * vRef_)
        {
            int32_t step = static_cast<int32_t>(currentTenths) - iRef_;
            if (step > CAL_END_IREF_STEP)
                step = CAL_END_IREF_STEP;
            if (step < -CAL_END_IREF_STEP)
                step = -CAL_END_IREF_STEP;
            iRef_ = static_cast<uint16_t>(iRef_ + step);
            if (v > vRef_)
                vRef_ = v;
            if (normalSamples_ < 0xFFU)
                ++normalSamples_;
        }
        const uint16_t med4 = medianBack(0, 4, nullptr);
        const uint16_t still = static_cast<uint16_t>(k - (hasForward_ ? lastForward_ : 0));

        if (switchEnd_)
        {
            const uint16_t last = hasForward_ ? lastForward_ : kArm_;
            if (k - last >= CAL_END_SWITCH_STILL_SAMPLES)
            {
                if (!stopRefValid_)
                    freezeStop(k, last);
                uint16_t worst = 0;
                for (uint8_t back = 0; back < CAL_END_SWITCH_STILL_SAMPLES; ++back)
                {
                    const uint16_t value = current_[slot(back)];
                    if (value > worst)
                        worst = value;
                }
                const uint32_t pct = stopFast_ ? CAL_END_SWITCH_FAST_PCT : CAL_END_SWITCH_SLOW_PCT;
                if (static_cast<uint32_t>(worst) * 100UL <= pct * stopRef_ &&
                    (stopFast_ ||
                     (worst < stopMin_ &&
                      static_cast<uint32_t>(worst) * 100UL <=
                          static_cast<uint32_t>(CAL_END_SWITCH_SLOW_IREF_PCT) * iRef_)) &&
                    switchOpen())
                {
                    return decide(CalibrationEnd::Switch, worst, stopRef_);
                }
            }
        }
        if (!retract_)
        {
            const bool slow = v * 100 <=
                static_cast<int32_t>(CAL_END_STOP_SPEED_PCT) * vRef_;
            const bool hot = static_cast<uint32_t>(med4) * 100UL >=
                static_cast<uint32_t>(CAL_END_STOP_CURRENT_PCT) * iRef_;
            stopStreak_ = slow && hot ? static_cast<uint8_t>(stopStreak_ + 1U) : 0;
            // A hip extend (switchEnd_) gives its extend switch the chance to
            // show its drop before the rising current counts as an obstacle.
            if (stopStreak_ >= CAL_END_STOP_HOLD_SAMPLES &&
                (!switchEnd_ || still >= CAL_END_SWITCH_STOP_STILL_SAMPLES ||
                 stopStreak_ >= CAL_END_SWITCH_STOP_LONG_SAMPLES))
            {
                return decide(CalibrationEnd::Stop, med4, iRef_);
            }
        }

        if (k >= kStallSamples && progressOver(kStallSamples) < CAL_HALL_STALL_MIN_EDGES &&
            static_cast<uint32_t>(still) * CAL_END_SAMPLE_MS >=
                (switchEnd_ ? CAL_END_SWITCH_STUCK_STILL_MS : CAL_END_STUCK_STILL_MS))
        {
            // A switch drop in progress, or a current already down at the
            // floor (not pressing): wait, bounded, for switchOpen's verdict.
            if (switchEnd_ && (currentFalling() || atFloor()) &&
                static_cast<uint32_t>(still) * CAL_END_SAMPLE_MS < CAL_END_STUCK_STILL_MAX_MS)
            {
                return CalibrationEnd::None;
            }
            const uint16_t ran = static_cast<uint16_t>(
                (hasForward_ && lastForward_ > kArm_ ? lastForward_ : kArm_) - kArm_);
            if (!leaving_ && normalSamples_ < CAL_END_SHORT_NORMAL_SAMPLES &&
                ran < CAL_END_SHORT_RUN_SAMPLES &&
                (!switchEnd_ || (med4 <= iRef_ && switchOpen())))
            {
                return decide(CalibrationEnd::Still, med4, switchEnd_ ? floor_ : iRef_);
            }
            return decide(CalibrationEnd::Stuck, med4, iRef_);
        }
        return CalibrationEnd::None;
    }

    bool isArmed() const { return armed_; }
    // The newest sample's drive edges and current tenths.
    int16_t lastSampleEdges() const { return edges_[head_]; }
    uint16_t lastSampleCurrent() const { return current_[head_]; }
    // Samples taken since begin().
    uint16_t samples() const { return count_; }
    // Armed samples at normal speed (they moved the references), saturating.
    uint8_t normalSamples() const { return normalSamples_; }
    // Running references (tenths of an IS count; edges per 200 ms), once armed.
    uint16_t currentReference() const { return iRef_; }
    int32_t speedReference() const { return vRef_; }
    // The decision's current and the reference it was compared with (tenths):
    // Switch: the highest of the last 3 samples / the frozen stop reference;
    // Stop and Stuck: the median of the last 4 samples / iRef; a short curl's
    // Still: the median of the last 4 / iRef (extend) or the floor (switch
    // end); an unarmed drive's Still or Stuck: the highest of the last
    // CAL_END_OPEN_SAMPLES / the floor (0 / 0 for an extend drive's Still).
    uint16_t endCurrent() const { return endCurrent_; }
    uint16_t endReference() const { return endReference_; }
    // The drive's floor (tenths; 65535 until sample CAL_END_FLOOR_FIRST_SAMPLE).
    uint16_t currentFloor() const { return floor_; }
    // The last CAL_END_OPEN_SAMPLES currents sit flat at the floor (the
    // switch-open signature).
    bool isSwitchOpen() const { return switchOpen(); }

private:
    static constexpr uint16_t kStallSamples =
        static_cast<uint16_t>(CAL_STALL_WINDOW_MS / CAL_END_SAMPLE_MS);
    static constexpr uint16_t kStillWindowSamples =
        static_cast<uint16_t>(CAL_END_STILL_WINDOW_MS / CAL_END_SAMPLE_MS);
    static constexpr uint16_t kUncurlStillSamples =
        static_cast<uint16_t>((CAL_START_GRACE_MS + CAL_STALL_WINDOW_MS) / CAL_END_SAMPLE_MS);
    static constexpr uint16_t kLearnSamples =
        static_cast<uint16_t>(CAL_END_LEARN_DELAY_MS / CAL_END_SAMPLE_MS);

    // Ring slot of the sample `back` samples before the newest (back < ring).
    uint8_t slot(uint16_t back) const
    {
        return static_cast<uint8_t>(head_ >= back ? head_ - back : head_ + CAL_END_RING - back);
    }

    int16_t edgesBack(uint16_t back) const { return edges_[slot(back)]; }

    // Drive edges gained over the last `samples` samples.
    int32_t progressOver(uint16_t samples) const
    {
        return static_cast<int32_t>(edgesBack(0)) - edgesBack(samples);
    }

    // v at sample k - back: edges over the 4 samples before it (0 before k = 4).
    int32_t speedAt(uint16_t k, uint16_t back) const
    {
        if (k < back + 4U)
            return 0;
        return static_cast<int32_t>(edgesBack(back)) - edgesBack(static_cast<uint16_t>(back + 4U));
    }

    // Median (sorted[n / 2]) of the currents of the n samples ending `back`
    // samples before the newest; min receives the smallest when given.
    uint16_t medianBack(uint16_t back, uint8_t n, uint16_t *min) const
    {
        uint16_t sorted[CAL_END_SWITCH_REF_SAMPLES];
        for (uint8_t m = 0; m < n; ++m)
        {
            const uint16_t value = current_[slot(static_cast<uint16_t>(back + m))];
            uint8_t at = m;
            while (at > 0 && sorted[at - 1] > value)
            {
                sorted[at] = sorted[at - 1];
                --at;
            }
            sorted[at] = value;
        }
        if (min != nullptr)
            *min = sorted[0];
        return sorted[n / 2];
    }

    // Median of the 6 speeds ending at the newest sample.
    int32_t medianSpeed(uint16_t k) const
    {
        int32_t sorted[CAL_END_ARM_SAMPLES];
        for (uint8_t m = 0; m < CAL_END_ARM_SAMPLES; ++m)
        {
            const int32_t value = speedAt(k, m);
            uint8_t at = m;
            while (at > 0 && sorted[at - 1] > value)
            {
                sorted[at] = sorted[at - 1];
                --at;
            }
            sorted[at] = value;
        }
        return sorted[CAL_END_ARM_SAMPLES / 2];
    }

    // The highest and lowest of the last CAL_END_OPEN_SAMPLES currents.
    void openSpan(uint16_t &high, uint16_t &low) const
    {
        high = 0;
        low = 0xFFFFU;
        for (uint8_t back = 0; back < CAL_END_OPEN_SAMPLES; ++back)
        {
            const uint16_t value = current_[slot(back)];
            if (value > high)
                high = value;
            if (value < low)
                low = value;
        }
    }

    // The switch-open signature: the last CAL_END_OPEN_SAMPLES currents within
    // CAL_END_OPEN_FLAT_TENTHS of each other and at most
    // CAL_END_OPEN_MARGIN_TENTHS above the floor. Needs a floor and enough
    // samples (callers judge only after CAL_END_OPEN_SAMPLES samples).
    bool switchOpen() const
    {
        if (floor_ == 0xFFFFU || count_ < CAL_END_OPEN_SAMPLES)
            return false;
        uint16_t high = 0;
        uint16_t low = 0;
        openSpan(high, low);
        return static_cast<uint32_t>(high) <= static_cast<uint32_t>(floor_) + CAL_END_OPEN_MARGIN_TENTHS &&
               high - low <= CAL_END_OPEN_FLAT_TENTHS;
    }

    // The newest current is at most CAL_END_OPEN_MARGIN_TENTHS above the floor.
    bool atFloor() const
    {
        return floor_ != 0xFFFFU &&
               static_cast<uint32_t>(current_[slot(0)]) <= static_cast<uint32_t>(floor_) + CAL_END_OPEN_MARGIN_TENTHS;
    }

    // The newest current is more than CAL_END_FALL_TENTHS below the higher of
    // the two before it.
    bool currentFalling() const
    {
        const uint16_t newest = current_[slot(0)];
        const uint16_t one = current_[slot(1)];
        const uint16_t two = current_[slot(2)];
        const uint16_t before = one > two ? one : two;
        return static_cast<uint32_t>(newest) + CAL_END_FALL_TENTHS < before;
    }

    // An unarmed stall: Still on an extend drive; on a switch-end drive, once
    // the count has been still CAL_END_OPEN_SETTLE_SAMPLES, Still with the
    // switch-open signature and Stuck without it.
    CalibrationEnd unarmedStall(uint16_t k)
    {
        if (!switchEnd_)
            return decide(CalibrationEnd::Still, 0, 0);
        const uint16_t still = static_cast<uint16_t>(k - (hasForward_ ? lastForward_ : 0));
        if (still < CAL_END_OPEN_SETTLE_SAMPLES)
            return CalibrationEnd::None;
        uint16_t high = 0;
        uint16_t low = 0;
        openSpan(high, low);
        if (switchOpen())
            return decide(CalibrationEnd::Still, high, floor_);
        if ((atFloor() || currentFalling()) && openWait_ < CAL_END_OPEN_WAIT_SAMPLES)
        {
            ++openWait_;
            return CalibrationEnd::None;
        }
        return decide(CalibrationEnd::Stuck, high, floor_);
    }

    CalibrationEnd unarmed(uint16_t k, uint16_t sinceRamp, int32_t v, int16_t e)
    {
        if (!leaving_)
        {
            const uint16_t window = e >= CAL_HALL_CURL_EVIDENCE_EDGES ? kStallSamples : kStillWindowSamples;
            if (k >= minStillSamples_ && k >= window && progressOver(window) < CAL_HALL_STALL_MIN_EDGES)
            {
                const CalibrationEnd kind = unarmedStall(k);
                if (kind != CalibrationEnd::None)
                    return kind;
            }
        }
        else if (sinceRamp >= kUncurlStillSamples &&
                 progressOver(kStallSamples) < CAL_HALL_STALL_MIN_EDGES)
        {
            const CalibrationEnd kind = unarmedStall(k);
            if (kind != CalibrationEnd::None)
                return kind;
        }
        if (sinceRamp < kLearnSamples)
            return CalibrationEnd::None;
        runStreak_ = v >= CAL_END_RUN_MIN_EDGES ? static_cast<uint8_t>(runStreak_ + 1U) : 0;
        if (runStreak_ >= CAL_END_ARM_SAMPLES && e >= CAL_HALL_CURL_EVIDENCE_EDGES)
        {
            armed_ = true;
            kArm_ = k;
            iRef_ = medianBack(0, CAL_END_ARM_SAMPLES, nullptr);
            vRef_ = medianSpeed(k);
        }
        return CalibrationEnd::None;
    }

    // The switch reference of this stop: the up to 20 samples before the last
    // forward edge (that sample's own current may already be falling), and
    // whether the joint arrived at speed.
    void freezeStop(uint16_t k, uint16_t last)
    {
        const uint16_t armFirst = static_cast<uint16_t>(kArm_ - (CAL_END_ARM_SAMPLES - 1U));
        uint16_t lo = last >= CAL_END_SWITCH_REF_SAMPLES
            ? static_cast<uint16_t>(last - CAL_END_SWITCH_REF_SAMPLES)
            : 0;
        if (armFirst > lo)
            lo = armFirst;
        // A stop is frozen at k = last + 3, or on the first sample after
        // arming (lo >= kArm - 5), so k - lo stays inside the ring
        // (static_asserts in calibration_constants.h).
        const uint16_t hi = last > lo ? last : static_cast<uint16_t>(lo + 1U);
        const uint8_t n = static_cast<uint8_t>(hi - lo);
        stopRef_ = medianBack(static_cast<uint16_t>(k - (hi - 1U)), n, &stopMin_);
        // The arrival speed: the fastest of the speeds at the last forward edge
        // and the CAL_END_ARRIVE_LOOKBACK_SAMPLES samples before it, so that one
        // straggling edge a sample late does not turn a fast arrival into a
        // slow one. A last forward edge older than the ring (only after a large
        // bounce back before arming) counts as a slow arrival.
        const uint16_t since = static_cast<uint16_t>(k - last);
        int32_t arrive = 0;
        for (uint8_t extra = 0; extra <= CAL_END_ARRIVE_LOOKBACK_SAMPLES; ++extra)
        {
            const uint16_t back = static_cast<uint16_t>(since + extra);
            if (back + 4U >= CAL_END_RING)
                break;
            const int32_t speed = speedAt(k, back);
            if (speed > arrive)
                arrive = speed;
        }
        stopFast_ = arrive * 100 >=
            static_cast<int32_t>(CAL_END_ARRIVE_FAST_PCT) * vRef_;
        stopRefValid_ = true;
    }

    CalibrationEnd decide(CalibrationEnd kind, uint16_t current, uint16_t reference)
    {
        endCurrent_ = current;
        endReference_ = reference;
        return kind;
    }

    int16_t edges_[CAL_END_RING];
    uint16_t current_[CAL_END_RING];
    uint32_t nextSampleMs_;
    uint32_t sum_;
    uint16_t loops_;
    uint16_t count_;
    uint16_t minStillSamples_;
    uint16_t kRamp_;
    uint16_t lastForward_;
    uint16_t kArm_;
    uint16_t iRef_;
    uint16_t stopRef_;
    uint16_t stopMin_;
    uint16_t endCurrent_;
    uint16_t endReference_;
    uint16_t floor_;
    int32_t vRef_;
    int16_t eMax_;
    uint8_t head_;
    uint8_t runStreak_;
    uint8_t stopStreak_;
    uint8_t normalSamples_;
    uint8_t openWait_;  // samples an unarmed switch-end stall waited for switchOpen
    bool retract_;
    bool leaving_;
    bool switchEnd_;  // a retract, or a hip extend: the end is a limit switch
    bool rampSeen_;
    bool hasForward_;
    bool armed_;
    bool stopRefValid_;
    bool stopFast_;
};
