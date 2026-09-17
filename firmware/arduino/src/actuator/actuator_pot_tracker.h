#pragma once

#include <stdint.h>

constexpr int POT_MINIMUM_RAW = 5; // at or below this the input reads as railed low
// Valid baselines must leave 10 counts of probe headroom below ADC saturation (1023).
constexpr int POT_MAXIMUM_RAW = 1014; // at or above this the input reads as railed high
constexpr int POT_IDLE_JITTER_MAX = 6; // max ADC drift per sample while undriven
constexpr uint8_t POT_INVALID_SAMPLE_LIMIT = 3; // consecutive bad samples before invalid

// With the AVR pull-up, attached pots moved 2 counts; open inputs read 1005-1020.
constexpr int POT_OPEN_PROBE_RISE = 10; // minimum ADC rise under pull-up indicating an open input

// Probe open inputs infrequently and only while reading.
constexpr uint32_t POT_PROBE_INTERVAL_MS = 2000;

class ActuatorPotTracker
{
public:
    ActuatorPotTracker()
        : previousAvgPot_(0), badSampleCount_(0), isValid_(true),
          isPositionOpen_(false), lastConnectionProbeMilliseconds_(0)
    {
    }

    bool isValid() const { return isValid_ && !isPositionOpen_; }

    void reset(float seedAvgPot)
    {
        previousAvgPot_ = seedAvgPot;
        badSampleCount_ = 0;
        isValid_ = true;
        isPositionOpen_ = false;
        lastConnectionProbeMilliseconds_ = 0;
    }

    bool isPositionOpen() const { return isPositionOpen_; }

    template <typename ReadConnectionProbeRise>
    bool update(
        float avgPot,
        bool isDriving,
        uint32_t nowMilliseconds,
        ReadConnectionProbeRise readConnectionProbeRise)
    {
        const bool isInSaneBand =
            avgPot > POT_MINIMUM_RAW && avgPot < POT_MAXIMUM_RAW;
        float delta = avgPot - previousAvgPot_;
        if (delta < 0)
            delta = -delta;
        previousAvgPot_ = avgPot;

        const bool isSlewValid = isDriving || delta <= POT_IDLE_JITTER_MAX;
        if (isInSaneBand && isSlewValid)
            badSampleCount_ = 0;
        else if (badSampleCount_ < POT_INVALID_SAMPLE_LIMIT)
            ++badSampleCount_;

        isValid_ = badSampleCount_ < POT_INVALID_SAMPLE_LIMIT;

        // An idle pull-up probe separates a low-impedance wiper from an open input.
        if (!isDriving &&
            nowMilliseconds - lastConnectionProbeMilliseconds_ >=
                POT_PROBE_INTERVAL_MS)
        {
            lastConnectionProbeMilliseconds_ = nowMilliseconds;
            const float rise = readConnectionProbeRise();
            isPositionOpen_ = rise >= POT_OPEN_PROBE_RISE;
        }

        return isValid();
    }

private:
    float previousAvgPot_;
    uint8_t badSampleCount_;  // saturates at POT_INVALID_SAMPLE_LIMIT
    bool isValid_;
    bool isPositionOpen_;
    uint32_t lastConnectionProbeMilliseconds_;
};
