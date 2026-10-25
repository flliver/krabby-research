#pragma once

#include <stdint.h>

enum class ActuatorCurrentEvidence : uint8_t
{
    Unknown,
    CurrentPresent,
    CurrentAbsent,
};

// Attached channels measured 1-7 ADC counts; empty channels never sustained 1.
// Current was flat across PWM 50-255.
constexpr int ACTUATOR_CURRENT_PRESENT_FLOOR = 1;
constexpr int ACTUATOR_CURRENT_PROBE_PWM = 50;
constexpr uint8_t ACTUATOR_CURRENT_DEBOUNCE_SAMPLES = 3;

class ActuatorCurrentTracker
{
public:
    ActuatorCurrentTracker()
        : evidence_(ActuatorCurrentEvidence::Unknown),
          presentEvidenceCount_(0), absentEvidenceCount_(0)
    {
    }

    ActuatorCurrentEvidence evidence() const { return evidence_; }

    void reset()
    {
        evidence_ = ActuatorCurrentEvidence::Unknown;
        presentEvidenceCount_ = 0;
        absentEvidenceCount_ = 0;
    }

    void update(
        bool isDriving,
        int commandedPwm,
        float measuredCurrent)
    {
        int pwmMagnitude = commandedPwm;
        if (pwmMagnitude < 0)
            pwmMagnitude = -pwmMagnitude;
        if (!isDriving ||
            pwmMagnitude < ACTUATOR_CURRENT_PROBE_PWM)
        {
            presentEvidenceCount_ = 0;
            absentEvidenceCount_ = 0;
            return;
        }

        if (measuredCurrent >= ACTUATOR_CURRENT_PRESENT_FLOOR)
        {
            absentEvidenceCount_ = 0;
            if (presentEvidenceCount_ < ACTUATOR_CURRENT_DEBOUNCE_SAMPLES)
                ++presentEvidenceCount_;
            if (presentEvidenceCount_ >= ACTUATOR_CURRENT_DEBOUNCE_SAMPLES)
                evidence_ = ActuatorCurrentEvidence::CurrentPresent;
            return;
        }

        presentEvidenceCount_ = 0;
        if (absentEvidenceCount_ < ACTUATOR_CURRENT_DEBOUNCE_SAMPLES)
            ++absentEvidenceCount_;
        if (absentEvidenceCount_ >= ACTUATOR_CURRENT_DEBOUNCE_SAMPLES)
            evidence_ = ActuatorCurrentEvidence::CurrentAbsent;
    }

private:
    ActuatorCurrentEvidence evidence_;
    uint8_t presentEvidenceCount_;
    uint8_t absentEvidenceCount_;
};
