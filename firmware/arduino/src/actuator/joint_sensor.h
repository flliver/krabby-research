#pragma once

#include <stdint.h>

#include "calibration_constants.h"

// Per-joint position sensor kind and the pure classifiers that tell a hall
// encoder from a pot during calibration moves. On quadrature builds the analog
// pins A0-A5 carry hall B, so a pot reading is only evidence when read at rest.

// Stored as 2 bits per joint in EEPROM; 3 reads as Unknown.
enum class JointSensor : uint8_t { Unknown = 0, Pot = 1, Hall = 2 };

// What one move (or nudge) showed about the joint's sensor.
enum class MoveFeedback : uint8_t { None, Hall, HallWrongWay, Pot, PotWrongWay };

// Edges counted in the driven direction. countDelta is the count now minus the
// count at move start (motor at rest then); pwmSign is +1 for +PWM.
inline int32_t hallDriveEdges(int8_t pwmSign, int32_t countDelta)
{
    return static_cast<int32_t>(pwmSign) * ACTUATOR_HALL_EXTEND_COUNT_SIGN * countDelta;
}

// Hall evidence while driven. A pot never makes quadrature edges (its A pin floats
// high and the ISR fires only on A), so enough edges in the driven direction
// mean an encoder whatever kind is stored.
inline MoveFeedback classifyHallMotion(bool quadrature, int8_t pwmSign, int32_t countDelta)
{
    if (!quadrature)
        return MoveFeedback::None;  // Rev 1 counts carry no direction
    const int32_t driveEdges = hallDriveEdges(pwmSign, countDelta);
    if (driveEdges >= CAL_HALL_DETECT_EDGES)
        return MoveFeedback::Hall;
    if (driveEdges <= -CAL_HALL_DETECT_EDGES)
        return MoveFeedback::HallWrongWay;
    return MoveFeedback::None;
}

inline bool isPotRestInBand(int snap)
{
    return snap >= CAL_POT_DETECT_MIN_RAW && snap <= CAL_POT_DETECT_MAX_RAW;
}

// Pot evidence from two at-rest snapshots. requiredInBand: 0 = non-quadrature
// build; 1 = quadrature, joint already Pot; 2 = quadrature, joint Unknown or Hall
// (classification).
inline MoveFeedback classifyPotAtRest(
    uint8_t requiredInBand,
    int8_t pwmSign,
    int32_t countDelta,
    int startSnap,
    int endSnap)
{
    if (requiredInBand > 0)
    {
        if (countDelta > CAL_POT_DETECT_MAX_EDGES || countDelta < -CAL_POT_DETECT_MAX_EDGES)
            return MoveFeedback::None;  // edges: not a pot
        if (static_cast<uint8_t>(isPotRestInBand(startSnap) + isPotRestInBand(endSnap)) <
            requiredInBand)
            return MoveFeedback::None;  // rail or rail flip: hall B
    }
    const int travel = pwmSign * (endSnap - startSnap);  // pot convention: +PWM reads higher
    if (travel >= CAL_MIN_TRAVEL_COUNTS)
        return MoveFeedback::Pot;
    if (travel <= -CAL_MIN_TRAVEL_COUNTS)
        return MoveFeedback::PotWrongWay;
    return MoveFeedback::None;
}
