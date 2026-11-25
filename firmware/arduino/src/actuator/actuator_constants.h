#pragma once

#include <stdint.h>

static constexpr int ACTUATOR_PWM_MAXIMUM_MAGNITUDE = 255;

static constexpr unsigned int POT_PROBE_SETTLE_US = 200;
static constexpr unsigned int POT_PROBE_SAMPLE_COUNT = 4;

// Hall encoder joints (quadrature builds). The signed count rises on retract and
// falls on extend, as seen on every hip and knee (FLHL, FLKL, FRKL, RLHL). The
// FRHY yaw counts the other way, but yaws are never driven. The position frame
// is x = SIGN * (count - hallHome), so a higher value means more extended, like
// a pot, and +PWM raises it.
static constexpr int ACTUATOR_HALL_EXTEND_COUNT_SIGN = -1;
// The frame is clamped to +-this before it is narrowed to int16.
static constexpr int ACTUATOR_HALL_POSITION_LIMIT = 30000;

// Encoder T control. PROVISIONAL: set from an on-robot stroke measurement and a
// low-amplitude tuning run with the user present. Edges are hall edges.
static constexpr float ACTUATOR_ENC_KP = 1.0f;
static constexpr int ACTUATOR_ENC_ERR_DEADBAND_EDGES = 12;
static constexpr int ACTUATOR_ENC_MAX_PWM = 150;
// T targets stay inside the ends calibration recorded. In the hall frame the
// minStop side is always a retracted end, the actuator's limit switch (hip curl
// end, knee uncurl end; recorded exactly where the count stopped, and an
// overshoot only opens the switch): ACTUATOR_ENC_SWITCH_INSET_EDGES inside it.
// The maxStop side is the knee's compliant leg stop (recorded 0-5 edges
// shallower than a hard push ends; 150 pushes 10-20 edges deeper than 80,
// 2026-11-24): ACTUATOR_ENC_STOP_INSET_EDGES inside it; or a hip's extend
// switch: ACTUATOR_ENC_HIP_EXTEND_INSET_EDGES inside it.
static constexpr int ACTUATOR_ENC_SWITCH_INSET_EDGES = 8;
static constexpr int ACTUATOR_ENC_STOP_INSET_EDGES = 25;
// A hall hip's maxStop is its extend switch, derived as home +
// CAL_HIP_STROKE_EDGES (not switch-verified): T stays this far inside it, so
// the switch remains a hardware backstop (fix round 2026-11-24: "soft limit
// about 10 counts short").
static constexpr int ACTUATOR_ENC_HIP_EXTEND_INSET_EDGES = 10;
// A homed encoder reading this far outside its stroke clears the home.
static constexpr int ACTUATOR_ENC_RANGE_MARGIN_EDGES = 60;

// T stall guard (pot and encoder): a joint driven at this PWM or more that
// moves less than the progress over the window is stopped. Edges or counts.
// Windows are wrapping 16-bit milliseconds.
static constexpr int ACTUATOR_T_STALL_MIN_PWM = 60;
static constexpr int ACTUATOR_T_STALL_MIN_PROGRESS = 3;
static constexpr uint16_t ACTUATOR_T_STALL_WINDOW_MS = 1000;
// Below ACTUATOR_T_STALL_MIN_PWM but above the drive deadband, a joint that
// moves less than the progress over this window is stopped (no ERR line, the
// home is kept: at low PWM no edges are lost).
static constexpr uint16_t ACTUATOR_T_STALL_SLOW_WINDOW_MS = 3000;
// Minimum interval between "ERR T refused" lines, per board.
static constexpr uint16_t ACTUATOR_T_REFUSED_LOG_INTERVAL_MS = 1000;
// SENS line refresh period when no joint state changed.
static constexpr uint16_t ACTUATOR_SENS_PERIOD_MS = 2000;

// LinearActuator::encFlags bits.
// Home proven this boot (a curl end was accepted).
static constexpr uint8_t ACTUATOR_FLAG_HOMED = 0x01;
// Pot limits measured, or the hall stroke known.
static constexpr uint8_t ACTUATOR_FLAG_LIMITS_VALID = 0x02;
// A home was cleared by a guard or a failed move.
static constexpr uint8_t ACTUATOR_FLAG_HOME_LOST = 0x04;
// An ERR line is pending.
static constexpr uint8_t ACTUATOR_FLAG_ERR_STALL = 0x08;
static constexpr uint8_t ACTUATOR_FLAG_ERR_HOME = 0x10;

static_assert(ACTUATOR_HALL_EXTEND_COUNT_SIGN * ACTUATOR_HALL_EXTEND_COUNT_SIGN == 1,
              "the hall count sign must be +1 or -1");
static_assert(ACTUATOR_HALL_POSITION_LIMIT > 0 && ACTUATOR_HALL_POSITION_LIMIT < 32767,
              "the hall frame must fit int16");
static_assert(ACTUATOR_ENC_MAX_PWM > 0 && ACTUATOR_ENC_MAX_PWM <= ACTUATOR_PWM_MAXIMUM_MAGNITUDE,
              "encoder PWM cap is outside the actuator PWM range");
static_assert(ACTUATOR_T_STALL_MIN_PWM > 0 && ACTUATOR_T_STALL_MIN_PWM <= ACTUATOR_ENC_MAX_PWM,
              "the stall guard must be reachable by the encoder PWM cap");
static_assert(ACTUATOR_T_STALL_SLOW_WINDOW_MS > ACTUATOR_T_STALL_WINDOW_MS,
              "the low-PWM stall window must be longer than the stall window");
