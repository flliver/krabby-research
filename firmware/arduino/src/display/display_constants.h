#pragma once

#include <stddef.h>
#include <stdint.h>

#include "../actuator/actuator_identity.h"

static constexpr uint32_t CONTROLLER_DISPLAY_TIMEOUT_MILLISECONDS = 500;
// A full frame takes ~29 ms at 400 kHz and ~115 ms at 100 kHz.
static constexpr uint32_t SSD1306_TRANSFER_BUS_CLOCK_HZ = 400000UL;
static constexpr uint8_t SSD1306_I2C_ADDRESS = 0x3D;
static constexpr uint8_t SSD1306_BAD_TICKS_BEFORE_RECOVERY = 3;
static constexpr uint32_t SSD1306_RECOVERY_RETRY_INTERVAL_MS = 1000UL;

// Keep the renderer independent of SparkFun headers.
static constexpr int SSD1306_COLOR_BLACK = 0;
static constexpr int SSD1306_COLOR_WHITE = 1;

// Separate header fields allow partial redraws. The 5x7 font advances 6 px.
static constexpr int SSD1306_CHAR_WIDTH = 6;
static constexpr int SSD1306_TEXT_HEIGHT = 8;
static constexpr int SSD1306_HEADER_Y = 0;
static constexpr int SSD1306_HEADER_RULE_Y = 9;
static constexpr int SSD1306_HEADER_ROLE_X = 0;
static constexpr int SSD1306_HEADER_ROLE_CHARS = 5;   // "UNKWN"
static constexpr int SSD1306_HEADER_TILT_X = 36;
static constexpr int SSD1306_HEADER_TILT_CHARS = 7;   // "+00/+00"
static constexpr int SSD1306_HEADER_VOLTS_X = 92;
static constexpr int SSD1306_HEADER_VOLTS_CHARS = 6;  // "24.8V"

static_assert(
    SSD1306_HEADER_ROLE_X + SSD1306_HEADER_ROLE_CHARS * SSD1306_CHAR_WIDTH <=
        SSD1306_HEADER_TILT_X,
    "role field overlaps the tilt field");
static_assert(
    SSD1306_HEADER_TILT_X + SSD1306_HEADER_TILT_CHARS * SSD1306_CHAR_WIDTH <=
        SSD1306_HEADER_VOLTS_X,
    "tilt field overlaps the voltage field");
static_assert(
    SSD1306_HEADER_VOLTS_X + SSD1306_HEADER_VOLTS_CHARS * SSD1306_CHAR_WIDTH <= 128,
    "voltage field runs off the panel");

static constexpr int SSD1306_GLYPH_SIZE = 9;
// One changed band takes ~5.8 ms versus ~120 ms for a full frame.
static constexpr int SSD1306_BAND_HEIGHT = SSD1306_GLYPH_SIZE + 1;
static constexpr int SSD1306_BODY_WIDTH = 32;
static constexpr int SSD1306_BODY_HEIGHT = SSD1306_BAND_HEIGHT * 3 + 1;
static constexpr int SSD1306_BODY_X = (128 - SSD1306_BODY_WIDTH) / 2;
static constexpr int SSD1306_BODY_Y = 22;
static constexpr int SSD1306_T_BAR_Y =
    SSD1306_BODY_Y + 2 * SSD1306_BAND_HEIGHT;
static constexpr int SSD1306_STEM_X =
    SSD1306_BODY_X + SSD1306_BODY_WIDTH / 2;
static constexpr int SSD1306_LEG_FIRST_OFFSET = 7;
static constexpr int SSD1306_LEG_JOINT_PITCH = 11;
static constexpr int SSD1306_OUTERMOST_ACTUATOR_OFFSET =
    SSD1306_LEG_FIRST_OFFSET + 2 * SSD1306_LEG_JOINT_PITCH;

inline int ssd1306ActuatorX(ActuatorId actuatorId)
{
    switch (actuatorId)
    {
        case ActuatorId::FLHY:
        case ActuatorId::MLHY:
        case ActuatorId::RLHY:
            return SSD1306_BODY_X - SSD1306_LEG_FIRST_OFFSET;
        case ActuatorId::FLHL:
        case ActuatorId::MLHL:
        case ActuatorId::RLHL:
            return SSD1306_BODY_X -
                (SSD1306_LEG_FIRST_OFFSET + SSD1306_LEG_JOINT_PITCH);
        case ActuatorId::FLKL:
        case ActuatorId::MLKL:
        case ActuatorId::RLKL:
            return SSD1306_BODY_X - SSD1306_OUTERMOST_ACTUATOR_OFFSET;
        case ActuatorId::FRHY:
        case ActuatorId::MRHY:
        case ActuatorId::RRHY:
            return SSD1306_BODY_X + SSD1306_BODY_WIDTH +
                SSD1306_LEG_FIRST_OFFSET;
        case ActuatorId::FRHL:
        case ActuatorId::MRHL:
        case ActuatorId::RRHL:
            return SSD1306_BODY_X + SSD1306_BODY_WIDTH +
                SSD1306_LEG_FIRST_OFFSET + SSD1306_LEG_JOINT_PITCH;
        case ActuatorId::FRKL:
        case ActuatorId::MRKL:
        case ActuatorId::RRKL:
            return SSD1306_BODY_X + SSD1306_BODY_WIDTH +
                SSD1306_OUTERMOST_ACTUATOR_OFFSET;
        default:
            return 0;
    }
}

inline int ssd1306ActuatorY(ActuatorId actuatorId)
{
    switch (actuatorId)
    {
        case ActuatorId::RLHY:
        case ActuatorId::RLHL:
        case ActuatorId::RLKL:
        case ActuatorId::RRHY:
        case ActuatorId::RRHL:
        case ActuatorId::RRKL:
            return SSD1306_BODY_Y + SSD1306_BAND_HEIGHT / 2;
        case ActuatorId::MLHY:
        case ActuatorId::MLHL:
        case ActuatorId::MLKL:
        case ActuatorId::MRHY:
        case ActuatorId::MRHL:
        case ActuatorId::MRKL:
            return SSD1306_BODY_Y + SSD1306_BAND_HEIGHT +
                SSD1306_BAND_HEIGHT / 2;
        case ActuatorId::FLHY:
        case ActuatorId::FLHL:
        case ActuatorId::FLKL:
        case ActuatorId::FRHY:
        case ActuatorId::FRHL:
        case ActuatorId::FRKL:
            return SSD1306_BODY_Y + 2 * SSD1306_BAND_HEIGHT +
                SSD1306_BAND_HEIGHT / 2;
        default:
            return 0;
    }
}

static_assert(
    SSD1306_LEG_JOINT_PITCH > SSD1306_GLYPH_SIZE,
    "leg joints are pitched closer than a glyph is wide, so they would overlap");
static_assert(
    SSD1306_BODY_X - (SSD1306_OUTERMOST_ACTUATOR_OFFSET +
                      SSD1306_GLYPH_SIZE / 2) >= 0,
    "outermost leg joint runs off the left of the panel");
static_assert(
    SSD1306_BODY_X + SSD1306_BODY_WIDTH +
        (SSD1306_OUTERMOST_ACTUATOR_OFFSET +
         SSD1306_GLYPH_SIZE / 2) < 128,
    "outermost leg joint runs off the right of the panel");

static constexpr int SSD1306_FACE_CENTER_X =
    SSD1306_BODY_X + SSD1306_BODY_WIDTH / 2;
static constexpr int SSD1306_FACE_TOP_Y =
    SSD1306_BODY_Y + SSD1306_BODY_HEIGHT - 1;
static constexpr int SSD1306_FACE_EYE_OFFSET_X = 6;
static constexpr int SSD1306_FACE_STALK_HEIGHT = 2;
static constexpr int SSD1306_FACE_EYE_Y =
    SSD1306_FACE_TOP_Y + SSD1306_FACE_STALK_HEIGHT + 1;
static constexpr int SSD1306_FACE_BOTTOM_Y = SSD1306_FACE_TOP_Y + 5;

static_assert(SSD1306_FACE_BOTTOM_Y < 64, "krab face runs off the bottom of the panel");
static_assert(
    SSD1306_FACE_CENTER_X + SSD1306_FACE_EYE_OFFSET_X + 1 <
        SSD1306_BODY_X + SSD1306_BODY_WIDTH,
    "krab face is wider than the body it sits on");

static constexpr int SSD1306_BATTERY_WIDTH = 18;
static constexpr int SSD1306_BATTERY_HEIGHT = 7;
static constexpr int SSD1306_BATTERY_Y = 11;
static constexpr int SSD1306_BATTERY_NUB_WIDTH = 2;
static constexpr int SSD1306_BATTERY_NUB_HEIGHT = 3;
static constexpr int SSD1306_BATTERY_FILL_WIDTH = SSD1306_BATTERY_WIDTH - 2;

static constexpr int SSD1306_BATTERY_LABEL_CHARS = 1;    // "A", "B"
static constexpr int SSD1306_BATTERY_VALUE_CHARS = 5;  // "13.3V"
static constexpr int SSD1306_BATTERY_VALUE_GAP = 2;

static constexpr int SSD1306_BATTERY_BAR_DX =
    SSD1306_BATTERY_LABEL_CHARS * SSD1306_CHAR_WIDTH;
static constexpr int SSD1306_BATTERY_VALUE_DX =
    SSD1306_BATTERY_BAR_DX + SSD1306_BATTERY_WIDTH + SSD1306_BATTERY_NUB_WIDTH +
    SSD1306_BATTERY_VALUE_GAP;
static constexpr int SSD1306_BATTERY_CELL_WIDTH =
    SSD1306_BATTERY_VALUE_DX +
    SSD1306_BATTERY_VALUE_CHARS * SSD1306_CHAR_WIDTH;
static constexpr int SSD1306_BATTERY_CELL_X = 6;
static constexpr int SSD1306_BATTERY_CELL_PITCH = 64;

static_assert(
    SSD1306_BATTERY_CELL_WIDTH <= SSD1306_BATTERY_CELL_PITCH,
    "the two battery gauges overlap");
static_assert(
    SSD1306_BATTERY_CELL_X + SSD1306_BATTERY_CELL_PITCH +
            SSD1306_BATTERY_CELL_WIDTH <= 128,
    "the second battery gauge runs off the panel");
static_assert(
    SSD1306_BATTERY_Y > SSD1306_HEADER_RULE_Y,
    "the battery gauges overwrite the header rule");
static_assert(
    SSD1306_BATTERY_Y + SSD1306_TEXT_HEIGHT <= SSD1306_BODY_Y,
    "the battery gauge row runs into the body");
