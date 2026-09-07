#pragma once

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "display_frame_model.h"
#include "display_constants.h"

// Canvas-independent renderer shared by firmware, native tests, and the sim.
template <class Canvas>
class DisplayRenderer
{
public:
    explicit DisplayRenderer(Canvas &canvas)
        : canvas_(canvas), previousFrame_(), hasPreviousFrame_(false)
    {
    }

    // Returns whether any draw calls were made. The caller flushes the canvas.
    bool render(const DisplayFrame &frame)
    {
        const bool isFullRedraw = !hasPreviousFrame_;
        if (!isFullRedraw && displayFramesEqual(previousFrame_, frame))
            return false;

        canvas_.useStatusFont();
        if (isFullRedraw)
        {
            canvas_.erase();
            canvas_.line(0, SSD1306_HEADER_RULE_Y, 127, SSD1306_HEADER_RULE_Y);
        }

        if (isFullRedraw || frame.role != previousFrame_.role)
            drawRole(frame);
        if (isFullRedraw || frame.roll.value() != previousFrame_.roll.value() ||
            frame.pitch.value() != previousFrame_.pitch.value())
            drawTilt(frame);
        if (isFullRedraw ||
            frame.packVoltage.value() != previousFrame_.packVoltage.value())
            drawVolts(frame);

        for (int battery = 0; battery < 2; ++battery)
            if (isFullRedraw ||
                frame.batteryLevel[battery] !=
                    previousFrame_.batteryLevel[battery] ||
                frame.batteryDecivolts[battery] !=
                    previousFrame_.batteryDecivolts[battery])
                drawBattery(frame, battery);

        if (isFullRedraw || didControllerPresenceChange(frame))
            drawBody(frame);

        for (ActuatorId actuatorId = ActuatorId::FLHY;
             actuatorId < ActuatorId::ActuatorCount;
             ++actuatorId)
            if (isFullRedraw ||
                frame.actuators[actuatorId] !=
                    previousFrame_.actuators[actuatorId])
                drawActuator(frame, actuatorId);

        previousFrame_ = frame;
        hasPreviousFrame_ = true;
        return true;
    }

    void invalidate()
    {
        previousFrame_ = DisplayFrame{};
        hasPreviousFrame_ = false;
    }

private:

    bool didControllerPresenceChange(const DisplayFrame &frame) const
    {
        for (BoardRole role : ALL_BOARD_ROLES)
            if (frame.controllers[role] !=
                previousFrame_.controllers[role])
                return true;
        return false;
    }

    void clear(int x, int y, int width, int height)
    {
        canvas_.rectangleFill(x, y, width, height, SSD1306_COLOR_BLACK);
    }

    void drawTextField(int x, int chars, const char *text)
    {
        clear(x, SSD1306_HEADER_Y, chars * SSD1306_CHAR_WIDTH, SSD1306_TEXT_HEIGHT);
        canvas_.text(x, SSD1306_HEADER_Y, text);
    }

    void drawRole(const DisplayFrame &frame)
    {
        drawTextField(SSD1306_HEADER_ROLE_X, SSD1306_HEADER_ROLE_CHARS,
                      boardRoleLabel(frame.role));
    }

    void drawTilt(const DisplayFrame &frame)
    {
        char text[12];
        snprintf(text, sizeof(text), "%+03d/%+03d",
                 clampTilt(static_cast<int>(frame.roll.value())),
                 clampTilt(static_cast<int>(frame.pitch.value())));
        drawTextField(SSD1306_HEADER_TILT_X, SSD1306_HEADER_TILT_CHARS, text);
    }

    void drawVolts(const DisplayFrame &frame)
    {
        char text[16];
        const float volts = frame.packVoltage.value();
        if (!isfinite(volts) || volts < 0.0f || volts > 999.9f)
            snprintf(text, sizeof(text), "--.-V");
        else
        {
            const int decivolts = static_cast<int>(lround(volts * 10.0f));
            snprintf(text, sizeof(text), "%d.%dV", decivolts / 10, abs(decivolts % 10));
        }
        drawTextField(SSD1306_HEADER_VOLTS_X, SSD1306_HEADER_VOLTS_CHARS, text);
    }

    void drawBattery(const DisplayFrame &frame, int battery)
    {
        const int cellX = SSD1306_BATTERY_CELL_X + battery * SSD1306_BATTERY_CELL_PITCH;
        const int barX = cellX + SSD1306_BATTERY_BAR_DX;
        const int nubY = (SSD1306_BATTERY_HEIGHT - SSD1306_BATTERY_NUB_HEIGHT) / 2;

        // The 5x7 font blit can touch an eighth row.
        clear(cellX, SSD1306_BATTERY_Y, SSD1306_BATTERY_CELL_WIDTH,
              SSD1306_TEXT_HEIGHT);

        const char label[2] = {static_cast<char>('A' + battery), '\0'};
        canvas_.text(cellX, SSD1306_BATTERY_Y, label);

        canvas_.rectangle(barX, SSD1306_BATTERY_Y, SSD1306_BATTERY_WIDTH,
                          SSD1306_BATTERY_HEIGHT);
        canvas_.rectangleFill(barX + SSD1306_BATTERY_WIDTH, SSD1306_BATTERY_Y + nubY,
                              SSD1306_BATTERY_NUB_WIDTH, SSD1306_BATTERY_NUB_HEIGHT,
                              SSD1306_COLOR_WHITE);

        const float raw = frame.batteryLevel[battery];
        const float level = raw < 0.0f ? 0.0f : (raw > 1.0f ? 1.0f : raw);
        const int fillWidth =
            static_cast<int>(lround(SSD1306_BATTERY_FILL_WIDTH * level));
        if (fillWidth > 0)
            canvas_.rectangleFill(barX + 1, SSD1306_BATTERY_Y + 1, fillWidth,
                                  SSD1306_BATTERY_HEIGHT - 2, SSD1306_COLOR_WHITE);

        char voltage[16];
        const int decivolts = frame.batteryDecivolts[battery];
        if (decivolts == BATTERY_DECIVOLTS_NO_SIGNAL)
            snprintf(voltage, sizeof(voltage), "--.-V");
        else
            snprintf(voltage, sizeof(voltage), "%2d.%dV",
                     decivolts / 10, abs(decivolts % 10));
        canvas_.text(cellX + SSD1306_BATTERY_VALUE_DX, SSD1306_BATTERY_Y, voltage);
    }

    void drawBody(const DisplayFrame &frame)
    {
        clear(SSD1306_BODY_X, SSD1306_BODY_Y, SSD1306_BODY_WIDTH,
              SSD1306_FACE_BOTTOM_Y - SSD1306_BODY_Y + 1);
        canvas_.rectangle(SSD1306_BODY_X, SSD1306_BODY_Y, SSD1306_BODY_WIDTH,
                          SSD1306_BODY_HEIGHT);
        if (frame.controllers[ROLE_LEFT])
            canvas_.rectangleFill(SSD1306_BODY_X + 1, SSD1306_BODY_Y + 1,
                                  SSD1306_STEM_X - SSD1306_BODY_X - 1,
                                  SSD1306_T_BAR_Y - SSD1306_BODY_Y - 1,
                                  SSD1306_COLOR_WHITE);
        if (frame.controllers[ROLE_RIGHT])
            canvas_.rectangleFill(SSD1306_STEM_X, SSD1306_BODY_Y + 1,
                                  SSD1306_BODY_X + SSD1306_BODY_WIDTH - 1 - SSD1306_STEM_X,
                                  SSD1306_T_BAR_Y - SSD1306_BODY_Y - 1,
                                  SSD1306_COLOR_WHITE);
        if (frame.controllers[ROLE_FRONT])
            canvas_.rectangleFill(SSD1306_BODY_X + 1, SSD1306_T_BAR_Y,
                                  SSD1306_BODY_WIDTH - 2,
                                  SSD1306_BODY_Y + SSD1306_BODY_HEIGHT - 1 - SSD1306_T_BAR_Y,
                                  SSD1306_COLOR_WHITE);
        drawFace();
    }

    void drawFace()
    {
        const int centerX = SSD1306_FACE_CENTER_X;
        const int top = SSD1306_FACE_TOP_Y;
        const int eyeY = SSD1306_FACE_EYE_Y;
        for (int side = -1; side <= 1; side += 2)
        {
            const int eyeX = centerX + side * SSD1306_FACE_EYE_OFFSET_X;
            canvas_.line(eyeX, top + 1, eyeX, top + SSD1306_FACE_STALK_HEIGHT);
            // SparkFun omits rectangle sides below four pixels tall.
            canvas_.line(eyeX - 1, eyeY, eyeX + 1, eyeY);
            canvas_.line(eyeX - 1, eyeY + 2, eyeX + 1, eyeY + 2);
            canvas_.line(eyeX - 1, eyeY, eyeX - 1, eyeY + 2);
            canvas_.line(eyeX + 1, eyeY, eyeX + 1, eyeY + 2);
        }
        canvas_.pixel(centerX - 2, top + 4);
        canvas_.pixel(centerX + 2, top + 4);
        canvas_.line(centerX - 1, SSD1306_FACE_BOTTOM_Y, centerX + 1,
                     SSD1306_FACE_BOTTOM_Y);
    }

    void drawActuator(const DisplayFrame &frame, ActuatorId actuatorId)
    {
        const int radius = SSD1306_GLYPH_SIZE / 2;
        const int centerX = ssd1306ActuatorX(actuatorId);
        const int y = ssd1306ActuatorY(actuatorId);
        const int direction = centerX < SSD1306_BODY_X ? -1 : 1;
        const int edge = direction < 0
            ? SSD1306_BODY_X
            : SSD1306_BODY_X + SSD1306_BODY_WIDTH;
        // Clear only as far as the inboard joint.
        const int inner = abs(centerX - edge) == SSD1306_LEG_FIRST_OFFSET
            ? edge
            : centerX - direction * (SSD1306_LEG_JOINT_PITCH - radius);

        const int outer = centerX + direction * radius;
        const int left = direction < 0 ? outer : inner + 1;
        const int right = direction < 0 ? inner - 1 : outer;
        clear(left, y - radius, right - left + 1, SSD1306_GLYPH_SIZE);

        canvas_.line(inner, y, centerX - direction * radius, y);
        drawGlyph(centerX, y, frame.actuators[actuatorId]);
    }

    void drawGlyph(
        int centerX,
        int centerY,
        ActuatorGlyph glyph)
    {
        const int radius = SSD1306_GLYPH_SIZE / 2;
        const int triangleRadius = radius - 1;
        if (glyph == ActuatorGlyph::Extend ||
            glyph == ActuatorGlyph::Retract)
        {
            // Extend points down; retract points up.
            const int diameter = 2 * triangleRadius;
            for (int row = 0; row <= diameter; ++row)
            {
                const int widthRow =
                    glyph == ActuatorGlyph::Extend
                    ? diameter - row
                    : row;
                const int halfWidth = widthRow * triangleRadius / diameter;
                const int y = centerY - triangleRadius + row;
                canvas_.line(centerX - halfWidth, y, centerX + halfWidth, y);
            }
            return;
        }

        if (glyph == ActuatorGlyph::Hold || glyph == ActuatorGlyph::Unverified)
        {
            const int inner = glyph == ActuatorGlyph::Hold
                ? 0
                : (radius - 1) * (radius - 1);
            for (int dy = -radius; dy <= radius; ++dy)
                for (int dx = -radius; dx <= radius; ++dx)
                {
                    const int distance = dx * dx + dy * dy;
                    if (distance <= radius * radius && distance >= inner)
                        canvas_.pixel(centerX + dx, centerY + dy);
                }
            return;
        }

        canvas_.line(centerX - triangleRadius, centerY - triangleRadius,
                     centerX + triangleRadius, centerY + triangleRadius);
        canvas_.line(centerX - triangleRadius, centerY + triangleRadius,
                     centerX + triangleRadius, centerY - triangleRadius);
    }

    static int clampTilt(int value)
    {
        return value < -99 ? -99 : (value > 99 ? 99 : value);
    }

    Canvas &canvas_;
    DisplayFrame previousFrame_;
    bool hasPreviousFrame_;
};
