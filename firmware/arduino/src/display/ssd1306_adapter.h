#pragma once

#include <Arduino.h>
#include <SparkFun_Qwiic_OLED.h>
#include <Wire.h>
#include <res/qw_fnt_5x7.h>

#include "../i2c/arduino_i2c_bus.h"
#include "../i2c/i2c_recovery.h"
#include "../imu/imu_constants.h"
#include "display_frame_model.h"
#include "display_constants.h"

// Three consecutive failed ticks before a recovery attempt, then at most one a second.
static constexpr I2cRecoveryLimits SSD1306_RECOVERY_LIMITS = {3, 1000UL};

// Owns the SparkFun driver and is the DisplayRenderer's canvas.
class Ssd1306Adapter
{
public:
    static_assert(COLOR_BLACK == SSD1306_COLOR_BLACK &&
                      COLOR_WHITE == SSD1306_COLOR_WHITE,
                  "SparkFun's colour constants no longer match the renderer's");

    Ssd1306Adapter()
        : driver_(), recoveryPolicy_{}, stuckBusLatch_{},
          isInitialized_(false)
    {
    }

    bool isInitialized() const { return isInitialized_; }

    bool initialize()
    {
        recoveryPolicy_ = I2cRecoveryPolicy{};
        stuckBusLatch_ = I2cStuckBusLatch{};
        isInitialized_ = driver_.begin();
        return isInitialized_;
    }

    // Counts a failed tick and, once the policy allows, recovers the bus and resets the
    // panel. True only when the panel was reset, which clears it.
    bool recover()
    {
        return recoveryPolicy_.shouldAttemptRecovery(millis(), SSD1306_RECOVERY_LIMITS) && recoverAndConfigure();
    }

    // Probes the panel; one that does not answer must be recovered before drawing again.
    bool isResponding()
    {
        if (!isProbeAcknowledged())
        {
            isInitialized_ = false;
            recoveryPolicy_.shouldAttemptRecovery(millis(), SSD1306_RECOVERY_LIMITS);
            return false;
        }
        recoveryPolicy_.noteSuccess();
        return true;
    }

    // Flushes dirty pages. A full frame blocks the loop for ~115 ms at the default rate
    // but ~29 ms at fast mode, so the flush runs fast and the rest of the bus stays default.
    void display()
    {
        Wire.setClock(SSD1306_TRANSFER_BUS_CLOCK_HZ);
        driver_.display();
        Wire.setClock(I2C_DEFAULT_BUS_CLOCK_HZ);
    }

    // DisplayRenderer canvas: drawing changes only the driver's buffer until display().
    void useStatusFont() { driver_.setFont(QW_FONT_5X7); }
    void erase() { driver_.erase(); }

    void pixel(int x, int y) { driver_.pixel(narrow(x), narrow(y)); }

    void line(int x0, int y0, int x1, int y1)
    {
        driver_.line(narrow(x0), narrow(y0), narrow(x1), narrow(y1));
    }

    void rectangle(int x, int y, int width, int height)
    {
        driver_.rectangle(narrow(x), narrow(y), narrow(width), narrow(height));
    }

    void rectangleFill(int x, int y, int width, int height, int color)
    {
        driver_.rectangleFill(narrow(x), narrow(y), narrow(width), narrow(height),
                              static_cast<uint8_t>(color));
    }

    void text(int x, int y, const char *value)
    {
        driver_.text(narrow(x), narrow(y), value);
    }

private:
    // Match the driver's uint8_t coordinate conversion.
    static uint8_t narrow(int value) { return static_cast<uint8_t>(value); }

    bool isProbeAcknowledged()
    {
        Wire.clearWireTimeoutFlag();
        Wire.beginTransmission(SSD1306_I2C_ADDRESS);
        return Wire.endTransmission() == 0;
    }

    bool recoverAndConfigure()
    {
        if (isProbeAcknowledged())
            return resetPanel();

        if (!Wire.getWireTimeoutFlag())
            return false;

        Wire.end();
        ArduinoI2cBus bus(
            I2C_DEFAULT_BUS_CLOCK_HZ,
            I2C_BUS_TIMEOUT_MICROSECONDS);
        if (!stuckBusLatch_.mayAttempt(bus.isSdaHigh()))
            return false;

        const I2cBusRecovery result = recoverI2cBus(bus);
        stuckBusLatch_.noteResult(result);
        if (result == I2cBusRecovery::Stuck)
            return false;
        if (!isProbeAcknowledged())
            return false;
        return resetPanel();
    }

    bool resetPanel()
    {
        if (!driver_.reset(true))
            return false;
        isInitialized_ = true;
        return true;
    }

    Qwiic1in3OLED driver_;
    I2cRecoveryPolicy recoveryPolicy_;
    I2cStuckBusLatch stuckBusLatch_;
    bool isInitialized_;
};
