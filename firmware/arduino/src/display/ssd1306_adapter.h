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
#include "display_renderer.h"

// DisplayRenderer canvas backed by the SparkFun driver.
class Ssd1306Canvas
{
public:
    static_assert(COLOR_BLACK == SSD1306_COLOR_BLACK &&
                      COLOR_WHITE == SSD1306_COLOR_WHITE,
                  "SparkFun's colour constants no longer match the renderer's");

    bool begin() { return driver_.begin(); }
    bool reset() { return driver_.reset(true); }
    void display() { driver_.display(); }

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

    Qwiic1in3OLED driver_;
};

class Ssd1306Adapter
{
public:
    Ssd1306Adapter()
        : canvas_(), renderer_(canvas_), recoveryPolicy_{}, stuckBusLatch_{},
          isInitialized_(false)
    {
    }

    bool isInitialized() const { return isInitialized_; }

    bool initialize()
    {
        recoveryPolicy_ = I2cRecoveryPolicy{};
        stuckBusLatch_ = I2cStuckBusLatch{};
        return configure();
    }

    // Flush dirty pages at the display bus rate.
    bool render(const DisplayFrame &frame)
    {
        const I2cRecoveryLimits limits = {
            SSD1306_BAD_TICKS_BEFORE_RECOVERY,
            SSD1306_RECOVERY_RETRY_INTERVAL_MS,
        };

        if (!isInitialized_)
        {
            if (!recoveryPolicy_.noteFailure(millis(), limits) ||
                !recoverAndConfigure())
            {
                return false;
            }
            recoveryPolicy_.noteSuccess();
        }

        if (!responds())
        {
            isInitialized_ = false;
            recoveryPolicy_.noteFailure(millis(), limits);
            return false;
        }
        recoveryPolicy_.noteSuccess();

        if (!renderer_.render(frame))
            return false;

        Wire.setClock(SSD1306_TRANSFER_BUS_CLOCK_HZ);
        canvas_.display();
        Wire.setClock(I2C_DEFAULT_BUS_CLOCK_HZ);
        return true;
    }

    void invalidate() { renderer_.invalidate(); }

private:
    bool configure()
    {
        isInitialized_ = canvas_.begin();
        if (isInitialized_)
            canvas_.useStatusFont();
        invalidate();
        return isInitialized_;
    }

    bool responds()
    {
        Wire.clearWireTimeoutFlag();
        Wire.beginTransmission(SSD1306_I2C_ADDRESS);
        return Wire.endTransmission() == 0;
    }

    bool recoverAndConfigure()
    {
        if (responds())
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
        return result != I2cBusRecovery::Stuck && responds() && resetPanel();
    }

    bool resetPanel()
    {
        if (!canvas_.reset())
            return false;
        isInitialized_ = true;
        canvas_.useStatusFont();
        invalidate();
        return true;
    }

    Ssd1306Canvas canvas_;
    DisplayRenderer<Ssd1306Canvas> renderer_;
    I2cRecoveryPolicy recoveryPolicy_;
    I2cStuckBusLatch stuckBusLatch_;
    bool isInitialized_;
};
