#pragma once

// Canvas for DisplayRenderer on host: raster + Wire GDDRAM flush, not SparkFun driver.
// Intentional; see "Host OLED (no Arduino sensor libraries)" in I2C_SIM_SCOPE.md.

#include "src/display/display_constants.h"
#include "src/display/display_frame_model.h"

#include "Wire.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

// DisplayRenderer canvas: rasterizes primitives, flushes GDDRAM over sim Wire (0x40 stream).
class HostOledCanvas
{
public:
    bool isInitialized() const { return initialized_; }

    bool initialize()
    {
        Wire.begin();
        Wire.beginTransmission(SSD1306_I2C_ADDRESS);
        initialized_ = Wire.endTransmission() == 0;
        return initialized_;
    }

    bool recover() { return initialize(); }

    bool isResponding()
    {
        Wire.beginTransmission(SSD1306_I2C_ADDRESS);
        return Wire.endTransmission() == 0;
    }

    void useStatusFont() {}

    void erase()
    {
        std::memset(framebuffer_, 0, sizeof(framebuffer_));
    }

    void pixel(int x, int y) { setPixel(x, y, true); }

    void line(int x0, int y0, int x1, int y1) { drawLine(x0, y0, x1, y1); }

    void rectangle(int x, int y, int width, int height)
    {
        drawLine(x, y, x + width - 1, y);
        drawLine(x, y + height - 1, x + width - 1, y + height - 1);
        drawLine(x, y, x, y + height - 1);
        drawLine(x + width - 1, y, x + width - 1, y + height - 1);
    }

    void rectangleFill(int x, int y, int width, int height, int color)
    {
        for (int row = y; row < y + height; ++row)
        {
            for (int col = x; col < x + width; ++col)
            {
                setPixel(col, row, color != SSD1306_COLOR_BLACK);
            }
        }
    }

    void text(int /*x*/, int /*y*/, const char * /*value*/) {}

    void display();

    ActuatorGlyph lastGlyphForActuator(ActuatorId actuatorId) const
    {
        return lastGlyphs_[static_cast<size_t>(actuatorId)];
    }

    void setLastGlyphForActuator(ActuatorId actuatorId, ActuatorGlyph glyph)
    {
        lastGlyphs_[static_cast<size_t>(actuatorId)] = glyph;
    }

private:
    void setPixel(int x, int y, bool on);
    void drawLine(int x0, int y0, int x1, int y1);

    bool initialized_ = false;
    uint8_t framebuffer_[128 * 8];
    ActuatorGlyph lastGlyphs_[ActuatorId::ActuatorCount]{};
};
