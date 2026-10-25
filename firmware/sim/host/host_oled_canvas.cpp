#include "host_oled_canvas.hpp"

#include "src/display/display_frame_model.h"

namespace {

void clampCoord(int &value, int minValue, int maxValue)
{
    if (value < minValue) value = minValue;
    if (value > maxValue) value = maxValue;
}

}  // namespace

void HostOledCanvas::setPixel(int x, int y, bool on)
{
    clampCoord(x, 0, 127);
    clampCoord(y, 0, 63);
    const std::size_t page = static_cast<std::size_t>(y / 8);
    const uint8_t bit = static_cast<uint8_t>(1u << (y & 7));
    const std::size_t index = page * 128 + static_cast<std::size_t>(x);
    if (on)
    {
        framebuffer_[index] |= bit;
    }
    else
    {
        framebuffer_[index] &= static_cast<uint8_t>(~bit);
    }
}

void HostOledCanvas::drawLine(int x0, int y0, int x1, int y1)
{
    int dx = x1 - x0;
    int dy = y1 - y0;
    if (dx < 0) dx = -dx;
    if (dy < 0) dy = -dy;
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = dx - dy;
    while (true)
    {
        setPixel(x0, y0, true);
        if (x0 == x1 && y0 == y1) break;
        const int e2 = 2 * err;
        if (e2 > -dy)
        {
            err -= dy;
            x0 += sx;
        }
        if (e2 < dx)
        {
            err += dx;
            y0 += sy;
        }
    }
}

void HostOledCanvas::display()
{
    Wire.beginTransmission(SSD1306_I2C_ADDRESS);
    Wire.write(0x40);
    Wire.write(framebuffer_, sizeof(framebuffer_));
    Wire.endTransmission(true);
}
