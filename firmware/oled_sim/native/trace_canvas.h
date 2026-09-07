#pragma once

#include <stdio.h>
#include <stdint.h>

#include "src/display/display_constants.h"

// DisplayRenderer canvas that emits replayable draw calls.
class TraceCanvas
{
public:
    explicit TraceCanvas(FILE *out) : out_(out) {}

    void useStatusFont() { fprintf(out_, "font 5x7\n"); }
    void erase() { fprintf(out_, "erase\n"); }

    void pixel(int x, int y)
    {
        fprintf(out_, "pixel %u %u\n", narrow(x), narrow(y));
    }

    void line(int x0, int y0, int x1, int y1)
    {
        fprintf(out_, "line %u %u %u %u\n", narrow(x0), narrow(y0), narrow(x1),
                narrow(y1));
    }

    void rectangle(int x, int y, int width, int height)
    {
        fprintf(out_, "rect %u %u %u %u\n", narrow(x), narrow(y), narrow(width),
                narrow(height));
    }

    void rectangleFill(int x, int y, int width, int height, int color)
    {
        fprintf(out_, "fill %u %u %u %u %u\n", narrow(x), narrow(y), narrow(width),
                narrow(height), narrow(color));
    }

    // The string runs to end of line, so it needs no quoting and may hold spaces.
    void text(int x, int y, const char *value)
    {
        fprintf(out_, "text %u %u %s\n", narrow(x), narrow(y), value);
    }

private:
    static unsigned narrow(int value) { return static_cast<uint8_t>(value); }

    FILE *out_;
};
