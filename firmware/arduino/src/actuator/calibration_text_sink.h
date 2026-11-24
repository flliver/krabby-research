#pragma once

#include "flash_text.h"

// Where calibration status lines and follower commands are written. The
// firmware adapts an Arduino Print (PrintCalibrationSink in actuator_manager.h);
// native tests record the text.
class CalibrationTextSink
{
public:
    // Literal text: a KRABBY_FLASH_LITERAL pointer (flash on AVR).
    virtual void printFlash(const char *flashText) = 0;
    // Text held in RAM (joint names, role labels, forwarded line text).
    virtual void printText(const char *text) = 0;
    virtual void printNumber(long value) = 0;
    virtual void printChar(char value) = 0;
    // Ends the line (println()).
    virtual void endLine() = 0;

protected:
    ~CalibrationTextSink() = default;
};
