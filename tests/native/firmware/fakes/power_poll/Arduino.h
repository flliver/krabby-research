#pragma once

#include <cmath>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>

#include "environment.h"

inline uint32_t millis()
{
    powerPollFake::events.push_back("millis=" + std::to_string(powerPollFake::now));
    return powerPollFake::now;
}

// Deterministic host text sink. Not an emulator of AVR Print's floating-point
// rounding/overflow behavior; those limits are documented with the scenarios.
class Print
{
public:
    std::string output;
    void print(const char *value) { start(); output += value; }
    void print(char value) { start(); output += value; }
    void print(int value) { start(); output += std::to_string(value); }
    void print(uint8_t value) { print(static_cast<int>(value)); }
    void print(float value, int precision)
    {
        start();
        if (std::isnan(value)) { output += "nan"; return; }
        if (std::isinf(value)) { output += value < 0 ? "-inf" : "inf"; return; }
        std::ostringstream stream;
        stream.imbue(std::locale::classic());
        stream << std::fixed << std::setprecision(precision) << value;
        output += stream.str();
    }
private:
    void start()
    {
        if (output.empty())
            powerPollFake::events.push_back("telemetry.begin");
    }
};

// Out of line: the SparkFun Toolkit's I2C read path links against it.
void delay(unsigned long);
