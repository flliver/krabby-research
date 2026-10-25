#pragma once

#include <cstdint>

namespace krabby {
namespace sim {

constexpr int kPinModeUnset = -1;

struct PinRecord
{
    int mode = kPinModeUnset;
    int digital = 0;
    int pwm = 0;
    int analog_adc = 512;  // host analogRead; IS pins updated by current_sense_readback each step
};

class PinStateModel
{
public:
    static constexpr int kMaxMegaPin = 69;

    void reset();

    void pinMode(int pin, int mode);
    void digitalWrite(int pin, int value);
    void analogWrite(int pin, int duty);
    void setAnalogInput(int pin, int value);

    int getMode(int pin) const;
    int digitalRead(int pin) const;
    int analogInput(int pin) const;
    int pwmDuty(int pin) const;
    bool isDigitalHigh(int pin) const;
    int signedPwmCommand(int pinPwmR, int pinPwmL, int pinEn) const;

private:
    PinRecord& record(int pin);
    const PinRecord& record(int pin) const;

    PinRecord pins_[kMaxMegaPin + 1];
};

PinStateModel& pinState();

}  // namespace sim
}  // namespace krabby
