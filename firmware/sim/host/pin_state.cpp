#include "pin_state.hpp"

namespace krabby {
namespace sim {

namespace {

PinStateModel g_pin_state;

bool validPin(int pin)
{
    return pin >= 0 && pin <= PinStateModel::kMaxMegaPin;
}

}  // namespace

void PinStateModel::reset()
{
    for (PinRecord& pin : pins_)
    {
        pin = PinRecord();
    }
}

void PinStateModel::pinMode(int pin, int mode)
{
    if (!validPin(pin)) return;
    record(pin).mode = mode;
}

void PinStateModel::digitalWrite(int pin, int value)
{
    if (!validPin(pin)) return;
    record(pin).digital = value ? 1 : 0;
}

void PinStateModel::analogWrite(int pin, int duty)
{
    if (!validPin(pin)) return;
    if (duty < 0) duty = 0;
    if (duty > 255) duty = 255;
    record(pin).pwm = duty;
}

void PinStateModel::setAnalogInput(int pin, int value)
{
    if (!validPin(pin)) return;
    if (value < 0) value = 0;
    if (value > 1023) value = 1023;
    record(pin).analog_adc = value;
}

int PinStateModel::getMode(int pin) const
{
    if (!validPin(pin)) return kPinModeUnset;
    return record(pin).mode;
}

int PinStateModel::digitalRead(int pin) const
{
    if (!validPin(pin)) return 0;
    return record(pin).digital;
}

int PinStateModel::analogInput(int pin) const
{
    if (!validPin(pin)) return 512;
    return record(pin).analog_adc;
}

int PinStateModel::pwmDuty(int pin) const
{
    if (!validPin(pin)) return 0;
    return record(pin).pwm;
}

bool PinStateModel::isDigitalHigh(int pin) const
{
    return digitalRead(pin) == 1;
}

int PinStateModel::signedPwmCommand(int pinPwmR, int pinPwmL, int pinEn) const
{
    if (!isDigitalHigh(pinEn)) return 0;
    return pwmDuty(pinPwmR) - pwmDuty(pinPwmL);
}

PinRecord& PinStateModel::record(int pin) { return pins_[pin]; }

const PinRecord& PinStateModel::record(int pin) const { return pins_[pin]; }

PinStateModel& pinState() { return g_pin_state; }

}  // namespace sim
}  // namespace krabby
