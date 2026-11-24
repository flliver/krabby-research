#include "pin_state.hpp"

namespace krabby {
namespace sim {

namespace {

PinStateModel g_pin_state;

bool validPin(int pin)
{
    return pin >= 0 && pin <= PinStateModel::kMaxMegaPin;
}

bool validBoard(int board)
{
    return board >= 0 && board < kPinStateBoardCount;
}

constexpr int kPinModeInputPullUp = 2;     // Arduino.h INPUT_PULLUP
constexpr int kOpenInputPullUpAdc = 1020;  // a floating input pulled to the rail

}  // namespace

void PinStateModel::reset()
{
    for (PinRecord& pin : pins_)
    {
        pin = PinRecord();
    }
    for (auto& board : overrides_)
    {
        for (AnalogOverride& entry : board)
        {
            entry = AnalogOverride();
        }
    }
    active_board_ = 0;
    observer_ = nullptr;
}

void PinStateModel::setActiveBoard(int board)
{
    if (!validBoard(board)) return;
    active_board_ = board;
}

void PinStateModel::setWriteObserver(PinWriteObserver observer) { observer_ = observer; }

void PinStateModel::setBoardAnalogOverride(int board, int pin, int value, bool isOpenInput)
{
    if (!validBoard(board) || !validPin(pin)) return;
    if (value < 0) value = 0;
    if (value > 1023) value = 1023;
    AnalogOverride& entry = overrides_[board][pin];
    entry.isSet = true;
    entry.isOpenInput = isOpenInput;
    entry.value = value;
}

void PinStateModel::notifyWrite(int pin, bool isPwm, int value)
{
    if (!observer_) return;
    observer_(PinWriteEvent{active_board_, pin, isPwm, value});
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
    notifyWrite(pin, false, record(pin).digital);
}

void PinStateModel::analogWrite(int pin, int duty)
{
    if (!validPin(pin)) return;
    if (duty < 0) duty = 0;
    if (duty > 255) duty = 255;
    record(pin).pwm = duty;
    notifyWrite(pin, true, duty);
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
    const AnalogOverride& entry = overrides_[active_board_][pin];
    if (entry.isSet)
    {
        return entry.isOpenInput && record(pin).mode == kPinModeInputPullUp
            ? kOpenInputPullUpAdc
            : entry.value;
    }
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
