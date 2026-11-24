#pragma once

#include <cstdint>
#include <functional>

namespace krabby {
namespace sim {

constexpr int kPinModeUnset = -1;
constexpr int kPinStateBoardCount = 3;  // FRONT, LEFT, RIGHT (BoardId order)

// One output write, tagged with the board whose firmware made it (test hook).
struct PinWriteEvent
{
    int board;
    int pin;
    bool isPwm;  // analogWrite duty; otherwise digitalWrite level
    int value;   // after clamping, as stored
};
using PinWriteObserver = std::function<void(const PinWriteEvent &)>;

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

    // The three-board sim shares one pin map between boards. These hooks let a
    // test see which board wrote an output and give each board its own analog
    // inputs; with no observer and no overrides nothing changes. reset() clears them.
    void setActiveBoard(int board);
    int activeBoard() const { return active_board_; }
    void setWriteObserver(PinWriteObserver observer);
    // analogRead(pin) on `board` returns `value`; an open input reads near the
    // rail while the pin is in INPUT_PULLUP (the pot tracker's open probe).
    void setBoardAnalogOverride(int board, int pin, int value, bool isOpenInput = false);

private:
    struct AnalogOverride
    {
        bool isSet = false;
        bool isOpenInput = false;
        int value = 0;
    };

    PinRecord& record(int pin);
    const PinRecord& record(int pin) const;
    void notifyWrite(int pin, bool isPwm, int value);

    PinRecord pins_[kMaxMegaPin + 1];
    AnalogOverride overrides_[kPinStateBoardCount][kMaxMegaPin + 1];
    int active_board_ = 0;
    PinWriteObserver observer_;
};

PinStateModel& pinState();

}  // namespace sim
}  // namespace krabby
