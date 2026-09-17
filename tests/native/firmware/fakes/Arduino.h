#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cctype>
#include <deque>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>

#define INPUT 0
#define OUTPUT 1
#define INPUT_PULLUP 2
#define LOW 0
#define HIGH 1
#define constrain(value, low, high) ((value) < (low) ? (low) : ((value) > (high) ? (high) : (value)))

class String : public std::string
{
public:
    using std::string::string;
    String(const std::string &value) : std::string(value) {}
    String substring(int begin, int end) const { return substr(begin, end - begin); }
    float toFloat() const { return std::strtof(c_str(), nullptr); }
};

class Print
{
public:
    std::string output;
    template <typename T> void print(const T &value)
    {
        std::ostringstream stream;
        stream << value;
        output += stream.str();
    }
    void print(float value, int precision)
    {
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(precision) << value;
        output += stream.str();
    }
    void println(const char *value) { print(value); output += '\n'; }
};

namespace fakeArduino
{
struct Event
{
    char operation;
    int pin;
    int value;
};
extern uint32_t now;
extern int modes[32];
extern int digital[32];
extern int pwm[32];
extern int analog[32];
extern std::deque<int> readings[32];
extern std::vector<Event> events;
}

extern Print Serial;
inline unsigned long millis() { return fakeArduino::now; }
inline void pinMode(int pin, int mode)
{
    fakeArduino::modes[pin] = mode;
    fakeArduino::events.push_back({'m', pin, mode});
}
inline void digitalWrite(int pin, int value) { fakeArduino::digital[pin] = value; }
inline int digitalRead(int pin) { return fakeArduino::digital[pin]; }
inline void analogWrite(int pin, int value) { fakeArduino::pwm[pin] = value; }
inline int analogRead(int pin)
{
    int value = fakeArduino::analog[pin];
    if (!fakeArduino::readings[pin].empty())
    {
        value = fakeArduino::readings[pin].front();
        fakeArduino::readings[pin].pop_front();
    }
    fakeArduino::events.push_back({'r', pin, value});
    return value;
}
inline void delayMicroseconds(unsigned int duration)
{
    fakeArduino::events.push_back({'d', -1, static_cast<int>(duration)});
}
