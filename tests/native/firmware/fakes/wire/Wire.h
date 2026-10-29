#pragma once

#include <stddef.h>
#include <stdint.h>
#include <deque>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace wire_native {
struct Event
{
    std::string name;
    std::vector<long> args;
};

// Reported counts and supplied bytes can differ to exercise short/broken reads.
struct Transfer
{
    uint8_t address = 0;
    size_t written = 1;
    uint8_t status = 0;
    bool timeout = false;
    uint8_t reported = 0;
    std::vector<uint8_t> bytes;
};

// A device fake that answers every transaction at its address. transmit receives
// the bytes written since beginTransmission and returns the endTransmission status
// (5 also sets the timeout flag); receive returns the bytes for a read, and fewer
// than requested is a short read.
class Device
{
public:
    virtual ~Device() {}
    virtual uint8_t transmit(const std::vector<uint8_t> &bytes) = 0;
    virtual std::vector<uint8_t> receive(uint8_t count) = 0;
};

struct State
{
    bool begun = false;
    uint32_t clock = 100000;
    uint32_t timeoutMicroseconds = 0;
    bool resetOnTimeout = false;
    bool timeout = false;
    std::deque<Transfer> transfers;
    std::vector<Event> events;
    std::vector<std::string> errors;
    std::function<void(const Event &)> observe;
    // Devices answering their address in place of scripted transfers; not owned.
    std::map<uint8_t, Device *> devices;

    void record(const char *name, std::initializer_list<long> args = {});
};
}

class TwoWire
{
public:
    TwoWire() : state_(&owned_) {}
    explicit TwoWire(wire_native::State &state) : state_(&state) {}
    TwoWire(const TwoWire &) = delete;
    TwoWire &operator=(const TwoWire &) = delete;
    wire_native::State &state() { return *state_; }
    void reset();
    void begin();
    void end();
    void setClock(uint32_t);
    void setWireTimeout(uint32_t, bool);
    void clearWireTimeoutFlag();
    bool getWireTimeoutFlag();
    void beginTransmission(uint8_t);
    size_t write(uint8_t);
    size_t write(const uint8_t *, size_t);
    uint8_t endTransmission(uint8_t stop = true);
    uint8_t requestFrom(uint8_t, uint8_t, uint8_t stop = true);
    int available();
    int read();

private:
    void takeTransfer(uint8_t address);
    wire_native::State owned_;
    wire_native::State *state_;
    wire_native::Transfer active_;
    bool hasTransfer_ = false;
    bool receiving_ = false;
    size_t cursor_ = 0;
    wire_native::Device *device_ = nullptr;
    std::vector<uint8_t> pending_;
};

extern TwoWire Wire;
