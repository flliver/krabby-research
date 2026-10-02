#include "Wire.h"

TwoWire Wire;

static wire_native::Device *deviceAt(wire_native::State &state, uint8_t address)
{
    const auto found = state.devices.find(address);
    return found == state.devices.end() ? nullptr : found->second;
}

void wire_native::State::record(const char *name, std::initializer_list<long> args)
{
    const Event event{name, args};
    events.push_back(event);
    if (observe) observe(event);
}

void TwoWire::reset()
{
    state() = wire_native::State{};
    active_ = wire_native::Transfer{};
    hasTransfer_ = receiving_ = false;
    cursor_ = 0;
    device_ = nullptr;
    pending_.clear();
}
void TwoWire::begin() { state().begun = true; state().record("wire.begin"); }
void TwoWire::end()
{
    state().begun = false;
    hasTransfer_ = receiving_ = false;
    cursor_ = 0;
    device_ = nullptr;
    pending_.clear();
    state().record("wire.end");
}
void TwoWire::setClock(uint32_t hz) { state().clock = hz; state().record("wire.clock", {long(hz)}); }
void TwoWire::setWireTimeout(uint32_t us, bool reset)
{
    state().timeoutMicroseconds = us;
    state().resetOnTimeout = reset;
    state().record("wire.timeout", {long(us), reset});
}
void TwoWire::clearWireTimeoutFlag() { state().timeout = false; state().record("wire.clearTimeout"); }
bool TwoWire::getWireTimeoutFlag() { state().record("wire.getTimeout", {state().timeout}); return state().timeout; }

void TwoWire::takeTransfer(uint8_t address)
{
    active_ = wire_native::Transfer{};
    receiving_ = false;
    cursor_ = 0;
    hasTransfer_ = true;
    if (state().transfers.empty())
    {
        state().errors.push_back("unscripted Wire transaction");
        active_.address = address;
        active_.written = 0;
        active_.status = 2;
        return;
    }
    active_ = state().transfers.front();
    state().transfers.pop_front();
    if (active_.address != address) state().errors.push_back("wrong Wire transaction address");
}
void TwoWire::beginTransmission(uint8_t address)
{
    state().record("wire.transmit", {address});
    pending_.clear();
    device_ = deviceAt(state(), address);
    if (!device_)
    {
        takeTransfer(address);
        return;
    }
    active_ = wire_native::Transfer{};
    active_.address = address;
    hasTransfer_ = true;
    receiving_ = false;
    cursor_ = 0;
}
size_t TwoWire::write(uint8_t value)
{
    state().record("wire.write", {value});
    if (!hasTransfer_)
    {
        state().errors.push_back("Wire write without beginTransmission");
        return 0;
    }
    if (device_)
    {
        pending_.push_back(value);
        return 1;
    }
    return active_.written;
}

// Byte by byte, so buffered writes produce the same events and scripted counts.
size_t TwoWire::write(const uint8_t *data, size_t length)
{
    size_t written = 0;
    for (size_t index = 0; index < length; ++index)
        written += write(data[index]);
    return written;
}
uint8_t TwoWire::endTransmission(uint8_t stop)
{
    state().record("wire.endTransmission", {stop});
    if (!hasTransfer_)
    {
        state().errors.push_back("Wire endTransmission without beginTransmission");
        return 2;
    }
    if (device_)
    {
        active_.status = device_->transmit(pending_);
        active_.timeout = active_.status == 5;
        pending_.clear();
        device_ = nullptr;
    }
    state().timeout = state().timeout || active_.timeout;
    // A no-STOP register write and its following read share one scripted transfer.
    if (stop || active_.status != 0) hasTransfer_ = false;
    return active_.status;
}
uint8_t TwoWire::requestFrom(uint8_t address, uint8_t count, uint8_t stop)
{
    state().record("wire.request", {address, count, stop});
    if (wire_native::Device *device = deviceAt(state(), address))
    {
        if (hasTransfer_ && active_.address != address) state().errors.push_back("wrong Wire request address");
        active_ = wire_native::Transfer{};
        active_.address = address;
        active_.bytes = device->receive(count);
        active_.reported = static_cast<uint8_t>(active_.bytes.size());
    }
    else if (!hasTransfer_) takeTransfer(address);
    else if (active_.address != address) state().errors.push_back("wrong Wire request address");
    state().timeout = state().timeout || active_.timeout;
    hasTransfer_ = false;
    receiving_ = true;
    cursor_ = 0;
    return active_.reported;
}
int TwoWire::available() { return receiving_ ? static_cast<int>(active_.bytes.size() - cursor_) : 0; }
int TwoWire::read()
{
    if (!available())
    {
        state().errors.push_back("read beyond Wire receive buffer");
        return -1;
    }
    const int value = active_.bytes[cursor_++];
    state().record("wire.read", {value});
    return value;
}
