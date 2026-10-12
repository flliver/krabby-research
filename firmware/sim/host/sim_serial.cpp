#include "sim_serial.hpp"

namespace krabby {
namespace sim {

namespace {

SimSerialPort g_ports[3];

}  // namespace

void SimSerialPort::reset()
{
    tx_partial_.clear();
    tx_lines_.clear();
    rx_.clear();
}

void SimSerialPort::txWrite(uint8_t byte)
{
    if (byte == '\n')
    {
        flushTxLine();
        return;
    }
    if (byte != '\r')
    {
        tx_partial_.push_back(static_cast<char>(byte));
    }
}

void SimSerialPort::flushTxLine()
{
    tx_lines_.push_back(tx_partial_);
    tx_partial_.clear();
}

bool SimSerialPort::txLineAvailable() const { return !tx_lines_.empty(); }

std::string SimSerialPort::popTxLine()
{
    if (tx_lines_.empty()) return {};
    std::string line = tx_lines_.front();
    tx_lines_.pop_front();
    return line;
}

void SimSerialPort::rxPush(const uint8_t *data, std::size_t length)
{
    for (std::size_t i = 0; i < length; ++i)
    {
        rx_.push_back(data[i]);
    }
}

void SimSerialPort::rxPushByte(uint8_t byte) { rx_.push_back(byte); }

int SimSerialPort::rxAvailable() const
{
    return static_cast<int>(rx_.size());
}

int SimSerialPort::rxRead()
{
    if (rx_.empty()) return -1;
    uint8_t byte = rx_.front();
    rx_.pop_front();
    return static_cast<int>(byte);
}

int SimSerialPort::rxPeek()
{
    if (rx_.empty()) return -1;
    return static_cast<int>(rx_.front());
}

String SimSerialPort::readStringUntil(char terminator)
{
    std::string result;
    while (rxAvailable() > 0)
    {
        int next = rxPeek();
        if (next < 0) break;
        if (static_cast<char>(next) == terminator)
        {
            rxRead();
            break;
        }
        result.push_back(static_cast<char>(rxRead()));
    }
    return String(result);
}

SimSerialPort &serialPort(SerialPortId id)
{
    return g_ports[static_cast<int>(id)];
}

void resetHostSerial()
{
    for (SimSerialPort &port : g_ports)
    {
        port.reset();
    }
}

HardwareSerial::HardwareSerial(SerialPortId id) : id_(id) {}

SimSerialPort &HardwareSerial::port() { return serialPort(id_); }

int HardwareSerial::available() { return port().rxAvailable(); }

int HardwareSerial::read() { return port().rxRead(); }

int HardwareSerial::peek() { return port().rxPeek(); }

String HardwareSerial::readStringUntil(char terminator)
{
    return port().readStringUntil(terminator);
}

void HardwareSerial::write(uint8_t byte)
{
    port().txWrite(byte);
    Print::write(byte);
}

void HardwareSerial::write(const uint8_t *buffer, std::size_t size)
{
    for (std::size_t i = 0; i < size; ++i)
    {
        write(buffer[i]);
    }
}

}  // namespace sim
}  // namespace krabby

krabby::sim::HardwareSerial Serial(krabby::sim::SerialPortId::kUsb);
krabby::sim::HardwareSerial Serial1(krabby::sim::SerialPortId::kSerial1);
krabby::sim::HardwareSerial Serial2(krabby::sim::SerialPortId::kSerial2);
