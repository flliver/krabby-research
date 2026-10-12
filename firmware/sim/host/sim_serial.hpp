#pragma once

#include "arduino_io.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>

namespace krabby {
namespace sim {

enum class SerialPortId : int
{
    kUsb = 0,
    kSerial1 = 1,
    kSerial2 = 2,
};

class SimSerialPort
{
public:
    void reset();

    void txWrite(uint8_t byte);
    int txBytesAvailable() const;
    int popTxByte();
    bool txLineAvailable() const;
    std::string popTxLine();

    void rxPush(const uint8_t *data, std::size_t length);
    void rxPushByte(uint8_t byte);
    int rxAvailable() const;
    int rxRead();
    int rxPeek();

    String readStringUntil(char terminator);

private:
    void flushTxLine();

    std::string tx_partial_;
    std::deque<std::string> tx_lines_;
    std::deque<uint8_t> tx_bytes_;
    std::deque<uint8_t> rx_;
};

SimSerialPort &serialPort(SerialPortId id);
void resetHostSerial();

class HardwareSerial : public Print
{
public:
    explicit HardwareSerial(SerialPortId id);

    void begin(unsigned long /*baud*/) {}
    void flush() {}

    int available();
    int read();
    int peek();
    String readStringUntil(char terminator);

    void write(uint8_t byte) override;
    void write(const uint8_t *buffer, std::size_t size) override;

private:
    SerialPortId id_;
    SimSerialPort &port();
};

}  // namespace sim
}  // namespace krabby

extern krabby::sim::HardwareSerial Serial;
extern krabby::sim::HardwareSerial Serial1;
extern krabby::sim::HardwareSerial Serial2;
