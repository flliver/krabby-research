#pragma once

#include "Wire.h"

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace ina228_native {

enum Register : uint8_t
{
    CONFIG = 0x00,
    ADC_CONFIG = 0x01,
    SHUNT_CAL = 0x02,
    SHUNT_TEMPCO = 0x03,
    VSHUNT = 0x04,
    VBUS = 0x05,
    DIETEMP = 0x06,
    CURRENT = 0x07,
    POWER = 0x08,
    ENERGY = 0x09,
    CHARGE = 0x0A,
    DIAG_ALRT = 0x0B,
    SOVL = 0x0C,
    SUVL = 0x0D,
    BOVL = 0x0E,
    BUVL = 0x0F,
    TEMP_LIMIT = 0x10,
    PWR_LIMIT = 0x11,
    MANUFACTURER_ID = 0x3E,
    DEVICE_ID = 0x3F,
};

static constexpr uint16_t CONFIG_RST = 0x8000;
static constexpr uint16_t CONFIG_RSTACC = 0x4000;
static constexpr uint16_t CONFIG_ADCRANGE = 0x0010;
static constexpr uint16_t DIAG_CNVR = 0x4000;
static constexpr uint16_t INA228_DEVICE_ID = 0x2281;

struct Operation
{
    uint8_t reg;
    bool isWrite;
    uint16_t value;
    bool acknowledged;
};

// An INA228 at one address, answering the real driver register by register.
// Measurements follow from the physical inputs through the datasheet's integer
// relations and the SHUNT_CAL the driver wrote, so a wrong calibration reads back
// as a wrong current, power or charge.
class Device : public wire_native::Device
{
public:
    explicit Device(double shunt = 0.0);

    double shuntOhms;
    double busVolts = 0.0;
    double amps = 0.0;
    double coulombs = 0.0; // cleared by RST and RSTACC, like the accumulator

    bool present = true; // false NACKs the address; attempts are still logged
    uint16_t deviceId = INA228_DEVICE_ID;
    // Returning true NACKs that register read (isWrite false) or write.
    std::function<bool(uint8_t reg, bool isWrite, uint16_t value)> nack;
    // Called with each step's name as it is logged, NACKed steps included.
    std::function<void(const std::string &step)> onStep;

    unsigned resetCount = 0;
    unsigned accumulatorResetCount = 0;
    std::vector<Operation> operations; // register reads and writes, refused ones included

    uint16_t registerValue(uint8_t reg) const;
    // Adapter steps inferred from traffic, one per driver call: begin (identity
    // read); reset, accumulators, alert, mode, calibrate (writes); and voltage,
    // current, power, charge (measurement reads). Read-before-write reads are omitted.
    std::vector<std::string> steps() const;
    void clearLog() { operations.clear(); }

    uint8_t transmit(const std::vector<uint8_t> &bytes) override;
    std::vector<uint8_t> receive(uint8_t count) override;

private:
    void resetRegisters();
    void write(uint8_t reg, uint16_t value);
    uint64_t read(uint8_t reg) const;
    bool isReducedRange() const;
    int64_t shuntRaw() const;
    uint64_t busRaw() const;
    int64_t currentRaw() const;
    uint64_t powerRaw() const;
    int64_t chargeRaw() const;

    uint8_t pointer_ = CONFIG;
    uint16_t config_ = 0;
    std::map<uint8_t, uint16_t> registers_;
};

}
