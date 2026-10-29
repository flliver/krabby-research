#include "ina228.h"

#include <cmath>

namespace ina228_native {
namespace {

// Power-on values from the INA228 datasheet (SBOS938).
const uint16_t RESET_VALUES[][2] = {
    {ADC_CONFIG, 0xFB68}, {SHUNT_CAL, 0x1000}, {SHUNT_TEMPCO, 0x0000}, {DIAG_ALRT, 0x0001},
    {SOVL, 0x7FFF}, {SUVL, 0x8000}, {BOVL, 0x7FFF}, {BUVL, 0x0000},
    {TEMP_LIMIT, 0x7FFF}, {PWR_LIMIT, 0xFFFF},
};
constexpr double BUS_LSB_VOLTS = 195.3125e-6;
constexpr double SHUNT_LSB_VOLTS = 312.5e-9;
constexpr double REDUCED_SHUNT_LSB_VOLTS = 78.125e-9;
constexpr double CALIBRATION_SCALE = 13107.2e6;

int64_t clampSigned(int64_t value, int bits)
{
    const int64_t limit = int64_t(1) << (bits - 1);
    return value >= limit ? limit - 1 : (value < -limit ? -limit : value);
}

uint64_t clampUnsigned(int64_t value, int bits)
{
    const int64_t limit = (int64_t(1) << bits) - 1;
    return uint64_t(value < 0 ? 0 : (value > limit ? limit : value));
}

uint64_t twosComplement(int64_t value, int bits)
{
    return uint64_t(value) & ((uint64_t(1) << bits) - 1);
}

size_t widthOf(uint8_t reg)
{
    switch (reg)
    {
    case VSHUNT: case VBUS: case CURRENT: case POWER: return 3;
    case ENERGY: case CHARGE: return 5;
    default: return 2;
    }
}

const char *stepName(const Operation &operation)
{
    if (!operation.isWrite)
    {
        switch (operation.reg)
        {
        case DEVICE_ID: return "begin";
        case VBUS: return "voltage";
        case CURRENT: return "current";
        case POWER: return "power";
        case CHARGE: return "charge";
        default: return nullptr;
        }
    }
    switch (operation.reg)
    {
    case CONFIG:
        if (operation.value & CONFIG_RST) return "reset";
        return (operation.value & CONFIG_RSTACC) ? "accumulators" : nullptr;
    case DIAG_ALRT: return "alert";
    case ADC_CONFIG: return "mode";
    case SHUNT_CAL: return "calibrate";
    default: return nullptr;
    }
}

}

Device::Device(double shunt) : shuntOhms(shunt) { resetRegisters(); }

uint16_t Device::registerValue(uint8_t reg) const
{
    if (reg == CONFIG) return config_;
    const auto found = registers_.find(reg);
    return found == registers_.end() ? 0 : found->second;
}

std::vector<std::string> Device::steps() const
{
    std::vector<std::string> names;
    for (const Operation &operation : operations)
        if (const char *name = stepName(operation)) names.push_back(name);
    return names;
}

// A one-byte transaction sets the register pointer for a read; three bytes write a
// 16-bit register MSB first. An empty transaction is an address probe.
uint8_t Device::transmit(const std::vector<uint8_t> &bytes)
{
    if (bytes.empty()) return present ? 0 : 2;
    const uint8_t reg = bytes[0];
    const bool isWrite = bytes.size() > 1;
    const uint16_t value = bytes.size() == 3 ? uint16_t(bytes[1] << 8 | bytes[2]) : 0;
    const bool refused = !present || (isWrite && bytes.size() != 3) || (nack && nack(reg, isWrite, value));
    const Operation operation{reg, isWrite, value, !refused};
    operations.push_back(operation);
    if (onStep)
        if (const char *name = stepName(operation)) onStep(name);
    if (!present) return 2;
    if (refused) return 3;
    pointer_ = reg;
    if (isWrite) write(reg, value);
    return 0;
}

std::vector<uint8_t> Device::receive(uint8_t count)
{
    std::vector<uint8_t> bytes;
    if (!present) return bytes;
    const uint64_t value = read(pointer_);
    for (size_t index = widthOf(pointer_); index > 0 && bytes.size() < size_t(count); --index)
        bytes.push_back(uint8_t(value >> (8 * (index - 1))));
    return bytes;
}

void Device::resetRegisters()
{
    config_ = 0;
    registers_.clear();
    for (const auto &entry : RESET_VALUES) registers_[uint8_t(entry[0])] = entry[1];
    coulombs = 0.0;
}

void Device::write(uint8_t reg, uint16_t value)
{
    switch (reg)
    {
    case CONFIG:
        if (value & CONFIG_RST)
        {
            resetRegisters();
            ++resetCount;
            return;
        }
        if (value & CONFIG_RSTACC)
        {
            coulombs = 0.0;
            ++accumulatorResetCount;
        }
        config_ = uint16_t(value & ~(CONFIG_RST | CONFIG_RSTACC)); // both self-clear
        return;
    case SHUNT_CAL:
        registers_[reg] = uint16_t(value & 0x7FFF);
        return;
    case DIAG_ALRT: // configuration bits only; the flags below are read-only
        registers_[reg] = uint16_t((value & 0xF000) | (registerValue(reg) & 0x0FFF));
        return;
    default:
        if (registers_.count(reg)) registers_[reg] = value; // results and IDs are read-only
        return;
    }
}

uint64_t Device::read(uint8_t reg) const
{
    switch (reg)
    {
    case VSHUNT: return twosComplement(shuntRaw(), 20) << 4;
    case VBUS: return busRaw() << 4;
    case CURRENT: return twosComplement(currentRaw(), 20) << 4;
    case POWER: return powerRaw();
    case ENERGY: return 0;
    case CHARGE: return twosComplement(chargeRaw(), 40);
    case MANUFACTURER_ID: return 0x5449;
    case DEVICE_ID: return deviceId;
    default: return registerValue(reg);
    }
}

bool Device::isReducedRange() const { return (config_ & CONFIG_ADCRANGE) != 0; }

int64_t Device::shuntRaw() const
{
    const double lsb = isReducedRange() ? REDUCED_SHUNT_LSB_VOLTS : SHUNT_LSB_VOLTS;
    return clampSigned(std::llround(amps * shuntOhms / lsb), 20);
}

uint64_t Device::busRaw() const
{
    return clampUnsigned(std::llround(busVolts / BUS_LSB_VOLTS), 20);
}

// CURRENT_LSB * R = SHUNT_CAL / 13107.2e6 (x4 in the reduced range, where the shunt
// LSB is a quarter), so CURRENT = VSHUNT * 4096 / SHUNT_CAL in either range.
int64_t Device::currentRaw() const
{
    const uint16_t calibration = registerValue(SHUNT_CAL);
    if (calibration == 0) return 0;
    return clampSigned(shuntRaw() * 4096 / calibration, 20);
}

// POWER = CURRENT * VBUS * 195.3125e-6 / 3.2, an unsigned magnitude.
uint64_t Device::powerRaw() const
{
    const int64_t current = currentRaw();
    const uint64_t magnitude = uint64_t(current < 0 ? -current : current);
    return clampUnsigned(int64_t(magnitude * busRaw() / 16384), 24);
}

int64_t Device::chargeRaw() const
{
    const uint16_t calibration = registerValue(SHUNT_CAL);
    if (calibration == 0 || shuntOhms <= 0.0) return 0;
    const double currentLsb =
        calibration / (CALIBRATION_SCALE * (isReducedRange() ? 4.0 : 1.0) * shuntOhms);
    return clampSigned(std::llround(coulombs / currentLsb), 40);
}

}
