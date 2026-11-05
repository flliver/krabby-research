#pragma once

#include <SparkFun_INA2XX.h>
#include <Arduino.h>
#include <Wire.h>
#include <math.h>

#include "../i2c/i2c_recovery.h"
#include "power_monitor_constants.h"
#include "power_measurement.h"

class Ina228Adapter
{
public:
    // Fields are meaningful only when both diagnostic registers were read.
    struct Diagnostics
    {
        uint16_t adcConfig;
        uint8_t adcMode;
        uint32_t busVoltageRegister;
        Volts decodedBusVoltage;
        bool isValid;
    };

    explicit Ina228Adapter(uint8_t address)
        : Ina228Adapter(address, 0.0f, 0.0f)
    {
        hasShuntConfiguration_ = false;
    }

    explicit Ina228Adapter(uint8_t address, float shuntResistanceOhms,
        float shuntMaxCurrentAmps, bool resetChargeOnBegin = false)
        : address_(address), shuntResistanceOhms_(shuntResistanceOhms),
          shuntMaxCurrentAmps_(shuntMaxCurrentAmps), hasShuntConfiguration_(true),
          resetChargeOnBegin_(resetChargeOnBegin),
          wire_(nullptr), isUp_(false)
    {
    }

    bool begin(TwoWire *wire)
    {
        wire_ = wire;
        isUp_ = wire_ && start(wire_, false);
        return isUp_;
    }

    bool isUp() const { return isUp_; }

    Volts readBusVoltage()
    {
        float value = NAN;
        if (device_.getBusVoltage_V(value) != ksfTkErrOk) value = NAN;
        return Volts(value);
    }

    Amps readCurrent()
    {
        float value = NAN;
        if (device_.getCurrent_A(value) != ksfTkErrOk) value = NAN;
        return Amps(value);
    }

    Watts readPower()
    {
        float value = NAN;
        if (device_.getPower_W(value) != ksfTkErrOk) value = NAN;
        return Watts(value);
    }

    Coulombs readCharge()
    {
        double value = NAN;
        if (device_.getCharge_C(value) != ksfTkErrOk) value = NAN;
        return Coulombs(value);
    }

    PowerMonitorMeasurement measure()
    {
        const I2cRecoveryLimits limits = {
            POWER_MONITOR_REINIT_AFTER_BAD_TICKS,
            POWER_MONITOR_REINIT_INTERVAL_MILLISECONDS,
        };
        PowerMonitorMeasurement measurement;
        if (!wire_) return measurement;
        const bool wasUp = isUp_;
        if (!wasUp)
        {
            if (!recovery_.shouldAttemptRecovery(millis(), limits) || !recover())
                return measurement;
        }
        float voltage = NAN, current = NAN, power = NAN;
        double charge = NAN;
        const bool voltageRead = device_.getBusVoltage_V(voltage) == ksfTkErrOk;
        const bool currentRead = device_.getCurrent_A(current) == ksfTkErrOk;
        const bool powerRead = device_.getPower_W(power) == ksfTkErrOk;
        const bool chargeRead = device_.getCharge_C(charge) == ksfTkErrOk;
        measurement.voltage = Volts(voltageRead ? voltage : NAN);
        measurement.current = Amps(currentRead ? current : NAN);
        measurement.power = Watts(powerRead ? power : NAN);
        measurement.charge = Coulombs(chargeRead ? charge : NAN);
        measurement.isValid = voltageRead && currentRead && powerRead && chargeRead;
        if (measurement.isValid)
            recovery_.noteSuccess();
        else if (wasUp && recovery_.shouldAttemptRecovery(millis(), limits))
            recover();
        return measurement;
    }

    Diagnostics readDiagnostics()
    {
        static constexpr uint8_t ADC_CONFIG_REGISTER = 0x01;
        static constexpr uint8_t BUS_VOLTAGE_REGISTER = 0x05;
        const auto adc = readRegister(ADC_CONFIG_REGISTER, 2);
        const auto vbus = readRegister(BUS_VOLTAGE_REGISTER, 3);
        if (!adc.isValid || !vbus.isValid)
            return {};

        return {
            static_cast<uint16_t>(adc.value),
            static_cast<uint8_t>(adc.value >> 12),
            vbus.value,
            Volts(static_cast<float>(vbus.value >> 4) * 195.3125f / 1000000.0f),
            true,
        };
    }

    uint8_t address() const { return address_; }
    uint8_t badTicks() const { return recovery_.badTicks(); }

private:
    struct RegisterReading
    {
        uint32_t value;
        bool isValid;
    };

    // Diagnostic reads do not participate in measurement recovery or configuration.
    RegisterReading readRegister(uint8_t reg, uint8_t byteCount)
    {
        if (!wire_)
            return {};
        wire_->beginTransmission(address_);
        wire_->write(reg);
        if (wire_->endTransmission(false) != 0 ||
            wire_->requestFrom(address_, byteCount) != byteCount)
            return {};

        uint32_t value = 0;
        for (uint8_t i = 0; i < byteCount; ++i)
        {
            const int byte = wire_->read();
            if (byte < 0)
                return {};
            value = (value << 8) | static_cast<uint8_t>(byte);
        }
        return {value, true};
    }

    bool recover()
    {
        isUp_ = start(wire_, true);
        return isUp_;
    }

    bool start(TwoWire *wire, bool isRecovery)
    {
        if (!device_.begin(address_, *wire))
            return false;
        // Keep accumulated pack charge across recovery.
        if (!(isRecovery && resetChargeOnBegin_))
        {
            if (device_.reset() != ksfTkErrOk) return false;
        }
        if (device_.setConversionReadyAlert(true) != ksfTkErrOk ||
            device_.setADCMode(INA2XX_MODE_CONT_ALL) != ksfTkErrOk)
            return false;
        delay(2);
        if (hasShuntConfiguration_ &&
            device_.calibrate(shuntResistanceOhms_, shuntMaxCurrentAmps_) != ksfTkErrOk)
            return false;
        if (!isRecovery && resetChargeOnBegin_ && device_.resetAccumulators() != ksfTkErrOk)
            return false;
        return true;
    }

    SfeINA228ArdI2C device_;
    I2cRecoveryPolicy recovery_;
    uint8_t address_;
    float shuntResistanceOhms_;
    float shuntMaxCurrentAmps_;
    bool hasShuntConfiguration_;
    bool resetChargeOnBegin_;
    TwoWire *wire_;
    bool isUp_;
};
