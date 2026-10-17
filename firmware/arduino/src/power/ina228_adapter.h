#pragma once

#include <Adafruit_INA228.h>
#include <Wire.h>

#include "battery_measurement.h"
#include "ina228_constants.h"
#include "../imu/imu_constants.h"

enum class Ina228InitializationResult : uint8_t
{
    Ok = 0,
    NotDetected,
    ConfigurationFailed,
};

class Ina228BatteryCluster
{
public:
    Ina228InitializationResult initialize()
    {
        Wire.begin();
        Wire.setClock(I2C_DEFAULT_BUS_CLOCK_HZ);
        Wire.setWireTimeout(I2C_BUS_TIMEOUT_MICROSECONDS, true);

        if (!packMonitor_.begin(INA228_PACK_ADDRESS, &Wire))
        {
            return Ina228InitializationResult::NotDetected;
        }
        if (!midpointMonitor_.begin(INA228_MIDPOINT_ADDRESS, &Wire))
        {
            return Ina228InitializationResult::NotDetected;
        }
        packMonitor_.setShunt(
            INA228_PACK_SHUNT_OHMS, INA228_PACK_SHUNT_MAX_AMPS);
        online_ = true;
        return Ina228InitializationResult::Ok;
    }

    bool isOnline() const { return online_; }

    BatteryMeasurement measure() const
    {
        BatteryMeasurement out{};
        if (!online_) return out;

        out.packVolts = packMonitor_.readBusVoltage();
        out.packAmps = packMonitor_.readCurrent();
        out.packWatts = packMonitor_.readPower();
        out.packChargeCoulombs = packMonitor_.readCharge();
        out.battAVolts = midpointMonitor_.readBusVoltage();
        out.battBVolts = out.packVolts - out.battAVolts;
        out.powerState = 0;
        out.valid = true;
        updateBatteryDivergenceFlag(out, INA228_DIVERGENCE_THRESHOLD_VOLTS);
        return out;
    }

private:
    Adafruit_INA228 packMonitor_;
    Adafruit_INA228 midpointMonitor_;
    bool online_ = false;
};
