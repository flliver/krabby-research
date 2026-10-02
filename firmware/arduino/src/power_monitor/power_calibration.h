#pragma once

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "../units/electrical_units.h"
#include "power_monitor_constants.h"
#include "power_measurement.h"

#pragma pack(push, 1)
struct PowerCalibrationRecord
{
    uint8_t magic;
    uint8_t schema;
    float packVoltageOffset;
    float midpointVoltageOffset;
    float packShuntScale;
};
#pragma pack(pop)

static_assert(sizeof(PowerCalibrationRecord) == EEPROM_POWER_CAL_SIZE,
              "power-calibration EEPROM layout changed");
static_assert(POWER_CAL_PACK_REF_MAX_V > 0.0f,
              "pack reference limit must be positive");
static_assert(POWER_CAL_MID_REF_MAX_V > 0.0f,
              "midpoint reference limit must be positive");
static_assert(POWER_CAL_MAX_VOFFSET_V >= 0.0f,
              "voltage offset limit cannot be negative");
static_assert(POWER_CAL_MIN_SHUNT_TRIM_A > 0.0f,
              "minimum calibration current must be positive");
static_assert(POWER_CAL_MIN_GAIN > 0.0f &&
              POWER_CAL_MAX_GAIN >= POWER_CAL_MIN_GAIN,
              "invalid shunt scale limits");

class PowerCalibration
{
public:
    enum class SaveResult : uint8_t
    {
        Saved,
        InvalidInput,
        VerificationFailed,
    };

    PowerCalibration()
        : record_(identityRecord())
    {
    }

    template <typename Storage>
    bool load(Storage &storage)
    {
        PowerCalibrationRecord stored = identityRecord();
        storage.load(stored);
        if (!isValid(stored))
        {
            record_ = identityRecord();
            return false;
        }

        record_ = stored;
        return true;
    }

    template <typename Storage>
    SaveResult captureVoltage(
        Storage &storage,
        Volts rawPackVoltage,
        Volts rawMidpointVoltage,
        Volts packReference,
        Volts midpointReference)
    {
        if (!isReferenceValid(
                packReference, Volts(POWER_CAL_PACK_REF_MAX_V)) ||
            !isReferenceValid(
                midpointReference, Volts(POWER_CAL_MID_REF_MAX_V)))
            return SaveResult::InvalidInput;

        PowerCalibrationRecord candidate = record_;
        if (!calculateOffset(
                packReference,
                rawPackVoltage,
                candidate.packVoltageOffset) ||
            !calculateOffset(
                midpointReference,
                rawMidpointVoltage,
                candidate.midpointVoltageOffset))
            return SaveResult::InvalidInput;

        return persist(storage, candidate);
    }

    template <typename Storage>
    SaveResult captureCurrent(
        Storage &storage,
        Amps measuredCurrent,
        Amps knownCurrent)
    {
        const float measured = measuredCurrent.value();
        const float known = knownCurrent.value();
        if (!isfinite(measured) ||
            !isfinite(known) ||
            fabs(measured) < POWER_CAL_MIN_SHUNT_TRIM_A ||
            fabs(known) < POWER_CAL_MIN_SHUNT_TRIM_A)
            return SaveResult::InvalidInput;

        const float scale = known / measured;
        if (!isfinite(scale) ||
            scale < POWER_CAL_MIN_GAIN ||
            scale > POWER_CAL_MAX_GAIN)
            return SaveResult::InvalidInput;

        PowerCalibrationRecord candidate = record_;
        candidate.packShuntScale = scale;
        return persist(storage, candidate);
    }

    PowerMonitorMeasurement applyPackCalibration(const PowerMonitorMeasurement &raw) const
    {
        PowerMonitorMeasurement calibrated = raw;
        calibrated.voltage = raw.voltage + packVoltageOffset();
        calibrated.current = raw.current.scalarMultiply(record_.packShuntScale);
        calibrated.power = raw.power.scalarMultiply(record_.packShuntScale);
        calibrated.charge = raw.charge.scalarMultiply(record_.packShuntScale);
        return calibrated;
    }

    PowerMonitorMeasurement applyMidpointCalibration(const PowerMonitorMeasurement &raw) const
    {
        PowerMonitorMeasurement calibrated = raw;
        calibrated.voltage = raw.voltage + midpointVoltageOffset();
        return calibrated;
    }

    Volts packVoltageOffset() const
    {
        return Volts(record_.packVoltageOffset);
    }

    Volts midpointVoltageOffset() const
    {
        return Volts(record_.midpointVoltageOffset);
    }

    float packShuntScale() const
    {
        return record_.packShuntScale;
    }

private:
    static PowerCalibrationRecord identityRecord()
    {
        const PowerCalibrationRecord record = {
            EEPROM_POWER_CAL_INVALID_MAGIC, 0, 0.0f, 0.0f, 1.0f};
        return record;
    }

    static bool isReferenceValid(Volts reference, Volts maximum)
    {
        return isfinite(reference.value()) &&
            reference.value() > 0.0f &&
            reference.value() <= maximum.value();
    }

    static bool calculateOffset(
        Volts reference,
        Volts rawVoltage,
        float &result)
    {
        if (!isfinite(rawVoltage.value()))
            return false;

        const float offset = reference.value() - rawVoltage.value();
        if (!isfinite(offset) ||
            fabs(offset) > POWER_CAL_MAX_VOFFSET_V)
            return false;

        result = offset;
        return true;
    }

    static bool isValid(const PowerCalibrationRecord &record)
    {
        return record.magic == EEPROM_POWER_CAL_MAGIC &&
            record.schema == EEPROM_POWER_CAL_SCHEMA &&
            isfinite(record.packVoltageOffset) &&
            fabs(record.packVoltageOffset) <= POWER_CAL_MAX_VOFFSET_V &&
            isfinite(record.midpointVoltageOffset) &&
            fabs(record.midpointVoltageOffset) <= POWER_CAL_MAX_VOFFSET_V &&
            isfinite(record.packShuntScale) &&
            record.packShuntScale >= POWER_CAL_MIN_GAIN &&
            record.packShuntScale <= POWER_CAL_MAX_GAIN;
    }

    static bool recordsMatch(
        const PowerCalibrationRecord &left,
        const PowerCalibrationRecord &right)
    {
        return left.magic == right.magic &&
            left.schema == right.schema &&
            left.packVoltageOffset == right.packVoltageOffset &&
            left.midpointVoltageOffset == right.midpointVoltageOffset &&
            left.packShuntScale == right.packShuntScale;
    }

    template <typename Storage>
    SaveResult persist(
        Storage &storage,
        PowerCalibrationRecord candidate)
    {
        candidate.magic = EEPROM_POWER_CAL_MAGIC;
        candidate.schema = EEPROM_POWER_CAL_SCHEMA;

        PowerCalibrationRecord pending = candidate;
        pending.magic = EEPROM_POWER_CAL_INVALID_MAGIC;
        storage.writeRecord(pending);
        storage.updateMagic(EEPROM_POWER_CAL_MAGIC);

        PowerCalibrationRecord verified = identityRecord();
        storage.load(verified);
        if (!isValid(verified) || !recordsMatch(candidate, verified))
        {
            storage.updateMagic(EEPROM_POWER_CAL_INVALID_MAGIC);
            return SaveResult::VerificationFailed;
        }

        record_ = verified;
        return SaveResult::Saved;
    }

    PowerCalibrationRecord record_;
};
