#pragma once

#include <stdint.h>

#include "power_safety_policy.h"

enum class PowerSafetyLatchValue : uint8_t
{
    None = 0,
    Sleep = 1,
};

enum class PowerSafetyLatchLoadResult : uint8_t
{
    Empty,
    Loaded,
    Corrupt,
};

#pragma pack(push, 1)
struct PowerSafetyLatchRecord
{
    uint8_t magic;
    uint8_t schema;
    uint8_t value;
};
#pragma pack(pop)

static_assert(sizeof(PowerSafetyLatchRecord) == EEPROM_POWER_SAFETY_SIZE,
              "power-safety EEPROM layout changed");

class PowerSafetyLatch
{
public:
    enum class SaveResult : uint8_t
    {
        Saved,
        Unchanged,
        VerificationFailed,
    };

    template <typename Storage>
    PowerSafetyLatchLoadResult load(Storage &storage)
    {
        PowerSafetyLatchRecord record = erasedRecord();
        storage.load(record);
        if (isErased(record))
        {
            value_ = PowerSafetyLatchValue::None;
            isPersisted_ = false;
            return PowerSafetyLatchLoadResult::Empty;
        }
        if (!isValid(record))
        {
            value_ = PowerSafetyLatchValue::Sleep;
            isPersisted_ = false;
            return PowerSafetyLatchLoadResult::Corrupt;
        }

        value_ = static_cast<PowerSafetyLatchValue>(record.value);
        isPersisted_ = true;
        return PowerSafetyLatchLoadResult::Loaded;
    }

    template <typename Storage>
    SaveResult save(Storage &storage, PowerSafetyLatchValue value)
    {
        if (isPersisted_ && value_ == value)
            return SaveResult::Unchanged;

        const PowerSafetyLatchRecord expected = {
            EEPROM_POWER_SAFETY_MAGIC,
            EEPROM_POWER_SAFETY_SCHEMA,
            static_cast<uint8_t>(value),
        };
        PowerSafetyLatchRecord pending = expected;
        pending.magic = EEPROM_POWER_SAFETY_INVALID_MAGIC;
        storage.writeRecord(pending);
        storage.updateMagic(EEPROM_POWER_SAFETY_MAGIC);

        PowerSafetyLatchRecord verified = erasedRecord();
        storage.load(verified);
        if (!isValid(verified) || !recordsMatch(expected, verified))
        {
            storage.updateMagic(EEPROM_POWER_SAFETY_INVALID_MAGIC);
            value_ = value;
            isPersisted_ = false;
            return SaveResult::VerificationFailed;
        }

        value_ = value;
        isPersisted_ = true;
        return SaveResult::Saved;
    }

    PowerSafetyLatchValue value() const
    {
        return value_;
    }

private:
    static PowerSafetyLatchRecord erasedRecord()
    {
        const PowerSafetyLatchRecord record = {0xFF, 0xFF, 0xFF};
        return record;
    }

    static bool isErased(const PowerSafetyLatchRecord &record)
    {
        return record.magic == 0xFF &&
            record.schema == 0xFF &&
            record.value == 0xFF;
    }

    static bool isValid(const PowerSafetyLatchRecord &record)
    {
        return record.magic == EEPROM_POWER_SAFETY_MAGIC &&
            record.schema == EEPROM_POWER_SAFETY_SCHEMA &&
            record.value <= static_cast<uint8_t>(PowerSafetyLatchValue::Sleep);
    }

    static bool recordsMatch(
        const PowerSafetyLatchRecord &left,
        const PowerSafetyLatchRecord &right)
    {
        return left.magic == right.magic &&
            left.schema == right.schema &&
            left.value == right.value;
    }

    PowerSafetyLatchValue value_ = PowerSafetyLatchValue::None;
    bool isPersisted_ = false;
};
