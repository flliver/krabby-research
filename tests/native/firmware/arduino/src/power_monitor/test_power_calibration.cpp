#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "unity.h"

#include "src/power_monitor/power_calibration.h"

void setUp() {}
void tearDown() {}

struct FakeStorage
{
    enum class Operation : uint8_t { Load, WriteRecord, UpdateMagic };

    uint8_t bytes[sizeof(PowerCalibrationRecord)];
    std::vector<Operation> operations;
    int failBeforeWrite;
    int writeCount;
    bool shouldCorruptVerification;
    bool shouldSubstitutePlausibleRecord;

    FakeStorage()
        : bytes{}, failBeforeWrite(-1), writeCount(0),
          shouldCorruptVerification(false),
          shouldSubstitutePlausibleRecord(false)
    {
        memset(bytes, 0xff, sizeof(bytes));
    }

    void seed(const PowerCalibrationRecord &record)
    {
        memcpy(bytes, &record, sizeof(record));
    }

    void load(PowerCalibrationRecord &record)
    {
        operations.push_back(Operation::Load);
        memcpy(&record, bytes, sizeof(record));
        if (shouldCorruptVerification)
            record.schema = 0;
        if (shouldSubstitutePlausibleRecord)
            record.packVoltageOffset += 0.1f;
    }

    void writeByte(size_t index, uint8_t value)
    {
        if (writeCount == failBeforeWrite)
            throw std::runtime_error("simulated power interruption");
        bytes[index] = value;
        ++writeCount;
    }

    void writeRecord(const PowerCalibrationRecord &record)
    {
        operations.push_back(Operation::WriteRecord);
        const uint8_t *raw = reinterpret_cast<const uint8_t *>(&record);
        for (size_t index = 0; index < sizeof(record); ++index)
            writeByte(index, raw[index]);
    }

    void updateMagic(uint8_t magic)
    {
        operations.push_back(Operation::UpdateMagic);
        writeByte(0, magic);
    }
};

static PowerCalibrationRecord record(
    float packOffset = -0.03f,
    float midpointOffset = 0.02f,
    float shuntScale = 1.01f)
{
    const PowerCalibrationRecord result = {
        EEPROM_POWER_CAL_MAGIC,
        EEPROM_POWER_CAL_SCHEMA,
        packOffset,
        midpointOffset,
        shuntScale};
    return result;
}

static PowerMonitorMeasurement shuntReadings(float current, float power, float charge)
{
    PowerMonitorMeasurement raw;
    raw.current = Amps(current);
    raw.power = Watts(power);
    raw.charge = Coulombs(charge);
    return raw;
}

static void assertIdentity(const PowerCalibration &calibration)
{
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calibration.packVoltageOffset().value());
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calibration.midpointVoltageOffset().value());
    TEST_ASSERT_EQUAL_FLOAT(1.0f, calibration.packShuntScale());
}

static void test_layout_and_identity_are_stable()
{
    TEST_ASSERT_EQUAL_UINT32(14, sizeof(PowerCalibrationRecord));
    PowerCalibration calibration;
    assertIdentity(calibration);
    TEST_ASSERT_EQUAL_FLOAT(
        2.0f, calibration.applyPackCalibration(shuntReadings(2.0f, 0, 0)).current.value());
    TEST_ASSERT_EQUAL_FLOAT(
        3.0f, calibration.applyPackCalibration(shuntReadings(0, 3.0f, 0)).power.value());
    TEST_ASSERT_EQUAL_FLOAT(
        4.0f, calibration.applyPackCalibration(shuntReadings(0, 0, 4.0f)).charge.value());
}

static void test_load_accepts_valid_record_and_rejects_invalid_records()
{
    FakeStorage storage;
    storage.seed(record());
    PowerCalibration calibration;
    TEST_ASSERT_TRUE(calibration.load(storage));
    TEST_ASSERT_EQUAL_FLOAT(-0.03f, calibration.packVoltageOffset().value());
    TEST_ASSERT_EQUAL_FLOAT(0.02f, calibration.midpointVoltageOffset().value());
    TEST_ASSERT_EQUAL_FLOAT(1.01f, calibration.packShuntScale());

    PowerCalibrationRecord invalid = record();
    invalid.magic = 0;
    storage.seed(invalid);
    TEST_ASSERT_FALSE(calibration.load(storage));
    assertIdentity(calibration);

    invalid = record();
    invalid.packVoltageOffset = std::numeric_limits<float>::quiet_NaN();
    storage.seed(invalid);
    TEST_ASSERT_FALSE(calibration.load(storage));
    assertIdentity(calibration);

    invalid = record(POWER_CAL_MAX_VOFFSET_V + 0.01f);
    storage.seed(invalid);
    TEST_ASSERT_FALSE(calibration.load(storage));
    assertIdentity(calibration);
}

static void test_load_rejects_each_nonfinite_field_and_accepts_exact_bounds()
{
    const float nonfinite[] = {
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
    };
    FakeStorage storage;
    PowerCalibration calibration;

    for (size_t field = 0; field < 3; ++field)
    {
        for (size_t value = 0; value < 3; ++value)
        {
            PowerCalibrationRecord invalid = record();
            if (field == 0) invalid.packVoltageOffset = nonfinite[value];
            if (field == 1) invalid.midpointVoltageOffset = nonfinite[value];
            if (field == 2) invalid.packShuntScale = nonfinite[value];
            storage.seed(invalid);
            TEST_ASSERT_FALSE(calibration.load(storage));
            assertIdentity(calibration);
        }
    }

    storage.seed(record(
        POWER_CAL_MAX_VOFFSET_V,
        -POWER_CAL_MAX_VOFFSET_V,
        POWER_CAL_MIN_GAIN));
    TEST_ASSERT_TRUE(calibration.load(storage));
    storage.seed(record(0.0f, 0.0f, POWER_CAL_MAX_GAIN));
    TEST_ASSERT_TRUE(calibration.load(storage));

    storage.seed(record(0.0f, 0.0f, POWER_CAL_MIN_GAIN - 0.01f));
    TEST_ASSERT_FALSE(calibration.load(storage));
    storage.seed(record(0.0f, 0.0f, POWER_CAL_MAX_GAIN + 0.01f));
    TEST_ASSERT_FALSE(calibration.load(storage));
}

static void test_voltage_capture_persists_then_activates_both_offsets()
{
    FakeStorage storage;
    PowerCalibration calibration;

    TEST_ASSERT_EQUAL(
        static_cast<int>(PowerCalibration::SaveResult::Saved),
        static_cast<int>(calibration.captureVoltage(
            storage,
            Volts(25.0f), Volts(12.0f),
            Volts(25.5f), Volts(11.75f))));
    TEST_ASSERT_EQUAL_FLOAT(0.5f, calibration.packVoltageOffset().value());
    TEST_ASSERT_EQUAL_FLOAT(-0.25f, calibration.midpointVoltageOffset().value());
    TEST_ASSERT_EQUAL_UINT32(3, storage.operations.size());
    TEST_ASSERT_EQUAL(
        static_cast<int>(FakeStorage::Operation::WriteRecord),
        static_cast<int>(storage.operations[0]));
    TEST_ASSERT_EQUAL(
        static_cast<int>(FakeStorage::Operation::UpdateMagic),
        static_cast<int>(storage.operations[1]));
    TEST_ASSERT_EQUAL(
        static_cast<int>(FakeStorage::Operation::Load),
        static_cast<int>(storage.operations[2]));
    TEST_ASSERT_EQUAL_HEX8(EEPROM_POWER_CAL_MAGIC, storage.bytes[0]);
}

static void test_invalid_voltage_capture_preserves_active_record_without_writing()
{
    FakeStorage storage;
    storage.seed(record(0.2f, -0.1f, 1.1f));
    PowerCalibration calibration;
    TEST_ASSERT_TRUE(calibration.load(storage));
    storage.operations.clear();

    TEST_ASSERT_EQUAL(
        static_cast<int>(PowerCalibration::SaveResult::InvalidInput),
        static_cast<int>(calibration.captureVoltage(
            storage,
            Volts(20.0f), Volts(10.0f),
            Volts(POWER_CAL_PACK_REF_MAX_V + 0.01f), Volts(10.0f))));
    TEST_ASSERT_EQUAL_FLOAT(0.2f, calibration.packVoltageOffset().value());
    TEST_ASSERT_EQUAL_FLOAT(-0.1f, calibration.midpointVoltageOffset().value());
    TEST_ASSERT_TRUE(storage.operations.empty());
}

static void test_voltage_capture_rejects_each_invalid_input()
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float invalid[][4] = {
        {25.0f, 12.0f, 0.0f, 12.0f},
        {25.0f, 12.0f, nan, 12.0f},
        {25.0f, 12.0f, 25.0f, 0.0f},
        {25.0f, 12.0f, 25.0f, POWER_CAL_MID_REF_MAX_V + 0.01f},
        {25.0f, nan, 25.0f, 12.0f},
        {20.0f, 12.0f, 20.0f + POWER_CAL_MAX_VOFFSET_V + 0.01f, 12.0f},
        {25.0f, 8.0f, 25.0f, 8.0f + POWER_CAL_MAX_VOFFSET_V + 0.01f},
    };

    for (size_t index = 0; index < sizeof(invalid) / sizeof(invalid[0]); ++index)
    {
        FakeStorage storage;
        PowerCalibration calibration;
        TEST_ASSERT_EQUAL(
            static_cast<int>(PowerCalibration::SaveResult::InvalidInput),
            static_cast<int>(calibration.captureVoltage(
                storage,
                Volts(invalid[index][0]), Volts(invalid[index][1]),
                Volts(invalid[index][2]), Volts(invalid[index][3]))));
        TEST_ASSERT_TRUE(storage.operations.empty());
        assertIdentity(calibration);
    }
}

static void test_voltage_capture_accepts_signed_exact_bounds()
{
    FakeStorage storage;
    PowerCalibration calibration;

    TEST_ASSERT_EQUAL(
        static_cast<int>(PowerCalibration::SaveResult::Saved),
        static_cast<int>(calibration.captureVoltage(
            storage,
            Volts(24.0f), Volts(12.0f),
            Volts(26.0f), Volts(10.0f))));
    TEST_ASSERT_EQUAL_FLOAT(
        POWER_CAL_MAX_VOFFSET_V,
        calibration.packVoltageOffset().value());
    TEST_ASSERT_EQUAL_FLOAT(
        -POWER_CAL_MAX_VOFFSET_V,
        calibration.midpointVoltageOffset().value());
}

static void test_bad_voltage_measurement_preserves_both_offsets()
{
    const float nonfinite[] = {
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
    };

    for (size_t index = 0; index < 3; ++index)
    {
        FakeStorage storage;
        storage.seed(record(0.2f, -0.1f, 1.1f));
        PowerCalibration calibration;
        TEST_ASSERT_TRUE(calibration.load(storage));
        storage.operations.clear();

        TEST_ASSERT_EQUAL(
            static_cast<int>(PowerCalibration::SaveResult::InvalidInput),
            static_cast<int>(calibration.captureVoltage(
                storage,
                Volts(nonfinite[index]), Volts(12.0f),
                Volts(25.0f), Volts(12.0f))));
        TEST_ASSERT_EQUAL_FLOAT(0.2f, calibration.packVoltageOffset().value());
        TEST_ASSERT_EQUAL_FLOAT(-0.1f, calibration.midpointVoltageOffset().value());
        TEST_ASSERT_TRUE(storage.operations.empty());
    }
}

static void test_current_capture_applies_one_scale_to_all_shunt_measurements()
{
    FakeStorage storage;
    PowerCalibration calibration;

    TEST_ASSERT_EQUAL(
        static_cast<int>(PowerCalibration::SaveResult::Saved),
        static_cast<int>(calibration.captureCurrent(
            storage, Amps(10.0f), Amps(12.0f))));
    TEST_ASSERT_EQUAL_FLOAT(1.2f, calibration.packShuntScale());
    TEST_ASSERT_EQUAL_FLOAT(
        12.0f, calibration.applyPackCalibration(shuntReadings(10.0f, 0, 0)).current.value());
    TEST_ASSERT_EQUAL_FLOAT(
        24.0f, calibration.applyPackCalibration(shuntReadings(0, 20.0f, 0)).power.value());
    TEST_ASSERT_EQUAL_FLOAT(
        36.0f, calibration.applyPackCalibration(shuntReadings(0, 0, 30.0f)).charge.value());
}

static void test_invalid_current_capture_preserves_active_record()
{
    FakeStorage storage;
    storage.seed(record(0.0f, 0.0f, 1.1f));
    PowerCalibration calibration;
    TEST_ASSERT_TRUE(calibration.load(storage));
    storage.operations.clear();

    TEST_ASSERT_EQUAL(
        static_cast<int>(PowerCalibration::SaveResult::InvalidInput),
        static_cast<int>(calibration.captureCurrent(
            storage, Amps(0.0f), Amps(1.0f))));
    TEST_ASSERT_EQUAL_FLOAT(1.1f, calibration.packShuntScale());
    TEST_ASSERT_TRUE(storage.operations.empty());
}

static void test_current_capture_accepts_signed_and_exact_scale_bounds()
{
    FakeStorage storage;
    PowerCalibration calibration;

    TEST_ASSERT_EQUAL(
        static_cast<int>(PowerCalibration::SaveResult::Saved),
        static_cast<int>(calibration.captureCurrent(
            storage, Amps(-8.0f), Amps(-10.0f))));
    TEST_ASSERT_EQUAL_FLOAT(1.25f, calibration.packShuntScale());

    TEST_ASSERT_EQUAL(
        static_cast<int>(PowerCalibration::SaveResult::Saved),
        static_cast<int>(calibration.captureCurrent(
            storage, Amps(10.0f), Amps(5.0f))));
    TEST_ASSERT_EQUAL_FLOAT(POWER_CAL_MIN_GAIN, calibration.packShuntScale());

    TEST_ASSERT_EQUAL(
        static_cast<int>(PowerCalibration::SaveResult::Saved),
        static_cast<int>(calibration.captureCurrent(
            storage, Amps(5.0f), Amps(10.0f))));
    TEST_ASSERT_EQUAL_FLOAT(POWER_CAL_MAX_GAIN, calibration.packShuntScale());
}

static void test_current_capture_rejects_nonfinite_opposite_and_small_currents()
{
    const float nonfinite[] = {
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
    };
    FakeStorage storage;
    PowerCalibration calibration;

    for (size_t index = 0; index < 3; ++index)
    {
        TEST_ASSERT_EQUAL(
            static_cast<int>(PowerCalibration::SaveResult::InvalidInput),
            static_cast<int>(calibration.captureCurrent(
                storage, Amps(nonfinite[index]), Amps(1.0f))));
        TEST_ASSERT_EQUAL(
            static_cast<int>(PowerCalibration::SaveResult::InvalidInput),
            static_cast<int>(calibration.captureCurrent(
                storage, Amps(1.0f), Amps(nonfinite[index]))));
    }

    TEST_ASSERT_EQUAL(
        static_cast<int>(PowerCalibration::SaveResult::InvalidInput),
        static_cast<int>(calibration.captureCurrent(
            storage, Amps(-1.0f), Amps(1.0f))));
    TEST_ASSERT_EQUAL(
        static_cast<int>(PowerCalibration::SaveResult::InvalidInput),
        static_cast<int>(calibration.captureCurrent(
            storage,
            Amps(POWER_CAL_MIN_SHUNT_TRIM_A - 0.001f),
            Amps(1.0f))));
    TEST_ASSERT_EQUAL(
        static_cast<int>(PowerCalibration::SaveResult::InvalidInput),
        static_cast<int>(calibration.captureCurrent(
            storage,
            Amps(1.0f),
            Amps(POWER_CAL_MIN_SHUNT_TRIM_A - 0.001f))));
    TEST_ASSERT_EQUAL(
        static_cast<int>(PowerCalibration::SaveResult::InvalidInput),
        static_cast<int>(calibration.captureCurrent(
            storage, Amps(10.0f), Amps(4.9f))));
    TEST_ASSERT_EQUAL(
        static_cast<int>(PowerCalibration::SaveResult::InvalidInput),
        static_cast<int>(calibration.captureCurrent(
            storage, Amps(5.0f), Amps(10.1f))));
    TEST_ASSERT_TRUE(storage.operations.empty());
    assertIdentity(calibration);
}

static void test_failed_verification_preserves_active_record()
{
    FakeStorage storage;
    storage.seed(record(0.2f, -0.1f, 1.1f));
    PowerCalibration calibration;
    TEST_ASSERT_TRUE(calibration.load(storage));
    storage.operations.clear();
    storage.shouldCorruptVerification = true;

    TEST_ASSERT_EQUAL(
        static_cast<int>(PowerCalibration::SaveResult::VerificationFailed),
        static_cast<int>(calibration.captureVoltage(
            storage,
            Volts(25.0f), Volts(12.0f),
            Volts(25.5f), Volts(11.75f))));
    TEST_ASSERT_EQUAL_FLOAT(0.2f, calibration.packVoltageOffset().value());
    TEST_ASSERT_EQUAL_FLOAT(-0.1f, calibration.midpointVoltageOffset().value());
    TEST_ASSERT_EQUAL_FLOAT(1.1f, calibration.packShuntScale());
    TEST_ASSERT_EQUAL_HEX8(
        EEPROM_POWER_CAL_INVALID_MAGIC, storage.bytes[0]);
}

static void test_plausible_but_different_readback_fails_verification()
{
    FakeStorage storage;
    PowerCalibration calibration;
    storage.shouldSubstitutePlausibleRecord = true;

    TEST_ASSERT_EQUAL(
        static_cast<int>(PowerCalibration::SaveResult::VerificationFailed),
        static_cast<int>(calibration.captureVoltage(
            storage,
            Volts(25.0f), Volts(12.0f),
            Volts(25.5f), Volts(11.75f))));
    assertIdentity(calibration);
    TEST_ASSERT_EQUAL_HEX8(
        EEPROM_POWER_CAL_INVALID_MAGIC, storage.bytes[0]);
}

static void test_each_interrupted_write_preserves_active_calibration()
{
    for (int failure = 0;
         failure <= static_cast<int>(sizeof(PowerCalibrationRecord));
         ++failure)
    {
        FakeStorage storage;
        storage.seed(record(0.2f, -0.1f, 1.1f));
        PowerCalibration calibration;
        TEST_ASSERT_TRUE(calibration.load(storage));
        storage.operations.clear();
        storage.failBeforeWrite = failure;

        bool wasInterrupted = false;
        try
        {
            calibration.captureVoltage(
                storage,
                Volts(25.0f), Volts(12.0f),
                Volts(25.5f), Volts(11.75f));
        }
        catch (const std::runtime_error &)
        {
            wasInterrupted = true;
        }
        TEST_ASSERT_TRUE(wasInterrupted);
        TEST_ASSERT_EQUAL_FLOAT(0.2f, calibration.packVoltageOffset().value());
        TEST_ASSERT_EQUAL_FLOAT(-0.1f, calibration.midpointVoltageOffset().value());
        TEST_ASSERT_EQUAL_FLOAT(1.1f, calibration.packShuntScale());

        PowerCalibration reloaded;
        if (failure == 0)
            TEST_ASSERT_TRUE(reloaded.load(storage));
        else
            TEST_ASSERT_FALSE(reloaded.load(storage));
    }
}

static void test_measurement_correction_preserves_input()
{
    for (int configured = 0; configured < 2; ++configured)
    {
        PowerCalibration calibration;
        FakeStorage storage;
        if (configured)
        {
            storage.seed(record(0.5f, -0.25f, 1.25f));
            TEST_ASSERT_TRUE(calibration.load(storage));
        }
        for (int available = 0; available < 2; ++available)
        {
            PowerMonitorMeasurement raw;
            raw.isValid = available;
            raw.voltage = Volts(12.0f);
            raw.current = Amps(-8.0f);
            raw.power = Watts(16.0f);
            raw.charge = Coulombs(-24.0f);
            const auto pack = calibration.applyPackCalibration(raw);
            const auto midpoint = calibration.applyMidpointCalibration(raw);
            TEST_ASSERT_EQUAL(available, raw.isValid);
            TEST_ASSERT_EQUAL(available, pack.isValid);
            TEST_ASSERT_EQUAL(available, midpoint.isValid);
            TEST_ASSERT_EQUAL_FLOAT(configured ? 12.5f : 12.0f, pack.voltage.value());
            TEST_ASSERT_EQUAL_FLOAT(configured ? -10.0f : -8.0f, pack.current.value());
            TEST_ASSERT_EQUAL_FLOAT(configured ? 20.0f : 16.0f, pack.power.value());
            TEST_ASSERT_EQUAL_FLOAT(configured ? -30.0f : -24.0f, pack.charge.value());
            TEST_ASSERT_EQUAL_FLOAT(configured ? 11.75f : 12.0f, midpoint.voltage.value());
            TEST_ASSERT_EQUAL_FLOAT(-8.0f, midpoint.current.value());
            TEST_ASSERT_EQUAL_FLOAT(16.0f, midpoint.power.value());
            TEST_ASSERT_EQUAL_FLOAT(-24.0f, midpoint.charge.value());
            TEST_ASSERT_EQUAL_FLOAT(12.0f, raw.voltage.value());
            TEST_ASSERT_EQUAL_FLOAT(-8.0f, raw.current.value());
            TEST_ASSERT_EQUAL_FLOAT(16.0f, raw.power.value());
            TEST_ASSERT_EQUAL_FLOAT(-24.0f, raw.charge.value());
        }
    }
}

static void test_measurement_correction_preserves_nonfinite_values()
{
    PowerCalibration calibration;
    FakeStorage storage;
    storage.seed(record(0.5f, -0.25f, 1.25f));
    TEST_ASSERT_TRUE(calibration.load(storage));
    const float values[] = {NAN, INFINITY, -INFINITY};
    for (float value : values)
    {
        for (int available = 0; available < 2; ++available)
        {
            PowerMonitorMeasurement raw;
            raw.isValid = available;
            raw.voltage = Volts(value);
            raw.current = Amps(value);
            raw.power = Watts(value);
            raw.charge = Coulombs(value);
            const PowerMonitorMeasurement results[] = {
                calibration.applyPackCalibration(raw),
                calibration.applyMidpointCalibration(raw), raw};
            for (const auto &result : results)
            {
                TEST_ASSERT_EQUAL(available, result.isValid);
                const float fields[] = {result.voltage.value(), result.current.value(),
                    result.power.value(), result.charge.value()};
                for (float field : fields)
                {
                    if (isnan(value)) TEST_ASSERT_TRUE(isnan(field));
                    else TEST_ASSERT_TRUE(field == value);
                }
            }
        }
    }
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_measurement_correction_preserves_input);
    RUN_TEST(test_measurement_correction_preserves_nonfinite_values);
    RUN_TEST(test_layout_and_identity_are_stable);
    RUN_TEST(test_load_accepts_valid_record_and_rejects_invalid_records);
    RUN_TEST(test_load_rejects_each_nonfinite_field_and_accepts_exact_bounds);
    RUN_TEST(test_voltage_capture_persists_then_activates_both_offsets);
    RUN_TEST(test_invalid_voltage_capture_preserves_active_record_without_writing);
    RUN_TEST(test_voltage_capture_rejects_each_invalid_input);
    RUN_TEST(test_voltage_capture_accepts_signed_exact_bounds);
    RUN_TEST(test_bad_voltage_measurement_preserves_both_offsets);
    RUN_TEST(test_current_capture_applies_one_scale_to_all_shunt_measurements);
    RUN_TEST(test_invalid_current_capture_preserves_active_record);
    RUN_TEST(test_current_capture_accepts_signed_and_exact_scale_bounds);
    RUN_TEST(test_current_capture_rejects_nonfinite_opposite_and_small_currents);
    RUN_TEST(test_failed_verification_preserves_active_record);
    RUN_TEST(test_plausible_but_different_readback_fails_verification);
    RUN_TEST(test_each_interrupted_write_preserves_active_calibration);
    return UNITY_END();
}
