#include <string.h>
#include <vector>

#include "unity.h"

#include "src/power_safety/power_safety_latch.h"

struct FakeStorage
{
    enum class Operation { Load, WriteRecord, UpdateMagic };

    PowerSafetyLatchRecord record = {0xFF, 0xFF, 0xFF};
    std::vector<Operation> operations;
    bool corruptVerification = false;

    void load(PowerSafetyLatchRecord &destination)
    {
        operations.push_back(Operation::Load);
        destination = record;
        if (corruptVerification && operations.size() > 1)
            destination.value = 99;
    }

    void writeRecord(const PowerSafetyLatchRecord &source)
    {
        operations.push_back(Operation::WriteRecord);
        record = source;
    }

    void updateMagic(uint8_t magic)
    {
        operations.push_back(Operation::UpdateMagic);
        record.magic = magic;
    }
};

void setUp() {}
void tearDown() {}

static PowerSafetyLatchRecord validRecord(PowerSafetyLatchValue value)
{
    const PowerSafetyLatchRecord record = {
        EEPROM_POWER_SAFETY_MAGIC,
        EEPROM_POWER_SAFETY_SCHEMA,
        static_cast<uint8_t>(value),
    };
    return record;
}

static void test_erased_storage_is_an_empty_normal_boot()
{
    FakeStorage storage;
    PowerSafetyLatch latch;
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyLatchLoadResult::Empty, latch.load(storage));
    TEST_ASSERT_EQUAL_UINT8(PowerSafetyLatchValue::None, latch.value());
}

static void test_valid_values_load()
{
    const PowerSafetyLatchValue values[] = {
        PowerSafetyLatchValue::None,
        PowerSafetyLatchValue::Sleep,
    };
    for (const PowerSafetyLatchValue value : values)
    {
        FakeStorage storage;
        storage.record = validRecord(value);
        PowerSafetyLatch latch;
        TEST_ASSERT_EQUAL_UINT8(
            PowerSafetyLatchLoadResult::Loaded, latch.load(storage));
        TEST_ASSERT_EQUAL_UINT8(value, latch.value());
    }
}

static void test_any_non_erased_invalid_record_fails_safe()
{
    PowerSafetyLatchRecord records[] = {
        {EEPROM_POWER_SAFETY_INVALID_MAGIC, EEPROM_POWER_SAFETY_SCHEMA, 0},
        {EEPROM_POWER_SAFETY_MAGIC, 0, 0},
        {EEPROM_POWER_SAFETY_MAGIC, EEPROM_POWER_SAFETY_SCHEMA, 99},
        {0xFF, 0xFF, 0},
    };
    for (const PowerSafetyLatchRecord record : records)
    {
        FakeStorage storage;
        storage.record = record;
        PowerSafetyLatch latch;
        TEST_ASSERT_EQUAL_UINT8(
            PowerSafetyLatchLoadResult::Corrupt, latch.load(storage));
        TEST_ASSERT_EQUAL_UINT8(
            PowerSafetyLatchValue::Sleep, latch.value());
    }
}

static void test_save_is_invalid_first_magic_last_and_verified()
{
    FakeStorage storage;
    PowerSafetyLatch latch;
    latch.load(storage);
    storage.operations.clear();

    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyLatch::SaveResult::Saved,
        latch.save(storage, PowerSafetyLatchValue::Sleep));
    TEST_ASSERT_EQUAL_UINT(3, storage.operations.size());
    TEST_ASSERT_EQUAL_INT(
        static_cast<int>(FakeStorage::Operation::WriteRecord),
        static_cast<int>(storage.operations[0]));
    TEST_ASSERT_EQUAL_INT(
        static_cast<int>(FakeStorage::Operation::UpdateMagic),
        static_cast<int>(storage.operations[1]));
    TEST_ASSERT_EQUAL_INT(
        static_cast<int>(FakeStorage::Operation::Load),
        static_cast<int>(storage.operations[2]));
    TEST_ASSERT_EQUAL_HEX8(EEPROM_POWER_SAFETY_MAGIC, storage.record.magic);
}

static void test_unchanged_valid_value_does_not_write()
{
    FakeStorage storage;
    storage.record = validRecord(PowerSafetyLatchValue::Sleep);
    PowerSafetyLatch latch;
    latch.load(storage);
    storage.operations.clear();

    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyLatch::SaveResult::Unchanged,
        latch.save(storage, PowerSafetyLatchValue::Sleep));
    TEST_ASSERT_TRUE(storage.operations.empty());
}

static void test_verification_failure_invalidates_the_record()
{
    FakeStorage storage;
    PowerSafetyLatch latch;
    latch.load(storage);
    storage.operations.clear();
    storage.corruptVerification = true;

    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyLatch::SaveResult::VerificationFailed,
        latch.save(storage, PowerSafetyLatchValue::Sleep));
    TEST_ASSERT_EQUAL_HEX8(
        EEPROM_POWER_SAFETY_INVALID_MAGIC, storage.record.magic);
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_erased_storage_is_an_empty_normal_boot);
    RUN_TEST(test_valid_values_load);
    RUN_TEST(test_any_non_erased_invalid_record_fails_safe);
    RUN_TEST(test_save_is_invalid_first_magic_last_and_verified);
    RUN_TEST(test_unchanged_valid_value_does_not_write);
    RUN_TEST(test_verification_failure_invalidates_the_record);
    return UNITY_END();
}
