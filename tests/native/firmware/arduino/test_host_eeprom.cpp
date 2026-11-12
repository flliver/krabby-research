#include "unity.h"

#include "EEPROM.h"
#include "eeprom_storage.h"
#include "actuator_manager.h"
#include "sim_clock.hpp"
#include "src/controller/board_role.h"

namespace {

constexpr uint8_t EEPROM_ROLE_MAGIC = 0xAB;

LinearActuator makeActuator()
{
    return LinearActuator("FLHY", 0, 1, 2, 3, 4, 0);
}

}  // namespace

void setUp() { krabby::sim::resetHostSim(); }
void tearDown() {}

static void test_eeprom_read_write_update_round_trip_bytes()
{
    EEPROM.write(10, 0x12);
    TEST_ASSERT_EQUAL_HEX8(0x12, EEPROM.read(10));

    EEPROM.update(10, 0x12);
    TEST_ASSERT_EQUAL_UINT(1, EEPROM.writes);
    EEPROM.update(10, 0x34);
    TEST_ASSERT_EQUAL_HEX8(0x34, EEPROM.read(10));
    TEST_ASSERT_EQUAL_UINT(2, EEPROM.writes);
}

static void test_record_storage_isolates_addresses_and_commits_only_magic()
{
    struct Record
    {
        uint8_t schema;
        uint8_t magic;
        uint8_t value;
    };
    EepromStorage<Record, 100> first;
    EepromStorage<Record, 200> second;
    first.writeRecord({1, 0, 42});
    second.writeRecord({2, 0xBC, 99});
    first.updateMagic(0xAB);

    Record loaded{};
    first.load(loaded);
    TEST_ASSERT_EQUAL_UINT8(1, loaded.schema);
    TEST_ASSERT_EQUAL_HEX8(0xAB, loaded.magic);
    TEST_ASSERT_EQUAL_UINT8(42, loaded.value);
    second.load(loaded);
    TEST_ASSERT_EQUAL_UINT8(2, loaded.schema);
    TEST_ASSERT_EQUAL_HEX8(0xBC, loaded.magic);
    TEST_ASSERT_EQUAL_UINT8(99, loaded.value);
}

static void test_role_magic_round_trip_at_bytes_32_33()
{
    EEPROM.update(EEPROM_ROLE_ADDR, EEPROM_ROLE_MAGIC);
    EEPROM.update(EEPROM_ROLE_ADDR + 1, static_cast<uint8_t>(ROLE_LEFT));

    TEST_ASSERT_EQUAL_HEX8(EEPROM_ROLE_MAGIC, EEPROM.read(EEPROM_ROLE_ADDR));
    TEST_ASSERT_EQUAL_INT(ROLE_LEFT, EEPROM.read(EEPROM_ROLE_ADDR + 1));
}

static void test_cal_data_put_get_round_trip_deadbeef_magic()
{
    ActuatorManager::CalData saved{};
    saved.magic = 0xDEADBEEF;
    for (int i = 0; i < 6; ++i)
    {
        saved.minVals[i] = 100 + i;
        saved.maxVals[i] = 800 + i;
    }

    EEPROM.put(0, saved);

    ActuatorManager::CalData loaded{};
    EEPROM.get(0, loaded);
    TEST_ASSERT_EQUAL_HEX32(0xDEADBEEF, static_cast<unsigned>(loaded.magic));
    for (int i = 0; i < 6; ++i)
    {
        TEST_ASSERT_EQUAL_INT(100 + i, loaded.minVals[i]);
        TEST_ASSERT_EQUAL_INT(800 + i, loaded.maxVals[i]);
    }
}

static void test_actuator_manager_save_and_load_calibration()
{
    LinearActuator storage[] = {
        makeActuator(), makeActuator(), makeActuator(),
        makeActuator(), makeActuator(), makeActuator()};
    LinearActuator *acts[6];
    for (int i = 0; i < 6; ++i)
    {
        acts[i] = &storage[i];
        acts[i]->minStop = 110 + i;
        acts[i]->maxStop = 810 + i;
    }

    ActuatorManager manager(acts, 6);
    manager.saveCalibration();

    ActuatorManager::CalData raw{};
    EEPROM.get(0, raw);
    TEST_ASSERT_EQUAL_HEX32(0xDEADBEEF, static_cast<unsigned>(raw.magic));

    for (auto *actuator : acts)
    {
        actuator->minStop = 0;
        actuator->maxStop = 1023;
    }
    manager.loadCalibration();
    for (int i = 0; i < 6; ++i)
    {
        TEST_ASSERT_EQUAL_INT(110 + i, acts[i]->minStop);
        TEST_ASSERT_EQUAL_INT(810 + i, acts[i]->maxStop);
    }
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_eeprom_read_write_update_round_trip_bytes);
    RUN_TEST(test_record_storage_isolates_addresses_and_commits_only_magic);
    RUN_TEST(test_role_magic_round_trip_at_bytes_32_33);
    RUN_TEST(test_cal_data_put_get_round_trip_deadbeef_magic);
    RUN_TEST(test_actuator_manager_save_and_load_calibration);
    return UNITY_END();
}
