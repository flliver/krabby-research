#include "unity.h"

#include "EEPROM.h"
#include "actuator_manager.h"
#include "hall_sim.hpp"
#include "sim_clock.hpp"
#include "src/controller/board_role.h"

#include <cmath>
#include <cstring>

namespace {

constexpr uint8_t EEPROM_ROLE_MAGIC = 0xAB;

LinearActuator makeActuator()
{
    return LinearActuator("FLHY", 0, 1, 2, 3, 4, 0);
}

// Six actuators on board 0 of the host sim (a quadrature flag per board).
struct HostBoard
{
    LinearActuator storage[6] = {
        makeActuator(), makeActuator(), makeActuator(),
        makeActuator(), makeActuator(), makeActuator()};
    LinearActuator *acts[6];
    ActuatorManager manager{acts, 6};
    Print out;

    HostBoard()
    {
        for (int i = 0; i < 6; ++i) acts[i] = &storage[i];
        manager.setCalibrationOutput(&out, ROLE_FRONT);
    }
};

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

static void test_role_magic_round_trip_at_bytes_32_33()
{
    EEPROM.update(EEPROM_ROLE_ADDR, EEPROM_ROLE_MAGIC);
    EEPROM.update(EEPROM_ROLE_ADDR + 1, static_cast<uint8_t>(ROLE_LEFT));

    TEST_ASSERT_EQUAL_HEX8(EEPROM_ROLE_MAGIC, EEPROM.read(EEPROM_ROLE_ADDR));
    TEST_ASSERT_EQUAL_INT(ROLE_LEFT, EEPROM.read(EEPROM_ROLE_ADDR + 1));
}

// The record is the Mega's 30-byte v2 layout on every build, little-endian from
// address 0: six int16 minimums, six int16 maximums, the magic at 24, the sensor
// bits at 26 and the check at 28. The role record stays at 32.
static void test_cal_data_layout_is_the_30_byte_avr_record()
{
    TEST_ASSERT_EQUAL_UINT(EEPROM_ACTUATOR_CAL_SIZE, sizeof(ActuatorManager::CalData));
    TEST_ASSERT_EQUAL_UINT(30, sizeof(ActuatorManager::CalData));
    TEST_ASSERT_EQUAL_UINT(24, offsetof(ActuatorManager::CalData, magic));
    TEST_ASSERT_EQUAL_UINT(26, offsetof(ActuatorManager::CalData, sensorBits));
    TEST_ASSERT_EQUAL_UINT(28, offsetof(ActuatorManager::CalData, check));
    TEST_ASSERT_TRUE(EEPROM_ACTUATOR_CAL_ADDR + sizeof(ActuatorManager::CalData) <= EEPROM_ROLE_ADDR);
    TEST_ASSERT_EQUAL_UINT16(32, EEPROM_ROLE_ADDR);
    // 16-bit values, so the Mega stores and compares exactly these.
    TEST_ASSERT_EQUAL_HEX16(0xCA2B, ActuatorManager::CAL_DATA_MAGIC);
    TEST_ASSERT_EQUAL_HEX16(0xCA1B, ActuatorManager::CAL_DATA_MAGIC_V1);

    ActuatorManager::CalData saved{};
    saved.magic = ActuatorManager::CAL_DATA_MAGIC;
    saved.minVals[0] = 0x0102;
    saved.maxVals[5] = 1023;
    saved.sensorBits = 0x0921;
    saved.check = ActuatorManager::calDataCheck(saved);
    EEPROM.put(0, saved);
    TEST_ASSERT_EQUAL_HEX8(0x02, EEPROM.read(0));
    TEST_ASSERT_EQUAL_HEX8(0x01, EEPROM.read(1));
    TEST_ASSERT_EQUAL_HEX8(0xFF, EEPROM.read(22));  // maxVals[5] = 0x03FF
    TEST_ASSERT_EQUAL_HEX8(0x03, EEPROM.read(23));
    TEST_ASSERT_EQUAL_HEX8(0x2B, EEPROM.read(24));  // magic
    TEST_ASSERT_EQUAL_HEX8(0xCA, EEPROM.read(25));
    TEST_ASSERT_EQUAL_HEX8(0x21, EEPROM.read(26));  // sensor bits
    TEST_ASSERT_EQUAL_HEX8(0x09, EEPROM.read(27));
    // (sum of bytes 0..27) ^ 0x5A5A: 0x02 + 0x01 + 0xFF + 0x03 + 0x2B + 0xCA + 0x21 + 0x09.
    const uint16_t expected = static_cast<uint16_t>(0x02 + 0x01 + 0xFF + 0x03 + 0x2B + 0xCA + 0x21 + 0x09) ^ 0x5A5A;
    TEST_ASSERT_EQUAL_HEX8(expected & 0xFF, EEPROM.read(28));
    TEST_ASSERT_EQUAL_HEX8(expected >> 8, EEPROM.read(29));
    TEST_ASSERT_EQUAL_HEX8(0, EEPROM.read(30));
}

// A record whose magic is neither v2 nor v1 (erased EEPROM, or the old
// routine's 0xBEEF / 0xDEADBEEF) leaves the defaults in place.
static void test_load_ignores_records_without_the_magic()
{
    HostBoard board;
    for (const uint16_t magic : {static_cast<uint16_t>(0xFFFF), static_cast<uint16_t>(0xBEEF)})
    {
        ActuatorManager::CalData stale{};
        stale.magic = magic;
        for (int i = 0; i < 6; ++i)
        {
            stale.minVals[i] = 400;
            stale.maxVals[i] = 500;
        }
        EEPROM.put(0, stale);
        board.manager.loadCalibration();
        for (auto *actuator : board.acts)
        {
            TEST_ASSERT_EQUAL_INT(0, actuator->minStop);
            TEST_ASSERT_EQUAL_INT(1023, actuator->maxStop);
            TEST_ASSERT_EQUAL_INT(static_cast<int>(JointSensor::Pot), static_cast<int>(actuator->sensor));
        }
    }
    TEST_ASSERT_NOT_NULL(std::strstr(board.out.output.c_str(), "No calibration record: defaults\n"));
}

static void test_cal_data_put_get_round_trip_magic()
{
    ActuatorManager::CalData saved{};
    saved.magic = ActuatorManager::CAL_DATA_MAGIC;
    for (int i = 0; i < 6; ++i)
    {
        saved.minVals[i] = 100 + i;
        saved.maxVals[i] = 800 + i;
    }
    saved.sensorBits = 0x0555;
    saved.check = ActuatorManager::calDataCheck(saved);

    EEPROM.put(0, saved);

    ActuatorManager::CalData loaded{};
    EEPROM.get(0, loaded);
    TEST_ASSERT_EQUAL_HEX16(ActuatorManager::CAL_DATA_MAGIC, loaded.magic);
    TEST_ASSERT_EQUAL_HEX16(0x0555, loaded.sensorBits);
    TEST_ASSERT_EQUAL_HEX16(ActuatorManager::calDataCheck(saved), loaded.check);
    for (int i = 0; i < 6; ++i)
    {
        TEST_ASSERT_EQUAL_INT(100 + i, loaded.minVals[i]);
        TEST_ASSERT_EQUAL_INT(800 + i, loaded.maxVals[i]);
    }
}

// Without quadrature halls (the sim default) every joint is a pot.
static void test_actuator_manager_save_and_load_calibration()
{
    HostBoard board;
    board.manager.initAll();
    for (int i = 0; i < 6; ++i)
    {
        board.acts[i]->minStop = 110 + i;
        board.acts[i]->maxStop = 810 + i;
    }
    board.manager.saveCalibration();
    TEST_ASSERT_NOT_NULL(std::strstr(board.out.output.c_str(), "Limits saved\n"));

    ActuatorManager::CalData raw{};
    EEPROM.get(0, raw);
    TEST_ASSERT_EQUAL_HEX16(ActuatorManager::CAL_DATA_MAGIC, raw.magic);
    TEST_ASSERT_EQUAL_HEX16(0x0555, raw.sensorBits);  // six pots
    TEST_ASSERT_EQUAL_HEX16(ActuatorManager::calDataCheck(raw), raw.check);

    for (auto *actuator : board.acts)
    {
        actuator->minStop = 0;
        actuator->maxStop = 1023;
    }
    board.manager.loadCalibration();
    for (int i = 0; i < 6; ++i)
    {
        TEST_ASSERT_EQUAL_INT(110 + i, board.acts[i]->minStop);
        TEST_ASSERT_EQUAL_INT(810 + i, board.acts[i]->maxStop);
        TEST_ASSERT_TRUE(board.acts[i]->encFlags & ACTUATOR_FLAG_LIMITS_VALID);
    }
    TEST_ASSERT_NOT_NULL(std::strstr(board.out.output.c_str(), "Calibration v2 loaded\n"));
}

// A v1 record (26 bytes, magic 0xCA1B at 24) is migrated as pots without
// quadrature halls and ignored with them.
static void test_v1_record_migrates_without_quadrature_and_is_ignored_with_it()
{
    ActuatorManager::CalData v1{};
    v1.magic = ActuatorManager::CAL_DATA_MAGIC_V1;
    for (int i = 0; i < 6; ++i)
    {
        v1.minVals[i] = 120 + i;
        v1.maxVals[i] = 820 + i;
    }
    EEPROM.put(0, v1);

    HostBoard pots;
    pots.manager.initAll();
    pots.manager.loadCalibration();
    TEST_ASSERT_EQUAL_INT(121, pots.acts[1]->minStop);
    TEST_ASSERT_EQUAL_INT(821, pots.acts[1]->maxStop);
    TEST_ASSERT_NOT_NULL(std::strstr(pots.out.output.c_str(), "Calibration v1 loaded\n"));

    krabby::sim::hallSimSetQuadrature(0, true);
    HostBoard halls;
    halls.manager.initAll();
    halls.manager.loadCalibration();
    for (auto *actuator : halls.acts)
    {
        TEST_ASSERT_EQUAL_INT(static_cast<int>(JointSensor::Unknown), static_cast<int>(actuator->sensor));
        TEST_ASSERT_EQUAL_INT(0, actuator->minStop);
        TEST_ASSERT_EQUAL_INT(0, actuator->maxStop);
        TEST_ASSERT_TRUE(std::isnan(actuator->getPos()));
    }
    TEST_ASSERT_NOT_NULL(std::strstr(halls.out.output.c_str(), "Calibration v1 ignored: run C\n"));
}

// On a quadrature board the kinds survive a save and load; a hall stroke reloads
// without its home, so the joint has no position until a curl homes it.
static void test_quadrature_board_round_trips_kinds_and_strokes()
{
    krabby::sim::hallSimSetQuadrature(0, true);
    HostBoard board;
    board.manager.initAll();
    board.acts[1]->setSensor(JointSensor::Hall);
    board.acts[1]->minStop = 0;
    board.acts[1]->maxStop = CAL_HIP_SWEEP_EDGES;  // a hall hip's fixed sweep
    board.acts[2]->setSensor(JointSensor::Hall);
    board.acts[2]->minStop = -700;
    board.acts[2]->maxStop = 0;
    board.acts[4]->setSensor(JointSensor::Pot);
    board.acts[4]->minStop = 200;
    board.acts[4]->maxStop = 800;
    board.manager.saveCalibration();

    HostBoard reloaded;
    reloaded.manager.initAll();
    reloaded.manager.loadCalibration();
    TEST_ASSERT_EQUAL_INT(static_cast<int>(JointSensor::Hall), static_cast<int>(reloaded.acts[1]->sensor));
    TEST_ASSERT_EQUAL_INT(CAL_HIP_SWEEP_EDGES, reloaded.acts[1]->maxStop);
    TEST_ASSERT_EQUAL_INT(-700, reloaded.acts[2]->minStop);
    TEST_ASSERT_EQUAL_INT(static_cast<int>(JointSensor::Pot), static_cast<int>(reloaded.acts[4]->sensor));
    TEST_ASSERT_EQUAL_INT(200, reloaded.acts[4]->minStop);
    TEST_ASSERT_EQUAL_INT(static_cast<int>(JointSensor::Unknown), static_cast<int>(reloaded.acts[0]->sensor));
    TEST_ASSERT_EQUAL_CHAR('e', reloaded.acts[1]->sensorStateCode());
    TEST_ASSERT_TRUE(std::isnan(reloaded.acts[1]->getPos()));
    reloaded.acts[1]->setHome();
    TEST_ASSERT_EQUAL_CHAR('E', reloaded.acts[1]->sensorStateCode());
    TEST_ASSERT_EQUAL_FLOAT(0.0f, reloaded.acts[1]->getPos());
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_eeprom_read_write_update_round_trip_bytes);
    RUN_TEST(test_role_magic_round_trip_at_bytes_32_33);
    RUN_TEST(test_cal_data_layout_is_the_30_byte_avr_record);
    RUN_TEST(test_load_ignores_records_without_the_magic);
    RUN_TEST(test_cal_data_put_get_round_trip_magic);
    RUN_TEST(test_actuator_manager_save_and_load_calibration);
    RUN_TEST(test_v1_record_migrates_without_quadrature_and_is_ignored_with_it);
    RUN_TEST(test_quadrature_board_round_trips_kinds_and_strokes);
    return UNITY_END();
}
