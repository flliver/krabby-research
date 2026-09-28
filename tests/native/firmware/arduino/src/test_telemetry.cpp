#include <iomanip>
#include <sstream>
#include <string>

#include "src/imu/imu_constants.h"
#include "src/imu/imu_calibrator.h"
#include "src/telemetry.h"
#include "unity.h"

struct FakeOutput
{
    std::ostringstream text;
    void print(char value) { text << value; }
    void print(const char *value) { text << value; }
    void print(int value) { text << value; }
    void print(float value, int digits)
    {
        text << std::fixed << std::setprecision(digits) << value;
    }
    std::string str() const { return text.str(); }
};






void setUp() {}
void tearDown() {}








static void test_invalid_measurement_serializes_zero_values_and_valid_false()
{
    FakeOutput output;
    appendImuMeasurement(
        output, ImuMeasurement{false});
    TEST_ASSERT_EQUAL_STRING(
        ";IMU 0.000 0.000 0.000 0.0000 0.0000 0.0000 0.0 0",
        output.str().c_str());
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_invalid_measurement_serializes_zero_values_and_valid_false);
    return UNITY_END();
}
