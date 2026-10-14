#include "unity.h"

#include "host_api.h"
#include "host_display.hpp"
#include "sim_ssd1306.hpp"

using krabby::sim::host::hostDisplayInvalidate;
using krabby::sim::host::hostDisplaySetup;
using krabby::sim::host::hostDisplayStep;
using krabby::sim::host::hostOledBufferCrc32;
using krabby::sim::host::hostOledGlyphForActuator;
using krabby::sim::simSsd1306;

void setUp() { krabby_host_reset(); }

void tearDown() {}

static void renderOledAtMs(unsigned long ms)
{
    krabby_host_advance_millis(ms);
    hostDisplayStep(krabby_host_millis());
}

static void stageConnectedFrontActuators()
{
    static const char *kIds[] = {"FLHY", "FLHL", "FLKL", "FRHY", "FRHL", "FRKL"};
    for (const char *id : kIds)
    {
        krabby_host_set_actuator_stops(id, 100, 900);
    }
    for (int potPin = 0; potPin < 6; ++potPin)
    {
        krabby_host_set_analog_read(potPin, 500);
        krabby_host_set_analog_read(potPin + 6, 12);
    }
}

static void test_jog_changes_actuator_glyph_and_wire_buffer()
{
    stageConnectedFrontActuators();
    krabby_host_firmware_setup();
    krabby_host_update_all();
    hostDisplaySetup();

    renderOledAtMs(250);
    const ActuatorGlyph idleGlyph = hostOledGlyphForActuator(ActuatorId::FLHY);
    TEST_ASSERT_NOT_EQUAL(ActuatorGlyph::Extend, idleGlyph);
    const uint32_t idleCrc = hostOledBufferCrc32();
    TEST_ASSERT_GREATER_THAN(0, simSsd1306().totalDataBytesReceived());

    krabby_host_handle_jog("FLHY", 100);
    krabby_host_update_all();
    hostDisplayInvalidate();
    renderOledAtMs(250);

    TEST_ASSERT_EQUAL_INT(ActuatorGlyph::Extend, hostOledGlyphForActuator(ActuatorId::FLHY));
    TEST_ASSERT_NOT_EQUAL(idleCrc, hostOledBufferCrc32());
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_jog_changes_actuator_glyph_and_wire_buffer);
    return UNITY_END();
}
