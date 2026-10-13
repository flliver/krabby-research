// Host OLED: shipping DisplayRenderer + buildDisplayFrame only. We do not link
// Ssd1306Adapter or SparkFun_Qwiic_OLED here; see firmware/sim/I2C_SIM_SCOPE.md.

#include "host_display.hpp"

#include "host_firmware_internal.hpp"
#include "host_leader_imu.hpp"
#include "host_oled_canvas.hpp"
#include "sim_ssd1306.hpp"

#include "src/controller/board_role.h"
#include "src/controller/controller_freshness_tracker.h"
#include "src/display/display_frame_model.h"
#include "src/display/display_renderer.h"

namespace krabby {
namespace sim {
namespace host {

namespace {

constexpr unsigned long kOledRedrawIntervalMs = 250;
constexpr int kMoveThreshold = 20;

HostOledCanvas g_canvas;
DisplayRenderer<HostOledCanvas> g_renderer(g_canvas);
ControllerFreshnessTracker g_controllerFreshness[BOARD_ROLE_COUNT];
ActuatorStatus g_actuatorStatus[ActuatorId::ActuatorCount];
unsigned long g_lastOledDrawMs = 0;
bool g_setupDone = false;

void seedControllerFreshness(unsigned long nowMs)
{
    for (BoardRole role : ALL_BOARD_ROLES)
    {
        g_controllerFreshness[role] = ControllerFreshnessTracker{};
    }
    g_controllerFreshness[ROLE_FRONT] = ControllerFreshnessTracker::seenAt(nowMs);
}

void fillActuatorStatusFromHost()
{
    for (ActuatorId actuatorId = ActuatorId::FLHY;
         actuatorId < ActuatorId::ActuatorCount;
         ++actuatorId)
    {
        g_actuatorStatus[actuatorId] = ActuatorStatus{};
        g_actuatorStatus[actuatorId].actuatorId = actuatorId;
        g_actuatorStatus[actuatorId].connectionState = ActuatorConnection::Disconnected;
    }

    static const char *kFrontActuatorNames[] = {
        "FLHY", "FLHL", "FLKL", "FRHY", "FRHL", "FRKL",
    };
    for (const char *name : kFrontActuatorNames)
    {
        LinearActuator *actuator = krabby::sim::host::find_actuator(name);
        if (actuator == nullptr) continue;
        const ActuatorStatus status = actuator->getStatus();
        if (status.actuatorId < ActuatorId::ActuatorCount)
        {
            g_actuatorStatus[status.actuatorId] = status;
        }
    }
}

}  // namespace

void hostDisplaySetup()
{
    if (g_setupDone) return;
    g_renderer.initialize();
    g_lastOledDrawMs = 0;
    g_setupDone = true;
}

void hostDisplayReset()
{
    g_setupDone = false;
    g_lastOledDrawMs = 0;
    g_renderer.invalidate();
}

void hostDisplayInvalidate() { g_renderer.invalidate(); }

uint32_t hostOledBufferCrc32() { return simSsd1306().bufferCrc32(); }

ActuatorGlyph hostOledGlyphForActuator(ActuatorId actuatorId)
{
    return g_canvas.lastGlyphForActuator(actuatorId);
}

void hostDisplayStep(unsigned long nowMs)
{
    if (!g_setupDone) hostDisplaySetup();
    if (nowMs - g_lastOledDrawMs < kOledRedrawIntervalMs) return;

    seedControllerFreshness(nowMs);
    fillActuatorStatusFromHost();

    const ImuMeasurement imu = leaderImu().measure();
    const DisplayFrame frame = buildDisplayFrame(
        ROLE_FRONT,
        g_controllerFreshness,
        g_actuatorStatus,
        imu,
        static_cast<uint32_t>(nowMs),
        kMoveThreshold);

    for (ActuatorId actuatorId = ActuatorId::FLHY;
         actuatorId < ActuatorId::ActuatorCount;
         ++actuatorId)
    {
        g_canvas.setLastGlyphForActuator(actuatorId, frame.actuators[actuatorId]);
    }

    if (g_renderer.render(frame))
    {
        g_lastOledDrawMs = nowMs;
    }
}

}  // namespace host
}  // namespace sim
}  // namespace krabby
