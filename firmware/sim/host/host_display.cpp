// Host OLED: shipping DisplayRenderer + buildDisplayFrame only. We do not link
// Ssd1306Adapter or SparkFun_Qwiic_OLED here; see firmware/sim/I2C_SIM_SCOPE.md.

#include "host_display.hpp"

#include "host_battery_telemetry.hpp"
#include "host_firmware_internal.hpp"
#include "host_leader_imu.hpp"
#include "host_multi_board.hpp"
#include "host_oled_canvas.hpp"
#include "sim_ssd1306.hpp"

#include "actuator_manager.h"
#include "src/controller/board_role.h"
#include "src/controller/controller_freshness_tracker.h"
#include "src/display/display_constants.h"
#include "src/display/display_frame_model.h"
#include "src/display/display_renderer.h"
#include "src/telemetry.h"

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
DisplayFrame g_lastRenderedFrame{};

void resetActuatorStatusDefaults()
{
    for (ActuatorId actuatorId = ActuatorId::FLHY;
         actuatorId < ActuatorId::ActuatorCount;
         ++actuatorId)
    {
        g_actuatorStatus[actuatorId] = ActuatorStatus{};
        g_actuatorStatus[actuatorId].actuatorId = actuatorId;
        g_actuatorStatus[actuatorId].connectionState = ActuatorConnection::Disconnected;
    }
}

void touchFrontControllerFreshness(unsigned long nowMs)
{
    g_controllerFreshness[ROLE_FRONT] =
        ControllerFreshnessTracker::seenAt(nowMs);
}

void fillFrontActuatorStatusFromHost()
{
    static const char *kFrontActuatorNames[] = {
        "FLHY", "FLHL", "FLKL", "FRHY", "FRHL", "FRKL",
    };
    for (const char *name : kFrontActuatorNames)
    {
        LinearActuator *actuator = find_actuator(name);
        if (actuator == nullptr) continue;
        const ActuatorStatus status = actuator->getStatus();
        if (status.actuatorId < ActuatorId::ActuatorCount)
        {
            g_actuatorStatus[status.actuatorId] = status;
        }
    }
}

void fillAllActuatorStatusFromThreeBoardManagers()
{
    static const char *kAllNames[] = {
        "FLHY", "FLHL", "FLKL", "FRHY", "FRHL", "FRKL",
        "RLHY", "RLHL", "RLKL", "MLHY", "MLHL", "MLKL",
        "RRHY", "RRHL", "RRKL", "MRHY", "MRHL", "MRKL",
    };
    for (const char *name : kAllNames)
    {
        LinearActuator *actuator = findThreeBoardActuator(name);
        if (actuator == nullptr) continue;
        const ActuatorStatus status = actuator->getStatus();
        if (status.actuatorId < ActuatorId::ActuatorCount)
        {
            g_actuatorStatus[status.actuatorId] = status;
        }
    }
}

}  // namespace

void hostDisplayNoteTelemetryLine(const char *line, BoardRole boardRole)
{
    if (line == nullptr || boardRole == ROLE_UNKNOWN) return;

    ActuatorStatus parsed[CONTROLLER_ACTUATOR_COUNT];
    if (!parseActuatorStatus(line, boardRole, parsed)) return;

    g_controllerFreshness[boardRole] =
        ControllerFreshnessTracker::seenAt(millis());
    for (const ActuatorStatus &status : parsed)
    {
        if (status.actuatorId < ActuatorId::ActuatorCount)
        {
            g_actuatorStatus[status.actuatorId] = status;
        }
    }
}

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
    g_lastRenderedFrame = DisplayFrame{};
    for (BoardRole role : ALL_BOARD_ROLES)
    {
        g_controllerFreshness[role] = ControllerFreshnessTracker{};
    }
    resetActuatorStatusDefaults();
}

void hostDisplayInvalidate() { g_renderer.invalidate(); }

uint32_t hostOledBufferCrc32() { return simSsd1306().bufferCrc32(); }

ActuatorGlyph hostOledGlyphForActuator(ActuatorId actuatorId)
{
    return g_canvas.lastGlyphForActuator(actuatorId);
}

bool hostOledControllerShownFresh(BoardRole role, unsigned long nowMs)
{
    if (role == ROLE_UNKNOWN) return false;
    (void)nowMs;
    return g_lastRenderedFrame.controllers[role];
}

void hostDisplayStep(unsigned long nowMs)
{
    if (!g_setupDone) hostDisplaySetup();
    if (nowMs - g_lastOledDrawMs < kOledRedrawIntervalMs) return;

    touchFrontControllerFreshness(nowMs);
    if (threeBoardFirmwareLoopEnabled() || singleLeaderEighteenJointEnabled())
    {
        fillAllActuatorStatusFromThreeBoardManagers();
    }
    else
    {
        fillFrontActuatorStatusFromHost();
    }

    const ImuMeasurement imu = leaderImu().measure();
    DisplayFrame frame = buildDisplayFrame(
        ROLE_FRONT,
        g_controllerFreshness,
        g_actuatorStatus,
        imu,
        static_cast<uint32_t>(nowMs),
        kMoveThreshold);
    if (leaderBattery().isOnline())
    {
        const auto sample = leaderBattery().sample();
        const Volts batteryVoltage[2] = {
            Volts(sample.battAVolts), Volts(sample.battBVolts)};
        setBatteryVoltages(frame, batteryVoltage);
    }

    for (ActuatorId actuatorId = ActuatorId::FLHY;
         actuatorId < ActuatorId::ActuatorCount;
         ++actuatorId)
    {
        g_canvas.setLastGlyphForActuator(actuatorId, frame.actuators[actuatorId]);
    }

    if (g_renderer.render(frame))
    {
        g_lastOledDrawMs = nowMs;
        g_lastRenderedFrame = frame;
    }
}

}  // namespace host
}  // namespace sim
}  // namespace krabby
