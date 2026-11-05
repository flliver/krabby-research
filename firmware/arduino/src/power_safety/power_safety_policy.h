#pragma once

#include <stdint.h>

#include "power_safety_constants.h"

// Keep in sync with firmware.interfaces.battery_telemetry.
enum class PackVoltageRegion : uint8_t
{
    Normal = 0,
    Warn = 1,
    SoftCut = 2,
    HardCut = 3,
    OverVolt = 4,
    AboveRecovery = 5,
};

enum class PowerSafetyState : uint8_t
{
    Operating = 0,
    PoweringDown = 1,
    Sleep = 2,
    Resuming = 3,
    OverVoltageLockout = 4,
};

enum class PowerSafetyAction : uint8_t
{
    None = 0,
    BeginGracefulShutdown,
    BeginHardShutdown,
    BeginOverVoltageShutdown,
    BeginResume,
    CutOrinPower,
    CompleteResume,
};

struct PowerSafetyEvaluation
{
    PowerSafetyAction action;
    uint16_t belowSoftTicks;
};

inline PackVoltageRegion classifyPackVoltage(Volts packVoltage)
{
    if (packVoltage.value() > PACK_OVER_VOLT_THRESHOLD.value())
        return PackVoltageRegion::OverVolt;
    if (packVoltage.value() > PACK_RECOVERY_THRESHOLD.value())
        return PackVoltageRegion::AboveRecovery;
    if (packVoltage.value() > PACK_WARNING_THRESHOLD.value())
        return PackVoltageRegion::Normal;
    if (packVoltage.value() > PACK_SOFT_CUT_THRESHOLD.value())
        return PackVoltageRegion::Warn;
    if (packVoltage.value() > PACK_HARD_CUT_THRESHOLD.value())
        return PackVoltageRegion::SoftCut;
    return PackVoltageRegion::HardCut;
}

inline uint16_t incrementPowerDebounce(uint16_t ticks)
{
    return ticks < POWER_CUT_DEBOUNCE_TICKS
        ? static_cast<uint16_t>(ticks + 1)
        : POWER_CUT_DEBOUNCE_TICKS;
}

inline PowerSafetyEvaluation evaluatePowerSafetyAction(
    PowerSafetyState state,
    PackVoltageRegion region,
    bool isReadingValid,
    uint16_t belowSoftTicks)
{
    if (!isReadingValid || state == PowerSafetyState::OverVoltageLockout)
        return {PowerSafetyAction::None, 0};

    if (region == PackVoltageRegion::OverVolt)
        return {PowerSafetyAction::BeginOverVoltageShutdown, 0};

    if (region == PackVoltageRegion::HardCut &&
        state != PowerSafetyState::Sleep)
        return {PowerSafetyAction::BeginHardShutdown, 0};

    const bool isBelowSoft = region == PackVoltageRegion::SoftCut;
    const uint16_t nextBelowSoftTicks = isBelowSoft
        ? incrementPowerDebounce(belowSoftTicks)
        : 0;
    const bool isSoftCutLatched =
        nextBelowSoftTicks >= POWER_CUT_DEBOUNCE_TICKS;

    if (isSoftCutLatched &&
        (state == PowerSafetyState::Operating ||
         state == PowerSafetyState::Resuming))
        return {
            PowerSafetyAction::BeginGracefulShutdown,
            nextBelowSoftTicks};

    if (state == PowerSafetyState::Sleep &&
        region == PackVoltageRegion::AboveRecovery)
        return {PowerSafetyAction::BeginResume, 0};

    if (state == PowerSafetyState::Resuming)
        return {
            PowerSafetyAction::CompleteResume,
            nextBelowSoftTicks};

    return {PowerSafetyAction::None, nextBelowSoftTicks};
}

inline uint32_t orinForceOffDelay(bool isShutdownAcknowledged)
{
    return isShutdownAcknowledged ? ORIN_ACKED_OFF_MS : ORIN_FORCE_OFF_MS;
}

inline PowerSafetyAction evaluateOrinCutoffAction(
    uint32_t now,
    uint32_t armedAt,
    bool isArmed,
    bool isPowerCommandedOn,
    bool isShutdownAcknowledged)
{
    if (!isArmed || !isPowerCommandedOn)
        return PowerSafetyAction::None;
    const uint32_t elapsed = static_cast<uint32_t>(now - armedAt);
    return elapsed >= orinForceOffDelay(isShutdownAcknowledged)
        ? PowerSafetyAction::CutOrinPower
        : PowerSafetyAction::None;
}

inline PowerSafetyAction resolvePowerSafetyActions(
    PowerSafetyAction voltageAction,
    PowerSafetyAction cutoffAction)
{
    if (voltageAction == PowerSafetyAction::BeginOverVoltageShutdown ||
        voltageAction == PowerSafetyAction::BeginResume)
        return voltageAction;
    return cutoffAction == PowerSafetyAction::None
        ? voltageAction
        : cutoffAction;
}
