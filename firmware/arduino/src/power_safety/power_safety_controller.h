#pragma once

#include "../periodic_timer.h"
#include "power_safety_latch.h"

enum class PowerSafetyTransitionResult : uint8_t
{
    Applied,
    Rejected,
    PersistenceFailed,
};

class PowerSafetyController
{
public:
    PowerSafetyState state() const
    {
        return state_;
    }

    PackVoltageRegion lastPackVoltageRegion() const
    {
        return lastPackVoltageRegion_;
    }

    Volts lastPackVoltage() const
    {
        return lastPackVoltage_;
    }

    bool isLastPackReadingValid() const
    {
        return isLastPackReadingValid_;
    }

    bool isOrinPowerCommandedOn() const
    {
        return isOrinPowerCommandedOn_;
    }

    uint16_t belowSoftTicks() const
    {
        return belowSoftTicks_;
    }

    PowerSafetyAction determinePowerSafetyAction(
        Volts packVoltage, bool isReadingValid, uint32_t now)
    {
        if (state_ == PowerSafetyState::Sleep)
            recoveryPollTimer_.restart(now);
        lastPackVoltage_ = packVoltage;
        isLastPackReadingValid_ = isReadingValid;
        const PackVoltageRegion region = isReadingValid
            ? classifyPackVoltage(packVoltage)
            : lastPackVoltageRegion_;
        const PowerSafetyEvaluation evaluation = evaluatePowerSafetyAction(
            state_,
            region,
            isReadingValid,
            belowSoftTicks_);
        belowSoftTicks_ = evaluation.belowSoftTicks;
        if (isReadingValid)
            lastPackVoltageRegion_ = region;

        return resolvePowerSafetyActions(
            evaluation.action,
            determinePowerSafetyAction(now));
    }

    PowerSafetyAction determinePowerSafetyAction(uint32_t now) const
    {
        return evaluateOrinCutoffAction(
            now,
            orinCutoffArmedMilliseconds_,
            isOrinCutoffArmed_,
            isOrinPowerCommandedOn_,
            isShutdownAcknowledged_);
    }

    template <typename Storage>
    PowerSafetyLatchLoadResult restorePowerSafetyState(
        Storage &storage, uint32_t now)
    {
        const PowerSafetyLatchLoadResult result = latch_.load(storage);
        switch (latch_.value())
        {
            case PowerSafetyLatchValue::None:
                state_ = PowerSafetyState::Operating;
                isOrinPowerCommandedOn_ = true;
                break;
            case PowerSafetyLatchValue::Sleep:
                state_ = PowerSafetyState::Sleep;
                isOrinPowerCommandedOn_ = false;
                recoveryPollTimer_.restart(now);
                resetDebounce();
                break;
        }
        if (result == PowerSafetyLatchLoadResult::Corrupt)
            latch_.save(storage, PowerSafetyLatchValue::Sleep);
        return result;
    }

    PowerSafetyAction reconcilePowerSafetyAfterReset(
        Volts packVoltage,
        bool isReadingValid,
        uint32_t now)
    {
        if (state_ == PowerSafetyState::Operating)
            return PowerSafetyAction::None;

        return determinePowerSafetyAction(
            packVoltage, isReadingValid, now);
    }

    template <typename Storage>
    PowerSafetyTransitionResult beginPoweringDown(Storage &storage)
    {
        if (state_ == PowerSafetyState::Operating ||
            state_ == PowerSafetyState::Resuming)
            return transitionTo(PowerSafetyState::PoweringDown, storage);
        return PowerSafetyTransitionResult::Rejected;
    }

    template <typename Storage>
    PowerSafetyTransitionResult enterSleep(Storage &storage, uint32_t now)
    {
        if (state_ == PowerSafetyState::OverVoltageLockout)
            return PowerSafetyTransitionResult::Rejected;
        if (state_ == PowerSafetyState::Sleep)
            return PowerSafetyTransitionResult::Applied;
        const PowerSafetyTransitionResult result =
            transitionTo(PowerSafetyState::Sleep, storage);
        recoveryPollTimer_.restart(now);
        resetDebounce();
        return result;
    }

    template <typename Storage>
    PowerSafetyTransitionResult beginResuming(Storage &storage)
    {
        if (state_ == PowerSafetyState::Sleep)
            return transitionTo(PowerSafetyState::Resuming, storage);
        return PowerSafetyTransitionResult::Rejected;
    }

    template <typename Storage>
    PowerSafetyTransitionResult completeResume(Storage &storage)
    {
        if (state_ != PowerSafetyState::Resuming)
            return PowerSafetyTransitionResult::Rejected;
        return transitionTo(PowerSafetyState::Operating, storage);
    }

    template <typename Storage>
    PowerSafetyTransitionResult completeShutdown(
        Storage &storage, uint32_t now)
    {
        if (state_ == PowerSafetyState::Sleep)
            return PowerSafetyTransitionResult::Applied;
        if (state_ != PowerSafetyState::PoweringDown)
            return PowerSafetyTransitionResult::Rejected;
        return enterSleep(storage, now);
    }

    template <typename Storage>
    PowerSafetyTransitionResult enterOverVoltageLockout(Storage &storage)
    {
        if (state_ == PowerSafetyState::OverVoltageLockout)
            return PowerSafetyTransitionResult::Applied;
        const PowerSafetyTransitionResult result = transitionTo(
            PowerSafetyState::OverVoltageLockout, storage);
        resetDebounce();
        return result;
    }

    void armOrinCutoff(uint32_t now)
    {
        orinCutoffArmedMilliseconds_ = now;
        isOrinCutoffArmed_ = true;
        isShutdownAcknowledged_ = false;
    }

    void armOrinCutoffIfNeeded(uint32_t now)
    {
        if (!isOrinCutoffArmed_)
            armOrinCutoff(now);
    }

    void markOrinPowerCut()
    {
        isOrinPowerCommandedOn_ = false;
        isOrinCutoffArmed_ = false;
    }

    void markOrinPowerRestored()
    {
        isOrinPowerCommandedOn_ = true;
        isOrinCutoffArmed_ = false;
        isShutdownAcknowledged_ = false;
    }

    bool acceptShutdownAckIfExpected(uint32_t now)
    {
        if (state_ != PowerSafetyState::PoweringDown ||
            !isOrinCutoffArmed_ || !isOrinPowerCommandedOn_ ||
            isShutdownAcknowledged_)
            return false;
        orinCutoffArmedMilliseconds_ = now;
        isShutdownAcknowledged_ = true;
        return true;
    }

    void beginLowPowerIndicators()
    {
        isStatusLedOn_ = true;
    }

    void toggleStatusLed()
    {
        isStatusLedOn_ = !isStatusLedOn_;
    }

    void clearStatusLed()
    {
        isStatusLedOn_ = false;
    }

    bool isStatusLedOn() const
    {
        return isStatusLedOn_;
    }

    bool areActuatorsParked() const
    {
        return state_ == PowerSafetyState::PoweringDown ||
               state_ == PowerSafetyState::Sleep ||
               state_ == PowerSafetyState::OverVoltageLockout;
    }

    bool isNormalOperationSuspended() const
    {
        return state_ == PowerSafetyState::Sleep ||
               state_ == PowerSafetyState::OverVoltageLockout;
    }

    bool shouldCheckForRecovery(uint32_t now) const
    {
        return state_ == PowerSafetyState::Sleep &&
               recoveryPollTimer_.isExpired(now);
    }

private:
    static PowerSafetyLatchValue latchForState(PowerSafetyState state)
    {
        switch (state)
        {
            case PowerSafetyState::Operating:
                return PowerSafetyLatchValue::None;
            case PowerSafetyState::PoweringDown:
            case PowerSafetyState::Sleep:
            case PowerSafetyState::Resuming:
            case PowerSafetyState::OverVoltageLockout:
                return PowerSafetyLatchValue::Sleep;
        }
        return PowerSafetyLatchValue::Sleep;
    }

    template <typename Storage>
    PowerSafetyTransitionResult transitionTo(
        PowerSafetyState nextState, Storage &storage)
    {
        state_ = nextState;
        return latch_.save(storage, latchForState(state_)) ==
                PowerSafetyLatch::SaveResult::VerificationFailed
            ? PowerSafetyTransitionResult::PersistenceFailed
            : PowerSafetyTransitionResult::Applied;
    }

    void resetDebounce()
    {
        belowSoftTicks_ = 0;
    }

    PowerSafetyState state_ = PowerSafetyState::Operating;
    PackVoltageRegion lastPackVoltageRegion_ = PackVoltageRegion::Normal;
    uint16_t belowSoftTicks_ = 0;

    uint32_t orinCutoffArmedMilliseconds_ = 0;
    PeriodicTimer recoveryPollTimer_{POWER_RECOVERY_POLL_INTERVAL_MS};
    Volts lastPackVoltage_;
    bool isShutdownAcknowledged_ = false;
    bool isStatusLedOn_ = false;
    bool isOrinPowerCommandedOn_ = true;
    bool isOrinCutoffArmed_ = false;
    bool isLastPackReadingValid_ = false;
    PowerSafetyLatch latch_;
};
