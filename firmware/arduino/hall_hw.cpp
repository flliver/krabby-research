#include "hall_hw.h"

#include "board_pins.h"
#include <avr/interrupt.h>
#include <avr/io.h>

static volatile uint32_t g_hallEdgeCount[6];    // legacy cumulative edge count (debug)
static volatile int32_t  g_hallSignedCount[6];  // signed quadrature count = position (M17 Task 2 §5)

// --- Storm breaker ---
// A fast-shaft encoder (the hip-yaw counts the motor shaft BEFORE its 1:100 gearbox —
// ~170k+ edges/s at full jog PWM) can saturate the PCINT vector: once the edge interval
// drops below the ISR's own run time, PCINT (higher vector priority than the timers)
// re-fires back-to-back and NOTHING else executes — millis() freezes, serial dies, and
// loop() (telemetry, command parsing, the jog watchdog) never runs until the shaft stops.
// Observed on the bench 2026-07-03: FLHY jogged at PWM 200 → telemetry dead in <100 ms,
// stop command and jog watchdog both starved, only a motor-power cut recovered it.
// The only code guaranteed to keep running is the ISR itself, so it self-limits: count
// edges since the last loop() pass, and past the limit mask off the storming pin's own
// PCINT bit. The storm stops, loop() recovers within one pass, and ActuatorManager coasts
// the joint and emits `ERR <joint> hall_storm` (counts made during a storm are unreliable;
// the slot is re-armed once its motor has stopped).
// Limit: worst legit edges in one loop pass ≈ 80 ms of blocking flush() × ~170 kHz ≈ 14k;
// 40000 gives ~3x margin and still trips within ~0.2 s of true saturation.
static const uint16_t    HALL_STORM_EDGE_LIMIT = 40000;
static volatile uint16_t g_edgesSinceLoopPet;
static volatile uint8_t  g_hallStormMask;       // bit N = slot N tripped off, awaiting re-arm

// Quadrature step from one HallA edge, given the current A and B levels: +1 when A != B,
// -1 when A == B. That yields a count that climbs in one travel direction and falls in
// the other (counts both A edges → 2x A resolution). Absolute sign is arbitrary; the
// calibration's direction-flip (sensorReversed) normalizes it.
static inline int8_t quadStep(uint8_t aLevel, uint8_t bLevel)
{
    return (aLevel != bLevel) ? 1 : -1;
}

#if KRABBY_PIN_REV == 3

// FL Halls: HallA D50/D51/D52 (PB3/PB2/PB1, PCINT0); HallB A0/A1/A2 (PF0/PF1/PF2)
// FR Halls: HallA A12/A13/A14 (PK4/PK5/PK6, PCINT2); HallB A3/A4/A5 (PF3/PF4/PF5)
// HallB shares the POT/HALLB analog pins (a pot wiper on a pot actuator, the Hall B
// channel on a Hall actuator). We sample it digitally via PINF for the quadrature phase.
static const uint8_t kHallPins[6] = { 50, 51, 52, A12, A13, A14 };

static uint8_t s_lastPortB;
static uint8_t s_lastPortK;

void hallHwInit()
{
    for (uint8_t i = 0; i < 6; i++)
    {
        g_hallEdgeCount[i] = 0;
        g_hallSignedCount[i] = 0;
        pinMode(kHallPins[i], INPUT_PULLUP);
    }

    s_lastPortB = PINB & 0x0E;
    PCMSK0 |= 0x0E;
    PCICR  |= _BV(PCIE0);

    s_lastPortK = PINK & 0x70;
    PCMSK2 |= 0x70;
    PCICR  |= _BV(PCIE2);
}

ISR(PCINT0_vect)
{
    uint8_t b = PINB & 0x0E;
    uint8_t chg = b ^ s_lastPortB;
    s_lastPortB = b;
    uint8_t f = PINF;  // HallB: slot0=PF0, slot1=PF1, slot2=PF2
    if (chg & _BV(3)) { g_hallEdgeCount[0]++; g_hallSignedCount[0] += quadStep((b >> 3) & 1, (f >> 0) & 1); } // D50/A0
    if (chg & _BV(2)) { g_hallEdgeCount[1]++; g_hallSignedCount[1] += quadStep((b >> 2) & 1, (f >> 1) & 1); } // D51/A1
    if (chg & _BV(1)) { g_hallEdgeCount[2]++; g_hallSignedCount[2] += quadStep((b >> 1) & 1, (f >> 2) & 1); } // D52/A2
    if (++g_edgesSinceLoopPet >= HALL_STORM_EDGE_LIMIT)
    {
        PCMSK0 &= ~chg;  // silence only the pin(s) that just fired; loop() recovers and re-arms later
        if (chg & _BV(3)) g_hallStormMask |= _BV(0);
        if (chg & _BV(2)) g_hallStormMask |= _BV(1);
        if (chg & _BV(1)) g_hallStormMask |= _BV(2);
        g_edgesSinceLoopPet = 0;  // innocent slots start a fresh window
    }
}

ISR(PCINT2_vect)
{
    uint8_t k = PINK & 0x70;
    uint8_t chg = k ^ s_lastPortK;
    s_lastPortK = k;
    uint8_t f = PINF;  // HallB: slot3=PF3, slot4=PF4, slot5=PF5
    if (chg & _BV(4)) { g_hallEdgeCount[3]++; g_hallSignedCount[3] += quadStep((k >> 4) & 1, (f >> 3) & 1); } // A12/A3
    if (chg & _BV(5)) { g_hallEdgeCount[4]++; g_hallSignedCount[4] += quadStep((k >> 5) & 1, (f >> 4) & 1); } // A13/A4
    if (chg & _BV(6)) { g_hallEdgeCount[5]++; g_hallSignedCount[5] += quadStep((k >> 6) & 1, (f >> 5) & 1); } // A14/A5
    if (++g_edgesSinceLoopPet >= HALL_STORM_EDGE_LIMIT)
    {
        PCMSK2 &= ~chg;
        if (chg & _BV(4)) g_hallStormMask |= _BV(3);
        if (chg & _BV(5)) g_hallStormMask |= _BV(4);
        if (chg & _BV(6)) g_hallStormMask |= _BV(5);
        g_edgesSinceLoopPet = 0;
    }
}

#elif KRABBY_PIN_REV == 1

// PORT C PC0–PC2 and PC5–PC3: D37,D36,D35,D32,D33,D34. No HallB wired on Rev 1, so the
// signed count just tracks edges without direction (Rev 1 is the legacy breadboard).
static const uint8_t kHallPins[6] = { 37, 36, 35, 32, 33, 34 };

static uint8_t s_lastPortCLow6;

void hallHwInit()
{
    for (uint8_t i = 0; i < 6; i++)
    {
        g_hallEdgeCount[i] = 0;
        g_hallSignedCount[i] = 0;
    }

    for (uint8_t i = 0; i < 6; i++)
        pinMode(kHallPins[i], INPUT_PULLUP);

    s_lastPortCLow6 = PINC & 0x3F;
    PCMSK1 |= 0x3F;
    PCICR |= _BV(PCIE1);
}

ISR(PCINT1_vect)
{
    uint8_t c = PINC & 0x3F;
    uint8_t chg = c ^ s_lastPortCLow6;
    s_lastPortCLow6 = c;
    if (chg & _BV(0)) { g_hallEdgeCount[0]++; g_hallSignedCount[0]++; }
    if (chg & _BV(1)) { g_hallEdgeCount[1]++; g_hallSignedCount[1]++; }
    if (chg & _BV(2)) { g_hallEdgeCount[2]++; g_hallSignedCount[2]++; }
    if (chg & _BV(5)) { g_hallEdgeCount[3]++; g_hallSignedCount[3]++; }
    if (chg & _BV(4)) { g_hallEdgeCount[4]++; g_hallSignedCount[4]++; }
    if (chg & _BV(3)) { g_hallEdgeCount[5]++; g_hallSignedCount[5]++; }
}

#else

// Rev 2 (Uno v0.1) — no Hall sensors wired
void hallHwInit()
{
    for (uint8_t i = 0; i < 6; i++)
    {
        g_hallEdgeCount[i] = 0;
        g_hallSignedCount[i] = 0;
    }
}

#endif

uint32_t hallHwGetEdgeCount(uint8_t hallSlot)
{
    if (hallSlot >= 6)
        return 0;
    uint8_t oldSreg = SREG;
    cli();
    uint32_t c = g_hallEdgeCount[hallSlot];
    SREG = oldSreg;
    return c;
}

int32_t hallHwGetSignedCount(uint8_t hallSlot)
{
    if (hallSlot >= 6)
        return 0;
    uint8_t oldSreg = SREG;
    cli();
    int32_t c = g_hallSignedCount[hallSlot];
    SREG = oldSreg;
    return c;
}

void hallHwResetCount(uint8_t hallSlot)
{
    if (hallSlot >= 6)
        return;
    uint8_t oldSreg = SREG;
    cli();
    g_hallEdgeCount[hallSlot] = 0;
    g_hallSignedCount[hallSlot] = 0;
    SREG = oldSreg;
}

void hallHwLoopPet()
{
    uint8_t oldSreg = SREG;
    cli();  // uint16 store isn't atomic on AVR
    g_edgesSinceLoopPet = 0;
    SREG = oldSreg;
}

uint8_t hallHwStormMask()
{
    return g_hallStormMask;  // single-byte read is atomic
}

void hallHwStormRearm(uint8_t hallSlot)
{
    if (hallSlot >= 6)
        return;
    // Pin bit per slot: slots 0-2 = PB3/PB2/PB1 (PCMSK0), slots 3-5 = PK4/PK5/PK6 (PCMSK2).
    static const uint8_t kPinBit[6] = { _BV(3), _BV(2), _BV(1), _BV(4), _BV(5), _BV(6) };
    uint8_t oldSreg = SREG;
    cli();
    g_hallStormMask &= ~_BV(hallSlot);
#if KRABBY_PIN_REV == 3
    // Refresh the last-port snapshot so re-arming doesn't fabricate a phantom edge.
    if (hallSlot < 3) { s_lastPortB = PINB & 0x0E; PCMSK0 |= kPinBit[hallSlot]; }
    else              { s_lastPortK = PINK & 0x70; PCMSK2 |= kPinBit[hallSlot]; }
#else
    (void)kPinBit;  // Rev 1/2 never trip a storm (slow or absent Halls)
#endif
    SREG = oldSreg;
}
