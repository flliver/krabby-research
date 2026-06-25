#include "hall_hw.h"

#include "board_pins.h"
#include <avr/interrupt.h>
#include <avr/io.h>

static volatile uint32_t g_hallEdgeCount[6];    // legacy cumulative edge count (telemetry)
static volatile int32_t  g_hallSignedCount[6];  // signed quadrature count = position (M17 Task 2 §5)

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
