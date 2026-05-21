#include "hall_hw.h"

#include "board_pins.h"
#include <avr/interrupt.h>
#include <avr/io.h>

static volatile uint32_t g_hallEdgeCount[6];

#if KRABBY_PIN_REV == 3

// FL Halls: D50 (PB3/PCINT3), D51 (PB2/PCINT2), D52 (PB1/PCINT1) → PCINT0_vect
// FR Halls: A12 (PK4/PCINT20), A13 (PK5/PCINT21), A14 (PK6/PCINT22) → PCINT2_vect
static const uint8_t kHallPins[6] = { 50, 51, 52, A12, A13, A14 };

static uint8_t s_lastPortB;
static uint8_t s_lastPortK;

void hallHwInit()
{
    for (uint8_t i = 0; i < 6; i++)
    {
        g_hallEdgeCount[i] = 0;
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
    if (chg & _BV(3)) g_hallEdgeCount[0]++; // D50 → slot 0
    if (chg & _BV(2)) g_hallEdgeCount[1]++; // D51 → slot 1
    if (chg & _BV(1)) g_hallEdgeCount[2]++; // D52 → slot 2
}

ISR(PCINT2_vect)
{
    uint8_t k = PINK & 0x70;
    uint8_t chg = k ^ s_lastPortK;
    s_lastPortK = k;
    if (chg & _BV(4)) g_hallEdgeCount[3]++; // A12 → slot 3
    if (chg & _BV(5)) g_hallEdgeCount[4]++; // A13 → slot 4
    if (chg & _BV(6)) g_hallEdgeCount[5]++; // A14 → slot 5
}

#elif KRABBY_PIN_REV == 1

// PORT C PC0–PC2 and PC5–PC3: D37,D36,D35,D32,D33,D34
static const uint8_t kHallPins[6] = { 37, 36, 35, 32, 33, 34 };

static uint8_t s_lastPortCLow6;

void hallHwInit()
{
    for (uint8_t i = 0; i < 6; i++)
        g_hallEdgeCount[i] = 0;

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
    if (chg & _BV(0))
        g_hallEdgeCount[0]++;
    if (chg & _BV(1))
        g_hallEdgeCount[1]++;
    if (chg & _BV(2))
        g_hallEdgeCount[2]++;
    if (chg & _BV(5))
        g_hallEdgeCount[3]++;
    if (chg & _BV(4))
        g_hallEdgeCount[4]++;
    if (chg & _BV(3))
        g_hallEdgeCount[5]++;
}

#else

// Rev 2 (Uno v0.1) — no Hall sensors wired
void hallHwInit()
{
    for (uint8_t i = 0; i < 6; i++)
        g_hallEdgeCount[i] = 0;
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
