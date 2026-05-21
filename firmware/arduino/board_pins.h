#pragma once

// Pin revision — select at compile time (or pass -DKRABBY_PIN_REV=N).
//
//  1 = Original breadboard: EN D22,D23,D24,D28,D26,D27; Hall D37,D36,D35,D32,D33,D34 (PCINT1/Port C).
//  2 = Krabby Uno v0.1:    EN FL D22,D23,D24 / FR D28,D26,D27; no Hall.
//  3 = Krabby Uno v0.2:    EN interleaved D22,D24,D26 / D23,D25,D27; Hall D50-D52 (PCINT0) + A12-A14 (PCINT2).
//
// PWM (D2-D13) and analog (IS A6-A11, POT A0-A5) are the same across all revisions.
#ifndef KRABBY_PIN_REV
#define KRABBY_PIN_REV 2
#endif

// --- PWM pins — identical for all revisions ---
#define PIN_S0_PWMR  2
#define PIN_S0_PWML  3
#define PIN_S1_PWMR  4
#define PIN_S1_PWML  5
#define PIN_S2_PWMR  6
#define PIN_S2_PWML  7
#define PIN_S3_PWMR  8
#define PIN_S3_PWML  9
#define PIN_S4_PWMR 10
#define PIN_S4_PWML 11
#define PIN_S5_PWMR 12
#define PIN_S5_PWML 13

// --- EN pins — differ per revision ---
#if KRABBY_PIN_REV == 1

#define PIN_S0_EN 22
#define PIN_S1_EN 23
#define PIN_S2_EN 24
#define PIN_S3_EN 28
#define PIN_S4_EN 26
#define PIN_S5_EN 27

#elif KRABBY_PIN_REV == 2

#define PIN_S0_EN 22   // FL board
#define PIN_S1_EN 23
#define PIN_S2_EN 24
#define PIN_S3_EN 28   // FR board
#define PIN_S4_EN 26
#define PIN_S5_EN 27

#elif KRABBY_PIN_REV == 3

#define PIN_S0_EN 22   // FL
#define PIN_S1_EN 24
#define PIN_S2_EN 26
#define PIN_S3_EN 23   // FR
#define PIN_S4_EN 25
#define PIN_S5_EN 27

#else
#error "KRABBY_PIN_REV must be 1, 2, or 3"
#endif

inline const char* boardPinRevisionLabel()
{
#if KRABBY_PIN_REV == 1
    return "PINS_REV1_ORIGINAL";
#elif KRABBY_PIN_REV == 2
    return "PINS_REV2_UNO_V01";
#else
    return "PINS_REV3_UNO_V02";
#endif
}
