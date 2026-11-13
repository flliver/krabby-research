# Krabby Uno v0.3 — STM32CubeMX IOC handoff

Generated 2026-11-09 from the original EasyEDA Pro archive, extracted U6 physical pad/net mapping, hardware-definition JSON and CSV. Target **STM32G474RBT3**, LQFP64.

## Status

**Prepared for CubeMX import, not yet validated by STM32CubeMX or the ST code generator.** STM32CubeMX is not installed in the execution environment. The project was checked for unique pin assignments, consistency with the extracted 64-pad map, and clock/timer arithmetic. The `.ioc` must be opened in a compatible CubeMX installation and checked for any unavailable settings or reassigned pin modes before code is generated.

## Intended configuration

- HSE: 8 MHz crystal. PLL: 8 MHz / M2 × N40 / R2 = **80 MHz SYSCLK**. Q2 provides 80 MHz FDCAN kernel clock. USB: HSI48 48 MHz (separate clock source).
- TIM1: PA8/PA9/PA10 PWM CH1/CH2/CH3, 80 MHz / (3999+1) = **20 kHz**, 0% startup duty. Do not start TIM1 outputs or set advanced-timer MOE before explicit arming.
- TIM2 CH1/CH2: PA5/PB3; TIM8 CH1/CH2: PC6/PC7 (physical motor 2 A/B is reversed); TIM5 CH1/CH2: PB2/PC12; full quadrature x4; initial filters zero. Verify CubeMX's encoder interface generation.
- ADC1: PA0/PA1/PA2/PC0; ADC2: PA6/PA7/PC4/PC5, four intended regular conversion ranks on each ADC. Polling; no DMA until validated. Reconfirm ADC clock prescaler and sampling periods on the board.
- FDCAN2: PB12/PB13, classic CAN nominal 1 Mbps, 80 MHz kernel / 80 time quanta. Std filter count 1; application must configure reception filter/start controller, and check transceiver enable/power conditions.
- I2C1: PA15/PB7, external DAC7574 at 7-bit 0x4C. `TIMINGR` is provisionally `0x10909CEC` and MUST be regenerated/confirmed in CubeMX with actual risetime/falltime and bus pullups, rather than assumed accurate at 80 MHz.
- USB device peripheral PA11/PA12 (USB device stack/class middleware not yet selected).
- All three DRV8262 SLEEP outputs PB10/PB14/PB15 initialized LOW. Motor direction outputs and indicator LEDs LOW. External pull-downs already fitted on sleep nets; sleep is the primary hardware shutoff. PWM 0 while awake is brake, not coast.
- Retain PA13/PA14 SWD, PF0/PF1 HSE crystal.

## Critical generated-code safety step

**The IOC alone is not a motor safety controller.** Cube-generated init functions may configure TIM1 PA8–PA10 into alternate function immediately. No `HAL_TIM_PWM_Start`, TIM1 `MOE` enable, driver sleep release, or DAC write permitting unexpected current is allowed until application logic has established current limits, detected relevant faults, and explicitly authorized arming. Review reset/default GPIO output levels and put the three SLEEP GPIOs LOW immediately in `MX_GPIO_Init` (BEFORE enabling anything else). Confirm the motor signal truth table and actual physical motor behavior before motor power is connected.

## CubeMX acceptance tests

1. Open `.ioc`, verify STM32G474RBTx, 64-pin package, correct pin locations, no conflicts or invalid pin signals; save to canonicalize it.
2. Review RCC Clock Configuration; SYSCLK 80 MHz, USB 48 MHz, FDCAN 80 MHz, ADC kernel frequency within datasheet limits, and I2C1 clock/timing valid.
3. Verify **three independent encoder interfaces** in TIM2, TIM5, TIM8, with both channels actually selected and TIM8 A/B reversal documented.
4. Check three 20 kHz PWM channels with zero initial compare; no PWM automatically started. Inspect generated `HAL_TIM_MspPostInit` behavior and driver SLEEP initialization ordering.
5. Verify both ADC regular sequencers contain the intended four unique channels with 12-bit conversion and sufficiently long sampling time. Read generated C for ranks: this is particularly important because CubeMX versions differ in how they serialize ADC channel ranks.
6. Verify FDCAN2 classic frame, 1 Mbps, RAM/filter configuration, interrupt enables; configure filters in firmware before enabling receive.
7. Verify USB PCD initialization and optionally add USB CDC middleware in CubeMX; USB enumeration cannot be guaranteed by selecting pins alone.
8. Generate code for selected toolchain, compile, and inspect GPIO reset values. Do initial bring-up with motor driver power isolated.

This document is a verification checklist, **not** a claim that STM32CubeMX has successfully loaded or compiled this file.
