# Krabby Uno v0.3: firmware hardware definition

**Source:** EasyEDA Pro project archive, PCB pad connectivity and exported BOM. **MCU:** STM32G474RBT3 (LQFP64).

**Status:** Pin/net mapping and candidate alternate functions mapped; safe-state policy derived from board circuitry. Actual electrical behavior not yet bench validated. Not certified for energizing motors.

## Peripheral allocation

- **PWM:** `{"timer": "TIM1", "channels": {"M1": "CH1 PA8 AF6", "M2": "CH2 PA9 AF6", "M3": "CH3 PA10 AF6"}, "startup": "MOE=0 / output disabled; CCR=0; nSLEEP=0"}`
- **encoders:** `{"M1": {"timer": "TIM2", "A": "CH1 PA5 AF1", "B": "CH2 PB3 AF1"}, "M2": {"timer": "TIM8", "A": "CH2 PC7 AF4", "B": "CH1 PC6 AF4", "note": "A/B swapped compared with CH1/CH2"}, "M3": {"timer": "TIM5", "A": "CH1 PB2 AF2", "B": "CH2 PC12 AF1"}}`
- **ADC:** `{"ADC1": {"PC0": "IN6 board NTC", "PA0": "IN1 M1 IPROPI", "PA1": "IN2 M2 IPROPI", "PA2": "IN3 M3 IPROPI"}, "ADC2": {"PA6": "IN3 USB VBUS monitor", "PA7": "IN4 M1 pot", "PC4": "IN5 M2 pot", "PC5": "IN11 M3 pot"}}`
- **FDCAN2:** `{"RX": "PB12 AF9", "TX": "PB13 AF9"}`
- **I2C1:** `{"SCL": "PA15 AF4", "SDA": "PB7 AF4", "device": "U13 DAC7574", "7bit_address": "0x4C", "outputs": "A=M1_VREF_DAC, B=M2_VREF_DAC, C=M3_VREF_DAC, D=NC"}`
- **USB_FS:** `{"DM": "PA11", "DP": "PA12", "clock": "48MHz USB clock must be generated / validated"}`
- **HSE:** `{"frequency_Hz": 8000000, "pins": ["PF0", "PF1"]}`

## Hardware safe startup (required sequence)

1. The three DRV8262 nSLEEP signals have 10k physical pull-downs; never rely solely on internal GPIO state to shut motors off.
1. Immediately configure PB10, PB15, PB14 LOW (M1, M2, M3 SLEEP) with output-data register set LOW before changing mode to output.
1. Configure PWM GPIOs PA8/PA9/PA10 LOW before enabling TIM1 alternate functions. Initialize TIM1 channels with CCR=0, outputs disabled (and MOE off) until arming.
1. Configure PB0/PB1/PC1 motor PH pins to a defined LOW initially. No direction changes under drive without a validated neutral/dead-time plan.
1. Initialize sensor GPIO inputs, encoder timers, ADCs, I2C1, FDCAN2 and safety state machine with motor outputs still disabled.
1. Program U13 DAC7574 current-limit VREF to a verified conservative threshold and read/verify as feasible. Check supply and temperature, encoder plausibility and faults.
1. Only after an explicit arm command and validated readiness: enable a single driver nSLEEP, allow wake time, qualify nFAULT, then permit PWM duty to rise under a current and timeout limit.
1. On any critical fault, immediately turn off TIM1 outputs and drive ALL nSLEEP LOW, independently of CAN/USB tasks; require a deliberate re-arm.

## Important verified circuit relationships

- U10/U11/U12 = DRV8262DDWR; MODE1 pin28 pulled to each DVDD via 10k, MODE2 pin29 GND: **single H-bridge PH/EN** configuration.
- nSLEEP pulled low externally by 10k. nFAULT pulled high to 3V3 externally by 10k; active-low fault.
- U13 = DAC7574IDGSR I2C DAC; 1k output feed and 100k to GND at each VREF. Driver current-limit threshold is firmware-controlled and must be initialized while motor is disabled. DAC A1/A0 pins 10/9 are grounded, giving 7-bit I2C address 0x4C. DAC channel D pin5 is unconnected.
- Encoder inputs buffered by SN74LVC2G17. TIM2 for M1, TIM8 for M2 (A on CH2), TIM5 for M3.
- Node IDs SW1: all 3 GPIOs external pull-up 10k; switch-to-GND active-low.
- Boot PB8 pulled low with 100k, button to 3V3.
- PA6 USB VBUS divider 68k top / 82k bottom, ratio 0.5467.
- On-board temp divider 10k 3V3 pull-up / NTC 10k to GND.
- LED GPIO outputs drive resistors toward LED anodes, so LOW = off, HIGH = on.
- 8MHz HSE crystal on PF0/PF1; SWD on PA13/PA14 retained.

## Critical PWM behavior

The DRV8262 is strapped for single H-bridge PH/EN. Driver IN1 pin 41 is **enable/PWM**, IN2 pin 40 is **phase/direction**. The TI truth table states that when nSLEEP is HIGH, **EN=0 commands high-side braking, not coast**. Only nSLEEP=LOW gives guaranteed high-impedance output. Do not equate 0% PWM with coasting or a hard safety-off state.

## Verification gates and unresolved items

- **BLOCKER_FOR_ARMING — DAC current threshold:** U13 DAC7574 outputs through R44/R45/R46 1k to U10/U11/U12 VREF nodes with 100k pull-down R48/R50/R52. Verify DAC I2C address, power-up behavior, VREF->I_TRIP transfer, and write a conservative nonzero current limit before nSLEEP HIGH.
- **BLOCKER_FOR_ARMING — Driver modes:** DRV8262 MODE1 pin28 pulled up to DVDD by R62/R64/R66, MODE2 pin29 tied GND; selected single H-bridge PH/EN (pin40 phase, pin41 enable). Verify on first hardware and bench-test PH/EN truth table; do not assume PWM interface IN1/IN2.
- **VERIFY_ON_HARDWARE — Fault handling:** nFAULT active low, pulled up 10k to 3V3, may assert when drivers asleep. Mask fault logic until ready/awake then latch and immediately disarm on any fault.
- **VERIFY_ON_HARDWARE — USB clock / connection:** USB 48MHz clock tree, VBUS sensing scale, D+ pullup connection and enum not yet validated experimentally.
- **VERIFY_ON_HARDWARE — Encoder direction, filtering:** M2 encoder signal A maps CH2; sign conventions and RC input filtering require empirical test.
- **VERIFY_ON_HARDWARE — Analog acquisition:** ADC sample times, source impedances, sensor rails and conversion scaling need measurement; analog levels may be power-domain dependent.
- **VERIFY_SCHEMATIC — Schematic vs PCB:** This mapping follows extracted PCB pad nets; full independent 15-sheet schematic-to-PCB LVS not performed.
- **VERIFY_STARTUP — BOOT0 option bytes:** PB8 externally low 100k except button to 3V3; confirm nBOOT0 option bytes and documented boot behavior.

## Per-pin reference

| Package pin | GPIO | PCB net | Peripheral / role | AF | initial state |
|---:|---|---|---|---|---|
| 1 | VBAT | VOUT_3V3 | POWER | n/a | n/a |
| 2 | PC13 | NODE_ID0 | GPIO | none | input external pullup HIGH |
| 3 | PC14 | M1_FAULT | GPIO_EXTI optional | none | input (10k external pullup) |
| 4 | PC15 | M2_FAULT | GPIO_EXTI optional | none | input (10k external pullup) |
| 5 | PF0 | OSC_IN | RCC_OSC_IN | dedicated | crystal input |
| 6 | PF1 | OSC_OUT | RCC_OSC_OUT | dedicated | crystal output |
| 7 | PG10 | NRST_STM | NRST | dedicated | reset input |
| 8 | PC0 | PCB_TEMP_ADC | ADC1_IN6 / ADC2_IN6 | analog | high-Z analog |
| 9 | PC1 | M3_DIR | GPIO | none | LOW |
| 10 | PC2 | M1_LEFT_LED_CTRL | GPIO | none | LOW off |
| 11 | PC3 | M1_RIGHT_LED_CTRL | GPIO | none | LOW off |
| 12 | PA0 | M1_IPROPI | ADC1_IN1 / ADC2_IN1 | analog | high-Z analog |
| 13 | PA1 | M2_IPROPI | ADC1_IN2 / ADC2_IN2 | analog | high-Z analog |
| 14 | PA2 | M3_IPROPI | ADC1_IN3 | analog | high-Z analog |
| 15 | VSS | GND | POWER | n/a | n/a |
| 16 | VDD | VOUT_3V3 | POWER | n/a | n/a |
| 17 | PA3 | M2_LEFT_LED_CTRL | GPIO | none | LOW off |
| 18 | PA4 | M2_RIGHT_LED_CTRL | GPIO | none | LOW off |
| 19 | PA5 | M1_POS_A | TIM2_CH1 | AF1 | input high-Z |
| 20 | PA6 | USB_VBUS_SENSE | ADC2_IN3 | analog | high-Z analog |
| 21 | PA7 | M1_POT_ADC | ADC2_IN4 | analog | high-Z analog |
| 22 | PC4 | M2_POT_ADC | ADC2_IN5 | analog | high-Z analog |
| 23 | PC5 | M3_POT_ADC | ADC2_IN11 | analog | high-Z analog |
| 24 | PB0 | M1_DIR | GPIO | none | LOW |
| 25 | PB1 | M2_DIR | GPIO | none | LOW |
| 26 | PB2 | M3_POS_A | TIM5_CH1 | AF2 | input high-Z |
| 27 | VSSA | GND | POWER | n/a | n/a |
| 28 | VREF+ | VDDA | POWER | n/a | n/a |
| 29 | VDDA | VDDA | POWER | n/a | n/a |
| 30 | PB10 | M1_SLEEP | GPIO | none | LOW |
| 31 | VSS | GND | POWER | n/a | n/a |
| 32 | VDD | VOUT_3V3 | POWER | n/a | n/a |
| 33 | PB11 | M3_FAULT | GPIO_EXTI optional | none | input (10k external pullup) |
| 34 | PB12 | FDCAN2_RX | FDCAN2_RX | AF9 | input |
| 35 | PB13 | FDCAN2_TX | FDCAN2_TX | AF9 | recessive HIGH before AF |
| 36 | PB14 | M3_SLEEP | GPIO | none | LOW |
| 37 | PB15 | M2_SLEEP | GPIO | none | LOW |
| 38 | PC6 | M2_POS_B | TIM8_CH1 | AF4 | input high-Z |
| 39 | PC7 | M2_POS_A | TIM8_CH2 | AF4 | input high-Z |
| 40 | PC8 | STATUS_LED | GPIO | none | LOW off |
| 41 | PC9 | CAN_ACTIVITY_LED | GPIO | none | LOW off |
| 42 | PA8 | M1_PWM | TIM1_CH1 | AF6 | GPIO output LOW then timer PWM output at 0 duty |
| 43 | PA9 | M2_PWM | TIM1_CH2 | AF6 | GPIO output LOW then timer PWM output at 0 duty |
| 44 | PA10 | M3_PWM | TIM1_CH3 | AF6 | GPIO output LOW then timer PWM output at 0 duty |
| 45 | PA11 | STM_USB_DN | USB_FS_DM | USB dedicated | do not drive GPIO |
| 46 | PA12 | STM_USB_DP | USB_FS_DP | USB dedicated | do not drive GPIO |
| 47 | VSS | GND | POWER | n/a | n/a |
| 48 | VDD | VOUT_3V3 | POWER | n/a | n/a |
| 49 | PA13 | SWDIO | SYS_JTMS-SWDIO | AF0 | retain SWD |
| 50 | PA14 | SWCLK | SYS_JTCK-SWCLK | AF0 | retain SWD |
| 51 | PA15 | SCL | I2C1_SCL | AF4 | released HIGH |
| 52 | PC10 | — | UNCONNECTED | none | Hi-Z |
| 53 | PC11 | — | UNCONNECTED | none | Hi-Z |
| 54 | PC12 | M3_POS_B | TIM5_CH2 | AF1 | input high-Z |
| 55 | PD2 | M3_RIGHT_LED_CTRL | GPIO | none | LOW off |
| 56 | PB3 | M1_POS_B | TIM2_CH2 | AF1 | input high-Z |
| 57 | PB4 | NODE_ID1 | GPIO | none | input external pullup HIGH |
| 58 | PB5 | NODE_ID2 | GPIO | none | input external pullup HIGH |
| 59 | PB6 | M3_LEFT_LED_CTRL | GPIO | none | LOW off |
| 60 | PB7 | SDA | I2C1_SDA | AF4 | released HIGH |
| 61 | PB8 | BOOT0 | BOOT0 strap | none | external pull-down |
| 62 | PB9 | — | UNCONNECTED | none | Hi-Z |
| 63 | VSS | GND | POWER | n/a | n/a |
| 64 | VDD | VOUT_3V3 | POWER | n/a | n/a |
