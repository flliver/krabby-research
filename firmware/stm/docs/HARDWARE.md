# Leg controller hardware

## Purpose

Electrical and interconnect contract for one leg controller PCB driving three joints over CAN-FD.

## Components

- **MCU:** **STM32G474RBT3** (LQFP-64), application + CAN-FD controller
- **CAN-FD:** **MCP2562FD-E/SN** transceiver, **FDCAN2** on the MCU
- **Motor bridges:** three **DRV8262** H-bridges (`U10`–`U12`), **PH/EN** strapped single-bridge per channel
- **Motor current limit:** **DAC7574** (**I2C** `0x4C`) programs **`M#_VREF`** per channel before motors leave sleep
- **Input power:** pack input via **XT30** (`VM`), **LM5069MM-2** hot-swap/limit, buck **LMR33620ADDAR** → logic **`3V3`**
- **Harness:** motor **XT30** ×3, sensor **JST-PH** ×3, **CAN-FD** daisy **JST-PH** ×2 (`U8`/`U9`)
- **Leg identity:** three **node ID** switches (`NODE_ID0`–`NODE_ID2`) — see [`PROVISIONING.md`](./PROVISIONING.md)
- Authoritative drawings: leg controller **`SCH-001`** schematic and **`BRD-001`** BOM under repo **`hardware/`**
- Control semantics: [`LEG_BOARD_CONTROL.md`](./LEG_BOARD_CONTROL.md)

## Critical points

- **Joint ↔ channel map (fixed on every identical PCB):** **`M1` = `hip_yaw` (index 0)**, **`M2` = `hip_pitch` (1)**, **`M3` = `knee` (2)**. Firmware and CAN telemetry use this order.
- **Committed bridge (profile B):** **DRV8262DDWR** ×3 — **`M#_PWM`** → EN, **`M#_DIR`** → PH, **`M#_SLEEP`**, active-low **`M#_FAULT`**, **`M#_IPROPI`**. Firmware **default profile B** for current **`SCH-001`** (`LEG_BOARD_CONTROL.md`).
- **Alternate bridge (profile A):** **STSPIN9P2x** dual-PWM + **EN** if a PCB spin substitutes silicon; same joint map **`M1`/`M2`/`M3`**.
- **Flash:** **128 KB** on **STM32G474RBT3** — application + OpenBLT must fit linker regions in `FLASH.md`.
- **Pack / VM rail:** actuator VM **`VM_PROTECTED`** after **LM5069**; per-channel fused feeds **`VM_FUSED_M1`**, **`VM_FUSED_M2`**, **`VM_FUSED_M3`** to each bridge.
- **Logic rails:** **`VOUT_3V3`** supplies MCU and signal conditioning; **`EN_5V`** gates board logic enable; **`CAN_5V_PASS`** feeds the transceiver segment (see schematic power tree).
- **Interlocks / safe disable:** bridge **`M#_SLEEP`** (active level per DRV8262/STSPIN datasheets), **`M#_FAULT`** inputs to MCU, **`LM5069`** shutdown on input abuse, firmware safe state in `LEG_BOARD_CONTROL.md` (all bridges off on stale CAN).
- **CAN-FD physical:** **`CANH`**, **`CANL`**, common-mode choke **`CANH_CHOKE`**, **`CANL_CHOKE`**; MCU **`FDCAN2_TX`**, **`FDCAN2_RX`** to **MCP2562FD**. **No 120 Ω termination on the PCB** — bus termination is **external** at each physical bus end (`CAN_PROTOCOL.md`). Daisy: one **4-pin** CAN harness through the board; schematic note requires **CAN connector pin 4** connected per **`SCH-001`**.
- **Status LEDs:** **`STATUS_LED`** (**`PC8`**) boot/identity; **`CAN_ACTIVITY_LED`** (**`PC9`**) toggles on CAN traffic (operator only).
- **FDCAN2 MCU pins:** **`PB12`** RX, **`PB13`** TX to **MCP2562FD**; transceiver **STB** tied **low** (always active — firmware does not gate standby).

### On-board vs off-board

| Function | On PCB | Off PCB (per joint harness) |
|----------|--------|-----------------------------|
| Position feedback | **`M#_POT_ADC`** conditioning | Pot wiper on linear actuator (`M#_POT` at **JST-PH 5p** `U14`/`U16`/`U18`) |
| Hall (optional) | **`M#_HALL_A_BUF`**, **`M#_HALL_B_BUF`** | Hall sensors `M#_HALL_A` / `M#_HALL_B` on harness |
| Current evidence | **`M#_IPROPI`** / **`M#_IPROPI_DRV`** (bridge profile B) | Motor current via bridge sense |
| Motor power | Bridge **`M#_MOTOR+`**, **`M#_MOTOR-`** | Actuator leads on **XT30** `MOTOR1`–`MOTOR3` |
| Pack power | **XT30** input, fusing, hot-swap | Robot battery harness |
| CAN | Transceiver + connectors **`U8`**, **`U9`** | Daisy to Orin host and adjacent leg boards |
| PCB temperature | **`PCB_TEMP_ADC`** | — |
| Leg identity | **`NODE_ID0`–`NODE_ID2`** (`PC13`, `PB4`, `PB5`) | Closed switch = logic **1** (bit weight `NODE_ID0` = LSB); encodes leg **1–6** at boot (`PROVISIONING.md`) |

- **Analog scaling:** pot and current signals route to **12-bit ADC** inputs named in schematic **`M#_POT_ADC`**; scaling constants live in firmware `board_hw.c` derived from **`SCH-001`** divider values.
- **Pot divider:** **1 kΩ** + **15 kΩ** series from **5 V**, **22 kΩ** to ground at ADC — full travel ≈ **3590** of **4095** counts (send raw ADC in telemetry).
- **Firmware pin map:** generate **`firmware/stm/board_pins.h`** from **`SCH-001`** pinout in **`hardware/`**; table below matches current schematic nets — reconcile PDF if the spin edits labels.

## Reference

### Key BOM MPNs (`BRD-001`)

| Designator | MPN | Role |
|------------|-----|------|
| `U6` | **STM32G474RBT3** | MCU |
| `U7` | **MCP2562FD-E/SN** | CAN-FD transceiver |
| `U1` | **LM5069MM-2/NOPB** | Input protection |
| `U3` | **LMR33620ADDAR** | `3V3` buck |
| `U10`, `U11`, `U12` | **DRV8262DDWR** (`SCH-001`) | Motor bridges (bring-up) |
| `U15`, `U17`, `U19` | **SN74LVC2G17DBVR** | Hall line buffers |
| `U8`, `U9` | **S4B-PH-K-S** | CAN daisy connectors |
| `U14`, `U16`, `U18` | **S5B-PH-K-S** | Sensor harness |
| Power/motor | **XT30PW-F** | `VM` in, `MOTOR1`–`MOTOR3` out |

| `U4` (typ.) | **DAC7574** | Per-motor **VREF** / current limit (program before **`M#_SLEEP`** high) |

Update MPN rows when **`BRD-001`** changes.

### Schematic net → joint function

| Net prefix | Joint index | `LEG_BOARD_CONTROL` joint | Host-facing role |
|------------|-------------|---------------------------|------------------|
| `M1_` | 0 | `hip_yaw` | Yaw linear actuator |
| `M2_` | 1 | `hip_pitch` | Hip pitch actuator |
| `M3_` | 2 | `knee` | Knee actuator |

Per joint **`#` ∈ {1, 2, 3}** (schematic motor index):

| Schematic net | Bridge profile A (**STSPIN9P2x**) | Bridge profile B (**DRV8262**, `SCH-001`) |
|-------------|-----------------------------------|------------------------------------------|
| Enable | `EN` | `M#_SLEEP` |
| Retract PWM | `PWM_L` / phase IN1 | `M#_PWM` + `M#_DIR` = retract |
| Extend PWM | `PWM_R` / phase IN2 | `M#_PWM` + `M#_DIR` = extend |
| Current sense | `ISENSE` (STSPIN) | `M#_IPROPI` |
| Driver fault | `nFAULT` | `M#_FAULT` |
| Position ADC | `M#_POT_ADC` | `M#_POT_ADC` |
| Hall edges | `M#_HALL_A_BUF`, `M#_HALL_B_BUF` | same |

### CAN connector (daisy)

| Signal | Schematic net | Notes |
|--------|---------------|--------|
| CAN high | `CANH` / `CANH_CHOKE` | Through choke to transceiver side |
| CAN low | `CANL` / `CANL_CHOKE` | Through choke to transceiver side |
| Pin 4 | per **`SCH-001`** note | Required connection; do not leave floating |

### MCU pin map (`SCH-001`, motor + CAN)

| Signal | Motor 0 (yaw) | Motor 1 (hip) | Motor 2 (knee) | Peripheral |
|--------|---------------|---------------|----------------|------------|
| PWM | `PA8` TIM1 CH1 | `PA9` TIM1 CH2 | `PA10` TIM1 CH3 | **TIM1** ~20 kHz; enable **MOE** before output |
| DIR | `PB0` | `PB1` | `PC1` | GPIO |
| SLEEP | `PB10` | `PB15` | `PB14` | GPIO; pull-down holds off through reset |
| FAULT | `PC14` | `PC15` | `PB11` | GPIO in, active low |
| POT ADC | `PA7` | `PC4` | `PC5` | **ADC2** |
| Hall A | `PA5` | `PC7` | `PB2` | **TIM2** CH1 / **TIM3** CH2 / **TIM5** CH1 encoder |
| Hall B | `PB3` | `PC6` | `PC12` | **TIM2** CH2 / **TIM3** CH1 / **TIM5** CH2 encoder |
| CAN | `PB12` RX, `PB13` TX | — | — | **FDCAN2** |
| Node ID | `PC13`, `PB4`, `PB5` | — | — | Switches, see `PROVISIONING.md` |
| I2C (DAC) | `PA15` SCL, `PB7` SDA | — | — | **I2C1** |
