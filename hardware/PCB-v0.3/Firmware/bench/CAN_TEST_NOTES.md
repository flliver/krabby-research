# CAN bridge bench configurations

Recorded 2026-11-11. No commits or branches created.

**Historical pre-OpenBLT configuration.** OpenBLT is now installed with DBANK=0 and
bootloader write protection. The standalone restore commands below require the
additional recovery steps in [openblt/README.md](openblt/README.md). Current application
vectors are at 0x08003000; the CAN electrical timing remains unchanged.

## Hardware and prerequisites

- Krabby v0.3, STM32G474RBT3, FDCAN2 PB12 RX / PB13 TX AF9.
- FDCAN kernel: PCLK1, 80 MHz; peripheral divider 1.
- U7 MCP2562FD: **5 V transceiver supply must be present**, plus 3.3 V VIO.
  A missing 5 V rail caused the original silent-bus failure while USB/MCU remained alive.
- CANable2 firmware version reported: `16e7497-dirty github.com/normaldotcom/canable2.git`.
- STM32 USB CDC COM3; CANable2 COM7; both serial handles opened at 115200.
- Approximately six inches of CAN-H, CAN-L, and ground between two nodes.
- CANable termination jumper installed; 120 ohm dongle across H/L at Krabby's second connector.
- No scope probes attached during the passing tests.
- Motor sleep GPIOs stay LOW, PWM remains stopped. USB `hello`, `status`, LED commands remain.
- Automatic scope transmitter, its commands, and automatic bus-off restart were removed.

## Configurations

All values below are HAL functional values, not register encodings.

| Parameter | Passing baseline | Passing high-speed experiment |
|---|---:|---:|
| Nominal bitrate | 500,000 | 1,000,000 |
| Nominal prescaler / seg1 / seg2 / SJW | 10 / 13 / 2 / 1 | 5 / 13 / 2 / 1 |
| Nominal sample point | 87.5% | 87.5% |
| Data bitrate | 2,000,000 | 5,000,000 |
| Data prescaler / seg1 / seg2 / SJW | 5 / 6 / 1 / 1 | 1 / 13 / 2 / 1 |
| Data sample point | 87.5% | 87.5% |
| Transmitter delay compensation | disabled | enabled |
| TDC offset / filter | 30 / 0 (inactive) | 13 / 0 |
| NBTP | `00090C01` | `00040C01` |
| DBTP | `00040500` | `00800C10` |
| TDCR | `00001E00` | `00000D00` |
| Adapter settings | `S6`, `Y2` | `S8`, `Y5` |

Normal mode, CAN FD+BRS capability, ISO protocol default, automatic retransmission enabled,
transmit pause disabled, protocol exception handling disabled. No individual filters;
global filter accepts standard/extended frames to FIFO0. Software RX queue holds 64 frames.

Rate = 80 MHz / (prescaler * (1 + seg1 + seg2)).
Sample point = (1 + seg1) / (1 + seg1 + seg2).

The adapter's published source at revision 16e7497 uses 170 MHz and 17 quanta per bit,
with an 88.24% sample point. Different clock/prescaler values are expected.
Its `dirty` suffix means the source cannot fully establish the installed binary's settings.

## Evidence and limitations

Baseline: user confirmed a full 140-delivery matrix with per-frame status checks,
ending TEC=0, REC=0, EP=0, BO=0, zero dropped/hwloss/rxerr.

High speed: the agent ran the connected bench on 2026-11-11, ten matrices with varying
payloads, 1,400 independently received deliveries. Every delivery and final status passed.
Final rx=700, TEC=0, REC=0, EP=0, BO=0, dropped=0, hwloss=0, rxerr=0,
IR=`00000200` (transmission-complete flag; no PEA/PED error flags).
Full transcript: [can_1m_5m/test.log](can_1m_5m/test.log).

Each matrix covers both directions, standard and extended identifiers, classic CAN lengths
0/4/8, and FD with/without BRS at lengths 0..8,12,16,20,24,32,48,64.
Repeated matrices offset payload bytes by the round index. This is a sequential
functional test, not a measurement of maximum sustained throughput, bus load, USB busy
resilience, overflow behavior, long cables, temperature, or supply margins.

Earlier 80% sample-point configurations produced STM32 BRS transmit errors despite some
successful deliveries. Enabling TDC did not resolve those errors. Moving to 87.5% with
data prescaler 5 initially kept TDC enabled, which is an invalid combination; disabling
it produced the passing baseline. Do not infer a single proven root cause from these changes.
M_CAN permits TDC only with functional data prescaler 1 or 2 (encoded DBRP 0 or 1).
The bridge now enables/disables TDC according to that constraint.

1 Mbps / 5 Mbps is the maximum rate pair exposed by the current CANable2 serial
firmware, not a claim about the absolute limit of the MCU or transceiver.

## Repeat tests

Close COM3 terminals and release COM7 from any PowerShell SerialPort object first.
Power the transceiver, connect the terminated bus, and reset STM32 before each fresh run.
From `Firmware/tools`:

```powershell
# Current high-speed firmware; ten complete matrices, 1,400 deliveries.
python .\test_can_bridge.py --nominal 1000000 --data 5000000 --check-status --repeat 10

# Only after restoring baseline firmware; 140 deliveries.
python .\test_can_bridge.py --nominal 500000 --data 2000000 --check-status
```

The script configures the adapter, checks STM32 bitrate registers before opening CAN,
rejects invalid TDC/prescaler combinations, checks latched PEA/PED protocol errors,
and fails on delivery mismatches/timeouts, error state/counters, or increased loss counters.
`--mode classic`, `--mode fd`, and `--mode brs` isolate individual modes.
`--verbose` prints serial traffic. `--repeat` defaults to 1. Default rates remain baseline.
The script closes the adapter CAN channel on exit; rerunning configures and opens it again.

## Restore either configuration

Exact prebuilt ELF and IOC snapshots are in `can_500k_2m/` and `can_1m_5m/`.
Each directory includes SHA-256 fingerprints of the snapshots and relevant source files
at capture time. Historical source fingerprints are expected to differ after later edits.
Keep these artifacts even if `build/` is cleaned.

To restore the exact baseline executable, run from the Firmware directory:

```powershell
& 'C:/Users/Me/AppData/Local/stm32cube/bundles/programmer/2.23.0/bin/STM32_Programmer_CLI.exe' -c port=SWD -w 'bench/can_500k_2m/krabby_v03_bringup.elf' -v -rst
```

To restore the high-speed executable, use `bench/can_1m_5m/krabby_v03_bringup.elf` instead.
Flashing a snapshot does not modify the working source configuration.

To restore baseline source timing, copy its IOC to the root project IOC, then regenerate
with CubeMX, update the timing comment in `Src/main.c`, build, and flash the new build:

```powershell
Copy-Item bench/can_500k_2m/krabby_v03_bringup.ioc krabby_v03_bringup.ioc
```

Regeneration is required after restoring/changing the IOC. Keep `App/Src/can_bridge.c`
with its conditional TDC setup, existing USB integration, and motor-safe GPIO settings.
For exact historical behavior regardless of future source changes, use the archived ELF.

## Primary references

- [CANable2 firmware commands](https://github.com/normaldotcom/canable2-fw#supported-commands)
- [CANable2 timing source, revision 16e7497](https://github.com/normaldotcom/canable2-fw/blob/16e7497/src/can.c)
- [MCP2561/2FD datasheet](https://ww1.microchip.com/downloads/en/DeviceDoc/20005284A.pdf)
- [Bosch M_CAN manual, DBTP/TDC restriction in section 2.3.4](https://www.bosch-semiconductors.com/media/ip_modules/pdf_2/m_can/mcan_users_manual_v331.pdf)
