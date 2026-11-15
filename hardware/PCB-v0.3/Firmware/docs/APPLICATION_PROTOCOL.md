# Application protocol and engineering decisions, version 1

This implements the delegated design work under the autonomous implementation
assignment. Older planning documents' separate approval gates are superseded by
that assignment. No closed-loop position controller or optional I2C devices.

## Addressing and framing

Standard CAN FD+BRS only, 1 Mbit/s arbitration, 5 Mbit/s data. Request ID 0x500+node,
response 0x580+node, periodic telemetry 0x600+node, node=physical switches 0..7.
No broadcast commands. Exactly one master per node; stop-and-wait request/response.
OpenBLT 0x667+node / 0x7E1+node remains unchanged. Lost replies: read queries may
retry; retrying ARM clears commands, and SAVE repeats a flash erase, so do not
blindly retry mutating requests. Motor packets are complete output snapshots.

Request header: u8 version=1, sequence, opcode, argument, offset, count (6 bytes).
SET adds count payload bytes; MOTOR adds 3 signed bytes; REBOOT_NOW adds u32 key.
All other requests have no payload. Count/offset describe GET/SET byte ranges.
CAN FD frames pad to the next legal DLC with zero bytes; nonzero padding rejected.
USB `app HEX` sends the identical unpadded request, receives `APP HEXRESPONSE`.
Response: u8 version, sequence, opcode, result, payload_length, then payload;
CAN padding zero. Results 0=OK,1=format,2=range/mode,3=disarmed,4=hardware,
5=no saved config,6=reboot authorization,7=busy/armed. All integers little endian.
Unknown opcodes or malformed requests never refresh watchdog or change outputs.

Opcodes: 1 VERSION, 2 STATUS, 3 SENSORS, 4 GET, 5 SET, 6 DEFAULTS,
7 SAVE, 8 LOAD, 9 ARM, 10 DISARM, 11 MOTOR, 12 REBOOT_ARM, 13 REBOOT_NOW, 14 UID.
Unused arg/offset/count must be zero except the documented parameters.
GET/SET arg=page 0..3, offset and count multiples of 4, count=4..48 for node, 4..56 for motors,
range within page. DEFAULTS arg=0..3 for one page, 255 for all.
SENSORS arg=0 board or 1..3 motor. MOTOR 3 int8 values -127..127; -128 invalid.
REBOOT_ARM returns u32 0x4B524242; NOW must echo it within 500 ms over the SAME
transport, then permission is consumed. Wrong key consumes permission too.
Reboot first disarms and sleeps all drivers. Replies are queued before reset delay.
VERSION returns u32 protocol=1, config_schema=4, app_base=0x08003000, capabilities=63 (bit 5 advertises UID support).
UID returns 12 bytes: HAL_GetUIDw0/w1/w2, each uint32 little-endian. Host display concatenates the three words in that order as 24 uppercase hexadecimal digits. It is the factory silicon identifier, independent of node switches and saved configuration. Read-only, no watchdog refresh or motor/configuration changes. USB text: node uid. Python CLI: python tools/krabby.py --node 1 uid.
STATUS payload: u32 uptime, u8 node, node_status, motor_status[3], int8 commands[3],
u32 adc_samples, adc_errors, transport_drops. SENSORS payload is documented in host.
Telemetry rotates STATUS (sequence=0, opcode=2), then SENSORS board/M1/M2/M3
(sequence=1/2/3/4, opcode=3), one frame per telemetry interval.

## Configuration schema 4

Node page: 48 bytes / 12 little-endian int32 fields. Each motor page: 56 bytes / 14 fields. Numeric offsets are field*4. A complete motor SET is 62 bytes including its six-byte header, fitting one CAN FD frame.
Node page: 0 watchdog_mode=0 only; 1 watchdog_timeout=25 (uint8,10 ms units,
0 disables); 2 telemetry_ms=0 (off or 10..60000); 3 board_filter_shift=3 (0..8);
4 ntc_beta=0 (0..10000 K, zero means unknown); 5 ntc_nominal_ohms=10000;
6 ntc_pullup_ohms=10000 (both 1..1000000 ohm);
7 ntc_nominal_mC=25000 (-50000..150000);
8..11 reserved zero. No motor flags are stored on the node page.
Motor pages 1..3: 0 control_mode=0 only; 1 feedback_source=0 pot/1 encoder;
2 report_mode=0 native/1 normalized; 3 current_limit_mode=0 only;
4 current_vref_mv=100 (1..3000 at DRIVER VREF, not DAC pin);
5 lower_limit=0; 6 upper_limit=4095; 7 calibration_min=0;
8 calibration_max=4095; 9 adc_filter_shift=3 (0..8);
10 safe_state=0 brake/1 Hi-Z; 11 travel_limits_enabled=0/1;
12 output_reversed=0/1 (offset 48); 13 normalized_limits=0/1 (offset 52).
Commands are logical: positive=increasing selected feedback. Firmware applies the motor output_reversed field only to electrical PH direction, after travel checks.
Limits use native selected feedback units unless that motor normalized_limits field is 1; then limits are 0..65535 calibrated units, independent of report_mode. Internal normalized comparisons remain unsaturated beyond calibration endpoints.
Calibration endpoints must increase; normalize to 0..65535 with saturation.
If limits enabled and feedback stale/ADC railed, inhibit output. At/beyond lower,
negative duty is suppressed; at/beyond upper, positive suppressed; retreat allowed.
Set the board polarity using supervised Jog calibration before enabling limits. All command paths share this polarity, including USB, CAN, Jog and PC position control. Limit status flags use the same units as enforcement.
Configuration changes reseed filters from latest ADC samples instead of creating a false zero ramp.
Schema-2/3 flash configurations migrate on LOAD/startup without modifying flash. Schema 2 initializes the new motor fields to zero; schema 3 moves each node-mask bit into its motor field. Node masks are cleared. SAVE writes schema 4. New GUIs require schema 4; older GUIs reject it. JSON imports support schemas 2/3/4 with equivalent migration. Motor-page SET updates limits, polarity and units atomically. Node page remains 48 bytes; motor pages grow to 56 bytes. No flash partition or CAN timing changes.
SET validates the entire prospective dataset and disarms before hardware application.
Defaults/load also disarm. No config completeness or saved-image ARM prerequisite.

Flash 0x0801F000..0x0801FFFF: 16-byte header {u32 magic=0x4B524346,schema=4,size=224,CRC32} then
224-byte stored dataset: four internal 56-byte records, with two zero padding int32 fields after the node page. Wire pages total 216 bytes; the stored node padding is not exposed. CRC32/ISO-HDLC of the stored dataset. Header written LAST. One erase page,
no redundant copy; interruption can lose saved configuration. SAVE/LOAD require
disarmed, force Hi-Z during flash work. LOAD failure leaves active RAM unchanged.
NO_CONFIG indicates no verified stored dataset; RAM edits/defaults leave it unchanged.
Successful verified SAVE/LOAD clears it. Invalid saved data uses compiled defaults.

## Safety and hardware decisions

Boot Hi-Z, disarmed. ARM starts watchdog immediately, clears commands, preserves safe
outputs; only subsequent MOTOR may energize. Queries/config/ARM do not refresh a
running watchdog. Only valid addressed MOTOR while armed refreshes it (USB motor
service is equivalent for bench commissioning). Timeout disarms, zeroes all commands,
applies safe states, requires explicit rearm. Timeout flag clears only on ARM.
SysTick enforces timeout independently of USB/CAN output congestion. Fatal exceptions
and bootloader entry force all sleeps LOW. Reversal enters neutral/sleep for at least
2 ms, then sets phase and waits 2 ms for wake before PWM. nFAULT after wake disarms
all and forces Hi-Z; no fault history/automatic rearming.

Node bits: 0 READY (sensors and DAC initialized),1 NO_CONFIG,2 WATCHDOG_TIMEOUT,
3 ARMED,4 HARDWARE_INHIBIT,5 SENSOR_ERROR,6 TRANSPORT_OVERFLOW,7 reserved.
Motor bits: 0 READY,1 ACTIVE (nonzero applied duty),2 LOWER,3 UPPER,4 reserved
(current chopping not observable),5 DRIVER_FAULT (current pin),6 INHIBITED,7 reserved.

DAC limits configured with selected-channel update, 12-bit code based on 3.3V
supply and 100k/(100k+1k) divider; report commanded code, do not claim measured VREF.
ADC raw current and mV are always available. Current mA uses board archive
R59/R60/R61=1.43k and TI nominal 212uA/A mirror; these are nominal, not calibrated
or proof of loaded regulation. NTC R33 has no manufacturer/beta data in the archive.
Celsius is INT32_MIN when ntc_beta=0; known coefficients enable beta-model milliC.
Motor power VM has no ADC. Do not infer VM availability from MCU, USB, or DAC ACK.
Unattended bench tests keep nSLEEP LOW unless supply prerequisites independently
established; missing fuses and unknown VM prohibit energized output validation.

ADC paced at 1 kHz, four ranks per ADC, bounded polling burst with interrupts enabled;
filter y += (x-y)/2^shift with fractional accumulator. Encoders sampled 1 kHz,
16-bit modulo delta extended to int32; M2 delta inverted to account for A/B mapping.
No claim of motion/calibration verification on disconnected sensors.

At 6 nodes, 50 Hz commands plus replies and 50 Hz telemetry frames is 900 frames/s.
Conservative 200 us/frame allowance gives <20% bus occupancy; this is a sizing
estimate, not a measured six-node utilization or worst-case latency guarantee.
Telemetry is best-effort; replies use a bounded queue. Diagnostic USB backlog never
blocks application CAN processing or SysTick safety. One request per node in flight.


## Sensor response payloads

Nine 32-bit little-endian words (36 bytes). Board page 0: raw NTC ADC,
filtered NTC ADC, raw VBUS ADC, filtered VBUS ADC, nominal VBUS mV,
NTC resistance ohm (UINT32_MAX at open circuit), completed sample groups,
ADC errors, signed milliC (INT32_MIN if uncalibrated/invalid). First eight unsigned.
Motor pages 1..3: raw current ADC, filtered current ADC, nominal current-sense mV,
raw pot ADC, filtered pot ADC, signed extended encoder count, signed reported
position, readback-verified 12-bit DAC code, nominal measured current mA.
ADC codes are 0..4095; voltage conversions assume nominal 3.3V VDDA. VBUS uses
68k/82k divider. Current mA = ADC*3300*1e6/(4095*1430*212).
Calibrated position is unsigned range 0..65535 held in an int32 field.
DAC code round-to-nearest = VREF_mV*4096*101/(3300*100), ceiling 4095.
Temperature uses 1/T=1/Tnom+ln(R/Rnom)/beta, Kelvin internally.

## USB service and host examples

Existing hello/status/led/tx commands remain. New human commands use the same
service and return `APP ` followed by the binary response as hexadecimal:
`node version`, `node status`, `node sensors 0`, `node page 1`, `node get 1 4`,
`node set 1 4 100`, `node defaults 255`, `node save`, `node load`, `node arm`,
`node disarm`, `node motor 0 0 0`. Page/field numbers are decimal (0x prefix allowed).
Page/field metadata is machine-readable in config_schema.json. Raw binary service:
`app 010102000000` queries STATUS with sequence 1. Malformed hex is rejected.

From Firmware/tools:

```powershell
python .\krabby.py status
python .\krabby.py page 1
python .\krabby.py set 1 4 100
python .\krabby.py --usb --port COM3 sensors 0
python .\krabby.py save
```

The default host port is COM7, node 0. Close any competing handles. The host library
Node.page/set/status/sensors/motor/request is the master-side reference. Ordinary
motor snapshots require ARM acknowledgment first. Do not automatically ARM on retry.
After firmware update or reboot, reopen USB rather than retaining a stale handle.

## Primary hardware references

- TI DAC7574 datasheet https://www.ti.com/lit/ds/symlink/dac7574.pdf
  (selected-channel update and repeated-start register readback).
- TI DRV8262 datasheet https://www.ti.com/lit/ds/symlink/drv8262.pdf
  (PH/EN truth table and IPROPI conversion).
- Local PCB netlist and EPRO2 attributes identify R59/R60/R61=1.43k;
  the R33 thermistor part/beta remains unavailable. No guessed temperature calibration.


Capability bits (VERSION capabilities=31): 0 config RAM, 1 sensors, 2 direct motor
commands, 3 persistent configuration, 4 periodic telemetry. Reboot is mandatory in v1.
READY means software interfaces initialized; it does not prove VM, fuses, calibration
or loaded hardware readiness. Motor READY and raw DRIVER_FAULT can coexist while
asleep; the raw nFAULT input is not fabricated from a DAC setting.

Example node-0 ARM: ID 0x500, `01 01 09 00 00 00`; reply ID 0x580,
`01 01 09 00 00`. Motor +127/-127/0: `01 02 0B 00 00 00 7F 81 00`
padded with three zero bytes to DLC 9 (12 bytes). GET motor1 VREF: request
`01 03 04 01 10 04`; response `01 03 04 00 04 64 00 00 00` for 100 mV,
padded to 12 bytes. Positive logical command is PH HIGH when output_reversed=0, PH LOW when output_reversed=1; carrier duty is abs(cmd)/127
(20 kHz, CCR=abs(cmd)*4000/127, ARR=3999). Board polarity has been physically tested; see bench/motor_pages/README.md.

Only a received transaction from the current ARM generation can apply motor output.
CAN packets capture generation at RX interrupt; USB captures it at the first line
byte. This rejects pre-ARM backlog even when another interface performs ARM.
PWM main outputs stay disabled at boot and become neutral-enabled at ARM; zero
duty while awake uses the configured safe policy with wake-time fault qualification.
Configuration writes, SAVE/LOAD and fatal failures force Hi-Z while changing hardware.

Future I2C extensions should share hi2c1 from main context, with bounded transactions
and no interrupt-side I2C. Add new configuration version/pages inside the existing
4 KiB reservation when required; never expand into application/boot flash. No IMU
or LCD driver or optional device protocol is implemented.
