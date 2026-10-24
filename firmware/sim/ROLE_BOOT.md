# Three-board role boot (host sim)

Host role election follows the same rules as `determineRole()` in `firmware/arduino/arduino.ino`, implemented in `firmware/arduino/src/controller/board_role_election.hpp`:

- Broadcast `SYNC` on `Serial1` and `Serial2` every 10 ms during a 3 s window.
- Print `ROLE_HINT: FRONT|LEFT|RIGHT` on USB from EEPROM bytes 32–33 (magic `0xAB` at 32) before election.
- FRONT assigns followers with `ROLE:LEFT` / `ROLE:RIGHT` on the inter-board links when it sees SYNC on both sides (or when it is the I2C/IMU host).
- Followers without the IMU accept those assignments; timeout with no dual-SYNC defaults to **ROLE_UNKNOWN** with front actuators.

Shipping firmware uses real `hasFrontImu()` (LSM6DSO on I2C). On the sim:

| Concern | Host behavior |
|--------|------------------|
| **IMU / I2C host** | Only the FRONT board instance is treated as having the LSM6DSO (`hostBoardHasFrontImu` in `host/host_board_role.hpp`). That board can become **FRONT** when probed alone. |
| **Inter-board links** | FRONT `Serial1` ↔ LEFT `Serial1`, FRONT `Serial2` ↔ RIGHT `Serial2` via `wireDefaultThreeBoardSerialLinks()` (called from `resetHostSim()`). |
| **EEPROM** | One 4 KB image per simulated board; call `setActiveEepromBoard(BoardId)` before role load/save on that board. |
| **Time** | Cooperative election advances `simClock()` (tests step 10 ms at a time). AVR firmware blocks on real `millis()`. |

## API entry points

- Per-board step: `hostBoardRoleElectionBegin` / `hostBoardRoleElectionStep` (`host/host_board_role.hpp`).
- Three instances together: `runThreeBoardRoleElectionCooperative()` (C: `krabby_host_run_three_board_role_election`).
- Python: `boot_three_board_emulators()` in `three_board_boot.py`; PTY on leader USB via `ThreeBoardMcuHost` in `three_board_mcu_host.py`.

The SDK host port (`krabby_host_serial_usb_*`) is always the **FRONT** board’s USB (`BoardId::kFront`). Followers do not expose USB to the host; they use inter-board serial only.

The single-board USB path (`host/host_firmware_loop.cpp`) still assumes an already-elected **FRONT** leader. Call `boot_three_board_emulators()` (or `ThreeBoardMcuHost.start()`) before connecting `KrabbyMCUSDK` when simulating three boards.

With `enable_three_board_firmware_loop(True)`, the host runs `host_multi_board.cpp`: USB `T`/`B`/`J`/… commands apply on FRONT and relay to followers; follower telemetry on inter-board serial is forwarded to USB so the SDK sees `FRONT;`, `LEFT ;`, and `RIGHT;` each 50 ms tick.

## Optional: single-leader 18-joint mode (stepping stone)

For early `KrabbyMCUSDK` tests without booting three emulators or running SYNC election, enable **single-leader 18-joint** mode:

| Switch | Effect |
|--------|--------|
| `KRABBY_SIM_SINGLE_LEADER_18=1` | `ThreeBoardRobotCore.reset()` skips election and uses one USB leader for all 18 joints. |
| `HostActuatorBridge.enable_single_leader_eighteen_joint(True)` | Same host loop from Python/C tests. |
| `SingleLeaderEighteenMcuHost` | PTY + background tick for SDK connect tests (`firmware/sim/single_leader_mcu_host.py`). |

The host runs `singleLeaderEighteenFirmwareLoopOnce()`: commands route to the FRONT/LEFT/RIGHT actuator managers by joint name; USB emits all three role-prefixed telemetry lines each `TELEMETRY_INTERVAL_MS`. No inter-board serial relay.

Production sim paths should use three-instance election + relay (`ThreeBoardMcuHost`). Single-leader mode is a shortcut for early SDK tests only.

## Tests

Native coverage: `tests/native/firmware/arduino/test_host_board_role_election.cpp` (`make -C firmware test-native`).
