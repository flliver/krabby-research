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
- Three instances together: `runThreeBoardRoleElectionCooperative()`.

The single-board USB path (`host/host_firmware_loop.cpp`) still assumes an already-elected **FRONT** leader. Full three-instance boot runs cooperative election first.

## Tests

Native coverage: `tests/native/firmware/arduino/test_host_board_role_election.cpp` (`make -C firmware test-native`).
