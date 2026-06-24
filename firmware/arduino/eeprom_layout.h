#pragma once
#include <Arduino.h>
#include <EEPROM.h>
#include <stdint.h>
#include <stddef.h>

// ============================================================================
// Persistent board configuration — the single source of truth for everything the
// board remembers across power cycles (currently its role and serial number).
//
// The whole struct lives at EEPROM address 0 and is read/written as one unit via
// EEPROM.get/put. On load it is checked against a magic word, a schema version, and
// a CRC32; if any check fails (blank chip, corruption, or a struct written by a
// different schema version) the board falls back to ROLE_UNKNOWN instead of trusting
// garbage. Add a config field by editing the struct and bumping EEPROM_SCHEMA_VER —
// there are no hand-rolled byte offsets to keep in sync.
// ============================================================================

constexpr uint16_t EEPROM_MAGIC      = 0x4B17;  // sentinel marking an initialized struct
constexpr uint8_t  EEPROM_SCHEMA_VER = 1;       // bump when EepromLayout changes
constexpr int      EEPROM_BASE_ADDR  = 0;       // single struct lives at addr 0

enum BoardRole : uint8_t {
    ROLE_UNKNOWN = 0,
    ROLE_FRONT   = 1,
    ROLE_LEFT    = 2,
    ROLE_RIGHT   = 3,
};

constexpr size_t EEPROM_SERIAL_LEN = 16;  // zero-padded ASCII; "" if unset

struct EepromLayout {
    uint16_t  magic;                     // EEPROM_MAGIC when valid
    uint8_t   schema_version;            // EEPROM_SCHEMA_VER; bumped on struct change
    BoardRole role;                      // FRONT / LEFT / RIGHT / UNKNOWN
    char      serial[EEPROM_SERIAL_LEN]; // zero-padded ASCII; serial[0]=='\0' if unset
    // add new config fields here (before crc32), then bump EEPROM_SCHEMA_VER
    uint32_t  crc32;                     // over all bytes before this field
};

// --- CRC32 (IEEE 802.3, poly 0xEDB88320), bytewise — no table, AVR-friendly ---
inline uint32_t eepromCrc32(const uint8_t* data, size_t len) {
    uint32_t crc = 0xFFFFFFFFUL;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (uint8_t b = 0; b < 8; b++)
            crc = (crc & 1u) ? (crc >> 1) ^ 0xEDB88320UL : (crc >> 1);
    }
    return ~crc;
}

// Persist cfg: stamps magic + schema + crc32, then writes the whole struct.
inline void eepromSave(EepromLayout& cfg) {
    cfg.magic = EEPROM_MAGIC;
    cfg.schema_version = EEPROM_SCHEMA_VER;
    cfg.crc32 = eepromCrc32(reinterpret_cast<const uint8_t*>(&cfg),
                            offsetof(EepromLayout, crc32));
    EEPROM.put(EEPROM_BASE_ADDR, cfg);
}

// Load cfg from EEPROM. Returns false (and leaves cfg defaulted) when the stored
// bytes are invalid: bad magic, wrong schema, or CRC mismatch.
inline bool eepromLoad(EepromLayout& cfg) {
    EEPROM.get(EEPROM_BASE_ADDR, cfg);
    const uint32_t want = eepromCrc32(reinterpret_cast<const uint8_t*>(&cfg),
                                      offsetof(EepromLayout, crc32));
    const bool valid = cfg.magic == EEPROM_MAGIC
                    && cfg.schema_version == EEPROM_SCHEMA_VER
                    && cfg.crc32 == want;
    if (!valid)
        cfg = EepromLayout{};  // value-init: role defaults to ROLE_UNKNOWN (0)
    return valid;
}

// --- role <-> config string (for SET/GET role). Distinct from the fixed-width
// telemetry roleName() in arduino.ino, which stays as-is for the wire telemetry. ---
inline const char* roleConfigName(BoardRole r) {
    switch (r) {
        case ROLE_FRONT: return "FRONT";
        case ROLE_LEFT:  return "LEFT";
        case ROLE_RIGHT: return "RIGHT";
        default:         return "UNKNOWN";
    }
}

inline bool parseRole(const String& s, BoardRole& out) {
    if (s == "FRONT")   { out = ROLE_FRONT;   return true; }
    if (s == "LEFT")    { out = ROLE_LEFT;    return true; }
    if (s == "RIGHT")   { out = ROLE_RIGHT;   return true; }
    if (s == "UNKNOWN") { out = ROLE_UNKNOWN; return true; }
    return false;
}

// ============================================================================
// Per-joint calibration (M17 Task 2) — a SEPARATE EEPROM block from the config
// struct above, at its own base address. Kept separate (rather than a field of
// EepromLayout) so a re-cal doesn't rewrite role/serial, and so adding cal doesn't
// invalidate the Task-1 config already stored on existing boards. It has its own
// magic + schema + CRC and is loaded/saved as one unit, exactly like EepromLayout.
//
// EEPROM map:  [0 .. sizeof(EepromLayout)-1]                       = board config
//              [JOINTCAL_BASE_ADDR .. +sizeof(JointCalBlock)-1]    = per-joint cal
// Mega 2560 EEPROM is 4096 bytes; this uses < 200, leaving room for a future M16 cal.
// ============================================================================

constexpr uint16_t JOINTCAL_MAGIC      = 0xCA17;  // "cal-17" (M17)
constexpr uint8_t  JOINTCAL_SCHEMA_VER = 1;       // bump when JointCal/JointCalBlock changes
constexpr int      JOINTCAL_BASE_ADDR  = 64;      // clears EepromLayout at addr 0
constexpr uint8_t  JOINTCAL_SLOTS      = 6;       // actuators per board

// Which position sensor a joint's motor cable carries (each motor has exactly one).
// Auto-detected during the calibration nudge (Task 2 §3).
enum SensorType : uint8_t {
    SENSOR_POT  = 0,   // potentiometer — absolute by physics; the default
    SENSOR_HALL = 1,   // Hall encoder — incremental signed count
};

// Recorded calibration for one joint. Min = retract-stop, Max = extend-stop, stored
// PRE-flip; sensorReversed says whether to invert the reading at the read site (2b).
struct JointCal {
    uint16_t potMin;          // raw ADC at retract-stop  (used when sensorType==SENSOR_POT)
    uint16_t potMax;          // raw ADC at extend-stop
    int32_t  hallMin;         // signed count at retract-stop (used when sensorType==SENSOR_HALL)
    int32_t  hallMax;         // signed count at extend-stop
    uint8_t  sensorType;      // SensorType auto-detected for this slot
    uint8_t  sensorReversed;  // 1 = sensor reads inverted vs drive direction
    uint8_t  calibrated;      // 1 = min/max recorded; 0 = slot never calibrated
    uint8_t  reserved[5];     // pad to 20 bytes; room for future per-joint fields
};

// All six slots for this board, persisted as one unit with magic + schema + CRC.
struct JointCalBlock {
    uint16_t magic;                  // JOINTCAL_MAGIC when valid
    uint8_t  schema_version;         // JOINTCAL_SCHEMA_VER
    uint8_t  joint_count;            // JOINTCAL_SLOTS
    JointCal joints[JOINTCAL_SLOTS];
    uint32_t crc32;                  // over all bytes before this field
};

static_assert(sizeof(EepromLayout) <= JOINTCAL_BASE_ADDR,
              "EepromLayout overlaps the JointCal region");

// Persist the joint-cal block: stamp magic + schema + crc, then write as one unit.
inline void jointCalSave(JointCalBlock& blk) {
    blk.magic = JOINTCAL_MAGIC;
    blk.schema_version = JOINTCAL_SCHEMA_VER;
    blk.joint_count = JOINTCAL_SLOTS;
    blk.crc32 = eepromCrc32(reinterpret_cast<const uint8_t*>(&blk),
                            offsetof(JointCalBlock, crc32));
    EEPROM.put(JOINTCAL_BASE_ADDR, blk);
}

// Load the joint-cal block. Returns false (and zero-inits blk, so every slot reads
// calibrated==0) when the stored bytes are blank, corrupt, or a different schema.
inline bool jointCalLoad(JointCalBlock& blk) {
    EEPROM.get(JOINTCAL_BASE_ADDR, blk);
    const uint32_t want = eepromCrc32(reinterpret_cast<const uint8_t*>(&blk),
                                      offsetof(JointCalBlock, crc32));
    const bool valid = blk.magic == JOINTCAL_MAGIC
                    && blk.schema_version == JOINTCAL_SCHEMA_VER
                    && blk.crc32 == want;
    if (!valid)
        blk = JointCalBlock{};  // value-init: all zero, every slot calibrated==0
    return valid;
}
