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
