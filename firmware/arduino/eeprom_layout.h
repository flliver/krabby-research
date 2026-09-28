#pragma once
#include <Arduino.h>
#include <EEPROM.h>
#include <stdint.h>
#include <stddef.h>

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

// ============================================================================
// Per-joint calibration block (own address, magic, and CRC) — kept clear of the
// board role bytes (32-33, arduino.ino) so writing calibration can never clobber
// the role. Slots are this board's actuator indices 0-5 (which joints those
// are follows from the board's role). Each slot's flags byte marks which stops
// have been recorded — directional calibration ("C <joint> retract|extend")
// writes one stop at a time, so a slot can be half-done. jointCalLoad()
// zero-fills on any validation failure (all flags clear = nothing calibrated).
// ============================================================================

constexpr uint16_t JOINTCAL_MAGIC      = 0xCA17;  // sentinel marking an initialized block
constexpr uint8_t  JOINTCAL_SCHEMA_VER = 2;       // bump when JointCalBlock changes
constexpr int      JOINTCAL_BASE_ADDR  = 64;      // clear of the role bytes at 32-33
constexpr size_t   JOINTCAL_COUNT     = 6;        // one slot per actuator on this board

constexpr uint8_t JOINTCAL_FLAG_MIN  = 0x01;      // minStop (retract stop) recorded
constexpr uint8_t JOINTCAL_FLAG_MAX  = 0x02;      // maxStop (extend stop) recorded
constexpr uint8_t JOINTCAL_FLAG_BOTH = JOINTCAL_FLAG_MIN | JOINTCAL_FLAG_MAX;

struct JointCalBlock {
    uint16_t magic;                        // JOINTCAL_MAGIC when valid
    uint8_t  schema_version;               // JOINTCAL_SCHEMA_VER
    uint16_t minStop[JOINTCAL_COUNT];      // raw ADC at full retract, per slot
    uint16_t maxStop[JOINTCAL_COUNT];      // raw ADC at full extend, per slot
    uint8_t  flags[JOINTCAL_COUNT];        // JOINTCAL_FLAG_* per slot
    uint32_t crc32;                        // over all bytes before this field
};

inline void jointCalSave(JointCalBlock& cal) {
    cal.magic = JOINTCAL_MAGIC;
    cal.schema_version = JOINTCAL_SCHEMA_VER;
    cal.crc32 = eepromCrc32(reinterpret_cast<const uint8_t*>(&cal),
                            offsetof(JointCalBlock, crc32));
    EEPROM.put(JOINTCAL_BASE_ADDR, cal);
}

inline bool jointCalLoad(JointCalBlock& cal) {
    EEPROM.get(JOINTCAL_BASE_ADDR, cal);
    const uint32_t want = eepromCrc32(reinterpret_cast<const uint8_t*>(&cal),
                                      offsetof(JointCalBlock, crc32));
    const bool valid = cal.magic == JOINTCAL_MAGIC
                    && cal.schema_version == JOINTCAL_SCHEMA_VER
                    && cal.crc32 == want;
    if (!valid)
        cal = JointCalBlock{};  // value-init: all slots 0 = uncalibrated
    return valid;
}
