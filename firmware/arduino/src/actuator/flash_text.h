#pragma once

// Literal text for calibration lines and prefix matching in host-testable
// headers. On AVR a KRABBY_FLASH_LITERAL lives in flash (PSTR) and is read only
// through KRABBY_FLASH_CHAR or a CalibrationTextSink; host builds get plain
// const char pointers.
#ifdef __AVR__
#include <avr/pgmspace.h>
#define KRABBY_FLASH_LITERAL(literal) (PSTR(literal))
#define KRABBY_FLASH_CHAR(pointer) (static_cast<char>(pgm_read_byte(pointer)))
#else
#define KRABBY_FLASH_LITERAL(literal) (literal)
#define KRABBY_FLASH_CHAR(pointer) (*(pointer))
#endif
