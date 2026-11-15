#ifndef APP_BOOT_H
#define APP_BOOT_H
#define KRABBY_APPLICATION_BASE 0x08003000UL
#define KRABBY_CONFIG_BASE 0x0801F000UL
#define KRABBY_CONFIG_SIZE 0x1000UL
#ifndef APP_BLINK_MS
#define APP_BLINK_MS 250U
#endif
#ifndef APP_VARIANT
#define APP_VARIANT 0
#endif
#ifdef LOAD_ME_FIRST
#define APP_LED BOARD_LED_STATUS
#define APP_IDENTITY "LoadMeFirst | eight-LED chase, 100 ms/step | motors disabled"
#elif APP_VARIANT == 1
#define APP_LED BOARD_LED_M1_LEFT
#define APP_IDENTITY "APP-1 M1 LEFT LED (PC2)"
#elif APP_VARIANT == 2
#define APP_LED BOARD_LED_M2_LEFT
#define APP_IDENTITY "APP-2 M2 LEFT LED (PA3)"
#elif APP_VARIANT == 3
#define APP_LED BOARD_LED_M3_LEFT
#define APP_IDENTITY "APP-3 M3 LEFT LED (PB6)"
#else
#define APP_LED BOARD_LED_STATUS
#define APP_IDENTITY "node application v1 STATUS LED (PC8)"
#endif
#endif
