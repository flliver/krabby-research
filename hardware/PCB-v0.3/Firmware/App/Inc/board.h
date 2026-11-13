
#ifndef BOARD_H
#define BOARD_H

#include <stdint.h>
#include <stdbool.h>

/* Board identification */
uint8_t Board_GetNodeId(void);

/* Board LEDs */
typedef enum
{
    BOARD_LED_M1_LEFT,
    BOARD_LED_M1_RIGHT,
    BOARD_LED_M2_LEFT,
    BOARD_LED_M2_RIGHT,
    BOARD_LED_M3_LEFT,
    BOARD_LED_M3_RIGHT,
    BOARD_LED_STATUS,
    BOARD_LED_CAN_ACTIVITY,

    BOARD_LED_COUNT
} Board_LED_t;

void Board_LED_Set(Board_LED_t led, bool on);
void Board_LED_Toggle(Board_LED_t led);
void Board_LED_AllOff(void);

#endif /* BOARD_H */
