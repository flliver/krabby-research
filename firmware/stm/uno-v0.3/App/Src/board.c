#include "board.h"
#include "main.h"

/* -------------------------------------------------------
 * Board identification switches
 * Active LOW, external pull-ups
 * ------------------------------------------------------- */



uint8_t Board_GetNodeId(void)
{
    uint8_t id = 0;

    if (HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_13) == GPIO_PIN_RESET)
        id |= 4U;

    if (HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_4) == GPIO_PIN_RESET)
        id |= 2U;

    if (HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_5) == GPIO_PIN_RESET)
        id |= 1U;

    return id;
}
 

/* -------------------------------------------------------
 * Board LEDs
 * All LEDs active HIGH
 * GPIO initialization is handled by CubeMX
 * ------------------------------------------------------- */

typedef struct
{
    GPIO_TypeDef *port;
    uint16_t pin;
} Board_LED_Map_t;

static const Board_LED_Map_t led_map[BOARD_LED_COUNT] =
{
    [BOARD_LED_M1_LEFT]      = {GPIOC, GPIO_PIN_2},
    [BOARD_LED_M1_RIGHT]     = {GPIOC, GPIO_PIN_3},
    [BOARD_LED_M2_LEFT]      = {GPIOA, GPIO_PIN_3},
    [BOARD_LED_M2_RIGHT]     = {GPIOA, GPIO_PIN_4},
    [BOARD_LED_M3_LEFT]      = {GPIOB, GPIO_PIN_6},
    [BOARD_LED_M3_RIGHT]     = {GPIOD, GPIO_PIN_2},
    [BOARD_LED_STATUS]       = {GPIOC, GPIO_PIN_8},
    [BOARD_LED_CAN_ACTIVITY] = {GPIOC, GPIO_PIN_9}
};

void Board_LED_Set(Board_LED_t led, bool on)
{
    if ((unsigned)led >= BOARD_LED_COUNT)
        return;

    HAL_GPIO_WritePin(
        led_map[led].port,
        led_map[led].pin,
        on ? GPIO_PIN_SET : GPIO_PIN_RESET
    );
}

void Board_LED_Toggle(Board_LED_t led)
{
    if ((unsigned)led >= BOARD_LED_COUNT)
        return;

    HAL_GPIO_TogglePin(
        led_map[led].port,
        led_map[led].pin
    );
}

void Board_LED_AllOff(void)
{
    for (unsigned i = 0; i < BOARD_LED_COUNT; i++)
    {
        Board_LED_Set((Board_LED_t)i, false);
    }
}
