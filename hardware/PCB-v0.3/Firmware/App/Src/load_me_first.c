/* First-install application. No motor, configuration, ADC or DAC service. */
#include "app_service.h"
#include "app_boot.h"
#include "board.h"
#include "main.h"
#include <stdio.h>
#include <string.h>
extern FDCAN_HandleTypeDef hfdcan2;
static uint8_t node, led, mailbox[64];
static volatile unsigned pending;
static uint32_t led_at, reboot_at, authorization_at;
static int reboot_pending, authorized;

void App_EmergencySleep(void) {
    GPIOB->BRR = GPIO_PIN_10 | GPIO_PIN_14 | GPIO_PIN_15;
    CLEAR_BIT(TIM1->BDTR, TIM_BDTR_MOE);
}
void App_Tick(void) {}
uint32_t App_CommandEpoch(void) { return 0; }
void App_Init(void) {
    App_EmergencySleep();
    node = Board_GetNodeId();
    Board_LED_AllOff();
    Board_LED_Set(BOARD_LED_M1_LEFT, true);
    led_at = HAL_GetTick();
}
unsigned App_Request(const uint8_t *in, unsigned len, uint8_t out[64]) {
    memset(out, 0, 64);
    out[0] = 1;
    out[1] = len > 1 ? in[1] : 0;
    out[2] = len > 2 ? in[2] : 0;
    out[3] = 1;
    if (len < 6 || in[0] != 1 || in[3] || in[4] || in[5]) return 5;
    unsigned used = in[2] == 13 ? 10 : 6;
    if (len < used) return 5;
    for (unsigned i = used; i < len; i++) if (in[i]) return 5;
    out[3] = 0;
    switch (in[2]) {
    case 1: {
        /* Schema zero prevents motor GUIs connecting; capability bit 5 advertises UID. */
        uint32_t version[4] = {1, 0, KRABBY_APPLICATION_BASE, 32};
        memcpy(out + 5, version, sizeof(version));
        out[4] = sizeof(version);
        break;
    }
    case 14: {
        uint32_t uid[3] = {HAL_GetUIDw0(), HAL_GetUIDw1(), HAL_GetUIDw2()};
        memcpy(out + 5, uid, sizeof(uid));
        out[4] = sizeof(uid);
        break;
    }
    case 12:
        authorized = 1;
        authorization_at = HAL_GetTick();
        memcpy(out + 5, "BBRK", 4);
        out[4] = 4;
        break;
    case 13:
        if (!authorized || HAL_GetTick() - authorization_at >= 500 ||
            memcmp(in + 6, "BBRK", 4)) out[3] = 6;
        else { reboot_pending = 1; reboot_at = HAL_GetTick(); }
        authorized = 0;
        break;
    default: out[3] = 2; break; /* Including ARM, MOTOR, SET, SAVE. */
    }
    return 5 + out[4];
}
int App_CANReceive(uint32_t id, const uint8_t *data, unsigned len) {
    if (id < 0x500 || id > 0x507) return 0;
    if (id != 0x500U + node || pending || len > sizeof(mailbox)) return 1;
    memcpy(mailbox, data, len);
    __DMB();
    pending = len;
    return 1;
}
void App_Poll(void) {
    App_EmergencySleep();
    uint32_t now = HAL_GetTick();
    if (reboot_pending && now - reboot_at >= 30) NVIC_SystemReset();
    if (now - led_at >= 100) {
        Board_LED_Set((Board_LED_t)led, false);
        led = (led + 1) % BOARD_LED_COUNT;
        Board_LED_Set((Board_LED_t)led, true);
        led_at = now;
    }
    unsigned len = pending;
    if (len && HAL_FDCAN_GetTxFifoFreeLevel(&hfdcan2)) {
        uint8_t out[64];
        unsigned length = App_Request(mailbox, len, out);
        FDCAN_TxHeaderTypeDef header = {0};
        header.Identifier = 0x580U + node;
        header.IdType = FDCAN_STANDARD_ID;
        header.TxFrameType = FDCAN_DATA_FRAME;
        header.FDFormat = FDCAN_FD_CAN;
        header.BitRateSwitch = FDCAN_BRS_ON;
        header.DataLength = length > 8 ? FDCAN_DLC_BYTES_24 : FDCAN_DLC_BYTES_8;
        HAL_FDCAN_AddMessageToTxFifoQ(&hfdcan2, &header, out);
        __DMB();
        pending = 0;
    }
}
unsigned App_USB(char *line, char *out, size_t size, uint32_t epoch) {
    (void)epoch;
    if (!strcmp(line, "node uid"))
        return snprintf(out, size, "%08lX%08lX%08lX\r\n",
                        (unsigned long)HAL_GetUIDw0(), (unsigned long)HAL_GetUIDw1(),
                        (unsigned long)HAL_GetUIDw2());
    if (!strcmp(line, "node reboot")) {
        reboot_pending = 1; reboot_at = HAL_GetTick();
        return snprintf(out, size, "OK LoadMeFirst reboot requested\r\n");
    }
    return snprintf(out, size, "ERR LoadMeFirst: use hello, status, or node reboot; motor/config services unavailable\r\n");
}
