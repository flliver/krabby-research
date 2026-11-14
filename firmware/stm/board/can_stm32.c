#include "board_can.h"

#include <stddef.h>
#include <string.h>

#define KRABBY_STM32_RX_DEPTH 16

typedef struct {
    uint32_t id;
    uint8_t data[64];
    uint8_t len;
} can_rx_slot_t;

typedef struct {
    can_rx_slot_t queue[KRABBY_STM32_RX_DEPTH];
    uint8_t head;
    uint8_t tail;
} can_rx_queue_t;

static can_rx_queue_t s_rx_queue;

void can_init(leg_t *leg) {
    (void)leg;
    memset(&s_rx_queue, 0, sizeof(s_rx_queue));
    /* FDCAN2 init — reference can_bridge.c in hardware/PCB-v0.3/Firmware/. */
}

bool can_send(leg_t *leg, uint32_t id, const uint8_t *data, uint8_t len) {
    (void)leg;
    (void)id;
    (void)data;
    (void)len;
    /* HAL_FDCAN_AddMessageToTxFifoQ. */
    return false;
}

bool can_receive(leg_t *leg, uint32_t *id, uint8_t *data, uint8_t *len) {
    (void)leg;
    if (id == NULL || data == NULL || len == NULL) {
        return false;
    }
    if (s_rx_queue.head == s_rx_queue.tail) {
        return false;
    }
    const can_rx_slot_t *slot = &s_rx_queue.queue[s_rx_queue.tail];
    s_rx_queue.tail = (uint8_t)((s_rx_queue.tail + 1u) % KRABBY_STM32_RX_DEPTH);
    *id = slot->id;
    *len = slot->len;
    memcpy(data, slot->data, slot->len);
    return true;
}