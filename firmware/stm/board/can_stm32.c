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
    volatile uint8_t head;
    volatile uint8_t tail;
} can_rx_queue_t;

static can_rx_queue_t s_rx_queue;

#ifdef USE_HAL_DRIVER

#include "hw_handles.h"

#include "stm32_bringup.h"

static const uint8_t s_fdcan_dlc_bytes[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64};

static uint8_t krabby_fdcan_dlc_for_length(uint8_t len) {
    for (uint8_t dlc = 0; dlc < 16u; ++dlc) {
        if (s_fdcan_dlc_bytes[dlc] >= len) {
            return dlc;
        }
    }
    return 0xFFu;
}

static void krabby_fdcan_start(void) {
    if (HAL_FDCAN_ConfigTxDelayCompensation(&hfdcan2, hfdcan2.Init.DataTimeSeg1 * hfdcan2.Init.DataPrescaler, 0) !=
            HAL_OK ||
        (hfdcan2.Init.DataPrescaler <= 2U ? HAL_FDCAN_EnableTxDelayCompensation(&hfdcan2)
                                        : HAL_FDCAN_DisableTxDelayCompensation(&hfdcan2)) != HAL_OK) {
        Error_Handler();
    }
    if (HAL_FDCAN_ConfigGlobalFilter(&hfdcan2, FDCAN_ACCEPT_IN_RX_FIFO0, FDCAN_ACCEPT_IN_RX_FIFO0, FDCAN_FILTER_REMOTE,
                                     FDCAN_FILTER_REMOTE) != HAL_OK ||
        HAL_FDCAN_Start(&hfdcan2) != HAL_OK ||
        HAL_FDCAN_ActivateNotification(&hfdcan2, FDCAN_IT_RX_FIFO0_NEW_MESSAGE | FDCAN_IT_RX_FIFO0_MESSAGE_LOST,
                                       0) != HAL_OK) {
        Error_Handler();
    }
}

static void krabby_fdcan_push_rx(uint32_t id, const uint8_t *data, uint8_t len) {
    uint8_t next = (uint8_t)((s_rx_queue.head + 1u) % KRABBY_STM32_RX_DEPTH);
    if (next == s_rx_queue.tail) {
        return;
    }
    can_rx_slot_t *slot = &s_rx_queue.queue[s_rx_queue.head];
    slot->id = id;
    slot->len = len;
    memcpy(slot->data, data, len);
    __DMB();
    s_rx_queue.head = next;
}

void HAL_FDCAN_RxFifo0Callback(FDCAN_HandleTypeDef *can, uint32_t flags) {
    if (can != &hfdcan2) {
        return;
    }
    (void)flags;
    while (HAL_FDCAN_GetRxFifoFillLevel(can, FDCAN_RX_FIFO0) > 0U) {
        FDCAN_RxHeaderTypeDef header;
        uint8_t payload[64];
        if (HAL_FDCAN_GetRxMessage(can, FDCAN_RX_FIFO0, &header, payload) != HAL_OK) {
            break;
        }
        if (header.IdType != FDCAN_STANDARD_ID || header.RxFrameType != FDCAN_DATA_FRAME) {
            continue;
        }
        const uint8_t dlc = (uint8_t)(header.DataLength & 15U);
        uint8_t len = s_fdcan_dlc_bytes[dlc];
        if (header.FDFormat == FDCAN_CLASSIC_CAN && len > 8u) {
            len = 8u;
        }
        if (len == 0u) {
            continue;
        }
        krabby_fdcan_push_rx(header.Identifier, payload, len);
    }
}

void can_init(leg_t *leg) {
    (void)leg;
    memset(&s_rx_queue, 0, sizeof(s_rx_queue));
    krabby_fdcan_start();
}

bool can_send(leg_t *leg, uint32_t id, const uint8_t *data, uint8_t len) {
    (void)leg;
    if (data == NULL || len == 0u || len > 64u || id > 0x7FFu) {
        return false;
    }
    const uint8_t dlc = krabby_fdcan_dlc_for_length(len);
    if (dlc == 0xFFu) {
        return false;
    }
    FDCAN_ProtocolStatusTypeDef status;
    HAL_FDCAN_GetProtocolStatus(&hfdcan2, &status);
    if (status.BusOff != 0U) {
        return false;
    }
    if (HAL_FDCAN_GetTxFifoFreeLevel(&hfdcan2) == 0U) {
        return false;
    }
    FDCAN_TxHeaderTypeDef header = {0};
    header.Identifier = id;
    header.IdType = FDCAN_STANDARD_ID;
    header.TxFrameType = FDCAN_DATA_FRAME;
    header.DataLength = dlc;
    header.ErrorStateIndicator = FDCAN_ESI_ACTIVE;
    header.BitRateSwitch = FDCAN_BRS_ON;
    header.FDFormat = FDCAN_FD_CAN;
    header.TxEventFifoControl = FDCAN_NO_TX_EVENTS;
    uint8_t wire[64] = {0};
    memcpy(wire, data, len);
    return HAL_FDCAN_AddMessageToTxFifoQ(&hfdcan2, &header, wire) == HAL_OK;
}

#else

void can_init(leg_t *leg) {
    (void)leg;
    memset(&s_rx_queue, 0, sizeof(s_rx_queue));
}

bool can_send(leg_t *leg, uint32_t id, const uint8_t *data, uint8_t len) {
    (void)leg;
    (void)id;
    (void)data;
    (void)len;
    return false;
}

#endif

bool can_receive(leg_t *leg, uint32_t *id, uint8_t *data, uint8_t *len) {
    (void)leg;
    if (id == NULL || data == NULL || len == NULL) {
        return false;
    }
    if (s_rx_queue.head == s_rx_queue.tail) {
        return false;
    }
    can_rx_slot_t slot;
#ifdef USE_HAL_DRIVER
    const uint32_t primask = __get_PRIMASK();
    __disable_irq();
    slot = s_rx_queue.queue[s_rx_queue.tail];
    s_rx_queue.tail = (uint8_t)((s_rx_queue.tail + 1u) % KRABBY_STM32_RX_DEPTH);
    __set_PRIMASK(primask);
#else
    slot = s_rx_queue.queue[s_rx_queue.tail];
    s_rx_queue.tail = (uint8_t)((s_rx_queue.tail + 1u) % KRABBY_STM32_RX_DEPTH);
#endif
    *id = slot.id;
    *len = slot.len;
    memcpy(data, slot.data, slot.len);
    return true;
}