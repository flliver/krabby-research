#include "can_bridge.h"
#include "main.h"
#include "board.h"
#include "app_service.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

extern FDCAN_HandleTypeDef hfdcan2;
static const uint8_t lengths[16] = {0,1,2,3,4,5,6,7,8,12,16,20,24,32,48,64};
#define QUEUE_SIZE 64U
typedef struct {
    FDCAN_RxHeaderTypeDef header;
    uint8_t data[64];
} Frame;
static Frame queue[QUEUE_SIZE];
/* Monotonic single-producer/single-consumer indices; all 64 slots usable. */
static volatile uint32_t head, tail;
static volatile uint32_t dropped, hardware_lost, rx_errors;
static volatile uint32_t received, rx_interrupts;

uint16_t CAN_BridgeStatus(char *out, size_t size)
{
    FDCAN_ProtocolStatusTypeDef status = {0};
    FDCAN_ErrorCountersTypeDef errors = {0};
    /* Reading protocol status resets last-error indications to 'no change'. */
    HAL_FDCAN_GetProtocolStatus(&hfdcan2, &status);
    HAL_FDCAN_GetErrorCounters(&hfdcan2, &errors);
    return (uint16_t)snprintf(out, size,
        "CAN clk=%lu INIT=%u BO=%lu EP=%lu TEC=%lu REC=%lu LEC=%lu DLEC=%lu\r\n"
        "irq=%lu rx=%lu fifo=%lu queued=%lu dropped=%lu hwloss=%lu rxerr=%lu RXpin=%u\r\n"
        "CCCR=%08lX NBTP=%08lX DBTP=%08lX IR=%08lX IE=%08lX TDCR=%08lX\r\n",
        (unsigned long)HAL_RCCEx_GetPeriphCLKFreq(RCC_PERIPHCLK_FDCAN),
        (hfdcan2.Instance->CCCR & FDCAN_CCCR_INIT) != 0,
        (unsigned long)status.BusOff, (unsigned long)status.ErrorPassive,
        (unsigned long)errors.TxErrorCnt, (unsigned long)errors.RxErrorCnt,
        (unsigned long)status.LastErrorCode, (unsigned long)status.DataLastErrorCode,
        (unsigned long)rx_interrupts, (unsigned long)received,
        (unsigned long)HAL_FDCAN_GetRxFifoFillLevel(&hfdcan2, FDCAN_RX_FIFO0),
        (unsigned long)(head - tail), (unsigned long)dropped,
        (unsigned long)hardware_lost, (unsigned long)rx_errors,
        HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_12) == GPIO_PIN_SET,
        (unsigned long)hfdcan2.Instance->CCCR, (unsigned long)hfdcan2.Instance->NBTP,
        (unsigned long)hfdcan2.Instance->DBTP, (unsigned long)hfdcan2.Instance->IR,
        (unsigned long)hfdcan2.Instance->IE, (unsigned long)hfdcan2.Instance->TDCR);
}

void CAN_BridgeInit(void)
{
    /* TDCO follows ST's G4 example. M_CAN permits TDC only with functional
       data prescaler 1 or 2 (encoded DBRP 0 or 1). Larger prescalers must
       use the normal data sample point for transmitter bit monitoring. */
    if (HAL_FDCAN_ConfigTxDelayCompensation(&hfdcan2,
            hfdcan2.Init.DataTimeSeg1 * hfdcan2.Init.DataPrescaler, 0) != HAL_OK ||
        (hfdcan2.Init.DataPrescaler <= 2U ?
            HAL_FDCAN_EnableTxDelayCompensation(&hfdcan2) :
            HAL_FDCAN_DisableTxDelayCompensation(&hfdcan2)) != HAL_OK)
        Error_Handler();
    /* No filters needed: accept all standard/extended data and remote frames. */
    if (HAL_FDCAN_ConfigGlobalFilter(&hfdcan2, FDCAN_ACCEPT_IN_RX_FIFO0,
            FDCAN_ACCEPT_IN_RX_FIFO0, FDCAN_FILTER_REMOTE, FDCAN_FILTER_REMOTE) != HAL_OK ||
        HAL_FDCAN_Start(&hfdcan2) != HAL_OK ||
        HAL_FDCAN_ActivateNotification(&hfdcan2,
            FDCAN_IT_RX_FIFO0_NEW_MESSAGE | FDCAN_IT_RX_FIFO0_MESSAGE_LOST, 0) != HAL_OK)
        Error_Handler();
}

void HAL_FDCAN_RxFifo0Callback(FDCAN_HandleTypeDef *can, uint32_t flags)
{
    if (can != &hfdcan2) return;
    ++rx_interrupts;
    if (flags & FDCAN_IT_RX_FIFO0_MESSAGE_LOST) ++hardware_lost;
    /* Drain hardware even when software queue is full; never wait for USB. */
    while (HAL_FDCAN_GetRxFifoFillLevel(can, FDCAN_RX_FIFO0)) {
        Frame frame;
        if (HAL_FDCAN_GetRxMessage(can, FDCAN_RX_FIFO0, &frame.header, frame.data) != HAL_OK) {
            ++rx_errors;
            break;
        }
        ++received;
        /* Same XCP CONNECT re-entry used by OpenBLT's demo application.
           The bootloader then accepts the host's repeated CONNECT request. */
        if (frame.header.IdType == FDCAN_STANDARD_ID &&
            frame.header.RxFrameType == FDCAN_DATA_FRAME &&
            frame.header.Identifier == 0x667U + Board_GetNodeId() &&
            frame.header.DataLength == FDCAN_DLC_BYTES_2 && frame.data[0] == 0xFFU) {
            App_EmergencySleep();
            NVIC_SystemReset();
        }
        if (frame.header.IdType == FDCAN_STANDARD_ID &&
            frame.header.RxFrameType == FDCAN_DATA_FRAME &&
            frame.header.FDFormat == FDCAN_FD_CAN && frame.header.BitRateSwitch == FDCAN_BRS_ON &&
            App_CANReceive(frame.header.Identifier, frame.data, lengths[frame.header.DataLength & 15U])) continue;
        if ((uint32_t)(head - tail) >= QUEUE_SIZE) {
            ++dropped;
        } else {
            queue[head % QUEUE_SIZE] = frame;
            __DMB();
            ++head;
        }
    }
}

uint16_t CAN_BridgeRead(char *out, size_t size)
{
    static uint32_t seen_drop, seen_lost, seen_errors;
    uint32_t d = dropped, h = hardware_lost, e = rx_errors;
    if (d != seen_drop || h != seen_lost || e != seen_errors) {
        seen_drop = d; seen_lost = h; seen_errors = e;
        return (uint16_t)snprintf(out, size,
            "CAN overflow: queue_dropped=%lu hw_loss_events=%lu rx_errors=%lu\r\n",
            (unsigned long)d, (unsigned long)h, (unsigned long)e);
    }
    if (tail == head) return 0;
    __DMB();
    Frame frame = queue[tail % QUEUE_SIZE];
    __DMB();
    ++tail;
    FDCAN_RxHeaderTypeDef *r = &frame.header;
    unsigned dlc = r->DataLength & 15U;
    unsigned length = lengths[dlc];
    if (r->FDFormat == FDCAN_CLASSIC_CAN && length > 8) length = 8;
    int n = snprintf(out, size, "RX %s %08lX %s %s BRS=%u DLC=%X len=%u",
        r->IdType == FDCAN_STANDARD_ID ? "STD" : "EXT", (unsigned long)r->Identifier,
        r->FDFormat == FDCAN_FD_CAN ? "FD" : "CAN",
        r->RxFrameType == FDCAN_REMOTE_FRAME ? "RTR" : "DATA",
        r->BitRateSwitch == FDCAN_BRS_ON, dlc, length);
    if (r->RxFrameType != FDCAN_REMOTE_FRAME)
        for (unsigned i = 0; i < length; ++i)
            n += snprintf(out + n, size - (size_t)n, " %02X", frame.data[i]);
    n += snprintf(out + n, size - (size_t)n, "\r\n");
    return (uint16_t)n;
}

static int hex_value(const char *s, uint32_t max, uint32_t *value)
{
    char *end;
    if (!s || !*s || *s == '-' || *s == '+') return 0;
    unsigned long v = strtoul(s, &end, 16);
    if (*end || v > max) return 0;
    *value = (uint32_t)v;
    return 1;
}

uint16_t CAN_BridgeCommand(char *line, char *out, size_t size)
{
    /* tx = classic, txfd = FD+BRS, txfdn = FD without BRS.
       Append x to command for an explicit 29-bit identifier, including low IDs. */
    char *cmd = strtok(line, " \t");
    unsigned extended = cmd && (!strcmp(cmd, "txx") ||
        !strcmp(cmd, "txfdx") || !strcmp(cmd, "txfdnx"));
    if (extended) cmd[strlen(cmd)-1] = 0;
    unsigned fd = cmd && (!strcmp(cmd, "txfd") || !strcmp(cmd, "txfdn"));
    unsigned brs = cmd && !strcmp(cmd, "txfd");
    uint32_t id, byte;
    uint8_t data[64] = {0};
    unsigned count = 0, dlc;
    if (!cmd || (!fd && strcmp(cmd, "tx")) ||
        !hex_value(strtok(NULL, " \t"), extended ? 0x1FFFFFFFU : 0x7FFU, &id)) goto invalid;
    char *token;
    while ((token = strtok(NULL, " \t")) != NULL) {
        if (count >= (fd ? 64U : 8U) || !hex_value(token, 255, &byte)) goto invalid;
        data[count++] = (uint8_t)byte;
    }
    for (dlc = 0; dlc < 16 && lengths[dlc] != count; ++dlc) {}
    if (dlc == 16) goto invalid;
    FDCAN_ProtocolStatusTypeDef status;
    HAL_FDCAN_GetProtocolStatus(&hfdcan2, &status);
    if (status.BusOff)
        return (uint16_t)snprintf(out, size, "ERR CAN bus-off; fix bus and reset board\r\n");
    if (!HAL_FDCAN_GetTxFifoFreeLevel(&hfdcan2))
        return (uint16_t)snprintf(out, size, "ERR CAN TX full; check peer/ACK\r\n");
    FDCAN_TxHeaderTypeDef tx = {0};
    tx.Identifier = id;
    tx.IdType = extended ? FDCAN_EXTENDED_ID : FDCAN_STANDARD_ID;
    tx.TxFrameType = FDCAN_DATA_FRAME;
    tx.DataLength = dlc;
    tx.ErrorStateIndicator = FDCAN_ESI_ACTIVE;
    tx.BitRateSwitch = brs ? FDCAN_BRS_ON : FDCAN_BRS_OFF;
    tx.FDFormat = fd ? FDCAN_FD_CAN : FDCAN_CLASSIC_CAN;
    tx.TxEventFifoControl = FDCAN_NO_TX_EVENTS;
    return (uint16_t)snprintf(out, size, "%s\r\n",
        HAL_FDCAN_AddMessageToTxFifoQ(&hfdcan2, &tx, data) == HAL_OK ?
        "OK TX queued (not confirmation of delivery)" : "ERR CAN TX enqueue");
invalid:
    return (uint16_t)snprintf(out, size,
        "ERR use tx/txfd/txfdn HEX_ID [HEX_BYTES]; append x for EXT (txfdx); "
        "CAN:0..8 FD:0..8,12,16,20,24,32,48,64 bytes\r\n");
}
