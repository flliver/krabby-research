#include "can_sim.h"

#include "board_can.h"

#include <stddef.h>
#include <string.h>

#define KRABBY_CAN_SIM_LEGS 6
#define KRABBY_CAN_SIM_RX_DEPTH 8

typedef struct {
    uint32_t id;
    uint8_t data[64];
    uint8_t len;
} can_rx_slot_t;

typedef struct {
    uint8_t leg_index;
    can_rx_slot_t queue[KRABBY_CAN_SIM_RX_DEPTH];
    uint8_t head;
    uint8_t tail;
    uint32_t last_tx_id;
    uint8_t last_tx_data[64];
    uint8_t last_tx_len;
} can_sim_bus_t;

static can_sim_bus_t s_bus[KRABBY_CAN_SIM_LEGS];

void can_init(leg_t *leg) {
    if (leg == NULL || leg->leg_index >= KRABBY_CAN_SIM_LEGS) {
        return;
    }
    can_sim_bus_t *bus = &s_bus[leg->leg_index];
    memset(bus, 0, sizeof(*bus));
    bus->leg_index = leg->leg_index;
}

bool can_send(leg_t *leg, uint32_t can_id, const uint8_t *data, uint8_t len) {
    if (leg == NULL || data == NULL || leg->leg_index >= KRABBY_CAN_SIM_LEGS || len > 64u) {
        return false;
    }
    can_sim_bus_t *bus = &s_bus[leg->leg_index];
    bus->last_tx_id = can_id;
    bus->last_tx_len = len;
    memcpy(bus->last_tx_data, data, len);
    return true;
}

bool can_receive(leg_t *leg, uint32_t *id, uint8_t *data, uint8_t *len) {
    if (leg == NULL || id == NULL || data == NULL || len == NULL || leg->leg_index >= KRABBY_CAN_SIM_LEGS) {
        return false;
    }
    can_sim_bus_t *bus = &s_bus[leg->leg_index];
    if (bus->head == bus->tail) {
        return false;
    }
    const can_rx_slot_t *slot = &bus->queue[bus->tail];
    bus->tail = (uint8_t)((bus->tail + 1u) % KRABBY_CAN_SIM_RX_DEPTH);
    *id = slot->id;
    *len = slot->len;
    memcpy(data, slot->data, slot->len);
    return true;
}

void krabby_sim_inject_rx(leg_t *leg, uint32_t can_id, const uint8_t *data, uint8_t len) {
    if (leg == NULL || data == NULL || leg->leg_index >= KRABBY_CAN_SIM_LEGS || len > 64u) {
        return;
    }
    can_sim_bus_t *bus = &s_bus[leg->leg_index];
    const uint8_t next = (uint8_t)((bus->head + 1u) % KRABBY_CAN_SIM_RX_DEPTH);
    if (next == bus->tail) {
        return;
    }
    can_rx_slot_t *slot = &bus->queue[bus->head];
    slot->id = can_id;
    slot->len = len;
    memcpy(slot->data, data, len);
    bus->head = next;
}

int krabby_sim_last_tx(leg_t *leg, uint32_t *can_id, uint8_t *data, uint8_t *len) {
    if (leg == NULL || can_id == NULL || data == NULL || len == NULL || leg->leg_index >= KRABBY_CAN_SIM_LEGS) {
        return -1;
    }
    const can_sim_bus_t *bus = &s_bus[leg->leg_index];
    *can_id = bus->last_tx_id;
    *len = bus->last_tx_len;
    memcpy(data, bus->last_tx_data, *len);
    return 0;
}