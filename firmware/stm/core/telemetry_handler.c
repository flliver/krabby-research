/*
 * TX path: six LegTelemetry_* frames cover the full robot joint snapshot (18 joints),
 * one aggregated frame per board per tick — not 18 per-motor frames.
 *
 * Each LegTelemetry_* packs telemetry for all three joints on this leg, then one
 * can_send() with krabby_can_leg_telemetry_id(leg_index). Schedule NodeStatus
 * separately (1 Hz + on boot), not inside every joint-rate tick.
 */
#include "telemetry_handler.h"

void telemetry_handler_tick(leg_t *leg, uint32_t now_ms) {
    (void)leg;
    (void)now_ms;
    /* TX: krabby_can_leg_telemetry_id(leg_index) — one frame, three joints packed. */
    /* TX: krabby_can_node_status_id(leg_index) — NodeStatus at 1 Hz, separate from LegTelemetry. */
}
