#include "app.h"
#include "board.h"
#include "board_pins.h"

#include "krabby/version.h"

int main(void) {
    leg_t leg = {0};
    (void)KRABBY_LEG_FW_VERSION;
    (void)KRABBY_LEG_FW_GIT_SHA;
    app_init(&leg, board_pins_stm32());
    for (;;) {
        const uint32_t now_ms = leg_now_ms(&leg);
        app_tick(&leg, now_ms);
    }
}