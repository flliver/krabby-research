/**
 * Leg-board ISRs for krabby_leg_app (no reference app_service / App_Tick).
 * CubeMX HAL MSP still comes from firmware/stm/uno-v0.3.
 */
#include "main.h"
#include "stm32g4xx_it.h"

extern FDCAN_HandleTypeDef hfdcan2;

void HardFault_Handler(void) {
    while (1) {
    }
}

void MemManage_Handler(void) {
    while (1) {
    }
}

void BusFault_Handler(void) {
    while (1) {
    }
}

void UsageFault_Handler(void) {
    while (1) {
    }
}

void SysTick_Handler(void) {
    HAL_IncTick();
}

void FDCAN2_IT0_IRQHandler(void) {
    HAL_FDCAN_IRQHandler(&hfdcan2);
}