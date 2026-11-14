#pragma once

#ifdef USE_HAL_DRIVER

#include "stm32g4xx_hal.h"

/* OpenBLT application base — keep aligned with hardware/PCB-v0.3/Firmware/App/Inc/app_boot.h */
#define KRABBY_APPLICATION_BASE 0x08003000UL

extern FDCAN_HandleTypeDef hfdcan2;
extern ADC_HandleTypeDef hadc1;
extern ADC_HandleTypeDef hadc2;
extern I2C_HandleTypeDef hi2c1;
extern TIM_HandleTypeDef htim1;
extern TIM_HandleTypeDef htim2;
extern TIM_HandleTypeDef htim5;
extern TIM_HandleTypeDef htim8;

#define KRABBY_CONFIG_BASE 0x0801F000UL

#endif