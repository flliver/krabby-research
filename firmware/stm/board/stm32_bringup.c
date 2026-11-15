#include "stm32_bringup.h"

#ifdef USE_HAL_DRIVER

#include "hw_handles.h"

#include "stm32_cubemx_init.h"

#include "stm32g4xx_hal.h"

FDCAN_HandleTypeDef hfdcan2;

extern ADC_HandleTypeDef hadc1;
extern ADC_HandleTypeDef hadc2;
extern I2C_HandleTypeDef hi2c1;
extern TIM_HandleTypeDef htim1;
extern TIM_HandleTypeDef htim2;
extern TIM_HandleTypeDef htim5;
extern TIM_HandleTypeDef htim8;

static void system_clock_config(void);
static void mx_fdcan2_init(void);

void Error_Handler(void) {
    __disable_irq();
    for (;;) {
    }
}

static void system_clock_config(void) {
    RCC_OscInitTypeDef osc = {0};
    RCC_ClkInitTypeDef clk = {0};

    HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1);

    osc.OscillatorType = RCC_OSCILLATORTYPE_HSI48 | RCC_OSCILLATORTYPE_HSE;
    osc.HSEState = RCC_HSE_ON;
    osc.HSI48State = RCC_HSI48_ON;
    osc.PLL.PLLState = RCC_PLL_ON;
    osc.PLL.PLLSource = RCC_PLLSOURCE_HSE;
    osc.PLL.PLLM = RCC_PLLM_DIV1;
    osc.PLL.PLLN = 20;
    osc.PLL.PLLP = RCC_PLLP_DIV2;
    osc.PLL.PLLQ = RCC_PLLQ_DIV2;
    osc.PLL.PLLR = RCC_PLLR_DIV2;
    if (HAL_RCC_OscConfig(&osc) != HAL_OK) {
        Error_Handler();
    }

    clk.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    clk.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    clk.AHBCLKDivider = RCC_SYSCLK_DIV1;
    clk.APB1CLKDivider = RCC_HCLK_DIV1;
    clk.APB2CLKDivider = RCC_HCLK_DIV1;
    if (HAL_RCC_ClockConfig(&clk, FLASH_LATENCY_2) != HAL_OK) {
        Error_Handler();
    }
}

/* CubeMX FDCAN2 — 1 Mbit/s nominal, 5 Mbit/s data @ PCLK1 80 MHz (hardware/PCB-v0.3/Firmware/Src/main.c). */
static void mx_fdcan2_init(void) {
    hfdcan2.Instance = FDCAN2;
    hfdcan2.Init.ClockDivider = FDCAN_CLOCK_DIV1;
    hfdcan2.Init.FrameFormat = FDCAN_FRAME_FD_BRS;
    hfdcan2.Init.Mode = FDCAN_MODE_NORMAL;
    hfdcan2.Init.AutoRetransmission = ENABLE;
    hfdcan2.Init.TransmitPause = DISABLE;
    hfdcan2.Init.ProtocolException = DISABLE;
    hfdcan2.Init.NominalPrescaler = 5;
    hfdcan2.Init.NominalSyncJumpWidth = 1;
    hfdcan2.Init.NominalTimeSeg1 = 13;
    hfdcan2.Init.NominalTimeSeg2 = 2;
    hfdcan2.Init.DataPrescaler = 1;
    hfdcan2.Init.DataSyncJumpWidth = 1;
    hfdcan2.Init.DataTimeSeg1 = 13;
    hfdcan2.Init.DataTimeSeg2 = 2;
    hfdcan2.Init.StdFiltersNbr = 0;
    hfdcan2.Init.ExtFiltersNbr = 0;
    hfdcan2.Init.TxFifoQueueMode = FDCAN_TX_FIFO_OPERATION;
    if (HAL_FDCAN_Init(&hfdcan2) != HAL_OK) {
        Error_Handler();
    }
}

void stm32_hw_init(void) {
    SCB->VTOR = KRABBY_APPLICATION_BASE;
    HAL_Init();
    system_clock_config();
    stm32_cubemx_periph_init();
    mx_fdcan2_init();
}

#else

void stm32_hw_init(void) {}

#endif