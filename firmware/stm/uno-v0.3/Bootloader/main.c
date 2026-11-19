/* Krabby v0.3 OpenBLT board integration. GPL-3.0-or-later. */
#include "main.h"
#include "boot.h"
static unsigned node_id;
unsigned long KrabbyBootRxId(void) { return 0x667UL + node_id; }
unsigned long KrabbyBootTxId(void) { return 0x7E1UL + node_id; }
void SystemClock_Config(void);
void SysTick_Handler(void) { HAL_IncTick(); }
void Error_Handler(void) { __disable_irq(); for (;;) {} }
void HAL_FDCAN_MspInit(FDCAN_HandleTypeDef *can)
{
    (void)can;
    RCC_PeriphCLKInitTypeDef clock = {0};
    clock.PeriphClockSelection = RCC_PERIPHCLK_FDCAN;
    clock.FdcanClockSelection = RCC_FDCANCLKSOURCE_PCLK1;
    if (HAL_RCCEx_PeriphCLKConfig(&clock) != HAL_OK) Error_Handler();
    __HAL_RCC_FDCAN_CLK_ENABLE();
    GPIO_InitTypeDef pins = {0};
    pins.Pin = GPIO_PIN_12 | GPIO_PIN_13;
    pins.Mode = GPIO_MODE_AF_PP;
    pins.Alternate = GPIO_AF9_FDCAN2;
    HAL_GPIO_Init(GPIOB, &pins);
}
int main(void)
{
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_GPIOC_CLK_ENABLE();
    /* Assert motor sleep LOW before HAL, clock, or CAN initialization. */
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_10|GPIO_PIN_14|GPIO_PIN_15, GPIO_PIN_RESET);
    GPIO_InitTypeDef pins = {0};
    pins.Pin = GPIO_PIN_10|GPIO_PIN_14|GPIO_PIN_15;
    pins.Mode = GPIO_MODE_OUTPUT_PP;
    HAL_GPIO_Init(GPIOB, &pins);
    pins.Pin = GPIO_PIN_8;
    HAL_GPIO_Init(GPIOC, &pins);
    pins.Mode = GPIO_MODE_INPUT;
    pins.Pin = GPIO_PIN_13;
    HAL_GPIO_Init(GPIOC, &pins);
    pins.Pin = GPIO_PIN_4|GPIO_PIN_5;
    HAL_GPIO_Init(GPIOB, &pins);
    node_id = (HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_13)==GPIO_PIN_RESET ? 4U:0U)
            | (HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_4)==GPIO_PIN_RESET ? 2U:0U)
            | (HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_5)==GPIO_PIN_RESET ? 1U:0U);
    HAL_Init();
    SystemClock_Config();
    BootInit();
    unsigned long led_tick = 0;
    for (;;) {
        BootTask();
        if ((unsigned long)(TimerGet()-led_tick) >= 100U) {
            led_tick = TimerGet();
            HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_8);
        }
    }
}

void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI48|RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.HSI48State = RCC_HSI48_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = RCC_PLLM_DIV1;
  RCC_OscInitStruct.PLL.PLLN = 20;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
  RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}
