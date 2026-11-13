/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32g4xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

void HAL_TIM_MspPostInit(TIM_HandleTypeDef *htim);

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define NODE_ID0_Pin GPIO_PIN_13
#define NODE_ID0_GPIO_Port GPIOC
#define M1_FAULT_Pin GPIO_PIN_14
#define M1_FAULT_GPIO_Port GPIOC
#define M2_FAULT_Pin GPIO_PIN_15
#define M2_FAULT_GPIO_Port GPIOC
#define M3_DIR_Pin GPIO_PIN_1
#define M3_DIR_GPIO_Port GPIOC
#define M1_LEFT_LED_CTRL_Pin GPIO_PIN_2
#define M1_LEFT_LED_CTRL_GPIO_Port GPIOC
#define M1_RIGHT_LED_CTRL_Pin GPIO_PIN_3
#define M1_RIGHT_LED_CTRL_GPIO_Port GPIOC
#define M2_LEFT_LED_CTRL_Pin GPIO_PIN_3
#define M2_LEFT_LED_CTRL_GPIO_Port GPIOA
#define M2_RIGHT_LED_CTRL_Pin GPIO_PIN_4
#define M2_RIGHT_LED_CTRL_GPIO_Port GPIOA
#define M1_DIR_Pin GPIO_PIN_0
#define M1_DIR_GPIO_Port GPIOB
#define M2_DIR_Pin GPIO_PIN_1
#define M2_DIR_GPIO_Port GPIOB
#define M1_SLEEP_Pin GPIO_PIN_10
#define M1_SLEEP_GPIO_Port GPIOB
#define M3_FAULT_Pin GPIO_PIN_11
#define M3_FAULT_GPIO_Port GPIOB
#define M3_SLEEP_Pin GPIO_PIN_14
#define M3_SLEEP_GPIO_Port GPIOB
#define M2_SLEEP_Pin GPIO_PIN_15
#define M2_SLEEP_GPIO_Port GPIOB
#define STATUS_LED_Pin GPIO_PIN_8
#define STATUS_LED_GPIO_Port GPIOC
#define CAN_ACTIVITY_LED_Pin GPIO_PIN_9
#define CAN_ACTIVITY_LED_GPIO_Port GPIOC
#define M1_PWM_Pin GPIO_PIN_8
#define M1_PWM_GPIO_Port GPIOA
#define M2_PWM_Pin GPIO_PIN_9
#define M2_PWM_GPIO_Port GPIOA
#define M3_PWM_Pin GPIO_PIN_10
#define M3_PWM_GPIO_Port GPIOA
#define M3_RIGHT_LED_CTRL_Pin GPIO_PIN_2
#define M3_RIGHT_LED_CTRL_GPIO_Port GPIOD
#define NODE_ID1_Pin GPIO_PIN_4
#define NODE_ID1_GPIO_Port GPIOB
#define NODE_ID2_Pin GPIO_PIN_5
#define NODE_ID2_GPIO_Port GPIOB
#define M3_LEFT_LED_CTRL_Pin GPIO_PIN_6
#define M3_LEFT_LED_CTRL_GPIO_Port GPIOB

/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
