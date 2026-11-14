#pragma once

void stm32_hw_init(void);

#ifdef USE_HAL_DRIVER
void Error_Handler(void);
#endif