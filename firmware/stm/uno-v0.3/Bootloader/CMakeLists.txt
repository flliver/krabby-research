cmake_minimum_required(VERSION 3.22)
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_C_STANDARD 11)
project(krabby_boot C ASM)
set(ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(BLT ${ROOT}/vendor/openblt/Target/Source)
set(HAL ${ROOT}/Drivers/STM32G4xx_HAL_Driver)
add_executable(krabby_boot main.c hooks.c can.c
    ${ROOT}/startup_stm32g474xx.s ${ROOT}/Src/system_stm32g4xx.c
    ${BLT}/boot.c ${BLT}/asserts.c ${BLT}/backdoor.c ${BLT}/com.c
    ${BLT}/cop.c ${BLT}/xcp.c
    ${BLT}/ARMCM4_STM32G4/cpu.c ${BLT}/ARMCM4_STM32G4/GCC/cpu_comp.c
    ${BLT}/ARMCM4_STM32G4/flash.c ${BLT}/ARMCM4_STM32G4/nvm.c
    ${BLT}/ARMCM4_STM32G4/timer.c
    ${HAL}/Src/stm32g4xx_hal.c ${HAL}/Src/stm32g4xx_hal_cortex.c
    ${HAL}/Src/stm32g4xx_hal_gpio.c ${HAL}/Src/stm32g4xx_hal_fdcan.c
    ${HAL}/Src/stm32g4xx_hal_rcc.c ${HAL}/Src/stm32g4xx_hal_rcc_ex.c
    ${HAL}/Src/stm32g4xx_hal_flash.c ${HAL}/Src/stm32g4xx_hal_flash_ex.c
    ${HAL}/Src/stm32g4xx_hal_pwr.c ${HAL}/Src/stm32g4xx_hal_pwr_ex.c)
target_include_directories(krabby_boot PRIVATE . ${BLT} ${BLT}/ARMCM4_STM32G4
    ${ROOT}/Inc ${HAL}/Inc ${ROOT}/Drivers/CMSIS/Include
    ${ROOT}/Drivers/CMSIS/Device/ST/STM32G4xx/Include)
target_compile_definitions(krabby_boot PRIVATE STM32G474xx USE_HAL_DRIVER NDEBUG)
target_link_options(krabby_boot PRIVATE --specs=nosys.specs)
add_custom_command(TARGET krabby_boot POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:krabby_boot> krabby_boot.bin
    COMMAND ${CMAKE_OBJCOPY} -O srec $<TARGET_FILE:krabby_boot> krabby_boot.srec)
