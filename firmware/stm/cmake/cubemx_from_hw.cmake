# Reuse STM32CubeMX HAL + startup from hardware/PCB-v0.3/Firmware (OpenBLT linker layout).
# KRABBY_HW_FIRMWARE defaults there via Makefile / gcc-arm-none-eabi.cmake — update all three
# if the CubeMX tree moves or is copied under firmware/stm/.

if(NOT KRABBY_HW_FIRMWARE)
    message(FATAL_ERROR "KRABBY_HW_FIRMWARE must point at hardware/PCB-v0.3/Firmware")
endif()

set(_HW "${KRABBY_HW_FIRMWARE}")
set(_MX "${_HW}/cmake/stm32cubemx")

set(KRABBY_MX_Defines USE_HAL_DRIVER STM32G474xx)
set(KRABBY_MX_Include_Dirs
    ${_HW}/Inc
    ${_HW}/Drivers/STM32G4xx_HAL_Driver/Inc
    ${_HW}/Drivers/STM32G4xx_HAL_Driver/Inc/Legacy
    ${_HW}/Middlewares/ST/STM32_USB_Device_Library/Core/Inc
    ${_HW}/Middlewares/ST/STM32_USB_Device_Library/Class/CDC/Inc
    ${_HW}/Drivers/CMSIS/Device/ST/STM32G4xx/Include
    ${_HW}/Drivers/CMSIS/Include
)

set(KRABBY_MX_Crt_Src
    ${_HW}/startup_stm32g474xx.s
    ${_HW}/Src/system_stm32g4xx.c
    ${_HW}/Src/sysmem.c
    ${_HW}/Src/syscalls.c
    ${_HW}/Src/stm32g4xx_it.c
    ${_HW}/Src/stm32g4xx_hal_msp.c
)

set(KRABBY_STM32_HAL_Src
    ${_HW}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_hal_pcd.c
    ${_HW}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_hal_pcd_ex.c
    ${_HW}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_ll_usb.c
    ${_HW}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_hal.c
    ${_HW}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_hal_rcc.c
    ${_HW}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_hal_rcc_ex.c
    ${_HW}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_hal_flash.c
    ${_HW}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_hal_flash_ex.c
    ${_HW}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_hal_flash_ramfunc.c
    ${_HW}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_hal_gpio.c
    ${_HW}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_hal_exti.c
    ${_HW}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_hal_dma.c
    ${_HW}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_hal_dma_ex.c
    ${_HW}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_hal_pwr.c
    ${_HW}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_hal_pwr_ex.c
    ${_HW}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_hal_cortex.c
    ${_HW}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_hal_adc.c
    ${_HW}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_hal_adc_ex.c
    ${_HW}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_ll_adc.c
    ${_HW}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_hal_fdcan.c
    ${_HW}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_hal_i2c.c
    ${_HW}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_hal_i2c_ex.c
    ${_HW}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_hal_tim.c
    ${_HW}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_hal_tim_ex.c
)

set(KRABBY_USB_Src
    ${_HW}/Middlewares/ST/STM32_USB_Device_Library/Core/Src/usbd_core.c
    ${_HW}/Middlewares/ST/STM32_USB_Device_Library/Core/Src/usbd_ctlreq.c
    ${_HW}/Middlewares/ST/STM32_USB_Device_Library/Core/Src/usbd_ioreq.c
    ${_HW}/Middlewares/ST/STM32_USB_Device_Library/Class/CDC/Src/usbd_cdc.c
)

add_library(krabby_cubemx_config INTERFACE)
target_include_directories(krabby_cubemx_config INTERFACE ${KRABBY_MX_Include_Dirs})
target_compile_definitions(krabby_cubemx_config INTERFACE ${KRABBY_MX_Defines})

add_library(krabby_cubemx_hal OBJECT ${KRABBY_STM32_HAL_Src})
target_link_libraries(krabby_cubemx_hal PUBLIC krabby_cubemx_config)

add_library(krabby_cubemx_usb OBJECT ${KRABBY_USB_Src})
target_link_libraries(krabby_cubemx_usb PUBLIC krabby_cubemx_config)

function(krabby_stm_link_cubemx app_target)
    target_sources(${app_target} PRIVATE ${KRABBY_MX_Crt_Src})
    target_link_libraries(${app_target} PRIVATE
        krabby_cubemx_hal
        krabby_cubemx_usb
        m
    )
    set_target_properties(${app_target} PROPERTIES
        OUTPUT_NAME krabby_leg_app
        SUFFIX ".elf"
    )
endfunction()