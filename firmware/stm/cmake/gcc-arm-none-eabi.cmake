# Toolchain for STM32G474 leg firmware (same CPU/FPU flags as hardware/PCB-v0.3/Firmware).

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER_ID GNU)

set(TOOLCHAIN_PREFIX arm-none-eabi-)
set(CMAKE_C_COMPILER ${TOOLCHAIN_PREFIX}gcc)
set(CMAKE_ASM_COMPILER ${CMAKE_C_COMPILER})
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}g++)
set(CMAKE_LINKER ${TOOLCHAIN_PREFIX}g++)
set(CMAKE_OBJCOPY ${TOOLCHAIN_PREFIX}objcopy)
set(CMAKE_SIZE ${TOOLCHAIN_PREFIX}size)

set(CMAKE_EXECUTABLE_SUFFIX_ASM ".elf")
set(CMAKE_EXECUTABLE_SUFFIX_C ".elf")
set(CMAKE_EXECUTABLE_SUFFIX_CXX ".elf")

set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(_TARGET_FLAGS "-mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard ")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${_TARGET_FLAGS}")
set(CMAKE_ASM_FLAGS "${CMAKE_C_FLAGS} -x assembler-with-cpp -MMD -MP")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -fdata-sections -ffunction-sections")

set(CMAKE_C_FLAGS_DEBUG "-O0 -g3")
set(CMAKE_C_FLAGS_RELEASE "-Os -g0")

if(NOT KRABBY_HW_FIRMWARE)
    set(KRABBY_HW_FIRMWARE "${CMAKE_SOURCE_DIR}/../../hardware/PCB-v0.3/Firmware" CACHE PATH
        "Reference CubeMX tree (linker + startup + HAL)")
endif()

set(_LINKER_SCRIPT "${KRABBY_HW_FIRMWARE}/STM32G474xx_FLASH.ld")
if(NOT EXISTS "${_LINKER_SCRIPT}")
    message(FATAL_ERROR "Missing linker script: ${_LINKER_SCRIPT}")
endif()

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${_TARGET_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -T \"${_LINKER_SCRIPT}\"")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --specs=nano.specs")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-Map=krabby_leg_app.map -Wl,--gc-sections")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--print-memory-usage")
set(TOOLCHAIN_LINK_LIBRARIES m)