cmake_minimum_required(VERSION 3.22)

project(krabby_leg_board LANGUAGES C)

if(CMAKE_CROSSCOMPILING)
    enable_language(ASM)
endif()

set(KRABBY_LEG_FW_VERSION "0.1.0-dev" CACHE STRING "Leg firmware version string")
set(KRABBY_LEG_FW_GIT_SHA "dev-local" CACHE STRING "Git SHA embedded in NodeStatus")
option(KRABBY_STM_BUILD_FIRMWARE "Build MCU app/bootloader targets" OFF)
option(KRABBY_STM_HOST_LIBRARY "Build host libkrabby_stm_host for sim/tests" OFF)

set(KRABBY_STM_CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/core)
set(KRABBY_STM_DBC_GEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dbc/generated)

add_library(krabby_leg_version INTERFACE)
target_include_directories(krabby_leg_version INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${KRABBY_STM_CORE_DIR}
)

set(KRABBY_STM_CORE_SOURCES
    ${KRABBY_STM_CORE_DIR}/app.c
    ${KRABBY_STM_CORE_DIR}/calibration.c
    ${KRABBY_STM_CORE_DIR}/can_handler.c
    ${KRABBY_STM_CORE_DIR}/leg_access.c
    ${KRABBY_STM_CORE_DIR}/telemetry_handler.c
)

if(KRABBY_STM_HOST_LIBRARY)
    add_library(krabby_stm_host SHARED
        ${KRABBY_STM_CORE_SOURCES}
        ${KRABBY_STM_DBC_GEN_DIR}/krabby.c
        ${CMAKE_CURRENT_SOURCE_DIR}/sim/board_pins_isaac.c
        ${CMAKE_CURRENT_SOURCE_DIR}/sim/can_sim.c
    )
    target_include_directories(krabby_stm_host PUBLIC
        ${KRABBY_STM_CORE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/sim
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${KRABBY_STM_DBC_GEN_DIR}
    )
    set_target_properties(krabby_stm_host PROPERTIES
        OUTPUT_NAME krabby_stm_host
        POSITION_INDEPENDENT_CODE ON
    )
endif()

if(KRABBY_STM_BUILD_FIRMWARE)
    if(CMAKE_CROSSCOMPILING)
        if(NOT KRABBY_HW_FIRMWARE)
            set(KRABBY_HW_FIRMWARE "${CMAKE_CURRENT_SOURCE_DIR}/../../hardware/PCB-v0.3/Firmware" CACHE PATH
                "Reference CubeMX tree (linker + startup + HAL)")
        endif()
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/cubemx_from_hw.cmake)
    endif()

    add_library(krabby_leg_core STATIC
        ${KRABBY_STM_CORE_SOURCES}
        ${KRABBY_STM_DBC_GEN_DIR}/krabby.c
    )
    target_include_directories(krabby_leg_core PUBLIC
        ${KRABBY_STM_CORE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${KRABBY_STM_DBC_GEN_DIR}
    )
    set(KRABBY_LEG_BOARD_HW_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/board/board_pins_stm32.c
        ${CMAKE_CURRENT_SOURCE_DIR}/board/can_stm32.c
        ${CMAKE_CURRENT_SOURCE_DIR}/board/stm32_bringup.c
    )
    if(CMAKE_CROSSCOMPILING)
        list(APPEND KRABBY_LEG_BOARD_HW_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/board/stm32_cubemx_init.c)
    endif()
    add_library(krabby_leg_board_hw STATIC ${KRABBY_LEG_BOARD_HW_SOURCES})
    target_include_directories(krabby_leg_board_hw PUBLIC
        ${KRABBY_STM_CORE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/board
    )
    target_link_libraries(krabby_leg_board_hw PUBLIC krabby_leg_core)
    if(CMAKE_CROSSCOMPILING)
        target_link_libraries(krabby_leg_board_hw PUBLIC krabby_cubemx_config)
    endif()

    add_executable(krabby_leg_app ${CMAKE_CURRENT_SOURCE_DIR}/board/main.c)
    target_include_directories(krabby_leg_app PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${KRABBY_STM_CORE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/board
    )
    target_compile_definitions(krabby_leg_app PRIVATE
        KRABBY_LEG_FW_VERSION="${KRABBY_LEG_FW_VERSION}"
        KRABBY_LEG_FW_GIT_SHA="${KRABBY_LEG_FW_GIT_SHA}"
    )
    target_link_libraries(krabby_leg_app PRIVATE krabby_leg_core krabby_leg_board_hw krabby_leg_version)

    if(CMAKE_CROSSCOMPILING)
        target_link_libraries(krabby_leg_app PRIVATE krabby_cubemx_config)
        krabby_stm_link_cubemx(krabby_leg_app)
    endif()
else()
    if(NOT KRABBY_STM_HOST_LIBRARY)
        add_custom_target(krabby_leg_stub ALL
            COMMENT "STM leg tree (enable KRABBY_STM_BUILD_FIRMWARE or KRABBY_STM_HOST_LIBRARY)"
        )
    endif()
endif()
