cmake_minimum_required(VERSION 3.22)

project(krabby_leg_board LANGUAGES C)

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
    ${KRABBY_STM_CORE_DIR}/can_handler.c
    ${KRABBY_STM_CORE_DIR}/telemetry_handler.c
)

if(KRABBY_STM_HOST_LIBRARY)
    add_library(krabby_stm_host SHARED
        ${KRABBY_STM_CORE_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/sim/board_sim.c
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
    add_library(krabby_leg_core STATIC ${KRABBY_STM_CORE_SOURCES})
    target_include_directories(krabby_leg_core PUBLIC
        ${KRABBY_STM_CORE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${KRABBY_STM_DBC_GEN_DIR}
    )
    add_library(krabby_leg_board_hw STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/board/board_hw.c
    )
    target_include_directories(krabby_leg_board_hw PUBLIC ${KRABBY_STM_CORE_DIR})
    target_link_libraries(krabby_leg_board_hw PUBLIC krabby_leg_core)
    add_subdirectory(app)
else()
    if(NOT KRABBY_STM_HOST_LIBRARY)
        add_custom_target(krabby_leg_stub ALL
            COMMENT "STM leg tree (enable KRABBY_STM_BUILD_FIRMWARE or KRABBY_STM_HOST_LIBRARY)"
        )
    endif()
endif()
