# Host Arduino shim (pin state + sim clock). Linked by native tests and future host firmware targets.

set(KRABBY_ARDUINO_DIR "${CMAKE_CURRENT_LIST_DIR}/../../arduino")

add_library(krabby_sim_host STATIC
    pin_state.cpp
    sim_clock.cpp
    sim_i2c.cpp
    sim_imu_model.cpp
    Wire.cpp
    host_leader_imu.cpp
    eeprom_stub.cpp
    hall_sim.cpp
    hall_hw_stub.cpp
    sim_serial.cpp
    host_board_role.cpp
)
target_include_directories(krabby_sim_host PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(krabby_sim_host PRIVATE "${KRABBY_ARDUINO_DIR}")
target_compile_definitions(krabby_sim_host PUBLIC KRABBY_HOST_FIRMWARE=1)
# krabby_host links these statics into a .so, so they need PIC as well.
set_target_properties(krabby_sim_host PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
    target_compile_options(krabby_sim_host PRIVATE /W4 /WX)
else()
    target_compile_options(krabby_sim_host PRIVATE -Wall -Wextra -Werror)
endif()

add_library(krabby_firmware_actuator_host STATIC actuator_manager_host.cpp)
target_compile_definitions(krabby_firmware_actuator_host PUBLIC KRABBY_HOST_FIRMWARE=1)
target_include_directories(krabby_firmware_actuator_host BEFORE PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_include_directories(krabby_firmware_actuator_host PUBLIC "${KRABBY_ARDUINO_DIR}")
target_link_libraries(krabby_firmware_actuator_host PUBLIC krabby_sim_host)
set_target_properties(krabby_firmware_actuator_host PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
    target_compile_options(krabby_firmware_actuator_host PRIVATE /W4 /WX)
else()
    target_compile_options(krabby_firmware_actuator_host PRIVATE -Wall -Wextra -Werror -Wno-sign-compare)
endif()

add_library(krabby_host SHARED host_api.cpp host_firmware_loop.cpp host_multi_board.cpp)
target_sources(
    krabby_host
    PRIVATE
        "${KRABBY_ARDUINO_DIR}/src/telemetry.cpp"
)
target_link_libraries(krabby_host PRIVATE krabby_firmware_actuator_host)
target_include_directories(krabby_host PRIVATE "${KRABBY_ARDUINO_DIR}")
if(MSVC)
    target_compile_options(krabby_host PRIVATE /W4 /WX)
else()
    target_compile_options(krabby_host PRIVATE -Wall -Wextra -Werror -Wno-sign-compare)
endif()
set_target_properties(krabby_host PROPERTIES POSITION_INDEPENDENT_CODE ON)
