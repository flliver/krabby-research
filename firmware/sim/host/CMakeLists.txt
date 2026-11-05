# Host Arduino shim (pin state + sim clock). Linked by native tests and future host firmware targets.

set(KRABBY_ARDUINO_DIR "${CMAKE_CURRENT_LIST_DIR}/../../arduino")

# Shared register model has no Arduino/Wire dependency.
add_library(ina228_native_device STATIC ../devices/ina228.cpp)
target_include_directories(ina228_native_device PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../devices")
set_target_properties(ina228_native_device PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
    target_compile_options(ina228_native_device PRIVATE /W4 /WX)
else()
    target_compile_options(ina228_native_device PRIVATE -Wall -Wextra -Werror)
endif()

# This driver build sees runtime Wire, never the test-only TwoWire fake.
add_library(sparkfun_sim_runtime STATIC ../vendor/SPI.cpp)
target_include_directories(sparkfun_sim_runtime PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../vendor")
set_target_properties(sparkfun_sim_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/sparkfun_ina228.cmake")
add_sparkfun_ina228_driver(sparkfun_ina2xx_sim sparkfun_sim_runtime)

add_library(krabby_sim_host STATIC
    pin_state.cpp
    sim_clock.cpp
    sim_i2c.cpp
    sim_imu_model.cpp
    sim_battery_model.cpp
    sim_ssd1306.cpp
    Wire.cpp
    host_leader_imu.cpp
    host_battery_telemetry.cpp
    host_oled_canvas.cpp
    eeprom_stub.cpp
    hall_sim.cpp
    hall_hw_stub.cpp
    sim_serial.cpp
    host_board_role.cpp
)
target_link_libraries(krabby_sim_host PUBLIC ina228_native_device sparkfun_ina2xx_sim)
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
        host_display.cpp
        "${KRABBY_ARDUINO_DIR}/src/display/display_frame_model.cpp"
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
