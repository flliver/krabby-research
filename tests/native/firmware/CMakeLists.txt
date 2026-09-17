cmake_minimum_required(VERSION 3.20)

project(krabby_firmware_native_tests LANGUAGES C CXX)

option(KRABBY_NATIVE_COVERAGE "Instrument native firmware tests for coverage" OFF)

include(CTest)
include(FetchContent)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(KRABBY_NATIVE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-O0 -g --coverage)
        add_link_options(--coverage)
    else()
        message(FATAL_ERROR "Native coverage requires GCC or Clang")
    endif()
endif()

FetchContent_Declare(
    unity
    URL https://github.com/ThrowTheSwitch/Unity/archive/refs/tags/v2.6.1.tar.gz
    URL_HASH SHA256=b41a66d45a6b99758fb3202ace6178177014d52fc524bf1f72687d93e9867292
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(unity)

set(KRABBY_ARDUINO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../firmware/arduino")
set(KRABBY_OLED_SIM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../firmware/oled_sim/native")

# Test paths mirror firmware/arduino; target names use the basename.
function(add_firmware_native_test test_path)
    get_filename_component(test_name "${test_path}" NAME)
    add_executable("${test_name}" "${test_path}.cpp")
    target_include_directories("${test_name}" PRIVATE "${KRABBY_ARDUINO_DIR}")
    target_link_libraries("${test_name}" PRIVATE unity)
    if(MSVC)
        target_compile_options("${test_name}" PRIVATE /W4 /WX)
    else()
        target_compile_options("${test_name}" PRIVATE -Wall -Wextra -Werror)
    endif()
    add_test(NAME "${test_name}" COMMAND "${test_name}")
endfunction()

set(KRABBY_FIRMWARE_NATIVE_TESTS
    arduino/test_units
    arduino/test_actuator_manager
    arduino/src/test_telemetry
    arduino/src/imu/test_imu_calibrator
    arduino/src/imu/test_imu_measurement
    arduino/src/i2c/test_i2c_recovery
    arduino/src/display/test_display_frame_model
    arduino/src/display/test_display_renderer
    arduino/src/actuator/test_actuator_current_tracker
    arduino/src/actuator/test_actuator_status
    arduino/src/actuator/test_actuator_identity
    arduino/test_board_pins
    arduino/src/actuator/test_actuator_pot_tracker
    arduino/src/controller/test_board_role
    arduino/src/controller/test_controller_freshness_tracker
)

# Require every discovered test to appear in the manifest.
foreach(test_path IN LISTS KRABBY_FIRMWARE_NATIVE_TESTS)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${test_path}.cpp")
        message(FATAL_ERROR
            "'${test_path}' is listed in KRABBY_FIRMWARE_NATIVE_TESTS but "
            "${test_path}.cpp does not exist. Add the file or drop the entry.")
    endif()
    add_firmware_native_test("${test_path}")
endforeach()

target_include_directories(test_actuator_manager PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/fakes")
# Legacy calibration compares Arduino int fields with size_t and a magic literal.
if(NOT MSVC)
    target_compile_options(test_actuator_manager PRIVATE -Wno-sign-compare)
endif()

file(GLOB_RECURSE discovered_test_sources CONFIGURE_DEPENDS
     RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.cpp")
foreach(discovered IN LISTS discovered_test_sources)
    string(REGEX REPLACE "\\.cpp$" "" discovered_path "${discovered}")
    if(NOT discovered_path IN_LIST KRABBY_FIRMWARE_NATIVE_TESTS)
        message(FATAL_ERROR
            "${discovered} is not in KRABBY_FIRMWARE_NATIVE_TESTS, so it would "
            "never build or run. Add '${discovered_path}' to the list.")
    endif()
endforeach()

target_sources(
    test_display_frame_model
    PRIVATE
        "${KRABBY_ARDUINO_DIR}/src/display/display_frame_model.cpp"
        "${KRABBY_ARDUINO_DIR}/src/telemetry.cpp"
)

target_sources(
    test_display_renderer
    PRIVATE
        "${KRABBY_ARDUINO_DIR}/src/display/display_frame_model.cpp"
        "${KRABBY_ARDUINO_DIR}/src/telemetry.cpp"
)
target_include_directories(test_display_renderer PRIVATE "${KRABBY_OLED_SIM_DIR}")

# Build the simulator's trace emitter with the test toolchain.
add_executable(oled_trace
    "${KRABBY_OLED_SIM_DIR}/oled_trace.cpp"
    "${KRABBY_ARDUINO_DIR}/src/display/display_frame_model.cpp"
    "${KRABBY_ARDUINO_DIR}/src/telemetry.cpp"
)
target_include_directories(oled_trace PRIVATE "${KRABBY_ARDUINO_DIR}" "${KRABBY_OLED_SIM_DIR}")
if(KRABBY_NATIVE_COVERAGE)
    set_target_properties(oled_trace PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()
if(MSVC)
    target_compile_options(oled_trace PRIVATE /W4 /WX)
else()
    target_compile_options(oled_trace PRIVATE -Wall -Wextra -Werror)
endif()
