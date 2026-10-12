# Host Arduino shim (pin state + sim clock). Linked by native tests and future host firmware targets.

set(KRABBY_ARDUINO_DIR "${CMAKE_CURRENT_LIST_DIR}/../../arduino")

add_library(krabby_sim_host STATIC
    pin_state.cpp
    sim_clock.cpp
    eeprom_stub.cpp
    hall_sim.cpp
    hall_hw_stub.cpp
    sim_serial.cpp
)
target_include_directories(krabby_sim_host PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(krabby_sim_host PRIVATE "${KRABBY_ARDUINO_DIR}")
# krabby_host links these statics into a .so, so they need PIC as well.
set_target_properties(krabby_sim_host PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
    target_compile_options(krabby_sim_host PRIVATE /W4 /WX)
else()
    target_compile_options(krabby_sim_host PRIVATE -Wall -Wextra -Werror)
endif()

add_library(krabby_firmware_actuator_host STATIC actuator_manager_host.cpp)
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

add_library(krabby_host SHARED host_api.cpp)
target_link_libraries(krabby_host PRIVATE krabby_firmware_actuator_host)
if(MSVC)
    target_compile_options(krabby_host PRIVATE /W4 /WX)
else()
    target_compile_options(krabby_host PRIVATE -Wall -Wextra -Werror -Wno-sign-compare)
endif()
set_target_properties(krabby_host PROPERTIES POSITION_INDEPENDENT_CODE ON)
