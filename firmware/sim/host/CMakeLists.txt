# Host Arduino shim (pin state + sim clock). Linked by native tests and future host firmware targets.

set(KRABBY_ARDUINO_DIR "${CMAKE_CURRENT_LIST_DIR}/../../arduino")

add_library(krabby_sim_host STATIC
    pin_state.cpp
    sim_clock.cpp
    eeprom_stub.cpp
    hall_hw_stub.cpp
    serial_stub.cpp
)
target_include_directories(krabby_sim_host PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(krabby_sim_host PRIVATE "${KRABBY_ARDUINO_DIR}")
if(MSVC)
    target_compile_options(krabby_sim_host PRIVATE /W4 /WX)
else()
    target_compile_options(krabby_sim_host PRIVATE -Wall -Wextra -Werror)
endif()
