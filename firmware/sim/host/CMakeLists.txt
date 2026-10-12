# Host Arduino shim (pin state + sim clock). Linked by native tests and future host firmware targets.

add_library(krabby_sim_host STATIC
    pin_state.cpp
    sim_clock.cpp
)
target_include_directories(krabby_sim_host PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
if(MSVC)
    target_compile_options(krabby_sim_host PRIVATE /W4 /WX)
else()
    target_compile_options(krabby_sim_host PRIVATE -Wall -Wextra -Werror)
endif()
