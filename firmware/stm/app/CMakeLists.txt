add_executable(krabby_leg_app main.c)
target_include_directories(krabby_leg_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../core
)
target_compile_definitions(krabby_leg_app PRIVATE
    KRABBY_LEG_FW_VERSION="${KRABBY_LEG_FW_VERSION}"
    KRABBY_LEG_FW_GIT_SHA="${KRABBY_LEG_FW_GIT_SHA}"
)
target_link_libraries(krabby_leg_app PRIVATE krabby_leg_board_hw krabby_leg_core krabby_leg_version)
