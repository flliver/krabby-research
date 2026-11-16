cmake_minimum_required(VERSION 3.22)

#
# This file is generated only once,
# and is not re-generated if converter is called multiple times.
#
# User is free to modify the file as much as necessary
#

# Setup compiler settings
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)
option(LOAD_ME_FIRST "Build motor-disabled first-install LED application" OFF)
set(APP_BLINK_MS 250 CACHE STRING "Status LED toggle interval for update verification")
set(APP_VARIANT 0 CACHE STRING "LED demo: 0=original, 1=M1 left, 2=M2 left, 3=M3 left")
if(NOT APP_VARIANT MATCHES "^[0-3]$")
    message(FATAL_ERROR "APP_VARIANT must be 0, 1, 2, or 3")
endif()


# Define the build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

# Set the project name
set(CMAKE_PROJECT_NAME krabby_v03_bringup)

# Enable compile command to ease indexing with e.g. clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# Core project settings
project(${CMAKE_PROJECT_NAME})
message("Build type: " ${CMAKE_BUILD_TYPE})

# Enable CMake support for ASM and C languages
enable_language(C ASM)

# Create an executable object type
add_executable(${CMAKE_PROJECT_NAME})

# Add STM32CubeMX generated sources
add_subdirectory(cmake/stm32cubemx)

# Link directories setup
target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined library search paths
)

# Add sources to executable
target_sources(${CMAKE_PROJECT_NAME} PRIVATE
    $<IF:$<BOOL:${LOAD_ME_FIRST}>,App/Src/load_me_first.c,App/Src/app_service.c>
    App/Src/board.c
    App/Src/can_bridge.c
    # Add user sources here
)

# Add include paths
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    App/Inc
    # Add user defined include paths
)

# Add project symbols (macros)
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
    APP_BLINK_MS=$<IF:$<BOOL:${LOAD_ME_FIRST}>,100,${APP_BLINK_MS}>
    APP_VARIANT=${APP_VARIANT}
    $<$<BOOL:${LOAD_ME_FIRST}>:LOAD_ME_FIRST=1>
    # Add user defined symbols
)

# Remove wrong libob.a library dependency when using cpp files
list(REMOVE_ITEM CMAKE_C_IMPLICIT_LINK_LIBRARIES ob)

# Add linked libraries
target_link_libraries(${CMAKE_PROJECT_NAME}
    stm32cubemx

    # Add user defined libraries
)
add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} -O srec $<TARGET_FILE:${CMAKE_PROJECT_NAME}>
        ${CMAKE_PROJECT_NAME}.srec
    # Convert addressed records, not ELF sections: an empty loadable TLS section
    # can otherwise make objcopy pad the binary from flash up to a RAM address.
    COMMAND ${CMAKE_OBJCOPY} -I srec --gap-fill 0xff -O binary
        ${CMAKE_PROJECT_NAME}.srec
        ${CMAKE_PROJECT_NAME}.bin)
