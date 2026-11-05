# One set of firmware-matching vendor pins, compiled separately for each Wire ABI.
include_guard(GLOBAL)
include(FetchContent)

FetchContent_Declare(
    sparkfun_toolkit
    URL https://github.com/sparkfun/SparkFun_Toolkit/archive/refs/tags/v1.2.0.tar.gz
    URL_HASH SHA256=44e02fceaa0db83765b021b964de701e6d0a057fff62ff3a1eea80d98d4b034d
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
)
FetchContent_Declare(
    sparkfun_ina2xx
    URL https://github.com/sparkfun/SparkFun_INA2XX_Arduino_Library/archive/refs/tags/v1.0.0.tar.gz
    URL_HASH SHA256=51ea7aa21b219707f9f43504b3e815a797b848d2a1e2077babb9e56cf4b50265
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
)

function(add_sparkfun_ina228_driver name runtime)
    FetchContent_MakeAvailable(sparkfun_toolkit sparkfun_ina2xx)
    add_library(${name}_toolkit STATIC
        "${sparkfun_toolkit_SOURCE_DIR}/src/sfTk/sfToolkit.cpp"
        "${sparkfun_toolkit_SOURCE_DIR}/src/sfTkArdI2C.cpp"
        "${sparkfun_toolkit_SOURCE_DIR}/src/sfTkArdSPI.cpp"
        "${sparkfun_toolkit_SOURCE_DIR}/src/sfTkArduino.cpp")
    target_include_directories(${name}_toolkit SYSTEM PUBLIC "${sparkfun_toolkit_SOURCE_DIR}/src")
    target_link_libraries(${name}_toolkit PUBLIC ${runtime})
    add_library(${name} STATIC
        "${sparkfun_ina2xx_SOURCE_DIR}/src/sfTk/sfDevINA2XX.cpp"
        "${sparkfun_ina2xx_SOURCE_DIR}/src/sfTk/sfDevINA228.cpp"
        "${sparkfun_ina2xx_SOURCE_DIR}/src/sfTk/sfDevINA237.cpp")
    target_include_directories(${name} SYSTEM PUBLIC "${sparkfun_ina2xx_SOURCE_DIR}/src")
    target_link_libraries(${name} PUBLIC ${name}_toolkit)
    foreach(target IN ITEMS ${name} ${name}_toolkit)
        set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
        if(NOT MSVC)
            target_compile_options(${target} PRIVATE -w)
        endif()
    endforeach()
endfunction()
