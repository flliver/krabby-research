# Real driver compiled against the scripted native-test Wire and Arduino shims.
include("${KRABBY_ARDUINO_DIR}/../sim/cmake/sparkfun_ina228.cmake")
add_library(sparkfun_host_runtime STATIC "${KRABBY_ARDUINO_DIR}/../sim/vendor/SPI.cpp")
target_include_directories(sparkfun_host_runtime PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/fakes/vendor_runtime"
    "${KRABBY_ARDUINO_DIR}/../sim/vendor")
target_link_libraries(sparkfun_host_runtime PUBLIC wire_native_support)
add_sparkfun_ina228_driver(sparkfun_ina2xx_native sparkfun_host_runtime)
