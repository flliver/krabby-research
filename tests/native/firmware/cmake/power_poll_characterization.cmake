# Temporary migration seam: compile verbatim regions of the sketch, without
# modifying firmware or maintaining a copied polling implementation.
set(KRABBY_POWER_POLL_SKETCH "${KRABBY_ARDUINO_DIR}/arduino.ino" CACHE FILEPATH
    "Sketch source used by the power polling characterization tests")
get_filename_component(power_poll_sketch "${KRABBY_POWER_POLL_SKETCH}" ABSOLUTE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${power_poll_sketch}")
file(READ "${power_poll_sketch}" power_poll_source)
set(power_poll_generated "${CMAKE_CURRENT_BINARY_DIR}/power_poll_generated")
file(MAKE_DIRECTORY "${power_poll_generated}")

function(extract_power_poll_region output start_marker end_marker)
    string(FIND "${power_poll_source}" "${start_marker}" start)
    string(FIND "${power_poll_source}" "${end_marker}" end)
    if(start LESS 0 OR end LESS 0 OR end LESS_EQUAL start)
        message(FATAL_ERROR "Power characterization seam moved: ${start_marker}")
    endif()
    foreach(marker IN ITEMS "${start_marker}" "${end_marker}")
        string(FIND "${power_poll_source}" "${marker}" position)
        string(LENGTH "${marker}" marker_length)
        math(EXPR after_marker "${position} + ${marker_length}")
        string(SUBSTRING "${power_poll_source}" ${after_marker} -1 remaining)
        string(FIND "${remaining}" "${marker}" duplicate)
        if(NOT duplicate EQUAL -1)
            message(FATAL_ERROR "Ambiguous power characterization seam: ${marker}")
        endif()
    endforeach()
    math(EXPR length "${end} - ${start}")
    string(SUBSTRING "${power_poll_source}" ${start} ${length} region)
    string(SUBSTRING "${power_poll_source}" 0 ${start} prefix)
    string(REGEX MATCHALL "\n" newlines "${prefix}")
    list(LENGTH newlines line)
    math(EXPR line "${line} + 1")
    file(WRITE "${power_poll_generated}/${output}"
        "// Generated verbatim from arduino.ino; do not edit.\n#line ${line} \"${power_poll_sketch}\"\n${region}\n")
endfunction()

extract_power_poll_region(state.inc
    "Ina228Adapter packPowerMonitor("
    "// --- I2C sensor cluster (Milestone 16)")
extract_power_poll_region(calibration.inc
    "PowerCalibration powerCalibration;"
    "static void logImuInitFailure(")
extract_power_poll_region(poll.inc
    "static void readPowerMeasurements()"
    "static void imuSetup()")
extract_power_poll_region(telemetry.inc
    "            const bool isDiverged ="
    "\n        }\n        mainSerial->println();")
extract_power_poll_region(display.inc
    "        DisplayFrame displayFrame = buildDisplayFrame("
    "        const bool isActuatorDisconnected =")

target_include_directories(test_power_poll PRIVATE
    "${power_poll_generated}"
    "${CMAKE_CURRENT_SOURCE_DIR}/fakes/power_poll"
    "${KRABBY_OLED_SIM_DIR}")
target_compile_definitions(test_power_poll PRIVATE
    KRABBY_POWER_POLL_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/power_poll")
target_sources(test_power_poll PRIVATE
    "${KRABBY_ARDUINO_DIR}/src/display/display_frame_model.cpp"
    "${KRABBY_ARDUINO_DIR}/src/telemetry.cpp")
