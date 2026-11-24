#!/bin/sh
# Run inside the container, cwd = /scratch/ced-verify/trace-replay
set -e
T=/src/tests/native/firmware/arduino/src/actuator
A=/src/firmware/arduino
build() { # $1 = name, $2 = include dir override ('' for pristine)
  inc=""; [ -n "$2" ] && inc="-I$2"
  g++ -std=c++11 -O1 -w -Ishim $inc -I$A -I$T -o out/test_$1 $T/test_calibration_end_detector.cpp 2> out/test_$1.build.log || { echo "$1: BUILD FAILED"; tail -5 out/test_$1.build.log; return; }
  out/test_$1 > out/test_$1.log 2>&1 || true
  echo "== $1: $(tail -1 out/test_$1.log)"; grep '^FAIL' out/test_$1.log | cut -c1-260 || true
}
build pristine ""
mk() { # $1 name, $2 sed expr
  rm -rf mut/$1; mkdir -p mut/$1/src/actuator
  cp $A/src/actuator/*.h mut/$1/src/actuator/
  sed -i 's/\r$//' mut/$1/src/actuator/calibration_end_detector.h
  cp mut/$1/src/actuator/calibration_end_detector.h mut/$1/orig.h
  sed -i "$2" mut/$1/src/actuator/calibration_end_detector.h
  if cmp -s mut/$1/orig.h mut/$1/src/actuator/calibration_end_detector.h; then echo "$1: MUTATION DID NOT APPLY"; return; fi
  diff mut/$1/orig.h mut/$1/src/actuator/calibration_end_detector.h | head -6
  build $1 mut/$1
}
mk F1_no_iref_guard 's|static_cast<uint32_t>(CAL_END_SWITCH_SLOW_IREF_PCT) \* iRef_)))|4000000000UL)))|'
mk F2_no_fallback 's|if (!leaving_ \&\& normalSamples_ < CAL_END_SHORT_NORMAL_SAMPLES \&\&|if (false \&\& normalSamples_ < CAL_END_SHORT_NORMAL_SAMPLES \&\&|'
mk F2_no_retract_guard 's/(!retract_ || med4 <= iRef_))/(true))/'
mk F2_no_normal_count 's|if (!leaving_ \&\& normalSamples_ < CAL_END_SHORT_NORMAL_SAMPLES \&\&|if (!leaving_ \&\&|'
mk F2_no_run_bound 's|ran < CAL_END_SHORT_RUN_SAMPLES \&\& |true \&\& |'
mk F2_fallback_for_uncurl 's|if (!leaving_ \&\& normalSamples_ < CAL_END_SHORT_NORMAL_SAMPLES \&\&|if (normalSamples_ < CAL_END_SHORT_NORMAL_SAMPLES \&\&|'
mk F4_short_window 's|e >= CAL_HALL_CURL_EVIDENCE_EDGES ? kStallSamples : kStillWindowSamples|true ? kStillWindowSamples : kStallSamples|'
mk F1_below_min_removed 's|(worst < stopMin_ \&\&|(true \&\&|'
