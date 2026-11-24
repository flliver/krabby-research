set -e
cd /scratch/ced-verify/robot-run
g++ -std=c++17 -O1 -Wall -I/src/firmware/arduino/src/actuator -I/src/firmware/arduino -o /tmp/h h.cpp
for f in "$@"; do /tmp/h < $f > ${f%.txt}.out; done
