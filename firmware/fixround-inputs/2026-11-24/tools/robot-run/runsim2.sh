set -e
cd /scratch/ced-verify/robot-run
g++ -std=c++17 -O1 -Wall -I/src/firmware/arduino/src/actuator -o /tmp/sim sim.cpp
echo "##### A2 SEEK"; /tmp/sim A2 1
echo "##### AF SEEK+SWEEP (belly at 941)"; /tmp/sim AF 3
echo "##### AX SEEK+SWEEP (no belly, extend switch 1069)"; /tmp/sim AX 3
