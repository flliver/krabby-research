set -e
cd /scratch/ced-verify/robot-run
g++ -std=c++17 -O1 -Wall -I/src/firmware/arduino/src/actuator -o /tmp/sim sim.cpp
for s in H A B B2 C K; do echo "##### $s"; /tmp/sim $s 1; done
