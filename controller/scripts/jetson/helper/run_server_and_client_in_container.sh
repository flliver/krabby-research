#!/usr/bin/env bash
# Run HAL server and client in the background inside the container.
# Used as entrypoint when starting the container with start_gamepad_hal_and_client_detached.sh.
# On SIGTERM/SIGINT (e.g. docker stop), both processes are killed and the container exits.

set -e

MCU_PORT="${KRABBY_MCU_PORT:-/dev/ttyACM0}"
if [ ! -e "$MCU_PORT" ]; then
  echo "Error: MCU device not found at $MCU_PORT. Connect the Jetson HAL (MCU) before starting the container." >&2
  echo "Set KRABBY_MCU_PORT if using a different port (e.g. docker run -e KRABBY_MCU_PORT=/dev/ttyUSB0 ...)." >&2
  exit 1
fi

krabby-hal-server-jetson \
  --control-source gamepad \
  --observation-bind tcp://*:6001 --command-bind tcp://*:6002 &
SERVER_PID=$!

sleep 2

python3 -m controller.scripts.jetson.run_gamepad_to_krabby_client \
  --observation_endpoint tcp://localhost:6001 --command_endpoint tcp://localhost:6002 &
CLIENT_PID=$!

trap 'kill $SERVER_PID $CLIENT_PID 2>/dev/null; wait 2>/dev/null; exit' SIGTERM SIGINT

wait
