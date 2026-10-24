#!/usr/bin/env bash
# Start both HAL server and client in one container (detached). No second terminal needed.
# Stop with ./controller/scripts/jetson/helper/stop_gamepad_hal_container.sh
# Logs: docker logs hal-gamepad or docker logs -f hal-gamepad
# Usage: ./controller/scripts/jetson/helper/start_gamepad_hal_and_client_detached.sh

set -e
MCU_PORT="${KRABBY_MCU_PORT:-/dev/ttyACM0}"
if [ ! -e "$MCU_PORT" ]; then
  echo "Error: MCU device not found at $MCU_PORT. Connect the Jetson HAL (MCU) before starting." >&2
  echo "Set KRABBY_MCU_PORT if using a different port (e.g. export KRABBY_MCU_PORT=/dev/ttyUSB0)." >&2
  exit 1
fi
IMAGE="${KRABBY_LOCOMOTION_IMAGE:-krabby-locomotion:latest}"
RUN_OPTS=(-d --name hal-gamepad --rm --privileged --runtime=nvidia
  -v /dev:/dev
  -p 6001:6001 -p 6002:6002
  -e XDG_RUNTIME_DIR=/tmp/runtime-krabby
  --entrypoint /workspace/controller/scripts/jetson/helper/run_server_and_client_in_container.sh)
[ -n "${KRABBY_MCU_PORT:-}" ] && RUN_OPTS+=(-e "KRABBY_MCU_PORT=$KRABBY_MCU_PORT")
docker run "${RUN_OPTS[@]}" "$IMAGE"
