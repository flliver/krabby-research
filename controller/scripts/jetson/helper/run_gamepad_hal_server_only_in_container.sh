#!/usr/bin/env bash
# Run the gamepad-only HAL server (server only) inside the locomotion container.
# Requires: krabby-locomotion image pulled (sudo krabby install).
# Usage: ./controller/scripts/jetson/helper/run_gamepad_hal_server_only_in_container.sh
# Then run the client on the host: krabby-uno  (pip install krabby-controller)
# Note: plain `krabby run` already starts this server + client together; this helper
# is for the two-process debug flow where you want the server on its own.

set -e
IMAGE="${KRABBY_LOCOMOTION_IMAGE:-krabby-locomotion:latest}"
docker run --rm --privileged --runtime=nvidia \
  -v /dev:/dev \
  -p 6001:6001 -p 6002:6002 \
  --entrypoint krabby-hal-server-jetson \
  "$IMAGE" \
  --control-source gamepad \
  --observation-bind tcp://*:6001 --command-bind tcp://*:6002 \
  "$@"
