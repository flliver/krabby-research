#!/usr/bin/env bash
# Run the teleop portal (HTTP viewer + WebSocket signaling) inside the x86 test Docker image,
# with TCP published to the host so you can:
#   - Open the viewer on the x86 machine: http://localhost:<host-port>/
#   - Point Jetson teleop edge at ws://<x86-host-LAN-ip>:<host-port>/ws/robot
#
# The portal serves /, /api/teleop-config, /ws/browser, and /ws/robot on one listener (default 9000).
#
# Stops/removes any existing container named KRABBY_TELEOP_CONTAINER_NAME (default: krabby-teleop-portal) before starting.
#
# Prerequisites:
#   make build-test-image    # produces krabby-testing-x86:latest
#
# Environment (optional):
#   KRABBY_TESTING_X86_IMAGE   Docker image (default: krabby-testing-x86:latest)
#   TELEOP_HOST_PORT           Host TCP port to publish (default: 9000)
#   TELEOP_CONTAINER_PORT      Port inside the container (default: 9000; must match portal --port)
#   KRABBY_TELEOP_DOCKER_RUN_EXTRA   Extra args to `docker run`, e.g. `--gpus all` (not required for the portal)
#   KRABBY_TELEOP_CONTAINER_NAME     Container name (default: krabby-teleop-portal)
#
# Extra CLI arguments are forwarded to krabby-teleop-portal (e.g. --help).
#
# WebRTC media is browser ↔ robot after signaling; open host firewall for UDP if ICE fails.

set -euo pipefail

IMAGE="${KRABBY_TESTING_X86_IMAGE:-krabby-testing-x86:latest}"
HOST_PORT="${TELEOP_HOST_PORT:-9000}"
CONTAINER_PORT="${TELEOP_CONTAINER_PORT:-9000}"
CONTAINER_NAME="${KRABBY_TELEOP_CONTAINER_NAME:-krabby-teleop-portal}"
DOCKER_RUN_EXTRA="${KRABBY_TELEOP_DOCKER_RUN_EXTRA:-}"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

# shellcheck disable=SC2086
exec docker run --rm -it \
  --name "${CONTAINER_NAME}" \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  ${DOCKER_RUN_EXTRA} \
  "$IMAGE" \
  krabby-teleop-portal --host 0.0.0.0 --port "${CONTAINER_PORT}" \
  "$@"
