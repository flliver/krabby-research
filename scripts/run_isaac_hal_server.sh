#!/bin/bash
# Isaac Sim HAL + joystick demo in Docker. Build: make build-isaacsim-image
#
#   ./scripts/run_isaac_hal_server.sh              # Go2 play + krabby-uno-sim (HAL TCP :5555 / :5556 on host)
#   ./scripts/run_isaac_hal_server.sh --teleop-ip 127.0.0.1     # Go2 play + portal WebRTC (in-process policy; no krabby-uno-sim)
#   ./scripts/run_isaac_hal_server.sh --teleop-ip 127.0.0.1 --joystick  # Portal + manual drive (start krabby-uno-sim on host)
#   ./scripts/run_isaac_hal_server.sh --hexapod  # Hex USD + --backend fullsim (host firmware)
#
# Uses Docker --network host (Linux) so HAL and teleop edge can use ws://127.0.0.1:9000/ws/robot when the
# portal runs on the same machine (e.g. another container publishing 9000 on the host). Omitting bridge
# NAT avoids 127.0.0.1 inside the Isaac container pointing at the wrong namespace.
#
# Image wheels (krabby-hal-server-isaac 0.1.2+ on PyPI) are the default after make build-isaacsim-image.
# Set KRABBY_ISAAC_DEV_MOUNTS=1 to bind-mount hal/, compute/, teleop/, firmware/, and parkour/ over the image.
#
# Optional flags (--teleop-ip, --hexapod) may appear anywhere among script arguments.
# Everything else is passed through to the container entrypoint.

set -e

IMAGE="krabby-isaacsim:latest"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Demo pins (edit here if the demo changes)
GO2_TASK="Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0"
HEX_USD="/workspace/assets/crab_hex_ref.usd"
GO2_TELEOP_DEFAULT_CHECKPOINT="/workspace/test_assets/checkpoints/unitree_go2_parkour_teacher.pt"

want_teleop_ip=""
want_hexapod=0
want_joystick=0
passthrough=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --teleop-ip)
      if [ "$#" -lt 2 ]; then
        echo "Error: --teleop-ip requires a host argument" >&2
        exit 1
      fi
      want_teleop_ip="$2"
      shift 2
      ;;
    --hexapod)
      want_hexapod=1
      shift
      ;;
    --joystick)
      want_joystick=1
      shift
      ;;
    *)
      passthrough+=("$1")
      shift
      ;;
  esac
done

has_checkpoint_in_pass=0
i=0
while [ "$i" -lt "${#passthrough[@]}" ]; do
  if [ "${passthrough[$i]}" = "--checkpoint" ]; then
    has_checkpoint_in_pass=1
    break
  fi
  i=$((i + 1))
done

py=()

if [ "$want_hexapod" -eq 1 ]; then
  py+=(--usd "$HEX_USD" --backend fullsim)
  if [ -n "$want_teleop_ip" ]; then
    py+=(--teleop-ip "$want_teleop_ip")
    # Image does not bundle a hex policy; drive over TCP unless user passes --checkpoint …
    if [ "$want_joystick" -eq 1 ] || [ "$has_checkpoint_in_pass" -eq 0 ]; then
      py+=(--joystick)
      if [ "$has_checkpoint_in_pass" -eq 0 ] && [ "$want_joystick" -eq 0 ]; then
        echo "NOTE: hexapod --teleop-ip defaults to --joystick (krabby-uno-sim); pass --checkpoint … for in-process inference." >&2
      fi
    fi
  else
    py+=(--joystick)
  fi
else
  py+=(--task "$GO2_TASK")
  if [ -n "$want_teleop_ip" ]; then
    py+=(--robot go2 --teleop-ip "$want_teleop_ip")
    if [ "$want_joystick" -eq 1 ]; then
      py+=(--joystick)
    elif [ "$has_checkpoint_in_pass" -eq 1 ]; then
      :
    else
      py+=(--checkpoint "$GO2_TELEOP_DEFAULT_CHECKPOINT")
    fi
  else
    py+=(--joystick)
  fi
fi

py+=("${passthrough[@]}")

want_headless=0
for arg in "${passthrough[@]}"; do
  if [ "$arg" = "--headless" ]; then
    want_headless=1
    break
  fi
done

docker_env=()
docker_mounts=()
if [ "$want_headless" -eq 1 ]; then
  # Avoid GLFW probing a host DISPLAY that is not valid inside the container.
  docker_env+=(-e DISPLAY=)
else
  xhost +local:docker 2>/dev/null || true
  docker_mounts+=(-v /tmp/.X11-unix:/tmp/.X11-unix)
  docker_env+=(-e "DISPLAY=${DISPLAY}")
fi
if [ "${KRABBY_ISAAC_DEV_MOUNTS:-0}" = 1 ]; then
  docker_mounts+=(
    -v "${REPO_ROOT}/hal:/workspace/hal"
    -v "${REPO_ROOT}/compute:/workspace/compute"
    -v "${REPO_ROOT}/teleop:/workspace/teleop"
    -v "${REPO_ROOT}/firmware:/workspace/firmware"
    -v "${REPO_ROOT}/parkour:/workspace/parkour"
  )
fi
if [ "$want_hexapod" -eq 1 ]; then
  docker_mounts+=(-v "${REPO_ROOT}/assets:/workspace/assets")
  # fullsim attaches firmware.sim on the Isaac env (Python tree; .so is baked into the image).
  docker_mounts+=(-v "${REPO_ROOT}/firmware:/workspace/firmware")
  HOST_LIB="${REPO_ROOT}/build/native-firmware-tests/krabby_sim_host/libkrabby_host.so"
  if [ -f "$HOST_LIB" ]; then
    docker_mounts+=(
      -v "${REPO_ROOT}/build/native-firmware-tests:/workspace/build/native-firmware-tests:ro"
    )
  fi
fi

exec docker run --rm --gpus all \
  --network host \
  "${docker_env[@]}" \
  "${docker_mounts[@]}" \
  "$IMAGE" "${py[@]}"
