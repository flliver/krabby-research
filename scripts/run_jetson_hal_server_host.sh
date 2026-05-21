#!/usr/bin/env bash
# Launch Jetson HAL server from host into the Jetson locomotion image.
# All runtime values are hardcoded in this script by design.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

print_usage() {
  cat <<'EOF'
Usage: scripts/run_jetson_hal_server_host.sh

Launches Jetson HAL server in Docker using hardcoded settings in this script.

Data collector notes:
  - This launcher currently enables recording via:
      --data-collector-output-dir /workspace/bags
  - HAL entrypoint also supports:
      --data-collector-config <path-to-collector.yaml>
    (edit this script if you want to pass that flag through).

Control source notes:
  - This launcher passes:
      --control-source portal
  - Set CONTROL_SOURCE below to "inference" to use policy commands instead.

To customize runtime values (image, checkpoint, collector path, teleop URL),
edit the "Hardcoded launch settings" section in this file.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_usage
  exit 0
fi

if [ ! -f "$PROJECT_ROOT/Makefile" ]; then
  echo "Error: Could not find repo Makefile at $PROJECT_ROOT/Makefile" >&2
  exit 1
fi

ensure_dir_exists() {
  local dir="$1"
  if mkdir -p "$dir" 2>/dev/null; then
    return 0
  fi
  echo "Directory not creatable as current user, creating with sudo: $dir"
  sudo mkdir -p "$dir"
  if [ ! -d "$dir" ]; then
    echo "Error: directory does not exist after sudo create: $dir" >&2
    exit 1
  fi
}

# -------------------------
# Hardcoded launch settings
# -------------------------
IMAGE="krabby-locomotion:latest"
# Locomotion image does not bundle Isaac Sim test_assets; mount a host dir (see docs/JETSON_DEPLOYMENT.md).
CHECKPOINT_FILENAME="unitree_go2_parkour_teacher.pt"
CHECKPOINT_PATH="/workspace/checkpoints/$CHECKPOINT_FILENAME"
# Prefer repo ./checkpoints/; also accept parkour/assets/weights/ (see parkour/scripts README).
CHECKPOINT_HOST_DIR=""
for _dir in "$PROJECT_ROOT/checkpoints" "$PROJECT_ROOT/parkour/assets/weights"; do
  if [ -f "$_dir/$CHECKPOINT_FILENAME" ]; then
    CHECKPOINT_HOST_DIR="$(realpath "$_dir")"
    break
  fi
done
TASK_NAME="Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0"
ROBOT_TYPE="go2"
CONTROL_SOURCE="portal"   # "portal" (default) or "inference"
TELEOP_WS_URL="ws://10.0.0.130:9000/ws/robot"
COLLECTOR_HOST_DIR="/tmp/krabby_bags"
CONTAINER_COLLECTOR_DIR="/workspace/bags"
ZED_CACHE_HOST_DIR="${HOME}/zed-resources"
ZED_RESOURCES_CONTAINER_DIR="/usr/local/zed/resources"
ZED_SETTINGS_CONTAINER_DIR="/usr/local/zed/settings"
CREATE_COLLECTOR_DIR_IF_MISSING=1

if [ -z "$CHECKPOINT_HOST_DIR" ]; then
  echo "Error: checkpoint not found: $CHECKPOINT_FILENAME" >&2
  echo "Place it in one of:" >&2
  echo "  $PROJECT_ROOT/checkpoints/$CHECKPOINT_FILENAME" >&2
  echo "  $PROJECT_ROOT/parkour/assets/weights/$CHECKPOINT_FILENAME" >&2
  echo "Weights are not in git or the Docker image; copy from training output or another machine (scp, USB, etc.)." >&2
  echo "See docs/JETSON_DEPLOYMENT.md — checkpoints must exist on the host before running." >&2
  exit 1
fi

COLLECTOR_HOST_DIR="$(realpath -m "$COLLECTOR_HOST_DIR")"
if [ ! -d "$COLLECTOR_HOST_DIR" ]; then
  if [ "$CREATE_COLLECTOR_DIR_IF_MISSING" -eq 1 ]; then
    mkdir -p "$COLLECTOR_HOST_DIR"
  else
    echo "Error: collector directory does not exist: $COLLECTOR_HOST_DIR" >&2
    exit 1
  fi
fi

ZED_CACHE_HOST_DIR="$(realpath -m "$ZED_CACHE_HOST_DIR")"
ZED_RESOURCES_SUBDIR="$ZED_CACHE_HOST_DIR/resources"
ZED_SETTINGS_SUBDIR="$ZED_CACHE_HOST_DIR/settings"
ensure_dir_exists "$ZED_CACHE_HOST_DIR"
ensure_dir_exists "$ZED_RESOURCES_SUBDIR"
ensure_dir_exists "$ZED_SETTINGS_SUBDIR"

echo "Using data collector host folder: $COLLECTOR_HOST_DIR"
echo "Using image: $IMAGE"
echo "Using checkpoint (host): $CHECKPOINT_HOST_DIR/$CHECKPOINT_FILENAME"
echo "Using checkpoint (container): $CHECKPOINT_PATH"
echo "Using task reference: $TASK_NAME"
echo "Using robot definition: $ROBOT_TYPE"
echo "Using control source: $CONTROL_SOURCE"
echo "Using teleop signaling URL: $TELEOP_WS_URL"
echo "Using side MaixSense endpoint: from sensor catalog"
echo "Using ZED resources mount: $ZED_RESOURCES_SUBDIR:$ZED_RESOURCES_CONTAINER_DIR"
echo "Using ZED settings mount: $ZED_SETTINGS_SUBDIR:$ZED_SETTINGS_CONTAINER_DIR"

PY_BOOTSTRAP="$(cat <<'PYEOF'
import sys

import teleop.edge.robot_settings as teleop_settings
from hal.server.jetson.sensor_backend_jetson import JETSON_SENSOR_CATALOG

task_name = "Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0"
teleop_url = "ws://10.0.0.130:9000/ws/robot"

if not teleop_url:
    raise SystemExit("KRABBY_TELEOP_SIGNALING_WS_URL is empty; refusing to launch teleop")

front = next((e for e in JETSON_SENSOR_CATALOG if e.is_primary and e.id == "front_rgbd"), None)
if front is None:
    raise SystemExit("JETSON_SENSOR_CATALOG is missing required primary row id='front_rgbd'")
if front.camera_driver != "zed":
    raise SystemExit(
        f"Expected front_rgbd camera_driver='zed', got {front.camera_driver!r}. "
        "Update catalog to the requested front ZED configuration."
    )

side = next(
    (
        e
        for e in JETSON_SENSOR_CATALOG
        if (not e.is_primary)
        and e.hal_open_rgbd
        and e.camera_driver == "maixsense_a075v"
    ),
    None,
)
if side is None:
    raise SystemExit(
        "Expected a non-primary side MaixSense catalog row "
        "(camera_driver='maixsense_a075v', hal_open_rgbd=True), but none was found."
)
side_host = (side.maixsense_host or "").strip() or "(unset)"
if side.maixsense_port is not None:
    side_port = side.maixsense_port
elif side.maixsense_port_env:
    side_port = (__import__("os").environ.get(side.maixsense_port_env, "")).strip() or "(unset)"
else:
    side_port = "(unset)"

# Force teleop config from launcher.
teleop_settings.TELEOP_EDGE_MODE = "agent"
teleop_settings.SERVER_SIGNALING_WS_URL = teleop_url

print(
    f"[launcher] Camera config OK: front={front.id}({front.camera_driver}), "
    f"side={side.id}({side.camera_driver})"
)
print(f"[launcher] Teleop signaling URL: {teleop_url}")
print(f"[launcher] Side MaixSense endpoint: {side_host}:{side_port}")
if task_name:
    print(f"[launcher] Task reference: {task_name} (Jetson main does not consume --task)")

sys.argv = [sys.argv[0], "--teleop", *sys.argv[1:]]
from hal.server.jetson.main import main

main()
PYEOF
)"

exec sudo docker run --rm --runtime=nvidia --network host \
  -v "$CHECKPOINT_HOST_DIR:/workspace/checkpoints" \
  -v "$COLLECTOR_HOST_DIR:$CONTAINER_COLLECTOR_DIR" \
  -v "$ZED_RESOURCES_SUBDIR:$ZED_RESOURCES_CONTAINER_DIR" \
  -v "$ZED_SETTINGS_SUBDIR:$ZED_SETTINGS_CONTAINER_DIR" \
  -v /dev:/dev \
  --privileged \
  --entrypoint python3 \
  "$IMAGE" \
  -c "$PY_BOOTSTRAP" \
  --checkpoint "$CHECKPOINT_PATH" \
  --control-source "$CONTROL_SOURCE" \
  --robot "$ROBOT_TYPE" \
  --data-collector-output-dir "$CONTAINER_COLLECTOR_DIR"
