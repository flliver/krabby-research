#!/usr/bin/env bash
# Run firmware @pytest.mark.isaacsim tests inside krabby-isaacsim (GPU + Isaac Lab).
#
# From repo root (after make build-isaacsim-image):
#   ./images/isaacsim/run_firmware_isaac_pytest.sh
#   ./images/isaacsim/run_firmware_isaac_pytest.sh tests/integration/firmware/ -v
#
# Mounts host firmware/, tests/, hal/, compute/, assets/, and parkour/ over the image so
# local edits are exercised without rebuilding HAL wheels. Uses /isaac-sim/python.sh (not
# AppLauncher can construct SimulationApp — same requirement as run_test_runner.sh.
#
# Pass/fail is determined on the HOST from build/firmware-isaac-pytest/ (outcomes.jsonl +
# junit.xml) via summarize_firmware_isaac_results.py. The container process often exits 0
# even when pytest failed, so its exit code is informational only.
#
# Environment:
#   KRABBY_ISAACSIM_IMAGE  docker tag (default: krabby-isaacsim:latest)
#   KRABBY_ISAAC_DEVICE    passed to AppLauncher (default: cuda:0)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMAGE="${KRABBY_ISAACSIM_IMAGE:-krabby-isaacsim:latest}"
DEVICE="${KRABBY_ISAAC_DEVICE:-cuda:0}"
RESULTS_DIR="${REPO_ROOT}/build/firmware-isaac-pytest"

if [[ $# -eq 0 ]]; then
  set -- tests/integration/firmware -v
fi

mkdir -p "$RESULTS_DIR"
rm -f "$RESULTS_DIR"/{outcomes.jsonl,collected.json,session.json,junit.xml,pytest.rc}

docker run --rm --gpus all \
  -v "$REPO_ROOT/firmware:/workspace/firmware" \
  -v "$REPO_ROOT/hal:/workspace/hal" \
  -v "$REPO_ROOT/compute:/workspace/compute" \
  -v "$REPO_ROOT/tests:/workspace/tests" \
  -v "$REPO_ROOT/assets:/workspace/assets:ro" \
  -v "$REPO_ROOT/parkour:/workspace/parkour" \
  -v "$REPO_ROOT/build/native-firmware-tests:/workspace/build/native-firmware-tests:ro" \
  -v "$RESULTS_DIR:/results:rw" \
  -e RUN_FIRMWARE_ISAAC=1 \
  -e FIRMWARE_ISAAC_RESULTS_DIR=/results \
  -e "KRABBY_ISAAC_DEVICE=$DEVICE" \
  -e PYTHONUNBUFFERED=1 \
  --entrypoint /bin/bash \
  "$IMAGE" \
  -c '
    set -uo pipefail
    source /workspace/testenv/bin/activate
    pip install -q pyserial
    VENV_SITE_PACKAGES=$(/workspace/testenv/bin/python -c "import site; print(site.getsitepackages()[0])")
    export PYTHONPATH="/workspace/isaaclab/source/isaaclab_tasks:/workspace/isaaclab/source:/workspace/isaaclab/source/isaaclab:/workspace/parkour:/workspace:/isaac-sim/python_packages:${VENV_SITE_PACKAGES}"
    /isaac-sim/python.sh -m pytest -o log_cli=false --junitxml=/results/junit.xml "$@"
    pytest_rc=$?
    echo "$pytest_rc" > /results/pytest.rc
    exit "$pytest_rc"
  ' -- "$@"
container_rc=$?

echo ""
echo "Container process exit code: ${container_rc} (informational; verdict below is authoritative)"
echo ""

exec python3 "$SCRIPT_DIR/summarize_firmware_isaac_results.py" "$RESULTS_DIR"
