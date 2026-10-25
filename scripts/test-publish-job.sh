#!/usr/bin/env bash
# Run the same build-and-test steps as the publish workflow for a package (or all).
# Usage: from repo root with venv activated:
#   ./scripts/test-publish-job.sh <package-key>
#   ./scripts/test-publish-job.sh all
# Package keys: hal-client, hal-server, compute-parkour, controller, hal-tools, hal-server-isaac, hal-server-jetson, data-collection, firmware
# Does not upload to PyPI.

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! python -m build --version &>/dev/null; then
  echo "Missing 'build'. Install with: pip install build"
  exit 1
fi

run_one_job() {
  local key="$1"
  local path deps build_self_first test_path dep

  case "$key" in
    hal-server-isaac)
      path="hal/server/isaac"
      deps="hal/client hal/server compute/parkour data_collection"
      build_self_first=""
      test_path="tests/unit/hal/server/isaac/"
      ;;
    hal-server-jetson)
      path="hal/server/jetson"
      deps="hal/client hal/server compute/parkour firmware data_collection"
      build_self_first=""
      test_path="tests/unit/hal/"
      ;;
    teleop-edge)
      path="teleop/edge"
      deps=""
      build_self_first=""
      test_path="tests/unit/teleop/"
      ;;
    data-collection)
      path="data_collection"
      deps="hal/client"
      build_self_first=""
      test_path="tests/unit/data_collection/"
      ;;
    firmware)
      path="firmware"
      deps=""
      build_self_first=""
      test_path=""
      ;;
    hal-client)
      path="hal/client"
      deps="hal/server compute/parkour"
      build_self_first="hal/client"
      test_path="tests/unit/hal/"
      ;;
    hal-server)
      path="hal/server"
      deps=""
      build_self_first=""
      test_path="tests/unit/hal/test_server.py tests/unit/hal/test_zmq_push_pull.py"
      ;;
    compute-parkour)
      path="compute/parkour"
      deps="hal/client"
      build_self_first=""
      test_path="tests/unit/test_compute_parkour_policy.py"
      ;;
    controller)
      path="controller"
      deps="hal/client"
      build_self_first=""
      test_path="tests/unit/controller/"
      ;;
    hal-tools)
      path="hal/tools"
      deps="hal/client hal/server"
      build_self_first=""
      test_path=""
      ;;
    *)
      echo "Unknown package key: $key"
      echo "Valid keys: hal-client, hal-server, compute-parkour, controller, hal-tools, hal-server-isaac, hal-server-jetson, teleop-edge, data-collection, firmware"
      exit 1
      ;;
  esac

  echo "===== test-publish-job: $key ====="

  if [[ -n "$build_self_first" ]]; then
    echo "Building and installing $build_self_first first..."
    cd "$ROOT/$build_self_first" && python -m build --wheel --no-isolation && pip install dist/*.whl && cd "$ROOT"
  fi

  for dep in $deps; do
    [[ -z "$dep" ]] && continue
    echo "Building and installing $dep..."
    cd "$ROOT/$dep" && python -m build --wheel --no-isolation && pip install dist/*.whl && cd "$ROOT"
  done

  echo "Building package wheel: $path"
  cd "$ROOT/$path" && python -m build --wheel --no-isolation && cd "$ROOT"

  echo "Installing package and test deps..."
  cd "$ROOT/$path" && pip install dist/*.whl && pip install pytest pytest-cov pytest-timeout pytest-asyncio keyboard pyserial
  if [[ "$key" == hal-server-isaac ]] || [[ "$key" == hal-server-jetson ]] || [[ "$key" == compute-parkour ]] || [[ "$key" == hal-client ]]; then
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install scipy
  fi
  cd "$ROOT"

  if [[ -n "$test_path" ]]; then
    echo "Running tests: pytest $test_path -v --timeout=300"
    pytest $test_path -v --timeout=300
  else
    echo "No tests for $key (test_path empty)."
  fi

  echo "===== $key done ====="
}

if [[ "$1" == "all" ]]; then
  for key in firmware hal-server-isaac hal-server-jetson hal-client hal-server compute-parkour data-collection controller hal-tools; do
    run_one_job "$key"
  done
elif [[ -n "$1" ]]; then
  run_one_job "$1"
else
  echo "Usage: $0 <package-key|all>"
  echo "  package-key: hal-client, hal-server, compute-parkour, controller, hal-tools, hal-server-isaac, hal-server-jetson, firmware"
  echo "  all: run all eight jobs in sequence"
  exit 1
fi
