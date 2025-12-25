#!/bin/bash
# Wrapper script to run test_runner.py
# Uses /isaac-sim/python.sh to get omni modules, but test_runner.py will check for existing instance

set -e

# Activate the virtual environment
source /workspace/testenv/bin/activate

# Set up PYTHONPATH
VENV_SITE_PACKAGES=$(/workspace/testenv/bin/python -c "import site; print(site.getsitepackages()[0])")
export PYTHONPATH="/workspace/isaaclab/source/isaaclab_tasks:/workspace/isaaclab/source:/workspace/isaaclab/source/isaaclab:/workspace/parkour:/workspace:/isaac-sim/python_packages:${VENV_SITE_PACKAGES}:${PYTHONPATH}"

# Filter out Isaac Sim arguments before passing to test runner
filtered_args=()
isaac_sim_flags=("--portable" "--no-window" "--allow-root" "--ext-folder")
skip_next=false

for arg in "$@"; do
    if [ "$skip_next" = true ]; then
        skip_next=false
        continue
    fi
    
    # Check if it's an Isaac Sim flag
    is_isaac_flag=false
    for flag in "${isaac_sim_flags[@]}"; do
        if [ "$arg" = "$flag" ]; then
            is_isaac_flag=true
            if [ "$arg" = "--ext-folder" ]; then
                skip_next=true
            fi
            break
        fi
    done
    
    # Skip Isaac Sim arguments (those starting with --/)
    if [[ "$arg" == --/* ]] || [ "$is_isaac_flag" = true ]; then
        continue
    fi
    
    filtered_args+=("$arg")
done

# Use /isaac-sim/python.sh (required for omni modules)
# Both hal/server/isaac/main.py and test_runner.py need /isaac-sim/python.sh
# The CUDA context issue is a known limitation - AppLauncher should handle it
exec /isaac-sim/python.sh /workspace/test_runner.py "${filtered_args[@]}"

