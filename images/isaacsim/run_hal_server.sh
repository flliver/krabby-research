#!/bin/bash
# Wrapper script to run hal.server.isaac.main
# Uses /isaac-sim/python.sh to get omni modules, but main.py will use AppLauncher

set -e

# Activate the virtual environment
source /workspace/testenv/bin/activate

# Set up PYTHONPATH
VENV_SITE_PACKAGES=$(/workspace/testenv/bin/python -c "import site; print(site.getsitepackages()[0])")
export PYTHONPATH="/workspace/isaaclab/source/isaaclab_tasks:/workspace/isaaclab/source:/workspace/isaaclab/source/isaaclab:/workspace/parkour:/workspace:/isaac-sim/python_packages:${VENV_SITE_PACKAGES}:${PYTHONPATH}"

# Filter out Isaac Sim arguments before passing to main
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

# Crab hex launches use host firmware in the loop unless the caller sets --backend.
_hal_has_backend=0
_hal_hex_launch=0
_i=0
while [ "$_i" -lt "${#filtered_args[@]}" ]; do
    _a="${filtered_args[$_i]}"
    case "$_a" in
        --backend|--backend=*)
            _hal_has_backend=1
            ;;
        --usd)
            _hal_hex_launch=1
            ;;
        --robot=hex)
            _hal_hex_launch=1
            ;;
        --robot)
            _next="${filtered_args[$((_i + 1))]:-}"
            if [ "$_next" = "hex" ]; then
                _hal_hex_launch=1
            fi
            ;;
        --task|--task=*)
            if [ "$_a" = "--task" ]; then
                _task="${filtered_args[$((_i + 1))]:-}"
            else
                _task="${_a#--task=}"
            fi
            case "$_task" in
                *CrabHex*|*crab*hex*|*Crab-Hex*)
                    _hal_hex_launch=1
                    ;;
            esac
            ;;
    esac
    _i=$((_i + 1))
done

if [ "$_hal_has_backend" -eq 0 ] && [ "$_hal_hex_launch" -eq 1 ]; then
    filtered_args=(--backend fullsim "${filtered_args[@]}")
fi

# Use /isaac-sim/python.sh (required for omni modules)
# AppLauncher will handle Isaac Sim initialization
exec /isaac-sim/python.sh -m hal.server.isaac.main "${filtered_args[@]}"

