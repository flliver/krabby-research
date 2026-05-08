#!/bin/bash
# Live status for an in-flight MAtCha run on bbeeprz/tbeeprz.
#
# Usage:
#   run_status.sh [--watch] [--host HOST] [--log LOG] [--out OUT]
#
# Default host: tbeeprz
# Default log:  /tmp/run-curated-12-strong.log
# Default out:  /home/jeremy/outposts/krabby/data/011-scene-reconstruction/scenes/004-sky-house-curated-12-strong
#
# Snapshot mode (default): print a one-shot status, exit.
# Watch mode (--watch):    re-print every 5 seconds until Ctrl+C.

set -uo pipefail

HOST=tbeeprz
LOG=/tmp/run-curated-12-strong.log
OUT=/home/jeremy/outposts/krabby/data/011-scene-reconstruction/scenes/004-sky-house-curated-12-strong
WATCH=0
INTERVAL=5

while [[ $# -gt 0 ]]; do
    case "$1" in
        --watch) WATCH=1; shift ;;
        --host)  HOST="$2"; shift 2 ;;
        --log)   LOG="$2"; shift 2 ;;
        --out)   OUT="$2"; shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        -h|--help)
            head -15 "$0" | tail -14 | sed 's/^# //; s/^#$//'
            exit 0
            ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

snap() {
    local ts
    ts=$(date '+%Y-%m-%dT%H:%M:%S')
    ssh -o ConnectTimeout=3 "$HOST" "
        echo \"=== $HOST status @ $ts ===\"
        echo
        echo '== current pipeline stage =='
        grep -hE '^Start|^Done|^=== \[|^MATCHA|^POSTPROCESS|^TOTAL|RESULT|OutOfMemory|Error|Traceback' \"$LOG\" 2>/dev/null | tail -8
        echo
        echo '== output dir =='
        if [ -d \"$OUT\" ]; then
            for d in \"$OUT\"/*/ ; do
                if [ -d \"\$d\" ]; then
                    n=\$(find \"\$d\" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d ' ')
                    sz=\$(du -sh \"\$d\" 2>/dev/null | cut -f1)
                    echo \"  \$(basename \"\$d\")/  \$n files, \$sz\"
                fi
            done
        else
            echo '  (output dir does not exist yet)'
        fi
        echo
        echo '== GPU =='
        nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>&1
        echo
        echo '== compute apps =='
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1 | head -5
        echo
        echo '== process =='
        docker exec matcha-build pgrep -af 'python train|run_mast3r|extract_tetra' 2>&1 | head -2 | sed 's/[^[:space:]]\\+ //'
    " 2>&1
}

if [ "$WATCH" -eq 1 ]; then
    while true; do
        clear
        snap
        echo
        echo "(watching every ${INTERVAL}s — Ctrl+C to exit)"
        sleep "$INTERVAL"
    done
else
    snap
fi
