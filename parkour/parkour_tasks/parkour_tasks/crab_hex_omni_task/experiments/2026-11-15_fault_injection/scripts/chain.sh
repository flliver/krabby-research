#!/bin/bash
# Fault sweep: run the grid phases serially in one detached process (nominals -> es1 -> full read -> es2 -> kind
# -> c1 -> closing). Stops at the first phase that does not end ok. Last line: CHAIN END status=<ok|stop> phase=<p>.
set -u
D=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=/home/nickmagus/krabby/isaac_venv/bin/python
R=/home/nickmagus/krabby/krabby-research/parkour/outputs/fault_sweep/lr
cd "$D"
for p in ${PHASES:-nominals es1 READ es2 kind c1 closing}; do
  if [ "$p" = READ ]; then
    echo "CHAIN read after es1 $(date -Is)"
    if ! $PY scripts/fault_read.py --root "$R"; then echo "CHAIN END status=stop phase=READ"; exit 1; fi
    continue
  fi
  echo "CHAIN phase $p start $(date -Is)"
  $PY -u scripts/fault_queue.py run --phase "$p"
  rc=$?
  if [ $rc -ne 0 ]; then echo "CHAIN END status=stop phase=$p rc=$rc"; exit 1; fi
done
echo "CHAIN END status=ok phase=closing"
