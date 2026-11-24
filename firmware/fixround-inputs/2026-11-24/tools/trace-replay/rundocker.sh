#!/bin/sh
# usage: rundocker.sh '<cmd run in /scratch/ced-verify/trace-replay>'
S="C:/Users/nickm/AppData/Local/Temp/claude/78c86551260afdf9/720ce176-f876-4a62-9959-4e2b88a4d5da/scratchpad"
MSYS_NO_PATHCONV=1 docker run --rm --entrypoint /bin/sh -v "C:/Users/nickm/krabby/krabby-research:/src:ro" -v "$S:/scratch" krabby-fwtest:latest -c "cd /scratch/ced-verify/trace-replay && $1"
