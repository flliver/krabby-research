#!/usr/bin/env bash
# sync-tools.sh — keep fastmap's baked krabby-tools/ in lockstep with the
# canonical real2sim/ sources, or (--check) fail the build if they drift.
# STO-SCN-157 (EPI-SCN-FLEET-IMAGE-DEPLOY).
#
# Single-source rule (T-023): real2sim/<f> is CANONICAL. The copy under
# images/fastmap/krabby-tools/<f> is a build-time mirror that gets baked into
# the image via `COPY krabby-tools/`. Without a guard the mirror silently
# drifts from canonical (it did: the audit found covis_graph.py + lib_progress.sh
# stale, so the registry fastmap ran old covis logic). This script removes the
# manual "remember to sync" footgun (T-003 — fix the root cause).
#
# Files in krabby-tools/ with NO real2sim/ counterpart (e.g. run_fastmap.sh)
# are image-local and are left untouched.
#
# Usage:
#   images/fastmap/sync-tools.sh            # sync canonical -> krabby-tools (default)
#   images/fastmap/sync-tools.sh --check    # exit 1 if any drift (CI / pre-build gate)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$HERE/krabby-tools"
CANON="$(cd "$HERE/../../real2sim" && pwd)"
MODE="${1:-sync}"

drift=0
synced=0
for f in "$TOOLS"/*; do
  b="$(basename "$f")"
  src="$CANON/$b"
  [ -f "$src" ] || continue                 # image-local file: no canonical source
  cmp -s "$src" "$f" && continue            # already in sync
  if [ "$MODE" = "--check" ]; then
    echo "DRIFT: krabby-tools/$b != real2sim/$b"
    drift=1
  else
    cp "$src" "$f"
    echo "synced: $b  (real2sim -> krabby-tools)"
    synced=$((synced + 1))
  fi
done

if [ "$MODE" = "--check" ]; then
  if [ "$drift" -eq 0 ]; then
    echo "krabby-tools in sync with real2sim ✓"
  else
    echo "krabby-tools DRIFT — run 'images/fastmap/sync-tools.sh' before building (STO-SCN-157)" >&2
    exit 1
  fi
else
  echo "sync complete (${synced} file(s) updated)"
fi
