#!/usr/bin/env bash
# Assemble Next.js `output: "standalone"` into a runnable directory tree.
# Usage: assemble-standalone.sh <portal-src-dir> <dest-dir>
set -euo pipefail

SRC="${1:?portal source dir}"
DEST="${2:?destination dir}"

if [[ ! -f "$SRC/.next/standalone/server.js" ]]; then
  echo "error: $SRC/.next/standalone/server.js missing — run npm run build first" >&2
  exit 1
fi

rm -rf "$DEST"
mkdir -p "$DEST"
cp -a "$SRC/.next/standalone/." "$DEST/"
mkdir -p "$DEST/.next"
cp -a "$SRC/.next/static" "$DEST/.next/static"
if [[ -d "$SRC/public" ]]; then
  cp -a "$SRC/public" "$DEST/public"
fi
