#!/usr/bin/env bash
# Thin wrapper: canonical script ships in krabby-launcher
# (krabby/scripts/pair_pro_controller.sh → `krabby pair-pro`).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PACKAGED="$ROOT/krabby/scripts/pair_pro_controller.sh"
if [[ -f "$PACKAGED" ]]; then
  exec bash "$PACKAGED" "$@"
fi
if command -v krabby >/dev/null 2>&1; then
  exec krabby pair-pro "$@"
fi
echo "[err] pairing script not found at $PACKAGED and krabby pair-pro is unavailable" >&2
exit 1
