#!/usr/bin/env bash
# install.sh — install krabby-scene-sync + systemd user units on a Linux fleet
# host (STO-SCN-030). Source-controlled in krabby-research/scripts/scene-sync/
# (HUG-KRB-002) — never hand-copy units.
#
# Usage:
#   ./install.sh [--config-enabled | --config-disabled]
#
#   --config-enabled    write ~/.config/krabby/scene-sync.toml with
#                       enabled=true IF NO CONFIG EXISTS (never overwrites)
#   --config-disabled   same, with enabled=false
#   (no flag)           install script+units only; the gate stays whatever
#                       the host's config says (absent config = sync OFF)
#
# What it does:
#   1. krabby-scene-sync        -> ~/.local/bin/
#   2. units (timer interval templated from config interval_minutes, default 30)
#                               -> ~/.config/systemd/user/
#   3. systemctl --user daemon-reload && enable --now krabby-scene-sync.timer
#   4. checks loginctl linger (user timers need it on headless hosts);
#      attempts enable-linger, warns if it cannot
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$HOME/.config/krabby/scene-sync.toml"

write_config() { # write_config <true|false>
    if [[ -f "$CONFIG" ]]; then
        echo "config exists, NOT overwriting: $CONFIG"
        return
    fi
    mkdir -p "$(dirname "$CONFIG")"
    cat > "$CONFIG" <<EOF
# krabby scene-store auto-sync gate (STO-SCN-030).
# This file is THE per-host opt-in: absent or enabled=false => no sync, ever.
[sync]
enabled  = $1
remote   = "j"
interval_minutes = 30
lfs      = "full"
store    = "~/krabby/scenes"
EOF
    echo "wrote $CONFIG (enabled = $1)"
}

case "${1:-}" in
    --config-enabled)  write_config true ;;
    --config-disabled) write_config false ;;
    "") ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
esac

# 1. script
install -D -m 0755 "$HERE/krabby-scene-sync" "$HOME/.local/bin/krabby-scene-sync"
echo "installed ~/.local/bin/krabby-scene-sync"

# 2. units (interval from config, default 30)
INTERVAL=30
if [[ -f "$CONFIG" ]]; then
    v="$(awk '/^\[/{s=($0~/^\[sync\]/)} s && /^[ \t]*interval_minutes[ \t]*=/{sub(/^[^=]*=[ \t]*/,"");sub(/[ \t]*(#.*)?$/,"");print;exit}' "$CONFIG")"
    [[ "$v" =~ ^[0-9]+$ ]] && INTERVAL="$v"
fi
UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$UNIT_DIR"
install -m 0644 "$HERE/krabby-scene-sync.service" "$UNIT_DIR/krabby-scene-sync.service"
sed "s/@INTERVAL@/$INTERVAL/" "$HERE/krabby-scene-sync.timer" > "$UNIT_DIR/krabby-scene-sync.timer"
echo "installed user units (timer interval ${INTERVAL}min)"

# 3. enable
systemctl --user daemon-reload
systemctl --user enable --now krabby-scene-sync.timer
echo "timer enabled: $(systemctl --user is-enabled krabby-scene-sync.timer) / $(systemctl --user is-active krabby-scene-sync.timer)"

# 4. linger — user timers don't run without a session unless linger is on
if [[ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != "yes" ]]; then
    if loginctl enable-linger "$USER" 2>/dev/null; then
        echo "enabled linger for $USER"
    else
        echo "WARNING: linger is OFF and could not be enabled — timer only runs while $USER has a session. Run: sudo loginctl enable-linger $USER" >&2
    fi
else
    echo "linger already enabled"
fi
echo "done. gate: $([[ -f "$CONFIG" ]] && grep -m1 '^enabled' "$CONFIG" || echo 'no config => sync OFF')"
