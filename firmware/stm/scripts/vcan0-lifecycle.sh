#!/usr/bin/env bash
# Virtual CAN bring-up/teardown for SocketCAN FD on vcan (krabby.dbc SetDuty).
#
# Only deletes the interface on "down" if this script created it (stamp file).
# Pre-existing vcan0 (from another session) is left in place.
#
# Usage:
#   vcan0-lifecycle.sh up
#   vcan0-lifecycle.sh down
#   vcan0-lifecycle.sh run -- <command>   # up, run command, down on EXIT/INT/TERM
#
set -euo pipefail

IFACE="${KRABBY_CAN_IFACE:-vcan0}"
MTU="${KRABBY_VCAN_MTU:-72}"
STATE_FILE="${KRABBY_VCAN_STATE_FILE:-${XDG_RUNTIME_DIR:-/tmp}/krabby-${IFACE}.created-by-lifecycle}"

as_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

stamp_owned() {
  mkdir -p "$(dirname "$STATE_FILE")"
  date -u +"%Y-%m-%dT%H:%M:%SZ pid=$$" >"$STATE_FILE"
}

we_own_iface() {
  [[ -f "$STATE_FILE" ]]
}

cmd_up() {
  if ip link show "$IFACE" &>/dev/null; then
    echo "$IFACE already exists (not marked for delete unless this script created it)"
  else
    as_root modprobe vcan
    as_root ip link add dev "$IFACE" type vcan
    stamp_owned
    echo "Created $IFACE (will delete on: $0 down or end of 'run')"
  fi
  as_root ip link set "$IFACE" mtu "$MTU"
  as_root ip link set up "$IFACE"
  echo "$IFACE up, mtu=$MTU"
}

cmd_down() {
  if we_own_iface; then
    as_root ip link set down "$IFACE" 2>/dev/null || true
    as_root ip link del "$IFACE" 2>/dev/null || true
    rm -f "$STATE_FILE"
    echo "Removed $IFACE (lifecycle teardown)"
  else
    echo "No lifecycle stamp for $IFACE — leaving interface unchanged"
    echo "  (delete manually: sudo ip link del $IFACE)"
  fi
}

cmd_run() {
  if [[ $# -lt 1 || "$1" != "--" ]]; then
    echo "usage: $0 run -- <command> [args...]" >&2
    exit 2
  fi
  shift
  cmd_up
  trap 'cmd_down' EXIT INT TERM
  "$@"
}

case "${1:-}" in
  up) shift || true; cmd_up ;;
  down) shift || true; cmd_down ;;
  run) shift; cmd_run "$@" ;;
  *)
    echo "usage: $0 {up|down|run -- COMMAND ...}" >&2
    exit 2
    ;;
esac