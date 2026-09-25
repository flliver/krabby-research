#!/usr/bin/env bash
# Light reset for the bench harness on a dual-use Orin.
#
# Clears enough state that the next Install stage is a fresh PyPI install +
# image pull, without the full pristine wipe of jetson-reset.sh.
#
# NEVER removes ~/.venv-krabby (manual cold-start venv).
# ALWAYS removes ~/.venv-krabby-bench (bench/CI venv) when present.
#
# Usage (as the kit user, from anywhere):
#   ./scripts/jetson/bench-reset.sh
#   ./scripts/jetson/bench-reset.sh --rmi          # also delete locomotion images
#   ./scripts/jetson/bench-reset.sh --keep-locomotion-unit  # leave unit enabled
#
# After reset, recreate the bench venv:
#   python3 -m venv ~/.venv-krabby-bench
#   source ~/.venv-krabby-bench/bin/activate
#   pip install -U pip && pip install krabby-launcher
#   sudo -E env PATH="$PATH" "$(which krabby)" install --no-launch-on-startup

set -euo pipefail

RMI=0
KEEP_LOCOMOTION_UNIT=0
for arg in "$@"; do
  case "$arg" in
    --rmi) RMI=1 ;;
    --keep-locomotion-unit) KEEP_LOCOMOTION_UNIT=1 ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      echo "error: unknown argument: $arg (try --help)" >&2
      exit 2
      ;;
  esac
done

# Resolve the kit user's home even if invoked via sudo.
if [[ -n "${SUDO_USER:-}" ]]; then
  KIT_USER="$SUDO_USER"
  KIT_HOME="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
else
  KIT_USER="$(id -un)"
  KIT_HOME="$HOME"
fi

BENCH_VENV="${KIT_HOME}/.venv-krabby-bench"
COLD_VENV="${KIT_HOME}/.venv-krabby"
STATE_JSON="${KIT_HOME}/.config/krabby/state.json"
ECR_REPO="public.ecr.aws/t7t7b3i3/krabby-locomotion"

echo "==> Dual-use light reset (user=${KIT_USER}, home=${KIT_HOME})"
echo "    Preserving cold-start venv: ${COLD_VENV}"
echo "    Removing bench venv:        ${BENCH_VENV}"

echo "==> Stopping locomotion container name races (F6)"
if command -v docker >/dev/null 2>&1; then
  docker rm -f krabby 2>/dev/null || true
  # Best-effort: any other container with "krabby" in the name except orphans we keep.
  while read -r cid; do
    [[ -n "$cid" ]] || continue
    docker rm -f "$cid" 2>/dev/null || true
  done < <(docker ps -aq --filter name=krabby 2>/dev/null || true)
else
  echo "    [skip] docker not on PATH"
fi

echo "==> Stopping krabby-locomotion so it cannot restart the container mid-reset"
if command -v systemctl >/dev/null 2>&1; then
  sudo systemctl stop krabby-locomotion.service 2>/dev/null || true
  if [[ "$KEEP_LOCOMOTION_UNIT" -eq 0 ]]; then
    sudo systemctl disable krabby-locomotion.service 2>/dev/null || true
    echo "    disabled krabby-locomotion.service (bench mode)"
  else
    echo "    left krabby-locomotion enablement unchanged (--keep-locomotion-unit)"
  fi
else
  echo "    [skip] systemctl not found"
fi

echo "==> Clearing install state (${STATE_JSON})"
# Shared by both venvs — must clear so Install re-pulls and re-writes state.
rm -f "$STATE_JSON"
# Leave the directory; only the digest/ref file is the Install marker.

echo "==> Removing bench venv only"
if [[ -e "$COLD_VENV" && "$(readlink -f "$BENCH_VENV" 2>/dev/null || true)" == "$(readlink -f "$COLD_VENV" 2>/dev/null || true)" ]]; then
  echo "error: bench venv path resolves to cold-start venv; refusing to delete" >&2
  exit 1
fi
if [[ -d "$BENCH_VENV" ]]; then
  rm -rf "$BENCH_VENV"
  echo "    removed ${BENCH_VENV}"
else
  echo "    [ok] ${BENCH_VENV} already absent"
fi
if [[ -d "$COLD_VENV" ]]; then
  echo "    [ok] preserved ${COLD_VENV}"
else
  echo "    [warn] cold-start venv ${COLD_VENV} not found (ok if not created yet)"
fi

if [[ "$RMI" -eq 1 ]]; then
  echo "==> Removing locomotion Docker images (--rmi)"
  if command -v docker >/dev/null 2>&1; then
    docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' \
      | awk -v repo="$ECR_REPO" '$1 ~ repo || $1 ~ /krabby-locomotion/ { print $2 }' \
      | sort -u \
      | while read -r id; do
          [[ -n "$id" ]] || continue
          docker rmi -f "$id" 2>/dev/null || true
        done
  fi
else
  echo "==> Keeping Docker images (default). Pass --rmi to force a cold pull."
  echo "    Note: krabby install always docker-pulls; local layers only speed the pull."
fi

echo ""
echo "Done (light reset)."
echo ""
echo "Not cleared (dual-use gap vs bare Orin):"
echo "  - ${COLD_VENV}"
echo "  - repo clone, SSH, BT Pro Controller bond"
echo "  - udev / dialout / hid_nintendo (re-asserted by next krabby install)"
echo "  - /etc/krabby/iot/ (enroll identity; wipe only when testing enroll)"
echo ""
echo "Next — recreate bench venv and Install stage:"
echo "  python3 -m venv ${BENCH_VENV}"
echo "  source ${BENCH_VENV}/bin/activate"
echo "  pip install -U pip && pip install krabby-launcher"
echo "  sudo -E env PATH=\"\$PATH\" \"\$(which krabby)\" install --no-launch-on-startup"
echo "  krabby firmware show"
echo ""
echo "Bring-up mode later: source ${COLD_VENV}/bin/activate"
echo "  (re-run krabby install if you need state.json / boot unit again)"
