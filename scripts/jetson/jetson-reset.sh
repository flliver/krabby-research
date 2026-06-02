#!/usr/bin/env bash
# Reset the Jetson to a pristine state with respect to krabby software.
# Removes all packages, services, config, udev rules, and Docker images.
# SSH keys and the krabby user account are left intact.
# Must be run as root (sudo).
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "error: run as root: sudo $0" >&2
    exit 1
fi

echo "==> Stopping and removing krabby-bench service"
systemctl stop krabby-bench 2>/dev/null || true
systemctl disable krabby-bench 2>/dev/null || true
rm -f /etc/systemd/system/krabby-bench.service \
      /etc/systemd/system/krabby-bench.timer
systemctl daemon-reload
systemctl reset-failed 2>/dev/null || true

echo "==> Removing config and state directories"
rm -rf /etc/krabby-bench /var/lib/krabby-bench

echo "==> Removing udev rules"
rm -f /etc/udev/rules.d/99-krabby-mega.rules
udevadm control --reload-rules

echo "==> Uninstalling system-wide pip packages"
pip3 uninstall -y \
    krabby krabby-bench krabby-controller krabby-firmware \
    krabby-hal-client krabby-hal-server krabby-launcher 2>/dev/null || true

echo "==> Uninstalling krabby user pip packages (removes user-install shadows)"
sudo -u krabby pip3 uninstall -y \
    krabby krabby-bench krabby-controller krabby-firmware \
    krabby-hal-client krabby-hal-server krabby-launcher 2>/dev/null || true

echo "==> Removing Docker images"
docker images --format '{{.Repository}}:{{.Tag}}' \
    | grep -i krabby \
    | xargs -r docker rmi -f 2>/dev/null || true

echo ""
echo "Done. Jetson is back to pristine state."
echo "Verify with: pip3 list | grep krabby && docker images | grep krabby"
