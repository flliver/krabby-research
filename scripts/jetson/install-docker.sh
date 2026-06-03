#!/bin/bash
# Install Docker Engine on a Jetson Orin (Ubuntu 22.04 / jammy, arm64).
# Implements section 2 ("Docker Installation") of docs/JETSON_DEPLOYMENT.md.
#
# Run this BEFORE scripts/jetson/setup-docker-gpu.sh: this installs Docker
# Engine, that one wires up the NVIDIA container runtime for GPU access.
#
# Usage (on the Jetson):
#     ./scripts/jetson/install-docker.sh
# Escalates per-command via sudo; you may be prompted for your password.

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "Installing Docker Engine..."

# Skip if Docker Engine is already present.
if command -v docker &> /dev/null; then
    echo "✓ docker already installed ($(docker --version)); nothing to do."
    exit 0
fi

# Sanity-check the platform; the apt repo below targets Debian-based distros.
if ! command -v apt-get &> /dev/null; then
    echo "ERROR: apt-get not found. This script targets Ubuntu/Debian (Jetson)."
    exit 1
fi

echo "==> Updating package index and installing prerequisites"
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release

echo "==> Adding Docker's official GPG key"
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | sudo gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "==> Configuring the Docker apt repository"
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

echo "==> Installing Docker Engine packages"
sudo apt-get update
sudo apt-get install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin

echo "==> Adding '${SUDO_USER:-$USER}' to the docker group"
sudo usermod -aG docker "${SUDO_USER:-$USER}"

echo "==> Enabling and starting the Docker service"
sudo systemctl enable --now docker

echo ""
echo "✓ Docker Engine installed:"
docker --version
docker compose version || true

echo ""
echo "IMPORTANT: group membership for 'docker' takes effect on next login."
echo "Run 'newgrp docker' or log out/in before using docker without sudo."
echo "Next: run scripts/jetson/setup-docker-gpu.sh to enable GPU (NVIDIA) runtime."
