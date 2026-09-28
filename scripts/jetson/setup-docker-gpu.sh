#!/bin/bash
# Script to configure Docker for GPU support
# Based on official NVIDIA Container Toolkit installation guide:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

set -e

echo "Setting up Docker GPU support..."

# Check if nvidia-smi works
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Please install NVIDIA drivers first."
    echo "See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#prerequisites"
    exit 1
fi

echo "✓ NVIDIA drivers detected"

# Detect distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO=$ID
else
    echo "ERROR: Cannot detect Linux distribution"
    exit 1
fi

# Check if nvidia-container-toolkit is already installed
if command -v nvidia-ctk &> /dev/null; then
    echo "✓ nvidia-container-toolkit already installed"
    INSTALLED=true
else
    INSTALLED=false
fi

# Install nvidia-container-toolkit based on distribution
if [ "$INSTALLED" = false ]; then
    echo "Installing nvidia-container-toolkit..."
    
    if command -v apt-get &> /dev/null; then
        # Ubuntu/Debian installation (from official docs)
        echo "Detected Debian-based distribution. Installing prerequisites..."
        sudo apt-get update && sudo apt-get install -y --no-install-recommends \
            curl \
            gnupg2
        
        echo "Configuring production repository..."
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        
        echo "Updating package list..."
        sudo apt-get update
        
        echo "Installing nvidia-container-toolkit packages..."
        # Install latest version (remove version pinning to get latest)
        sudo apt-get install -y \
            nvidia-container-toolkit \
            nvidia-container-toolkit-base \
            libnvidia-container-tools \
            libnvidia-container1
        
        echo "✓ nvidia-container-toolkit installed"
        
    elif command -v dnf &> /dev/null; then
        # RHEL/CentOS/Fedora installation
        echo "Detected RPM-based distribution. Installing prerequisites..."
        sudo dnf install -y curl
        
        echo "Configuring production repository..."
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
            sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
        
        echo "Installing nvidia-container-toolkit packages..."
        sudo dnf install -y \
            nvidia-container-toolkit \
            nvidia-container-toolkit-base \
            libnvidia-container-tools \
            libnvidia-container1
        
        echo "✓ nvidia-container-toolkit installed"
        
    elif command -v zypper &> /dev/null; then
        # OpenSUSE/SLE installation
        echo "Detected zypper-based distribution. Configuring repository..."
        sudo zypper ar https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo
        
        echo "Installing nvidia-container-toolkit packages..."
        sudo zypper --gpg-auto-import-keys install -y \
            nvidia-container-toolkit \
            nvidia-container-toolkit-base \
            libnvidia-container-tools \
            libnvidia-container1
        
        echo "✓ nvidia-container-toolkit installed"
    else
        echo "ERROR: Unsupported package manager. Please install nvidia-container-toolkit manually."
        echo "See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
        exit 1
    fi
fi

# Configure Docker to use nvidia runtime
echo "Configuring Docker runtime..."
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker daemon
echo "Restarting Docker daemon..."
sudo systemctl restart docker

echo ""
echo "✓ Docker GPU support configured!"
echo ""
echo "Verifying configuration..."
sleep 2  # Give Docker a moment to restart
if docker info 2>/dev/null | grep -qi nvidia; then
    echo "✓ NVIDIA runtime detected in Docker"
else
    echo "WARNING: nvidia runtime not found in docker info."
    echo "You may need to restart Docker manually or check the configuration."
fi

echo ""
echo "Testing GPU access in Docker..."
if docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    echo "✓ GPU test passed - Docker can access NVIDIA GPUs"
else
    echo "WARNING: GPU test failed."
    echo "You may need to:"
    echo "  - Restart your terminal session"
    echo "  - Log out and log back in"
    echo "  - Check that Docker daemon is running: sudo systemctl status docker"
fi

