# Virtualenv selection logic:
# 1. If a venv is already activated (VIRTUAL_ENV is set), use that
# 2. Otherwise, if ./testenv exists, use it
# 3. Otherwise, fail with an error

# Default target
.DEFAULT_GOAL := test

ifdef VIRTUAL_ENV
VENV_ROOT := $(VIRTUAL_ENV)
else
VENV_ROOT := $(CURDIR)/testenv
endif

ifeq ($(OS),Windows_NT)
    VENV_PYTHON := $(VENV_ROOT)/Scripts/python.exe
    VENV_PIP    := $(VENV_ROOT)/Scripts/pip.exe
else
    VENV_PYTHON := $(VENV_ROOT)/bin/python
    VENV_PIP    := $(VENV_ROOT)/bin/pip
endif

.PHONY: venv
venv:
	 python -m venv $(VENV_ROOT)
	 $(VENV_PYTHON) -m pip install --progress-bar off --upgrade pip
	 $(VENV_PYTHON) -m pip install --progress-bar off build

# Allow `make venv` to run without pre-existing environment; gate the check for other targets.
ifeq ($(filter venv,$(MAKECMDGOALS)),)
ifeq ($(wildcard $(VENV_PYTHON)),)
$(error No Python virtual environment found. Activate a venv (VIRTUAL_ENV) or create ./testenv with: python3.11 -m venv testenv)
endif
endif

PYTHON := $(VENV_PYTHON)
PIP    := $(VENV_PIP)

# Docker availability check for Docker-dependent targets
ifneq ($(filter build-test-image build-test-image-arm build-isaacsim-image build-locomotion-image test test-coverage,$(MAKECMDGOALS)),)
ifeq ($(OS),Windows_NT)
DOCKER_BIN := $(shell where docker 2>NUL)
else
DOCKER_BIN := $(shell command -v docker 2>/dev/null)
endif
ifeq ($(strip $(DOCKER_BIN)),)
$(error Docker CLI not found on PATH. Install Docker Desktop and restart your shell, or ensure `docker` is available.)
endif
endif

.PHONY: build-wheels
build-wheels:
	@echo "Building wheels for all packages..."
	@cd hal/client && $(PYTHON) -m build --wheel
	@cd hal/server && $(PYTHON) -m build --wheel
	@cd hal/server/isaac && $(PYTHON) -m build --wheel
	@cd hal/server/jetson && $(PYTHON) -m build --wheel
	@cd hal/tools && $(PYTHON) -m build --wheel
	@cd compute/parkour && $(PYTHON) -m build --wheel
	@$(PYTHON) scripts/wheel-build/build_parkour_wheel.py
	@echo "Wheels built in dist/ directories"

.PHONY: clean
clean:
	rm -rf hal/*/dist hal/*/build hal/*/*.egg-info
	rm -rf hal/server/*/dist hal/server/*/build hal/server/*/*.egg-info
	rm -rf compute/*/dist compute/*/build compute/*/*.egg-info
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

.PHONY: install-editable
install-editable:
	@echo "Installing packages in editable mode (for development)..."
	@echo "This allows you to edit source files in wheel package directories and see changes immediately."
	@$(PIP) install -e hal/client
	@$(PIP) install -e hal/server
	@$(PIP) install -e hal/server/isaac
	@$(PIP) install -e hal/server/jetson
	@$(PIP) install -e hal/tools
	@$(PIP) install -e compute/parkour
	@echo "Packages installed in editable mode. Edit files in hal/*/ and compute/*/ directories."

# Build cache directory (for heavy downloads like Isaac Lab, reused across Docker builds)
BUILD_CACHE := $(CURDIR)/.build-cache
ISAACLAB_CACHE := $(BUILD_CACHE)/isaaclab

.PHONY: isaaclab-cache
isaaclab-cache:
	@echo "Setting up Isaac Lab git cache..."
	@if [ ! -d "$(ISAACLAB_CACHE)" ]; then \
		echo "Cloning Isaac Lab (this may take a while, progress visible below)..."; \
		git clone --depth 1 --branch main https://github.com/isaac-sim/IsaacLab.git $(ISAACLAB_CACHE); \
		echo "Isaac Lab cloned to $(ISAACLAB_CACHE)"; \
	else \
		echo "Isaac Lab cache already exists at $(ISAACLAB_CACHE)"; \
	fi
	@echo "Isaac Lab cache ready at $(ISAACLAB_CACHE)"


.PHONY: build-test-image
build-test-image: build-wheels
	@echo "Building x86 test Docker image..."
	docker build -f images/testing/x86/Dockerfile -t krabby-testing-x86:latest .
	@echo "Test image built: krabby-testing-x86:latest"

.PHONY: build-isaacsim-image
build-isaacsim-image: build-wheels isaaclab-cache
	@echo "Building Isaac Sim Docker image..."
	@echo "Note: Requires NVIDIA NGC authentication for base image"
	docker build -f images/isaacsim/Dockerfile -t krabby-isaacsim:latest .
	@echo "Isaac Sim image built: krabby-isaacsim:latest"

.PHONY: build-locomotion-image
build-locomotion-image: build-wheels
	@echo "Building locomotion Docker image (for Jetson/ARM64)..."
	@echo "Note: This target is for building on Jetson hardware (native ARM64)"
	@echo "      For cross-platform builds from x86_64, use buildx manually"
	docker build -f images/locomotion/Dockerfile -t krabby-locomotion:latest .
	@echo "Locomotion image built: krabby-locomotion:latest"

.PHONY: build-test-image-arm
build-test-image-arm: build-wheels
	@echo "Building ARM test Docker image..."
	@echo "Note: This target is for building on ARM testing environment (native ARM64)"
	@echo "      For cross-platform builds from x86_64, use buildx manually"
	docker build -f images/testing/arm/Dockerfile -t krabby-testing-arm:latest .
	@echo "ARM test image built: krabby-testing-arm:latest"

.PHONY: test
test: build-test-image
	@echo "Running all tests (excluding Jetson and Isaac Sim tests) in Docker container..."
	docker run --rm --gpus all \
		krabby-testing-x86:latest \
		pytest tests/ -v -m "not jetson and not isaacsim"

.PHONY: test-coverage
test-coverage: build-test-image
	@echo "Running tests with coverage (excluding Jetson and Isaac Sim tests) in Docker container..."
	docker run --rm --gpus all \
		-v $$(pwd)/tests/coverage:/workspace/tests/coverage \
		krabby-testing-x86:latest \
		pytest tests/ -v -m "not jetson and not isaacsim" --cov=hal --cov=compute --cov-report=html --cov-report=term

.PHONY: test-isaacsim
test-isaacsim: build-isaacsim-image
	@echo "Running all Isaac Sim tests on Isaac Sim container..."
	@echo "Note: These tests require a checkpoint file and Isaac Lab packages"
	@echo "Note: To run a specific test with recommended options:"
	@echo "  PYTHONUNBUFFERED=1 timeout 300 docker run --rm --gpus all \\"
	@echo "    --entrypoint /workspace/run_test_runner.sh \\"
	@echo "    krabby-isaacsim:latest <test_name>"
	@echo "See test_runner.py and run_test_runner.sh for more information"
	PYTHONUNBUFFERED=1 timeout 600 docker run --rm --gpus all \
		--entrypoint /workspace/run_test_runner.sh \
		krabby-isaacsim:latest

