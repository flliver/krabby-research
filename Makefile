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
ifneq ($(filter build-test-image build-test-image-arm build-isaacsim-image build-locomotion-image build-scene-reconstruction-base-image build-matcha-image build-mast3r-image build-slam3r-image build-vggt-image build-m11-images test test-coverage,$(MAKECMDGOALS)),)
ifeq ($(OS),Windows_NT)
DOCKER_BIN := $(shell where docker 2>NUL)
else
DOCKER_BIN := $(shell command -v docker 2>/dev/null)
endif
ifeq ($(strip $(DOCKER_BIN)),)
$(error Docker CLI not found on PATH. Install Docker Desktop and restart your shell, or ensure `docker` is available.)
endif
endif

# BuildKit enables RUN --mount (pip cache, etc.). Docker Desktop enables it by default; set explicitly on Linux CI.
ifeq ($(OS),Windows_NT)
DOCKER_BUILD := docker build
else
DOCKER_BUILD := DOCKER_BUILDKIT=1 docker build
endif

# Arduino flashing defaults (override with FQBN/PORT in env or CLI)
FQBN ?=
PORT ?=

.PHONY: flash-task2
flash-task2:
ifeq ($(OS),Windows_NT)
	@powershell -NoProfile -Command "\
		$$cli = Join-Path (Resolve-Path '$(CURDIR)/../tools') 'arduino-cli.exe'; \
		if (-not (Test-Path $$cli)) { Write-Error 'arduino-cli.exe not found in ../tools. Run make setup-tools first.'; exit 1 } ; \
		$$ports = & $$cli board list --format json | ConvertFrom-Json; \
		$$match = $$ports.detected_ports | Where-Object { $$_.matching_boards -and $$_.matching_boards.Count -gt 0 } | Select-Object -First 1; \
		if (-not $$match) { Write-Error 'No boards detected. Plug in the Arduino and retry.'; exit 1 } ; \
		$$b = $$match.matching_boards[0]; \
		$$fqbn = $$b.fqbn; \
		$$port = $$match.port.address; \
		$$inc = Join-Path (Resolve-Path '$(CURDIR)') 'firmware/Task2_SixAxis/Six_Axis_Controller'; \
		if (-not $$fqbn -or -not $$port) { Write-Error 'Could not determine FQBN/port from arduino-cli board list.'; exit 1 } ; \
		Write-Host ('Using FQBN=' + $$fqbn + ' PORT=' + $$port); \
		Write-Host 'Compiling...'; \
		if (-not (& $$cli compile --fqbn $$fqbn --build-property compiler.cpp.extra_flags=-I$$inc firmware/Task2_SixAxis/Six_Axis_Controller)) { exit 1 }; \
		Write-Host 'Uploading...'; \
		if (-not (& $$cli upload --fqbn $$fqbn -p $$port firmware/Task2_SixAxis/Six_Axis_Controller)) { exit 1 }; \
		Write-Host 'Upload complete.'"
else
	@echo "flash-task2: not implemented for this OS" && exit 1
endif

.PHONY: setup-tools
setup-tools:
ifeq ($(OS),Windows_NT)
	@echo "Setting up arduino-cli in ../tools (Windows)..."
	@powershell -NoProfile -Command "\
	$$tools = Resolve-Path '$(CURDIR)/../tools'; \
	New-Item -ItemType Directory -Force -Path $$tools | Out-Null; \
	$$zip = Join-Path $$tools 'arduino-cli.zip'; \
	$$exe = Join-Path $$tools 'arduino-cli.exe'; \
	if (-Not (Test-Path $$exe)) { \
		Write-Host 'Downloading arduino-cli...'; \
		Invoke-WebRequest -Uri https://downloads.arduino.cc/arduino-cli/arduino-cli_latest_Windows_64bit.zip -OutFile $$zip; \
		Expand-Archive -Path $$zip -DestinationPath $$tools -Force; \
		$$found = Get-ChildItem -Path $$tools -Filter 'arduino-cli*.exe' | Select-Object -First 1; \
		if ($$found) { Move-Item $$found.FullName $$exe -Force; } \
		Remove-Item $$zip -Force; \
	} else { Write-Host 'arduino-cli.exe already present at' $$exe; } \
	$$old = [Environment]::GetEnvironmentVariable('Path','User'); \
	if ($$old -notlike ('*' + $$tools + '*')) { \
		[Environment]::SetEnvironmentVariable('Path', $$old + ';' + $$tools, 'User'); \
		Write-Host 'Added to user PATH - open a new shell and restart VSCode to pick up changes.'; \
	} else { Write-Host 'tools already on PATH - you may need to restart VSCode if its not in env:path.'; }"
else
	@echo "setup-tools: not implemented for this OS" && exit 1
endif

.PHONY: build-wheels
# `python -m build` leaves prior wheels in dist/ when a version is bumped. Docker images
# COPY dist/*.whl into one directory and pip install fails if two versions of the same
# package are present (e.g. krabby-compute-parkour 0.1.0 and 0.1.1). Clear dist first.
build-wheels:
	@echo "Building wheels for all packages..."
	@cd hal/client && rm -rf dist && $(PYTHON) -m build --wheel
	@cd controller && rm -rf dist && $(PYTHON) -m build --wheel
	@cd hal/server && rm -rf dist && $(PYTHON) -m build --wheel
	@cd hal/server/isaac && rm -rf dist && $(PYTHON) -m build --wheel
	@cd hal/server/jetson && rm -rf dist && $(PYTHON) -m build --wheel
	@cd hal/tools && rm -rf dist && $(PYTHON) -m build --wheel
	@cd data_collection && rm -rf dist && $(PYTHON) -m build --wheel
	@cd compute/parkour && rm -rf dist && $(PYTHON) -m build --wheel
	@rm -rf parkour/dist && $(PYTHON) scripts/wheel-build/build_parkour_wheel.py
	@cd teleop/edge && rm -rf dist && $(PYTHON) -m build --wheel
	@cd teleop/portal && rm -rf dist && $(PYTHON) -m build --wheel
	@echo "Wheels built in dist/ directories"

.PHONY: clean
# `python -m build` does not remove old wheels from dist/ after a version bump. Docker images
# COPY dist/*.whl into one directory and pip install fails if two versions of the same package
# are present (e.g. krabby-compute-parkour 0.1.0 and 0.1.1).
clean:
	rm -rf hal/*/dist hal/*/build hal/*/*.egg-info
	rm -rf hal/server/*/dist hal/server/*/build hal/server/*/*.egg-info
	rm -rf compute/*/dist compute/*/build compute/*/*.egg-info
	rm -rf parkour/dist parkour/build
	rm -rf data_collection/dist data_collection/build data_collection/*.egg-info
	rm -rf teleop/edge/dist teleop/edge/build teleop/edge/*.egg-info
	rm -rf teleop/portal/dist teleop/portal/build teleop/portal/*.egg-info
	rm -rf controller/dist controller/build controller/*.egg-info
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

.PHONY: install-editable
install-editable:
	@echo "Installing packages in editable mode (for development)..."
	@echo "This allows you to edit source files in wheel package directories and see changes immediately."
	@$(PIP) install -e hal/client
	@$(PIP) install -e controller
	@$(PIP) install -e hal/server
	@$(PIP) install -e hal/server/isaac
	@$(PIP) install -e hal/server/jetson
	@$(PIP) install -e hal/tools
	@$(PIP) install -e data_collection
	@$(PIP) install -e compute/parkour
	@echo "Packages installed in editable mode. Edit files in hal/*/, controller/, and compute/*/ directories."

# Run the same build-and-test steps as the publish workflow for a package (or all).
# Usage: make test-publish-job PKG=hal-client  or  make test-publish-job PKG=all
# Package keys: hal-client, hal-server, compute-parkour, controller, hal-tools, hal-server-isaac, hal-server-jetson, data-collection
.PHONY: test-publish-job
test-publish-job:
	@./scripts/test-publish-job.sh $(PKG)

# Build cache directory (for heavy downloads like Isaac Lab, reused across Docker builds)
BUILD_CACHE := $(CURDIR)/.build-cache
ISAACLAB_CACHE := $(BUILD_CACHE)/isaaclab
# Pin Isaac Lab to specific commit for reproducibility
# Update this commit hash when you need to use a different version
ISAACLAB_COMMIT := 64ecea24f51bd008d2ae75dc2a233db5db515a71

.PHONY: isaaclab-cache
isaaclab-cache:
	@echo "Setting up Isaac Lab git cache..."
	@if [ ! -d "$(ISAACLAB_CACHE)" ]; then \
		echo "Cloning Isaac Lab (this may take a while, progress visible below)..."; \
		git clone https://github.com/isaac-sim/IsaacLab.git $(ISAACLAB_CACHE); \
		cd $(ISAACLAB_CACHE) && git checkout $(ISAACLAB_COMMIT); \
		echo "Isaac Lab cloned to $(ISAACLAB_CACHE) at commit $(ISAACLAB_COMMIT)"; \
	else \
		echo "Isaac Lab cache already exists at $(ISAACLAB_CACHE)"; \
		echo "Using existing cache (assuming it was created at commit $(ISAACLAB_COMMIT))"; \
	fi
	@echo "Isaac Lab cache ready at $(ISAACLAB_CACHE) (commit: $(ISAACLAB_COMMIT))"


.PHONY: build-test-image
build-test-image: build-wheels isaaclab-cache
	@echo "Building x86 test Docker image..."
	$(DOCKER_BUILD) -f images/testing/x86/Dockerfile -t krabby-testing-x86:latest .
	@echo "Test image built: krabby-testing-x86:latest"

.PHONY: build-isaacsim-image
build-isaacsim-image: build-wheels isaaclab-cache
	@echo "Building Isaac Sim Docker image..."
	@echo "Note: Requires NVIDIA NGC authentication for base image"
	$(DOCKER_BUILD) -f images/isaacsim/Dockerfile -t krabby-isaacsim:latest .
	@echo "Isaac Sim image built: krabby-isaacsim:latest"

.PHONY: build-locomotion-image
build-locomotion-image: build-wheels
	@echo "Building locomotion Docker image (for Jetson/ARM64)..."
	@echo "Note: This target is for building on Jetson hardware (native ARM64)"
	@echo "      For cross-platform builds from x86_64, use buildx manually"
	@echo "      --network=host allows the build to access the network during the build"
	$(DOCKER_BUILD) --network=host -f images/locomotion/Dockerfile -t krabby-locomotion:latest .
	@echo "Locomotion image built: krabby-locomotion:latest"

.PHONY: build-test-image-arm
build-test-image-arm: build-wheels
	@echo "Building ARM test Docker image..."
	@echo "Note: This target is for building on ARM testing environment (native ARM64)"
	@echo "      For cross-platform builds from x86_64, use buildx manually"
	$(DOCKER_BUILD) -f images/testing/arm/Dockerfile -t krabby-testing-arm:latest .
	@echo "ARM test image built: krabby-testing-arm:latest"

# -------------------------------------------------------------------
# M11 scene-reconstruction images
# -------------------------------------------------------------------
# Each is self-contained: Dockerfile + requirements.txt + patches/ live
# under images/<name>/, and the build context is the image's own dir
# (not the repo root) — they don't need access to HAL wheels or the
# isaaclab cache.

.PHONY: build-scene-reconstruction-base-image
build-scene-reconstruction-base-image:
	@echo "Building scene-reconstruction-base Docker image (COLMAP + Open3D)..."
	$(DOCKER_BUILD) -f images/scene-reconstruction-base/Dockerfile \
		-t krabby-scene-reconstruction-base:latest \
		images/scene-reconstruction-base/
	@echo "Base image built: krabby-scene-reconstruction-base:latest"

.PHONY: build-matcha-image
build-matcha-image:
	@echo "Building matcha Docker image (primary T1 watertight-mesh pipeline)..."
	@echo "Note: ~30-min build on RTX 5080; pre-downloads ~4.3 GB of model checkpoints"
	$(DOCKER_BUILD) -f images/matcha/Dockerfile -t krabby-matcha:latest images/matcha/
	@echo "Matcha image built: krabby-matcha:latest"

.PHONY: build-mast3r-image
build-mast3r-image:
	@echo "Building mast3r Docker image (T0 alternative SLAM pipeline)..."
	@echo "Note: ~36 GB image; uses NGC PyTorch 25.10 base"
	$(DOCKER_BUILD) -f images/mast3r/Dockerfile -t krabby-mast3r:latest images/mast3r/
	@echo "Mast3r image built: krabby-mast3r:latest"

.PHONY: build-slam3r-image
build-slam3r-image:
	@echo "Building slam3r Docker image (Appendix A alternative)..."
	$(DOCKER_BUILD) -f images/slam3r/Dockerfile -t krabby-slam3r:latest images/slam3r/
	@echo "Slam3r image built: krabby-slam3r:latest"

.PHONY: build-vggt-image
build-vggt-image:
	@echo "Building vggt Docker image (Appendix A alternative)..."
	$(DOCKER_BUILD) -f images/vggt/Dockerfile -t krabby-vggt:latest images/vggt/
	@echo "Vggt image built: krabby-vggt:latest"

.PHONY: build-m11-images
build-m11-images: build-scene-reconstruction-base-image build-matcha-image build-mast3r-image build-slam3r-image build-vggt-image
	@echo "All M11 scene-reconstruction images built."

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
		pytest tests/ -v -m "not jetson and not isaacsim" --cov=. --cov-report=html --cov-report=term

.PHONY: test-isaacsim
test-isaacsim: build-isaacsim-image
	@echo "Running inference loop test on Isaac Sim container..."
	@echo "Note: This test requires a checkpoint file and Isaac Lab packages"
	@echo "Note: The environment is not currently resetting correctly between tests,"
	@echo "      so we only run the inference loop test (test_inference_latency_requirement)"
	@echo "      to avoid issues when running multiple tests in sequence."
	@echo "Note: To run a specific test with recommended options:"
	@echo "  PYTHONUNBUFFERED=1 timeout 300 docker run --rm --gpus all \\"
	@echo "    --entrypoint /workspace/run_test_runner.sh \\"
	@echo "    krabby-isaacsim:latest <test_name>"
	@echo "See test_runner.py and run_test_runner.sh for more information"
	PYTHONUNBUFFERED=1 timeout 600 docker run --rm --gpus all \
		--entrypoint /workspace/run_test_runner.sh \
		krabby-isaacsim:latest \
		test_inference_latency_requirement

