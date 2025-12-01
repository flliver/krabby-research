# Virtualenv selection logic:
# 1. If a venv is already activated (VIRTUAL_ENV is set), use that
# 2. Otherwise, if ./testenv exists, use it
# 3. Otherwise, fail with an error

ifdef VIRTUAL_ENV
VENV_ROOT := $(VIRTUAL_ENV)
else
VENV_ROOT := $(CURDIR)/testenv
endif

VENV_PYTHON := $(VENV_ROOT)/bin/python
VENV_PIP    := $(VENV_ROOT)/bin/pip

ifeq ($(wildcard $(VENV_PYTHON)),)
$(error No Python virtual environment found. Activate a venv (VIRTUAL_ENV) or create ./testenv with: python3.11 -m venv testenv)
endif

PYTHON := $(VENV_PYTHON)
PIP    := $(VENV_PIP)

.PHONY: build-wheels
build-wheels:
	@echo "Building wheels for all packages..."
	@cd hal/client && $(PYTHON) -m build --wheel
	@cd hal/server && $(PYTHON) -m build --wheel
	@cd hal/server/isaac && $(PYTHON) -m build --wheel
	@cd hal/server/jetson && $(PYTHON) -m build --wheel
	@cd hal/tools && $(PYTHON) -m build --wheel
	@echo "Wheels built in dist/ directories"

.PHONY: clean
clean:
	rm -rf hal/*/dist hal/*/build hal/*/*.egg-info
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
	@echo "Packages installed in editable mode. Edit files in hal/*/ directories."

.PHONY: build-test-image
build-test-image: build-wheels
	@echo "Building x86 test Docker image..."
	docker build -f images/testing/x86/Dockerfile -t krabby-testing-x86:latest .
	@echo "Test image built: krabby-testing-x86:latest"

.PHONY: test
test: build-test-image
	@echo "Running all tests (excluding Jetson tests) in Docker container..."
	docker run --rm --gpus all \
		krabby-testing-x86:latest \
		pytest tests/ -v -k "not jetson"

.PHONY: test-coverage
test-coverage: build-test-image
	@echo "Running tests with coverage (excluding Jetson tests) in Docker container..."
	docker run --rm --gpus all \
		-v $$(pwd)/tests/coverage:/workspace/tests/coverage \
		krabby-testing-x86:latest \
		pytest tests/ -v -k "not jetson" --cov=hal --cov=compute --cov-report=html --cov-report=term

