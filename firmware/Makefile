# Krabby Arduino Mega 2560 firmware — compile and upload via arduino-cli.
#
# Usage (from krabby-research):
#   make -C firmware upload-firmware
#   make -C firmware upload-firmware PORT=COM5
#   make -C firmware upload-firmware PIN_REV=1
#   make -C firmware compile-firmware          # compile only, no upload
#
# To push the host-side Python (the menu / SDK / CLI / TCP bridge in this dir) to
# a bench host over ssh, e.g. for `python -m firmware.gui --remote krabby-orin`:
#   make -C firmware sync-remote REMOTE=krabby-orin
# __pycache__ is excluded so host .pyc files don't litter the Jetson.
#
# Prerequisites: arduino-cli on PATH, pyserial for auto-detect (pip install pyserial).
# Windows: winget install GnuWin32.Make, then add GnuWin32\bin to PATH.

ARDUINO_CLI ?= arduino-cli
FQBN        ?= arduino:avr:mega
PIN_REV     ?= 3
SKETCH      := $(CURDIR)/arduino

# sync-remote: where the host-side Python lands on REMOTE. Must match
# DEFAULT_REMOTE_DIR in firmware/gui/remote.py (the --remote bridge launcher).
# NOTE: no trailing inline comments on these — make would fold the padding
# spaces into the value and corrupt the ssh target / path.
REMOTE      ?=
RSYNC       ?= rsync
REMOTE_DIR  ?= ~/krabby-fw-bench/firmware/

ifeq ($(OS),Windows_NT)
PYTHON ?= python
else
PYTHON ?= python3
endif

# Always bake in the 256-byte serial RX buffer. The leader forwards ~200-byte
# follower telemetry lines from Serial1/Serial2; the AVR core's 64-byte default
# RX buffer drops the middle of those lines and the host logs "can't keep up".
# CI (.github/workflows/publish-firmware.yml) and firmware/install.py's
# platform.local.txt already pass this define; passing it here too makes
# `make compile/upload-firmware` produce an identical binary regardless of
# whether install.py ran, the host OS, or which AVR core version is installed.
# KRABBY_PIN_REV is passed explicitly as well (board_pins.h defaults to 3 when
# undefined, so -DKRABBY_PIN_REV=3 is a harmless no-op for the default build).
BUILD_PROPS := --build-property "compiler.cpp.extra_flags=-DSERIAL_RX_BUFFER_SIZE=256 -DKRABBY_PIN_REV=$(PIN_REV)" \
               --build-property "compiler.c.extra_flags=-DSERIAL_RX_BUFFER_SIZE=256 -DKRABBY_PIN_REV=$(PIN_REV)"

# Auto-detect serial port for upload-firmware (only when PORT not provided).
ifneq ($(filter upload-firmware,$(MAKECMDGOALS)),)
ifeq ($(strip $(PORT)),)
ifeq ($(OS),Windows_NT)
PORT := $(shell $(PYTHON) "$(CURDIR)/scripts/detect_mcu_port.py" 2>NUL)
else
PORT := $(shell $(PYTHON) "$(CURDIR)/scripts/detect_mcu_port.py" 2>/dev/null)
endif
endif
ifeq ($(strip $(PORT)),)
$(error No serial port found. Pass PORT=COMx or install pyserial for auto-detect)
endif
$(info Using PORT=$(PORT))
endif

.PHONY: arduino-core compile-firmware upload-firmware sync-remote

arduino-core:
	$(ARDUINO_CLI) core install arduino:avr

compile-firmware: arduino-core
	$(ARDUINO_CLI) compile --fqbn $(FQBN) $(BUILD_PROPS) $(SKETCH)

upload-firmware: compile-firmware
	$(ARDUINO_CLI) upload -p "$(PORT)" --fqbn $(FQBN) $(SKETCH)

# Push this firmware/ tree (host-side Python: the interactive menu, SDK, CLI,
# serial/TCP bridge) to REMOTE for bench use, without touching any board's
# flash. Excludes __pycache__ so host-compiled .pyc files don't litter the
# Jetson (its Python recompiles its own).
sync-remote:
	@test -n "$(REMOTE)" || { echo "ERROR: set REMOTE=user@host (the bench machine, e.g. krabby-orin)"; exit 1; }
	$(RSYNC) -az --exclude='__pycache__' "$(CURDIR)/" "$(REMOTE):$(REMOTE_DIR)"
