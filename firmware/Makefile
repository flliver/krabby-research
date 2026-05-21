# Krabby Arduino Mega 2560 firmware — compile and upload via arduino-cli.
#
# Usage (from krabby-research):
#   make -C firmware upload-firmware
#   make -C firmware upload-firmware PORT=COM5
#   make -C firmware upload-firmware PIN_REV=1
#   make -C firmware compile-firmware          # compile only, no upload
#
# Prerequisites: arduino-cli on PATH, pyserial for auto-detect (pip install pyserial).
# Windows: winget install GnuWin32.Make, then add GnuWin32\bin to PATH.

ARDUINO_CLI ?= arduino-cli
FQBN        ?= arduino:avr:mega
PIN_REV     ?= 2
SKETCH      := $(CURDIR)/arduino

ifeq ($(OS),Windows_NT)
PYTHON ?= python
else
PYTHON ?= python3
endif

ifeq ($(PIN_REV),2)
BUILD_PROPS :=
else
BUILD_PROPS := --build-property compiler.cpp.extra_flags=-DKRABBY_PIN_REV=$(PIN_REV) \
               --build-property compiler.c.extra_flags=-DKRABBY_PIN_REV=$(PIN_REV)
endif

# Auto-detect serial port for upload-firmware (only when PORT not provided).
ifneq ($(filter upload-firmware,$(MAKECMDGOALS)),)
ifeq ($(strip $(PORT)),)
PORT := $(shell $(PYTHON) "$(CURDIR)/scripts/detect_mcu_port.py" 2>NUL)
endif
ifeq ($(strip $(PORT)),)
$(error No serial port found. Pass PORT=COMx or install pyserial for auto-detect)
endif
$(info Using PORT=$(PORT))
endif

.PHONY: arduino-core compile-firmware upload-firmware

arduino-core:
	$(ARDUINO_CLI) core install arduino:avr

compile-firmware: arduino-core
	$(ARDUINO_CLI) compile --fqbn $(FQBN) $(BUILD_PROPS) $(SKETCH)

upload-firmware: compile-firmware
	$(ARDUINO_CLI) upload -p "$(PORT)" --fqbn $(FQBN) $(SKETCH)
