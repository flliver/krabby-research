# Krabby Arduino Mega 2560 firmware — compile and upload via arduino-cli.
#
# Usage (from krabby-research):
#   make -C firmware upload-firmware
#   make -C firmware upload-firmware PORT=COM5
#   make -C firmware upload-firmware PIN_REV=1
#   make -C firmware compile-firmware          # compile only, no upload
#
# Remote flashing (boards on another host you reach over SSH, e.g. a Jetson Orin
# with the USB hub): compile here, copy the .hex over, run avrdude on the remote.
#   make -C firmware flash-remote REMOTE=orin1 PORT=/dev/ttyUSB0
# PORT is the serial device *on REMOTE*; the remote needs avrdude installed
# (apt install avrdude) and permission on the port (dialout group).
#
# To flash every board on REMOTE in one shot (compile + copy once, then loop):
#   make -C firmware flash-remote-all REMOTE=orin1
#   make -C firmware flash-remote-all REMOTE=orin1 PORTS="/dev/ttyUSB0 /dev/ttyUSB2"
# With no PORTS, the boards are discovered on REMOTE by USB VID:PID (the same
# ID set firmware/cli.py's _MEGA_USB_IDS uses), not a bare /dev/ttyUSB* glob, so
# unrelated USB-serial devices are skipped. The same hex goes on every board —
# roles live in EEPROM, not the firmware.
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

# Remote flashing (flash-remote / flash-remote-all): compile here, flash on
# REMOTE via avrdude. The avrdude args live in one variable so the two flash
# recipes can't drift; -c wiring talks to the stock Arduino Mega bootloader,
# same as cmd_update's _flash() in firmware/cli.py.
AVRDUDE     ?= avrdude
AVRDUDE_FLASH = $(AVRDUDE) -p m2560 -c wiring -b 115200 -D
SSH         ?= ssh
SCP         ?= scp
BUILD_DIR   := $(CURDIR)/build
HEX         := $(BUILD_DIR)/arduino.ino.hex
REMOTE_HEX  ?= /tmp/krabby-firmware.hex

# flash-remote-all: explicit board list overrides discovery; otherwise the boards
# are found on REMOTE by USB VID:PID via this self-contained detector (run on
# REMOTE with the script piped over stdin — nothing to scp or leave behind).
PORTS       ?=
LISTER      := $(CURDIR)/scripts/list_mcu_ports.py

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

.PHONY: arduino-core compile-firmware upload-firmware flash-remote flash-remote-all sync-remote

arduino-core:
	$(ARDUINO_CLI) core install arduino:avr

# Single compile recipe: always export the binaries to BUILD_DIR so every upload
# path (local upload-firmware, remote flash-remote) flashes the same artifact.
compile-firmware: arduino-core
	$(ARDUINO_CLI) compile --fqbn $(FQBN) $(BUILD_PROPS) --output-dir "$(BUILD_DIR)" $(SKETCH)

upload-firmware: compile-firmware
	$(ARDUINO_CLI) upload -p "$(PORT)" --fqbn $(FQBN) --input-dir "$(BUILD_DIR)" $(SKETCH)

# Flash that same locally-built image onto a board attached to REMOTE over SSH (no
# S3): copy the .hex over and run avrdude there. avrdude -c wiring talks to the stock
# Arduino bootloader, same as cmd_update's _flash().
flash-remote: compile-firmware
	@test -n "$(REMOTE)" || { echo "ERROR: set REMOTE=user@host (the machine the boards are plugged into)"; exit 1; }
	@test -n "$(strip $(PORT))" || { echo "ERROR: set PORT=/dev/ttyACMx (the board's serial device on $(REMOTE))"; exit 1; }
	$(SCP) "$(HEX)" "$(REMOTE):$(REMOTE_HEX)"
	$(SSH) "$(REMOTE)" "$(AVRDUDE_FLASH) -P '$(PORT)' -U flash:w:$(REMOTE_HEX):i"

# Flash every board on REMOTE: compile + copy the hex once, discover the boards
# by USB VID:PID on REMOTE (unless PORTS is given), then flash each in turn. The
# same hex goes on all of them — the board's role lives in EEPROM, not firmware.
#
# Assumes you can ssh into the Jetson REMOTE, and that all shields are
# connected to that Jetson via a USB hub. Primarily used in bench testing.
flash-remote-all: compile-firmware
	@test -n "$(REMOTE)" || { echo "ERROR: set REMOTE=user@host (the machine the boards are plugged into)"; exit 1; }
	$(SCP) "$(HEX)" "$(REMOTE):$(REMOTE_HEX)"
	@ports="$(PORTS)"; \
	if [ -z "$$ports" ]; then \
		ports=`$(SSH) "$(REMOTE)" "$(PYTHON) -" < "$(LISTER)"`; \
	fi; \
	if [ -z "$$ports" ]; then echo "ERROR: no boards found on $(REMOTE) (pass PORTS=\"/dev/ttyUSB0 ...\" to override)"; exit 1; fi; \
	echo "Flashing boards on $(REMOTE): $$ports"; \
	for p in $$ports; do \
		echo "==> $$p"; \
		$(SSH) "$(REMOTE)" "$(AVRDUDE_FLASH) -P '$$p' -U flash:w:$(REMOTE_HEX):i" || exit 1; \
	done

# Push this firmware/ tree (host-side Python: the interactive menu, SDK, CLI,
# serial/TCP bridge) to REMOTE for bench use, without touching any board's
# flash. Excludes __pycache__ so host-compiled .pyc files don't litter the
# Jetson (its Python recompiles its own).
sync-remote:
	@test -n "$(REMOTE)" || { echo "ERROR: set REMOTE=user@host (the bench machine, e.g. krabby-orin)"; exit 1; }
	$(RSYNC) -az --exclude='__pycache__' "$(CURDIR)/" "$(REMOTE):$(REMOTE_DIR)"
