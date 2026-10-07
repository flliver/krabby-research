# Krabby Research

Locomotion stack for the Krabby hexapod robot — firmware, HAL, policy inference, and deployment tooling.

## Start here

| If you want to… | Read in this order |
| --- | --- |
| Get a robot running on Orin | [Software quick-start](#software-quick-start) → [firmware/SETUP.md](firmware/SETUP.md) |
| Pair a Pro Controller and drive | [CONNECT_PRO_CONTROLLER.md](controller/scripts/jetson/CONNECT_PRO_CONTROLLER.md) (`krabby pair-pro`) → [E2E_GAMEPAD_KRABBY.md](controller/scripts/jetson/E2E_GAMEPAD_KRABBY.md) → [Software quick-start §6](#6-drive-with-a-gamepad) (`krabby run`) |
| Jog joints with the firmware GUI | [Software quick-start](#software-quick-start) + `pip install krabby-firmware`, then `python -m firmware.gui` or `krabby-firmware-gui` (needs `python3-tk` on Orin) |
| Change code or find where things live | [docs/FOLDER_LAYOUT.md](docs/FOLDER_LAYOUT.md) → [DEVELOPER.md](DEVELOPER.md) → [krabby/README.md](krabby/README.md) |
| Stand up fleet / AWS (one-time) | [fleet/ENROLL.md](fleet/ENROLL.md) → [fleet/SETUP-FLEET.md](fleet/SETUP-FLEET.md) → [fleet/FIELD-TELEOP.md](fleet/FIELD-TELEOP.md) |
| Run the continuous bench watchdog | [bench/README.md](bench/README.md) |
| Train or evaluate policies in sim | [crab-hex forward task](parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/README.md) → [docs/crab-hexapod-plant.md](docs/crab-hexapod-plant.md) |

## Kit

| Item | Qty |
|------|-----|
| Jetson Orin (Seeed reComputer J401 or equivalent) | 1 |
| Arduino Mega 2560 | 3 |
| Krabby H-bridge board (BTS7960) | 6 |
| USB hub (powered) | 1 |
| Bench power supply (12 V) | 1 |
| USB cables (Mega → hub) | 3 |
| Nintendo Switch Pro Controller (optional, for manual drive) | 1 |

Full robot assembly notes are in the Milestone 12 deliverables. This repo covers software from bare OS to running locomotion.

---

## Software quick-start

### 1. Install the CLI

On Orin (and other hosts where system Python is not writable), create a user venv first — bare `pip install` into system Python fails with `PermissionError`:

```bash
python3 -m venv ~/.venv-krabby
source ~/.venv-krabby/bin/activate
pip install -U pip
pip install krabby-launcher
```

Keep the venv activated for later `krabby` commands in this guide.

### 2. Pull the locomotion image and set up the host

With the venv still activated, run install via the venv `krabby` — bare `sudo krabby install` often resolves to a different binary and fails with `No such command 'install'`:

```bash
sudo -E env PATH="$PATH" "$(which krabby)" install
```

This pulls `release-latest` from ECR (the stable release channel), writes the udev rule for the Mega 2560 boards, adds you to the `dialout` group, and installs a systemd unit so the stack starts on boot. Replug USB after this step. (Pass `--no-launch-on-startup` to skip the boot autostart; see [krabby/README.md](krabby/README.md#start-on-boot).)

### 3. Verify the boards

```bash
krabby firmware show
```

All three boards should appear with their role (`front`, `left`, `right`) and version.

### 4. Flash all three boards (first time or after a firmware update)

```bash
krabby firmware update
```

Run once per board — replug USB between boards. Boards are auto-detected from `/dev/ttyACM*` and `/dev/ttyUSB*`. See [firmware/SETUP.md](firmware/SETUP.md) for the full three-board procedure.

### 5. Wire the serial harness

Connect all three Megas to the Jetson via the powered USB hub.

### 6. Drive with a gamepad

Pair a Pro Controller over Bluetooth **before** starting the stack
([CONNECT_PRO_CONTROLLER.md](controller/scripts/jetson/CONNECT_PRO_CONTROLLER.md)):

```bash
sudo -E env PATH="$PATH" "$(which krabby)" pair-pro
# If boot autostart is enabled (default after install), stop it so Restart=always
# does not reclaim the container name from your foreground session:
sudo systemctl stop krabby-locomotion
krabby run
```

`krabby run` starts the whole gamepad stack — HAL server, `krabby-uno` client, and controller — in one container, so the paired controller drives the robot immediately. No second command. It clears any existing container named `krabby` first (same as the boot unit’s `ExecStartPre`), so you no longer need a manual `docker rm -f krabby`. The container starts with GPU, serial, and input device passthrough; logs stream to stdout and Ctrl+C stops it. To return to boot autostart afterward: `sudo systemctl start krabby-locomotion`.

To drive from a *separate* client instead (e.g. a second terminal, or another host against a server-only run), use the `krabby-uno` console script from the controller package (`pip install krabby-controller`). See [controller/scripts/jetson/E2E_GAMEPAD_KRABBY.md](controller/scripts/jetson/E2E_GAMEPAD_KRABBY.md) for the full E2E and debug guide.

### 7. Inference mode (optional)

To run a trained policy instead of the gamepad:

```bash
krabby run -- --checkpoint /path/to/checkpoint.pt
```

---

## Continuous bench watchdog

A systemd service polls ECR every 60 s for a new `mainline-latest` digest. When one appears it runs a firmware smoke test and emails (or opens a GitHub Issue) on failure.

Complete Software quick-start §1–2 first, then:

```bash
source ~/.venv-krabby/bin/activate
pip install krabby-bench
sudo BENCH_SMTP_TO=alerts@example.com BENCH_GITHUB_REPO=owner/repo BENCH_GITHUB_TOKEN=ghp_... \
  krabby-bench install
```

See [bench/README.md](bench/README.md) for config reference and forced-failure testing.

---

## Further reading

| Document | Contents |
|----------|----------|
| [firmware/SETUP.md](firmware/SETUP.md) | S3 firmware store, V protocol, three-board update procedure |
| [images/locomotion/README.md](images/locomotion/README.md) | Production Docker image, ECR tags, pin bumping |
| [krabby/README.md](krabby/README.md) | Full `krabby` CLI reference |
| [controller/scripts/jetson/E2E_GAMEPAD_KRABBY.md](controller/scripts/jetson/E2E_GAMEPAD_KRABBY.md) | Gamepad E2E guide |
| [bench/README.md](bench/README.md) | Bench watchdog setup and alerter config |
| [docs/crab-hexapod-plant.md](docs/crab-hexapod-plant.md) | Simulation training plant: generated `assets/crab.usda` (A15+B), plant selection, provenance |
| [docs/crab-hex-forward-policy-config.md](docs/crab-hex-forward-policy-config.md) | Forward-walk policy/MDP configuration, KRABBY_* env vars, phase presets |
| [docs/FOLDER_LAYOUT.md](docs/FOLDER_LAYOUT.md) | Repository map (task package, experiments/ keep-set, policy of record) |
| [parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/README.md](parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/README.md) | Crab-hex forward-walk RL task: phases, training, evaluation |
