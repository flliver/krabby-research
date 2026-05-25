# Krabby Research

Locomotion stack for the Krabby hexapod robot — firmware, HAL, policy inference, and deployment tooling.

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

```bash
pip install krabby-launcher
```

### 2. Pull the locomotion image and set up the host

```bash
sudo krabby install
```

This pulls `mainline-latest` from ECR, writes the udev rule for the Mega 2560 boards, and adds you to the `dialout` group. Replug USB after this step.

### 3. Verify the boards

```bash
krabby firmware show
```

All three boards should appear with their role (`primary`, `left`, `right`) and version.

### 4. Flash all three boards (first time or after a firmware update)

```bash
krabby firmware update
```

Run once per board — replug USB between boards. Boards are auto-detected from `/dev/ttyACM*` and `/dev/ttyUSB*`. See [firmware/SETUP.md](firmware/SETUP.md) for the full three-board procedure.

### 5. Wire the serial harness

Connect all three Megas to the Jetson via the powered USB hub.

### 6. Start the locomotion stack

```bash
krabby run
```

The container starts with GPU, serial, and input device passthrough. Logs stream to stdout; Ctrl+C stops it.

### 7. Drive with a gamepad (optional)

Pair a Pro Controller over Bluetooth ([CONNECT_PRO_CONTROLLER.md](controller/scripts/jetson/CONNECT_PRO_CONTROLLER.md)), then from a second terminal:

```bash
krabby uno
```

See [controller/scripts/jetson/E2E_GAMEPAD_KRABBY.md](controller/scripts/jetson/E2E_GAMEPAD_KRABBY.md) for the full E2E guide.

---

## Continuous bench watchdog

A systemd service polls ECR every 60 s for a new `mainline-latest` digest. When one appears it runs a firmware smoke test and emails (or opens a GitHub Issue) on failure.

```bash
sudo pip3 install krabby-bench
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
