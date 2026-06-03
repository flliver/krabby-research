# Jetson bring-up scripts

Scripts for getting a fresh NVIDIA Jetson Orin (Ubuntu 22.04 / jammy, arm64)
ready to run the Krabby locomotion stack. Run them from the repo root, on the
Jetson. See [docs/JETSON_DEPLOYMENT.md](../../docs/JETSON_DEPLOYMENT.md) for the
full guide.

## Quick start

**`bootstrap.sh`** chains the whole bring-up in one idempotent command. Run it
from anywhere in the repo, as your normal user (not with `sudo`):

```bash
./scripts/jetson/bootstrap.sh
```

It runs steps 1–2 below, then installs `python3-pip` + `krabby-launcher` and
runs `sudo krabby install` (udev rules, `dialout` group, boot-autostart unit).
Re-running is safe; each step self-skips when its work is already done.

Flags: `--skip-docker` (Docker already configured), `--no-krabby-install`
(install the launcher package but don't run `krabby install`), and
`--ssm-prefix PREFIX` (SSM path for the watchdog; default `/krabby/bench`). See
`./scripts/jetson/bootstrap.sh --help`.

After bring-up, pull/run the locomotion image (see the guide).

### Error reporting (optional)

To also install the `krabby-bench` watchdog so this device reports smoke-test
failures via SMTP/GitHub, pass its **read-only IAM key** in the environment:

```bash
BENCH_AWS_KEY_ID=AKIA... BENCH_AWS_SECRET_KEY=... ./scripts/jetson/bootstrap.sh
```

Step 6 then runs `pip install krabby-bench` and `krabby-bench install
--ssm-prefix /krabby/bench`. Without those keys the step is skipped. The shared
SMTP/GitHub secrets the watchdog reads are seeded fleet-wide in AWS SSM by an
operator (off-device) — see [bench/README.md](../../bench/README.md) and
`set-ssm-params.sh`; bootstrap does not seed them.

## Bring-up order (individual steps)

`bootstrap.sh` runs these for you; use them on their own to re-run or debug a
single step.

1. **Remove `brltty`**: `sudo apt-get purge -y brltty`. Its udev rules grab
   CH340/Arduino-class USB serial adapters as Braille displays, so the Mega/MCU
   boards vanish from `/dev/ttyACM*` (see Notes).
2. **`install-docker.sh`** — install Docker Engine (section 2 of the guide).
3. **`setup-docker-gpu.sh`** — install the NVIDIA Container Toolkit and wire up
   the `nvidia` Docker runtime for GPU access (section 3). Requires the NVIDIA
   drivers (`nvidia-smi`) to already be present.
4. Install the launcher: `sudo pip3 install krabby-launcher && sudo krabby
   install`, then pull/run the locomotion image.
5. _(Optional)_ Install the error-reporting watchdog: `sudo pip3 install
   krabby-bench && sudo BENCH_AWS_KEY_ID=… BENCH_AWS_SECRET_KEY=… krabby-bench
   install --ssm-prefix /krabby/bench`. See
   [bench/README.md](../../bench/README.md).

## Other scripts

- **`run_jetson_hal_server_host.sh`** — launch the Jetson HAL server from the
  host into the locomotion image (resolves checkpoints, ZED cache, etc.).
- **`jetson-reset.sh`** — reset a Jetson to pristine: uninstall krabby packages,
  remove systemd units, config, udev rules, and Docker images. Leaves SSH keys
  and the user account intact.
- **`pair_pro_controller.sh`** — pair a Nintendo Switch Pro Controller over
  Bluetooth and persist the link key so it auto-reconnects across Bluetooth
  restarts. See [CONNECT_PRO_CONTROLLER.md](../../controller/scripts/jetson/CONNECT_PRO_CONTROLLER.md).

## Notes

- Some Jetson kernels lack the iptables modules Docker's default bridge network
  needs; run containers with `--network host` (the locomotion stack does this).
- `docker` group membership only takes effect on next login — run `newgrp
  docker` or log out/in to use Docker without `sudo`.
- **`brltty`** ships by default on Ubuntu and its udev rules claim
  CH340/Arduino-class USB serial adapters as Braille displays, which makes the
  Mega/MCU boards disappear from `/dev/ttyACM*` shortly after they enumerate.
  `bootstrap.sh` purges it (step 1); to do it by hand: `sudo apt-get purge -y
  brltty`.
