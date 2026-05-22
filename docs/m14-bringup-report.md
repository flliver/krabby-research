# M14 Bring-Up Report — Jetson Orin Bench

**Date:** 2026-05-22  
**Hardware:** Seeed reComputer J401 (Jetson Orin), 3× Arduino Mega 2560, 6× Krabby H-bridge boards, powered USB hub, bench PSU  
**Branch:** `m14`  
**Image digest tested:** `sha256:c25f47e2ad673d9b2e41079762b3a9739be467790434a3306a6a3de4f8fce5bf` (`m14-latest`)

---

## 1. Reset to pristine

Ran `scripts/jetson-reset.sh` as root to uninstall all krabby packages, remove systemd units, config, udev rules, and Docker images. SSH keys and the `krabby` user account were left intact.

---

## 2. Task 1 — Firmware updater package

```
pip install krabby-firmware
sudo krabby-firmware install          # wrote udev rule, added krabby to dialout
krabby-firmware show                  # confirmed 3 boards: primary/left/right @ 0.2.9
krabby-firmware update release/0.2.9  # flashed all 3 boards per-port
krabby-firmware show                  # all 3 confirmed same version post-flash
```

All three Megas reported `0.2.9 (release/0.2.9 05e5e3e)`. Version matches the S3 manifest for `release/0.2.9`.

---

## 3. Task 2 — Locomotion image

ECR Public image `public.ecr.aws/t7t7b3i3/krabby-locomotion:m14-latest` built by CI on push to `m14`. Tags present: `sha256:c25f47e2a`, `m14-latest`. Post-build CI verification confirmed:

- `pip freeze` for all `krabby-*` packages matches `requirements.release.txt`
- `avrdude`, `arduino-cli`, `krabby-firmware` present in image
- `arduino:avr` core installed

```
docker run --rm --device /dev/ttyACM0 ... --entrypoint krabby-firmware \
  public.ecr.aws/t7t7b3i3/krabby-locomotion:m14-latest show
# → primary/left/right all at 0.2.9
```

---

## 4. Task 3 — krabby CLI

```
sudo pip3 install krabby-launcher
sudo krabby install                   # pulled m14-latest, wrote udev rule, added to dialout
krabby firmware show                  # 3 boards via container, no host krabby-firmware
krabby run --entrypoint nvidia-smi    # GPU visible (NVIDIA-SMI 540.4.0, CUDA 12.6)
krabby run --entrypoint ls -- /dev/input/  # /dev/input/* visible (gamepad passthrough)
```

Pro Controller E2E (paired per `CONNECT_PRO_CONTROLLER.md`): controller drove joints through `krabby run` container. Verified previously on this hardware.

---

## 5. Task 4 — Bench watchdog

### Install

```
sudo pip3 install krabby-bench
sudo BENCH_SMTP_HOST=smtppro.zoho.com BENCH_SMTP_PORT=465 \
     BENCH_SMTP_USER=... BENCH_SMTP_PASSWORD=... \
     BENCH_SMTP_FROM=krabby-errors@anchornorth.tech \
     BENCH_SMTP_TO=krabby-errors@anchornorth.tech \
     BENCH_GITHUB_REPO=jchilders/krabby-research \
     BENCH_GITHUB_TOKEN=... \
     krabby-bench install --ecr-tag m14-latest
```

Service enabled and started: `systemctl status krabby-bench` → `active (running)`.

### Normal-path pass

```
2026-05-22 15:32:53 INFO  krabby-bench watchdog starting (interval=60s)
2026-05-22 15:32:54 INFO  New digest sha256:c25f47e2a — running update + smoke
2026-05-22 15:33:48 INFO  Smoke passed: sha256:c25f47e2a all boards ['0.2.9', '0.2.9', '0.2.9']
```

Smoke completed in ~54 s. Subsequent polls of the same digest: silent (dedup working).

### Forced-failure alert

One Mega unplugged. State file cleared and service restarted to simulate new-digest trigger:

```
2026-05-22 15:38:36 INFO     krabby-bench watchdog starting (interval=60s)
2026-05-22 15:38:36 INFO     New digest sha256:c25f47e2a — running update + smoke
2026-05-22 15:38:49 WARNING  Smoke failed: sha256:c25f47e2a / firmware_show_ports — expected 3 ports, got 2
2026-05-22 15:38:50 INFO     Alert email sent to krabby-errors@anchornorth.tech
2026-05-22 15:38:51 INFO     GitHub issue opened in jchilders/krabby-research
2026-05-22 15:38:51 WARNING  Alert sent for sha256:c25f47e2a / firmware_show_ports
```

Exactly one email delivered and one GitHub issue opened. Repeat polls of the same (digest, step) pair: no re-alarm (dedup window = 1 h).

---

## 6. Gaps surfaced and fixed

| Gap | Fix | Commit |
|-----|-----|--------|
| `sudo krabby-bench install` failed with `ModuleNotFoundError` — user-installed package shadowed system install under sudo | Changed `pip install` → `sudo pip3 install` in both READMEs | `0789b62` |
| `krabby-bench` PyPI 0.1.0 was pre-argparse (no `install` subcommand) | Bumped to 0.1.1, published via `bench-v0.1.1` tag | `5221a07` |
| `_VERSION` in `krabby/__main__.py` did not match `pyproject.toml` | Fixed to `0.1.2`, published via `krabby-v0.1.2` tag | `5221a07` |

---

## 7. Deferred items

- **AC4 Task 4 (browser teleop):** Blocked on M10 fleet portal, which is expected to complete after M14.
- **`krabby firmware update mainline`:** `mainline-latest` tag will exist in ECR once `m14` merges to `main`.
