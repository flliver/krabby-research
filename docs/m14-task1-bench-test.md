# M14 Task 1 — Bench Test Walkthrough

## Phase 1 — Trigger a CI build

The CI workflow triggers on pushes to `mainline` or `release/**`.

**Option A — push to mainline** (if m14 is ready to merge):
```bash
git push origin m14:mainline
```

**Option B — cut a release branch** (recommended — exercises `latest_release_branch()` end-to-end):
```bash
git push origin m14:release/0.2.0
```

Watch the `Publish firmware to S3` workflow in GitHub Actions. It should:
1. Compile the sketch with `arduino-cli`
2. Upload `firmware.hex` + `manifest.json` to `release/0.2.0/<build_key>/`
3. Write `release/0.2.0/latest.json` and `index.json`

Verify CI succeeded:
```bash
curl -s https://krabby-firmware-public.s3.amazonaws.com/index.json | python3 -m json.tool
```

---

## Phase 2 — Host setup on the Jetson

Install the package from the repo:
```bash
pip install -e /path/to/krabby-research/firmware
```

Run install (needs sudo for the udev rule):
```bash
sudo krabby-firmware install
```

Expected output:
```
[+]   wrote udev rule: /etc/udev/rules.d/99-krabby-mega.rules
[+]   added <user> to dialout group (re-login to take effect)
[ok]  avrdude already installed
...
Host setup complete. Replug your Mega boards before flashing.
```

Log out and back in (or `newgrp dialout`) so dialout group membership takes effect.

---

## Phase 3 — Show + flash each board

Plug in **one Mega at a time** via USB.

```bash
krabby-firmware show
```

Expected output before flashing new firmware:
```
Attached boards:
  /dev/ttyACM0  primary: dev-local (dev-local dev-local)

Available S3 builds:
  release/0.2.0                   build 20260514-HHMMSS-<sha7>
```

Flash it:
```bash
krabby-firmware update
```

Expected:
```
Branch: release/0.2.0  build 20260514-HHMMSS-<sha7>
Downloading https://krabby-firmware-public.s3.amazonaws.com/...
Saved to ~/.cache/krabby-firmware/release/0.2.0/<sha7>/firmware.hex
Flashing /dev/ttyACM0 ...
Flash complete.
```

Run `--show` again — the board should now report the release version:
```
/dev/ttyACM0  primary: 0.2.0 (release/0.2.0 <sha7>)
```

Repeat for the second and third Megas (unplug/replug each one).

---

## Phase 4 — Final AC check

With all three boards plugged in and flashed, run `--show` and verify `primary|left|right` versions appear. Then verify each acceptance criterion:

| AC | Check |
|----|-------|
| 1 | `curl .../index.json` has `release/0.2.0` branch |
| 2 | CI log shows upload to S3; manifest carries branch + commit |
| 3 | `VER 0.2.0\|0.2.0\|0.2.0 release/...\|...\|... <sha>\|...\|...` on serial |
| 4 | `pip install krabby-firmware` on clean venv; `krabby-firmware show` works |
| 5 | `--show` lists all boards + S3 branches |
| 6 | `--update` (no arg) picks `release/0.2.0`, not `mainline` |
| 7 | `--update mainline` flashes from mainline build |
| 8 | `firmware/SETUP.md` §4 documents bucket layout, V protocol, update procedure |
| 9 | Re-run `sudo krabby-firmware install` — all lines show `[ok]` (idempotent) |
| 10 | `testenv/bin/pytest tests/unit/firmware/ -q` — all 67 pass |
