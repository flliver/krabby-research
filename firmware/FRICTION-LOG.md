# Cold-start friction log

Milestone 23

## Task 1 open items (not done yet)

- **1d — Self-hosted runner:** Orin not registered as GitHub Actions runner yet (blocked on repo **admin** registration token) → Appendix C.
- **1a — Front video:** ZED needs USB 3 SuperSpeed cable → Appendix D (teleop control already works).

## Friction entries

| ID  | Stage                | Symptom                                                                                                                                                                                                                                                                                                                                                                          | Workaround                                                                                                                                                                                                                                | Component                                                                                                                               | Cause                                                                                                                                                                                                                                                                                                                                                                                                                    | Blocking                                                              | Cost                                                                 | Effort (h) | Who                                                                                |
| --- | -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------- | -------------------------------------------------------------------- | ---------- | ---------------------------------------------------------------------------------- |
| F1 | package-install | Following the README, `pip install krabby-launcher` on the Orin fails with `PermissionError: Permission denied` because pip tries to install into system Python. | Create a user venv, activate it, then install: `python3 -m venv ~/.venv-krabby && source ~/.venv-krabby/bin/activate && pip install -U pip && pip install krabby-launcher`. For later host setup, keep the venv on PATH: `sudo -E env PATH="$PATH" "$(which krabby)" install`. | README software quick-start; `krabby-launcher` | pip was writing into system Python, which a normal user cannot modify. **Fix landed:** `README.md` Software quick-start (and related bring-up docs) require a user venv before `pip install krabby-launcher` (same pattern as `fleet/ENROLL.md`). | yes | ~5 min; every fresh Orin | 2 | AI docs; Human verify on Orin (done) |
| F2 | package-install | After creating the venv (F1), `sudo krabby install` fails with `No such command 'install'` — sudo is running a different `krabby` binary than the one in the venv. | Compare `which krabby` vs `sudo which krabby`. Run install with the venv binary: `sudo -E env PATH="$PATH" "$(which krabby)" install`. | README `sudo krabby install`; sudo secure_path | sudo’s PATH often runs a different `krabby` than the venv (sometimes the unrelated PyPI project), so `install` is missing. **Fix landed:** document `sudo -E env PATH="$PATH" "$(which krabby)" install` in `README.md` and `krabby/README.md` (and related bring-up docs) wherever bare `sudo krabby` was the venv trap; system-wide bootstrap path still uses plain `sudo krabby install`. | yes | ~10 min; every venv + sudo | 1 | AI docs; Human sudo path on Orin (done) |
| F3 | firmware | With only published packages installed (no git clone), you cannot run the joint GUI or the Pro Controller pairing script — both live in the repo, not on PyPI. Keyboard jog via `python -m firmware` still works. | Install from PyPI: `pip install 'krabby-firmware>=0.2.17' 'krabby-launcher>=0.1.23'`, then `python -m firmware.gui` / `krabby-firmware-gui` and `sudo -E env PATH="$PATH" "$(which krabby)" pair-pro`. Keyboard jog still works via `python -m firmware`. | `krabby-firmware` / `krabby-launcher` packaging | GUI and pairing used to ship only in the git tree. **Fix landed:** GUI in `krabby-firmware` (`firmware.gui` / `krabby-firmware-gui`); pairing in `krabby-launcher` (`krabby pair-pro`). Published `firmware-v0.2.17` / `krabby-v0.1.23`; kit path verified on Orin without a clone. | was yes for GUI/pairing without clone | ~15–30 min finding this; every kit-only bring-up | 2 | AI package/docs; Human PyPI + Orin verify (done) |
| F4 | firmware / jog | With only the FRONT board on USB, `krabby firmware show` lists front/left/right and the GUI can jog FRONT motors, but LEFT/RIGHT H-bridges stay silent. The same LEFT/RIGHT motors move fine when USB is plugged straight into those boards. | Fixed in firmware — reflash all three boards with the patched sketch (`PIN_REV=2` was in use when this was found). No ongoing workaround once reflashed. | `firmware/arduino/arduino.ino` leader `J` forward; GUI `send_command_jog` | The leader forwarded jog as `J <name> <pwm>` (space after `J`); followers treated the name as empty and applied pwm 0. **Fix landed:** host/forward use `J<name> <pwm>`; parser skips leading spaces. Reflash all three boards with the patched sketch. | was yes for GUI jog | ~1–2 h misdiagnosing UART; every 3-board GUI session on unpatched FW | 4 | hardware + Human diagnose; AI assisted fix (done); Human reflash |
| F5 | controller | Pairing the Pro Controller often ends with `Connected: yes` but `Paired: no`, plus a warning that blames `hid_nintendo` even when that module is loaded. Reconnecting from Bluetooth cache (instead of holding Sync) also triggers “Found too quickly.” | Official Pro: power off (Home ~10s) → `bluetoothctl remove` → Sync until 4 LEDs flash → `pair-pro` until `Paired: yes` + `js*`. Third-party / clones: prefer USB or `bluetoothctl` with `pairable on` (see `CONNECT_PRO_CONTROLLER.md` §A/§C). | `krabby pair-pro` / packaged `pair_pro_controller.sh`; `CONNECT_PRO_CONTROLLER.md` | Incomplete bond (cache reconnect ≠ Sync); old script blamed `hid_nintendo` when the module was loaded. **Fix landed:** clearer `pair-pro` warns (`Found too quickly` continues; `Paired: no` says do not chase `hid_nintendo`; third-party → USB/`bluetoothctl`); CONNECT docs cover Sync vs Home, GUI-equivalent CLI (`pairable on`, one cmd at a time), and clones (e.g. XB-324). Verified on Orin: `Paired: no` + `hid_nintendo` still loaded. | was yes for gamepad drive | ~15–30 min; every first BT pair / bad reconnect | 3 | AI warn/docs; Human + HW Orin verify (done) |
| F6 | controller / run | After `krabby install` (boot autostart on), a manual `krabby run` fails because the name `/krabby` is already taken by the locomotion service container. | No ongoing workaround once on a launcher with the clear: `krabby run` removes `/krabby` first. For a stable foreground session while autostart is enabled: `sudo systemctl stop krabby-locomotion` then `krabby run` (unit `Restart=always` would otherwise reclaim the name). | `krabby run` / `gamepad_cmd`; `krabby-locomotion.service` | `krabby install` starts a boot service that already owns the container name `krabby`; plain `krabby run` did not clear it first. **Fix landed:** `krabby run` runs `docker rm -f krabby` before `docker run` (same as the unit’s `ExecStartPre`); boot-vs-manual documented in `README.md` §6 / `krabby/README.md`. Verified on Orin (editable `krabby-launcher`): prior Up container + second-terminal `krabby run` — no Conflict. | was yes for a fresh manual run | ~2–5 min; every boot+manual `krabby run` | 2 | AI docs/code; Human Orin verify (done) |
| F7 | controller / run | Docs say to set `KRABBY_MCU_PORT` when the board is not on `/dev/ttyACM0`, but `KRABBY_MCU_PORT=/dev/ttyUSB1 krabby run` still warns that the MCU is missing at `/dev/ttyACM0`. The HAL may still find the board and drive fine — the warning is misleading. | If logs already show `Connected to /dev/ttyUSB…` or `MCU connected successfully`, ignore the warning and keep going. | `krabby/_docker.py` `gamepad_cmd` vs `firmware_cmd`; `_gamepad_launch_script` | `krabby firmware` forwards `KRABBY_MCU_PORT` into the container; `krabby run` / gamepad launch does not, so the check always looks at `/dev/ttyACM0` even when the HAL later finds the board. **Proposed fix:** forward `KRABBY_MCU_PORT` from `gamepad_cmd` / `_gamepad_launch_script`, and align `E2E_GAMEPAD_KRABBY.md` with that behavior. | no (auto-detect worked); yes if operator trusts the warning and stops | ~5–15 min confusion; every CH340/`ttyUSB`* bring-up | 4 | AI can forward env; Human + Hardware CH340 path |
| F8 | controller / run | `krabby run` prints `error: XDG_RUNTIME_DIR is invalid or not set` before the controller starts. The stack still comes up and the Pro Controller works — the message is noise. | Ignore it — the stack still starts and the Pro Controller still shows up. | locomotion container / SDL-pygame | The locomotion container has no valid `XDG_RUNTIME_DIR`, so SDL prints an error before pygame starts. **Proposed fix:** set a valid `XDG_RUNTIME_DIR` in the locomotion container env (cosmetic; stack already works). | no | ~0–1 min; every `krabby run` | 3 | AI can patch container env; Human in locomotion container |
| F9 | firmware / flash | If a firmware upload is interrupted (Ctrl+C or the port disappears), the board can vanish: `cannot open port`, `krabby firmware show` says “No attached Mega,” and there is no `/dev/ttyACM*` or `ttyUSB*`. It feels bricked even when the sketch is probably fine. | Power-cycle the Mega (reboot the Orin if it stays wedged). Prefer a direct Orin USB port over a hub, use a known-good data cable, and avoid interrupting uploads. Press Mega RESET if an upload hangs. Swap cable/port or try a second Mega to separate USB-serial from a bad sketch. See Appendix B. | `arduino-cli upload` / CH340–ACM enum; powered hub | An interrupted upload plus flaky USB enum on a hub can drop the serial port even when the sketch is fine — it looks bricked. **Proposed fix (documentation only):** write/finish the recover path in Appendix B and link it from flash/update docs (power-cycle Mega ± Orin reboot; prefer direct USB over hub; known-good data cable; do not interrupt upload; Mega RESET if hang; how to tell “vanished port” from a bad sketch). No firmware or tooling code change. | yes until board reappears | ~15–45 min per stuck board; after local PIN_REV=2 flash / hub | 2 | AI docs; Human |
| F10 | firmware / show | `krabby firmware show` sometimes lists only the FRONT board even though LEFT and RIGHT are powered and wired. Unplug and replug the FRONT USB cable, then run show again — the other roles usually appear. | Unplug the FRONT USB cable, wait a few seconds, plug it back in, and re-run `krabby firmware show` until front/left/right all appear. | Leader USB / 3-board role election; `krabby firmware show` | After power or a serial reopen, the leader can miss role election: FRONT shows, LEFT/RIGHT stay invisible until FRONT USB is replugged (port is up — unlike F9). **Proposed fix:** document the FRONT unplug/replug recover step next to `firmware show` / flash docs; optionally add a retry/re-elect hint in `krabby firmware show` if roles are incomplete. | yes for flash/bringup that require 3 roles | ~1–2 min; intermittent after flash, harness, or long session | 3 | hardware + Human diagnose election; AI can draft docs/retry; Human |
| F11 | CI / Actions | Artifact-health’s locomotion-image job warns that Node.js 20 is deprecated because `docker/setup-qemu-action@v3` still targets Node 20 while runners use Node 24. Warning only — not what fails the job by itself. | Safe to ignore until GitHub drops Node 20 (~2026-09), or bump the action now to clear the warning. | `.github/workflows/artifact-health.yml` (`uses: docker/setup-qemu-action@v3`) | `docker/setup-qemu-action@v3` still targets Node 20 while GitHub runners use Node 24. **Proposed fix:** bump to `docker/setup-qemu-action@v4` in artifact-health (and any other `@v3` pins). | no (warn); yes after Node 20 removal if still on v3 | ~5 min when editing workflow | 4 | AI can bump action; Human review CI + confirm workflow green |
| F12 | CI / artifact-health | A manual Artifact health run posts Discord `FAIL` with failed stage **locomotion-image** (packages and firmware checks can still pass). The Node 20 warning (F11) is unrelated — diagnose from the Actions pull/smoke log. | Open the failed **Locomotion image** job in Actions and read **Pull and smoke-start channel tags**. Fix or relax that check once you know whether pull, QEMU run, or help-text matching failed. | `.github/workflows/artifact-health.yml` (`locomotion-image`); ECR `public.ecr.aws/t7t7b3i3/krabby-locomotion:{mainline,release}-latest` | Artifact-health’s locomotion-image stage fails while pulling/smoke-starting channel tags under QEMU (expects `--teleop-ip` in `docker run … --help`). **Proposed fix:** diagnose from the Actions pull/smoke log, then fix the image/tag/QEMU path or relax the check to match real entrypoint behavior; re-run workflow_dispatch until Discord PASS. | yes for green artifact-health | ~15–60 min once log is read | 4 | AI can patch workflow; Human confirm Discord PASS / image publish if broken |
| F13 | controller / run | If the Pro Controller powers off from idle (LEDs out) and you wake it with **Home**, Bluetooth and `/dev/input/js*` may come back, but a running `krabby run` never sees the pad again — sticks do nothing until you restart the locomotion stack. Different from F5 (first-time Sync / pairing). | No ongoing workaround once on a build with reopen: Home-wake (or USB replug); wait ~10s. Only if sticks stay dead: `docker rm -f krabby` then `krabby run`, or `sudo systemctl restart krabby-locomotion`. | gamepad / pygame–SDL joystick open in locomotion / HAL CLI; `controller/input/input_controller.py` | The joystick was opened once at start; after idle power-off + Home wake, BT/`js*` returned but the running HAL never reopened it. **Fix landed:** `InputController` detects disconnect (`attached()` / `CONTROLLERDEVICEREMOVED`), zeros state, and reopens on `CONTROLLERDEVICEADDED` or periodic joystick rescan. CONNECT docs + `verify_joystick_reopen.py`; local tree via `krabby run --mount "$PWD:/opt/krabby-research"`. **Orin HW verified** with third-party Pro: host reopen path (BT disconnect → rescan/Home-wake) and stack path (`krabby run --gamepad-only --mount "$PWD:/opt/krabby-research"` → USB unplug/replug → joints again without restart). Release image without mount still needs a new `krabby-controller` pin + locomotion ECR rebuild. | was yes for drive after idle sleep | ~1–3 min restart; every Pro idle power-off mid-session | 16 | AI reopen/hotplug + unit tests; Human + Hardware Orin verify (done) |




## Working command sequence (bring-up baseline)

Documented “few commands” goal (`README.md` Software quick-start): **~6 steps**
(venv + `pip install` → `sudo -E … krabby install` → `firmware show` → `firmware update` → wire hub → pair + `krabby run`).

**Proven path as of 2026-09-29** (front video still open — Appendix D). Tags:
`[doc]` = in quick-start; `[undoc]` = required but not there; `[manual]` = hands-on;
`[blocked]` = not run yet (later stages).

```text
# --- Orin host (assume imaged, networked, SSHable) ---
python3 -m venv ~/.venv-krabby                          # [doc] F1
source ~/.venv-krabby/bin/activate                      # [doc] F1
pip install -U pip                                      # [doc]
pip install krabby-launcher                             # [doc]

sudo -E env PATH="$PATH" "$(which krabby)" install      # [doc] F2
# (optional) sudo -E env PATH="$PATH" "$(which krabby)" install --no-launch-on-startup
# Replug USB after install.                                 # [doc]

# --- Firmware (three Megas) ---
# Flash one board at a time if first bring-up; then wire all three to powered hub.
krabby firmware show                                    # [doc]
krabby firmware update                                  # [doc] once per board; replug between # [manual]
# Wire FRONT/LEFT/RIGHT Megas → powered hub → Orin          # [doc] [manual]
krabby firmware show                                    # [doc] expect three roles + versions

# Optional GUI jog:
#   pip install krabby-firmware && python -m firmware.gui   # [doc] F3

# --- Pro Controller ---
# Official Pro: Hold Sync until all 4 LEDs flash rapidly           # [doc]/manual] F5
# Third-party: USB or bluetoothctl (pairable on) — CONNECT_PRO_CONTROLLER.md §A/§C
sudo -E env PATH="$PATH" "$(which krabby)" pair-pro     # [doc] F3; F5 warns if Paired: no
# If stale bond: power off (Home ~10s); bluetoothctl remove <MAC>  # [doc] F5
ls /dev/input/js*                                       # [doc] expect js0 (and often js1)

# --- Drive (gamepad stack) ---
# If boot autostart is on: sudo systemctl stop krabby-locomotion   # [doc] F6
source ~/.venv-krabby/bin/activate                      # if new shell
krabby run                                              # [doc] clears /krabby first (F6)
# Ignore: MCU not found at /dev/ttyACM0 (F7) if logs show Connected to /dev/ttyUSB*
# Ignore: XDG_RUNTIME_DIR … (F8)
# Drive: hold RT = FR, left stick Y = FRHL, etc.               # [manual]

# --- Fleet (proven for orin1 — enroll + live portal telemetry) ---
# export AWS_ACCESS_KEY_ID=… AWS_SECRET_ACCESS_KEY=… AWS_DEFAULT_REGION=…   # [undoc] ENROLL.md
# python -c "import boto3; print(boto3.client('sts').get_caller_identity())"
# sudo -E env PATH="$PATH" krabby enroll --thing-name orin1                 # [undoc]
# sudo systemctl start krabby-agent                                         # [undoc]
# source ~/.venv-krabby/bin/activate && krabby get telemetry                # [undoc] needs launcher w/ agent cmds
# Confirm portal fleet.krabbyco.com: orin1 online; detail timestamp advances  # [manual]
# Agent-only health may show locomotion inactive / mcu_missing — OK until HAL up
# Fleet HAL (enrolled): sudo -E env PATH="$PATH" "$(which krabby)" run      # [undoc]
#   (krabby-locomotion.service may be missing until `krabby install` — recreate if needed)
# Portal Open teleop: signaling + control proven; front video → Appendix D
```



### Gap to few-commands goal

**Headline:** goal ≈ **6** commands; working path ≈ **18** steps (**~12** undoc; **~5** manual). Remaining Task 1 gaps → **Task 1 open items**.

### Milestone 23 Items Task 1 list


| ID    | Item                                                | Status                                                                                                  |
| ----- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| 1a    | Portal live telemetry, teleop, cameras, gamepad     | **Partial:** telemetry + gamepad + teleop **control** OK; front video → Task 1 open items / Appendix D  |
| 1b–1c | Friction log + small items                          | **Done** (this file)                                                                                    |
| 1d    | Bench Orin as CI self-hosted runner                 | **Open** — Task 1 open items / Appendix C                                                               |
| 1e    | Harness reset + pip install from public index       | **Done** (manual harness / `bench-reset.sh`)                                                            |
| 1f    | Reflash from **published** S3 artifact              | **Waived on this Orin** — PIN_REV=2 local-only (Appendix B); harness skips S3 overwrite for `dev-local` |
| 1g    | Bringup + motion assert                             | **Done** (manual harness motion PASS)                                                                   |
| 1h    | Discord notify (result, commit, stage, run link)    | **Done** locally; CI injects secret when runner enabled                                                 |
| 1i    | Artifact health (packages / image / firmware)       | **Workflows + Discord notify exist;** green image check / F11–F12 → **Task 2**                          |
| 1j    | Watch `mainline` / `release/**`                     | **Done**                                                                                                |
| 1k    | Repro from notification / harness command in output | **Done** (harness logs stage + command)                                                                 |
| 1l    | Ordered command sequence + gap count                | **Done** (this section)                                                                                 |

---



## Appendix A — Dual-use Orin: cold-start vs bench venvs

**Constraint:** One Orin is used for both cold-start bring-up and the
bench harness (no second machine). Modes must not share a disposable venv.

### Two venvs (plan)


|                                | Cold-start (manual)                                                | Bench (CI / Install stage)                                     |
| ------------------------------ | ------------------------------------------------------------------ | -------------------------------------------------------------- |
| Path                           | `~/.venv-krabby`                                                   | `~/.venv-krabby-bench`                                         |
| Purpose                        | Day-to-day bring-up, Pro Controller, friction logging              | Fresh `pip install krabby-launcher` from PyPI each harness run |
| Created                        | Once during cold-start (F1)                                        | After each `bench-reset.sh`, or first bench Install            |
| Destroyed by `bench-reset.sh`? | **Never**                                                          | **Always** (when present)                                      |
| Typical packages               | `krabby-launcher`, optionally `krabby-bench` for harness inventory | `krabby-launcher` only (public index; no editable `./krabby`)  |
| Activate                       | `source ~/.venv-krabby/bin/activate`                               | `source ~/.venv-krabby-bench/bin/activate`                     |


**Shared (not per-venv):** `~/.config/krabby/state.json` (image ref/digest from
`krabby install`). Light reset **clears this file** so Install re-pulls and
rewrites state. After a bench cycle, cold-start mode may need
`krabby install` again (or restore from a known digest) before `krabby run` /
`krabby firmware` resolve the image via state.

**Also shared:** Docker images, udev, dialout, `hid_nintendo`, BT controller
bond, repo clone, SSH. Light reset leaves these (gap vs bare Orin).

### Modes on this Orin


| Mode     | Venv                   | Autostart                     | Actions runner (when present)     |
| -------- | ---------------------- | ----------------------------- | --------------------------------- |
| Bring-up | `~/.venv-krabby`       | `krabby-locomotion` optional  | Offline                           |
| Bench    | `~/.venv-krabby-bench` | **Disabled** (avoids F6 race) | Online only while running harness |


Do not leave bring-up and a live bench job overlapping on this host.

### Light reset vs full reset


| Script                           | Use                                                                                                                                                                         |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scripts/jetson/bench-reset.sh`  | Between bench runs (default). Stops containers + locomotion unit, clears `state.json`, removes **only** `~/.venv-krabby-bench`. Optional `--rmi` removes locomotion images. |
| `scripts/jetson/jetson-reset.sh` | Rare “closer to bare” wipe (packages, udev, all krabby images). Does **not** know about the two-venv split — use carefully on dual-use.                                     |




### First-time bench Install after reset

```bash
./scripts/jetson/bench-reset.sh
python3 -m venv ~/.venv-krabby-bench
source ~/.venv-krabby-bench/bin/activate
pip install -U pip && pip install krabby-launcher
sudo -E env PATH="$PATH" "$(which krabby)" install --no-launch-on-startup
krabby firmware show
```

Return to cold-start:

```bash
source ~/.venv-krabby/bin/activate
# if state.json was cleared by bench-reset, re-install (boot unit optional):
# sudo -E env PATH="$PATH" "$(which krabby)" install
```



### Gap vs bare Orin (record for bring-up)

Light reset does **not** re-image the Orin, purge Docker, remove udev/`hid_nintendo`,
unpair the Pro Controller, or delete `~/.venv-krabby` / the git clone. Install bugs
that only appear on a truly bare machine can still hide in that gap.

### Four-stage harness CLI

```bash
cd /path/to/krabby-research
source ~/.venv-krabby/bin/activate
pip install -e ./bench
# Full: reset + install + flash + bringup + motion
krabby-bench harness --repo-root "$(pwd)"
# Or continue after a proven Install without wiping again:
krabby-bench harness --skip-install --no-reset --repo-root "$(pwd)"
```

Implementation: `bench/krabby_bench/_harness.py` (`krabby-bench harness`).
Flash reuses `_smoke.py`. Motion jogs RLKL/RRKL/FLKL/FRHL/FRKL and asserts pot/hall
change on the MCU after stopping the container (HAL obs do not expose pot/hall —
KNOWN-ISSUES #6/#7).

---



## Appendix B — Flash / update firmware for PIN_REV=2 (local `dev-local` only)

The **local hardware used for testing** is wired for **pin revision 2**. The
published S3 / `krabby firmware update` artifact is built for **pin revision 3**
and **will not work** on this setup (wrong EN / Hall mapping). That is why
firmware update from S3 is **not** done on this bench — flash from the **repo
tree** with `arduino-cli` and `PIN_REV=2` so the sketch reports `dev-local`.

`PIN_REV=2` is for local Uno v0.1 testing only and will not be used going
forward once boards move to rev 3.

**Harness:** when all attached roles are `dev-local`, the flash stage **skips S3
overwrite**. Motion should use **host** repo `firmware/` parsers (`PYTHONPATH` /
`--repo-root`), not the ECR image’s bundled copy. Until boards move off rev 2,
treat published-firmware flash (Milestone item 1f) as waived on this bench.

Ensure `~/.local/bin` is on `PATH`. Install a real `arduino-cli` (not the snap):

```bash
source ~/.venv-krabby/bin/activate
curl -fsSL https://github.com/arduino/arduino-cli/releases/download/v1.1.1/arduino-cli_1.1.1_Linux_ARM64.tar.gz \
  | tar -xz -C ~/.local/bin arduino-cli

# confirm it's the new one, not snap
which arduino-cli   # should be ~/.local/bin/arduino-cli
arduino-cli version

make -C firmware upload-firmware PIN_REV=2
```

Flash **one Mega at a time** (replug USB between boards) if the hub/leader path
is not used for upload. Pass `PORT=` explicitly when the default is wrong
(e.g. `PORT=/dev/ttyUSB0`). **Do not Ctrl+C mid-upload** (F9). If the board
vanishes (`error -110` / no tty): power-cycle, try a direct Orin port, then
re-run upload once `/dev/ttyUSB*` or `/dev/ttyACM*` is back.

After all three are updated:

```bash
krabby firmware show
# expect front/left/right … dev-local
```

Then continue harness motion with `--skip-install --no-reset --repo-root` (Appendix A).

---



## Appendix C — Self-hosted Orin runner (bench CI)

**Status:** `~/actions-runner` (linux-**arm64**) unpacked on the Orin; blocked on a
fresh GitHub **admin** registration token (stale → `404`). Then `config.sh` +
`svc` + `BENCH_RUNNER_ENABLED=true`. Wrong arch (`linux-x64`) → `Exec format error`.

Full procedure (admin + operator + failures):
`[bench/SELF-HOSTED-RUNNER.md](../bench/SELF-HOSTED-RUNNER.md)`.

---



## Appendix D — ZED 2i USB 3 cable (front teleop video)

Teleop **signaling and control** on `orin1` work. Front **video** does not until the
ZED is on a real USB 3 path.

**Symptom:** `lsusb` shows Stereolabs ZED 2i, but HAL logs
`Failed to create ZED camera: CAMERA STREAM FAILED TO START`. Portal Open teleop
sends black frames for `front_rgbd`. `lsusb -t` may show ZED video under a **480M**
(USB 2) hub chain (e.g. Bus 01), not Bus 02/04 SuperSpeed. A wrong USB-C cable
can fail to enumerate at all.

**Resolution:** use a Stereolabs (or known-good) **USB 3 SuperSpeed** cable; plug so
`lsusb -t` shows ZED video at **5000M**. Then restart fleet HAL (`krabby run`) and
confirm `ZED camera initialized` / `front_rgbd ready`, then portal Open teleop with
real front video. Side MaixSense cameras are optional for 1a front video.

**Check before chasing HAL bugs:**

```bash
lsusb | grep -i stereolabs
lsusb -t   # ZED video line must be 5000M, not 480M
```

