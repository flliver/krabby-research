# krabby-launcher

CLI for installing, updating, and running the Krabby locomotion stack on a Jetson Orin host.

PyPI package name is `krabby-launcher` (the `krabby` name was already taken); the installed command is still `krabby`.

## Install

```bash
pip install krabby-launcher
```

## Usage

```
krabby install            # pull release-latest, set up udev + dialout, enable boot autostart
krabby install --image <ref>          # pull a specific tag or digest
krabby install --no-launch-on-startup # set up the host but DON'T start on boot

krabby update             # re-pull the last installed image
krabby update --image <ref>    # pull a different tag

krabby run                # full gamepad stack: HAL server + krabby-uno client + controller
krabby run --gamepad-only      # explicit alias for the gamepad stack
krabby run -- --device-id 1    # forward client args to krabby-uno (gamepad mode)
krabby run -- --checkpoint /path/to/ckpt.pt   # inference mode (policy checkpoint)

krabby firmware show           # run krabby-firmware show inside the container
krabby firmware show <branch>  # list a branch's builds newest-first, paged
krabby firmware update         # run krabby-firmware update inside the container
krabby firmware <args>         # any krabby-firmware subcommand/flags

krabby --version
krabby --help
```

`krabby run` starts the **whole** gamepad stack in one container — the HAL server and
the `krabby-uno` client/controller together — so a paired gamepad drives the robot with
a single command. To run just the client (e.g. the two-process debug flow), use the
`krabby-uno` console script from the controller package (`pip install krabby-controller`)
against the server's TCP endpoints (`tcp://host:6001` / `:6002`).

## Start on boot

`krabby install` installs a systemd unit (`krabby-locomotion.service`) that runs
`krabby run` on boot — **enabled by default**. It runs as the invoking user, starts
after `docker.service`, and retries if the MCU hasn't enumerated yet at boot.

```bash
sudo krabby install                       # enables boot autostart (default)
sudo krabby install --no-launch-on-startup  # host setup only; no autostart
sudo systemctl disable krabby-locomotion    # turn off later
journalctl -u krabby-locomotion -f          # service logs
```

The unit starts the **gamepad** stack. ⚠️ On boot the stack goes live; if a gamepad
is paired/connected it can drive the robot unattended. To run a policy instead, edit
`ExecStart` in the unit to `krabby run -- --checkpoint /path/to/ckpt.pt`.

## Image refs

The default image is pulled from ECR:

```
public.ecr.aws/t7t7b3i3/krabby-locomotion:release-latest
```

`release-latest` tracks the newest `release/*` build — the stable channel for kit
owners (CI moves the tag on each release build; see `images/locomotion/README.md`).
Use `--image mainline-latest` to follow the development line instead.

A bare tag (e.g. `--image v1.2.3`) is expanded to the full ECR URI automatically.
Pass a fully-qualified URI to use a different registry entirely.

## State

The last installed image ref and digest are recorded at `~/.config/krabby/state.json`.
`krabby update` and `krabby run` read this file when `--image` is omitted.

## GPU

On `aarch64` (Jetson) the container is started with `--runtime=nvidia`.
On `x86_64` it uses `--gpus all`.

## Firmware pass-through

There are two related but distinct commands — same flash CLI, different host requirements:

| Command | Where it runs | When to use |
|---------|---------------|-------------|
| `krabby firmware <args>` | the flash CLI **inside the installed image** | the normal path — no host flash tools needed (you only `pip install krabby-launcher`) |
| `krabby-firmware <args>` | the flash CLI **directly on the host** | running the firmware package standalone (`pip install krabby-firmware`), e.g. on a laptop without the locomotion image |

`krabby firmware` forwards every argument verbatim to `krabby-firmware` in a transient
container, mounts `~/.cache/krabby-firmware` so cached artifacts are shared across runs,
and passes serial devices (`/dev/ttyACM*`, `/dev/ttyUSB*`) through via `--device`. When
your shell is interactive it allocates a TTY, so the interactive menu and paged
`show <branch>` output behave the same as running `krabby-firmware` directly.
