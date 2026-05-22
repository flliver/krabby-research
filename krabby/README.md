# krabby-launcher

CLI for installing, updating, and running the Krabby locomotion stack on a Jetson Orin host.

PyPI package name is `krabby-launcher` (the `krabby` name was already taken); the installed command is still `krabby`.

## Install

```bash
pip install krabby-launcher
```

## Usage

```
krabby install            # pull mainline-latest, set up udev + dialout
krabby install --image <ref>   # pull a specific tag or digest

krabby update             # re-pull the last installed image
krabby update --image <ref>    # pull a different tag

krabby run                # start the locomotion container
krabby run --image <ref> -- --checkpoint /path/to/ckpt.pt

krabby firmware show      # run krabby-firmware show inside the container
krabby firmware update    # run krabby-firmware update inside the container
krabby firmware <args>    # any krabby-firmware subcommand/flags

krabby --version
krabby --help
```

## Image refs

The default image is pulled from ECR:

```
public.ecr.aws/t7t7b3i3/krabby-locomotion:mainline-latest
```

A bare tag (e.g. `--image v1.2.3`) is expanded to the full ECR URI automatically.
Pass a fully-qualified URI to use a different registry entirely.

## State

The last installed image ref and digest are recorded at `~/.config/krabby/state.json`.
`krabby update` and `krabby run` read this file when `--image` is omitted.

## GPU

On `aarch64` (Jetson) the container is started with `--runtime=nvidia`.
On `x86_64` it uses `--gpus all`.

## Firmware pass-through

`krabby firmware` mounts `~/.cache/krabby-firmware` into the container so cached
firmware artifacts are shared across runs.  Serial devices (`/dev/ttyACM*`,
`/dev/ttyUSB*`) are passed through automatically via `--device`.
