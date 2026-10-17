# Fleet registration and teleop check (simulated krab, firmware in the loop)

Confirm that a **simulated krab** running the **isaacsim** image with **`--backend fullsim`**
**registers with the fleet manager**, **appears in the UI**, **connects**, and is
**teleoperable** with the **firmware in the loop**; **stop** the sim and assert it **shows
offline** for teleop (see pass/fail below). HAL command-path integration tests are separate
(`tests/integration/firmware/test_hal_fullsim_teleop_e2e.py`, etc.).

On a GPU host you run **two processes yourself**: **`krabby agent`** (fleet MQTT + teleop
shim) and Isaac HAL (`./scripts/run_isaac_hal_server.sh --hexapod --teleop-ip 127.0.0.1`).
That is **not** the enrolled production path **`krabby run`** → Jetson locomotion container.

See also: [`FIELD-TELEOP.md`](FIELD-TELEOP.md), [`ENROLL.md`](ENROLL.md), [`SETUP-FLEET.md`](SETUP-FLEET.md),
[`hal/server/isaac/README.md`](../hal/server/isaac/README.md).

---

## Path under test

Same fleet teleop signaling as a field robot; locomotion is Isaac HAL (**fullsim**) instead
of Jetson HAL inside **`krabby-locomotion`**.

```text
Operator browser → fleet portal / service (Cognito)
  → MQTT teleop/{thingName}/signaling/in|out
  → krabby agent (ws://127.0.0.1:9000/ws/robot)
  → Isaac HAL (--teleop-ip 127.0.0.1, --backend fullsim)
  → WebRTC (ICE / coturn)
```

Do **not** run **`krabby-locomotion`** on the same host while Isaac is up.
**`--hexapod`** on `run_isaac_hal_server.sh` selects **`--backend fullsim`**
(`images/isaacsim/run_hal_server.sh`).

---

## Bring the simulated krab up

### Prerequisites

- Deployed **fleet manager** (portal + service): [`SETUP-FLEET.md`](SETUP-FLEET.md)
- IoT **thing** name: either a **long-lived** sim lab device you keep enrolled, or a
  **throwaway** name you will **decommission** after the check
- **`krabby-launcher` ≥ 0.1.22** ([`BENCH-TELEOP.md`](BENCH-TELEOP.md))
- Local **isaacsim** image: **`make build-isaacsim-image`** (embeds **`krabby-hal-server-isaac`**
  **0.1.2** wheels; same release is on [PyPI](https://pypi.org/project/krabby-hal-server-isaac/0.1.2/))

### Step 1 — Register with the fleet manager

Enroll needs **`krabby-enroll` IAM keys** in the shell first ([`ENROLL.md`](ENROLL.md)); enroll
does not write AWS keys to disk. **`sudo -E`** below keeps these exports visible to `krabby
enroll`.

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=<region>
python -c "import boto3; print(boto3.client('sts').get_caller_identity())"
```

Expect the caller ARN to end with `user/krabby-enroll`. Then:

```bash
sudo -E env PATH="$PATH" krabby enroll --thing-name <sim-thing-name> \
  --locomotion-control-source portal \
  --locomotion-robot hex

sudo systemctl stop krabby-locomotion.service
sudo systemctl disable krabby-locomotion.service
systemctl is-active krabby-locomotion.service    # expect: inactive
systemctl is-enabled krabby-locomotion.service   # expect: disabled

sudo systemctl enable --now krabby-agent.service
ss -tlnp | grep 9000
```

`krabby enroll` enables **`krabby-locomotion.service`** (Jetson HAL via **`krabby run`**). Stop
and disable it before Isaac fullsim so two HAL processes do not run on the same host.

### Step 2 — Appears in the UI

From an operator workstation, `krabby-fleet list` or portal **`/devices`**: thing listed,
shadow **`timestamp`** updating.

### Step 3 — Start the isaacsim image (fullsim)

On the sim host (repo clone optional if the image is current):

```bash
./scripts/run_isaac_hal_server.sh --hexapod --teleop-ip 127.0.0.1
```

Run **without** `--headless` on the sim host (local `DISPLAY` / `xhost +local:docker` as in
`scripts/run_isaac_hal_server.sh`). `--teleop-ip` enables sim RGB-D cameras (`enable_cameras`);
headless + RTX/Hydra has segfaulted on some stacks (RTX 5080, swap exhausted). Headless is fine
for HAL+fullsim **without** `--teleop-ip` (joystick/TCP smoke only).

Uses wheels baked into **`krabby-isaacsim:latest`** (rebuild after HAL releases:
`make build-isaacsim-image`). For unreleased HAL edits only, set
**`KRABBY_ISAAC_DEV_MOUNTS=1`** when running the script from a **krabby-research** clone.

If `main.py: error: unrecognized arguments: --backend fullsim` or **`--teleop-ip`**, rebuild
the image — it predates **0.1.2** — or use **`KRABBY_ISAAC_DEV_MOUNTS=1`** from a current clone.

If the entrypoint adds **`--joystick`**, use portal **operator override** so commands go
through teleop HAL → **`apply_command`** → **firmware in the loop**, not only TCP
`krabby-uno-sim`.

### Step 4 — Connects and teleoperates

Portal **Open teleop** for `<sim-thing-name>`; viewer **Playing**; **operator override** on;
leg command produces sim motion. On the sim host, `krabby get telemetry` should report
`teleop_edge_connected` true ([`FIELD-TELEOP.md`](FIELD-TELEOP.md)).

> **Sim-only host:** with no USB MCU, `health.mcu_present` false and red flag `mcu_missing`
> are expected; not a failure for this check.

---

## Tear down the simulated krab

Undo **Bring the simulated krab up** in reverse order (**4 → 3 → 2 → 1**), then undo
**Prerequisites**. The first step satisfies **shows offline when stopped** in pass/fail.

### Step 4 — Connects and teleoperates (reverse bring-up step 4)

Close the portal **teleop** tab. Stop the Isaac container/process (`Ctrl+C`, or
`docker stop <container>` — see
[`isaacsim_demo_runbook.md`](../controller/scripts/isaac/isaacsim_demo_runbook.md)).
Confirm `teleop_edge_connected` is false in `krabby get telemetry`.

### Step 3 — Start the isaacsim image (reverse bring-up step 3)

Confirm Isaac HAL is not running. Stop **`krabby-uno-sim`** if you used the optional
**`--joystick`** TCP path.

### Step 1 — Register with the fleet manager (reverse bring-up step 1)

**Long-lived** sim lab thing: leave **`krabby-agent.service`** enabled.

**Throwaway** thing: decommission per [Decommission throwaway thing](#decommission-throwaway-thing) below.

### Step 2 — Appears in the UI (reverse bring-up step 2)

Throwaway: confirm the thing is **absent** from portal **`/devices`** / `krabby-fleet list`.
Long-lived: thing remains listed.

### Prerequisites (reverse)

If you stopped **`krabby-locomotion.service`** for the check:
`sudo systemctl enable --now krabby-locomotion.service` (Jetson locomotion hosts only).

Stack-wide IoT cleanup notes: [`infra/control-plane.md`](infra/control-plane.md).

### Decommission throwaway thing

On the sim host (reverse of enable agent + write `/etc/krabby/iot/`):

```bash
sudo systemctl stop krabby-agent.service
sudo systemctl disable krabby-agent.service
sudo rm -rf /etc/krabby/iot
```

In AWS (reverse of create thing → cert → attach policy → attach thing principal):

```bash
export AWS_DEFAULT_REGION=<region>
THING=<sim-thing-name>

for arn in $(aws iot list-thing-principals --thing-name "$THING" --query principals[] --output text); do
  aws iot detach-thing-principal --thing-name "$THING" --principal "$arn"
  aws iot detach-policy --policy-name KrabDevicePolicy --target "$arn"
  cert_id="${arn##*/}"
  aws iot update-certificate --certificate-id "$cert_id" --new-status INACTIVE
  aws iot delete-certificate --certificate-id "$cert_id"
done

aws iot delete-thing --thing-name "$THING"
```

Same cert/thing order as scratch teardown in
[`test_control_plane_e2e.py`](../fleet/infra/tests_e2e/test_control_plane_e2e.py).

---

## Pass / fail

Maps to: register → UI → connect → teleoperate → offline when stopped.

| Assert | Pass | Fail |
|--------|------|------|
| **Registers with the fleet manager** | Enrolled thing; **`krabby-agent`** active; **:9000** listening | No identity, agent down, or no shim |
| **Appears in the UI** | Portal / `krabby-fleet list`; recent **`timestamp`** | Missing or stale |
| **Firmware in the loop** | Isaac HAL **`--backend fullsim`** (hex **`--hexapod`** launch) | **`direct`** backend or Jetson locomotion HAL |
| **Connects** | `teleop_edge_connected` true while Isaac HAL is up | false while HAL running |
| **Teleoperates** | **Playing**; operator override moves sim | ICE/signaling stuck or no motion |
| **Shows offline when stopped** | `teleop_edge_connected` false after HAL stop; teleop session ended | Teleop still live with HAL stopped |

The **agent** may remain MQTT-online after HAL stop; this check is about teleop/HAL
disconnect unless fleet semantics for “offline” are extended later.

Record: date, thing name, launcher version, **`krabby-hal-server-isaac`** wheel version (expect
**0.1.2**+), pass/fail per row.

---

## Missing pieces (from this check)

| Missing piece | Effect |
|---------------|--------|
| **isaacsim** Docker image not in a registry; HAL wheels on PyPI (**`krabby-hal-server-isaac` 0.1.2**) | Manual **`make build-isaacsim-image`** on sim host |
| **`krabby run`** launches Jetson HAL, not Isaac **fullsim** | Start **agent** and Isaac HAL separately |
| No install flow when **`mcu_present`** is false | Operator starts Isaac manually |
| No GPU preflight | Container fails on wrong hardware |
| No automatic resume after sim stop | Restart Isaac by hand |
| Shadow **`reported_image`** is locomotion image, not isaacsim | UI image field misleading |
| Fleet live pytest E2E uses bench Jetson HAL, not Isaac **fullsim** | This procedure is the check |
