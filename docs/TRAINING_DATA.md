# Training data from fleet camera recordings

This guide is for anyone who wants to **use robot camera (and proprioception) recordings for training or offline analysis**: where finished segments live in S3, how access is authorized, how to download them, and how to turn H.265 topics into arrays.

Operator / format detail lives elsewhere — this doc does not restate the full collector contract:

- [DATA_COLLECTOR.md](DATA_COLLECTOR.md) — topics, rates, H.265 encode, bag rotation, `decode_h265`
- [SENSOR_INTERFACE.md](SENSOR_INTERFACE.md) — catalog cameras and channels
- [JETSON_DEPLOYMENT.md](JETSON_DEPLOYMENT.md) — enabling collector / offload on a kit
- [PUBLISHING.md](PUBLISHING.md) — locomotion `release-latest` (must include `krabby-data-collection`)
- [fleet/ENROLL.md](../fleet/ENROLL.md) — device cert via on-device CSR (`krabby enroll`)
- [fleet/infra/control-plane.md](../fleet/infra/control-plane.md) — `KrabBags` CDK resources

> Bucket and thing names change per environment. Prefer `fleet/config/fleet.toml` `[bags] bucket` or CFN output `KrabBagsBucketName` over any hard-coded example here.

---

## 1. What is in a segment (summary)

Each **closed** segment is a **rosbag2 v9 / MCAP** directory (`metadata.yaml` + mcap), default **~5 minutes**. Default content:

- Cameras as H.265 `sensor_msgs/CompressedImage` on `/camera/{catalog_id}/{channel}_h265` (`front_rgbd` RGB+depth @ 10 Hz; ToF depth-only @ 20 Hz).
- Proprioception: `/joints/state`, `/joints/command`, `/imu`, `/base/twist`.

Depth is 8-bit quantized over the catalog meter range before HEVC; decode recovers float32 meters from `format`’s `range=` metadata.

Full topic map, YAML knobs, and rotation/quota rules: [DATA_COLLECTOR.md](DATA_COLLECTOR.md).

---

## 2. End-to-end data path

```text
HAL observations  →  in-process collector  →  /var/krabby/bags/krabby_<seq>_<YYYYMMDD_HHMMSS>/
                         ↓ (segment closes; active open segment is never uploaded)
                   krabby agent offloader
                         ↓ (device cert → IoT credential provider → temp AWS creds)
                   s3://<KrabBags>/<thing>/<YYYY-MM-DD>/<HHMMSS>_<seq>/
                         ↓ (HeadObject size verify)
                   local segment deleted
```

**Not** `KrabReportedImages` (small teleop/UI stills). Training archives are **`KrabBags`** only.

---

## 3. Authentication and authorization

There are **two different identities**: robots that **upload**, and people/tools that **download**. Robots never hold long-lived AWS access keys for bag upload.

### 3.1 Upload (robot → S3)

1. **Device identity** — at enroll, the robot gets an X.509 cert/key under `/etc/krabby/iot/` and an IoT thing name (on-device CSR; private key never leaves the Orin — see [fleet/ENROLL.md](../fleet/ENROLL.md)). Enroll also stores `credential_endpoint`, `bags_bucket`, and `bags_role_alias` (when the CFN export exists).
2. **IoT authorization** — the shared device policy (`KrabDevicePolicy`) allows  
   `iot:AssumeRoleWithCertificate` on `arn:aws:iot:*:*:rolealias/KrabBagsRoleAlias`  
   (`AssumeBagsUploadRole`). Only a principal presenting a cert attached to that policy can request that alias.
3. **Credential exchange** — `krabby` offload calls  
   `https://<credential_endpoint>/role-aliases/KrabBagsRoleAlias/credentials`  
   with **client TLS** using the device cert (mTLS). AWS IoT’s Credentials Provider returns **short-lived** AWS access key / secret / session token.
4. **IAM authorization** — those temp creds are for IAM role **`KrabBagsUploadRole`**, trusted only by `credentials.iot.amazonaws.com`. That role may:
   - `s3:PutObject` (+ related put variants) and multipart abort/list on the bags bucket
   - `s3:GetObject` / list-style reads used to **verify** size after upload before local delete  
   It is **not** a general admin role (no bucket delete, no IAM changes, no other buckets by design).
5. **Bucket hardening** — `KrabBags` is not public (`BlockPublicAccess`), SSE-S3 encrypted, SSL-only (`enforce_ssl`). Objects are not world-readable.
6. **Prefix convention** — keys are `<thing-name>/<YYYY-MM-DD>/<HHMMSS>_<seq>/…`. Isolation is **by convention in the offloader**, not a per-thing IAM condition on the shared role (same pattern as reported-image upload). Any enrolled robot that can assume `KrabBagsRoleAlias` can write under that shared role’s permissions; fleet robots are treated as trusted peers.
7. **Enable gates** — nothing uploads unless `"bags_offload": true` in locomotion config **and** collector has closed a segment. Offload also **pauses while teleop** is connected so uploads do not compete with live video.

`krabby-enroll` IAM user keys (if you re-enroll) are **only for enrollment APIs**, not for S3 bag traffic.

### 3.2 Download / training (human or CI → S3)

Device certs and `KrabBagsUploadRole` are the **upload** path. Training machines typically use a normal **AWS principal** (SSO, IAM user/role, CI OIDC) that has been granted **read** (and optionally list) on `KrabBags`.

- Default ControlPlane deploy does **not** automatically grant every AWS account user read access to bags; someone with admin/SSO must use an identity that can `s3:ListBucket` / `s3:GetObject` on that bucket (console, CLI, or a dedicated read role you attach for researchers).
- Do **not** put device private keys on laptops or training clusters.
- Do **not** confuse this with Cognito portal login — fleet portal auth is separate from bag-bucket IAM.

### 3.3 Quick comparison

| Actor | Authenticates with | Authorized to |
|-------|--------------------|---------------|
| Robot offloader | Device X.509 → IoT role alias → temp AWS creds | Put (+ verify Get) on `KrabBags` under shared upload role |
| Training consumer | Your AWS SSO / IAM / CI role | Whatever read (or write) your admin attached to that principal |
| `krabby enroll` | `krabby-enroll` access key (out of band) | IoT enroll APIs + `cloudformation:ListExports` for bucket name — **not** bag Put as the main path |

---

## 4. Robot prerequisites (operator)

Skip if segments already appear in S3. Short checklist; details in [JETSON_DEPLOYMENT.md](JETSON_DEPLOYMENT.md) and [PUBLISHING.md](PUBLISHING.md):

1. Release locomotion image includes pinned `krabby-data-collection` (`release/*` promotes `release-latest`; `main` alone does not).
2. `/etc/krabby/locomotion.json`: `"data_collector": true`, `"bags_offload": true`.
3. IoT config has `credential_endpoint`, `bags_bucket`, `bags_role_alias` (re-enroll or hand-edit).
4. Drive ≥ one closed segment (~5 min); wait until teleop ends if you need upload to proceed.

Host bags dir: `/var/krabby/bags` → container `/data/krabby_bags`.

---

## 5. Find and download segments from S3

Use an AWS identity with **list/get** on `KrabBags` (§3.2).

```bash
# From fleet.toml [bags] bucket, or:
# aws cloudformation describe-stacks --stack-name ControlPlaneStack \
#   --query "Stacks[0].Outputs[?OutputKey=='KrabBagsBucketName'].OutputValue" --output text
BUCKET=controlplanestack-krabbags1adb796a-pwhkt6yzgph0
THING=<iot-thing-name>
DATE=<YYYY-MM-DD>

aws s3 ls "s3://${BUCKET}/${THING}/${DATE}/"

SEG=<HHMMSS>_<seq>
mkdir -p ~/krabby_bags/${THING}/${DATE}/${SEG}
aws s3 sync \
  "s3://${BUCKET}/${THING}/${DATE}/${SEG}/" \
  ~/krabby_bags/${THING}/${DATE}/${SEG}/
```

Key layout:

```text
s3://<bucket>/<thing-name>/<YYYY-MM-DD>/<HHMMSS>_<seq>/metadata.yaml
s3://<bucket>/<thing-name>/<YYYY-MM-DD>/<HHMMSS>_<seq>/…mcap…
```

After a successful offload, that segment is **removed** from `/var/krabby/bags` on the robot.

---

## 6. Inspect and decode

Inspect:

```bash
BAG=~/krabby_bags/${THING}/${DATE}/${SEG}
ros2 bag info "$BAG"   # if ROS 2 is available
```

Or list topics with `rosbags` `AnyReader` (no ROS distro required) — see [DATA_COLLECTOR.md](DATA_COLLECTOR.md#playback).

Decode H.265 → `.npy` (`{catalog_id}_{channel}_{NNNNN}.npy`):

```bash
python -m data_collection.tools.decode_h265 "$BAG" \
  --catalog-id front_rgbd --channel rgb --out-dir /tmp/front_rgb

python -m data_collection.tools.decode_h265 "$BAG" \
  --catalog-id belly_left_tof --channel depth --out-dir /tmp/belly_left_depth

python -m data_collection.tools.decode_h265 "$BAG" \
  --catalog-id belly_left_tof --channel depth --check-roundtrip
```

Needs OpenCV and/or GStreamer `avdec_h265` on the training machine. Raw `camera_encode: raw` bags use normal image topics; `decode_h265` is for the H.265 path only. Full playback notes: [DATA_COLLECTOR.md](DATA_COLLECTOR.md#playback).

---

## 7. Suggested training consumption pattern

No in-tree Dataset class is required. A practical pattern:

1. List downloaded segment dirs (one bag ≈ one short clip).
2. Pre-decode with `decode_h265` to `.npy`, or stream with `rosbags` and decode in workers.
3. Treat depth quantization as part of the sensor model; align modalities by **message timestamps** (do not assume `hz × duration` frame counts).

```python
from pathlib import Path
import numpy as np

frames = sorted(Path("/tmp/belly_left_depth").glob("*.npy"))
depth = np.load(frames[0])  # float32 meters, HxW
```

---

## 8. Checklist for a first training sample

- [ ] Your AWS principal can list/get the bags bucket (§3.2)
- [ ] Segment exists under `s3://…/<thing>/<date>/<hms>_<seq>/`
- [ ] Local copy has `metadata.yaml` + mcap
- [ ] Topics include `*_h265` cameras + joints/imu
- [ ] `decode_h265` yields RGB and/or float depth
- [ ] `--check-roundtrip` passes on a depth stream

---

## 9. Common issues

| Symptom | Likely cause |
|---------|----------------|
| Nothing in S3 | Collector/offload flags off; segment still open; teleop holding offload; image missing `krabby-data-collection`; missing `credential_endpoint` / bags bucket in IoT config |
| `AssumeRoleWithCertificate` / credential fetch fails | Device policy missing `AssumeBagsUploadRole`; wrong role alias; cert not attached to thing/policy |
| AccessDenied on `aws s3 ls/sync` | Your user/role lacks read on `KrabBags` (upload role ≠ download identity) |
| Empty local `/var/krabby/bags` but no objects you can see | Offload may have succeeded under another account/prefix, or your principal cannot list the bucket |
| Can’t decode HEVC | Missing OpenCV/GStreamer decoder plugins |
| Depth looks stepped / coarse | Expected 8-bit quant over the configured meter range |
| Wrong bucket | Orphan RETAIN leftover from a failed create — use `fleet.toml` / `KrabBagsBucketName` |
